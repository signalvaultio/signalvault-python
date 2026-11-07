"""What a call does with each pre-flight answer, in both fail modes.

A refusal from SignalVault (a block decision, or ``content_policy_violation``
in any answer) never reaches the provider. Any other answer without a
decision is up to ``fail_mode``: "closed" raises; "open" calls the provider and
first records the request itself, marked unchecked, with the status and the
reason. A 429 with ``"recorded": true`` means SignalVault stored it already.

Each row checks three things: whether the provider was called, what the
caller got, and exactly which events SignalVault received.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
import warnings
from typing import Any, List, NamedTuple, Optional, Tuple
from unittest.mock import MagicMock, patch

import httpx
import pytest

import signalvault
from signalvault import (
    AsyncSignalVaultClient,
    SignalVaultBlockedError,
    SignalVaultUnavailableError,
    SignalVaultWarning,
)

from .test_security import (
    MESSAGES,
    RECORDED_429,
    Server,
    TrackedBody,
    drop_records,
    fake_completion,
    provider_handler,
    unchecked_429,
)

SV_METADATA = {"tag": "t1"}
VIOLATION = {"rule_id": "r1", "type": "contains_secret", "severity": 9, "action": "block", "details": {}}
DASHBOARD = "https://signalvault.io/dashboard/requests/r1"
# The SignalVault proxy's block answer.
PROXY_BLOCK = {"error": {
    "type": "invalid_request_error", "code": "content_policy_violation",
    "message": "Request blocked by SignalVault guardrails.",
    "violations": [VIOLATION], "dashboard_url": DASHBOARD,
}}
UNSERIALIZABLE = "unserializable"  # the call's sv_metadata holds a cycle


def reply(status: int, body: Any = None, text: Optional[str] = None, headers: Optional[dict] = None) -> dict:
    spec: dict = {"status": status, "headers": headers}
    if text is not None:
        spec["text"] = text
    else:
        spec["json"] = {} if body is None else body
    return spec


class Row(NamedTuple):
    id: str
    answer: Any  # a reply, an httpx exception class (no answer), or UNSERIALIZABLE
    outcome: str  # "allow", "block", "recorded", "full" or "minimal"
    status: Optional[int] = None  # preflight_status, and SignalVaultUnavailableError.status
    reason: Optional[str] = None  # preflight_reason
    violations: Tuple[str, ...] = ()  # rule_ids on SignalVaultBlockedError
    dashboard_url: Optional[str] = None


ROWS = [
    # 2xx decisions.
    Row("200-allow", reply(200, {"decision": "allow", "violations": [], "redactions": []}), "allow"),
    Row("200-warn", reply(200, {"decision": "warn", "violations": [{**VIOLATION, "action": "warn"}]}), "allow"),
    Row("200-redact", reply(200, {
        "decision": "redact", "violations": [], "redactions": [{"type": "contains_pii", "count": 1}],
    }), "allow"),
    Row("200-block", reply(200, {"decision": "block", "violations": [VIOLATION], "dashboard_url": DASHBOARD}),
        "block", violations=("r1",), dashboard_url=DASHBOARD),
    # A refusal in any answer.
    Row("400-cpv-error", reply(400, {"error": "content_policy_violation"}), "block"),
    Row("400-cpv-error-code", reply(400, PROXY_BLOCK), "block", violations=("r1",), dashboard_url=DASHBOARD),
    Row("403-cpv-error-type", reply(403, {"error": {"type": "content_policy_violation"}}), "block"),
    Row("409-decision-block", reply(409, {"decision": "block", "violations": [VIOLATION], "dashboard_url": DASHBOARD}),
        "block", violations=("r1",), dashboard_url=DASHBOARD),
    # Stored unchecked by SignalVault.
    Row("429-recorded-rate-limit", RECORDED_429, "recorded", 429),
    Row("429-recorded-unscanned", unchecked_429("unscanned_content"), "recorded", 429),
    # No decision; the full record.
    Row("network", httpx.ConnectError, "full", None, "network"),
    Row("timeout", httpx.ReadTimeout, "full", None, "timeout"),
    Row("301", reply(301, text="", headers={"location": "https://elsewhere.example"}), "full", 301, "redirect"),
    Row("302", reply(302, text="", headers={"location": "https://elsewhere.example"}), "full", 302, "redirect"),
    Row("401", reply(401, {"errors": {"detail": "Unauthorized"}}), "full", 401, "http_401"),
    Row("402", reply(402, {"error": {"type": "subscription_inactive", "code": "subscription_inactive"}}),
        "full", 402, "subscription_inactive"),
    Row("403", reply(403, {"error": "environment_not_allowed", "message": "Not allowed"}),
        "full", 403, "environment_not_allowed"),
    Row("429-trial", reply(429, {"error": {
        "type": "trial_limit_exceeded", "code": "trial_limit_exceeded", "message": "Trial limit reached",
    }}), "full", 429, "trial_limit_exceeded"),
    Row("429-plain", reply(429, {"error": "rate_limited"}, headers={"retry-after": "30"}), "full", 429, "rate_limited"),
    Row("500", reply(500, text="Internal Server Error"), "full", 500, "http_500"),
    Row("503", reply(503, {"error": {"code": "encryption_unavailable", "message": "Try again"}},
                     headers={"retry-after": "5"}), "full", 503, "encryption_unavailable"),
    Row("200-invalid-json", reply(200, text="<html>maintenance</html>"), "full", 200, "invalid_response"),
    Row("200-invalid-decision", reply(200, {"decision": "maybe"}), "full", 200, "invalid_response"),
    # No decision; the minimal record.
    Row("400", reply(400, {"error": {"type": "invalid_request_error", "message": "Body is not JSON"}}),
        "minimal", 400, "invalid_request_error"),
    Row("404", reply(404, text="Not Found"), "minimal", 404, "http_404"),
    Row("409-unknown-4xx", reply(409, {"error": "conflict"}), "minimal", 409, "conflict"),
    Row("413", reply(413, text="Payload Too Large"), "minimal", 413, "http_413"),
    Row("422", reply(422, {"error": "model must be a string"}), "minimal", 422, "http_422"),
    Row(UNSERIALIZABLE, UNSERIALIZABLE, "minimal", None, "unserializable"),
]
ROW = {row.id: row for row in ROWS}
# A refusal, a full record and a minimal one, for the other clients.
ASYNC_ROWS = ["400-cpv-error-code", "network", "503", "413", UNSERIALIZABLE]
ANTHROPIC_ROWS = ["400-cpv-error-code", "503", "413"]


def _no_answer(answer: Any) -> bool:
    return isinstance(answer, type) and issubclass(answer, Exception)


class Answers(Server):
    """SignalVault: answers the pre-flight with ``answer``, every later event with 200.

    With an httpx exception class the pre-flight gets no answer, and isn't
    recorded. ``later_delay`` slows every send after the first.
    """

    def __init__(self, answer: Any, later_delay: float = 0.0):
        sent = not _no_answer(answer) and answer != UNSERIALIZABLE
        super().__init__({"ai.request": [answer] if sent else []})
        self.no_answer = answer if _no_answer(answer) else None
        self.later_delay = later_delay
        self.seen = 0

    def _reply(self, request: httpx.Request) -> httpx.Response:
        self.seen += 1
        if self.no_answer is not None:
            exc, self.no_answer = self.no_answer, None
            raise exc("no answer", request=request)
        return super()._reply(request)

    def sync_handler(self, request: httpx.Request) -> httpx.Response:
        if self.seen and self.later_delay:
            time.sleep(self.later_delay)
        return self._reply(request)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        if self.seen and self.later_delay:
            await asyncio.sleep(self.later_delay)
        return self._reply(request)


def label(body: dict) -> str:
    if body["type"] == "ai.response":
        return "response"
    if body["type"] == "ai.request" and body["payload"].get("preflight_unavailable"):
        return "fallback"
    return "preflight" if body["type"] == "ai.request" else body["type"]


def labels(server: Server) -> List[str]:
    return [label(b) for b in server.bodies]


def sv_warnings(caught) -> List[str]:
    return [str(w.message) for w in caught if issubclass(w.category, SignalVaultWarning)]


def call_metadata(row: Row) -> dict:
    if row.answer == UNSERIALIZABLE:
        loop: dict = {}
        loop["self"] = loop
        return {"loop": loop}
    return SV_METADATA


def no_waits(client) -> None:
    """Retries go out without waiting."""
    client._random = lambda: 0.5
    if isinstance(client, signalvault.client._BaseAsyncClient):
        async def sleep(_):
            pass

        client._sleep = sleep
    else:
        client._sleep = lambda _: None


class Provider:
    """The provider: counts the calls, and notes what SignalVault had received at each."""

    def __init__(self, client, server: Server, result: Any, error: Optional[BaseException] = None):
        self.client, self.server, self.result, self.error = client, server, result, error
        self.calls = 0
        self.seen: List[List[str]] = []

    def sync_create(self, **_: Any) -> Any:
        self.calls += 1
        self.client.flush()  # what was queued before the call is sent now
        self.seen.append(labels(self.server))
        if self.error is not None:
            raise self.error
        return self.result

    async def async_create(self, **_: Any) -> Any:
        self.calls += 1
        await self.client.flush()
        self.seen.append(labels(self.server))
        if self.error is not None:
            raise self.error
        return self.result


def anthropic_message():
    message = MagicMock()
    message.content = [MagicMock(type="text", text="hello")]
    message.usage = MagicMock(input_tokens=3, output_tokens=1)
    return message


def build(kind: str, server: Server, fail_mode: str, error: Optional[BaseException] = None):
    """A client of ``kind`` talking to ``server``, its provider, and the call to make."""
    from signalvault import AnthropicSignalVaultClient, AsyncAnthropicSignalVaultClient, SignalVaultClient

    cls = {
        "openai": SignalVaultClient, "async-openai": AsyncSignalVaultClient,
        "anthropic": AnthropicSignalVaultClient, "async-anthropic": AsyncAnthropicSignalVaultClient,
    }[kind]
    if kind.endswith("openai"):
        client = cls(api_key="sk_test_abc", openai_api_key="sk-fake", fail_mode=fail_mode)
    else:
        with patch.dict("sys.modules", {"anthropic": MagicMock()}):
            client = cls(api_key="sk_test_abc", anthropic_api_key="sk-ant", fail_mode=fail_mode)
    is_async = kind.startswith("async")
    transport = httpx.MockTransport(server.async_handler if is_async else server.sync_handler)
    client._http = httpx.AsyncClient(transport=transport) if is_async else httpx.Client(transport=transport)
    no_waits(client)
    provider = Provider(client, server, fake_completion() if kind.endswith("openai") else anthropic_message(), error)
    create = provider.async_create if is_async else provider.sync_create
    if kind.endswith("openai"):
        client._openai = MagicMock()
        client._openai.chat.completions.create = create
        return client, provider, client.chat.completions.create
    client._anthropic = MagicMock()
    client._anthropic.messages.create = create
    return client, provider, functools.partial(client.messages.create, max_tokens=5)


def run(kind: str, row: Row, fail_mode: str):
    server = Answers(row.answer)
    client, provider, create = build(kind, server, fail_mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = create(model="m", messages=MESSAGES, sv_metadata=call_metadata(row))
        except (SignalVaultBlockedError, SignalVaultUnavailableError) as exc:
            result = exc
        client.flush()
    return server, provider, result, sv_warnings(caught)


async def run_async(kind: str, row: Row, fail_mode: str):
    server = Answers(row.answer)
    client, provider, create = build(kind, server, fail_mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = await create(model="m", messages=MESSAGES, sv_metadata=call_metadata(row))
        except (SignalVaultBlockedError, SignalVaultUnavailableError) as exc:
            result = exc
        await client.flush()
    return server, provider, result, sv_warnings(caught)


def expected_fallback(row: Row, request_id: str, provider_name: str) -> dict:
    payload: dict = {
        "preflight_unavailable": True, "fail_mode": "open",
        "preflight_status": row.status, "preflight_reason": row.reason,
    }
    body = {
        "type": "ai.request", "request_id": request_id, "environment": "production",
        "provider": provider_name, "model": "m", "payload": payload,
    }
    if row.outcome == "full":
        body["metadata"] = SV_METADATA
        payload["messages"] = MESSAGES
    return body


def check(row: Row, fail_mode: str, kind: str, outcome) -> None:
    server, provider, result, warned = outcome
    events = labels(server)
    preflight = [] if _no_answer(row.answer) or row.answer == UNSERIALIZABLE else ["preflight"]
    unchecked = [w for w in warned if "Guardrails were NOT applied" in w]
    request_ids = {b["request_id"] for b in server.bodies}
    assert len(request_ids) <= 1  # every event is about the one call

    if row.outcome == "allow":
        assert provider.calls == 1
        assert result is provider.result
        assert events == ["preflight", "response"]
        assert unchecked == []
        return

    if row.outcome == "block":
        assert provider.calls == 0
        assert isinstance(result, SignalVaultBlockedError)
        assert result.request_id == server.bodies[0]["request_id"]
        assert tuple(v.rule_id for v in result.violations) == row.violations
        assert result.dashboard_url == row.dashboard_url
        assert str(result) == "[SignalVault] Request blocked by guardrails ({}).".format(
            "contains_secret" if row.violations else "policy")
        assert events == ["preflight"]
        assert unchecked == []
        return

    if fail_mode == "closed":
        assert provider.calls == 0
        assert isinstance(result, SignalVaultUnavailableError)
        assert result.status == (429 if row.outcome == "recorded" else row.status)
        assert events == preflight
        return

    assert provider.calls == 1
    assert result is provider.result
    assert len(unchecked) == 1
    if row.outcome == "recorded":
        # SignalVault stored the request: no second record.
        assert provider.seen == [preflight]
        assert events == preflight + ["response"]
        return
    # The record is sent before the provider call, then the response after it.
    assert provider.seen == [preflight + ["fallback"]]
    if row.answer == UNSERIALIZABLE:
        # The response carries the same metadata: it can't be sent either, and is
        # dropped, logged and counted. The record leaves the metadata out.
        assert events == ["fallback"]
        assert [w for w in warned if "dropped" in w] == [
            "[SignalVault] ai.response event dropped: it could not be serialized "
            "(ValueError: Circular reference detected).",
        ]
    else:
        assert events == preflight + ["fallback", "response"]
    fallback = server.bodies[len(preflight)]
    provider_name = "openai" if kind.endswith("openai") else "anthropic"
    assert fallback == expected_fallback(row, server.bodies[-1]["request_id"], provider_name)
    if events[-1] == "response":
        assert server.bodies[-1]["payload"]["output"] == "hello"


@pytest.mark.parametrize("fail_mode", ["open", "closed"])
@pytest.mark.parametrize("row", ROWS, ids=[row.id for row in ROWS])
def test_sync_openai(row, fail_mode):
    check(row, fail_mode, "openai", run("openai", row, fail_mode))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["open", "closed"])
@pytest.mark.parametrize("row_id", ASYNC_ROWS)
async def test_async_openai(row_id, fail_mode):
    row = ROW[row_id]
    check(row, fail_mode, "async-openai", await run_async("async-openai", row, fail_mode))


@pytest.mark.parametrize("fail_mode", ["open", "closed"])
@pytest.mark.parametrize("row_id", ANTHROPIC_ROWS)
def test_sync_anthropic(row_id, fail_mode):
    row = ROW[row_id]
    check(row, fail_mode, "anthropic", run("anthropic", row, fail_mode))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["open", "closed"])
@pytest.mark.parametrize("row_id", ANTHROPIC_ROWS)
async def test_async_anthropic(row_id, fail_mode):
    row = ROW[row_id]
    check(row, fail_mode, "async-anthropic", await run_async("async-anthropic", row, fail_mode))


# ---------------------------------------------------------------------------
# What the record holds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,body", [
    (403, {"error": "environment_not_allowed\n[SignalVault] All rules passed"}),
    (402, {"error": {"code": "[SignalVault] All rules passed"}}),
    (429, {"error": {"type": "Trial limit exceeded"}}),
    (409, {"error": "x" * 65}),
    (400, {"error": {"code": "Bad_Request"}}),
])
def test_server_text_is_never_put_in_the_record(status, body):
    server = Answers(reply(status, body))
    client, _, create = build("openai", server, "open")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SignalVaultWarning)
        create(model="m", messages=MESSAGES)
    client.flush()
    assert labels(server) == ["preflight", "fallback", "response"]
    assert server.bodies[1]["payload"]["preflight_reason"] == f"http_{status}"


def test_minimal_record_leaves_out_a_model_that_is_not_a_string():
    server = Answers(reply(422, {"error": "model must be a string"}))
    client, _, create = build("openai", server, "open")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SignalVaultWarning)
        create(model=123, messages=MESSAGES)
    client.flush()
    fallback = server.bodies[1]
    assert "model" not in fallback and "metadata" not in fallback
    assert fallback["payload"] == {
        "preflight_unavailable": True, "fail_mode": "open",
        "preflight_status": 422, "preflight_reason": "http_422",
    }


# ---------------------------------------------------------------------------
# The record goes before the provider call, and the response after it
# ---------------------------------------------------------------------------

class TestRecordBeforeTheProviderCall:
    def test_provider_error_still_leaves_the_record(self):
        server = Answers(reply(503))
        client, provider, create = build("openai", server, "open", error=RuntimeError("provider down"))
        with pytest.warns(SignalVaultWarning, match="NOT applied"), pytest.raises(RuntimeError, match="provider down"):
            create(model="m", messages=MESSAGES)
        client.flush()
        assert provider.calls == 1
        assert labels(server) == ["preflight", "fallback"]

    @pytest.mark.asyncio
    async def test_async_provider_error_still_leaves_the_record(self):
        server = Answers(reply(413))
        client, _, create = build("async-anthropic", server, "open", error=RuntimeError("provider down"))
        with pytest.warns(SignalVaultWarning, match="NOT applied"), pytest.raises(RuntimeError, match="provider down"):
            await create(model="m", messages=MESSAGES)
        await client.flush()
        assert labels(server) == ["preflight", "fallback"]
        assert "messages" not in server.bodies[1]["payload"]

    def _openai_stream_client(self, server: Server, async_: bool = False):
        import openai

        cls = AsyncSignalVaultClient if async_ else signalvault.SignalVaultClient
        client = cls(api_key="k", openai_api_key="x")
        transport = httpx.MockTransport(server.async_handler if async_ else server.sync_handler)
        client._http = httpx.AsyncClient(transport=transport) if async_ else httpx.Client(transport=transport)
        body = TrackedBody()
        provider_transport = httpx.MockTransport(provider_handler(body))
        client._openai = (openai.AsyncOpenAI if async_ else openai.OpenAI)(
            api_key="sk-fake", base_url="http://provider.test/v1", max_retries=0,
            http_client=(httpx.AsyncClient if async_ else httpx.Client)(transport=provider_transport),
        )
        return client, body

    def test_stream_that_is_never_read_leaves_the_record(self):
        server = Answers(reply(503))
        client, body = self._openai_stream_client(server)
        with pytest.warns(SignalVaultWarning, match="NOT applied"):
            stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        client.flush()
        assert labels(server) == ["preflight", "fallback"]  # held, unread: no response yet
        stream.close()
        client.flush()
        assert body.closed
        assert labels(server) == ["preflight", "fallback", "response"]

    @pytest.mark.asyncio
    async def test_async_stream_that_is_never_read_leaves_the_record(self):
        server = Answers(httpx.ConnectError)
        client, body = self._openai_stream_client(server, async_=True)
        with pytest.warns(SignalVaultWarning, match="NOT applied"):
            stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        await client.flush()
        assert labels(server) == ["fallback"]
        await stream.close()
        await client.flush()
        assert body.closed
        assert labels(server) == ["fallback", "response"]

    def test_response_waits_for_a_record_still_being_sent(self):
        gate = threading.Event()
        order: List[str] = []

        def handler(request):
            body = json.loads(request.content)
            if label(body) == "fallback":
                gate.wait(5)  # slow: the response is ready long before
            order.append(label(body))
            return httpx.Response(503 if len(order) == 1 else 200, json={})

        client, _, create = build("openai", Server(), "open")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        client._openai.chat.completions.create = lambda **_: fake_completion()
        with pytest.warns(SignalVaultWarning):
            create(model="m", messages=MESSAGES)
        # Delivery threads have taken everything queued, while the record is still being sent.
        deadline = time.monotonic() + 5
        while client._queue and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
        assert order == ["preflight"]
        gate.set()
        client.flush()
        assert order == ["preflight", "fallback", "response"]

    @pytest.mark.asyncio
    async def test_async_response_waits_for_a_record_still_being_sent(self):
        gate = asyncio.Event()
        order: List[str] = []

        async def handler(request):
            body = json.loads(request.content)
            if label(body) == "fallback":
                await gate.wait()
            order.append(label(body))
            return httpx.Response(503 if len(order) == 1 else 200, json={})

        client, _, create = build("async-openai", Server(), "open")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def provider(**_):
            return fake_completion()

        client._openai.chat.completions.create = provider
        with pytest.warns(SignalVaultWarning):
            await create(model="m", messages=MESSAGES)
        for _ in range(5):
            await asyncio.sleep(0)
        assert order == ["preflight"]
        gate.set()
        await client.flush()
        assert order == ["preflight", "fallback", "response"]

    def test_response_is_sent_when_the_full_queue_dropped_the_record(self, monkeypatch, caplog):
        import signalvault.client as client_module

        from .test_security import GatedServer

        monkeypatch.setattr(client_module, "_MAX_QUEUED_EVENTS", 3)
        monkeypatch.setattr(client_module, "_DELIVERY_WORKERS", 1)
        server = GatedServer()
        client, _, create = build("openai", server, "open")
        client._openai.chat.completions.create = lambda **_: fake_completion()
        client._fire_tool_call(signalvault.ToolRecordOptions(tool_name="t0"))
        assert server.first_arrived.wait(5)  # t0 is being sent; the rest queue up
        server.replies["ai.request"] = [reply(503)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SignalVaultWarning)
            create(model="m", messages=MESSAGES)  # queues the record, then the response
            for name in ("t1", "t2"):  # the queue is full: the record, the oldest, is dropped
                client._fire_tool_call(signalvault.ToolRecordOptions(tool_name=name))
        server.release()
        start = time.monotonic()
        client.flush()
        assert time.monotonic() - start < 2
        assert [label(b) for b in server.bodies] == ["agent.tool_call", "preflight", "response",
                                                     "agent.tool_call", "agent.tool_call"]
        [record] = drop_records(caplog)
        assert record.signalvault_drop_cause == "queue_full"
        assert record.signalvault_dropped == [
            f"request_id={server.bodies[1]['request_id']} (fallback ai.request, which has no event_id)",
        ]
        assert client.dropped_events == signalvault.DroppedEvents(queue_full=1)


class TestRecordWhenTheEventLoopEnds:
    """asyncio.run cancels the task sending the record when the caller returns first."""

    def test_record_goes_with_a_stream_finished_at_loop_shutdown_and_counts_no_drop(self, caplog):
        from .test_security import AnthropicProvider, anthropic_create, async_anthropic_streaming_client

        provider = AnthropicProvider()
        kept = []

        async def main():
            client, server = async_anthropic_streaming_client(provider, preflight=reply(503))
            server.delay = 0.05  # every send takes time, as over a network
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SignalVaultWarning)
                kept.append((await anthropic_create(client), server, client))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(main())
        _, server, client = kept[0]
        assert provider.body.closed
        assert server.types() == ["ai.request", "ai.request", "ai.response"]
        assert server.bodies[1]["payload"]["preflight_reason"] == "http_503"
        assert drop_records(caplog) == []
        assert [w for w in sv_warnings(caught) if "dropped" in w] == []
        assert client.dropped_events == signalvault.DroppedEvents()

    def test_record_cut_off_without_a_response_is_reported_once_at_exit(self, caplog):
        server = Answers(reply(503), later_delay=5)
        client, _, create = build("async-openai", server, "open", error=RuntimeError("provider down"))
        client._openai.chat.completions.create = _raise_provider_error

        async def main():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SignalVaultWarning)
                with pytest.raises(RuntimeError):
                    await create(model="m", messages=MESSAGES)
            await asyncio.sleep(0.01)  # the record's send starts and is left hanging

        asyncio.run(main())
        assert drop_records(caplog) == []  # it would go with a response; there is none
        with pytest.warns(SignalVaultWarning, match="the process exited while its event loop"):
            client._report_unsent_at_exit()
        [record] = drop_records(caplog)
        assert record.signalvault_dropped == [
            f"request_id={server.bodies[0]['request_id']} (fallback ai.request, which has no event_id)",
        ]
        assert client.dropped_events == signalvault.DroppedEvents(shutdown=1)


async def _raise_provider_error(**_):
    raise RuntimeError("provider down")
