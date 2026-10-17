"""Security and reliability behaviour, exercised through httpx's MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import warnings
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

import signalvault
from signalvault import (
    AsyncSignalVaultClient,
    SignalVaultBlockedError,
    SignalVaultClient,
    SignalVaultUnavailableError,
    SignalVaultWarning,
)
from signalvault.client import normalize_base_url
from signalvault.tools import sanitize_payload

ALLOW = {"status": 200, "json": {"decision": "allow", "violations": [], "redactions": []}}
MESSAGES = [{"role": "user", "content": "hi"}]


class Server:
    """Records requests; replies from per-event-type queues (default 200 {})."""

    def __init__(self, replies: Dict[str, List[dict]] | None = None, delay: float = 0.0):
        self.replies = replies or {}
        self.delay = delay
        self.requests: List[httpx.Request] = []

    @property
    def bodies(self) -> List[dict]:
        return [json.loads(r.content) for r in self.requests]

    def types(self) -> List[str]:
        return [b["type"] for b in self.bodies]

    def _reply(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.replies.get(json.loads(request.content)["type"])
        spec = queue.pop(0) if queue else {"status": 200, "json": {}}
        if "text" in spec:
            return httpx.Response(spec["status"], text=spec["text"], headers=spec.get("headers"))
        return httpx.Response(spec["status"], json=spec.get("json", {}), headers=spec.get("headers"))

    def sync_handler(self, request: httpx.Request) -> httpx.Response:
        if self.delay:
            time.sleep(self.delay)
        return self._reply(request)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._reply(request)


def fake_completion():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "hello"
    resp.usage = MagicMock(prompt_tokens=3, completion_tokens=1)
    return resp


def sync_client(server: Server, **kw) -> SignalVaultClient:
    client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake", **kw)
    client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
    client._openai = MagicMock()
    client._openai.chat.completions.create.return_value = fake_completion()
    return client


# ---------------------------------------------------------------------------
# base_url
# ---------------------------------------------------------------------------

class TestBaseUrl:
    def test_accepts_https_and_local_http(self):
        assert normalize_base_url("https://api.signalvault.io/") == "https://api.signalvault.io"
        assert normalize_base_url("http://localhost:4000") == "http://localhost:4000"
        assert normalize_base_url("http://127.0.0.1:4000") == "http://127.0.0.1:4000"
        assert normalize_base_url("http://[::1]:4000") == "http://[::1]:4000"

    def test_refuses_plaintext_to_remote_hosts(self):
        with pytest.raises(ValueError, match="must use https"):
            SignalVaultClient(api_key="k", openai_api_key="x", base_url="http://api.signalvault.io")

    def test_refuses_garbage_and_other_schemes(self):
        for bad in ("api.signalvault.io", "ftp://api.signalvault.io"):
            with pytest.raises(ValueError):
                normalize_base_url(bad)

    def test_rejects_unknown_fail_mode(self):
        with pytest.raises(ValueError, match="fail_mode"):
            SignalVaultClient(api_key="k", openai_api_key="x", fail_mode="maybe")

    def test_http_clients_do_not_follow_redirects(self):
        client = SignalVaultClient(api_key="k", openai_api_key="x")
        assert client._http.follow_redirects is False


# ---------------------------------------------------------------------------
# api_key
# ---------------------------------------------------------------------------

def anthropic_classes():
    from signalvault import AnthropicSignalVaultClient, AsyncAnthropicSignalVaultClient
    return [AnthropicSignalVaultClient, AsyncAnthropicSignalVaultClient]


def build_any(cls, api_key):
    if cls in (SignalVaultClient, AsyncSignalVaultClient):
        return cls(api_key=api_key, openai_api_key="sk-fake")
    with patch.dict("sys.modules", {"anthropic": MagicMock()}):
        return cls(api_key=api_key, anthropic_api_key="sk-ant")


class TestApiKey:
    @pytest.mark.parametrize("raw", ["sk_test_abc\n", "sk_test_abc\r\n", "  sk_test_abc ", "\tsk_test_abc", "\ufeffsk_test_abc\n"])
    def test_surrounding_whitespace_is_stripped_and_block_enforced(self, raw):
        # A key read from a file used to make every pre-flight fail locally,
        # which fail_mode="open" turned into "allow".
        violations = [{"rule_id": "r1", "type": "contains_secret", "severity": 9, "action": "block", "details": {}}]
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "block", "violations": violations, "redactions": [],
        }}]})
        client = SignalVaultClient(api_key=raw, openai_api_key="sk-fake")
        client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
        client._openai = MagicMock()
        with pytest.raises(SignalVaultBlockedError):
            client.chat.completions.create(model="m", messages=MESSAGES)
        assert server.requests[0].headers["authorization"] == "Bearer sk_test_abc"
        client._openai.chat.completions.create.assert_not_called()

    @pytest.mark.parametrize("bad,pattern", [
        ("", "empty"),
        (" \n", "empty"),
        ("sk_test abc_SECRET", "whitespace"),
        ("sk_test\nabc_SECRET", "whitespace"),
        ("sk_test\x00abc_SECRET", "control"),
        ("sk_test_abc_SECRETé", "non-ASCII"),
    ])
    def test_invalid_keys_are_refused_without_echoing_them(self, bad, pattern):
        with pytest.raises(ValueError, match=pattern) as exc:
            SignalVaultClient(api_key=bad, openai_api_key="sk-fake")
        assert "SECRET" not in str(exc.value)

    def test_non_string_key_is_a_type_error(self):
        with pytest.raises(TypeError, match="NoneType"):
            SignalVaultClient(api_key=None, openai_api_key="sk-fake")

    @pytest.mark.parametrize("index", range(4))
    def test_every_client_validates(self, index):
        cls = ([SignalVaultClient, AsyncSignalVaultClient] + anthropic_classes())[index]
        with pytest.raises(ValueError, match="whitespace"):
            build_any(cls, "sk_test abc")
        assert build_any(cls, "sk_test_abc\n")._config.api_key == "sk_test_abc"


# ---------------------------------------------------------------------------
# Headers & event_id
# ---------------------------------------------------------------------------

class TestHeaders:
    def test_accept_user_agent_and_event_id(self):
        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        headers = server.requests[0].headers
        assert headers["accept"] == "application/json"
        assert headers["user-agent"].startswith(f"signalvault-python/{signalvault.__version__} python/")
        response_event = server.bodies[1]
        assert response_event["type"] == "ai.response"
        assert len(response_event["event_id"]) == 36


# ---------------------------------------------------------------------------
# Pre-flight failures
# ---------------------------------------------------------------------------

class TestPreflightFailures:
    @pytest.mark.parametrize("status,body,pattern", [
        (401, {"errors": {"detail": "Unauthorized"}}, "API key"),
        (402, {"error": "subscription_inactive"}, "not active"),
        (403, {"error": "environment_not_allowed"}, "denied access"),
        (429, {"error": "rate_limited"}, "rate limit"),
        (429, {"error": {"type": "trial_limit_exceeded"}}, "trial limit"),
        (503, {}, "API error"),
    ])
    def test_fail_open_warns_without_debug(self, status, body, pattern):
        server = Server({"ai.request": [{"status": status, "json": body}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match=pattern):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()

    def test_warns_once_per_minute_per_cause(self):
        server = Server({"ai.request": [{"status": 401, "json": {}} for _ in range(3)]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                client.chat.completions.create(model="m", messages=MESSAGES)
        assert len([w for w in caught if issubclass(w.category, SignalVaultWarning)]) == 1

    def test_fail_closed_raises_and_skips_provider(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server, fail_mode="closed")
        with pytest.raises(SignalVaultUnavailableError) as exc:
            client.chat.completions.create(model="m", messages=MESSAGES)
        assert exc.value.status == 401
        client._openai.chat.completions.create.assert_not_called()

    def test_non_json_200_is_unavailable_not_a_crash(self):
        server = Server({"ai.request": [{"status": 200, "text": "<html>maintenance</html>"}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="invalid response"):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_block_raises_typed_error(self):
        violations = [{"rule_id": "r1", "type": "contains_secret", "severity": 9, "action": "block", "details": {}}]
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "block", "violations": violations, "redactions": [],
            "dashboard_url": "https://signalvault.io/x",
        }}]})
        client = sync_client(server)
        with pytest.raises(SignalVaultBlockedError) as exc:
            client.chat.completions.create(model="m", messages=MESSAGES)
        err = exc.value
        assert isinstance(err, RuntimeError)
        assert str(err) == "[SignalVault] Request blocked by guardrails (contains_secret)."
        assert err.violations[0].rule_id == "r1"
        assert err.dashboard_url == "https://signalvault.io/x"
        client._openai.chat.completions.create.assert_not_called()

    def test_redactions_array_and_request_unmodified(self):
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "redact", "violations": [], "redactions": [{"type": "contains_pii", "count": 1}],
        }}]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        assert client._openai.chat.completions.create.call_args.kwargs["messages"] == MESSAGES

        server.replies["ai.request"] = [{"status": 200, "json": {
            "decision": "redact", "violations": [], "redactions": [{"type": "contains_pii", "count": 1}],
        }}]
        decision = client._send_request("r", {"model": "m", "messages": MESSAGES}, {})
        assert decision.redactions == [{"type": "contains_pii", "count": 1}]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def anthropic_client(server: Server):
    from signalvault import AnthropicSignalVaultClient

    mock_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        client = AnthropicSignalVaultClient(api_key="k", anthropic_api_key="sk-ant")
    client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
    response = MagicMock()
    response.content = [
        MagicMock(type="text", text="Let me check. "),
        MagicMock(type="tool_use", text=None),
        MagicMock(type="text", text="Done."),
    ]
    response.usage = MagicMock(input_tokens=5, output_tokens=2)
    client._anthropic.messages.create.return_value = response
    return client


class TestAnthropic:
    def test_system_prompt_is_scanned_and_provider_call_unchanged(self):
        server = Server({"ai.request": [ALLOW]})
        client = anthropic_client(server)
        client.messages.create(model="claude", max_tokens=5, system="SYS sk-secret", messages=MESSAGES)
        client.flush()

        assert server.bodies[0]["payload"]["messages"][0] == {"role": "system", "content": "SYS sk-secret"}
        sent = client._anthropic.messages.create.call_args.kwargs
        assert sent["system"] == "SYS sk-secret"
        assert sent["messages"] == MESSAGES

    def test_records_every_text_block(self):
        server = Server({"ai.request": [ALLOW]})
        client = anthropic_client(server)
        client.messages.create(model="claude", max_tokens=5, messages=MESSAGES)
        client.flush()
        assert server.bodies[1]["payload"]["output"] == "Let me check. Done."


# ---------------------------------------------------------------------------
# Provider SDK objects in messages
# ---------------------------------------------------------------------------


def openai_tool_turn() -> list:
    """The usual tool loop: the assistant message is appended as the SDK object."""
    from openai.types.chat import ChatCompletionMessage

    assistant = ChatCompletionMessage.model_validate({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "London"}'}}],
    })
    return [
        {"role": "user", "content": "Weather in London?"},
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": "12C"},
    ]


EXPECTED_ASSISTANT = {
    "role": "assistant", "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "London"}'}}],
}


class TestProviderObjectsInMessages:
    def test_openai_message_object_is_scanned(self):
        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        messages = openai_tool_turn()
        client.chat.completions.create(model="m", messages=messages)
        client.flush()

        assert server.bodies[0]["payload"]["messages"][1] == EXPECTED_ASSISTANT
        # The provider still receives the caller's objects unchanged.
        assert client._openai.chat.completions.create.call_args.kwargs["messages"] == messages
        assert server.types() == ["ai.request", "ai.response"]

    @pytest.mark.asyncio
    async def test_async_openai_message_object_is_scanned(self):
        server = Server({"ai.request": [ALLOW]})
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
        client._openai = MagicMock()

        async def create(**_):
            return fake_completion()

        client._openai.chat.completions.create = create
        await client.chat.completions.create(model="m", messages=openai_tool_turn())
        await client.flush()
        assert server.bodies[0]["payload"]["messages"][1] == EXPECTED_ASSISTANT

    def test_mirror_mode_records_message_objects(self):
        server = Server()
        client = sync_client(server, mirror_mode=True)
        client.chat.completions.create(model="m", messages=openai_tool_turn())
        client.flush()
        assert server.types() == ["ai.request", "ai.response"]
        assert server.bodies[0]["payload"]["messages"][1] == EXPECTED_ASSISTANT

    def test_anthropic_content_blocks_are_scanned(self):
        types = pytest.importorskip("anthropic.types")
        if not hasattr(types, "ToolUseBlock"):
            pytest.skip("this anthropic version has no ToolUseBlock")
        blocks = [
            types.TextBlock(type="text", text="Let me check."),
            types.ToolUseBlock(type="tool_use", id="tu_1", name="weather", input={"city": "London"}),
        ]
        server = Server({"ai.request": [ALLOW]})
        client = anthropic_client(server)
        client.messages.create(model="claude", max_tokens=5, messages=[
            {"role": "user", "content": "Weather in London?"},
            {"role": "assistant", "content": blocks},
        ])
        client.flush()

        assert server.bodies[0]["payload"]["messages"][1]["content"] == [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "weather", "input": {"city": "London"}},
        ]


class TestEncoding:
    def test_lone_surrogate_is_still_scanned(self):
        server = Server({"ai.request": [{"status": 200, "json": {"decision": "block", "violations": []}}]})
        client = sync_client(server)
        with pytest.raises(SignalVaultBlockedError):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x\ud800y"}])
        assert b"x\\ud800y" in server.requests[0].content
        assert not client._openai.chat.completions.create.called

    def test_body_is_json(self):
        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "å"}])
        assert server.requests[0].headers["content-type"] == "application/json"
        assert server.bodies[0]["payload"]["messages"][0]["content"] == "å"

    def test_common_metadata_values(self):
        import datetime
        import decimal
        import enum
        import uuid

        class Plan(enum.Enum):
            PRO = "pro"

        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES, sv_metadata={
            "id": uuid.UUID(int=1), "at": datetime.datetime(2026, 1, 2, 3, 4, 5),
            "price": decimal.Decimal("1.50"), "plan": Plan.PRO, "tags": {"a"},
        })
        assert server.bodies[0]["metadata"] == {
            "id": "00000000-0000-0000-0000-000000000001", "at": "2026-01-02T03:04:05",
            "price": "1.50", "plan": "pro", "tags": ["a"],
        }

    def test_value_that_cannot_be_converted_falls_back_to_repr(self):
        import dataclasses
        import threading

        class Detached:
            def __repr__(self):
                raise RuntimeError("instance is not bound to a session")

        @dataclasses.dataclass
        class Holder:
            lock: threading.Lock

        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SignalVaultWarning)
            client.chat.completions.create(model="m", messages=MESSAGES, sv_metadata={
                "user": Detached(), "holder": Holder(threading.Lock()),
            })
        metadata = server.bodies[0]["metadata"]
        assert metadata["user"] == "<unrepresentable Detached>"
        assert metadata["holder"].startswith("TestEncoding.")  # repr of the dataclass
        assert client._openai.chat.completions.create.called

    def test_pydantic_v1_shim(self):
        """openai/anthropic under pydantic v1 expose model_dump, but reject mode="json"."""
        from pydantic import BaseModel

        class Message(BaseModel):
            role: str
            content: str = ""

            def model_dump(self, **kwargs):
                if "mode" in kwargs:
                    raise ValueError("mode is only supported in Pydantic v2")
                return super().model_dump(**kwargs)

        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=[Message(role="assistant")])
        assert server.bodies[0]["payload"]["messages"] == [{"role": "assistant"}]

    def test_mock_objects_do_not_hang(self):
        from signalvault.client import _encode_body

        assert b"MagicMock" in _encode_body({"message": MagicMock()})


class TestUnserializableRequest:
    def _cyclic_metadata(self):
        loop: dict = {}
        loop["self"] = loop
        return {"loop": loop}

    def test_fail_open_warns_and_calls_the_provider(self):
        server = Server()
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="could not be serialized"):
            client.chat.completions.create(model="m", messages=MESSAGES, sv_metadata=self._cyclic_metadata())
        client.flush()
        assert client._openai.chat.completions.create.called
        assert server.requests == []

    def test_fail_closed_raises(self):
        server = Server()
        client = sync_client(server, fail_mode="closed")
        with pytest.raises(SignalVaultUnavailableError, match="could not be serialized"):
            client.chat.completions.create(model="m", messages=MESSAGES, sv_metadata=self._cyclic_metadata())
        assert not client._openai.chat.completions.create.called

    def test_mirror_mode_warns_instead_of_dropping_silently(self):
        server = Server()
        client = sync_client(server, mirror_mode=True)
        with pytest.warns(SignalVaultWarning, match="event dropped: it could not be serialized") as record:
            client.chat.completions.create(model="m", messages=MESSAGES, sv_metadata=self._cyclic_metadata())
            client.flush()
        assert server.requests == []
        dropped = {str(w.message).split()[1] for w in record}
        assert dropped == {"ai.request", "ai.response"}


# ---------------------------------------------------------------------------
# Background delivery
# ---------------------------------------------------------------------------

class TestBackgroundDelivery:
    def test_retries_once_on_503_with_same_event_id(self):
        server = Server({"ai.request": [ALLOW], "ai.response": [{"status": 503}, {"status": 200}]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        responses = [b for b in server.bodies if b["type"] == "ai.response"]
        assert len(responses) == 2
        assert responses[0]["event_id"] == responses[1]["event_id"]

    def test_short_retry_after_retried_long_one_dropped_with_warning(self):
        server = Server({
            "ai.request": [ALLOW, ALLOW],
            "ai.response": [
                {"status": 429, "headers": {"retry-after": "0"}},
                {"status": 200},
                {"status": 429, "headers": {"retry-after": "60"}},
            ],
        })
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()
        assert server.types().count("ai.response") == 2

        with pytest.warns(SignalVaultWarning, match="being dropped"):
            client.chat.completions.create(model="m", messages=MESSAGES)
            client.flush()
        assert server.types().count("ai.response") == 3

    def test_flush_waits_for_pending_events(self):
        server = Server(delay=0.1)
        client = sync_client(server)
        client._fire_response("r", "m", "out", {"prompt_tokens": 1, "completion_tokens": 1}, {})
        assert server.requests == []
        client.flush()
        assert server.types() == ["ai.response"]

    def test_tool_input_captured_at_call_time(self):
        """Mutating the argument after the call must not change the recorded input."""
        server = Server()
        client = sync_client(server)
        held = []
        # Hold the background job until after the mutation, so the test is deterministic.
        with patch.object(client, "_submit", side_effect=lambda fn, *a: held.append((fn, a))):
            args = {"city": "London"}
            client.tool("lookup", lambda a: "ok")(args)
            args["city"] = "MUTATED"
        fn, a = held[0]
        fn(*a)
        assert server.bodies[0]["payload"]["tool_input"] == {"city": "London"}

    def test_invalid_tool_name_does_not_break_the_tool(self):
        server = Server()
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="tool_call event dropped"):
            assert client.tool("", lambda: 42)() == 42


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class TestAsync:
    @pytest.mark.asyncio
    async def test_background_tasks_are_referenced_and_flushed(self):
        server = Server({"ai.request": [ALLOW]}, delay=0.05)
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
        client._openai = MagicMock()

        async def create(**_):
            return fake_completion()

        client._openai.chat.completions.create = create
        await client.chat.completions.create(model="m", messages=MESSAGES)
        assert len(client._tasks) == 1
        await client.flush()
        assert client._tasks == set()
        assert server.types() == ["ai.request", "ai.response"]

    def test_fire_without_loop_warns_instead_of_silently_dropping(self):
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        with pytest.warns(SignalVaultWarning, match="No running event loop"):
            client._fire_tool_call(signalvault.ToolRecordOptions(tool_name="t"))


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def test_deep_nesting_does_not_raise():
    value = cur = {}
    for _ in range(5000):
        cur["a"] = {}
        cur = cur["a"]
    result = sanitize_payload(value)
    assert "[MaxDepth]" in json.dumps(result)


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------

class TestMalformedDecisions:
    @pytest.mark.parametrize("body", [
        {"decision": ["block"]},
        {"decision": {}},
        {"decision": "allow", "violations": 5},
    ])
    def test_wrong_types_fail_open_instead_of_raising(self, body):
        server = Server({"ai.request": [{"status": 200, "json": body}]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()

    def test_non_string_violation_type_in_block(self):
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "block", "violations": [{"type": 7}, "junk", {"type": "pii"}],
        }}]})
        client = sync_client(server)
        with pytest.raises(SignalVaultBlockedError, match=r"\(7, pii\)"):
            client.chat.completions.create(model="m", messages=MESSAGES)


class TestWarningsRepeat:
    def test_repeats_after_the_window_under_default_filters(self, monkeypatch):
        import signalvault.client as client_module

        monkeypatch.setattr(client_module, "_WARN_INTERVAL_SECONDS", 0.0)
        server = Server({"ai.request": [{"status": 401, "json": {}} for _ in range(3)]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                client._send_request("r", {"model": "m", "messages": []}, {})
        assert len(caught) == 3

    def test_points_at_the_callers_code(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning) as record:
            client.chat.completions.create(model="m", messages=MESSAGES)
        assert record[0].filename == __file__


class TestToolCallWarnings:
    def test_rejected_tool_call_warns_at_the_callers_code(self):
        server = Server({"agent.tool_call": [{"status": 401}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="tool_call rejected with 401") as record:
            client.tools.record(tool_name="t", duration_ms=1)
        assert record[0].filename == __file__

    def test_rejected_tool_call_warns_once_per_window(self):
        server = Server({"agent.tool_call": [{"status": 401}, {"status": 401}]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.tools.record(tool_name="t", duration_ms=1)
            client.tools.record(tool_name="t", duration_ms=1)
        assert len([w for w in caught if issubclass(w.category, SignalVaultWarning)]) == 1

    def test_failed_tool_call_warns_only_in_debug(self):
        server = Server({"agent.tool_call": [{"status": 503}, {"status": 503}]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync_client(server).tools.record(tool_name="t", duration_ms=1)
        assert caught == []
        with pytest.warns(SignalVaultWarning) as record:
            sync_client(server, debug=True).tools.record(tool_name="t", duration_ms=1)
        assert any("tool_call event failed: 503" in str(w.message) for w in record)

    @pytest.mark.asyncio
    async def test_async_rejected_tool_call_warns(self):
        server = Server({"agent.tool_call": [{"status": 403}]})
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
        with pytest.warns(SignalVaultWarning, match="tool_call rejected with 403"):
            await client.tools.record(tool_name="t", duration_ms=1)


class TestRetryPolicy:
    def test_warns_when_the_retry_is_rate_limited_too(self):
        server = Server({
            "ai.request": [ALLOW],
            "ai.response": [
                {"status": 429, "headers": {"retry-after": "0"}},
                {"status": 429, "headers": {"retry-after": "60"}},
            ],
        })
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="being dropped"):
            client.chat.completions.create(model="m", messages=MESSAGES)
            client.flush()
        assert server.types().count("ai.response") == 2

    def test_timeouts_are_not_retried(self):
        attempts = []

        def handler(request):
            attempts.append(json.loads(request.content)["type"])
            raise httpx.ReadTimeout("slow", request=request)

        client = SignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        client._fire_response("r", "m", "out", {"prompt_tokens": 1, "completion_tokens": 1}, {})
        client.flush()
        assert attempts == ["ai.response"]

    def test_manual_tool_record_does_not_retry(self):
        server = Server({"agent.tool_call": [{"status": 503}, {"status": 200}]})
        client = sync_client(server)
        client.tools.record(tool_name="t", duration_ms=1)
        assert server.types() == ["agent.tool_call"]


class TestAuditWhenPreflightFailed:
    def test_records_the_request_before_the_response_on_503(self):
        server = Server({"ai.request": [{"status": 503}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        assert server.types() == ["ai.request", "ai.request", "ai.response"]
        fallback = server.bodies[1]
        assert fallback["request_id"] == server.bodies[0]["request_id"]
        assert "event_id" not in fallback
        assert fallback["payload"] == {"messages": MESSAGES, "preflight_unavailable": True}

    def test_no_audit_events_after_a_refused_preflight(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()
        assert server.types() == ["ai.request"]

    @pytest.mark.asyncio
    async def test_async_records_the_request_on_network_error(self):
        calls = []

        async def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if len(calls) == 1:
                raise httpx.ConnectError("down", request=request)
            return httpx.Response(200, json={})

        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client._openai = MagicMock()

        async def create(**_):
            return fake_completion()

        client._openai.chat.completions.create = create
        with pytest.warns(SignalVaultWarning, match="unreachable"):
            await client.chat.completions.create(model="m", messages=MESSAGES)
        await client.flush()
        assert [c["type"] for c in calls] == ["ai.request", "ai.request", "ai.response"]
        assert calls[1]["payload"]["preflight_unavailable"] is True


class TestFailureClassification:
    def test_redirect_is_reported_as_a_base_url_problem(self):
        server = Server({"ai.request": [{"status": 307, "headers": {"location": "https://elsewhere"}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match=r"redirected \(307\).*base_url"):
            client.chat.completions.create(model="m", messages=MESSAGES)

    def test_fail_closed_on_network_error(self):
        def handler(request):
            raise httpx.ConnectError("down", request=request)

        client = sync_client(Server(), fail_mode="closed")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(SignalVaultUnavailableError, match="unreachable"):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_not_called()

    def test_base_url_with_query_or_fragment_is_refused(self):
        for bad in ("https://api.signalvault.io?x=1", "https://api.signalvault.io#frag"):
            with pytest.raises(ValueError, match="query or fragment"):
                normalize_base_url(bad)


# ---------------------------------------------------------------------------
# Real provider clients (only the HTTP layer is mocked)
# ---------------------------------------------------------------------------


class OpenAIStreamServer:
    """Answers like OpenAI's streaming API, including the usage-only final chunk."""

    def __init__(self):
        self.bodies: List[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return openai_sse()


def openai_sse_events() -> List[str]:
    def chunk(choices, usage=None):
        body = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": choices}
        if usage:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n"

    return [
        chunk([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]),
        chunk([{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]),
        chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        chunk([], usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}),
        "data: [DONE]\n\n",
    ]


def openai_sse() -> httpx.Response:
    content = "".join(openai_sse_events()).encode()
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


class TestRealOpenAIClient:
    """Runs against the installed openai package, so CI on the minimum versions catches API drift."""

    def test_stream(self):
        import openai

        server = Server({"ai.request": [ALLOW]})
        provider = OpenAIStreamServer()
        client = sync_client(server)
        client._openai = openai.OpenAI(
            api_key="sk-fake", base_url="http://provider.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(provider.handler)),
        )
        text = "".join(
            c.choices[0].delta.content or ""
            for c in client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        )
        client.flush()
        assert provider.bodies[0]["stream_options"] == {"include_usage": True}
        assert text == "Hello"
        response = server.bodies[1]["payload"]
        assert response["output"] == "Hello"
        assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}

    @pytest.mark.asyncio
    async def test_async_stream(self):
        import openai

        server = Server({"ai.request": [ALLOW]})
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
        provider = OpenAIStreamServer()
        client._openai = openai.AsyncOpenAI(
            api_key="sk-fake", base_url="http://provider.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
        )
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        text = "".join([c.choices[0].delta.content or "" async for c in stream])
        await client.flush()
        assert provider.bodies[0]["stream_options"] == {"include_usage": True}
        assert text == "Hello"
        assert server.bodies[1]["payload"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}


# ---------------------------------------------------------------------------
# OpenAI stream lifecycle (real openai.Stream / AsyncStream)
# ---------------------------------------------------------------------------

class TrackedBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    """An SSE response body that records whether it was closed.

    With ``fail`` it stops after the first event that carries "Hello" and then
    raises ``error`` (default: a dropped connection), or with ``hang`` it waits
    there until it's closed and then raises like a dropped connection (async only).
    """

    def __init__(
        self, fail: bool = False, error: Optional[BaseException] = None,
        hang: bool = False, events: Optional[List[str]] = None, close_delay: float = 0.0,
    ):
        events = openai_sse_events() if events is None else events
        if fail or hang:
            events = events[:next(i for i, e in enumerate(events) if '"Hello"' in e) + 1]
        self.parts = [e.encode() for e in events]
        self.fail = fail
        self.error = error
        self.hang = hang
        self.closed = False
        self._closing: Optional[asyncio.Event] = None
        self.close_delay = close_delay

    def _raise(self):
        raise self.error if self.error is not None else httpx.ReadError("connection lost")

    def __iter__(self):
        yield from self.parts
        if self.fail:
            self._raise()

    async def __aiter__(self):
        for part in self.parts:
            yield part
        if self.hang:
            self._closing = asyncio.Event()
            await self._closing.wait()
            self._raise()
        if self.fail:
            self._raise()

    def close(self):
        self.closed = True

    async def aclose(self):
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        self.closed = True
        if self._closing is not None:
            self._closing.set()


def provider_handler(body: TrackedBody):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    return handler


def streaming_client(body: TrackedBody, mirror: bool = False):
    import openai

    server = Server({"ai.request": [ALLOW]})
    client = sync_client(server, mirror_mode=mirror)
    client._openai = openai.OpenAI(
        api_key="sk-fake", base_url="http://provider.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(provider_handler(body))),
    )
    return client, server


def async_streaming_client(body: TrackedBody):
    import openai

    server = Server({"ai.request": [ALLOW]})
    client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
    client._openai = openai.AsyncOpenAI(
        api_key="sk-fake", base_url="http://provider.test/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider_handler(body))),
    )
    return client, server


def _dropped_errors():
    import openai
    # openai 1.x passes the httpx error through; later versions wrap it.
    return (httpx.ReadError, openai.APIConnectionError)


DROPPED = _dropped_errors()


def responses(server: Server) -> List[dict]:
    return [b["payload"] for b in server.bodies if b["type"] == "ai.response"]


def read_until_hello(stream) -> None:
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content == "Hello":
            break


class TestStreamLifecycle:
    def test_with_block_closes_and_records_on_break(self):
        body = TrackedBody()
        client, server = streaming_client(body)
        with client.chat.completions.create(model="m", messages=MESSAGES, stream=True) as stream:
            assert isinstance(stream, signalvault.SignalVaultStream)
            assert stream.response.status_code == 200
            read_until_hello(stream)
            assert not body.closed
        assert body.closed
        client.flush()
        [response] = responses(server)
        assert response["output"] == "Hello"

    def test_close_records_once(self):
        body = TrackedBody()
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        read_until_hello(stream)
        stream.close()
        stream.close()
        assert list(stream) == []
        del stream
        client.flush()
        assert body.closed
        assert len(responses(server)) == 1

    def test_close_from_two_threads_records_once(self, monkeypatch):
        body = TrackedBody()
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        read_until_hello(stream)

        # Each close() that reads _closed as False waits for the other thread
        # to read it too, so without the lock both see False and record.
        state = {"closed": False}
        both_read = threading.Barrier(2, timeout=0.5)

        def get_closed(_):
            closed = state["closed"]
            if not closed:
                try:
                    both_read.wait()
                except threading.BrokenBarrierError:
                    pass  # the other thread is waiting for the lock
            return closed

        def set_closed(_, value):
            state["closed"] = value

        monkeypatch.setattr(signalvault.SignalVaultStream, "_closed", property(get_closed, set_closed), raising=False)
        threads = [threading.Thread(target=stream.close) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        del stream
        client.flush()
        assert body.closed
        assert len(responses(server)) == 1

    def test_break_without_close_records_when_collected(self):
        body = TrackedBody()
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        read_until_hello(stream)
        del stream
        client.flush()
        assert body.closed
        assert [r["output"] for r in responses(server)] == ["Hello"]

    def test_exhausted_stream_records_and_closes(self):
        body = TrackedBody()
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        assert len(list(stream)) == 3
        assert body.closed
        client.flush()
        [response] = responses(server)
        assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}

    def test_error_mid_stream_closes_records_and_raises(self):
        body = TrackedBody(fail=True)
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        with pytest.raises(DROPPED):
            for _ in stream:
                pass
        assert body.closed
        client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    def test_keyboard_interrupt_mid_stream_closes_and_records(self):
        body = TrackedBody(fail=True, error=KeyboardInterrupt())
        client, server = streaming_client(body)
        stream = client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        with pytest.raises(KeyboardInterrupt):
            for _ in stream:
                pass
        assert body.closed
        client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    def test_mirror_mode_records_on_with_exit(self):
        body = TrackedBody()
        client, server = streaming_client(body, mirror=True)
        with client.chat.completions.create(model="m", messages=MESSAGES, stream=True) as stream:
            read_until_hello(stream)
        client.flush()
        assert body.closed
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_with_block_closes_and_records_on_break(self):
        body = TrackedBody()
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        assert isinstance(stream, signalvault.AsyncSignalVaultStream)
        async with stream:
            assert stream.response.status_code == 200
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content == "Hello":
                    break
            assert not body.closed
        assert body.closed
        await stream.close()
        await client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_error_mid_stream_closes_records_and_raises(self):
        body = TrackedBody(fail=True)
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        with pytest.raises(DROPPED):
            async for _ in stream:
                pass
        assert body.closed
        await client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_cancelled_read_closes_and_records(self):
        body = TrackedBody(hang=True)
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        seen = []

        async def read():
            async for chunk in stream:
                seen.append(chunk)

        task = asyncio.ensure_future(read())
        while len(seen) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
        await client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_break_then_collected_closes_and_records(self):
        body = TrackedBody()
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content == "Hello":
                break
        del stream, chunk
        for _ in range(5):
            await asyncio.sleep(0)
        await client.flush()
        assert body.closed
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_unread_stream_is_closed_and_recorded_when_collected(self):
        body = TrackedBody()
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        del stream
        for _ in range(5):
            await asyncio.sleep(0)
        await client.flush()
        assert body.closed
        assert server.types() == ["ai.request", "ai.response"]

    @pytest.mark.asyncio
    async def test_async_stream_built_directly_yields_only_upstream_items(self):
        from signalvault.client import _AnthropicEventCollector

        class Upstream:
            async def __aiter__(self):
                yield MagicMock(type="message_stop")

            async def close(self):
                pass

        recorded = []
        stream = signalvault.AsyncSignalVaultStream(Upstream(), _AnthropicEventCollector(), lambda *r: recorded.append(r))
        assert [e.type async for e in stream] == ["message_stop"]
        assert recorded == [("", {"input_tokens": 0, "output_tokens": 0})]

    @pytest.mark.asyncio
    async def test_async_close_while_another_task_reads(self):
        body = TrackedBody(hang=True)
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)

        async def read():
            async for _ in stream:
                pass

        reader = asyncio.ensure_future(read())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not reader.done()  # waiting on the provider
        await stream.close()
        assert body.closed
        with pytest.raises(DROPPED):
            await asyncio.wait_for(reader, timeout=1)  # the close ended the read
        await client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    async def test_async_overlapping_closes_record_once(self):
        body = TrackedBody(close_delay=0.01)
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        await stream.__anext__()
        await asyncio.gather(stream.close(), stream.close())
        assert body.closed
        await client.flush()
        assert len(responses(server)) == 1

    @pytest.mark.asyncio
    async def test_async_close_and_end_of_stream_do_not_wait_for_signalvault(self):
        body = TrackedBody()
        client, server = async_streaming_client(body)
        server.replies["ai.request"] = [ALLOW, ALLOW]
        streams = [await client.chat.completions.create(model="m", messages=MESSAGES, stream=True) for _ in range(2)]
        server.delay = 1.0  # the audit sends are slow
        start = time.monotonic()
        await streams[0].close()
        assert [c async for c in streams[1]]
        assert time.monotonic() - start < 0.5
        await client.flush()
        assert len(responses(server)) == 2

    @pytest.mark.asyncio
    async def test_async_close_twice_then_collected_records_once(self):
        body = TrackedBody()
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        await stream.__anext__()
        await stream.close()
        await stream.aclose()
        assert [c async for c in stream] == []
        del stream
        for _ in range(5):
            await asyncio.sleep(0)
        await client.flush()
        assert body.closed
        assert len(responses(server)) == 1

    @pytest.mark.asyncio
    async def test_async_aclose_and_aclosing(self):
        contextlib = pytest.importorskip("contextlib")
        body = TrackedBody()
        client, server = async_streaming_client(body)
        stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
        if hasattr(contextlib, "aclosing"):  # Python 3.10+
            async with contextlib.aclosing(stream):
                await stream.__anext__()
        else:
            await stream.__anext__()
            await stream.aclose()
        assert body.closed
        await client.flush()
        assert len(responses(server)) == 1

    # The event is sent while asyncio.run shuts the loop down, so httpx starts an
    # async generator after shutdown_asyncgens(), which asyncio reports with a
    # ResourceWarning. The event is still delivered, as it was with the old async generator.
    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    def test_async_stream_left_open_is_recorded_when_the_loop_shuts_down(self):
        body = TrackedBody()
        kept = []

        async def main():
            client, server = async_streaming_client(body)
            server.delay = 0.05  # the send takes time, as over a network
            stream = await client.chat.completions.create(model="m", messages=MESSAGES, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content == "Hello":
                    break
            kept.append((stream, server))

        asyncio.run(main())
        _, server = kept[0]
        assert body.closed
        assert [r["output"] for r in responses(server)] == ["Hello"]


# ---------------------------------------------------------------------------
# Anthropic streams (real anthropic.Stream / AsyncStream)
# ---------------------------------------------------------------------------

ANTHROPIC_EVENT_TYPES = [
    "message_start", "content_block_start", "content_block_delta",
    "content_block_stop", "message_delta", "message_stop",
]


def anthropic_sse_events(
    start_usage: Optional[dict] = None, delta_usage: Optional[dict] = None,
) -> List[str]:
    def event(data):
        return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"

    return [
        event({"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude",
            "stop_reason": None, "stop_sequence": None,
            "usage": start_usage or {"input_tokens": 7, "output_tokens": 1},
        }}),
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        event({"type": "content_block_stop", "index": 0}),
        event({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
               "usage": delta_usage or {"output_tokens": 2}}),
        event({"type": "message_stop"}),
    ]


def anthropic_httpx():
    """The HTTP library the installed anthropic SDK uses: httpx2 in recent versions, httpx before."""
    import anthropic._base_client as base

    return getattr(base, "httpx2", None) or base.httpx


class AnthropicProvider:
    """Answers like Anthropic's streaming Messages API and records the requests."""

    def __init__(
        self, fail: bool = False, status: int = 200,
        events: Optional[List[str]] = None, usage: Optional[dict] = None,
    ):
        hx = anthropic_httpx()
        body_class = TrackedBody
        if hx is not httpx:  # httpx2 only accepts its own stream classes
            body_class = type("TrackedBody", (TrackedBody, hx.SyncByteStream, hx.AsyncByteStream), {})
        self.body = body_class(fail=fail, events=events or anthropic_sse_events())
        self.status = status
        self.usage = usage  # of a non-streaming response
        self.requests: List[dict] = []

    def handler(self, request):
        hx = anthropic_httpx()
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            error = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
            return hx.Response(self.status, json=error)
        if not self.requests[-1].get("stream"):
            return hx.Response(200, json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": self.usage or {"input_tokens": 7, "output_tokens": 2},
            })
        return hx.Response(200, headers={"content-type": "text/event-stream"}, stream=self.body)


def anthropic_streaming_client(provider: AnthropicProvider, mirror: bool = False, preflight: Optional[dict] = None):
    import anthropic
    from signalvault import AnthropicSignalVaultClient

    server = Server({"ai.request": [preflight or ALLOW]})
    client = AnthropicSignalVaultClient(api_key="k", anthropic_api_key="sk-ant", mirror_mode=mirror)
    client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
    hx = anthropic_httpx()
    client._anthropic = anthropic.Anthropic(
        api_key="sk-ant", base_url="http://provider.test", max_retries=0,
        http_client=hx.Client(transport=hx.MockTransport(provider.handler)),
    )
    return client, server


def async_anthropic_streaming_client(
    provider: AnthropicProvider, mirror: bool = False, preflight: Optional[dict] = None,
):
    import anthropic
    from signalvault import AsyncAnthropicSignalVaultClient

    server = Server({"ai.request": [preflight or ALLOW]})
    client = AsyncAnthropicSignalVaultClient(api_key="k", anthropic_api_key="sk-ant", mirror_mode=mirror)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
    hx = anthropic_httpx()
    client._anthropic = anthropic.AsyncAnthropic(
        api_key="sk-ant", base_url="http://provider.test", max_retries=0,
        http_client=hx.AsyncClient(transport=hx.MockTransport(provider.handler)),
    )
    return client, server


def anthropic_create(client, **kw):
    return client.messages.create(model="claude", max_tokens=10, messages=MESSAGES, stream=True, **kw)


UNAVAILABLE = {"status": 503}

# The SignalVault events each mode sends for one streamed call.
STREAM_EVENTS = {
    "normal": ["ai.request", "ai.response"],
    "mirror": ["ai.request", "ai.response"],
    "preflight_unavailable": ["ai.request", "ai.request", "ai.response"],
}


def stream_mode_client(factory, provider: AnthropicProvider, mode: str):
    return factory(provider, mirror=mode == "mirror", preflight=UNAVAILABLE if mode == "preflight_unavailable" else None)


def read_until_hello_event(stream) -> None:
    for event in stream:
        if event.type == "content_block_delta":
            break


class TestAnthropicStream:
    def test_yields_the_raw_events_and_records_the_response(self):
        provider = AnthropicProvider()
        client, server = anthropic_streaming_client(provider)
        stream = anthropic_create(client)
        assert isinstance(stream, signalvault.SignalVaultStream)
        assert [e.type for e in stream] == ANTHROPIC_EVENT_TYPES
        assert provider.requests[0]["stream"] is True
        assert provider.body.closed
        client.flush()
        [response] = responses(server)
        assert response["output"] == "Hello"
        assert response["usage"] == {"input_tokens": 7, "output_tokens": 2}

    def test_request_is_sent_by_create_and_provider_errors_raise_there(self):
        import anthropic

        provider = AnthropicProvider()
        client, _ = anthropic_streaming_client(provider)
        anthropic_create(client)
        assert len(provider.requests) == 1

    @pytest.mark.parametrize("mirror", [False, True])
    def test_provider_error_at_create_records_no_response(self, mirror):
        import anthropic

        client, server = anthropic_streaming_client(AnthropicProvider(status=401), mirror=mirror)
        with pytest.raises(anthropic.AuthenticationError):
            anthropic_create(client)
        client.flush()
        assert server.types() == ([] if mirror else ["ai.request"])

    def test_with_block_closes_and_records_on_break(self):
        provider = AnthropicProvider()
        client, server = anthropic_streaming_client(provider)
        with anthropic_create(client) as stream:
            assert stream.response.status_code == 200
            read_until_hello_event(stream)
            assert not provider.body.closed
        assert provider.body.closed
        client.flush()
        [response] = responses(server)
        assert response["output"] == "Hello"
        assert response["usage"] == {"input_tokens": 7, "output_tokens": 1}

    def test_error_mid_stream_closes_records_and_raises(self):
        provider = AnthropicProvider(fail=True)
        client, server = anthropic_streaming_client(provider)
        stream = anthropic_create(client)
        with pytest.raises(Exception):
            for _ in stream:
                pass
        assert provider.body.closed
        client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    def test_mirror_mode(self):
        provider = AnthropicProvider()
        client, server = anthropic_streaming_client(provider, mirror=True)
        with anthropic_create(client, system="SYS") as stream:
            assert [e.type for e in stream] == ANTHROPIC_EVENT_TYPES
        client.flush()
        assert provider.requests[0]["system"] == "SYS"
        assert server.types() == ["ai.request", "ai.response"]
        request, response = server.bodies
        assert request["payload"]["monitor_mode"] is True
        assert response["payload"]["monitor_mode"] is True
        assert response["payload"]["output"] == "Hello"

    def test_preflight_unavailable_records_the_request_and_the_response(self):
        provider = AnthropicProvider()
        client, server = anthropic_streaming_client(provider, preflight=UNAVAILABLE)
        with pytest.warns(SignalVaultWarning):
            stream = anthropic_create(client)
        with stream:
            assert [e.type for e in stream] == ANTHROPIC_EVENT_TYPES
        client.flush()
        assert server.types() == STREAM_EVENTS["preflight_unavailable"]
        assert server.bodies[1]["payload"]["preflight_unavailable"] is True
        assert server.bodies[2]["payload"]["output"] == "Hello"

    @pytest.mark.parametrize("mode", list(STREAM_EVENTS))
    def test_unread_stream_is_closed_and_recorded_when_collected(self, mode):
        provider = AnthropicProvider()
        client, server = stream_mode_client(anthropic_streaming_client, provider, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SignalVaultWarning)
            stream = anthropic_create(client)
        assert len(provider.requests) == 1
        del stream
        client.flush()
        assert provider.body.closed
        assert server.types() == STREAM_EVENTS[mode]

    @pytest.mark.asyncio
    async def test_async_stream(self):
        provider = AnthropicProvider()
        client, server = async_anthropic_streaming_client(provider)
        stream = await anthropic_create(client)
        assert isinstance(stream, signalvault.AsyncSignalVaultStream)
        assert len(provider.requests) == 1
        async with stream:
            assert [e.type async for e in stream] == ANTHROPIC_EVENT_TYPES
        assert provider.body.closed
        await client.flush()
        [response] = responses(server)
        assert response["output"] == "Hello"
        assert response["usage"] == {"input_tokens": 7, "output_tokens": 2}

    @pytest.mark.asyncio
    async def test_async_with_block_closes_and_records_on_break(self):
        provider = AnthropicProvider()
        client, server = async_anthropic_streaming_client(provider)
        async with await anthropic_create(client) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    break
            assert not provider.body.closed
        assert provider.body.closed
        await client.flush()
        assert [r["output"] for r in responses(server)] == ["Hello"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mirror", [False, True])
    async def test_async_provider_error_at_create_records_no_response(self, mirror):
        import anthropic

        client, server = async_anthropic_streaming_client(AnthropicProvider(status=401), mirror=mirror)
        with pytest.raises(anthropic.AuthenticationError):
            await anthropic_create(client)
        await client.flush()
        assert server.types() == ([] if mirror else ["ai.request"])

    @pytest.mark.asyncio
    async def test_async_mirror_mode(self):
        provider = AnthropicProvider()
        client, server = async_anthropic_streaming_client(provider, mirror=True)
        async with await anthropic_create(client) as stream:
            assert [e.type async for e in stream] == ANTHROPIC_EVENT_TYPES
        await client.flush()
        assert server.types() == ["ai.request", "ai.response"]
        assert server.bodies[0]["payload"]["monitor_mode"] is True
        assert server.bodies[1]["payload"]["output"] == "Hello"

    @pytest.mark.asyncio
    async def test_async_preflight_unavailable_records_the_request_and_the_response(self):
        provider = AnthropicProvider()
        client, server = async_anthropic_streaming_client(provider, preflight=UNAVAILABLE)
        with pytest.warns(SignalVaultWarning):
            stream = await anthropic_create(client)
        async with stream:
            assert [e.type async for e in stream] == ANTHROPIC_EVENT_TYPES
        await client.flush()
        assert server.types() == STREAM_EVENTS["preflight_unavailable"]
        assert server.bodies[1]["payload"]["preflight_unavailable"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", list(STREAM_EVENTS))
    async def test_async_unread_stream_is_closed_and_recorded_when_collected(self, mode):
        provider = AnthropicProvider()
        client, server = stream_mode_client(async_anthropic_streaming_client, provider, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SignalVaultWarning)
            stream = await anthropic_create(client)
        assert len(provider.requests) == 1
        del stream
        for _ in range(5):
            await asyncio.sleep(0)
        await client.flush()
        assert provider.body.closed
        assert server.types() == STREAM_EVENTS[mode]

    # See test_async_stream_left_open_is_recorded_when_the_loop_shuts_down.
    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    @pytest.mark.parametrize("mode", list(STREAM_EVENTS))
    def test_async_unread_stream_is_recorded_when_the_loop_shuts_down(self, mode):
        provider = AnthropicProvider()
        kept = []

        async def main():
            client, server = stream_mode_client(async_anthropic_streaming_client, provider, mode)
            server.delay = 0.05  # the send takes time, as over a network
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SignalVaultWarning)
                kept.append((await anthropic_create(client), server))

        asyncio.run(main())
        _, server = kept[0]
        assert provider.body.closed
        assert server.types() == STREAM_EVENTS[mode]


# ---------------------------------------------------------------------------
# Anthropic usage: the server reads it under Anthropic's names and adds the
# cache counts to the input. Under OpenAI's prompt_tokens it would take the
# count as already including them, so the SDK must not rename.
# ---------------------------------------------------------------------------

CACHED_USAGE = {
    "input_tokens": 12,
    "cache_creation_input_tokens": 2_000,
    "cache_read_input_tokens": 90_000,
    "output_tokens": 300,
}


class TestAnthropicUsage:
    @pytest.mark.parametrize("mirror", [False, True])
    def test_is_sent_under_anthropic_names_with_the_cache_counts(self, mirror):
        client, server = anthropic_streaming_client(AnthropicProvider(usage=CACHED_USAGE), mirror=mirror)
        client.messages.create(model="claude", max_tokens=10, messages=MESSAGES)
        client.flush()
        [response] = responses(server)
        assert response["usage"] == CACHED_USAGE

    @pytest.mark.parametrize("mirror", [False, True])
    async def test_async_is_sent_under_anthropic_names_with_the_cache_counts(self, mirror):
        client, server = async_anthropic_streaming_client(AnthropicProvider(usage=CACHED_USAGE), mirror=mirror)
        await client.messages.create(model="claude", max_tokens=10, messages=MESSAGES)
        await client.flush()
        [response] = responses(server)
        assert response["usage"] == CACHED_USAGE

    def test_cache_counts_that_are_null_are_left_out(self):
        usage = {"input_tokens": 5, "cache_creation_input_tokens": None, "cache_read_input_tokens": 7, "output_tokens": 2}
        client, server = anthropic_streaming_client(AnthropicProvider(usage=usage))
        client.messages.create(model="claude", max_tokens=10, messages=MESSAGES)
        client.flush()
        [response] = responses(server)
        assert response["usage"] == {"input_tokens": 5, "cache_read_input_tokens": 7, "output_tokens": 2}

    def test_values_that_are_not_counts_are_ignored(self):
        from signalvault.client import _anthropic_usage

        assert _anthropic_usage(MagicMock(input_tokens=5, output_tokens=2)) == {"input_tokens": 5, "output_tokens": 2}
        assert _anthropic_usage({"input_tokens": -1, "cache_read_input_tokens": True, "output_tokens": "3"}) == {
            "input_tokens": 0, "output_tokens": 0,
        }
        assert _anthropic_usage(None) == {"input_tokens": 0, "output_tokens": 0}

    @pytest.mark.parametrize("mirror", [False, True])
    def test_stream_keeps_message_start_counts_and_takes_output_from_message_delta(self, mirror):
        events = anthropic_sse_events(
            start_usage={**CACHED_USAGE, "output_tokens": 1}, delta_usage={"output_tokens": 300},
        )
        client, server = anthropic_streaming_client(AnthropicProvider(events=events), mirror=mirror)
        with anthropic_create(client) as stream:
            list(stream)
        client.flush()
        [response] = responses(server)
        assert response["usage"] == CACHED_USAGE

    async def test_stream_counts_in_message_delta_override_message_start(self):
        events = anthropic_sse_events(
            start_usage={"input_tokens": 1, "cache_read_input_tokens": 1, "output_tokens": 1},
            delta_usage=CACHED_USAGE,
        )
        client, server = async_anthropic_streaming_client(AnthropicProvider(events=events))
        async with await anthropic_create(client) as stream:
            [e async for e in stream]
        await client.flush()
        [response] = responses(server)
        assert response["usage"] == CACHED_USAGE

    def test_stream_cut_short_records_the_message_start_counts(self):
        events = anthropic_sse_events(start_usage={**CACHED_USAGE, "output_tokens": 1})
        client, server = anthropic_streaming_client(AnthropicProvider(events=events))
        with anthropic_create(client) as stream:
            read_until_hello_event(stream)
        client.flush()
        [response] = responses(server)
        assert response["usage"] == {**CACHED_USAGE, "output_tokens": 1}
