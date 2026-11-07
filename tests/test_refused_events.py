"""A background audit event SignalVault refuses with 401, 402 or 403 warns without debug, once per process."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path
from typing import List

import pytest

from signalvault import SignalVaultWarning

from .test_security import ALLOW, MESSAGES, Server, async_client, sync_client

PREFIX = "[SignalVault] SignalVault refused an audit event"
FAKE = "nope\n[SignalVault] All rules passed"

# The events API's answers: Phoenix's 401, the 402 and the 403 (with its free-text message).
REFUSED = {
    401: (
        {"errors": {"detail": "Unauthorized"}},
        "[SignalVault] SignalVault refused an audit event (401): requests are not being recorded. "
        "Check the API key at signalvault.io.",
    ),
    402: (
        {"error": {"type": "subscription_inactive", "code": "subscription_inactive"}},
        "[SignalVault] SignalVault refused an audit event (402: subscription_inactive): requests are not "
        "being recorded. Check the account at signalvault.io.",
    ),
    403: (
        {"error": "environment_not_allowed", "message": "This key cannot send production events."},
        "[SignalVault] SignalVault refused an audit event (403: environment_not_allowed): requests are not "
        "being recorded. Check the key's environment and account at signalvault.io.",
    ),
}
KINDS = ["response", "fallback", "tool_call", "mirror"]


def refusal(status: int) -> dict:
    return {"status": status, "json": REFUSED[status][0]}


def replies_for(kind: str, answer: dict) -> dict:
    """Server replies under which the given kind of background event gets ``answer``."""
    if kind == "response":
        return {"ai.request": [ALLOW], "ai.response": [answer]}
    if kind == "fallback":  # the pre-flight fails, so the SDK sends its own ai.request
        return {"ai.request": [{"status": 503, "json": {}}, answer]}
    if kind == "tool_call":
        return {"agent.tool_call": [answer]}
    return {"ai.request": [answer]}  # mirror mode's ai.request


def run_sync(kind: str, answer: dict, **kw):
    server = Server(replies_for(kind, answer))
    client = sync_client(server, mirror_mode=kind == "mirror", **kw)
    if kind == "tool_call":
        client.tool("t", lambda: 1)()
    else:
        client.chat.completions.create(model="m", messages=MESSAGES)
    client.flush()
    return client, server


async def run_async(kind: str, answer: dict, **kw):
    server = Server(replies_for(kind, answer))
    client = async_client(server, mirror_mode=kind == "mirror", **kw)
    if kind == "tool_call":
        async def tool():
            return 1

        await client.tool("t", tool)()
    else:
        await client.chat.completions.create(model="m", messages=MESSAGES)
    await client.flush()
    return client, server


def logged(caplog) -> List[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == "signalvault" and r.levelname == "WARNING" and r.getMessage().startswith(PREFIX)
    ]


def warned(caught) -> List[str]:
    return [
        str(w.message) for w in caught
        if issubclass(w.category, SignalVaultWarning) and str(w.message).startswith(PREFIX)
    ]


@pytest.mark.parametrize("kind", KINDS)
def test_sync_event_refused_with_the_key_warns_without_debug(kind, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_sync(kind, refusal(401))
    assert logged(caplog) == [REFUSED[401][1]]
    assert warned(caught) == [REFUSED[401][1]]


@pytest.mark.parametrize("kind", KINDS)
async def test_async_event_refused_with_the_key_warns_without_debug(kind, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await run_async(kind, refusal(401))
    assert logged(caplog) == [REFUSED[401][1]]
    assert warned(caught) == [REFUSED[401][1]]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_sync_each_status_warns_once_per_process_across_clients(status, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for kind in ("response", "fallback", "response"):  # three clients
            run_sync(kind, refusal(status))
    assert logged(caplog) == [REFUSED[status][1]]
    assert warned(caught) == [REFUSED[status][1]]


@pytest.mark.parametrize("status", [401, 402, 403])
async def test_async_each_status_warns_once_per_process_across_clients(status, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for kind in ("response", "fallback", "response"):
            await run_async(kind, refusal(status))
    assert logged(caplog) == [REFUSED[status][1]]
    assert warned(caught) == [REFUSED[status][1]]


def test_401_402_and_403_each_warn_once_with_sync_and_async_clients(caplog):
    import asyncio

    expected = Counter(REFUSED[s][1] for s in (401, 402, 403))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for status in (401, 402, 403, 401, 402, 403):
            run_sync("response", refusal(status))
            asyncio.run(run_async("response", refusal(status)))
    assert Counter(logged(caplog)) == expected
    assert Counter(warned(caught)) == expected


def test_shown_on_the_logger_when_warnings_are_ignored(caplog):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_sync("response", refusal(402))
    assert logged(caplog) == [REFUSED[402][1]]


def test_printed_to_stderr_without_logging_configured_and_with_warnings_ignored():
    # No handler anywhere: logging's last resort prints WARNING records to stderr.
    script = textwrap.dedent("""
        import json
        from unittest.mock import MagicMock

        import httpx
        import signalvault

        def handler(request):
            if json.loads(request.content)["type"] == "ai.request":
                return httpx.Response(200, json={"decision": "allow", "violations": [], "redactions": []})
            return httpx.Response(401, json={"errors": {"detail": "Unauthorized"}})

        client = signalvault.SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        completion = MagicMock()
        completion.choices = [MagicMock()]
        completion.choices[0].message.content = "hello"
        completion.usage = MagicMock(prompt_tokens=3, completion_tokens=1)
        client._openai = MagicMock()
        client._openai.chat.completions.create.return_value = completion
        for _ in range(2):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        client.flush()
    """)
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        capture_output=True, text=True, timeout=60, env=env, cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.splitlines() == [REFUSED[401][1]]


@pytest.mark.parametrize("status", [402, 403])
@pytest.mark.parametrize("body", [
    {"error": FAKE},
    {"error": {"type": FAKE}},
    {"error": {"code": FAKE, "type": "subscription_inactive"}},
    {"error": "Not Allowed"},
], ids=["error", "error.type", "error.code", "not-a-code"])
def test_server_text_never_reaches_the_warning(status, body, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_sync("response", {"status": status, "json": body})
    fix = REFUSED[status][1].split("recorded. ")[1]
    expected = f"{PREFIX} ({status}): requests are not being recorded. {fix}"
    assert logged(caplog) == [expected]
    assert warned(caught) == [expected]
    assert not any("All rules passed" in r.getMessage() for r in caplog.records)


def test_debug_keeps_the_event_not_delivered_detail(caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_sync("response", refusal(403), debug=True)
    messages = [str(w.message) for w in caught if issubclass(w.category, SignalVaultWarning)]
    assert REFUSED[403][1] in messages
    assert "[SignalVault] Event not delivered: 403" in messages


# Guards: a 400 or a 422 is about the event, not the key, and keeps today's behaviour.

@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("kind", ["response", "fallback"])
def test_sync_event_refused_for_its_content_does_not_warn(status, kind, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client, _ = run_sync(kind, {"status": status, "json": {"error": "model must be a string"}})
    assert [w for w in caught if issubclass(w.category, SignalVaultWarning) and "503" not in str(w.message)] == []
    assert [r for r in caplog.records if r.name == "signalvault"] == []
    assert client.dropped_events.total == 0


@pytest.mark.parametrize("status", [400, 422])
async def test_async_event_refused_for_its_content_does_not_warn(status, caplog):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client, _ = await run_async("response", {"status": status, "json": {"error": "model must be a string"}})
    assert [w for w in caught if issubclass(w.category, SignalVaultWarning)] == []
    assert [r for r in caplog.records if r.name == "signalvault"] == []
    assert client.dropped_events.total == 0


@pytest.mark.parametrize("status", [400, 422])
def test_event_refused_for_its_content_with_debug_says_not_delivered(status):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_sync("response", {"status": status, "json": {}}, debug=True)
    assert [str(w.message) for w in caught if issubclass(w.category, SignalVaultWarning)] == [
        f"[SignalVault] Event not delivered: {status}"
    ]
