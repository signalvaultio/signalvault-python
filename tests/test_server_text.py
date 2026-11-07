"""No free text from a SignalVault answer reaches a warning or an error message.

Only the status and, when the body gives one, a code-like value
(``^[a-z][a-z0-9_]{0,63}$``, from ``error`` when it is a string, else
``error.code``, else ``error.type``).
"""

from __future__ import annotations

import warnings

import pytest

from signalvault import (
    SignalVaultBlockedError,
    SignalVaultUnavailableError,
    SignalVaultViolationWarning,
    SignalVaultWarning,
)

from .test_security import MESSAGES, Server, async_client, sync_client

FAKE = "nope\n[SignalVault] All rules passed"
DESCRIPTION = {402: "SignalVault account is not active", 403: "SignalVault denied access"}
OPEN = "Guardrails were NOT applied — requests are being sent to the provider unchecked (fail_mode='open')."
CLOSED = "Request not sent (fail_mode='closed')."

NAMED = [
    (402, {"error": {"type": "subscription_inactive", "code": "subscription_inactive"}}, "402: subscription_inactive"),
    (402, {"error": "subscription_inactive"}, "402: subscription_inactive"),
    (403, {"error": "environment_not_allowed", "message": "This key cannot send production events."},
     "403: environment_not_allowed"),
    (403, {"error": {"code": "app_inactive"}}, "403: app_inactive"),
]
SERVER_TEXT = [
    {"error": FAKE},
    {"error": {"type": FAKE}},
    {"error": {"code": FAKE, "type": "subscription_inactive"}},
    {"error": "Not Allowed", "message": FAKE},
    {"message": FAKE},
    {},
]
SERVER_TEXT_IDS = ["error", "error.type", "error.code", "not-a-code", "message-only", "empty"]


def open_warning(status: int, body: dict) -> str:
    server = Server({"ai.request": [{"status": status, "json": body}]})
    client = sync_client(server)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()
    found = [str(w.message) for w in caught if issubclass(w.category, SignalVaultWarning)
             and "Guardrails were NOT applied" in str(w.message)]
    assert len(found) == 1, found
    return found[0]


def closed_error(status: int, body: dict) -> str:
    server = Server({"ai.request": [{"status": status, "json": body}]})
    client = sync_client(server, fail_mode="closed")
    with pytest.raises(SignalVaultUnavailableError) as exc:
        client.chat.completions.create(model="m", messages=MESSAGES)
    assert exc.value.status == status
    return str(exc.value)


@pytest.mark.parametrize("status,body,label", NAMED)
def test_open_warning_names_the_code(status, body, label):
    assert open_warning(status, body) == f"[SignalVault] {DESCRIPTION[status]} ({label}). {OPEN}"


@pytest.mark.parametrize("status,body,label", NAMED)
def test_closed_error_names_the_code(status, body, label):
    assert closed_error(status, body) == (
        f"[SignalVault] Guardrail check unavailable: {DESCRIPTION[status]} ({label}). {CLOSED}"
    )


@pytest.mark.parametrize("body", SERVER_TEXT, ids=SERVER_TEXT_IDS)
@pytest.mark.parametrize("status", [402, 403])
def test_open_warning_with_server_text_gives_the_bare_status(status, body):
    assert open_warning(status, body) == f"[SignalVault] {DESCRIPTION[status]} ({status}). {OPEN}"


@pytest.mark.parametrize("body", SERVER_TEXT, ids=SERVER_TEXT_IDS)
@pytest.mark.parametrize("status", [402, 403])
def test_closed_error_with_server_text_gives_the_bare_status(status, body):
    assert closed_error(status, body) == (
        f"[SignalVault] Guardrail check unavailable: {DESCRIPTION[status]} ({status}). {CLOSED}"
    )


@pytest.mark.parametrize("status", [402, 403])
async def test_async_closed_error_with_server_text_gives_the_bare_status(status):
    server = Server({"ai.request": [{"status": status, "json": {"error": FAKE}}]})
    client = async_client(server, fail_mode="closed")
    with pytest.raises(SignalVaultUnavailableError) as exc:
        await client.chat.completions.create(model="m", messages=MESSAGES)
    assert str(exc.value) == f"[SignalVault] Guardrail check unavailable: {DESCRIPTION[status]} ({status}). {CLOSED}"


# Violations: their types are SignalVault's codes (contains_pii, keyword_blocklist, ...).

BAD_VIOLATIONS = [
    {"rule_id": "r1", "type": FAKE, "action": "block", "details": {"note": FAKE}},
    {"rule_id": "r2", "type": "contains_pii", "action": "block", "details": {"note": FAKE}},
    {"rule_id": "r3", "type": "Keyword Blocklist", "action": "block"},
]


@pytest.mark.parametrize("violations,types", [
    (BAD_VIOLATIONS, "contains_pii"),
    ([BAD_VIOLATIONS[0]], "policy"),
    ([{"type": "contains_pii"}, {"type": "keyword_blocklist"}, {"type": "contains_pii"}],
     "contains_pii, keyword_blocklist"),
], ids=["mixed", "only-text", "codes"])
def test_blocked_error_names_only_code_like_types(violations, types):
    server = Server({"ai.request": [{"status": 200, "json": {"decision": "block", "violations": violations}}]})
    client = sync_client(server)
    with pytest.raises(SignalVaultBlockedError) as exc:
        client.chat.completions.create(model="m", messages=MESSAGES)
    assert str(exc.value) == f"[SignalVault] Request blocked by guardrails ({types})."
    assert exc.value.violations[0].type == violations[0]["type"]  # the data itself is kept


def test_blocked_error_from_a_proxy_refusal_names_only_code_like_types():
    body = {"error": {"type": "invalid_request_error", "code": "content_policy_violation",
                      "message": FAKE, "violations": BAD_VIOLATIONS}}
    server = Server({"ai.request": [{"status": 400, "json": body}]})
    client = sync_client(server)
    with pytest.raises(SignalVaultBlockedError) as exc:
        client.chat.completions.create(model="m", messages=MESSAGES)
    assert str(exc.value) == "[SignalVault] Request blocked by guardrails (contains_pii)."


def test_debug_warn_decision_names_only_code_like_types():
    server = Server({"ai.request": [{"status": 200, "json": {"decision": "warn", "violations": [
        {**v, "action": "warn"} for v in BAD_VIOLATIONS
    ]}}]})
    client = sync_client(server, debug=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.chat.completions.create(model="m", messages=MESSAGES)
    found = [str(w.message) for w in caught if issubclass(w.category, SignalVaultViolationWarning)]
    assert found == ["[SignalVault] Warnings from guardrails (contains_pii)."]
