"""SignalVault client — wraps OpenAI and Anthropic with guardrails and audit logging."""

from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import dataclasses
import datetime as _datetime
import decimal
import enum
import functools
import inspect
import json
import logging
import os
import platform
import random
import sys
import threading
import time
import traceback
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

try:
    from pydantic import BaseModel as _PydanticModel
except ImportError:  # pragma: no cover - openai and anthropic both depend on pydantic
    _PydanticModel = None

from . import __version__ as SDK_VERSION
from .tools import (
    ToolContext,
    ToolRecordOptions,
    build_tool_call_body,
    get_current_request_id,
)


DEFAULT_BASE_URL = "https://api.signalvault.io"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Background (audit) events: up to this many attempts each, about 4, 8, 16,
# 32 and 60 seconds apart, so about two minutes in all.
_MAX_ATTEMPTS = 6
_BACKOFF_BASE_SECONDS = 4.0
# Longest wait before a retry, also when a 429's Retry-After asks for more.
_MAX_RETRY_DELAY_SECONDS = 60.0
# Audit events waiting to be sent, per client. Above it the oldest are dropped.
_MAX_QUEUED_EVENTS = 1_000
# Threads sending audit events, per sync client.
_DELIVERY_WORKERS = 4
# An idle delivery thread waits this long for more events before it exits.
_WORKER_IDLE_SECONDS = 1.0
# Default for exit_flush_timeout: how long interpreter exit waits for queued audit events.
_EXIT_FLUSH_SECONDS = 10.0
# While flush(), close() or interpreter exit is waiting, retries wait at most this long.
_FLUSH_RETRY_WAIT_SECONDS = 1.0
# A repeated warning is emitted at most once per this window.
_WARN_INTERVAL_SECONDS = 60.0
_DECISIONS = {"allow", "warn", "block", "redact"}
_FAIL_MODES = {"open", "closed"}
# Pre-flight failures where re-sending the ai.request later can succeed.
_RECORDABLE_CAUSES = {"timeout", "network", "5xx", "invalid-response"}
# Pre-flight failures where the server stored the ai.request without running the rules.
_RECORDED_CAUSES = {"429-recorded"}
# Passed as the fallback when no audit events should be sent at all.
_NO_AUDIT: Any = object()
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_MISSING: Any = object()

# Every dropped audit event is logged here, with its event_id, at WARNING.
logger = logging.getLogger("signalvault")


def _emit_warning(message: str, category: type = UserWarning) -> None:
    """Warns from the first stack frame outside this package, every time.

    ``warnings.warn`` would report from a line inside the SDK, and Python's
    default filter shows a given message from a given line only once per
    process — so a second outage hours later would be silent. A fresh
    registry per call avoids that; callers rate-limit instead.
    """
    frame = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR + os.sep):
        frame = frame.f_back
    if frame is None:
        filename, lineno, module = "<signalvault>", 0, "signalvault"
    else:
        filename, lineno = frame.f_code.co_filename, frame.f_lineno
        module = frame.f_globals.get("__name__", "")
    warnings.warn_explicit(message, category, filename, lineno, module=module, registry={})


# ---------------------------------------------------------------------------
# Config & shared types
# ---------------------------------------------------------------------------

@dataclass
class SignalVaultConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    environment: str = "production"
    debug: bool = False
    mirror_mode: bool = False
    # "open": allow and warn when no guardrail decision can be obtained.
    # "closed": raise SignalVaultUnavailableError instead.
    fail_mode: str = "open"
    # Timeout for pre-flight /v1/events call (critical path).
    preflight_timeout: float = 2.0
    # Timeout for background/post-flight calls.
    timeout: float = 30.0
    # Default metadata attached to every event. Merged with per-call sv_metadata.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Seconds the interpreter's exit waits for queued audit events (sync
    # clients). 0 sends nothing more at exit.
    exit_flush_timeout: float = _EXIT_FLUSH_SECONDS


@dataclass(frozen=True)
class DroppedEvents:
    """Audit events a client gave up on, by cause. Read it from ``client.dropped_events``.

    Each drop is also logged, with the event ids, on the ``signalvault`` logger.
    """

    # Every attempt failed (connection error, timeout, 429 or 5xx).
    retries_exhausted: int = 0
    # The queue was full, so the oldest waiting events were dropped.
    queue_full: int = 0
    # Still unsent when close() / aclose() or the interpreter's exit flush
    # ran out of time, or when the async task sending them was cancelled.
    shutdown: int = 0
    # The event could not be serialized to JSON.
    unserializable: int = 0

    @property
    def total(self) -> int:
        return self.retries_exhausted + self.queue_full + self.shutdown + self.unserializable


# How each cause reads in the log and the warnings.
_DROP_CAUSES = {
    "retries_exhausted": "every attempt failed",
    "queue_full": "the queue was full",
    "shutdown": "still unsent at shutdown",
    "unserializable": "could not be serialized",
}


@dataclass
class Violation:
    rule_id: Optional[str] = None
    type: str = ""
    severity: int = 0
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    decision: str = "allow"
    violations: List[Violation] = field(default_factory=list)
    # [{"type": ..., "count": ...}] for rules whose redact action matched.
    # Redaction applies to what SignalVault stores; the request sent to the
    # provider is not modified. Check len(), never truthiness.
    redactions: List[Dict[str, Any]] = field(default_factory=list)
    dashboard_url: Optional[str] = None
    # Set when no decision was obtained: "record" if the server may not have
    # stored the request (network, timeout, 5xx, invalid response), "skip" if
    # it refused it (401/402/403/429/3xx), "recorded" if it stored it without
    # running the rules (a 429 with "recorded": true). Internal.
    _preflight_failed: Optional[str] = field(default=None, repr=False, compare=False)


class SignalVaultWarning(UserWarning):
    """Emitted when SignalVault cannot check or record a request."""


class SignalVaultViolationWarning(SignalVaultWarning):
    """Emitted when a guardrail returns a ``warn`` decision (with ``debug=True``)."""


class SignalVaultBlockedError(RuntimeError):
    """Raised when a guardrail rule blocks the request.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` handlers
    keep working.
    """

    def __init__(
        self, request_id: str, violations: List[Violation], dashboard_url: Optional[str] = None,
    ) -> None:
        types = ", ".join(dict.fromkeys(str(v.type) for v in violations if v.type)) or "policy"
        super().__init__(f"[SignalVault] Request blocked by guardrails ({types}).")
        self.request_id = request_id
        self.violations = violations
        self.dashboard_url = dashboard_url


class SignalVaultUnavailableError(RuntimeError):
    """Raised with ``fail_mode="closed"`` when no guardrail decision could be obtained."""

    def __init__(self, request_id: str, reason: str, status: Optional[int] = None) -> None:
        super().__init__(
            f"[SignalVault] Guardrail check unavailable: {reason}. "
            f"Request not sent (fail_mode='closed')."
        )
        self.request_id = request_id
        self.status = status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_base_url(raw: str) -> str:
    """Strips trailing slashes and refuses plaintext HTTP except to localhost.

    The API key and full prompts travel in every request, so an ``http://``
    URL would expose both before the server could redirect to HTTPS.
    """
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"[SignalVault] base_url must be an https:// URL, got {raw!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"[SignalVault] base_url must not contain a query or fragment, got {raw!r}")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise ValueError(
            "[SignalVault] base_url must use https:// (plain http:// is only allowed for "
            "localhost). Your API key and prompts would otherwise be sent unencrypted."
        )
    return raw.rstrip("/")


def _normalize_api_key(raw: Any) -> str:
    """Strips surrounding whitespace and refuses keys that can't go in a header.

    A key read from a file or a secret often ends in a newline. Sent as is,
    every request fails before it leaves the process, so with fail_mode="open"
    every call would go through unchecked. Errors never include the key.
    """
    if not isinstance(raw, str):
        raise TypeError(f"[SignalVault] api_key must be a string, got {type(raw).__name__}")
    key = raw.lstrip("\ufeff").strip()  # a BOM from a file saved by some editors
    if not key:
        raise ValueError("[SignalVault] api_key is empty")
    if any(not "\x21" <= ch <= "\x7e" for ch in key):
        raise ValueError(
            "[SignalVault] api_key contains whitespace, control or non-ASCII characters. "
            "Check how the key is loaded."
        )
    return key


def _build_config(
    api_key: str, base_url: str, environment: str, debug: bool, mirror_mode: bool,
    fail_mode: str, preflight_timeout: float, timeout: float,
    metadata: Optional[Dict[str, Any]], exit_flush_timeout: float = _EXIT_FLUSH_SECONDS,
) -> SignalVaultConfig:
    if fail_mode not in _FAIL_MODES:
        raise ValueError("[SignalVault] fail_mode must be 'open' or 'closed'")
    if isinstance(exit_flush_timeout, bool) or not isinstance(exit_flush_timeout, (int, float)) \
            or not 0 <= exit_flush_timeout < float("inf"):
        raise ValueError("[SignalVault] exit_flush_timeout must be a number of seconds, 0 or more")
    return SignalVaultConfig(
        api_key=_normalize_api_key(api_key),
        base_url=normalize_base_url(base_url),
        environment=environment,
        debug=debug,
        mirror_mode=mirror_mode,
        fail_mode=fail_mode,
        preflight_timeout=preflight_timeout,
        timeout=timeout,
        metadata=metadata or {},
        exit_flush_timeout=float(exit_flush_timeout),
    )


def _with_event_id(body: dict) -> dict:
    """The body with an idempotency key, so a retry is never double-counted."""
    return {"event_id": str(uuid.uuid4()), **body}


def _event_with_id(body: dict, event_id: Optional[str]) -> dict:
    """``body`` with the event_id chosen when it was queued, or a new one."""
    return {"event_id": event_id, **body} if event_id else _with_event_id(body)


def _resolve(waiter: "asyncio.Future[Any]") -> None:
    if not waiter.done():
        waiter.set_result(None)


def _event_ref(event: dict) -> str:
    """How an event is named in the drop log: its event_id, or for the fallback ai.request its request_id."""
    if "event_id" in event:
        return f"event_id={event['event_id']}"
    return f"request_id={event.get('request_id')} (fallback ai.request, which has no event_id)"


def _merge_metadata(config_meta: Dict[str, Any], call_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**config_meta, **(call_meta or {})}


def _parse_decision(data: dict) -> Decision:
    known = Violation.__dataclass_fields__
    violations = [
        Violation(**{k: v for k, v in item.items() if k in known})
        for item in (data.get("violations") if isinstance(data.get("violations"), list) else [])
        if isinstance(item, dict)
    ]
    redactions = data.get("redactions")
    dashboard_url = data.get("dashboard_url")
    return Decision(
        decision=data.get("decision", "allow"),
        violations=violations,
        redactions=redactions if isinstance(redactions, list) else [],
        dashboard_url=dashboard_url if isinstance(dashboard_url, str) else None,
    )


def _anthropic_messages_for_audit(params: dict) -> list:
    """Includes Anthropic's top-level ``system`` prompt as a system message.

    The server scans ``payload.messages``; without this, secrets or PII in the
    system prompt go unchecked.
    """
    messages = list(params.get("messages") or [])
    system = params.get("system")
    if system in (None, "", []):
        return messages
    return [{"role": "system", "content": system}, *messages]


def _anthropic_output_text(response: Any) -> str:
    """Concatenates every text block; ``content[0]`` alone misses text after a tool_use block."""
    blocks = getattr(response, "content", None) or []
    return "".join(
        getattr(b, "text", "") or ""
        for b in blocks
        if getattr(b, "type", "text") == "text" and isinstance(getattr(b, "text", None), str)
    )


def _openai_usage(usage: Any) -> Dict[str, Any]:
    return {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


_ANTHROPIC_USAGE_KEYS = (
    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens",
)

# ``usage.cache_creation``: the cache writes by cache lifetime, priced differently.
_CACHE_CREATION_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


def _copy_counts(target: Dict[str, Any], source: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    for key in keys:
        n = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
            target[key] = n
    return target


def _anthropic_usage(*sources: Any) -> Dict[str, Any]:
    """Anthropic usage under Anthropic's own names, as the proxy records it.

    The server adds the cache counts to ``input_tokens``; renamed to
    ``prompt_tokens``, which for OpenAI already includes cached tokens, they
    would be lost. Later sources override earlier ones, but only with a count.
    """
    usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    cache_creation: Dict[str, Any] = {}
    for source in sources:
        _copy_counts(usage, source, _ANTHROPIC_USAGE_KEYS)
        breakdown = source.get("cache_creation") if isinstance(source, dict) else getattr(source, "cache_creation", None)
        _copy_counts(cache_creation, breakdown, _CACHE_CREATION_KEYS)
    if cache_creation:
        usage["cache_creation"] = cache_creation
    return usage


def _dump_model(obj: Any) -> Any:
    """The fields a pydantic model was given (by the provider or the caller), under pydantic v1 or v2."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json", exclude_unset=True)
        except Exception:
            # The openai/anthropic shim for pydantic v1 rejects mode="json".
            return obj.model_dump(exclude_unset=True)
    return obj.dict(exclude_unset=True)


def _json_default(obj: Any) -> Any:
    """Converts what ``json`` can't: provider SDK objects in ``messages`` and similar.

    The usual tool-calling loop appends ``response.choices[0].message`` (a
    pydantic model) or Anthropic content blocks to the conversation. Only the
    fields the provider returned are kept, which matches what the Node SDK
    sends for the same objects. Never raises: anything that can't be
    converted falls back to its ``repr``, so one odd value can't stop the
    request from being checked.
    """
    try:
        if _PydanticModel is not None and isinstance(obj, _PydanticModel):
            return _dump_model(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, (_datetime.date, _datetime.time)):
            return obj.isoformat()
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, (uuid.UUID, decimal.Decimal)):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", "replace")
    except Exception:
        pass
    try:
        return repr(obj)
    except Exception:
        return f"<unrepresentable {type(obj).__name__}>"


def _encode_body(body: dict) -> bytes:
    """Serializes an event body like httpx's ``json=``, plus ``_json_default``.

    ``ensure_ascii`` keeps a lone surrogate in user text encodable (as
    ``\\udXXX``, as Node's ``JSON.stringify`` sends it) instead of failing.
    Raises only for a cycle, NaN or extreme nesting.
    """
    return json.dumps(
        body, default=_json_default, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")


def _retryable(resp: Optional[httpx.Response]) -> bool:
    """Whether a later attempt can succeed: no response (connection error or timeout), 429 or 5xx."""
    return resp is None or resp.status_code == 429 or resp.status_code >= 500


def _retry_delay(resp: Optional[httpx.Response], attempt: int, rand: Callable[[], float]) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (from 1) before the next one.

    A 429's Retry-After is honoured, up to a minute. Otherwise exponential
    backoff with jitter: about 4, 8, 16, 32, then 60 seconds.
    """
    if resp is not None and resp.status_code == 429:
        try:
            retry_after = float(resp.headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = -1.0
        if retry_after >= 0:  # False for NaN too
            return min(retry_after, _MAX_RETRY_DELAY_SECONDS)
    backoff = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1) * (0.8 + 0.4 * rand())
    return min(backoff, _MAX_RETRY_DELAY_SECONDS)


def _undelivered_cause(resp: Optional[httpx.Response]) -> str:
    if resp is None:
        return "SignalVault API unreachable or timed out"
    if resp.status_code == 429:
        return "rate limited, 429"
    return f"SignalVault API error, {resp.status_code}"


def _describe_failure(resp: httpx.Response) -> Tuple[str, str]:
    """Maps a non-2xx pre-flight response to (warning key, human reason)."""
    try:
        body = resp.json()
    except Exception:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else error
    status = resp.status_code

    if status == 401:
        return "401", "SignalVault rejected the API key (401: invalid or revoked)"
    if status == 402:
        return "402", f"SignalVault account is not active (402: {error_type or 'payment required'})"
    if status == 403:
        return "403", f"SignalVault denied access (403: {error_type or 'forbidden'})"
    if status == 429:
        if isinstance(body, dict) and body.get("recorded") is True:
            # Over the rule-check rate limit: stored without a decision, so
            # the audit events that follow are accepted.
            return "429-recorded", "SignalVault rule checks are rate limited (429)"
        if error_type == "trial_limit_exceeded":
            return "429-trial", "SignalVault trial limit reached (429)"
        return "429", "SignalVault rate limit reached (429)"
    if 300 <= status < 400:
        return "3xx", f"SignalVault API redirected ({status}); check that base_url is the https:// API URL"
    if status >= 500:
        return "5xx", f"SignalVault API error ({status})"
    return str(status), f"SignalVault rejected the pre-flight check ({status})"


class _ClientCommon:
    """Config, headers, warnings and decision handling shared by all clients."""

    _provider: str = ""  # overridden by subclasses

    def _init_common(self, config: SignalVaultConfig) -> None:
        self._config = config
        self._last_warned: Dict[str, float] = {}
        self._warn_lock = threading.Lock()
        # Audit events given up on, by DroppedEvents field.
        self._dropped: Dict[str, int] = {name: 0 for name in _DROP_CAUSES}
        # Events a delivery thread or task has taken and not finished, and
        # those reported as dropped at shutdown while their sender may still run.
        self._unsent: Dict[str, None] = {}
        self._abandoned: set = set()
        self._refs_lock = threading.Lock()
        # Injectable for tests: jitter for the retry backoff.
        self._random: Callable[[], float] = random.random

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "User-Agent": f"signalvault-python/{SDK_VERSION} python/{platform.python_version()}",
        }

    def _audit_messages(self, params: dict) -> list:
        """Messages as sent to SignalVault. Anthropic clients add the system prompt."""
        return list(params.get("messages") or [])

    def _warn(self, key: str, message: str) -> None:
        """Warns at most once per minute per key. Not gated on debug."""
        now = time.monotonic()
        with self._warn_lock:
            last = self._last_warned.get(key)
            if last is not None and now - last < _WARN_INTERVAL_SECONDS:
                return
            self._last_warned[key] = now
        _emit_warning(message, SignalVaultWarning)

    def _unavailable(
        self, request_id: str, key: str, reason: str, status: Optional[int] = None,
    ) -> Decision:
        """No decision could be obtained: raise in fail_mode 'closed', else warn and allow."""
        if self._config.fail_mode == "closed":
            raise SignalVaultUnavailableError(request_id, reason, status)
        recorded = key in _RECORDED_CAUSES
        self._warn(
            f"preflight:{key}",
            f"[SignalVault] {reason}. Guardrails were NOT applied — requests are being sent "
            f"to the provider unchecked (fail_mode='open')."
            + (" They are still recorded in the audit log." if recorded else ""),
        )
        if recorded:
            return Decision(_preflight_failed="recorded")
        return Decision(_preflight_failed="record" if key in _RECORDABLE_CAUSES else "skip")

    def _fallback_request(self, decision: Decision, request_id: str, params: dict, metadata: dict) -> Any:
        """What to send before the response when the pre-flight got no decision.

        None: the server stored the pre-flight, with a decision or (over the
        rule-check rate limit) without one. ``_NO_AUDIT``: the server refused
        it, so a response event would be refused too. Otherwise the ai.request
        body, so an unchecked call still reaches the audit log.
        """
        if decision._preflight_failed in (None, "recorded"):
            return None
        if decision._preflight_failed == "skip":
            return _NO_AUDIT
        body = self._preflight_body(request_id, params, metadata)
        body["payload"]["preflight_unavailable"] = True
        return body

    def _encode_preflight(self, request_id: str, params: dict, metadata: dict) -> Any:
        """The encoded pre-flight body, or the fail-mode Decision when it can't be serialized."""
        try:
            return _encode_body(self._preflight_body(request_id, params, metadata))
        except Exception as exc:
            return self._unavailable(
                request_id, "unserializable",
                f"The request could not be serialized for SignalVault ({type(exc).__name__}: {exc})",
            )

    def _encode_event(self, event: dict) -> Optional[bytes]:
        """The encoded event, or None (dropped, with a warning) when it can't be serialized."""
        try:
            return _encode_body(event)
        except Exception as exc:
            if self._settle(_event_ref(event)):
                self._drop(
                    "unserializable", [_event_ref(event)], f"{type(exc).__name__}: {exc}",
                    warning=(
                        f"unserializable:{event.get('type')}",
                        f"[SignalVault] {event.get('type')} event dropped: it could not be serialized "
                        f"({type(exc).__name__}: {exc}).",
                    ),
                )
            return None

    def _preflight_body(self, request_id: str, params: dict, metadata: dict) -> dict:
        return {
            "type": "ai.request",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": params.get("model", ""),
            "metadata": metadata,
            # Tells the server whether a request it can't check is sent on
            # ("open") or not ("closed"). Mirror mode never waits for a
            # decision, so its ai.request has no fail_mode.
            "payload": {"messages": self._audit_messages(params), "fail_mode": self._config.fail_mode},
        }

    def _interpret_preflight(self, request_id: str, resp: httpx.Response) -> Decision:
        if not 200 <= resp.status_code < 300:
            key, reason = _describe_failure(resp)
            return self._unavailable(request_id, key, reason, resp.status_code)
        try:
            data = resp.json()
        except ValueError:
            return self._unavailable(request_id, "invalid-response", "SignalVault returned an invalid response")
        decision = data.get("decision") if isinstance(data, dict) else None
        if not isinstance(decision, str) or decision not in _DECISIONS:
            return self._unavailable(request_id, "invalid-response", "SignalVault returned an invalid decision")
        return _parse_decision(data)

    def _enforce(self, request_id: str, decision: Decision) -> None:
        if decision.decision == "block":
            raise SignalVaultBlockedError(request_id, decision.violations, decision.dashboard_url)
        if decision.decision == "warn" and self._config.debug:
            _emit_warning(f"[SignalVault] Warnings: {decision.violations}", SignalVaultViolationWarning)

    def _response_body(
        self, request_id: str, model: str, output: str,
        usage: Dict[str, Any], metadata: dict, monitor_mode: bool = False,
    ) -> dict:
        payload: Dict[str, Any] = {
            "output": output,
            "usage": usage,
        }
        if monitor_mode:
            payload["monitor_mode"] = True
        return {
            "type": "ai.response",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
            "payload": payload,
        }

    def _mirror_request_body(self, request_id: str, model: str, messages: list, metadata: dict) -> dict:
        return {
            "type": "ai.request",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
            "payload": {"messages": messages, "monitor_mode": True},
        }

    def _log_undelivered(self, resp: Optional[httpx.Response], attempts: int, ref: str) -> None:
        """Reports the outcome of an event's last attempt: a drop always, a rejection only with debug."""
        if not self._settle(ref):
            return  # already reported as dropped at shutdown
        if resp is not None and 200 <= resp.status_code < 300:
            return
        if _retryable(resp):
            self._drop(
                "retries_exhausted", [ref],
                f"{attempts} attempt{'s' if attempts != 1 else ''}, last: {_undelivered_cause(resp)}",
            )
        elif self._config.debug:
            _emit_warning(f"[SignalVault] Event not delivered: {resp.status_code}", SignalVaultWarning)

    @property
    def dropped_events(self) -> DroppedEvents:
        """The audit events this client has dropped so far, by cause. Read-only snapshot."""
        with self._refs_lock:
            return DroppedEvents(**self._dropped)

    def _drop(
        self, cause: str, refs: List[str], detail: str = "",
        warning: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Counts audit events that will never be sent, logs their ids, and warns.

        The last resort: events are retried and queued first. Every drop is
        logged on the ``signalvault`` logger with the event ids (the
        request_id for a fallback ai.request); the SignalVaultWarning repeats
        at most once a minute per cause. ``warning`` replaces the default one
        as (key, message).
        """
        if not refs:
            return
        with self._refs_lock:
            self._dropped[cause] += len(refs)
            total = sum(self._dropped.values())
        why = _DROP_CAUSES[cause] + (f" ({detail})" if detail else "")
        logger.warning(
            "SignalVault dropped %d audit event(s), %s: %s", len(refs), why, ", ".join(refs),
            extra={"signalvault_drop_cause": cause, "signalvault_dropped": list(refs)},
        )
        key, message = warning or (
            f"dropped:{cause}",
            f"[SignalVault] {len(refs)} audit event(s) dropped, {why}. {total} dropped by this "
            f"client so far; the ids are logged on the 'signalvault' logger.",
        )
        self._warn(key, message)

    def _settle(self, ref: str) -> bool:
        """Marks an event as finished. False if it was already reported as dropped at shutdown."""
        with self._refs_lock:
            self._unsent.pop(ref, None)
            if ref in self._abandoned:
                self._abandoned.discard(ref)
                return False
        return True

    def _claim_unsent(self, refs: Any, sender_may_continue: bool) -> List[str]:
        """Takes the events in ``refs`` not finished yet, to be reported as dropped.

        With ``sender_may_continue``, a later ``_settle`` of one of them is
        ignored, so nothing is reported twice.
        """
        with self._refs_lock:
            claimed = [r for r in refs if self._unsent.pop(r, _MISSING) is not _MISSING]
            if sender_may_continue:
                self._abandoned.update(claimed)
        return claimed

    def _warn_tool_call_status(self, status_code: int) -> None:
        """4xx → warns regardless of debug; anything else non-2xx → only when debug=True."""
        if 200 <= status_code < 300:
            return
        if 400 <= status_code < 500:
            self._warn(
                f"tool-{status_code}",
                f"[SignalVault] tool_call rejected with {status_code}. "
                f"Check your api_key and event payload.",
            )
        elif self._config.debug:
            _emit_warning(f"[SignalVault] tool_call event failed: {status_code}", SignalVaultWarning)

    def _prepare_tool_call(self, opts: ToolRecordOptions) -> Optional[dict]:
        """Builds the tool_call body now, so it reflects the arguments at call time.

        Returns None (and warns) instead of raising — a wrapper must never
        break the user's tool because the audit could not be built.
        """
        try:
            return build_tool_call_body(
                environment=self._config.environment,
                default_metadata=self._config.metadata,
                opts=opts,
            )
        except Exception as exc:
            self._warn("tool-build", f"[SignalVault] tool_call event dropped: {exc}")
            return None


# ---------------------------------------------------------------------------
# Base sync client — shared HTTP logic for OpenAI and Anthropic sync clients
# ---------------------------------------------------------------------------

class _BaseSyncClient(_ClientCommon):
    """Shared HTTP, config, and audit logic for sync SignalVault clients."""

    def __init__(self, config: SignalVaultConfig) -> None:
        self._init_common(config)
        # Never follow redirects: the API key must not be re-sent to another URL.
        self._http = httpx.Client(follow_redirects=False)
        # Audit sends waiting for a delivery thread: (fn, args, the events it sends).
        self._queue: Deque[Tuple[Callable[..., Any], tuple, List[str]]] = collections.deque()
        self._queue_cond = threading.Condition()
        self._queued_events = 0
        self._workers = 0  # delivery threads running
        self._busy = 0  # sends in progress
        self._closing = False
        # flush(), close() and interpreter exit in progress: retries wait less.
        self._hurrying = 0
        # Injectable for tests: waits before a retry.
        self._sleep: Callable[[float], Any] = self._wait_before_retry
        _SYNC_CLIENTS.add(self)

    # -- Resource management -------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Waits up to ``timeout`` seconds for queued audit events to be sent.

        Meanwhile retries wait at most a second. Events still unsent stay queued.
        """
        with self._hurry():
            left = self._wait_idle(time.monotonic() + timeout)
        if left:
            self._warn("flush-timeout", f"[SignalVault] flush() timed out with {left} event(s) still sending.")

    def close(self, timeout: float = 10.0) -> None:
        """Sends queued audit events (up to ``timeout`` seconds), then closes the HTTP pool.

        Nothing is dropped before the deadline. Events still unsent then can't
        be sent once the pool is closed: they are dropped, logged with their
        ids and counted in ``dropped_events``.
        """
        with self._hurry():
            self._wait_idle(time.monotonic() + timeout)
        self._shut_down("the client was closed before they were sent")
        self._http.close()

    def __enter__(self) -> "_BaseSyncClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- Delivery queue ------------------------------------------------------

    def _submit(self, fn: Callable[..., Any], *args: Any, refs: Optional[List[str]] = None) -> None:
        """Queues ``fn(*args)``, which sends the events ``refs`` names, for a delivery thread.

        When more than ``_MAX_QUEUED_EVENTS`` are waiting, the oldest are dropped.
        """
        refs = refs if refs is not None else [f"{getattr(fn, '__name__', 'send')} (no event_id)"]
        with self._queue_cond:
            closed = self._closing
            dropped: List[str] = []
            start = False
            if not closed:
                self._queue.append((fn, args, refs))
                self._queued_events += len(refs)
                while self._queued_events > _MAX_QUEUED_EVENTS and len(self._queue) > 1:
                    _, _, oldest = self._queue.popleft()
                    self._queued_events -= len(oldest)
                    dropped.extend(oldest)
                # One more thread if every running one is busy, up to _DELIVERY_WORKERS.
                start = self._workers < _DELIVERY_WORKERS and self._workers - self._busy < len(self._queue)
                if start:
                    self._workers += 1
                self._queue_cond.notify_all()  # flush() and retry waits use it too
        if closed:
            self._drop("shutdown", refs, "the client is closed")
            return
        if dropped:
            self._drop(
                "queue_full", dropped,
                f"{_MAX_QUEUED_EVENTS:,} events waiting; SignalVault is unreachable or rate limiting this app",
            )
        if start:
            self._start_worker()

    def _start_worker(self) -> None:
        try:
            threading.Thread(target=self._work, name="sv-audit", daemon=True).start()
        except RuntimeError:  # can't start threads: the interpreter is shutting down
            with self._queue_cond:
                self._workers -= 1
                self._queue_cond.notify_all()

    def _work(self) -> None:
        """A delivery thread: sends queued events until none arrive for a while."""
        while True:
            with self._queue_cond:
                if not self._queue and not self._closing:
                    self._queue_cond.wait(_WORKER_IDLE_SECONDS)
                if not self._queue or self._closing:
                    self._workers -= 1
                    self._queue_cond.notify_all()
                    return
                fn, args, refs = self._queue.popleft()
                self._queued_events -= len(refs)
                self._busy += 1
                with self._refs_lock:
                    self._unsent.update(dict.fromkeys(refs))
            try:
                fn(*args)
            except Exception:
                if self._config.debug:
                    traceback.print_exc()
            finally:
                # Normally each event was settled as it finished; what's left was stopped by an error.
                left = self._claim_unsent(refs, sender_may_continue=False)
                if left:
                    logger.warning(
                        "SignalVault dropped %d audit event(s), an internal error stopped their delivery: %s",
                        len(left), ", ".join(left),
                        extra={"signalvault_drop_cause": "error", "signalvault_dropped": left},
                    )
                with self._queue_cond:
                    self._busy -= 1
                    self._queue_cond.notify_all()

    def _wait_idle(self, deadline: float) -> int:
        """Waits until nothing is queued or sending, or until ``deadline``. Returns the events left."""
        with self._queue_cond:
            while self._queued_events or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._queued_events + len(self._unsent)
                self._queue_cond.wait(remaining)
        return 0

    @contextlib.contextmanager
    def _hurry(self) -> Any:
        """While flush(), close() or the exit flush waits, retries wait at most a second."""
        with self._queue_cond:
            self._hurrying += 1
            self._queue_cond.notify_all()  # cuts retry waits in progress short
        try:
            yield
        finally:
            with self._queue_cond:
                self._hurrying -= 1

    def _wait_before_retry(self, delay: float) -> None:
        """The default ``_sleep``: ``delay`` seconds, less once someone is waiting for the queue."""
        start = time.monotonic()
        with self._queue_cond:
            while not self._closing:
                limit = min(delay, _FLUSH_RETRY_WAIT_SECONDS) if self._hurrying else delay
                remaining = start + limit - time.monotonic()
                if remaining <= 0:
                    return
                self._queue_cond.wait(remaining)

    def _pause(self, delay: float) -> bool:
        """Waits before a retry. False once the client is closing."""
        if self._hurrying:
            delay = min(delay, _FLUSH_RETRY_WAIT_SECONDS)
        self._sleep(delay)
        return not self._closing

    def _shut_down(self, why: str) -> None:
        """Stops delivery; drops (and logs, and counts) every event not sent yet."""
        with self._queue_cond:
            if self._closing:
                return
            queued = [ref for _, _, refs in self._queue for ref in refs]
            self._queue.clear()
            self._queued_events = 0
            # A sender that finishes later must not report these again.
            sending = self._claim_unsent(list(self._unsent), sender_may_continue=True)
            self._closing = True
            self._queue_cond.notify_all()
        self._drop("shutdown", queued + sending, why)

    def _start_exit(self) -> None:
        with self._queue_cond:
            self._hurrying += 1  # for good: the process is exiting
            self._queue_cond.notify_all()

    def _drain_before_exit(self, deadline: float) -> None:
        """Interpreter exit: sends what it can until ``deadline``, then drops the rest."""
        if self._closing:
            return
        with self._queue_cond:
            start = bool(self._queue) and self._workers == 0
            if start:
                self._workers += 1
        if start:
            self._start_worker()
        self._wait_idle(deadline)
        self._shut_down(
            "the process exited before they were sent; raise exit_flush_timeout "
            "or call client.flush() before exiting",
        )

    # -- Internal HTTP -------------------------------------------------------

    def _post(self, content: bytes, timeout: float) -> httpx.Response:
        return self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            timeout=timeout,
            content=content,
        )

    def _attempt(self, content: bytes) -> Optional[httpx.Response]:
        """One POST of a background event; None on a connection error or timeout."""
        try:
            return self._post(content, self._config.timeout)
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return None
        except RuntimeError:
            if self._closing:  # the pool was closed under it
                return None
            raise

    def _post_background(
        self, body: dict, *, event_id: bool = True, attempt: int = 1,
    ) -> Optional[httpx.Response]:
        """POST, retrying connection errors, timeouts, 429 and 5xx, up to ``_MAX_ATTEMPTS`` in all.

        Waits between attempts as ``_retry_delay`` says. ``event_id`` adds an
        idempotency key (unless ``body`` has one) so a retry is never
        double-counted; it is off only for the fallback ai.request, which
        relies on the server's request_id de-duplication instead (the original
        pre-flight may have been stored before the client gave up), and that
        covers its retries too. ``attempt`` numbers the first attempt, for an
        event already tried once. Returns the last response, or None.
        """
        event = _with_event_id(body) if event_id and "event_id" not in body else body
        content = self._encode_event(event)
        if content is None:
            return None
        while True:
            resp = self._attempt(content)
            if not _retryable(resp) or attempt >= _MAX_ATTEMPTS:  # delivered, rejected, or out of attempts
                break
            if not self._pause(_retry_delay(resp, attempt, self._random)):
                break  # the client is closing
            attempt += 1
        self._log_undelivered(resp, attempt, _event_ref(event))
        return resp

    def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        content = self._encode_preflight(request_id, params, metadata)
        if isinstance(content, Decision):
            return content
        try:
            resp = self._post(content, self._config.preflight_timeout)
        except httpx.TimeoutException:
            return self._unavailable(
                request_id, "timeout",
                f"SignalVault pre-flight check timed out after {self._config.preflight_timeout}s",
            )
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return self._unavailable(request_id, "network", "SignalVault API is unreachable")
        return self._interpret_preflight(request_id, resp)

    def _fire_response(
        self, request_id: str, model: str, output: str,
        usage: Dict[str, Any], metadata: dict, fallback: Any = None,
    ) -> None:
        """Queues the response event (after the fallback ai.request, if any). Does not block the caller."""
        if fallback is _NO_AUDIT:
            return
        event_id = str(uuid.uuid4())
        self._submit(
            self._send_response_from_parts,
            request_id, model, output, usage, metadata, fallback, event_id,
            refs=([_event_ref(fallback)] if fallback is not None else []) + [f"event_id={event_id}"],
        )

    def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        usage: Dict[str, Any], metadata: dict, fallback: Any = None, event_id: Optional[str] = None,
    ) -> None:
        if fallback is _NO_AUDIT:
            return
        if fallback is not None:
            # Must land first: the server rejects an ai.response whose ai.request it has not stored.
            self._post_background(fallback, event_id=False)
        self._post_background(_event_with_id(
            self._response_body(request_id, model, output, usage, metadata), event_id,
        ))

    def _fire_audit(
        self, request_id: str, model: str, messages: list, output: str,
        usage: Dict[str, Any], metadata: dict,
    ) -> None:
        """Queues the mirror-mode request and response events. Does not block the caller."""
        event_ids = (str(uuid.uuid4()), str(uuid.uuid4()))
        self._submit(
            self._send_audit_from_parts,
            request_id, model, messages, output, usage, metadata, event_ids,
            refs=[f"event_id={i}" for i in event_ids],
        )

    def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        usage: Dict[str, Any], metadata: dict, event_ids: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Sequential on purpose: the server rejects an ai.response whose ai.request it has not stored."""
        request_event_id, response_event_id = event_ids or (None, None)
        self._post_background(_event_with_id(
            self._mirror_request_body(request_id, model, messages, metadata), request_event_id,
        ))
        self._post_background(_event_with_id(
            self._response_body(request_id, model, output, usage, metadata, monitor_mode=True),
            response_event_id,
        ))

    # -- Agent tool-use capture ---------------------------------------------

    def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Blocking POST of an agent.tool_call event. Surfaces 4xx via warnings.

        Used by the manual API (``client.tools.record``) directly. Raises
        ``ValueError`` on an invalid ``tool_name``. If the POST fails in a way
        a retry can fix, the event is retried in the background.
        """
        event = _with_event_id(build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        ))
        content = self._encode_event(event)
        if content is None:
            return
        resp = self._attempt(content)
        if _retryable(resp):
            self._submit(self._deliver_tool_call, event, 2, refs=[_event_ref(event)])
        else:
            self._warn_tool_call_status(resp.status_code)

    def _deliver_tool_call(self, body: dict, attempt: int = 1) -> None:
        resp = self._post_background(body, attempt=attempt)
        if not _retryable(resp):
            self._warn_tool_call_status(resp.status_code)

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Builds the event now and sends it in the background. Non-blocking."""
        body = self._prepare_tool_call(opts)
        if body is not None:
            event = _with_event_id(body)
            self._submit(self._deliver_tool_call, event, refs=[_event_ref(event)])

    @property
    def tools(self) -> "_SyncToolsAPI":
        """Manual tool-call recording API. ``client.tools.record(...)``."""
        return _SyncToolsAPI(self)

    def with_context(self, *, request_id: str) -> ToolContext:
        """Returns a context manager scoping ``request_id`` for tool correlation.

        Usable as both ``with client.with_context(request_id=...)`` and
        ``async with client.with_context(request_id=...)``.
        """
        return ToolContext(request_id)

    def tool(
        self,
        name: str,
        fn: Optional[Callable[..., Any]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Wraps a sync or async callable so each call records an agent.tool_call.

        Two call shapes — both are equivalent::

            wrapped = client.tool("fetch_weather", fetch_weather)
            wrapped = client.tool("fetch_weather")(fetch_weather)  # decorator

        For async functions used inside an async runtime, prefer
        :class:`AsyncSignalVaultClient.tool` — sync clients submit recording to
        a background thread, which is fine but blocks the event loop briefly
        when serializing arguments.
        """

        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(inner):
                # Async fn through a sync client — return an async wrapper.
                @functools.wraps(inner)
                async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                    started_at, start = _start_timing()
                    request_id = get_current_request_id()
                    try:
                        result = await inner(*args, **kwargs)
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, result, None,
                            start, started_at, request_id, metadata,
                        ))
                        return result
                    except Exception as exc:
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, None, exc,
                            start, started_at, request_id, metadata,
                        ))
                        raise

                return async_wrapped

            @functools.wraps(inner)
            def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                started_at, start = _start_timing()
                request_id = get_current_request_id()
                try:
                    result = inner(*args, **kwargs)
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, result, None,
                        start, started_at, request_id, metadata,
                    ))
                    return result
                except Exception as exc:
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, None, exc,
                        start, started_at, request_id, metadata,
                    ))
                    raise

            return sync_wrapped

        if fn is None:
            return decorator
        return decorator(fn)


# Sync clients whose queued audit events get a last chance at interpreter exit.
_SYNC_CLIENTS: "weakref.WeakSet[_BaseSyncClient]" = weakref.WeakSet()


# Async clients, whose unsent events are reported (not sent: their loop is gone) at exit.
_ASYNC_CLIENTS: "weakref.WeakSet[_BaseAsyncClient]" = weakref.WeakSet()


def _flush_at_exit() -> None:
    """Gives each sync client's queued audit events ``exit_flush_timeout`` seconds at interpreter exit.

    All clients flush at once, and retries wait at most a second meanwhile.
    Delivery threads are daemon threads, so this never keeps the process
    alive past the longest timeout; events still unsent then are dropped,
    logged with their ids. An async client's events can't be sent without
    its event loop, so any it still holds are only reported.
    """
    start = time.monotonic()
    clients = list(_SYNC_CLIENTS)
    for client in clients:
        client._start_exit()
    for client in clients:
        client._drain_before_exit(start + client._config.exit_flush_timeout)
    for async_client in list(_ASYNC_CLIENTS):
        async_client._report_unsent_at_exit()


atexit.register(_flush_at_exit)


class _SyncToolsAPI:
    """Manual tool-call API exposed via ``client.tools``."""

    __slots__ = ("_client",)

    def __init__(self, client: "_BaseSyncClient") -> None:
        self._client = client

    def record(
        self,
        *,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        duration_ms: int = 0,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronously POST an agent.tool_call event. Surfaces errors.

        ``tool_name`` is required and validated (raises ``ValueError`` if
        empty or non-string). ``request_id`` falls back to the surrounding
        ``with_context`` block when omitted.
        """
        opts = ToolRecordOptions(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
            started_at=started_at,
            request_id=request_id if request_id is not None else get_current_request_id(),
            metadata=metadata,
        )
        self._client._send_tool_call_event(opts)


# ---------------------------------------------------------------------------
# Helpers shared by sync and async tool wrappers
# ---------------------------------------------------------------------------

def _start_timing() -> tuple[str, float]:
    """Returns (iso8601_started_at, monotonic_start_seconds)."""
    return datetime.now(tz=timezone.utc).isoformat(), time.monotonic()


def _build_opts(
    name: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    exc: Optional[BaseException],
    start: float,
    started_at: str,
    request_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> ToolRecordOptions:
    duration_ms = int((time.monotonic() - start) * 1000)
    return ToolRecordOptions(
        tool_name=name,
        tool_input=_serialize_args(args, kwargs),
        tool_output=None if exc is not None else result,
        duration_ms=duration_ms,
        error=None if exc is None else (str(exc) if not isinstance(exc, str) else exc),
        started_at=started_at,
        request_id=request_id,
        metadata=metadata,
    )


def _serialize_args(args: tuple, kwargs: dict) -> Any:
    """Mirrors Node's serializeArgs.

    - No args → None
    - Single positional arg, no kwargs → that value directly (avoids ``[obj]``)
    - kwargs but no positional → the kwargs dict
    - Otherwise → ``{"args": [...], "kwargs": {...}}``
    """
    if not args and not kwargs:
        return None
    if len(args) == 1 and not kwargs:
        return args[0]
    if not args and kwargs:
        return dict(kwargs)
    return {"args": list(args), "kwargs": dict(kwargs)}


# ---------------------------------------------------------------------------
# Base async client — shared HTTP logic for OpenAI and Anthropic async clients
# ---------------------------------------------------------------------------

class _BaseAsyncClient(_ClientCommon):
    """Shared HTTP, config, and audit logic for async SignalVault clients."""

    def __init__(self, config: SignalVaultConfig) -> None:
        self._init_common(config)
        # Never follow redirects: the API key must not be re-sent to another URL.
        self._http = httpx.AsyncClient(follow_redirects=False)
        # Audit tasks, oldest first, with the events each sends.
        # Strong references: the event loop keeps only weak references to
        # tasks, so an unreferenced audit task can be garbage-collected mid-send.
        self._tasks: Dict["asyncio.Task[Any]", List[str]] = {}
        self._queued_events = 0
        self._closed = False
        # flush() or aclose() in progress: retries wait less.
        self._hurrying = 0
        self._retry_waiters: set = set()
        # Injectable for tests: waits before a retry.
        self._sleep: Callable[[float], Awaitable[Any]] = self._wait_before_retry
        _ASYNC_CLIENTS.add(self)

    # -- Resource management -------------------------------------------------

    async def flush(self, timeout: float = 5.0) -> None:
        """Waits up to ``timeout`` seconds for queued audit events to be sent.

        Meanwhile retries wait at most a second. Events still unsent stay queued.
        """
        pending = await self._wait_tasks(timeout)
        if pending:
            left = sum(len(self._tasks.get(t, ())) for t in pending)
            self._warn("flush-timeout", f"[SignalVault] flush() timed out with {left} event(s) still sending.")

    async def aclose(self, timeout: float = 10.0) -> None:
        """Sends queued audit events (up to ``timeout`` seconds), then closes the HTTP pool.

        Nothing is dropped before the deadline. Events still unsent then can't
        be sent once the pool is closed: they are dropped, logged with their
        ids and counted in ``dropped_events``.
        """
        pending = await self._wait_tasks(timeout)
        self._closed = True
        dropped: List[str] = []
        for task in pending:
            dropped += self._cancel(task)
        self._drop("shutdown", dropped, "the client was closed before they were sent")
        await self._http.aclose()

    async def __aenter__(self) -> "_BaseAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def _wait_tasks(self, timeout: float) -> set:
        """Waits up to ``timeout`` for the audit tasks, with short retry waits. Returns those left."""
        tasks = [t for t in self._tasks if not t.done()]
        if not tasks:
            return set()
        self._hurrying += 1
        self._wake_retry_waiters()
        try:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
        finally:
            self._hurrying -= 1
        return pending

    def _spawn(self, coro: Awaitable[Any], refs: Optional[List[str]] = None) -> "Optional[asyncio.Task[Any]]":
        """Runs ``coro``, which sends the events ``refs`` names, in the background.

        When more than ``_MAX_QUEUED_EVENTS`` are unsent, the oldest tasks are
        cancelled and their events dropped.
        """
        refs = refs if refs is not None else [f"{getattr(coro, '__name__', 'send')} (no event_id)"]
        if self._closed:
            coro.close()  # type: ignore[attr-defined]
            self._drop("shutdown", refs, "the client is closed")
            return None
        try:
            task = asyncio.get_running_loop().create_task(coro)  # type: ignore[arg-type]
        except RuntimeError:
            coro.close()  # type: ignore[attr-defined]
            self._warn(
                "no-loop",
                "[SignalVault] No running event loop: audit event dropped. "
                "Use the async client from async code.",
            )
            return None
        self._tasks[task] = refs
        self._queued_events += len(refs)
        with self._refs_lock:
            self._unsent.update(dict.fromkeys(refs))
        task.add_done_callback(self._task_done)
        dropped: List[str] = []
        while self._queued_events > _MAX_QUEUED_EVENTS and len(self._tasks) > 1:
            oldest = next(iter(self._tasks))
            if oldest.done():  # finished; its done callback just hasn't run yet
                self._task_done(oldest)
            else:
                dropped += self._cancel(oldest)
        if dropped:
            self._drop(
                "queue_full", dropped,
                f"{_MAX_QUEUED_EVENTS:,} events waiting; SignalVault is unreachable or rate limiting this app",
            )
        return task

    def _cancel(self, task: "asyncio.Task[Any]") -> List[str]:
        """Cancels an audit task on purpose. Returns its unsent events, for the caller to report."""
        refs = self._tasks.pop(task, [])
        self._queued_events -= len(refs)
        task.cancel()
        return self._claim_unsent(refs, sender_may_continue=False)

    def _task_done(self, task: "asyncio.Task[Any]") -> None:
        refs = self._tasks.pop(task, None)
        if refs is None:  # cancelled by _cancel, which reported it
            return
        self._queued_events -= len(refs)
        left = self._claim_unsent(refs, sender_may_continue=False)
        if task.cancelled():
            self._drop(
                "shutdown", left,
                "the task sending them was cancelled, usually because the event loop stopped; "
                "await client.aclose() before it ends",
            )
            return
        if task.exception() is not None and self._config.debug:
            exc = task.exception()
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        if left:  # normally each event was settled as it finished
            logger.warning(
                "SignalVault dropped %d audit event(s), an internal error stopped their delivery: %s",
                len(left), ", ".join(left),
                extra={"signalvault_drop_cause": "error", "signalvault_dropped": left},
            )

    def _report_unsent_at_exit(self) -> None:
        if not self._closed:
            self._drop(
                "shutdown", self._claim_unsent(list(self._unsent), sender_may_continue=False),
                "the process exited while its event loop still held them; "
                "await client.aclose() before the loop ends",
            )

    async def _wait_before_retry(self, delay: float) -> None:
        """The default ``_sleep``: ``delay`` seconds, or until flush() or aclose() starts waiting."""
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        handle = loop.call_later(delay, _resolve, waiter)
        self._retry_waiters.add(waiter)
        try:
            await waiter
        finally:
            handle.cancel()
            self._retry_waiters.discard(waiter)

    def _wake_retry_waiters(self) -> None:
        for waiter in list(self._retry_waiters):
            try:
                _resolve(waiter)
            except RuntimeError:  # its event loop is closed
                pass

    async def _pause(self, delay: float) -> None:
        if self._hurrying:
            delay = min(delay, _FLUSH_RETRY_WAIT_SECONDS)
        await self._sleep(delay)

    # -- Internal HTTP -------------------------------------------------------

    async def _post(self, content: bytes, timeout: float) -> httpx.Response:
        return await self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            timeout=timeout,
            content=content,
        )

    async def _attempt(self, content: bytes) -> Optional[httpx.Response]:
        """One POST of a background event; None on a connection error or timeout."""
        try:
            return await self._post(content, self._config.timeout)
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return None

    async def _post_background(
        self, body: dict, *, event_id: bool = True, attempt: int = 1,
    ) -> Optional[httpx.Response]:
        """POST, retrying connection errors, timeouts, 429 and 5xx, up to ``_MAX_ATTEMPTS`` in all.

        As in the sync client: see ``_BaseSyncClient._post_background``.
        """
        event = _with_event_id(body) if event_id and "event_id" not in body else body
        content = self._encode_event(event)
        if content is None:
            return None
        while True:
            resp = await self._attempt(content)
            if not _retryable(resp) or attempt >= _MAX_ATTEMPTS:  # delivered, rejected, or out of attempts
                break
            await self._pause(_retry_delay(resp, attempt, self._random))
            attempt += 1
        self._log_undelivered(resp, attempt, _event_ref(event))
        return resp

    async def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        content = self._encode_preflight(request_id, params, metadata)
        if isinstance(content, Decision):
            return content
        try:
            resp = await self._post(content, self._config.preflight_timeout)
        except httpx.TimeoutException:
            return self._unavailable(
                request_id, "timeout",
                f"SignalVault pre-flight check timed out after {self._config.preflight_timeout}s",
            )
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return self._unavailable(request_id, "network", "SignalVault API is unreachable")
        return self._interpret_preflight(request_id, resp)

    def _fire_response(
        self, request_id: str, model: str, output: str,
        usage: Dict[str, Any], metadata: dict, fallback: Any = None,
    ) -> "Optional[asyncio.Task[Any]]":
        """Sends the response event (after the fallback ai.request, if any) in the background."""
        if fallback is _NO_AUDIT:
            return None
        event_id = str(uuid.uuid4())
        return self._spawn(
            self._send_response_from_parts(
                request_id, model, output, usage, metadata, fallback=fallback, event_id=event_id,
            ),
            refs=([_event_ref(fallback)] if fallback is not None else []) + [f"event_id={event_id}"],
        )

    async def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        usage: Dict[str, Any], metadata: dict, fallback: Any = None, event_id: Optional[str] = None,
    ) -> None:
        if fallback is _NO_AUDIT:
            return
        if fallback is not None:
            # Must land first: the server rejects an ai.response whose ai.request it has not stored.
            await self._post_background(fallback, event_id=False)
        await self._post_background(_event_with_id(
            self._response_body(request_id, model, output, usage, metadata), event_id,
        ))

    def _fire_audit(
        self, request_id: str, model: str, messages: list, output: str,
        usage: Dict[str, Any], metadata: dict,
    ) -> "Optional[asyncio.Task[Any]]":
        """Sends the mirror-mode request and response events in the background."""
        event_ids = (str(uuid.uuid4()), str(uuid.uuid4()))
        return self._spawn(
            self._send_audit_from_parts(request_id, model, messages, output, usage, metadata, event_ids),
            refs=[f"event_id={i}" for i in event_ids],
        )

    async def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        usage: Dict[str, Any], metadata: dict, event_ids: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Sequential on purpose: the server rejects an ai.response whose ai.request it has not stored."""
        request_event_id, response_event_id = event_ids or (None, None)
        await self._post_background(_event_with_id(
            self._mirror_request_body(request_id, model, messages, metadata), request_event_id,
        ))
        await self._post_background(_event_with_id(
            self._response_body(request_id, model, output, usage, metadata, monitor_mode=True),
            response_event_id,
        ))

    # -- Agent tool-use capture ---------------------------------------------

    async def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Awaitable POST of an agent.tool_call event. Surfaces 4xx via warnings.

        If the POST fails in a way a retry can fix, the event is retried in the background.
        """
        event = _with_event_id(build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        ))
        content = self._encode_event(event)
        if content is None:
            return
        resp = await self._attempt(content)
        if _retryable(resp):
            self._spawn(self._deliver_tool_call(event, 2), refs=[_event_ref(event)])
        else:
            self._warn_tool_call_status(resp.status_code)

    async def _deliver_tool_call(self, body: dict, attempt: int = 1) -> None:
        resp = await self._post_background(body, attempt=attempt)
        if not _retryable(resp):
            self._warn_tool_call_status(resp.status_code)

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Builds the event now and schedules the send on the running loop. Non-blocking."""
        body = self._prepare_tool_call(opts)
        if body is not None:
            event = _with_event_id(body)
            self._spawn(self._deliver_tool_call(event), refs=[_event_ref(event)])

    @property
    def tools(self) -> "_AsyncToolsAPI":
        """Manual async tool-call recording API. ``await client.tools.record(...)``."""
        return _AsyncToolsAPI(self)

    def with_context(self, *, request_id: str) -> ToolContext:
        """Returns a context manager scoping ``request_id`` for tool correlation.

        Use as ``async with client.with_context(request_id=...):`` from async
        code.
        """
        return ToolContext(request_id)

    def tool(
        self,
        name: str,
        fn: Optional[Callable[..., Awaitable[Any]]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Wraps an async (or sync) callable so each call records an agent.tool_call.

        The audit POST is fire-and-forget on the running event loop so the
        wrapped function's latency isn't affected by SignalVault's network call.

        If ``inner`` is a sync function, it is invoked inline inside the async
        wrapper and will briefly block the event loop. For CPU-bound or
        long-running sync work prefer ``run_in_executor`` or use the sync
        :class:`SignalVaultClient`.
        """

        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(inner):
                @functools.wraps(inner)
                async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                    started_at, start = _start_timing()
                    request_id = get_current_request_id()
                    try:
                        result = await inner(*args, **kwargs)
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, result, None,
                            start, started_at, request_id, metadata,
                        ))
                        return result
                    except Exception as exc:
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, None, exc,
                            start, started_at, request_id, metadata,
                        ))
                        raise

                return async_wrapped

            # Sync fn through async client — wrap it as async so the wrapper
            # signature is uniform for callers of an AsyncSignalVaultClient.
            @functools.wraps(inner)
            async def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                started_at, start = _start_timing()
                request_id = get_current_request_id()
                try:
                    result = inner(*args, **kwargs)
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, result, None,
                        start, started_at, request_id, metadata,
                    ))
                    return result
                except Exception as exc:
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, None, exc,
                        start, started_at, request_id, metadata,
                    ))
                    raise

            return sync_wrapped

        if fn is None:
            return decorator
        return decorator(fn)


class _AsyncToolsAPI:
    """Async manual tool-call API exposed via ``client.tools``."""

    __slots__ = ("_client",)

    def __init__(self, client: "_BaseAsyncClient") -> None:
        self._client = client

    async def record(
        self,
        *,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        duration_ms: int = 0,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Awaitable manual recording. Surfaces errors back to the caller."""
        opts = ToolRecordOptions(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
            started_at=started_at,
            request_id=request_id if request_id is not None else get_current_request_id(),
            metadata=metadata,
        )
        await self._client._send_tool_call_event(opts)


# ---------------------------------------------------------------------------
# SignalVaultClient (sync, OpenAI)
# ---------------------------------------------------------------------------

def _with_stream_usage(kwargs: dict, stream: bool) -> bool:
    """Asks OpenAI to report token usage on a stream, unless the caller already decided.

    OpenAI then sends one extra final chunk with an empty ``choices`` list.
    Returns True when ``include_usage`` was added here, so the stream wrapper
    consumes that chunk instead of passing it to a caller who never asked
    for it. Other stream options the caller set are kept. ``stream_options=None``
    sends no stream options at all, for OpenAI-compatible servers that reject
    them. Matches ``withStreamUsage`` in the Node SDK.
    """
    if not stream:
        return False
    options = kwargs.get("stream_options", {})
    if options is None:
        del kwargs["stream_options"]
        return False
    if not isinstance(options, dict) or options.get("include_usage") is not None:
        return False
    kwargs["stream_options"] = {**options, "include_usage": True}
    return True


class _StreamCollector:
    """Collects the text and token usage of a provider stream."""

    def __init__(self) -> None:
        self.parts: List[str] = []
        self.usage: Dict[str, Any] = {}

    def add(self, item: Any) -> bool:
        """Records an item. Returns False to keep it from the caller."""
        raise NotImplementedError

    def result(self) -> Tuple[str, Dict[str, Any]]:
        return "".join(self.parts), self.usage


class _ChunkCollector(_StreamCollector):
    """OpenAI chat completion chunks."""

    def __init__(self, hide_usage_chunk: bool):
        super().__init__()
        self._hide_usage_chunk = hide_usage_chunk
        self.usage = _openai_usage(None)

    def add(self, chunk: Any) -> bool:
        """Records a chunk. Returns False for the usage-only chunk the caller didn't ask for."""
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            self.parts.append(delta.content)
        if getattr(chunk, "usage", None):
            self.usage = {
                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                "completion_tokens": chunk.usage.completion_tokens or 0,
            }
            if self._hide_usage_chunk and not chunk.choices:
                return False
        return True


class _AnthropicEventCollector(_StreamCollector):
    """Anthropic message stream events. Every event reaches the caller."""

    def __init__(self) -> None:
        super().__init__()
        self.usage = _anthropic_usage()

    def add(self, event: Any) -> bool:
        kind = getattr(event, "type", None)
        if kind == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) == "text_delta":
                self.parts.append(getattr(delta, "text", "") or "")
        # message_delta may repeat the input and cache counts; its values win.
        elif kind == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            if usage:
                self.usage = _anthropic_usage(self.usage, usage)
        elif kind == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                self.usage = _anthropic_usage(self.usage, usage)
        return True


class SignalVaultStream:
    """Returned by ``create(..., stream=True)`` on the sync OpenAI and Anthropic clients.

    Iterates the provider's stream and records the response once: when the
    stream ends, raises, or is closed. A ``for`` loop doesn't close its
    iterator on ``break``, so use ``with`` (or call ``close()``) when you may
    stop early. Otherwise the response is recorded when the stream is
    garbage-collected.
    """

    def __init__(self, upstream: Any, collector: _StreamCollector, record: Callable[[str, Dict[str, Any]], None]):
        self._upstream = upstream
        self._iterator = iter(upstream)
        self._collector = collector
        self._record = record
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def response(self) -> Any:
        """The provider's HTTP response (``Stream.response`` in the provider SDK)."""
        return self._upstream.response

    def __iter__(self) -> "SignalVaultStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            while True:
                chunk = next(self._iterator)
                if self._collector.add(chunk):
                    return chunk
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # keep the original exception, or the normal end of the stream
            raise

    def close(self) -> None:
        """Closes the provider stream and records the response. Safe to call twice, from any thread."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            close = getattr(self._upstream, "close", None)
            if callable(close):
                close()
        finally:
            self._record(*self._collector.result())

    def __enter__(self) -> "SignalVaultStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            pass


class _AsyncStreamFinisher:
    """Closes the provider stream and records the response, once.

    Holds no reference to the AsyncSignalVaultStream, so the stream can be
    collected while its event generator is still being finalized.
    """

    def __init__(
        self, upstream: Any, collector: _StreamCollector,
        record: Callable[[str, Dict[str, Any]], "Optional[asyncio.Task[Any]]"],
    ):
        self._upstream = upstream
        self._collector = collector
        self._record = record
        self._done = False
        # AsyncSignalVaultStream calls in progress. Zero when Python finalizes
        # the stream: on garbage collection or when asyncio.run shuts down.
        self.watched = 0

    async def __call__(self) -> None:
        # No await between the check and the set, so two tasks can't both pass it.
        if self._done:
            return
        self._done = True
        try:
            close = getattr(self._upstream, "close", None) or getattr(self._upstream, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except Exception:
            pass  # a failed close must not hide how the stream itself ended
        finally:
            sent = self._record(*self._collector.result())
            if sent is not None and not self.watched:
                # asyncio.run closes the loop without waiting for background
                # tasks, so wait here, as flush() would.
                await asyncio.wait([sent], timeout=5.0)


async def _async_events(upstream: Any, collector: _StreamCollector, finish: _AsyncStreamFinisher) -> Any:
    """Drives an async provider stream for AsyncSignalVaultStream.

    Python finalizes an async generator that has started and is then dropped,
    on garbage collection or when ``asyncio.run`` shuts down its loop, so the
    ``finally`` closes the provider stream and records the response even if
    the caller stops reading without closing. The request has already been
    sent, so ``AsyncSignalVaultStream._start()`` starts the generator before
    the caller gets the stream: a stream that is never read is recorded too.
    """
    try:
        yield None  # consumed by AsyncSignalVaultStream._start()
        async for item in upstream:
            if collector.add(item):
                yield item
    finally:
        await finish()


class AsyncSignalVaultStream:
    """Returned by ``create(..., stream=True)`` on the async OpenAI and Anthropic clients.

    The async counterpart of :class:`SignalVaultStream`. Use ``async with``
    (or ``await stream.close()``) when you may stop early.
    """

    def __init__(
        self, upstream: Any, collector: _StreamCollector,
        record: Callable[[str, Dict[str, Any]], "Optional[asyncio.Task[Any]]"],
    ):
        self._upstream = upstream
        self._finish = _AsyncStreamFinisher(upstream, collector, record)
        self._events = _async_events(upstream, collector, self._finish)
        self._started = False
        self._reading = 0
        self._closing = False

    @property
    def response(self) -> Any:
        """The provider's HTTP response (``AsyncStream.response`` in the provider SDK)."""
        return self._upstream.response

    def __aiter__(self) -> "AsyncSignalVaultStream":
        return self

    async def __anext__(self) -> Any:
        if not self._started:
            await self._start()
        self._reading += 1
        self._finish.watched += 1
        try:
            return await self._events.__anext__()
        finally:
            self._reading -= 1
            self._finish.watched -= 1

    async def _start(self) -> "AsyncSignalVaultStream":
        if not self._started:
            self._started = True
            await self._events.__anext__()  # the generator's first, empty item
        return self

    async def close(self) -> None:
        """Closes the provider stream and records the response.

        Safe to call twice, also at the same time, and while another task
        reads the stream: that read then raises the provider SDK's connection
        error.
        """
        self._finish.watched += 1
        try:
            if not self._reading and not self._closing:
                self._closing = True
                await self._events.aclose()  # runs the generator's finally
            # The generator can't be closed while another task is reading or
            # closing it, so close the provider stream directly.
            await self._finish()
        finally:
            self._finish.watched -= 1

    async def aclose(self) -> None:
        """Same as :meth:`close`, as on the provider SDK's ``AsyncStream`` and on async generators."""
        await self.close()

    async def __aenter__(self) -> "AsyncSignalVaultStream":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()


class _ChatCompletions:
    """Proxies `client.chat.completions.create(...)` with SignalVault guardrails."""

    def __init__(self, sv: "SignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        hide_usage_chunk = _with_stream_usage(kwargs, stream)

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(
                request_id, kwargs, response, metadata, mirror=False, fallback=fallback,
                hide_usage_chunk=hide_usage_chunk,
            )

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            (response.choices[0].message.content or "") if response.choices else "",
            _openai_usage(response.usage),
            metadata, fallback=fallback,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        hide_usage_chunk = _with_stream_usage(kwargs, stream)

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(
                request_id, kwargs, response, metadata, mirror=True, hide_usage_chunk=hide_usage_chunk,
            )

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            (response.choices[0].message.content or "") if response.choices else "",
            _openai_usage(response.usage),
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None, hide_usage_chunk: bool = False,
    ) -> SignalVaultStream:
        model = kwargs.get("model", "")

        def record(output: str, usage: Dict[str, Any]) -> None:
            if mirror:
                self._sv._fire_audit(
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, usage, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, usage, metadata, fallback=fallback,
                )

        return SignalVaultStream(stream, _ChunkCollector(hide_usage_chunk), record)


class _Chat:
    def __init__(self, sv: "SignalVaultClient"):
        self.completions = _ChatCompletions(sv)


class SignalVaultClient(_BaseSyncClient):
    """
    Sync OpenAI wrapper with SignalVault guardrails.

    Usage::

        from signalvault import SignalVaultClient

        client = SignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
            metadata={"user_id": "u_123"},
        )

        # Non-streaming
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            sv_metadata={"tool": "clip_detect", "job_id": "abc-123"},
        )

        # Streaming
        with client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            sv_metadata={"tool": "stream_test"},
        ) as stream:
            for chunk in stream:
                if chunk.choices:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)

        # Context manager (recommended for long-lived use)
        with SignalVaultClient(...) as client:
            response = client.chat.completions.create(...)
    """

    _provider = "openai"

    def __init__(
        self,
        api_key: str,
        openai_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
        exit_flush_timeout: float = _EXIT_FLUSH_SECONDS,
    ):
        from openai import OpenAI
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata, exit_flush_timeout,
        ))
        self._openai = OpenAI(api_key=openai_api_key)
        self.chat = _Chat(self)


# ---------------------------------------------------------------------------
# AsyncSignalVaultClient (async, OpenAI)
# ---------------------------------------------------------------------------

class _AsyncChatCompletions:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        hide_usage_chunk = _with_stream_usage(kwargs, stream)

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return await self._wrap_stream(
                request_id, kwargs, response, metadata, mirror=False, fallback=fallback,
                hide_usage_chunk=hide_usage_chunk,
            )

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            (response.choices[0].message.content or "") if response.choices else "",
            _openai_usage(response.usage),
            metadata, fallback=fallback,
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        hide_usage_chunk = _with_stream_usage(kwargs, stream)

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return await self._wrap_stream(
                request_id, kwargs, response, metadata, mirror=True, hide_usage_chunk=hide_usage_chunk,
            )

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            (response.choices[0].message.content or "") if response.choices else "",
            _openai_usage(response.usage),
            metadata,
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None, hide_usage_chunk: bool = False,
    ) -> AsyncSignalVaultStream:
        model = kwargs.get("model", "")

        def record(output: str, usage: Dict[str, Any]) -> "Optional[asyncio.Task[Any]]":
            if mirror:
                return self._sv._fire_audit(
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, usage, metadata,
                )
            return self._sv._fire_response(
                request_id, model, output, usage, metadata, fallback=fallback,
            )

        return await AsyncSignalVaultStream(stream, _ChunkCollector(hide_usage_chunk), record)._start()


class _AsyncChat:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self.completions = _AsyncChatCompletions(sv)


class AsyncSignalVaultClient(_BaseAsyncClient):
    """
    Async OpenAI wrapper with SignalVault guardrails. Use in FastAPI, async Django, etc.

    Usage::

        from signalvault import AsyncSignalVaultClient

        client = AsyncSignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        async with await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        ) as stream:
            async for chunk in stream:
                if chunk.choices:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)

        # Context manager (recommended)
        async with AsyncSignalVaultClient(...) as client:
            response = await client.chat.completions.create(...)
    """

    _provider = "openai"

    def __init__(
        self,
        api_key: str,
        openai_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import AsyncOpenAI
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
        ))
        self._openai = AsyncOpenAI(api_key=openai_api_key)
        self.chat = _AsyncChat(self)


# ---------------------------------------------------------------------------
# AnthropicSignalVaultClient (sync)
# ---------------------------------------------------------------------------

class _AnthropicMessages:
    def __init__(self, sv: "AnthropicSignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        response = self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False, fallback=fallback)

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            _anthropic_output_text(response),
            _anthropic_usage(response.usage),
            metadata, fallback=fallback,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        response = self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            _anthropic_output_text(response),
            _anthropic_usage(response.usage),
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
    ) -> SignalVaultStream:
        model = kwargs.get("model", "")

        def record(output: str, usage: Dict[str, Any]) -> None:
            if mirror:
                self._sv._fire_audit(
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, usage, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, usage, metadata, fallback=fallback,
                )

        return SignalVaultStream(stream, _AnthropicEventCollector(), record)


class AnthropicSignalVaultClient(_BaseSyncClient):
    """
    Sync Anthropic/Claude wrapper with SignalVault guardrails.

    Install: pip install signalvault[anthropic]

    Usage::

        from signalvault import AnthropicSignalVaultClient

        client = AnthropicSignalVaultClient(
            api_key="sk_live_...",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        with client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=1024,
            stream=True,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

        # Context manager (recommended)
        with AnthropicSignalVaultClient(...) as client:
            response = client.messages.create(...)
    """

    _provider = "anthropic"

    def _audit_messages(self, params: dict) -> list:
        return _anthropic_messages_for_audit(params)

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
        exit_flush_timeout: float = _EXIT_FLUSH_SECONDS,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install signalvault[anthropic]"
            )
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata, exit_flush_timeout,
        ))
        self._anthropic = _anthropic.Anthropic(api_key=anthropic_api_key)
        self.messages = _AnthropicMessages(self)


# ---------------------------------------------------------------------------
# AsyncAnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class _AsyncAnthropicMessages:
    def __init__(self, sv: "AsyncAnthropicSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        response = await self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return await self._wrap_stream(request_id, kwargs, response, metadata, mirror=False, fallback=fallback)

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            _anthropic_output_text(response),
            _anthropic_usage(response.usage),
            metadata, fallback=fallback,
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        response = await self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return await self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            _anthropic_output_text(response),
            _anthropic_usage(response.usage),
            metadata,
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
    ) -> AsyncSignalVaultStream:
        model = kwargs.get("model", "")

        def record(output: str, usage: Dict[str, Any]) -> "Optional[asyncio.Task[Any]]":
            if mirror:
                return self._sv._fire_audit(
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, usage, metadata,
                )
            return self._sv._fire_response(
                request_id, model, output, usage, metadata, fallback=fallback,
            )

        return await AsyncSignalVaultStream(stream, _AnthropicEventCollector(), record)._start()


class AsyncAnthropicSignalVaultClient(_BaseAsyncClient):
    """
    Async Anthropic/Claude wrapper with SignalVault guardrails.

    Install: pip install signalvault[anthropic]

    Usage::

        from signalvault import AsyncAnthropicSignalVaultClient

        client = AsyncAnthropicSignalVaultClient(
            api_key="sk_live_...",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        stream = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=1024,
            stream=True,
        )
        async with stream:
            async for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

        # Context manager (recommended)
        async with AsyncAnthropicSignalVaultClient(...) as client:
            response = await client.messages.create(...)
    """

    _provider = "anthropic"

    def _audit_messages(self, params: dict) -> list:
        return _anthropic_messages_for_audit(params)

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install signalvault[anthropic]"
            )
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
        ))
        self._anthropic = _anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.messages = _AsyncAnthropicMessages(self)
