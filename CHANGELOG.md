# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- OpenAI `create(..., stream=True)` returns a `SignalVaultStream` / `AsyncSignalVaultStream` instead of a generator / async generator (see Fixes). Iterating it, `close()` and `aclose()` work as before. Generator-only APIs (`send()`, `throw()`, `asend()`, `athrow()`) and `inspect.isgenerator()` / `inspect.isasyncgen()` checks no longer apply. (#15)
- Anthropic `messages.create(..., stream=True)` now calls the Anthropic SDK's `messages.create(..., stream=True)` instead of `messages.stream()`, as the Node SDK does. It yields the raw stream events, exactly as the Anthropic SDK does. **Note:** the extra events that `messages.stream()` added in recent Anthropic SDK versions (`text`, `input_json`, `citation`, ...) are no longer yielded; read the text from `content_block_delta` events instead. The request is now sent when `create()` is called, so provider errors (for example an invalid Anthropic key) are raised there rather than on the first iteration, and nothing more is recorded for them, as for non-streaming calls. Before, an empty response was recorded, plus the fallback request when the pre-flight check was unavailable. It returns a `SignalVaultStream` / `AsyncSignalVaultStream` like the OpenAI clients, so the note above on generator-only APIs applies here too. (#29)

### Fixes

- OpenAI streams (`chat.completions.create(..., stream=True)`) now return a `SignalVaultStream` (`AsyncSignalVaultStream` on the async client) instead of a plain generator, so they work like the OpenAI SDK's stream:
  - `with` / `async with` is supported; it previously raised `TypeError`.
  - `close()` / `await close()` (and `await aclose()`) and `.response` are available.
  - Closing the stream, or leaving the `with` block, closes the connection to OpenAI and records the response right away. Previously the connection stayed open until garbage collection, and a stream left after `break` was only recorded then.
  - Each response is still recorded exactly once: when the stream ends, raises, or is closed.
  - An async stream that is dropped before it is read is now closed and recorded too, when it is garbage-collected or at the latest when `asyncio.run` finishes. Previously its response was never recorded and the connection was not closed.
  - When an async stream that is left open and not being read is recorded as `asyncio.run` shuts down, the SDK waits up to 5 seconds for the event to be sent. Previously the loop could close first and the event was lost. To be sure all events are sent, close streams and call `await client.aclose()` before the loop ends.
  - `close()` can be called from another task while one is reading the stream, for example from a timeout. It closes the connection, and that read raises a connection error (`httpx.ReadError`, or `httpx2.ReadError` from recent `anthropic` versions), so catch it in the reading task.

  Both classes are exported for type hints. (#15)
- Anthropic streams get the same: `with` / `async with`, `close()`, `.response`, an immediate close and record on `close()` or at the end of the `with` block, and exactly one recorded response. (#29)

---

## [0.5.2] — 2026-10-14

Security fix for API keys with surrounding whitespace, provider SDK objects in `messages`, and a higher `openai` minimum.

### Security

- An `api_key` with surrounding whitespace, such as the trailing newline a key read from a file or a Kubernetes secret usually has, made every request to SignalVault fail before it was sent. With the default `fail_mode="open"`, every call then went to the provider without a guardrail check or an audit event, and the warning said the API was unreachable. With `debug=True` the traceback also showed the key. Surrounding whitespace is now stripped, and a key that is empty or contains other whitespace, control or non-ASCII characters raises `ValueError` when the client is created (`TypeError` if it isn't a string). The error never includes the key. (GHSA-m2xh-jwwm-2g56)

### Changed

- The minimum `openai` version is now 1.26. Streaming sends `stream_options`, which older versions reject with `TypeError: ... unexpected keyword argument 'stream_options'`, so `stream=True` failed on openai 1.0–1.25. If you pin `openai` below 1.26, upgrade it together with this release. (#23)

### Fixes

- Provider SDK objects in `messages` no longer raise `TypeError`. The usual tool-calling loop appends `response.choices[0].message` (or Anthropic content blocks) to the conversation; these are now sent with the fields the provider returned, as the Node SDK does, under pydantic v1 and v2. Previously the pre-flight check raised even with `fail_mode="open"`, and in mirror mode the audit event was dropped silently. (#22)
- Other values in `messages` and `sv_metadata` that JSON can't represent are converted instead of failing: dates and times as ISO 8601, `UUID` and `Decimal` as strings, enums as their value, sets as lists, and anything else as its `repr`. (#22)
- A request that still can't be serialized (a circular reference or NaN in `sv_metadata`) follows `fail_mode`: with `"open"` it emits a `SignalVaultWarning` and the request goes ahead, with `"closed"` it raises `SignalVaultUnavailableError`. A background event that can't be serialized is dropped with a `SignalVaultWarning` (at most once a minute per event type), not silently. (#22)

---

## [0.5.1] — 2026-10-14

Fixes for warning categories and OpenAI streaming.

### Added

- `SignalVaultViolationWarning`, a subclass of `SignalVaultWarning`, is used for a `warn` guardrail decision (with `debug=True`). Filter on it to handle policy warnings separately from delivery problems; existing `SignalVaultWarning` filters still match.

### Fixes

- A `warn` guardrail decision (with `debug=True`), a rejected `tool_call` (4xx) and a failed `tool_call` (with `debug=True`) now emit `SignalVaultWarning` instead of a plain `UserWarning`, so `warnings.simplefilter("error", SignalVaultWarning)` catches them. They are reported at the caller's line like the SDK's other warnings, and the rejected `tool_call` warning repeats at most once a minute per status code.
- OpenAI streams no longer end with a chunk whose `choices` list is empty, which made the README's `chunk.choices[0]` loop raise `IndexError`. The SDK adds `include_usage` to record token counts and now consumes that usage-only chunk itself, as the Node SDK does. If you set `include_usage` yourself, your setting is kept and, with `True`, the chunk is passed through.
- Streamed OpenAI responses are logged with token counts when you pass other `stream_options` (for example `{"include_obfuscation": False}`); `include_usage` is added alongside them. Previously any `stream_options` meant the response was logged with 0 tokens.
- `stream_options=None` now sends no stream options, for OpenAI-compatible servers that reject them. Previously `null` was sent.

---

## [0.5.0] — 2026-09-28

Security and reliability release.

### Breaking changes

- **Default `base_url` is now `https://api.signalvault.io`** (was `http://localhost:4000`). Pass `base_url` explicitly for local development.
- **Plain `http://` base URLs are refused** except for `localhost`, `127.0.0.1` and `::1`, because the API key and prompts would travel unencrypted. A `base_url` with a query string or fragment is refused too.
- **`metadata=` is no longer taken over by SignalVault.** It is passed through to the provider (OpenAI's `metadata`, Anthropic's `metadata={"user_id": ...}`). Use `sv_metadata=` for SignalVault metadata; the `metadata` fallback was deprecated in 0.3.0.
- **Blocked requests raise `SignalVaultBlockedError`** (a `RuntimeError` subclass) with `violations`, `request_id` and `dashboard_url`. The message lists violation types instead of the full violation dicts.
- **`Decision.redactions`** is a list of `{"type", "count"}` dicts, matching the API.

### Security

- A failed guardrail check is no longer silent. An invalid key (401), inactive account (402), denied access (403), rate or trial limit (429), server error, timeout or invalid response now emits a `SignalVaultWarning` (at most once a minute per cause) even with `debug=False`.
- New `fail_mode="open" | "closed"`. With `"closed"`, the SDK raises `SignalVaultUnavailableError` instead of calling the provider unchecked. Default stays `"open"`.
- The Anthropic `system` prompt is now included in the pre-flight scan and the audit record.
- HTTP clients never follow redirects, so the API key is never re-sent to another URL.
- When the pre-flight check could not be completed (unreachable, timeout, 5xx, invalid response) and the request went ahead unchecked, the request is now recorded in the background, marked `preflight_unavailable`. Previously only the response was sent, which the API rejects, so these calls were missing from the audit log. After a refused check (401/402/403/429) no audit events are sent.
- `SignalVaultWarning` is reported at the caller's line and repeats after the one-minute window. Previously Python's default warning filter showed each message only once per process, so a later outage could go unreported.

### Fixes

- A 200 pre-flight response with a non-JSON body, or with wrongly typed fields (for example a non-string `decision` or non-list `violations`), no longer raises into the caller.
- Deeply nested tool inputs no longer raise `RecursionError`; nesting beyond 100 levels is replaced with `"[MaxDepth]"`.
- Tool-call events are built when the tool returns, so later mutation of the arguments no longer changes what is recorded. An invalid tool name no longer breaks the wrapped tool.
- Async clients keep references to background tasks, so audit events are no longer lost to garbage collection. Scheduling without a running loop now warns instead of dropping silently.
- Anthropic responses record every text block, not only the first.

### Added

- `flush()` on sync clients and `await flush()` on async clients; `close()` / `aclose()` now send queued events first.
- Every background event carries an `event_id`. The SDK retries once on connection errors and 5xx (not timeouts, so a slow server holds a worker for at most one `timeout` per event), and on 429 only when `Retry-After` is 5 seconds or less, which the SignalVault API does not currently send. `tools.record()` does not retry; its events also carry an `event_id`.
- `Accept: application/json` and `User-Agent: signalvault-python/<version>` headers.
- LICENSE file.

---

## [0.4.0] — 2026-04-29

### Added

- **Agent tool-use capture.** `client.tool(name, fn)` wraps sync or async functions and records each call as an `agent.tool_call` event; `client.tools.record(...)` records manually; `client.with_context(request_id=...)` links tool calls to a parent request.

---

## [0.3.0] — 2026-03-26

### Added

- **Streaming support** for all four clients (`SignalVaultClient`, `AsyncSignalVaultClient`, `AnthropicSignalVaultClient`, `AsyncAnthropicSignalVaultClient`). Pass `stream=True` to `chat.completions.create()` or `messages.create()` and iterate the returned generator as normal. SignalVault fires the `ai.response` audit event automatically after the stream is fully consumed (or closed early).
- **`sv_metadata` parameter** on every `create()` call. Per-call metadata is merged on top of the client-level `metadata` config. The old `metadata` kwarg still works but now emits a `DeprecationWarning` unconditionally.
- **`warn` decision handling** — when the SignalVault backend returns `decision: "warn"` and the client is configured with `debug=True`, a `UserWarning` is now emitted. Previously this was silently ignored in the async clients.
- **Context manager support** for all clients. Sync clients implement `__enter__`/`__exit__`; async clients implement `__aenter__`/`__aexit__`. Use `with SignalVaultClient(...) as client:` or `async with AsyncSignalVaultClient(...) as client:` to ensure the HTTP connection pool is properly closed.
- **`close()` / `aclose()` methods** on all clients for explicit resource cleanup without a context manager.

### Changed

- **Anthropic streaming** now uses `anthropic.messages.stream()` as a proper context manager instead of the undocumented raw iterator path. This is more robust across Anthropic SDK versions. The `stream=True` kwarg is no longer forwarded to the underlying Anthropic client.
- **Anthropic stream text extraction** now checks `event.delta.type == "text_delta"` before appending content, matching the Anthropic SDK's documented event shape and aligning with the Node SDK behaviour.
- **`DeprecationWarning` for `metadata` kwarg** now fires regardless of `debug` mode. Deprecation warnings are always relevant to callers.
- **Internal refactor** — `_BaseSyncClient` and `_BaseAsyncClient` base classes now hold all shared HTTP, config, and audit logic. The four public client classes inherit from these bases, eliminating ~40% code duplication and ensuring fixes are applied consistently.
- **`_parse_decision`** now filters unknown fields from violation objects before constructing `Violation` dataclasses. This prevents `TypeError` crashes if the SignalVault backend adds new violation fields in a future release.
- **`asyncio.create_task()`** calls in async stream `finally` blocks are now guarded with `try/except RuntimeError` to handle the case where the event loop has already shut down, preventing a confusing error from masking the original exception.
- **`_wrap_stream` return type annotations** corrected to `Generator[Any, None, None]` and `AsyncGenerator[Any, None]`.

### Fixed

- `__version__` in `signalvault/__init__.py` was still `"0.2.0"` while `pyproject.toml` declared `0.3.0`. Both now report `0.3.0`.
- Async `_normal` methods in `_AsyncChatCompletions` and `_AsyncAnthropicMessages` were missing the `warn` decision branch, silently swallowing guardrail warnings.
- `httpx.Client` and `httpx.AsyncClient` instances were never closed, leaking OS connections and triggering `ResourceWarning` on garbage collection.

---

## [0.2.0]

Initial public release with sync and async OpenAI and Anthropic client wrappers, preflight guardrails, mirror mode, and configurable timeouts.
