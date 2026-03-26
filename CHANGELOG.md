# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
