# signalvault

AI audit logs and guardrails for your OpenAI and Anthropic Python applications.

## Installation

```bash
# OpenAI only
pip install signalvault openai

# Anthropic only
pip install signalvault[anthropic]

# Both
pip install signalvault openai signalvault[anthropic]
```

## Quick Start — OpenAI (sync)

```python
import os
from signalvault import SignalVaultClient

client = SignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.signalvault.io",
    environment="production",
)

# Use exactly like OpenAI SDK
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Quick Start — OpenAI (async, FastAPI / async Django)

```python
import os
from signalvault import AsyncSignalVaultClient

client = AsyncSignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.signalvault.io",
)

response = await client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Quick Start — Anthropic

```python
import os
from signalvault import AnthropicSignalVaultClient

client = AnthropicSignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.signalvault.io",
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
)
print(response.content[0].text)
```

## Streaming

Streaming is fully supported for all clients and providers:

```python
# OpenAI streaming (sync)
with client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
) as stream:
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

# OpenAI streaming (async)
stream = await async_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
async with stream:
    async for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)

# Anthropic streaming
with anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Write a poem"}],
    max_tokens=1024,
    stream=True,
) as stream:
    for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)
```

Streams yield what the provider SDK's `create(..., stream=True)` yields: OpenAI chunks (apart from the usage chunk described below), or Anthropic's raw stream events (`message_start`, `content_block_delta`, ...). The request is sent when `create()` is called, so provider errors such as an invalid key are raised there, and no response is recorded for them.

A stream is recorded once: when it ends, when it raises, or when it's closed. A `for` loop doesn't close the stream on `break`, so use `with` / `async with` (or call `stream.close()` / `await stream.close()`) if you may stop reading early. That records the response and closes the connection to the provider right away. Without it, both happen when the stream is garbage-collected (with the async clients, at the latest when `asyncio.run` finishes, which then waits up to 5 seconds for the event to be sent), even if it was never read. Like the provider SDK's stream, it also has `stream.response`, the HTTP response.

For OpenAI streams, the SDK sets `stream_options={"include_usage": True}` so the response is logged with its token counts. OpenAI then sends one extra final chunk that carries only the usage and no choices; the SDK consumes that chunk rather than passing it to you. If you set `include_usage` yourself, your setting is kept: with `True` you receive the usage chunk, whose `choices` list is empty (hence the `if chunk.choices:` guard above), and with `False` the response is logged without token counts. Other stream options you pass are kept. If your OpenAI-compatible server (set through `OPENAI_BASE_URL`) rejects `stream_options`, pass `stream_options=None` and the SDK sends none; responses are then logged without token counts.

## Agent Tool-Use Capture

Log every tool invocation made by your agents — name, input, output, duration, and any error — as auditable events alongside your LLM calls.

### Wrapper API (recommended)

```python
@client.tool("fetch_weather")
def fetch_weather(city: str):
    res = httpx.get(f"https://api.example.com/weather?city={city}")
    return res.json()

# Call it like the original — SignalVault auto-times and audits.
weather = fetch_weather("London")
```

The wrapper records the call asynchronously (no impact on your tool's latency) and passes the result through unchanged. Errors are recorded and re-raised.

```python
# Async client + async tool
@async_client.tool("fetch_weather")
async def fetch_weather(city: str):
    async with httpx.AsyncClient() as http:
        res = await http.get(f"https://api.example.com/weather?city={city}")
        return res.json()

weather = await fetch_weather("London")
```

### Manual API

```python
# Sync — blocks on the first HTTP send so a rejected event (4xx) warns at your call;
# if SignalVault is unreachable or rate limiting, the event is retried in the background
client.tools.record(
    tool_name="fetch_weather",
    tool_input={"city": "London"},
    tool_output={"temp": 12.3},
    duration_ms=142,
)

# Async
await async_client.tools.record(
    tool_name="fetch_weather",
    tool_input={"city": "London"},
    tool_output={"temp": 12.3},
    duration_ms=142,
)
```

### Linking tool calls to a parent LLM turn

Wrap your agent loop in `with_context` and tool calls inside auto-correlate to the given `request_id`:

```python
# Sync
with client.with_context(request_id="agent-turn-abc"):
    llm_response = client.chat.completions.create(...)
    fetch_weather("London")  # auto-linked to 'agent-turn-abc'

# Async
async with async_client.with_context(request_id="agent-turn-abc"):
    llm_response = await async_client.chat.completions.create(...)
    await fetch_weather("London")  # auto-linked to 'agent-turn-abc'
```

Without `with_context`, tool calls are recorded as orphans (no parent request).

### What gets captured (and what to keep out)

When you wrap a tool or call `tools.record()`, SignalVault captures:

- `tool_name` (truncated to 200 bytes)
- `tool_input` — the function arguments, JSON-serialized (capped at 256 KB; oversize values are truncated with a marker)
- `tool_output` — the function return value, JSON-serialized (same 256 KB cap)
- `error` if the tool raises (truncated to 1900 bytes)
- `duration_ms`, `started_at`, and any `metadata` you attach

`tool_input` and `tool_output` are encrypted at rest server-side. `tool_name`, `error`, `started_at` and `metadata` are stored as sent, **not encrypted**. All of it goes on the wire to SignalVault's API: **if you pass user PII, secrets, or API keys as tool arguments, those values will leave your process and be stored in SignalVault.** Recommendations:

- Sanitize sensitive arguments before invoking the wrapped tool, or use the manual `tools.record()` API and pass a redacted copy.
- Don't put secrets in error messages — they end up in `error` verbatim and unencrypted.
- Use `metadata` for non-sensitive identifiers (`user_id`, `feature`, `workspace_id`); avoid putting raw user content in metadata.

## Metadata

Attach contextual metadata to every event for user attribution, analytics, and audit trails:

```python
# Set defaults at client level
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    metadata={"workspace_id": "ws_abc", "env": "production"},
)

# Override per-call
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    sv_metadata={"user_id": "u_123", "feature": "support-chat"},
)
```

`sv_metadata` goes to SignalVault. A plain `metadata=` argument is passed through to the provider untouched (for example Anthropic's `metadata={"user_id": ...}`).

## When SignalVault Is Unavailable

The pre-flight guardrail check is in your request's critical path. If it cannot return a decision — timeout, network error, invalid or revoked API key (401), inactive subscription (402), access denied (403), rate or trial limit (429), server error, an invalid response, or a request that can't be serialized (for example a circular reference in `sv_metadata`) — `fail_mode` decides what happens:

- `"open"` (default): the request goes to the provider **without** guardrails, and the SDK emits a `SignalVaultWarning` (at most once a minute per cause, whether or not `debug` is on). If SignalVault was unreachable, timed out or errored, the request is still recorded in the background, marked `preflight_unavailable`, so it appears in your audit log.
- `"closed"`: the SDK raises `SignalVaultUnavailableError` and the provider is never called. Use this if every request must be checked.

The pre-flight check tells SignalVault which `fail_mode` the client uses, so the audit log shows whether a request that could not be checked went to the provider. `fail_mode` applies to the pre-flight check, so it has no effect in mirror mode.

**Rate limit.** Rule checks are limited to 120 requests per minute per app. Above that, SignalVault still records the request, without running your rules, and answers 429:

- With `fail_mode="open"` the request goes to the provider unchecked, with a `SignalVaultWarning`. Its response and tool calls are recorded as usual, so the audit log stays complete.
- With `fail_mode="closed"` the SDK raises `SignalVaultUnavailableError` (`status` 429) and the provider is never called.

If you run at that volume and need every request checked, use `fail_mode="closed"`. Audit events (responses, tool calls, mirror-mode events) have their own, higher limit of 3,000 per minute per app.

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    fail_mode="closed",      # "open" | "closed". Default: "open"
    preflight_timeout=2.0,   # seconds — pre-flight check. Default: 2.0
    timeout=30.0,            # seconds — background/post-flight calls. Default: 30.0
)
```

`base_url` defaults to `https://api.signalvault.io`. Plain `http://` is refused except for `localhost`, as is a URL with a query string or fragment.

### Background delivery

Audit events (responses, mirror-mode events, tool calls) are queued in memory and sent in the background, without blocking your call:

- **Retries.** An event that fails with a connection error, a timeout, a 5xx or a 429 is retried with exponential backoff: up to 6 attempts over about two minutes (waits of about 4, 8, 16, 32 and 60 seconds). A 429's `Retry-After` is honoured, up to 60 seconds. Other 4xx responses mean the event was rejected, and it is not retried. Each event carries an `event_id`, so a retry is never double-counted. `tools.record()` makes the first attempt itself; if that one can be retried, the rest happen in the background.
- **Queue.** Up to 1,000 events per client wait to be sent. When the queue is full, the oldest are dropped to make room.
- **Drops.** Dropping is the last resort, and the SDK says so: a `SignalVaultWarning` whenever events are dropped (a full queue, all attempts used, or events still queued when the client is closed, the event loop stops or the process exits), at most once a minute per cause, with the number dropped so far.
- **Order.** A call's request is sent before its response, retries included.

The sync clients send from up to 4 background threads per client; the async clients send from tasks on the running event loop.

## Shutting Down

Background events are queued in memory. Wait for them before a short-lived process exits:

```python
client.close()          # sync: sends queued events (up to 5s), then closes the connection pool
await client.aclose()   # async: same; required — pending async events are lost if the loop exits first

# Or only wait, keeping the client open:
client.flush()
await async_client.flush()
```

`close()`, `aclose()` and `flush()` take a `timeout` in seconds. Events still queued when `close()` or `aclose()` gives up are dropped, with a warning; `flush()` leaves them queued. An event waiting to be retried can take up to two minutes, so pass a longer `timeout` if you need every event through an outage.

If a sync client is never closed, the SDK gives its queued events up to 3 seconds when the interpreter exits, retrying without waiting for the backoff, and never holds the process longer. Events still queued then are dropped, with a warning.

## Error Handling

```python
from signalvault import SignalVaultBlockedError, SignalVaultUnavailableError

try:
    response = client.chat.completions.create(model="gpt-4", messages=[...])
except SignalVaultBlockedError as e:
    print("Blocked by guardrails:", [v.type for v in e.violations])
    print("Details:", e.dashboard_url)
except SignalVaultUnavailableError as e:
    # Only raised with fail_mode="closed"
    print("Guardrail check unavailable:", e.status)
```

Both are subclasses of `RuntimeError`.

### Warnings

Everything the SDK warns about is a `SignalVaultWarning`. A guardrail `warn` decision (emitted with `debug=True`) uses the subclass `SignalVaultViolationWarning`, so you can treat policy warnings separately from delivery problems:

```python
import warnings
from signalvault import SignalVaultViolationWarning

warnings.simplefilter("error", SignalVaultViolationWarning)
```

## Mirror Mode

In mirror mode, requests go directly to the AI provider first and SignalVault audits them asynchronously — no latency added, never blocks:

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    mirror_mode=True,
)
```

## Features

- **Automatic Logging** — Every request and response is recorded
- **Pre-flight Guardrails** — Block requests that break your rules before they reach the AI provider
- **Redaction at rest** — Rules with the `redact` action remove matches from what SignalVault stores. The request sent to the AI provider is **not** modified.
- **PII Detection** — Detect emails, phone numbers, SSNs
- **Secret Detection** — Block API keys and tokens
- **Token Limits** — Enforce cost controls
- **Model Allowlists** — Restrict which models can be used
- **Streaming** — Full streaming support for OpenAI and Anthropic
- **Async Support** — `AsyncSignalVaultClient` and `AsyncAnthropicSignalVaultClient` for async codebases
- **Mirror Mode** — Observe without blocking
- **Metadata** — Tag every event with user_id, feature, workspace_id, etc.
- **Multi-provider** — OpenAI and Anthropic/Claude support

## License

MIT
