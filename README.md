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
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# OpenAI streaming (async)
stream = await async_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# Anthropic streaming
stream = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Write a poem"}],
    max_tokens=1024,
    stream=True,
)
for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text or "", end="", flush=True)
```

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
    metadata={"user_id": "u_123", "feature": "support-chat"},
)
```

## Timeout Configuration

The pre-flight guardrail check is in your request's critical path. SignalVault uses a short timeout and **fails open** — your request always goes through even if the SignalVault API is unreachable:

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    preflight_timeout=2.0,   # seconds — pre-flight check (fails open). Default: 2.0
    timeout=30.0,            # seconds — background/post-flight calls. Default: 30.0
)
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
- **Pre-flight Guardrails** — Block or redact requests before they reach the AI provider
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
