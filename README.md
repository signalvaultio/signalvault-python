# signalvault

AI audit logs and guardrails for your OpenAI Python applications.

## Installation

```bash
pip install signalvault openai
```

## Quick Start

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

## Mirror Mode

Monitor without blocking — requests go directly to OpenAI and are audited asynchronously:

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    mirror_mode=True,
)
```

## Features

- **Automatic Logging** — Every request and response is recorded
- **Pre-flight Guardrails** — Block or redact requests before OpenAI
- **PII Detection** — Detect emails, phone numbers, SSNs
- **Secret Detection** — Block API keys and tokens
- **Token Limits** — Enforce cost controls
- **Model Allowlists** — Restrict which models can be used
- **Mirror Mode** — Observe without blocking

## License

MIT
