"""SignalVault Python SDK — AI audit logs and guardrails for OpenAI and Anthropic applications."""

# Defined before the imports below: client.py reads it for the User-Agent header.
__version__ = "0.7.0"

from .client import (  # noqa: E402
    AnthropicSignalVaultClient,
    AsyncAnthropicSignalVaultClient,
    AsyncSignalVaultClient,
    AsyncSignalVaultStream,
    DroppedEvents,
    SignalVaultBlockedError,
    SignalVaultClient,
    SignalVaultStream,
    SignalVaultUnavailableError,
    SignalVaultViolationWarning,
    SignalVaultWarning,
)
from .tools import ToolContext, ToolRecordOptions  # noqa: E402

__all__ = [
    "SignalVaultClient",
    "AsyncSignalVaultClient",
    "AnthropicSignalVaultClient",
    "AsyncAnthropicSignalVaultClient",
    "SignalVaultBlockedError",
    "SignalVaultUnavailableError",
    "SignalVaultWarning",
    "SignalVaultViolationWarning",
    "SignalVaultStream",
    "AsyncSignalVaultStream",
    "DroppedEvents",
    "ToolContext",
    "ToolRecordOptions",
]
