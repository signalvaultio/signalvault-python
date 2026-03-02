"""Unit tests for SignalVault Python SDK (no network calls)."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from signalvault import (
    SignalVaultClient,
    AsyncSignalVaultClient,
    AnthropicSignalVaultClient,
    AsyncAnthropicSignalVaultClient,
)
from signalvault.client import Decision, _merge_metadata


# ---------------------------------------------------------------------------
# SignalVaultClient (sync, OpenAI)
# ---------------------------------------------------------------------------

class TestSignalVaultClientConstruction:
    def test_basic_construction(self):
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        assert client is not None
        assert client.chat is not None
        assert client.chat.completions is not None

    def test_defaults(self):
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        assert client._config.base_url == "http://localhost:4000"
        assert client._config.environment == "production"
        assert client._config.debug is False
        assert client._config.mirror_mode is False
        assert client._config.preflight_timeout == 2.0
        assert client._config.timeout == 30.0
        assert client._config.metadata == {}

    def test_custom_config(self):
        client = SignalVaultClient(
            api_key="sk_test_abc",
            openai_api_key="sk-fake",
            base_url="https://api.signalvault.io/",
            environment="staging",
            debug=True,
            mirror_mode=True,
            preflight_timeout=0.5,
            timeout=5.0,
            metadata={"user_id": "u_123"},
        )
        assert client._config.base_url == "https://api.signalvault.io"
        assert client._config.environment == "staging"
        assert client._config.debug is True
        assert client._config.mirror_mode is True
        assert client._config.preflight_timeout == 0.5
        assert client._config.timeout == 5.0
        assert client._config.metadata == {"user_id": "u_123"}

    def test_strips_trailing_slash(self):
        client = SignalVaultClient(
            api_key="sk_test_abc", openai_api_key="sk-fake",
            base_url="https://api.signalvault.io///",
        )
        assert client._config.base_url == "https://api.signalvault.io"


class TestMetadata:
    def test_merge_metadata_config_only(self):
        result = _merge_metadata({"user_id": "u_1"}, None)
        assert result == {"user_id": "u_1"}

    def test_merge_metadata_call_overrides(self):
        result = _merge_metadata({"user_id": "u_1", "env": "prod"}, {"user_id": "u_2", "feature": "chat"})
        assert result == {"user_id": "u_2", "env": "prod", "feature": "chat"}

    def test_merge_metadata_empty(self):
        result = _merge_metadata({}, {})
        assert result == {}


class TestTimeout:
    def test_preflight_timeout_default(self):
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client._config.preflight_timeout == 2.0

    def test_custom_preflight_timeout(self):
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake", preflight_timeout=0.5
        )
        assert client._config.preflight_timeout == 0.5

    def test_fail_open_on_connection_error(self):
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            base_url="http://127.0.0.1:1",  # unreachable
            preflight_timeout=0.1,
        )
        decision = client._send_request("req-1", {"model": "gpt-4", "messages": []}, {})
        assert decision.decision == "allow"
        assert decision.violations == []


class TestStreamingNotImplemented:
    def test_streaming_no_longer_raises(self):
        """Streaming was previously NotImplementedError — now it should be implemented."""
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        # We can't actually call OpenAI in tests, but we verify the method exists
        # and does NOT raise NotImplementedError for stream=True at dispatch time
        import inspect
        completions = client.chat.completions
        assert hasattr(completions, "create")
        source = inspect.getsource(completions.create)
        assert "NotImplementedError" not in source


# ---------------------------------------------------------------------------
# AsyncSignalVaultClient
# ---------------------------------------------------------------------------

class TestAsyncSignalVaultClient:
    def test_construction(self):
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client is not None
        assert client.chat is not None
        assert client.chat.completions is not None

    def test_defaults(self):
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client._config.preflight_timeout == 2.0
        assert client._config.timeout == 30.0

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout(self):
        client = AsyncSignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            base_url="http://127.0.0.1:1",
            preflight_timeout=0.1,
        )
        decision = await client._send_request("req-1", {"model": "gpt-4", "messages": []}, {})
        assert decision.decision == "allow"


# ---------------------------------------------------------------------------
# AnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class TestAnthropicSignalVaultClient:
    def test_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install signalvault"):
                AnthropicSignalVaultClient(
                    api_key="sk_test", anthropic_api_key="sk-ant-fake"
                )

    def test_construction_with_mock_sdk(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            )
            assert client is not None
            assert client.messages is not None
            assert client._config.preflight_timeout == 2.0

    def test_metadata_config(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test",
                anthropic_api_key="sk-ant-fake",
                metadata={"workspace": "ws_1"},
            )
            assert client._config.metadata == {"workspace": "ws_1"}

    def test_fail_open_on_connection_error(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="http://127.0.0.1:1",
                preflight_timeout=0.1,
            )
            decision = client._send_request("req-1", {"model": "claude-3-5-sonnet-20241022", "messages": []}, {})
            assert decision.decision == "allow"


# ---------------------------------------------------------------------------
# AsyncAnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class TestAsyncAnthropicSignalVaultClient:
    def test_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install signalvault"):
                AsyncAnthropicSignalVaultClient(
                    api_key="sk_test", anthropic_api_key="sk-ant-fake"
                )

    def test_construction_with_mock_sdk(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            )
            assert client is not None
            assert client.messages is not None

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="http://127.0.0.1:1",
                preflight_timeout=0.1,
            )
            decision = await client._send_request(
                "req-1", {"model": "claude-3-5-sonnet-20241022", "messages": []}, {}
            )
            assert decision.decision == "allow"
