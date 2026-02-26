"""Basic unit tests for SignalVaultClient (no network calls)."""

from signalvault import SignalVaultClient


def test_client_construction():
    client = SignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
    )
    assert client is not None
    assert client.chat is not None
    assert client.chat.completions is not None


def test_client_config_defaults():
    client = SignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
    )
    assert client._config.base_url == "http://localhost:4000"
    assert client._config.environment == "production"
    assert client._config.debug is False
    assert client._config.mirror_mode is False


def test_client_custom_config():
    client = SignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
        base_url="https://api.signalvault.io/",
        environment="staging",
        debug=True,
        mirror_mode=True,
    )
    assert client._config.base_url == "https://api.signalvault.io"
    assert client._config.environment == "staging"
    assert client._config.debug is True
    assert client._config.mirror_mode is True


def test_streaming_raises_not_implemented():
    client = SignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
    )
    try:
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Streaming" in str(e)
