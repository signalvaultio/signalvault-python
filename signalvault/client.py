"""SignalVault client — wraps OpenAI with guardrails and audit logging."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion


@dataclass
class SignalVaultConfig:
    api_key: str
    openai_api_key: str
    base_url: str = "http://localhost:4000"
    environment: str = "production"
    debug: bool = False
    mirror_mode: bool = False


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
    redactions: list = field(default_factory=list)


class _ChatCompletions:
    """Proxies `client.chat.completions.create(...)` with SignalVault guardrails."""

    def __init__(self, sv: "SignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any) -> ChatCompletion:
        request_id = str(uuid.uuid4())
        stream = kwargs.get("stream", False)

        if stream:
            raise NotImplementedError(
                "Streaming is not yet supported in the Python SDK. "
                "Use stream=False or the Node.js SDK for streaming."
            )

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs)

        return self._normal(request_id, kwargs)

    # ------------------------------------------------------------------
    # Normal mode — pre-flight check, then OpenAI, then post-flight
    # ------------------------------------------------------------------

    def _normal(self, request_id: str, kwargs: dict) -> ChatCompletion:
        decision = self._sv._send_request(request_id, kwargs)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )

        response = self._sv._openai.chat.completions.create(**kwargs)
        self._sv._send_response(request_id, response, kwargs.get("model", "gpt-4"))
        return response

    # ------------------------------------------------------------------
    # Mirror mode — OpenAI first, audit asynchronously
    # ------------------------------------------------------------------

    def _mirror(self, request_id: str, kwargs: dict) -> ChatCompletion:
        response = self._sv._openai.chat.completions.create(**kwargs)
        try:
            self._sv._send_audit(request_id, kwargs, response)
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()
        return response


class _Chat:
    def __init__(self, sv: "SignalVaultClient"):
        self.completions = _ChatCompletions(sv)


class SignalVaultClient:
    """
    Drop-in OpenAI wrapper with SignalVault guardrails.

    Usage::

        from signalvault import SignalVaultClient

        client = SignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    """

    def __init__(
        self,
        api_key: str,
        openai_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
    ):
        self._config = SignalVaultConfig(
            api_key=api_key,
            openai_api_key=openai_api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
        )
        self._openai = OpenAI(api_key=openai_api_key)
        self._http = httpx.Client(timeout=30)
        self.chat = _Chat(self)

    # ------------------------------------------------------------------
    # Internal API helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    def _send_request(self, request_id: str, params: dict) -> Decision:
        try:
            resp = self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": params.get("model", "gpt-4"),
                    "metadata": {},
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                violations = [
                    Violation(**v) for v in data.get("violations", [])
                ]
                return Decision(
                    decision=data.get("decision", "allow"),
                    violations=violations,
                    redactions=data.get("redactions", []),
                )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

        return Decision()

    def _send_response(
        self, request_id: str, response: ChatCompletion, model: str
    ) -> None:
        try:
            output = (response.choices[0].message.content or "") if response.choices else ""
            usage = response.usage
            self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": model,
                    "metadata": {},
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": usage.completion_tokens if usage else 0,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

    def _send_audit(
        self, request_id: str, params: dict, response: ChatCompletion
    ) -> None:
        output = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage

        # ai.request
        self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            json={
                "type": "ai.request",
                "request_id": request_id,
                "environment": self._config.environment,
                "provider": "openai",
                "model": params.get("model", "gpt-4"),
                "metadata": {},
                "payload": {"messages": params.get("messages", []), "monitor_mode": True},
            },
        )

        # ai.response
        self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            json={
                "type": "ai.response",
                "request_id": request_id,
                "environment": self._config.environment,
                "provider": "openai",
                "model": params.get("model", "gpt-4"),
                "metadata": {},
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                    },
                    "monitor_mode": True,
                },
            },
        )
