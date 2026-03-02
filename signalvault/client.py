"""SignalVault client — wraps OpenAI and Anthropic with guardrails and audit logging."""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, Iterator, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Config & shared types
# ---------------------------------------------------------------------------

@dataclass
class SignalVaultConfig:
    api_key: str
    base_url: str = "http://localhost:4000"
    environment: str = "production"
    debug: bool = False
    mirror_mode: bool = False
    # Timeout for pre-flight /v1/events call (critical path). Fails open on timeout.
    preflight_timeout: float = 2.0
    # Timeout for background/post-flight calls.
    timeout: float = 30.0
    # Default metadata attached to every event. Merged with per-call metadata.
    metadata: Dict[str, Any] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_metadata(config_meta: Dict[str, Any], call_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**config_meta, **(call_meta or {})}


def _parse_decision(data: dict) -> Decision:
    violations = [Violation(**v) for v in data.get("violations", [])]
    return Decision(
        decision=data.get("decision", "allow"),
        violations=violations,
        redactions=data.get("redactions", []),
    )


# ---------------------------------------------------------------------------
# SignalVaultClient (sync, OpenAI)
# ---------------------------------------------------------------------------

class _ChatCompletions:
    """Proxies `client.chat.completions.create(...)` with SignalVault guardrails."""

    def __init__(self, sv: "SignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any):
        request_id = str(uuid.uuid4())
        metadata = _merge_metadata(self._sv._config.metadata, kwargs.pop("metadata", None))
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        decision = self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )
        if decision.decision == "warn" and self._sv._config.debug:
            import warnings
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}")

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        self._sv._send_response(request_id, response, kwargs.get("model", ""), metadata)
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        try:
            self._sv._send_audit(request_id, kwargs, response, metadata)
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool
    ) -> Generator:
        chunks: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                chunks.append(delta.content)
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            yield chunk

        output = "".join(chunks)
        model = kwargs.get("model", "")
        try:
            if mirror:
                self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, prompt_tokens, completion_tokens, metadata
                )
            else:
                self._sv._send_response_from_parts(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata
                )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()


class _Chat:
    def __init__(self, sv: "SignalVaultClient"):
        self.completions = _ChatCompletions(sv)


class SignalVaultClient:
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
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        self._config = SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        )
        self._openai = OpenAI(api_key=openai_api_key)
        self._http = httpx.Client()
        self.chat = _Chat(self)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                import traceback
                traceback.print_exc()
        return Decision()

    def _send_response(self, request_id: str, response: Any, model: str, metadata: dict) -> None:
        from openai.types.chat import ChatCompletion
        output = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        self._send_response_from_parts(
            request_id, model, output,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            metadata,
        )

    def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict
    ) -> None:
        try:
            self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

    def _send_audit(self, request_id: str, params: dict, response: Any, metadata: dict) -> None:
        output = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        self._send_audit_from_parts(
            request_id, params.get("model", ""), params.get("messages", []),
            output,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            metadata,
        )

    def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict
    ) -> None:
        url = f"{self._config.base_url}/v1/events"
        headers = self._headers()
        timeout = self._config.timeout

        def post_request():
            self._http.post(url, headers=headers, timeout=timeout, json={
                "type": "ai.request",
                "request_id": request_id,
                "environment": self._config.environment,
                "provider": "openai",
                "model": model,
                "metadata": metadata,
                "payload": {"messages": messages, "monitor_mode": True},
            })

        def post_response():
            self._http.post(url, headers=headers, timeout=timeout, json={
                "type": "ai.response",
                "request_id": request_id,
                "environment": self._config.environment,
                "provider": "openai",
                "model": model,
                "metadata": metadata,
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "monitor_mode": True,
                },
            })

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(post_request)
            f2 = executor.submit(post_response)
            for f in (f1, f2):
                try:
                    f.result()
                except Exception:
                    if self._config.debug:
                        import traceback
                        traceback.print_exc()


# ---------------------------------------------------------------------------
# AsyncSignalVaultClient (async, OpenAI)
# ---------------------------------------------------------------------------

class _AsyncChatCompletions:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any):
        request_id = str(uuid.uuid4())
        metadata = _merge_metadata(self._sv._config.metadata, kwargs.pop("metadata", None))
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        await self._sv._send_response(request_id, response, kwargs.get("model", ""), metadata)
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        try:
            await self._sv._send_audit(request_id, kwargs, response, metadata)
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool
    ) -> AsyncGenerator:
        chunks: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                chunks.append(delta.content)
            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            yield chunk

        output = "".join(chunks)
        model = kwargs.get("model", "")
        try:
            if mirror:
                await self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, prompt_tokens, completion_tokens, metadata
                )
            else:
                await self._sv._send_response_from_parts(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata
                )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()


class _AsyncChat:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self.completions = _AsyncChatCompletions(sv)


class AsyncSignalVaultClient:
    """
    Async OpenAI wrapper with SignalVault guardrails. Use in FastAPI, async Django, etc.

    Usage::

        from signalvault import AsyncSignalVaultClient

        client = AsyncSignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        response = await client.chat.completions.create(
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
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import AsyncOpenAI
        self._config = SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        )
        self._openai = AsyncOpenAI(api_key=openai_api_key)
        self._http = httpx.AsyncClient()
        self.chat = _AsyncChat(self)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    async def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                import traceback
                traceback.print_exc()
        return Decision()

    async def _send_response(self, request_id: str, response: Any, model: str, metadata: dict) -> None:
        output = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        await self._send_response_from_parts(
            request_id, model, output,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            metadata,
        )

    async def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict
    ) -> None:
        try:
            await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "openai",
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

    async def _send_audit(self, request_id: str, params: dict, response: Any, metadata: dict) -> None:
        output = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        await self._send_audit_from_parts(
            request_id, params.get("model", ""), params.get("messages", []),
            output,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            metadata,
        )

    async def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict
    ) -> None:
        url = f"{self._config.base_url}/v1/events"
        base = {
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": "openai",
            "model": model,
            "metadata": metadata,
        }
        await asyncio.gather(
            self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.request",
                "payload": {"messages": messages, "monitor_mode": True},
            }),
            self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.response",
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "monitor_mode": True,
                },
            }),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# AnthropicSignalVaultClient (sync)
# ---------------------------------------------------------------------------

class _AnthropicMessages:
    def __init__(self, sv: "AnthropicSignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any):
        request_id = str(uuid.uuid4())
        metadata = _merge_metadata(self._sv._config.metadata, kwargs.pop("metadata", None))
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        decision = self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )

        response = self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        self._sv._send_response_from_parts(
            request_id, kwargs.get("model", ""),
            response.content[0].text if response.content else "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        response = self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        try:
            self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), kwargs.get("messages", []),
                response.content[0].text if response.content else "",
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata,
            )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool
    ) -> Generator:
        chunks: List[str] = []
        input_tokens = 0
        output_tokens = 0

        for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta" and hasattr(event, "delta"):
                    chunks.append(getattr(event.delta, "text", "") or "")
                elif event.type == "message_start" and hasattr(event, "message"):
                    usage = getattr(event.message, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                elif event.type == "message_delta" and hasattr(event, "usage"):
                    output_tokens = getattr(event.usage, "output_tokens", 0) or 0
            yield event

        output = "".join(chunks)
        model = kwargs.get("model", "")
        try:
            if mirror:
                self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, input_tokens, output_tokens, metadata
                )
            else:
                self._sv._send_response_from_parts(
                    request_id, model, output, input_tokens, output_tokens, metadata
                )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()


class AnthropicSignalVaultClient:
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

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
        )
    """

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
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
        self._config = SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        )
        self._anthropic = _anthropic.Anthropic(api_key=anthropic_api_key)
        self._http = httpx.Client()
        self.messages = _AnthropicMessages(self)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "anthropic",
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                import traceback
                traceback.print_exc()
        return Decision()

    def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        input_tokens: int, output_tokens: int, metadata: dict
    ) -> None:
        try:
            self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "anthropic",
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

    def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        input_tokens: int, output_tokens: int, metadata: dict
    ) -> None:
        url = f"{self._config.base_url}/v1/events"
        headers = self._headers()
        timeout = self._config.timeout
        base = {
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": "anthropic",
            "model": model,
            "metadata": metadata,
        }

        def post_request():
            self._http.post(url, headers=headers, timeout=timeout, json={
                **base,
                "type": "ai.request",
                "payload": {"messages": messages, "monitor_mode": True},
            })

        def post_response():
            self._http.post(url, headers=headers, timeout=timeout, json={
                **base,
                "type": "ai.response",
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                    },
                    "monitor_mode": True,
                },
            })

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(post_request)
            f2 = executor.submit(post_response)
            for f in (f1, f2):
                try:
                    f.result()
                except Exception:
                    if self._config.debug:
                        import traceback
                        traceback.print_exc()


# ---------------------------------------------------------------------------
# AsyncAnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class _AsyncAnthropicMessages:
    def __init__(self, sv: "AsyncAnthropicSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any):
        request_id = str(uuid.uuid4())
        metadata = _merge_metadata(self._sv._config.metadata, kwargs.pop("metadata", None))
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )

        response = await self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        await self._sv._send_response_from_parts(
            request_id, kwargs.get("model", ""),
            response.content[0].text if response.content else "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata,
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool):
        response = await self._sv._anthropic.messages.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        try:
            await self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), kwargs.get("messages", []),
                response.content[0].text if response.content else "",
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata,
            )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool
    ) -> AsyncGenerator:
        chunks: List[str] = []
        input_tokens = 0
        output_tokens = 0

        async for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta" and hasattr(event, "delta"):
                    chunks.append(getattr(event.delta, "text", "") or "")
                elif event.type == "message_start" and hasattr(event, "message"):
                    usage = getattr(event.message, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                elif event.type == "message_delta" and hasattr(event, "usage"):
                    output_tokens = getattr(event.usage, "output_tokens", 0) or 0
            yield event

        output = "".join(chunks)
        model = kwargs.get("model", "")
        try:
            if mirror:
                await self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, input_tokens, output_tokens, metadata
                )
            else:
                await self._sv._send_response_from_parts(
                    request_id, model, output, input_tokens, output_tokens, metadata
                )
        except Exception:
            if self._sv._config.debug:
                import traceback
                traceback.print_exc()


class AsyncAnthropicSignalVaultClient:
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

        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
        )
    """

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
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
        self._config = SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        )
        self._anthropic = _anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._http = httpx.AsyncClient()
        self.messages = _AsyncAnthropicMessages(self)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    async def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "anthropic",
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                import traceback
                traceback.print_exc()
        return Decision()

    async def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        input_tokens: int, output_tokens: int, metadata: dict
    ) -> None:
        try:
            await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": "anthropic",
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                import traceback
                traceback.print_exc()

    async def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        input_tokens: int, output_tokens: int, metadata: dict
    ) -> None:
        url = f"{self._config.base_url}/v1/events"
        base = {
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": "anthropic",
            "model": model,
            "metadata": metadata,
        }
        await asyncio.gather(
            self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.request",
                "payload": {"messages": messages, "monitor_mode": True},
            }),
            self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.response",
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                    },
                    "monitor_mode": True,
                },
            }),
            return_exceptions=True,
        )
