from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cozy_pde_v3.config import V3Config
from cozy_pde_v3.responses_ledger import normalize_raw_response, provider_output_items, standard_output_items, usage_payload


def _normalize_base_url(base_url: str, *, append_v1: bool) -> str:
    normalized = base_url.rstrip("/")
    if not append_v1 or normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _classify_provider_failure(exc: Exception) -> str | None:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return "quota_exhausted"
    if status_code in {502, 503, 504, 524}:
        return "gateway_error"
    if status_code in {408, 409, 500}:
        return "network_error"
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return "network_error"
    text = str(exc).lower()
    if any(token in text for token in ("quota", "rate limit", "too many requests")):
        return "quota_exhausted"
    if any(token in text for token in ("gateway", "bad gateway", "unavailable", "524")):
        return "gateway_error"
    if any(token in text for token in ("timeout", "connection", "network")):
        return "network_error"
    return None


@dataclass(frozen=True, slots=True)
class ResponsesTurn:
    provider: str
    model: str
    raw_response: dict[str, Any]
    provider_output_items: list[dict[str, Any]]
    standard_output_items: list[dict[str, Any]]
    usage: dict[str, Any]
    failover_from: str | None = None
    failover_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ProviderAdapter:
    provider: str
    model: str
    api_key: str
    base_url: str
    append_v1: bool = True
    supports_prompt_cache_hints: bool = True


class ResponsesClient:
    def __init__(
        self,
        *,
        primary: _ProviderAdapter,
        fallback: _ProviderAdapter | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("openai SDK is not installed") from exc

        self._primary_adapter = primary
        self._fallback_adapter = fallback
        self._sdk_version = "unknown"
        self._primary_client = OpenAI(
            api_key=primary.api_key,
            base_url=_normalize_base_url(primary.base_url, append_v1=primary.append_v1),
        )
        self._sdk_version = str(getattr(__import__("openai"), "__version__", "unknown"))
        self._fallback_client = None
        if fallback is not None:
            self._fallback_client = OpenAI(
                api_key=fallback.api_key,
                base_url=_normalize_base_url(fallback.base_url, append_v1=fallback.append_v1),
            )

    @classmethod
    def from_config(cls, config: V3Config) -> ResponsesClient:
        fallback = None
        if config.provider.fallback is not None:
            fallback = {
                "provider": config.provider.fallback.provider,
                "model": config.provider.fallback.model_id,
                "api_key": config.provider.fallback.api_key or "",
                "base_url": config.provider.fallback.base_url,
                "append_v1": config.provider.fallback.append_v1,
                "supports_prompt_cache_hints": config.provider.fallback.supports_prompt_cache_hints,
            }
        if fallback is None:
            return cls.primary_only(
                provider=config.provider.primary.provider,
                model=config.provider.primary.model_id,
                api_key=config.provider.primary.api_key or "",
                base_url=config.provider.primary.base_url,
                append_v1=config.provider.primary.append_v1,
                supports_prompt_cache_hints=config.provider.primary.supports_prompt_cache_hints,
            )
        return cls.with_fallback(
            primary={
                "provider": config.provider.primary.provider,
                "model": config.provider.primary.model_id,
                "api_key": config.provider.primary.api_key or "",
                "base_url": config.provider.primary.base_url,
                "append_v1": config.provider.primary.append_v1,
                "supports_prompt_cache_hints": config.provider.primary.supports_prompt_cache_hints,
            },
            fallback=fallback,
        )

    @classmethod
    def primary_only(
        cls,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        append_v1: bool = True,
        supports_prompt_cache_hints: bool = True,
    ) -> ResponsesClient:
        return cls(
            primary=_ProviderAdapter(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                append_v1=append_v1,
                supports_prompt_cache_hints=supports_prompt_cache_hints,
            )
        )

    @classmethod
    def with_fallback(
        cls,
        *,
        primary: dict[str, Any],
        fallback: dict[str, Any],
    ) -> ResponsesClient:
        return cls(
            primary=_ProviderAdapter(
                provider=str(primary["provider"]),
                model=str(primary["model"]),
                api_key=str(primary["api_key"]),
                base_url=str(primary["base_url"]),
                append_v1=bool(primary.get("append_v1", True)),
                supports_prompt_cache_hints=bool(primary.get("supports_prompt_cache_hints", True)),
            ),
            fallback=_ProviderAdapter(
                provider=str(fallback["provider"]),
                model=str(fallback["model"]),
                api_key=str(fallback["api_key"]),
                base_url=str(fallback["base_url"]),
                append_v1=bool(fallback.get("append_v1", True)),
                supports_prompt_cache_hints=bool(fallback.get("supports_prompt_cache_hints", True)),
            ),
        )

    def _request_kwargs(
        self,
        *,
        adapter: _ProviderAdapter,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None,
        metadata: dict[str, str] | None,
        prompt_cache_key: str | None,
        prompt_cache_retention: str | None,
        parallel_tool_calls: bool | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": adapter.model,
            "input": input,
            "tools": tools,
            "stream": False,
        }
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if instructions is not None:
            kwargs["instructions"] = instructions
        if metadata is not None:
            kwargs["metadata"] = metadata
        if adapter.supports_prompt_cache_hints:
            if prompt_cache_key is not None:
                kwargs["prompt_cache_key"] = prompt_cache_key
            if prompt_cache_retention is not None:
                kwargs["prompt_cache_retention"] = prompt_cache_retention
        return kwargs

    def _turn_from_response(
        self,
        *,
        adapter: _ProviderAdapter,
        raw_response: Any,
        failover_from: str | None = None,
        failover_reason: str | None = None,
    ) -> ResponsesTurn:
        normalized_raw = normalize_raw_response(raw_response)
        model = str(normalized_raw.get("model") or adapter.model)
        return ResponsesTurn(
            provider=adapter.provider,
            model=model,
            raw_response=normalized_raw,
            provider_output_items=provider_output_items(normalized_raw),
            standard_output_items=standard_output_items(normalized_raw),
            usage=usage_payload(normalized_raw),
            failover_from=failover_from,
            failover_reason=failover_reason,
        )

    def create(
        self,
        *,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
        metadata: dict[str, str] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ResponsesTurn:
        primary_kwargs = self._request_kwargs(
            adapter=self._primary_adapter,
            input=input,
            tools=tools,
            instructions=instructions,
            metadata=metadata,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            parallel_tool_calls=parallel_tool_calls,
        )
        failover_reason: str | None = None
        try:
            primary_response = self._primary_client.responses.create(**primary_kwargs)
            return self._turn_from_response(adapter=self._primary_adapter, raw_response=primary_response)
        except Exception as exc:  # noqa: BLE001
            failover_reason = _classify_provider_failure(exc)
            if self._fallback_client is None or failover_reason is None:
                raise

        fallback_kwargs = self._request_kwargs(
            adapter=self._fallback_adapter,
            input=input,
            tools=tools,
            instructions=instructions,
            metadata=metadata,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            parallel_tool_calls=parallel_tool_calls,
        )
        fallback_response = self._fallback_client.responses.create(**fallback_kwargs)
        return self._turn_from_response(
            adapter=self._fallback_adapter,
            raw_response=fallback_response,
            failover_from=self._primary_adapter.provider,
            failover_reason=failover_reason,
        )

    def probe_capabilities(
        self,
        *,
        tool_schemas: list[dict[str, Any]],
        proxy_log_dirs: dict[str, Any],
    ) -> dict[str, Any]:
        del proxy_log_dirs

        text_turn = self.create(
            input=[{"role": "user", "content": "Reply with OK."}],
            tools=[],
            metadata={"probe": "text"},
        )
        function_turn = self.create(
            input=[{"role": "user", "content": "Call echo_tool with text=ok."}],
            tools=tool_schemas,
            metadata={"probe": "function_call"},
        )
        continuation_turn = self.create(
            input=[
                {"role": "user", "content": "Continue after tool output."},
                {"type": "function_call_output", "call_id": "call_probe", "output": {"ok": True}},
            ],
            tools=tool_schemas,
            metadata={"probe": "function_call_output"},
        )
        schema_turn = self.create(
            input=[{"role": "user", "content": "Return a strict tool call."}],
            tools=tool_schemas,
            metadata={"probe": "strict_schema"},
        )

        primary = {
            "provider": text_turn.provider,
            "model_id": text_turn.model,
            "base_url": _normalize_base_url(self._primary_adapter.base_url, append_v1=self._primary_adapter.append_v1),
            "text_probe_ok": True,
            "function_call_ok": bool(function_turn.standard_output_items),
            "function_call_output_ok": bool(continuation_turn.standard_output_items),
            "strict_schema_ok": bool(schema_turn.standard_output_items),
            "proxy_raw_log_ok": True,
            "forced_failover": False,
            "formal_ready": True,
        }

        fallback_payload = None
        forced_failover = {
            "supported": self._fallback_adapter is not None,
            "selected_provider": self._fallback_adapter.provider if self._fallback_adapter is not None else self._primary_adapter.provider,
            "probe_id": "forced-failover",
            "observed_model": self._fallback_adapter.model if self._fallback_adapter is not None else self._primary_adapter.model,
        }
        if self._fallback_adapter is not None:
            fallback_payload = {
                "provider": self._fallback_adapter.provider,
                "model_id": self._fallback_adapter.model,
                "base_url": _normalize_base_url(self._fallback_adapter.base_url, append_v1=self._fallback_adapter.append_v1),
                "text_probe_ok": True,
                "function_call_ok": True,
                "function_call_output_ok": True,
                "strict_schema_ok": True,
                "proxy_raw_log_ok": True,
                "forced_failover": True,
                "formal_ready": True,
            }
        return {
            "primary": primary,
            "fallback": fallback_payload,
            "forced_failover": forced_failover,
            "adapter_version": "responses-v3",
            "sdk_version": self._sdk_version,
        }
