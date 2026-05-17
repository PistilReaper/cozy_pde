from __future__ import annotations

import json
from typing import Any

from .config import FallbackProviderConfig, LLMProfile, OpenAIEndpointConfig


def _normalize_base_url(base_url: str, *, append_v1: bool) -> str:
    normalized = base_url.rstrip("/")
    if not append_v1 or normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _is_gateway_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(code in text for code in ("502", "503", "504")) and any(
        token in text for token in ("gateway", "bad gateway", "timeout", "unavailable")
    )


def _message_text(item: dict[str, Any]) -> str:
    texts: list[str] = []
    for content_item in item.get("content", []):
        text = content_item.get("text")
        if text:
            texts.append(str(text))
    return "\n".join(texts)


def _tool_result_message(item: dict[str, Any]) -> str:
    output_value = item.get("output")
    if isinstance(output_value, str):
        try:
            output_value = json.loads(output_value)
        except json.JSONDecodeError:
            pass
    return json.dumps(
        {
            "type": "tool_result",
            "call_id": item.get("call_id"),
            "output": output_value,
        },
        ensure_ascii=False,
    )


def _ledger_to_messages(input_items: list[dict[str, Any]], instructions: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role", "user"))
            content = _message_text(item)
            if content:
                messages.append({"role": role, "content": content})
            continue
        if item_type == "function_call_output":
            messages.append({"role": "user", "content": _tool_result_message(item)})
    return messages


def _extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")
    return str(content or "")


def _normalize_response(*, text: str, provider: str, model: str, raw_response: Any) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "raw_response": raw_response,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


class ResponsesClient:
    def __init__(
        self,
        endpoint: OpenAIEndpointConfig,
        _responses_config: object | None = None,
        fallback_provider: FallbackProviderConfig | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.fallback_provider = fallback_provider or FallbackProviderConfig(enabled=False)
        self.last_model_used = ""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("openai SDK is not installed") from exc

        if not endpoint.api_key:
            raise RuntimeError(f"Environment variable {endpoint.api_key_env} is not set")

        self._openai_cls = OpenAI
        self.client = OpenAI(
            api_key=endpoint.api_key,
            base_url=_normalize_base_url(endpoint.base_url, append_v1=endpoint.append_v1),
        )
        self.fallback_client = None
        if self.fallback_provider.enabled and self.fallback_provider.api_key:
            self.fallback_client = OpenAI(
                api_key=self.fallback_provider.api_key,
                base_url=_normalize_base_url(
                    self.fallback_provider.base_url,
                    append_v1=self.fallback_provider.append_v1,
                ),
            )

    def _fallback_model_for_profile(self, profile: LLMProfile) -> str | None:
        if profile.name in {"strong_planner", "coder", "router"}:
            return self.fallback_provider.pro_model
        if profile.name in {"log_summarizer", "json_judge"}:
            return self.fallback_provider.flash_model
        return self.fallback_provider.pro_model

    def _create_with_client(
        self,
        *,
        client: Any,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        profile: LLMProfile,
        metadata: dict[str, str] | None,
    ) -> dict[str, Any]:
        raw_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            stream=False,
            metadata=metadata or {},
        )
        return _normalize_response(
            text=_extract_chat_text(raw_response),
            provider=provider,
            model=model,
            raw_response=raw_response,
        )

    def create(
        self,
        *,
        profile: LLMProfile,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Any:
        del tools
        messages = _ledger_to_messages(input_items, instructions)
        self.last_model_used = profile.model
        try:
            return self._create_with_client(
                client=self.client,
                provider=self.endpoint.provider,
                model=profile.model,
                messages=messages,
                profile=profile,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_gateway_error(exc) or self.fallback_client is None:
                raise
            fallback_model = self._fallback_model_for_profile(profile)
            if not fallback_model:
                raise
            self.last_model_used = fallback_model
            return self._create_with_client(
                client=self.fallback_client,
                provider=self.fallback_provider.provider,
                model=fallback_model,
                messages=messages,
                profile=profile,
                metadata=metadata,
            )
