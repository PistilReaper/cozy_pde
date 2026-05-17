from __future__ import annotations

from typing import Any

from .config import LLMProfile, OpenAIEndpointConfig


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


class ResponsesClient:
    def __init__(self, endpoint: OpenAIEndpointConfig) -> None:
        self.endpoint = endpoint
        self.last_model_used = ""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("openai SDK is not installed") from exc

        if not endpoint.api_key:
            raise RuntimeError(f"Environment variable {endpoint.api_key_env} is not set")

        self.client = OpenAI(
            api_key=endpoint.api_key,
            base_url=_normalize_base_url(endpoint.base_url),
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
        self.last_model_used = profile.model
        return self.client.responses.create(
            model=profile.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
            tool_choice="auto",
            temperature=profile.temperature,
            max_output_tokens=profile.max_tokens,
            reasoning={"effort": profile.reasoning_effort},
            text={"verbosity": profile.verbosity},
            store=self.endpoint.store,
            stream=self.endpoint.streaming,
            metadata=metadata or {},
        )
