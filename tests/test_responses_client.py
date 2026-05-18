from __future__ import annotations

import sys
import types

from agent_runner.config import FallbackProviderConfig, LLMProfile, OpenAIEndpointConfig
from agent_runner.json_action_client import JsonActionClient
from agent_runner.responses_items import user_text


def test_json_action_client_uses_chat_completions_and_normalizes_text_output(monkeypatch):
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured["create"] = kwargs
            return types.SimpleNamespace(
                id="chatcmpl_1",
                model="gpt-5.4",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"type":"final","message":"RUNNER_FINALIZED"}'
                        )
                    )
                ],
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    client = JsonActionClient(
        OpenAIEndpointConfig(base_url="https://example.com", api_key_env="LLM_API_KEY", api_key="test-key"),
    )
    response = client.create(
        profile=LLMProfile(name="probe", model="gpt-5.4", reasoning_effort="high", verbosity="medium"),
        input_items=[user_text("Reply with JSON.")],
        tools=[],
        instructions="Reply with ok.",
    )

    assert captured["init"]["base_url"] == "https://example.com/v1"
    assert captured["create"]["model"] == "gpt-5.4"
    assert captured["create"]["messages"][0]["role"] == "system"
    assert captured["create"]["messages"][1]["role"] == "user"
    assert response["provider"] == "third_party_openai_compatible"
    assert response["output"][0]["content"][0]["text"] == '{"type":"final","message":"RUNNER_FINALIZED"}'


def test_json_action_client_falls_back_to_deepseek_on_gateway_error(monkeypatch):
    class GatewayError(RuntimeError):
        status_code = 502

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            if "deepseek" not in self.base_url:
                raise GatewayError("502 bad gateway")
            return types.SimpleNamespace(
                id="chatcmpl_2",
                model=kwargs["model"],
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"type":"final","message":"RUNNER_FINALIZED"}'
                        )
                    )
                ],
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    client = JsonActionClient(
        OpenAIEndpointConfig(base_url="https://example.com", api_key_env="LLM_API_KEY", api_key="test-key"),
        fallback_provider=FallbackProviderConfig(
            enabled=True,
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            api_key="deepseek-key",
            append_v1=False,
            pro_model="deepseek-v4-pro",
            flash_model="deepseek-v4-flash",
        ),
    )
    response = client.create(
        profile=LLMProfile(name="coder", model="gpt-5.4", reasoning_effort="high", verbosity="medium"),
        input_items=[user_text("Reply with JSON.")],
        tools=[],
        instructions="Reply with ok.",
    )

    assert client.last_model_used == "deepseek-v4-pro"
    assert response["provider"] == "deepseek_openai_compatible"
    assert response["model"] == "deepseek-v4-pro"
