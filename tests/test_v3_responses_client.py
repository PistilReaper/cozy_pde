from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from cozy_pde_v3.provider_capabilities import write_provider_capability_report
from cozy_pde_v3.responses_client import ResponsesClient, ResponsesTurn


class FakeSDKItem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSDKResponse:
    def __init__(self, *, response_id: str, model: str, output: list[object], usage: object):
        self.id = response_id
        self.model = model
        self.output = output
        self.usage = usage

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "model": self.model,
            "output": [
                item.model_dump() if hasattr(item, "model_dump") else dict(vars(item)) for item in self.output
            ],
            "usage": self.usage.model_dump() if hasattr(self.usage, "model_dump") else dict(vars(self.usage)),
        }


class FakeSDKDumpable(FakeSDKItem):
    def model_dump(self) -> dict[str, object]:
        return dict(vars(self))


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.base_url = kwargs["base_url"]
        self.responses = types.SimpleNamespace(create=self._create)
        self.calls: list[dict[str, object]] = []
        self._planned_responses: list[object] = []
        self._planned_errors: list[Exception] = []

    def queue_response(self, response: object) -> None:
        self._planned_responses.append(response)

    def queue_error(self, error: Exception) -> None:
        self._planned_errors.append(error)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._planned_errors:
            raise self._planned_errors.pop(0)
        if not self._planned_responses:
            raise AssertionError("no queued fake response")
        return self._planned_responses.pop(0)


class GatewayError(RuntimeError):
    status_code = 502


class QuotaError(RuntimeError):
    status_code = 429


def _install_fake_openai(monkeypatch, clients: list[FakeOpenAI]) -> None:
    class Factory:
        def __init__(self, **kwargs):
            client = FakeOpenAI(**kwargs)
            clients.append(client)
            self.responses = client.responses
            self.calls = client.calls
            self.base_url = client.base_url
            self.queue_response = client.queue_response
            self.queue_error = client.queue_error

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Factory, __version__="2.37.0"))


def test_dict_raw_response_normalizes_into_turn_and_preserves_usage(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.primary_only(
        provider="primary",
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://example.com",
    )
    clients[0].queue_response(
        {
            "id": "resp_dict",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        }
    )

    turn = client.create(
        input=[{"role": "user", "content": "Say hello"}],
        tools=[],
        metadata={"task_id": "task-2", "step_id": "step-1"},
    )

    assert isinstance(turn, ResponsesTurn)
    assert turn.provider == "primary"
    assert turn.raw_response["id"] == "resp_dict"
    assert turn.provider_output_items[0]["type"] == "message"
    assert turn.standard_output_items[0]["type"] == "message"
    assert turn.standard_output_items[0]["content"][0]["text"] == "hello"
    assert turn.usage == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert clients[0].calls[0]["metadata"] == {"task_id": "task-2", "step_id": "step-1"}


def test_sdk_object_raw_response_normalizes_to_dict_like_payload(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.primary_only(
        provider="primary",
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://example.com/",
    )
    clients[0].queue_response(
        FakeSDKResponse(
            response_id="resp_sdk",
            model="gpt-5.4",
            output=[
                FakeSDKDumpable(
                    type="message",
                    role="assistant",
                    content=[FakeSDKDumpable(type="output_text", text="sdk text")],
                )
            ],
            usage=FakeSDKDumpable(input_tokens=3, output_tokens=2, total_tokens=5),
        )
    )

    turn = client.create(input=[{"role": "user", "content": "Say hello"}], tools=[])

    assert turn.raw_response["id"] == "resp_sdk"
    assert isinstance(turn.raw_response, dict)
    assert turn.provider_output_items[0]["content"][0]["text"] == "sdk text"
    assert turn.standard_output_items[0]["content"][0]["text"] == "sdk text"
    assert turn.usage["total_tokens"] == 5


def test_gateway_failure_before_tool_boundary_fails_over_with_stable_reason(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.with_fallback(
        primary={
            "provider": "primary",
            "model": "gpt-5.4",
            "api_key": "primary-key",
            "base_url": "https://primary.example.com",
        },
        fallback={
            "provider": "fallback",
            "model": "deepseek-v4-pro",
            "api_key": "fallback-key",
            "base_url": "https://fallback.example.com",
            "append_v1": False,
        },
    )
    clients[0].queue_error(GatewayError("502 bad gateway"))
    clients[1].queue_response(
        {
            "id": "resp_fallback",
            "model": "deepseek-v4-pro",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "fallback ok"}],
                }
            ],
            "usage": {"total_tokens": 9},
        }
    )

    turn = client.create(input=[{"role": "user", "content": "Hello"}], tools=[])

    assert turn.provider == "fallback"
    assert turn.model == "deepseek-v4-pro"
    assert turn.failover_from == "primary"
    assert turn.failover_reason == "gateway_error"


def test_quota_failure_before_tool_boundary_fails_over_with_stable_reason(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.with_fallback(
        primary={
            "provider": "primary",
            "model": "gpt-5.4",
            "api_key": "primary-key",
            "base_url": "https://primary.example.com",
        },
        fallback={
            "provider": "fallback",
            "model": "deepseek-v4-pro",
            "api_key": "fallback-key",
            "base_url": "https://fallback.example.com",
            "append_v1": False,
        },
    )
    clients[0].queue_error(QuotaError("quota exceeded"))
    clients[1].queue_response(
        {
            "id": "resp_quota_fallback",
            "model": "deepseek-v4-pro",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "fallback ok"}],
                }
            ],
            "usage": {"total_tokens": 10},
        }
    )

    turn = client.create(input=[{"role": "user", "content": "Hello"}], tools=[])

    assert turn.provider == "fallback"
    assert turn.failover_from == "primary"
    assert turn.failover_reason == "quota_exhausted"


def test_primary_function_call_response_does_not_fail_over(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.with_fallback(
        primary={
            "provider": "primary",
            "model": "gpt-5.4",
            "api_key": "primary-key",
            "base_url": "https://primary.example.com",
        },
        fallback={
            "provider": "fallback",
            "model": "deepseek-v4-pro",
            "api_key": "fallback-key",
            "base_url": "https://fallback.example.com",
        },
    )
    clients[0].queue_response(
        {
            "id": "resp_tool",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "function_call",
                    "name": "echo_tool",
                    "call_id": "call_1",
                    "arguments": json.dumps({"text": "hello"}),
                }
            ],
            "usage": {"total_tokens": 4},
        }
    )

    turn = client.create(
        input=[{"role": "user", "content": "Call the tool"}],
        tools=[{"type": "function", "name": "echo_tool"}],
    )

    assert turn.provider == "primary"
    assert turn.failover_from is None
    assert len(clients[1].calls) == 0
    assert turn.standard_output_items[0]["type"] == "function_call"


def test_base_url_normalization_matches_primary_and_fallback(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    ResponsesClient.with_fallback(
        primary={
            "provider": "primary",
            "model": "gpt-5.4",
            "api_key": "primary-key",
            "base_url": "https://primary.example.com/",
        },
        fallback={
            "provider": "fallback",
            "model": "deepseek-v4-pro",
            "api_key": "fallback-key",
            "base_url": "https://fallback.example.com/v1/",
        },
    )

    assert clients[0].base_url == "https://primary.example.com/v1"
    assert clients[1].base_url == "https://fallback.example.com/v1"


def test_cache_hint_kwargs_are_omitted_when_provider_does_not_support_them(monkeypatch):
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)

    client = ResponsesClient.primary_only(
        provider="primary",
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://example.com",
        supports_prompt_cache_hints=False,
    )
    clients[0].queue_response(
        {
            "id": "resp_cache",
            "model": "gpt-5.4",
            "output": [],
            "usage": {"total_tokens": 1},
        }
    )

    client.create(
        input=[{"role": "user", "content": "Hello"}],
        tools=[],
        prompt_cache_key="cache-key",
        prompt_cache_retention="ephemeral",
    )

    create_kwargs = clients[0].calls[0]
    assert "prompt_cache_key" not in create_kwargs
    assert "prompt_cache_retention" not in create_kwargs


def test_provider_report_writes_hashes_freshness_and_provider_metadata(tmp_path: Path):
    report_path = tmp_path / "nested" / "reports" / "provider_report.json"

    report = write_provider_capability_report(
        report_path,
        config_payload={"base": "config"},
        tool_schemas=[{"name": "echo_tool", "parameters": {"type": "object"}}],
        proxy_payload={"version": "2026-05-21"},
        adapter_version="responses-v3",
        sdk_version="2.37.0",
        checked_at="2026-05-21T09:00:00Z",
        expires_at="2026-05-21T10:00:00Z",
        forced_failover={
            "supported": True,
            "selected_provider": "fallback",
            "probe_id": "forced-failover-1",
            "observed_model": "deepseek-v4-pro",
        },
        primary={
            "provider": "primary",
            "model_id": "gpt-5.4",
            "base_url": "https://primary.example.com/v1",
            "forced_failover": False,
            "formal_ready": True,
            "strict_schema_ok": True,
            "function_call_ok": True,
        },
        fallback={
            "provider": "fallback",
            "model_id": "deepseek-v4-pro",
            "base_url": "https://fallback.example.com/v1",
            "forced_failover": True,
            "formal_ready": False,
            "strict_schema_ok": False,
            "function_call_ok": True,
        },
    )

    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report
    assert report["config_hash"]
    assert report["tool_schema_hash"]
    assert report["proxy_version_hash"]
    assert report["adapter_version"] == "responses-v3"
    assert report["sdk_version"] == "2.37.0"
    assert report["checked_at"] == "2026-05-21T09:00:00Z"
    assert report["expires_at"] == "2026-05-21T10:00:00Z"
    assert report["forced_failover"] == {
        "supported": True,
        "selected_provider": "fallback",
        "probe_id": "forced-failover-1",
        "observed_model": "deepseek-v4-pro",
    }
    assert report["formal_ready"] is True
    assert report["primary"]["model_id"] == "gpt-5.4"
    assert report["primary"]["base_url_hash"]
    assert report["primary"]["strict_schema_ok"] is True
    assert report["primary"]["function_call_ok"] is True
    assert report["primary"]["primary"] is True
    assert report["primary"]["fallback"] is False
    assert report["primary"]["forced_failover"] is False
    assert report["primary"]["formal_ready"] is True
    assert report["fallback"]["model_id"] == "deepseek-v4-pro"
    assert report["fallback"]["strict_schema_ok"] is False
    assert report["fallback"]["function_call_ok"] is True
    assert report["fallback"]["primary"] is False
    assert report["fallback"]["fallback"] is True
    assert report["fallback"]["forced_failover"] is True
    assert report["fallback"]["formal_ready"] is False
