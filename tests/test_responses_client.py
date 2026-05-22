from __future__ import annotations

import json
import sys
import types

from cozy_pde_v3.responses_client import ResponsesClient, ResponsesTurn


class FakeSDKItem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self) -> dict[str, object]:
        return dict(vars(self))


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
            "output": [item.model_dump() for item in self.output],
            "usage": self.usage.model_dump(),
        }


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.base_url = kwargs["base_url"]
        self.calls: list[dict[str, object]] = []
        self._responses: list[object] = []
        self._errors: list[Exception] = []
        self.responses = types.SimpleNamespace(create=self._create)

    def queue_response(self, response: object) -> None:
        self._responses.append(response)

    def queue_error(self, error: Exception) -> None:
        self._errors.append(error)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return self._responses.pop(0)


class GatewayError(RuntimeError):
    status_code = 502


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

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Factory, __version__="2.0.0"))


def test_responses_client_returns_responses_turn_with_normalized_sdk_payload(monkeypatch) -> None:
    clients: list[FakeOpenAI] = []
    _install_fake_openai(monkeypatch, clients)
    client = ResponsesClient.primary_only(
        provider="primary",
        model="gpt-5.4",
        api_key="test-key",
        base_url="https://example.com",
    )
    clients[0].queue_response(
        FakeSDKResponse(
            response_id="resp_1",
            model="gpt-5.4",
            output=[
                FakeSDKItem(
                    type="message",
                    role="assistant",
                    content=[FakeSDKItem(type="output_text", text="hello")],
                )
            ],
            usage=FakeSDKItem(input_tokens=3, output_tokens=2, total_tokens=5),
        )
    )

    turn = client.create(input=[{"role": "user", "content": "hello"}], tools=[])

    assert isinstance(turn, ResponsesTurn)
    assert turn.provider == "primary"
    assert turn.raw_response["id"] == "resp_1"
    assert turn.standard_output_items[0]["content"][0]["text"] == "hello"
    assert turn.usage["total_tokens"] == 5
    assert clients[0].base_url == "https://example.com/v1"


def test_gateway_failure_fails_over_before_tool_boundary(monkeypatch) -> None:
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
    clients[0].queue_error(GatewayError("bad gateway"))
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

    turn = client.create(input=[{"role": "user", "content": "hello"}], tools=[])

    assert turn.provider == "fallback"
    assert turn.failover_from == "primary"
    assert turn.failover_reason == "gateway_error"
    assert turn.standard_output_items[0]["content"][0]["text"] == "fallback ok"


def test_function_call_output_stays_on_primary_and_normalizes_arguments(monkeypatch) -> None:
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
                    "name": "write_file",
                    "call_id": "call-1",
                    "arguments": json.dumps({"path": "submission/code/app.py", "content": "print('ok')\n"}),
                }
            ],
            "usage": {"total_tokens": 4},
        }
    )

    turn = client.create(
        input=[{"role": "user", "content": "write a file"}],
        tools=[{"type": "function", "name": "write_file"}],
    )

    assert turn.provider == "primary"
    assert turn.failover_from is None
    assert len(clients[1].calls) == 0
    assert turn.standard_output_items[0]["type"] == "function_call"
    assert json.loads(turn.standard_output_items[0]["arguments"])["path"] == "submission/code/app.py"
