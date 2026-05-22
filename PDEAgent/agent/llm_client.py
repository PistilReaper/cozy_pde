"""
LLM API 客户端，带合规日志记录

日志格式要求（每行一条合法JSON）：
- timestamp: ISO 8601 时间戳，含时区
- elapsed_seconds: 本次LLM调用耗时（秒）
- response 或 tool_calls: 至少存在其一
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from openai import APIStatusError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(name, default)
    return default


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _get_field(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(value)


class LLMClient:
    """OpenAI 官方 SDK 客户端，自动记录合规日志"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        log_path: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.log_path = log_path

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def _log(
        self,
        elapsed: float,
        response_text: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        error: Optional[str] = None,
    ):
        """写入一条合规日志记录"""
        if not self.log_path:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "model": self.model,
        }

        if error:
            record["error"] = error
        if response_text is not None:
            record["response"] = response_text
        if tool_calls is not None:
            record["tool_calls"] = tool_calls

        if "response" not in record and "tool_calls" not in record:
            record["response"] = ""

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _error_message(self, exc: Exception) -> str:
        message = str(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            body = getattr(response, "text", None)
            if body:
                return str(body)
        return message

    def _send_request(self, payload: Dict[str, Any], max_retries: int = 5):
        """
        发送请求，带自动重试和错误恢复
        - 温度不兼容：自动调整为 1 并重试
        - 速率限制：指数退避
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**payload)
            except BadRequestError as e:
                last_exception = e
                err_msg = self._error_message(e)
                if "temperature" in err_msg.lower() and "only 1 is allowed" in err_msg.lower():
                    print("[LLMClient] Model requires temperature=1, auto-adjusting...")
                    payload["temperature"] = 1
                    self.temperature = 1
                    continue
                raise
            except RateLimitError as e:
                last_exception = e
                wait = min(2 ** attempt, 30)
                print(f"[LLMClient] Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                last_exception = e
                wait = min(2 ** attempt, 30)
                status_code = getattr(e, "status_code", "unknown")
                print(f"[LLMClient] HTTP error on attempt {attempt+1}/{max_retries}: {status_code}, retrying in {wait}s...")
                time.sleep(wait)
            except APITimeoutError as e:
                last_exception = e
                wait = min(2 ** attempt, 30)
                print(f"[LLMClient] Request timeout on attempt {attempt+1}/{max_retries}: {e}, retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                last_exception = e
                wait = min(2 ** attempt, 30)
                print(f"[LLMClient] Request error on attempt {attempt+1}/{max_retries}: {e}, retrying in {wait}s...")
                time.sleep(wait)

        raise last_exception

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        发送聊天请求，返回解析后的响应字典

        返回格式模拟 OpenAI ChatCompletion:
        {
            "content": "...",
            "tool_calls": [{"name": "...", "arguments": {...}}],
            "finish_reason": "...",
        }
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        start = time.time()
        try:
            data = self._send_request(payload)
            elapsed = time.time() - start

            choices = _get_field(data, "choices", []) or []
            choice = choices[0] if choices else {}
            message = _get_field(choice, "message", {}) or {}
            content = _coerce_text(_get_field(message, "content", ""))
            reasoning = _coerce_text(_get_field(message, "reasoning_content", ""))
            if not content and reasoning:
                content = reasoning

            tool_calls_raw = _get_field(message, "tool_calls", []) or []
            finish_reason = _get_field(choice, "finish_reason", "") or ""

            parsed_tools = []
            for tc in tool_calls_raw:
                if _get_field(tc, "type", "") != "function":
                    continue
                func = _get_field(tc, "function", {}) or {}
                arguments = _get_field(func, "arguments", {})
                parsed_tools.append(
                    {
                        "id": _get_field(tc, "id", ""),
                        "name": _get_field(func, "name", ""),
                        "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    }
                )

            self._log(
                elapsed=elapsed,
                response_text=content if not parsed_tools else None,
                tool_calls=parsed_tools if parsed_tools else None,
            )

            return {
                "content": content,
                "tool_calls": parsed_tools,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            elapsed = time.time() - start
            self._log(elapsed=elapsed, error=str(e))
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """流式聊天，逐字返回内容（不记录详细日志，仅记录总耗时）"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        start = time.time()
        full_content = []
        try:
            stream = self.client.chat.completions.create(**payload)
            for chunk in stream:
                choices = _get_field(chunk, "choices", []) or []
                if not choices:
                    continue
                delta = _get_field(choices[0], "delta", {}) or {}
                token = _coerce_text(_get_field(delta, "content", ""))
                if token:
                    full_content.append(token)
                    yield token
        finally:
            elapsed = time.time() - start
            self._log(elapsed=elapsed, response_text="".join(full_content))

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
