"""调用 OpenAI Chat Completions 兼容服务的模型后端。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import httpx

from se_smoothllm.backends.base import Backend
from se_smoothllm.models import Generation

JBB_VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

_RESERVED_BODY_FIELDS = frozenset({"model", "messages", "temperature", "max_tokens"})
_RETRYABLE_STATUS_CODES = frozenset({408, 429})


class BackendRequestError(RuntimeError):
    """后端请求在有限次数尝试后仍未成功。"""


@dataclass(slots=True)
class OpenAICompatibleBackend(Backend):
    """调用本地或远程 OpenAI 兼容的 ``/chat/completions`` 接口。"""

    base_url: str
    model: str
    api_key: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 150
    timeout: float = 60.0
    max_retries: int = 2
    retry_backoff: float = 0.25
    extra_body: dict[str, Any] = field(default_factory=dict)
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty_text("base_url", self.base_url)
        _require_non_empty_text("model", self.model)
        if self.system_prompt is not None:
            _require_non_empty_text("system_prompt", self.system_prompt)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be a number between 0 and 2")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if isinstance(self.retry_backoff, bool) or not isinstance(
            self.retry_backoff, (int, float)
        ):
            raise TypeError("retry_backoff must be a number")
        if self.retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        reserved = _RESERVED_BODY_FIELDS.intersection(self.extra_body)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"extra_body cannot override reserved fields: {names}")

    def generate(self, prompt: str) -> Generation:
        """生成一次回答，并汇总所有重试所消耗的墙钟时间。"""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }

        started_at = perf_counter()
        response = self._request(headers=headers, payload=payload)
        latency_ms = (perf_counter() - started_at) * 1_000
        data = _response_json(response)
        content = _response_content(data)
        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        reported_model = data.get("model")
        actual_model = (
            reported_model.strip()
            if isinstance(reported_model, str) and reported_model.strip()
            else self.model
        )
        return Generation(
            text=content,
            model=actual_model,
            prompt_tokens=_token_count(usage, "prompt_tokens"),
            completion_tokens=_token_count(usage, "completion_tokens"),
            latency_ms=latency_ms,
        )

    def _request(self, *, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        attempts = self.max_retries + 1
        with httpx.Client(transport=self.transport, timeout=self.timeout) as client:
            for attempt in range(attempts):
                try:
                    response = client.post(endpoint, headers=headers, json=payload)
                except httpx.TimeoutException as exc:
                    if attempt < self.max_retries:
                        self._wait_before_retry(attempt)
                        continue
                    raise BackendRequestError(
                        f"backend request timed out after {attempts} attempts: {exc}"
                    ) from exc
                except httpx.TransportError as exc:
                    if attempt < self.max_retries:
                        self._wait_before_retry(attempt)
                        continue
                    raise BackendRequestError(
                        f"backend transport failed after {attempts} attempts: {exc}"
                    ) from exc

                if response.is_success:
                    return response
                if _is_retryable_status(response.status_code) and attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise _http_error(response, attempt + 1)

        raise AssertionError("backend retry loop ended without returning or raising")

    def _wait_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay > 0:
            time.sleep(delay)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("backend response is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("backend response JSON must be an object")
    return data


def _response_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("backend response does not contain chat completion content") from exc
    if not isinstance(content, str):
        raise ValueError("backend chat completion content must be a string")
    return content


def _token_count(usage: dict[str, Any], name: str) -> int | None:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def _http_error(response: httpx.Response, attempts: int) -> BackendRequestError:
    detail = " ".join(response.text.split())
    if len(detail) > 500:
        detail = f"{detail[:497]}..."
    if not detail:
        detail = "<empty response body>"
    return BackendRequestError(
        f"backend request failed with HTTP {response.status_code} "
        f"after {attempts} attempts: {detail}"
    )


def _require_non_empty_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
