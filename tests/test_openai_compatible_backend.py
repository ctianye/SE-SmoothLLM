import json
from collections.abc import Callable

import httpx
import pytest

from se_smoothllm.backends.openai_compatible import (
    JBB_VICUNA_SYSTEM_PROMPT,
    BackendRequestError,
    OpenAICompatibleBackend,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _backend(handler: Handler, **overrides: object) -> OpenAICompatibleBackend:
    arguments: dict[str, object] = {
        "base_url": "http://backend.test/v1/",
        "model": "requested-model",
        "transport": httpx.MockTransport(handler),
        "retry_backoff": 0.0,
    }
    arguments.update(overrides)
    return OpenAICompatibleBackend(**arguments)  # type: ignore[arg-type]


def _success_response(
    request: httpx.Request,
    *,
    model: str | None = "served-model",
    usage: object = None,
) -> httpx.Response:
    payload: dict[str, object] = {
        "choices": [{"message": {"content": "model response"}}],
    }
    if model is not None:
        payload["model"] = model
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload, request=request)


def test_generate_sends_vicuna_prompt_defaults_and_records_metadata() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return _success_response(
            request,
            usage={"prompt_tokens": 17, "completion_tokens": 9},
        )

    backend = _backend(
        handler,
        api_key="secret-token",
        system_prompt=JBB_VICUNA_SYSTEM_PROMPT,
        extra_body={"top_p": 0.9},
    )

    generation = backend.generate("user prompt")

    assert captured["url"] == "http://backend.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-token"
    assert captured["payload"] == {
        "model": "requested-model",
        "messages": [
            {"role": "system", "content": JBB_VICUNA_SYSTEM_PROMPT},
            {"role": "user", "content": "user prompt"},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 0.9,
    }
    assert generation.text == "model response"
    assert generation.model == "served-model"
    assert generation.prompt_tokens == 17
    assert generation.completion_tokens == 9
    assert generation.latency_ms >= 0


def test_optional_system_prompt_and_missing_usage_are_supported() -> None:
    captured_payload: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.append(json.loads(request.content))
        return _success_response(request, model=None)

    generation = _backend(handler).generate("prompt without system message")

    assert captured_payload[0]["messages"] == [
        {"role": "user", "content": "prompt without system message"}
    ]
    assert generation.model == "requested-model"
    assert generation.prompt_tokens is None
    assert generation.completion_tokens is None


def test_invalid_token_values_are_reported_as_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response(
            request,
            usage={"prompt_tokens": True, "completion_tokens": -1},
        )

    generation = _backend(handler).generate("prompt")

    assert generation.prompt_tokens is None
    assert generation.completion_tokens is None


def test_timeout_is_retried_a_finite_number_of_times() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ReadTimeout("simulated timeout", request=request)

    with pytest.raises(BackendRequestError, match="timed out after 3 attempts"):
        _backend(handler, max_retries=2).generate("prompt")

    assert len(requests) == 3


def test_transport_error_is_retried_a_finite_number_of_times() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("simulated connection failure", request=request)

    with pytest.raises(BackendRequestError, match="transport failed after 2 attempts"):
        _backend(handler, max_retries=1).generate("prompt")

    assert len(requests) == 2


def test_retryable_500_response_recovers_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(500, json={"error": "temporary"}, request=request)
        return _success_response(request)

    monkeypatch.setattr(
        "se_smoothllm.backends.openai_compatible.time.sleep",
        delays.append,
    )
    generation = _backend(handler, max_retries=2, retry_backoff=0.25).generate("prompt")

    assert generation.text == "model response"
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_persistent_500_error_includes_status_attempts_and_body() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": "server exploded"}, request=request)

    with pytest.raises(BackendRequestError) as captured:
        _backend(handler, max_retries=1).generate("prompt")

    message = str(captured.value)
    assert "HTTP 500" in message
    assert "after 2 attempts" in message
    assert "server exploded" in message
    assert attempts == 2


def test_non_retryable_400_error_fails_immediately() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="invalid model", request=request)

    with pytest.raises(BackendRequestError, match="HTTP 400.*invalid model"):
        _backend(handler, max_retries=2).generate("prompt")

    assert attempts == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
    ],
)
def test_missing_chat_completion_fields_are_rejected(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(ValueError, match="does not contain chat completion content"):
        _backend(handler).generate("prompt")


def test_non_string_content_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"choices": [{"message": {"content": None}}]}
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(ValueError, match="content must be a string"):
        _backend(handler).generate("prompt")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "not valid JSON"),
        ("[]", "JSON must be an object"),
    ],
)
def test_invalid_json_responses_are_rejected(content: str, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=content, request=request)

    with pytest.raises(ValueError, match=message):
        _backend(handler).generate("prompt")


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"base_url": ""}, ValueError, "base_url"),
        ({"model": ""}, ValueError, "model"),
        ({"system_prompt": " "}, ValueError, "system_prompt"),
        ({"temperature": True}, ValueError, "temperature"),
        ({"temperature": 3}, ValueError, "temperature"),
        ({"max_tokens": True}, TypeError, "max_tokens"),
        ({"max_tokens": 0}, ValueError, "max_tokens"),
        ({"timeout": True}, TypeError, "timeout"),
        ({"timeout": 0}, ValueError, "timeout"),
        ({"max_retries": True}, TypeError, "max_retries"),
        ({"max_retries": -1}, ValueError, "max_retries"),
        ({"retry_backoff": True}, TypeError, "retry_backoff"),
        ({"retry_backoff": -1}, ValueError, "retry_backoff"),
        ({"extra_body": {"temperature": 1}}, ValueError, "reserved fields"),
    ],
)
def test_invalid_backend_configuration_is_rejected(
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _success_response(request)

    with pytest.raises(error_type, match=message):
        _backend(handler, **overrides)
