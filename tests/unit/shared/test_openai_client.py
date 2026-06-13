from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from app.shared.config import Settings
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)
from app.shared.infrastructure import openai_client as oc_mod
from app.shared.infrastructure.openai_client import OpenAIClient, get_openai_client
from tests.conftest import (
    json_object_format,
    make_chat_completion,
    make_embedding_response,
    user_messages,
)

_REQ = httpx.Request("POST", "https://api.openai.com/v1/x")


def _timeout():
    return APITimeoutError(request=_REQ)


def _conn():
    return APIConnectionError(request=_REQ)


def _rate():
    return RateLimitError("rate", response=httpx.Response(429, request=_REQ), body=None)


def _ise():
    return InternalServerError(
        "boom", response=httpx.Response(500, request=_REQ), body=None
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make exponential backoff instant during tests."""
    monkeypatch.setattr(oc_mod.asyncio, "sleep", AsyncMock())


@pytest.fixture
def client(monkeypatch) -> OpenAIClient:
    settings = Settings(OPENAI_MAX_RETRIES=2, OPENAI_RETRY_BASE_DELAY_SECONDS=0.01)
    c = OpenAIClient(settings)
    # Replace the real AsyncOpenAI instance with a mock surface.
    c.client = MagicMock()
    c.client.chat = MagicMock()
    c.client.chat.completions = MagicMock()
    c.client.chat.completions.create = AsyncMock()
    c.client.embeddings = MagicMock()
    c.client.embeddings.create = AsyncMock()
    return c


_MESSAGES = user_messages("hi")


class TestChatCompletionSuccess:
    async def test_returns_completion(self, client):
        client.client.chat.completions.create.return_value = make_chat_completion()
        result = await client.create_chat_completion(
            endpoint="ask", request_id="r", messages=_MESSAGES
        )
        assert result.model == "gpt-4o-mini"

    async def test_passes_model_and_messages(self, client):
        client.client.chat.completions.create.return_value = make_chat_completion()
        await client.create_chat_completion(
            endpoint="ask", request_id="r", messages=_MESSAGES
        )
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == _MESSAGES
        assert "temperature" in kwargs
        assert "max_tokens" in kwargs

    async def test_response_format_included_only_when_set(self, client):
        client.client.chat.completions.create.return_value = make_chat_completion()
        await client.create_chat_completion(
            endpoint="ask", request_id="r", messages=_MESSAGES
        )
        assert (
            "response_format"
            not in client.client.chat.completions.create.call_args.kwargs
        )

        await client.create_chat_completion(
            endpoint="ask",
            request_id="r",
            messages=_MESSAGES,
            response_format=json_object_format(),
        )
        assert client.client.chat.completions.create.call_args.kwargs[
            "response_format"
        ] == {"type": "json_object"}

    async def test_custom_temperature_and_max_tokens(self, client):
        client.client.chat.completions.create.return_value = make_chat_completion()
        await client.create_chat_completion(
            endpoint="ask",
            request_id="r",
            messages=_MESSAGES,
            temperature=1.5,
            max_tokens=42,
        )
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 1.5
        assert kwargs["max_tokens"] == 42

    async def test_tools_are_included_when_set(self, client):
        client.client.chat.completions.create.return_value = make_chat_completion()
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        await client.create_chat_completion(
            endpoint="tool-assistant",
            request_id="r",
            messages=_MESSAGES,
            tools=tools,
            tool_choice="auto",
        )
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


class TestChatCompletionRetries:
    async def test_retries_then_succeeds(self, client):
        client.client.chat.completions.create.side_effect = [
            _timeout(),
            make_chat_completion(content="ok"),
        ]
        result = await client.create_chat_completion(
            endpoint="ask", request_id="r", messages=_MESSAGES
        )
        assert result.choices[0].message.content == "ok"
        assert client.client.chat.completions.create.call_count == 2

    async def test_timeout_exhausted_raises_llm_timeout(self, client):
        client.client.chat.completions.create.side_effect = _timeout()
        with pytest.raises(LLMTimeoutError):
            await client.create_chat_completion(
                endpoint="ask", request_id="r", messages=_MESSAGES
            )
        # initial try + 2 retries
        assert client.client.chat.completions.create.call_count == 3

    async def test_rate_limit_exhausted_raises(self, client):
        client.client.chat.completions.create.side_effect = _rate()
        with pytest.raises(LLMRateLimitError):
            await client.create_chat_completion(
                endpoint="ask", request_id="r", messages=_MESSAGES
            )

    async def test_connection_error_maps_to_temporary(self, client):
        client.client.chat.completions.create.side_effect = _conn()
        with pytest.raises(LLMTemporaryError):
            await client.create_chat_completion(
                endpoint="ask", request_id="r", messages=_MESSAGES
            )

    async def test_internal_server_error_maps_to_temporary(self, client):
        client.client.chat.completions.create.side_effect = _ise()
        with pytest.raises(LLMTemporaryError):
            await client.create_chat_completion(
                endpoint="ask", request_id="r", messages=_MESSAGES
            )

    async def test_generic_openai_error_is_not_retried(self, client):
        client.client.chat.completions.create.side_effect = OpenAIError("weird")
        with pytest.raises(LLMServiceError):
            await client.create_chat_completion(
                endpoint="ask", request_id="r", messages=_MESSAGES
            )
        assert client.client.chat.completions.create.call_count == 1


class TestEmbeddings:
    async def test_returns_vector(self, client):
        client.client.embeddings.create.return_value = make_embedding_response(
            [0.1, 0.2]
        )
        vector = await client.create_embedding(
            endpoint="embeddings", request_id="r", text="hello"
        )
        assert vector == [0.1, 0.2]

    async def test_uses_default_embedding_model(self, client):
        client.client.embeddings.create.return_value = make_embedding_response([0.0])
        await client.create_embedding(endpoint="embeddings", request_id="r", text="x")
        assert (
            client.client.embeddings.create.call_args.kwargs["model"]
            == "text-embedding-3-small"
        )

    async def test_custom_model_override(self, client):
        client.client.embeddings.create.return_value = make_embedding_response([0.0])
        await client.create_embedding(
            endpoint="embeddings", request_id="r", text="x", model="custom-model"
        )
        assert (
            client.client.embeddings.create.call_args.kwargs["model"] == "custom-model"
        )

    async def test_retries_then_succeeds(self, client):
        client.client.embeddings.create.side_effect = [
            _conn(),
            make_embedding_response([1.0]),
        ]
        vector = await client.create_embedding(
            endpoint="embeddings", request_id="r", text="x"
        )
        assert vector == [1.0]
        assert client.client.embeddings.create.call_count == 2

    async def test_timeout_exhausted_raises(self, client):
        client.client.embeddings.create.side_effect = _timeout()
        with pytest.raises(LLMTimeoutError):
            await client.create_embedding(
                endpoint="embeddings", request_id="r", text="x"
            )

    async def test_rate_limit_maps(self, client):
        client.client.embeddings.create.side_effect = _rate()
        with pytest.raises(LLMRateLimitError):
            await client.create_embedding(
                endpoint="embeddings", request_id="r", text="x"
            )

    async def test_generic_error_maps_to_service_error(self, client):
        client.client.embeddings.create.side_effect = OpenAIError("weird")
        with pytest.raises(LLMServiceError):
            await client.create_embedding(
                endpoint="embeddings", request_id="r", text="x"
            )

    async def test_none_vector_raises_service_error(self, client):
        client.client.embeddings.create.return_value = make_embedding_response(None)
        with pytest.raises(LLMServiceError, match="without vector data"):
            await client.create_embedding(
                endpoint="embeddings", request_id="r", text="x"
            )


class TestStream:
    async def test_returns_stream_object(self, client):
        sentinel = SimpleNamespace(name="stream")
        client.client.chat.completions.create.return_value = sentinel
        result = await client.create_chat_completion_stream(
            endpoint="ask-stream", request_id="r", messages=_MESSAGES
        )
        assert result is sentinel
        assert client.client.chat.completions.create.call_args.kwargs["stream"] is True

    async def test_timeout_exhausted_raises(self, client):
        client.client.chat.completions.create.side_effect = _timeout()
        with pytest.raises(LLMTimeoutError):
            await client.create_chat_completion_stream(
                endpoint="ask-stream", request_id="r", messages=_MESSAGES
            )

    async def test_retries_then_succeeds(self, client):
        sentinel = SimpleNamespace(name="stream")
        client.client.chat.completions.create.side_effect = [_conn(), sentinel]
        result = await client.create_chat_completion_stream(
            endpoint="ask-stream", request_id="r", messages=_MESSAGES
        )
        assert result is sentinel
        assert client.client.chat.completions.create.call_count == 2

    async def test_rate_limit_maps(self, client):
        client.client.chat.completions.create.side_effect = _rate()
        with pytest.raises(LLMRateLimitError):
            await client.create_chat_completion_stream(
                endpoint="ask-stream", request_id="r", messages=_MESSAGES
            )

    async def test_connection_error_maps_to_temporary(self, client):
        client.client.chat.completions.create.side_effect = _conn()
        with pytest.raises(LLMTemporaryError):
            await client.create_chat_completion_stream(
                endpoint="ask-stream", request_id="r", messages=_MESSAGES
            )

    async def test_generic_error_maps_to_service_error(self, client):
        client.client.chat.completions.create.side_effect = OpenAIError("weird")
        with pytest.raises(LLMServiceError):
            await client.create_chat_completion_stream(
                endpoint="ask-stream", request_id="r", messages=_MESSAGES
            )


class TestResolvers:
    def test_resolve_temperature_default_and_override(self):
        c = OpenAIClient(Settings(OPENAI_TEMPERATURE=0.3))
        assert c._resolve_temperature(None) == 0.3
        assert c._resolve_temperature(0.9) == 0.9

    def test_resolve_max_tokens_default_and_override(self):
        c = OpenAIClient(Settings(OPENAI_MAX_TOKENS=300))
        assert c._resolve_max_tokens(None) == 300
        assert c._resolve_max_tokens(10) == 10


def test_get_openai_client_is_cached():
    get_openai_client.cache_clear()
    first = get_openai_client()
    second = get_openai_client()
    assert first is second
    get_openai_client.cache_clear()
