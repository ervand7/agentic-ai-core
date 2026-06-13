"""Unit tests for OpenAILLMProvider (OpenAIClient mocked)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domains.ai_tasks.infrastructure.openai_llm_provider import OpenAILLMProvider
from tests.conftest import (
    json_object_format,
    make_chat_completion,
    make_stream_chunk,
    user_messages,
)


class TestComplete:
    async def test_maps_completion_to_result(self):
        client = AsyncMock()
        client.create_chat_completion.return_value = make_chat_completion(
            content="hi", model="m", total_tokens=9
        )
        provider = OpenAILLMProvider(client)

        result = await provider.complete(
            endpoint="ask", request_id="r", messages=user_messages("x")
        )
        assert result.content == "hi"
        assert result.model == "m"
        assert result.tokens_used == 9

    async def test_forwards_all_kwargs(self):
        client = AsyncMock()
        client.create_chat_completion.return_value = make_chat_completion()
        provider = OpenAILLMProvider(client)

        await provider.complete(
            endpoint="classify",
            request_id="r",
            messages=user_messages("x"),
            response_format=json_object_format(),
            temperature=0.0,
            max_tokens=50,
        )
        kwargs = client.create_chat_completion.call_args.kwargs
        assert kwargs["endpoint"] == "classify"
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 50

    async def test_none_content_and_missing_usage(self):
        client = AsyncMock()
        client.create_chat_completion.return_value = make_chat_completion(
            content=None, total_tokens=None
        )
        provider = OpenAILLMProvider(client)
        result = await provider.complete(endpoint="ask", request_id="r", messages=[])
        assert result.content == ""
        assert result.tokens_used == 0


class TestCompleteWithTools:
    async def test_maps_tool_calls_to_result(self):
        client = AsyncMock()
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"location":"Yerevan"}',
            ),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        choice = SimpleNamespace(message=message)
        client.create_chat_completion.return_value = SimpleNamespace(
            choices=[choice],
            model="m",
            usage=SimpleNamespace(total_tokens=12),
        )
        provider = OpenAILLMProvider(client)

        result = await provider.complete_with_tools(
            endpoint="tool-assistant",
            request_id="r",
            messages=user_messages("weather"),
            tools=[{"type": "function"}],
        )

        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == '{"location":"Yerevan"}'
        assert result.tokens_used == 12
        kwargs = client.create_chat_completion.call_args.kwargs
        assert kwargs["tools"] == [{"type": "function"}]
        assert kwargs["tool_choice"] == "auto"


class TestStream:
    @staticmethod
    def _client_with_stream(chunks):
        client = AsyncMock()
        client.settings = SimpleNamespace(OPENAI_MODEL="gpt-4o-mini")

        async def _gen():
            for c in chunks:
                yield c

        client.create_chat_completion_stream = AsyncMock(return_value=_gen())
        return client

    async def test_yields_content_deltas_only(self):
        chunks = [
            make_stream_chunk(content="Hel", model="m"),
            make_stream_chunk(content=None, model="m"),
            make_stream_chunk(content="lo", model="m", total_tokens=4),
        ]
        provider = OpenAILLMProvider(self._client_with_stream(chunks))

        out = [
            chunk
            async for chunk in provider.stream(
                endpoint="ask-stream", request_id="r", messages=[]
            )
        ]
        assert [c.content for c in out] == ["Hel", "lo"]

    async def test_tracks_model_and_tokens(self):
        chunks = [
            make_stream_chunk(content="a", model="updated-model"),
            make_stream_chunk(content="b", model="updated-model", total_tokens=12),
        ]
        provider = OpenAILLMProvider(self._client_with_stream(chunks))

        out = [
            chunk
            async for chunk in provider.stream(
                endpoint="ask-stream", request_id="r", messages=[]
            )
        ]
        assert out[-1].model == "updated-model"
        assert out[-1].tokens_used == 12

    async def test_empty_stream_yields_nothing(self):
        provider = OpenAILLMProvider(self._client_with_stream([]))
        out = [
            c
            async for c in provider.stream(
                endpoint="ask-stream", request_id="r", messages=[]
            )
        ]
        assert out == []
