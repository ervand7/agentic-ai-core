"""Unit tests for AITasksService with a mocked LLMProvider port."""

import json
from unittest.mock import AsyncMock

import pytest

from app.domains.ai_tasks.application.services import AITasksService
from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.ports import (
    CompletionResult,
    LLMProvider,
    StreamChunk,
    ToolCall,
    ToolCompletionResult,
)
from app.shared.config import Settings
from app.shared.exceptions import LLMServiceError
from tests.conftest import async_mock_method, make_llm_provider_mock


def _complete(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete")


def _complete_with_tools(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete_with_tools")


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def provider() -> LLMProvider:
    return make_llm_provider_mock()


@pytest.fixture
def service(provider: LLMProvider, settings: Settings) -> AITasksService:
    return AITasksService(provider, settings)


class TestAsk:
    async def test_returns_answer_with_model_and_tokens(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content="the answer", model="gpt-4o-mini", tokens_used=11
        )
        resp = await service.ask("question?", "req-1")
        assert resp.answer == "the answer"
        assert resp.model == "gpt-4o-mini"
        assert resp.tokens_used == 11

    async def test_uses_ask_endpoint_and_two_messages(self, service, provider):
        await service.ask("hi", "req-1")
        complete = _complete(provider)
        complete.assert_awaited_once()
        call_kwargs = complete.await_args.kwargs
        assert call_kwargs["endpoint"] == Endpoint.ASK
        assert call_kwargs["request_id"] == "req-1"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["content"] == "hi"


class TestAskStream:
    async def test_yields_only_non_empty_content(self, service, provider):
        async def _stream(**_kwargs):
            yield StreamChunk(content="Hel", model="m", tokens_used=None)
            yield StreamChunk(content="", model="m", tokens_used=None)
            yield StreamChunk(content="lo", model="m", tokens_used=5)

        provider.stream = _stream
        tokens = [tok async for tok in service.ask_stream("hi", "req-1")]
        assert tokens == ["Hel", "lo"]

    async def test_stream_uses_stream_endpoint(self, service, provider):
        captured: list[dict] = []

        async def _tracking_stream(**kwargs):
            captured.append(kwargs)
            yield StreamChunk(content="x", model="m", tokens_used=None)

        provider.stream = _tracking_stream
        _ = [t async for t in service.ask_stream("hi", "req-2")]
        assert captured[0]["endpoint"] == Endpoint.ASK_STREAM


class TestClassify:
    async def test_parses_structured_json(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content=json.dumps(
                {"sentiment": "positive", "summary": "good", "keywords": ["a", "b"]}
            ),
            model="gpt-4o-mini",
            tokens_used=11,
        )
        resp = await service.classify("I love it", "req-1")
        assert resp.sentiment == "positive"
        assert resp.summary == "good"
        assert resp.keywords == ["a", "b"]

    async def test_uses_classify_temperature_and_response_format(
        self, service, provider, settings
    ):
        _complete(provider).return_value = CompletionResult(
            content=json.dumps(
                {"sentiment": "neutral", "summary": "s", "keywords": []}
            ),
            model="gpt-4o-mini",
            tokens_used=11,
        )
        await service.classify("text", "req-1")
        call_kwargs = _complete(provider).await_args.kwargs
        assert call_kwargs["temperature"] == settings.OPENAI_TEMPERATURE_CLASSIFY
        assert call_kwargs["response_format"] is not None

    async def test_invalid_json_raises_llm_service_error(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content="not json {", model="gpt-4o-mini", tokens_used=11
        )
        with pytest.raises(LLMServiceError, match="invalid JSON"):
            await service.classify("text", "req-1")


class TestSummarize:
    async def test_returns_summary(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content="a short summary", model="gpt-4o-mini", tokens_used=11
        )
        resp = await service.summarize("long text", "req-1")
        assert resp.summary == "a short summary"
        assert resp.model == "gpt-4o-mini"


class TestExtractKeywords:
    async def test_returns_keywords_from_json(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content=json.dumps({"keywords": ["x", "y", "z"]}),
            model="gpt-4o-mini",
            tokens_used=11,
        )
        resp = await service.extract_keywords("text", "req-1")
        assert resp.keywords == ["x", "y", "z"]

    async def test_missing_keywords_key_defaults_to_empty_list(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content=json.dumps({"other": 1}), model="gpt-4o-mini", tokens_used=11
        )
        resp = await service.extract_keywords("text", "req-1")
        assert resp.keywords == []


class TestTranslate:
    async def test_returns_translation(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content="hola", model="gpt-4o-mini", tokens_used=11
        )
        resp = await service.translate("hello", "Spanish", "req-1")
        assert resp.translation == "hola"

    async def test_target_language_injected_into_system_prompt(self, service, provider):
        await service.translate("hello", "French", "req-1")
        system_msg = _complete(provider).await_args.kwargs["messages"][0]["content"]
        assert "French" in system_msg


class TestAnalyzeText:
    async def test_parses_combined_response_with_prompt_version(
        self, service, provider
    ):
        _complete(provider).return_value = CompletionResult(
            content=json.dumps(
                {
                    "summary": "s",
                    "sentiment": "negative",
                    "keywords": ["k"],
                    "language": "en",
                }
            ),
            model="gpt-4o-mini",
            tokens_used=11,
        )
        resp = await service.analyze_text("text", "req-1")
        assert resp.summary == "s"
        assert resp.sentiment == "negative"
        assert resp.keywords == ["k"]
        assert resp.language == "en"
        assert resp.prompt_version == "analyze_text_v1"
        assert resp.model == "gpt-4o-mini"

    async def test_invalid_json_raises(self, service, provider):
        _complete(provider).return_value = CompletionResult(
            content="}{", model="gpt-4o-mini", tokens_used=11
        )
        with pytest.raises(LLMServiceError):
            await service.analyze_text("text", "req-1")


class TestToolAssistant:
    async def test_returns_direct_answer_when_no_tool_called(self, service, provider):
        _complete_with_tools(provider).return_value = ToolCompletionResult(
            content="direct answer",
            model="gpt-4o-mini",
            tokens_used=9,
            tool_calls=[],
        )
        resp = await service.tool_assistant("hello", "req-1")
        assert resp.answer == "direct answer"
        assert resp.tool_calls == []
        assert resp.prompt_version == "tool_assistant_v1"

    async def test_executes_tool_and_sends_result_back_to_model(
        self, service, provider
    ):
        _complete_with_tools(provider).return_value = ToolCompletionResult(
            content="",
            model="gpt-4o-mini",
            tokens_used=10,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments=json.dumps({"location": "Yerevan"}),
                )
            ],
        )
        _complete(provider).return_value = CompletionResult(
            content="Weather summary", model="gpt-4o-mini", tokens_used=7
        )

        resp = await service.tool_assistant("weather in Yerevan", "req-1")

        assert resp.answer == "Weather summary"
        assert resp.tokens_used == 17
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].risk == "read_only"
        assert resp.tool_calls[0].status == "executed"
        assert resp.tool_calls[0].result["location"] == "Yerevan"
        final_messages = _complete(provider).await_args.kwargs["messages"]
        assert final_messages[-1]["role"] == "tool"

    async def test_create_ticket_is_only_a_draft(self, service, provider):
        _complete_with_tools(provider).return_value = ToolCompletionResult(
            content="",
            model="gpt-4o-mini",
            tokens_used=5,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="create_ticket",
                    arguments=json.dumps(
                        {
                            "title": "Improve docs",
                            "description": "Add tool-calling examples.",
                            "priority": "high",
                        }
                    ),
                )
            ],
        )
        _complete(provider).return_value = CompletionResult(
            content="Draft created.", model="gpt-4o-mini", tokens_used=5
        )

        resp = await service.tool_assistant("create ticket", "req-1")

        result = resp.tool_calls[0].result
        assert resp.tool_calls[0].risk == "draft"
        assert result["status"] == "draft_created"
        assert result["requires_human_confirmation"] is True
