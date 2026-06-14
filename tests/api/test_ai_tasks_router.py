"""API tests for the ai_tasks router using FastAPI dependency overrides."""

from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.domains.ai_tasks.api.dependencies import (
    get_ai_tasks_service,
    get_research_agent_service,
)
from app.domains.ai_tasks.api.schemas import (
    AgentIterationResult,
    AnalyzeTextResponse,
    AskResponse,
    ClassifyResponse,
    ExtractKeywordsResponse,
    ResearchAgentResponse,
    SummarizeResponse,
    ToolAssistantResponse,
    ToolExecutionResult,
    TranslateResponse,
)
from app.main import app
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def ai_tasks_service() -> MagicMock:
    """Fully mocked ``AITasksService`` injected via dependency override."""
    mock = MagicMock()

    mock.ask = AsyncMock(
        side_effect=lambda question, request_id: AskResponse(
            answer=f"echo:{question}", model="m", tokens_used=1
        )
    )

    async def _ask_stream(question, request_id):
        for token in ("Hel", "lo"):
            yield token

    mock.ask_stream = _ask_stream
    mock.classify = AsyncMock(
        return_value=ClassifyResponse(sentiment="positive", summary="s", keywords=["k"])
    )
    mock.summarize = AsyncMock(
        return_value=SummarizeResponse(summary="short", model="m", tokens_used=2)
    )
    mock.extract_keywords = AsyncMock(
        return_value=ExtractKeywordsResponse(
            keywords=["a", "b"], model="m", tokens_used=3
        )
    )
    mock.translate = AsyncMock(
        side_effect=lambda text, target_language, request_id: TranslateResponse(
            translation=f"{target_language}:{text}", model="m", tokens_used=4
        )
    )
    mock.analyze_text = AsyncMock(
        return_value=AnalyzeTextResponse(
            summary="s",
            sentiment="neutral",
            keywords=["k"],
            language="en",
            model="m",
            tokens_used=5,
            prompt_version="analyze_text_v1",
        )
    )
    mock.tool_assistant = AsyncMock(
        return_value=ToolAssistantResponse(
            answer="tool answer",
            tool_calls=[
                ToolExecutionResult(
                    name="get_weather",
                    arguments={"location": "Yerevan"},
                    result={"temperature": 21},
                )
            ],
            model="m",
            tokens_used=6,
            prompt_version="tool_assistant_v1",
        )
    )

    app.dependency_overrides[get_ai_tasks_service] = lambda: mock
    return mock


def _agent_response(topic: str) -> ResearchAgentResponse:
    tool_call = ToolExecutionResult(
        name="search_docs",
        arguments={"query": topic},
        result={"results": []},
    )
    return ResearchAgentResponse(
        topic=topic,
        report=f"report:{topic}",
        plan=["step one", "step two"],
        iterations=[
            AgentIterationResult(iteration=1, thought="", tool_calls=[tool_call]),
            AgentIterationResult(iteration=2, thought="done", tool_calls=[]),
        ],
        tool_calls=[tool_call],
        critique="No issues found.",
        stop_reason="completed",
        iterations_used=2,
        model="m",
        tokens_used=42,
        prompt_version="research_agent_v1",
    )


@pytest.fixture
def research_agent_service() -> MagicMock:
    """Mocked ``ResearchAgentService`` injected via dependency override."""
    mock = MagicMock()
    mock.run = AsyncMock(
        side_effect=lambda topic, request_id, max_iterations=None: _agent_response(
            topic
        )
    )
    app.dependency_overrides[get_research_agent_service] = lambda: mock
    return mock


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


class TestAsk:
    def test_ask(self, client, ai_tasks_service):
        resp = client.post("/ask", json={"question": "hi"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "echo:hi"
        ai_tasks_service.ask.assert_awaited_once()

    def test_ask_validation_error(self, client, ai_tasks_service):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422
        ai_tasks_service.ask.assert_not_awaited()

    def test_response_includes_request_id_header(self, client, ai_tasks_service):
        resp = client.post(
            "/ask", json={"question": "hi"}, headers={"x-request-id": "abc"}
        )
        assert resp.headers["x-request-id"] == "abc"


class TestStreaming:
    def test_ask_stream_concatenates_tokens(self, client, ai_tasks_service):
        resp = client.post("/ask-stream", json={"question": "hi"})
        assert resp.status_code == 200
        assert resp.text == "Hello"


class TestOtherTasks:
    def test_classify(self, client, ai_tasks_service):
        resp = client.post("/classify", json={"text": "great"})
        assert resp.status_code == 200
        assert resp.json()["sentiment"] == "positive"
        ai_tasks_service.classify.assert_awaited_once()

    def test_summarize(self, client, ai_tasks_service):
        resp = client.post("/summarize", json={"text": "long text"})
        assert resp.json()["summary"] == "short"

    def test_extract_keywords(self, client, ai_tasks_service):
        resp = client.post("/extract-keywords", json={"text": "x"})
        assert resp.json()["keywords"] == ["a", "b"]

    def test_translate(self, client, ai_tasks_service):
        resp = client.post(
            "/translate", json={"text": "hello", "target_language": "Spanish"}
        )
        assert resp.json()["translation"] == "Spanish:hello"

    def test_analyze_text(self, client, ai_tasks_service):
        resp = client.post("/analyze-text", json={"text": "x"})
        body = resp.json()
        assert body["language"] == "en"
        assert body["prompt_version"] == "analyze_text_v1"

    def test_tool_assistant(self, client, ai_tasks_service):
        resp = client.post("/tool-assistant", json={"message": "weather in Yerevan"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "tool answer"
        assert body["tool_calls"][0]["name"] == "get_weather"
        ai_tasks_service.tool_assistant.assert_awaited_once()


class TestResearchAgent:
    def test_research_agent_returns_report_and_trace(
        self, client, research_agent_service
    ):
        resp = client.post("/research-agent", json={"topic": "retry logic"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["report"] == "report:retry logic"
        assert body["plan"] == ["step one", "step two"]
        assert body["stop_reason"] == "completed"
        assert body["iterations"][0]["tool_calls"][0]["name"] == "search_docs"
        research_agent_service.run.assert_awaited_once()

    def test_research_agent_validation_error_on_empty_topic(
        self, client, research_agent_service
    ):
        resp = client.post("/research-agent", json={"topic": ""})
        assert resp.status_code == 422
        research_agent_service.run.assert_not_awaited()

    def test_research_agent_passes_max_iterations(self, client, research_agent_service):
        resp = client.post("/research-agent", json={"topic": "x", "max_iterations": 3})
        assert resp.status_code == 200
        assert research_agent_service.run.await_args.kwargs["max_iterations"] == 3


class TestErrorMapping:
    @pytest.mark.parametrize(
        "error,status_code",
        [
            (LLMTimeoutError("t"), 504),
            (LLMRateLimitError("r"), 429),
            (LLMTemporaryError("temp"), 503),
            (LLMServiceError("svc"), 502),
        ],
    )
    def test_llm_errors_map_to_status(
        self, client, ai_tasks_service, error, status_code
    ):
        ai_tasks_service.ask.side_effect = error
        resp = client.post("/ask", json={"question": "hi"})
        assert resp.status_code == status_code
        assert "detail" in resp.json()
