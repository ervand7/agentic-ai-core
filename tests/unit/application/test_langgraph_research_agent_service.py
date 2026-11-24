"""Unit tests for the LangGraph research agent: workers, critic, approvals, retries."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.domains.ai_tasks.application.langgraph_research_agent_service import (
    LangGraphResearchAgentService,
    ResearchRunNotFoundError,
)
from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.ports import (
    CompletionResult,
    LLMProvider,
    ToolCall,
    ToolCompletionResult,
)
from app.shared.config import Settings
from app.shared.exceptions import LLMServiceError, LLMTimeoutError
from tests.conftest import async_mock_method, make_llm_provider_mock


def _complete(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete")


def _complete_with_tools(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete_with_tools")


def _plan_result(steps: list[str], tokens: int = 10) -> CompletionResult:
    return CompletionResult(
        content=json.dumps({"steps": steps}), model="m", tokens_used=tokens
    )


def _report(content: str, tokens: int = 3) -> CompletionResult:
    return CompletionResult(content=content, model="m", tokens_used=tokens)


def _critique_result(
    *, approved: bool, issues: list[str], revised: str = ""
) -> CompletionResult:
    return CompletionResult(
        content=json.dumps(
            {"approved": approved, "issues": issues, "revised_answer": revised}
        ),
        model="m",
        tokens_used=4,
    )


def _final(content: str, tokens: int = 3) -> ToolCompletionResult:
    return ToolCompletionResult(
        content=content, model="m", tokens_used=tokens, tool_calls=[]
    )


def _tool_step(
    name: str, arguments: dict, call_id: str = "call_1"
) -> ToolCompletionResult:
    return ToolCompletionResult(
        content="",
        model="m",
        tokens_used=3,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
    )


def _ticket_step() -> ToolCompletionResult:
    return _tool_step(
        "create_ticket", {"title": "Improve retries", "description": "Add jitter"}
    )


def _writer_brief(provider: LLMProvider) -> str:
    for call in _complete(provider).await_args_list:
        if call.kwargs["endpoint"] == Endpoint.RESEARCH_AGENT_WRITER:
            return call.kwargs["messages"][-1]["content"]
    raise AssertionError("writer was not called")


@pytest.fixture
def provider() -> LLMProvider:
    return make_llm_provider_mock()


def _service(
    provider: LLMProvider, settings: Settings, **kwargs
) -> LangGraphResearchAgentService:
    return LangGraphResearchAgentService(
        llm_provider=provider, settings=settings, **kwargs
    )


class TestCompletion:
    async def test_writer_turns_findings_into_report(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [
            _plan_result(["a", "b"]),
            _report("Final report"),
        ]
        _complete_with_tools(provider).return_value = _final("Research findings")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.status == "completed"
        assert resp.thread_id
        assert resp.report == "Final report"
        assert resp.stop_reason == "completed"
        assert resp.plan == ["a", "b"]
        assert resp.iterations_used == 1
        assert resp.tokens_used == 16
        assert resp.pending_tool_calls == []
        assert "Research findings" in _writer_brief(provider)

    async def test_read_only_tool_runs_without_pausing(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Report with evidence"),
        ]
        _complete_with_tools(provider).side_effect = [
            _tool_step("search_docs", {"query": "retry logic"}),
            _final("Findings"),
        ]

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.status == "completed"
        assert resp.report == "Report with evidence"
        assert resp.tool_calls[0].name == "search_docs"
        assert resp.tool_calls[0].status == "executed"
        assert len(resp.iterations) == 2
        assert "retry logic" in _writer_brief(provider)

    async def test_budget_stop_hands_off_to_writer(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Forced report", tokens=5),
        ]
        _complete_with_tools(provider).return_value = _tool_step(
            "search_docs", {"query": "x"}
        )

        resp = await _service(provider, settings).run(
            "topic", "req-1", max_iterations=2
        )

        assert resp.stop_reason == "max_iterations"
        assert resp.iterations_used == 2
        assert resp.report == "Forced report"
        assert "stopped early (max_iterations)" in _writer_brief(provider)


class TestSupervisor:
    async def test_hands_off_planner_researcher_writer_critic(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Report"),
            _critique_result(approved=True, issues=[]),
        ]
        _complete_with_tools(provider).return_value = _final("Findings")

        resp = await _service(provider, settings).run("topic", "req-1")

        endpoints = [
            call.kwargs["endpoint"] for call in _complete(provider).await_args_list
        ]
        assert endpoints == [
            Endpoint.RESEARCH_AGENT_PLAN,
            Endpoint.RESEARCH_AGENT_WRITER,
            Endpoint.RESEARCH_AGENT_CRITIC,
        ]
        assert resp.report == "Report"
        assert resp.critique == "No issues found."


class TestCriticRevisions:
    async def test_rejected_report_goes_back_for_another_round(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True, AGENT_MAX_REVISIONS=1)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Draft report"),
            _critique_result(approved=False, issues=["missing citations"]),
            _report("Improved report"),
            _critique_result(approved=True, issues=[]),
        ]
        _complete_with_tools(provider).side_effect = [
            _final("Draft findings"),
            _final("Improved findings"),
        ]

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Improved report"
        assert resp.revisions == 1
        assert resp.critique == "No issues found."
        second_round_messages = (
            _complete_with_tools(provider).await_args_list[1].kwargs["messages"]
        )
        assert second_round_messages[-2] == {
            "role": "assistant",
            "content": "Draft findings",
        }
        assert "missing citations" in second_round_messages[-1]["content"]

    async def test_revision_cap_falls_back_to_critic_revised_answer(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True, AGENT_MAX_REVISIONS=0)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Draft report"),
            _critique_result(approved=False, issues=["too short"], revised="Better"),
        ]
        _complete_with_tools(provider).return_value = _final("Findings")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Better"
        assert resp.revisions == 0
        assert _complete_with_tools(provider).await_count == 1


class TestRetries:
    async def test_transient_llm_error_retries_the_node(self, provider):
        settings = Settings(
            AGENT_ENABLE_REFLECTION=False,
            AGENT_NODE_MAX_ATTEMPTS=2,
            OPENAI_RETRY_BASE_DELAY_SECONDS=0.01,
        )
        _complete(provider).side_effect = [
            LLMTimeoutError("timeout"),
            _plan_result(["a"]),
            _report("Report"),
        ]
        _complete_with_tools(provider).return_value = _final("Findings")

        with patch("langgraph.pregel._retry.random.uniform", return_value=0):
            resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Report"
        assert _complete(provider).await_count == 3

    async def test_non_transient_error_is_not_retried(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False, AGENT_NODE_MAX_ATTEMPTS=3)
        _complete(provider).side_effect = LLMServiceError("bad request")

        with pytest.raises(LLMServiceError):
            await _service(provider, settings).run("topic", "req-1")

        assert _complete(provider).await_count == 1


class TestHumanApproval:
    async def test_risky_tool_pauses_run_for_approval(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).side_effect = [_ticket_step()]

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.status == "awaiting_approval"
        assert resp.stop_reason == "awaiting_approval"
        assert resp.report == ""
        assert resp.pending_tool_calls[0].name == "create_ticket"
        assert resp.tool_calls == []

    async def test_approved_resume_executes_tool_and_completes(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        service = _service(provider, settings)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Report, ticket drafted"),
        ]
        _complete_with_tools(provider).side_effect = [
            _ticket_step(),
            _final("Findings"),
        ]

        paused = await service.run("topic", "req-1")
        resp = await service.resume(paused.thread_id, "req-2", approved=True)

        assert resp.status == "completed"
        assert resp.thread_id == paused.thread_id
        assert resp.report == "Report, ticket drafted"
        assert resp.tool_calls[0].name == "create_ticket"
        assert resp.tool_calls[0].status == "executed"
        assert resp.tool_calls[0].result["draft_id"].startswith("TICKET-DRAFT-")

    async def test_rejected_resume_skips_tool_and_completes(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        service = _service(provider, settings)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _report("Report without ticket"),
        ]
        _complete_with_tools(provider).side_effect = [
            _ticket_step(),
            _final("Findings"),
        ]

        paused = await service.run("topic", "req-1")
        resp = await service.resume(paused.thread_id, "req-2", approved=False)

        assert resp.status == "completed"
        assert resp.tool_calls[0].status == "rejected"
        assert resp.tool_calls[0].approved is False
        last_messages = _complete_with_tools(provider).await_args.kwargs["messages"]
        assert "Rejected by the human reviewer" in last_messages[-1]["content"]

    async def test_resume_unknown_thread_raises(self, provider):
        service = _service(provider, Settings())
        with pytest.raises(ResearchRunNotFoundError, match="missing-thread"):
            await service.resume("missing-thread", "req-1", approved=True)

    async def test_resume_finished_thread_raises(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        service = _service(provider, settings)
        _complete(provider).side_effect = [_plan_result(["a"]), _report("done")]
        _complete_with_tools(provider).return_value = _final("Findings")

        finished = await service.run("topic", "req-1")

        with pytest.raises(ResearchRunNotFoundError):
            await service.resume(finished.thread_id, "req-2", approved=True)
