"""Unit tests for the LangGraph research agent: loop, critic revisions, approvals."""

import json
from unittest.mock import AsyncMock

import pytest

from app.domains.ai_tasks.application.langgraph_research_agent_service import (
    LangGraphResearchAgentService,
    ResearchRunNotFoundError,
)
from app.domains.ai_tasks.domain.ports import (
    CompletionResult,
    LLMProvider,
    ToolCall,
    ToolCompletionResult,
)
from app.shared.config import Settings
from tests.conftest import async_mock_method, make_llm_provider_mock


def _complete(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete")


def _complete_with_tools(provider: LLMProvider) -> AsyncMock:
    return async_mock_method(provider, "complete_with_tools")


def _plan_result(steps: list[str], tokens: int = 10) -> CompletionResult:
    return CompletionResult(
        content=json.dumps({"steps": steps}), model="m", tokens_used=tokens
    )


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


@pytest.fixture
def provider() -> LLMProvider:
    return make_llm_provider_mock()


def _service(
    provider: LLMProvider, settings: Settings
) -> LangGraphResearchAgentService:
    return LangGraphResearchAgentService(llm_provider=provider, settings=settings)


class TestCompletion:
    async def test_completes_when_model_stops_calling_tools(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a", "b"])]
        _complete_with_tools(provider).return_value = _final("Final report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.status == "completed"
        assert resp.thread_id
        assert resp.report == "Final report"
        assert resp.stop_reason == "completed"
        assert resp.plan == ["a", "b"]
        assert resp.iterations_used == 1
        assert resp.tokens_used == 13
        assert resp.pending_tool_calls == []

    async def test_read_only_tool_runs_without_pausing(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).side_effect = [
            _tool_step("search_docs", {"query": "retry logic"}),
            _final("Report with evidence"),
        ]

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.status == "completed"
        assert resp.report == "Report with evidence"
        assert resp.tool_calls[0].name == "search_docs"
        assert resp.tool_calls[0].status == "executed"
        assert len(resp.iterations) == 2

    async def test_budget_stop_forces_final_report(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            CompletionResult(content="Forced report", model="m", tokens_used=5),
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


class TestCriticRevisions:
    async def test_rejected_report_goes_back_for_another_round(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True, AGENT_MAX_REVISIONS=1)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _critique_result(approved=False, issues=["missing citations"]),
            _critique_result(approved=True, issues=[]),
        ]
        _complete_with_tools(provider).side_effect = [
            _final("Draft report"),
            _final("Improved report"),
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
            "content": "Draft report",
        }
        assert "missing citations" in second_round_messages[-1]["content"]

    async def test_revision_cap_falls_back_to_critic_revised_answer(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True, AGENT_MAX_REVISIONS=0)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _critique_result(approved=False, issues=["too short"], revised="Better"),
        ]
        _complete_with_tools(provider).return_value = _final("Draft report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Better"
        assert resp.revisions == 0
        assert _complete_with_tools(provider).await_count == 1


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
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).side_effect = [
            _ticket_step(),
            _final("Report, ticket drafted"),
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
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).side_effect = [
            _ticket_step(),
            _final("Report without ticket"),
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
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).return_value = _final("done")

        finished = await service.run("topic", "req-1")

        with pytest.raises(ResearchRunNotFoundError):
            await service.resume(finished.thread_id, "req-2", approved=True)
