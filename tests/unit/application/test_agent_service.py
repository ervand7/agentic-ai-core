"""Unit tests for the ResearchAgentService loop, budgets, and reflection."""

import json
from unittest.mock import AsyncMock

import pytest

from app.domains.ai_tasks.application.agent_service import ResearchAgentService
from app.domains.ai_tasks.constants import Endpoint
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


def _tool_step(query: str = "retry logic", tokens: int = 3) -> ToolCompletionResult:
    return ToolCompletionResult(
        content="",
        model="m",
        tokens_used=tokens,
        tool_calls=[
            ToolCall(
                id="call_1",
                name="search_docs",
                arguments=json.dumps({"query": query}),
            )
        ],
    )


@pytest.fixture
def provider() -> LLMProvider:
    return make_llm_provider_mock()


def _service(provider: LLMProvider, settings: Settings) -> ResearchAgentService:
    return ResearchAgentService(llm_provider=provider, settings=settings)


class TestNaturalCompletion:
    async def test_completes_when_model_stops_calling_tools(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a", "b"])]
        _complete_with_tools(provider).return_value = _final("Final report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Final report"
        assert resp.stop_reason == "completed"
        assert resp.iterations_used == 1
        assert resp.plan == ["a", "b"]
        assert resp.tool_calls == []
        assert resp.prompt_version == "research_agent_v1"

    async def test_sums_tokens_across_plan_and_loop(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"], tokens=10)]
        _complete_with_tools(provider).return_value = _final("done", tokens=3)

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.tokens_used == 13

    async def test_empty_plan_falls_back_to_default_step(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result([])]
        _complete_with_tools(provider).return_value = _final("done")

        resp = await _service(provider, settings).run("my topic", "req-1")

        assert resp.plan == ["Research the topic: my topic"]


class TestPlanResilience:
    async def test_non_json_plan_falls_back_instead_of_raising(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        # The planner returns prose, not JSON: the run must still succeed.
        _complete(provider).side_effect = [
            CompletionResult(content="1. look at retries", model="m", tokens_used=5)
        ]
        _complete_with_tools(provider).return_value = _final("done")

        resp = await _service(provider, settings).run("my topic", "req-1")

        assert resp.plan == ["Research the topic: my topic"]
        assert resp.report == "done"
        assert resp.stop_reason == "completed"

    async def test_plan_uses_dedicated_max_tokens(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).return_value = _final("done")

        await _service(provider, settings).run("topic", "req-1")

        plan_call = _complete(provider).await_args_list[0]
        assert plan_call.kwargs["max_tokens"] == settings.AGENT_PLAN_MAX_TOKENS


class TestToolLoop:
    async def test_executes_tools_then_completes(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).side_effect = [
            _tool_step(),
            _final("Report with evidence"),
        ]

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Report with evidence"
        assert resp.iterations_used == 2
        assert len(resp.iterations) == 2
        assert resp.tool_calls[0].name == "search_docs"
        assert resp.tool_calls[0].status == "executed"

    async def test_loop_uses_research_agent_endpoint(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).return_value = _final("done")

        await _service(provider, settings).run("topic", "req-1")

        assert (
            _complete_with_tools(provider).await_args.kwargs["endpoint"]
            == Endpoint.RESEARCH_AGENT
        )


class TestBudgets:
    async def test_stops_at_max_iterations_and_forces_final_report(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        # Plan call, then the forced final-report call.
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            CompletionResult(content="Forced report", model="m", tokens_used=5),
        ]
        # The model keeps calling tools forever; the cap must stop it.
        _complete_with_tools(provider).return_value = _tool_step()

        resp = await _service(provider, settings).run(
            "topic", "req-1", max_iterations=2
        )

        assert resp.stop_reason == "max_iterations"
        assert resp.iterations_used == 2
        assert resp.report == "Forced report"
        assert len(resp.tool_calls) == 2

    async def test_token_budget_stops_loop(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False, AGENT_MAX_TOTAL_TOKENS=5)
        _complete(provider).side_effect = [
            _plan_result(["a"], tokens=10),  # already over the 5-token budget
            CompletionResult(content="Forced report", model="m", tokens_used=1),
        ]
        _complete_with_tools(provider).return_value = _tool_step()

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.stop_reason == "token_budget"
        assert resp.iterations_used == 0
        assert resp.report == "Forced report"


class TestReflection:
    async def test_reflection_revises_report_when_not_approved(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _critique_result(
                approved=False, issues=["too short"], revised="Better report"
            ),
        ]
        _complete_with_tools(provider).return_value = _final("Draft report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Better report"
        assert resp.critique == "too short"

    async def test_critic_uses_dedicated_max_tokens(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _critique_result(approved=True, issues=[]),
        ]
        _complete_with_tools(provider).return_value = _final("Draft report")

        await _service(provider, settings).run("topic", "req-1")

        critic_call = _complete(provider).await_args_list[-1]
        assert critic_call.kwargs["max_tokens"] == settings.AGENT_CRITIC_MAX_TOKENS

    async def test_reflection_keeps_report_when_approved(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            _critique_result(approved=True, issues=[]),
        ]
        _complete_with_tools(provider).return_value = _final("Draft report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Draft report"
        assert resp.critique == "No issues found."

    async def test_reflection_disabled_skips_critic_call(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=False)
        _complete(provider).side_effect = [_plan_result(["a"])]
        _complete_with_tools(provider).return_value = _final("Draft report")

        resp = await _service(provider, settings).run("topic", "req-1")

        # Only the planning call should have hit `complete`.
        assert _complete(provider).await_count == 1
        assert resp.critique is None

    async def test_non_json_critique_keeps_draft_report(self, provider):
        settings = Settings(AGENT_ENABLE_REFLECTION=True)
        _complete(provider).side_effect = [
            _plan_result(["a"]),
            CompletionResult(content="not json", model="m", tokens_used=4),
        ]
        _complete_with_tools(provider).return_value = _final("Draft report")

        resp = await _service(provider, settings).run("topic", "req-1")

        assert resp.report == "Draft report"
        assert resp.critique is None
