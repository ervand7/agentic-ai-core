"""Unit tests for the agent loop control state and safety budgets."""

from app.domains.ai_tasks.domain.agent import (
    AgentBudget,
    AgentCritique,
    AgentState,
    AgentStopReason,
    ResearchPlan,
)


class TestAgentState:
    def test_records_tokens_and_keeps_latest_non_empty_model(self):
        state = AgentState(
            topic="t", budget=AgentBudget(max_iterations=3, max_total_tokens=100)
        )
        state.record_call(tokens_used=10, model="m1")
        state.record_call(tokens_used=5, model="")

        assert state.total_tokens == 15
        assert state.model == "m1"

    def test_iteration_budget_reached(self):
        state = AgentState(
            topic="t", budget=AgentBudget(max_iterations=2, max_total_tokens=100)
        )
        assert state.budget_stop_reason() is None

        state.iteration = 2
        assert state.iteration_budget_reached is True
        assert state.budget_stop_reason() is AgentStopReason.MAX_ITERATIONS

    def test_token_budget_reached(self):
        state = AgentState(
            topic="t", budget=AgentBudget(max_iterations=10, max_total_tokens=20)
        )
        state.total_tokens = 25

        assert state.token_budget_reached is True
        assert state.tokens_remaining == 0
        assert state.budget_stop_reason() is AgentStopReason.TOKEN_BUDGET

    def test_iteration_budget_takes_precedence_over_tokens(self):
        state = AgentState(
            topic="t", budget=AgentBudget(max_iterations=1, max_total_tokens=5)
        )
        state.iteration = 1
        state.total_tokens = 10

        assert state.budget_stop_reason() is AgentStopReason.MAX_ITERATIONS

    def test_tokens_remaining_never_negative(self):
        state = AgentState(
            topic="t", budget=AgentBudget(max_iterations=5, max_total_tokens=10)
        )
        state.total_tokens = 30
        assert state.tokens_remaining == 0


class TestStructuredModels:
    def test_research_plan_defaults_to_empty(self):
        assert ResearchPlan().steps == []

    def test_agent_critique_defaults_to_approved(self):
        critique = AgentCritique()
        assert critique.approved is True
        assert critique.issues == []
        assert critique.revised_answer is None
