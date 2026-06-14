from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentStopReason(str, Enum):
    """Why the agent loop terminated."""

    COMPLETED = "completed"
    """The model stopped requesting tools and produced a final answer."""

    MAX_ITERATIONS = "max_iterations"
    """The loop hit its iteration cap before the model finished."""

    TOKEN_BUDGET = "token_budget"
    """The accumulated token spend reached the configured budget."""


@dataclass(frozen=True)
class AgentBudget:
    """Hard safety limits for a single agent run."""

    max_iterations: int
    max_total_tokens: int


@dataclass
class AgentState:
    topic: str
    budget: AgentBudget
    iteration: int = 0
    total_tokens: int = 0
    model: str = ""
    plan: list[str] = field(default_factory=list)
    final_answer: str = ""
    critique: Optional[str] = None
    stop_reason: Optional[AgentStopReason] = None

    def record_call(self, *, tokens_used: int, model: str) -> None:
        """Account for one model call against the token budget."""
        self.total_tokens += tokens_used
        if model:
            self.model = model

    @property
    def iteration_budget_reached(self) -> bool:
        return self.iteration >= self.budget.max_iterations

    @property
    def token_budget_reached(self) -> bool:
        return self.total_tokens >= self.budget.max_total_tokens

    @property
    def tokens_remaining(self) -> int:
        return max(self.budget.max_total_tokens - self.total_tokens, 0)

    def budget_stop_reason(self) -> Optional[AgentStopReason]:
        if self.iteration_budget_reached:
            return AgentStopReason.MAX_ITERATIONS
        if self.token_budget_reached:
            return AgentStopReason.TOKEN_BUDGET
        return None


class ResearchPlan(BaseModel):
    """Structured planning output produced before the agent loop starts."""

    steps: list[str] = Field(default_factory=list)


class AgentCritique(BaseModel):
    """Structured reflection output used to optionally revise the report."""

    approved: bool = True
    issues: list[str] = Field(default_factory=list)
    revised_answer: Optional[str] = None


__all__ = [
    "AgentBudget",
    "AgentCritique",
    "AgentState",
    "AgentStopReason",
    "ResearchPlan",
]
