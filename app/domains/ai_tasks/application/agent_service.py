from __future__ import annotations

import logging
from typing import Any, Optional

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import ValidationError

from app.domains.ai_tasks.api.schemas import (
    AgentIterationResult,
    ResearchAgentResponse,
    ToolExecutionResult,
)
from app.domains.ai_tasks.application.conversation import (
    assistant_tool_call_message,
    parse_json_content,
    tool_result_message,
)
from app.domains.ai_tasks.application.tool_runner import ToolRunner
from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.agent import (
    AgentBudget,
    AgentCritique,
    AgentState,
    AgentStopReason,
    ResearchPlan,
)
from app.domains.ai_tasks.domain.ports import LLMProvider
from app.domains.ai_tasks.domain.prompts import get_prompts
from app.domains.ai_tasks.domain.response_formats import (
    AGENT_CRITIQUE_RESPONSE_FORMAT,
    RESEARCH_PLAN_RESPONSE_FORMAT,
)
from app.domains.ai_tasks.domain.tools import RESEARCH_TOOL_DEFINITIONS
from app.domains.documents.application.services import SearchDocumentsService
from app.shared.config import Settings
from app.shared.exceptions import LLMServiceError
from app.shared.openai_types import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

PLAN_RAW_LOG_LIMIT = 500
PLANNER_PROMPT_KEY = "research_agent_planner_v1"
AGENT_PROMPT_KEY = "research_agent_v1"
CRITIC_PROMPT_KEY = "research_agent_critic_v1"


class ResearchAgentService:
    """Goal-driven agent that researches a topic and writes a final report."""

    def __init__(
            self,
            *,
            llm_provider: LLMProvider,
            settings: Settings,
            document_search: Optional[SearchDocumentsService] = None,
            tool_runner: Optional[ToolRunner] = None,
            tools: Optional[list[dict[str, Any]]] = None,
    ):
        self._llm = llm_provider
        self._settings = settings
        self._tool_runner = tool_runner or ToolRunner(document_search=document_search)
        self._tools = tools if tools is not None else RESEARCH_TOOL_DEFINITIONS

    async def run(
            self,
            topic: str,
            request_id: str,
            *,
            max_iterations: Optional[int] = None,
    ) -> ResearchAgentResponse:
        """Run the full agent: plan, loop, (force final), reflect, report."""
        budget = AgentBudget(
            max_iterations=max_iterations or self._settings.AGENT_MAX_ITERATIONS,
            max_total_tokens=self._settings.AGENT_MAX_TOTAL_TOKENS,
        )
        state = AgentState(topic=topic, budget=budget)
        tool_executions: list[ToolExecutionResult] = []
        iterations: list[AgentIterationResult] = []

        plan = await self._plan(topic, request_id, state)
        state.plan = plan.steps

        messages = self._initial_messages(topic, plan)
        await self._run_loop(messages, state, request_id, tool_executions, iterations)

        # If the loop was cut short by a budget, the model never produced a
        # natural final answer -> force one synthesis call from what we gathered.
        if state.stop_reason != AgentStopReason.COMPLETED:
            await self._force_final_report(messages, state, request_id)

        if self._settings.AGENT_ENABLE_REFLECTION:
            await self._reflect(state, request_id)

        stop_reason = state.stop_reason or AgentStopReason.COMPLETED
        stop_reason_value: str = stop_reason
        logger.info(
            (
                "research_agent_completed request_id=%s iterations=%s tokens=%s "
                "stop_reason=%s tool_calls=%s reflected=%s"
            ),
            request_id,
            state.iteration,
            state.total_tokens,
            stop_reason_value,
            len(tool_executions),
            self._settings.AGENT_ENABLE_REFLECTION,
        )

        return ResearchAgentResponse(
            topic=topic,
            report=state.final_answer,
            plan=state.plan,
            iterations=iterations,
            tool_calls=tool_executions,
            critique=state.critique,
            stop_reason=stop_reason_value,
            iterations_used=state.iteration,
            model=state.model or self._settings.OPENAI_MODEL,
            tokens_used=state.total_tokens,
            prompt_version=get_prompts()[AGENT_PROMPT_KEY].prompt_version,
        )

    async def _plan(
            self, topic: str, request_id: str, state: AgentState
    ) -> ResearchPlan:
        """Ask the model for an ordered research plan (structured output)."""
        prompt = get_prompts()[PLANNER_PROMPT_KEY]
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": prompt.system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": f"Research topic:\n{topic}",
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]
        result = await self._llm.complete(
            endpoint=Endpoint.RESEARCH_AGENT_PLAN,
            request_id=request_id,
            messages=messages,
            temperature=0.0,
            max_tokens=self._settings.AGENT_PLAN_MAX_TOKENS,
            response_format=RESEARCH_PLAN_RESPONSE_FORMAT,
        )
        state.record_call(tokens_used=result.tokens_used, model=result.model)
        return self._parse_plan(result.content, topic, request_id)

    def _parse_plan(self, content: str, topic: str, request_id: str) -> ResearchPlan:
        fallback = ResearchPlan(steps=[f"Research the topic: {topic}"])
        try:
            plan = ResearchPlan.model_validate(parse_json_content(content))
        except (LLMServiceError, ValidationError) as exc:
            logger.warning(
                "research_agent_plan_parse_failed request_id=%s error=%s raw=%r",
                request_id,
                exc,
                (content or "")[:PLAN_RAW_LOG_LIMIT],
            )
            return fallback
        return plan if plan.steps else fallback

    def _initial_messages(
            self, topic: str, plan: ResearchPlan
    ) -> list[ChatCompletionMessageParam]:
        """Seed the loop with the agent system prompt + topic + plan."""
        prompt = get_prompts()[AGENT_PROMPT_KEY]
        plan_text = "\n".join(
            f"{index}. {step}" for index, step in enumerate(plan.steps, start=1)
        )
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": prompt.system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": (
                f"Research topic:\n{topic}\n\n"
                f"Suggested plan:\n{plan_text}\n\n"
                "Use the available tools to gather evidence. Call tools as many "
                "times as you need. When you have enough information, stop calling "
                "tools and write the final report: a short summary, the key "
                "findings as bullet points, and citations to the documents you "
                "relied on. If the documents do not contain enough information, "
                "say so honestly instead of guessing."
            ),
        }
        return [system_message, user_message]

    async def _run_loop(
            self,
            messages: list[ChatCompletionMessageParam],
            state: AgentState,
            request_id: str,
            tool_executions: list[ToolExecutionResult],
            iterations: list[AgentIterationResult],
    ) -> None:
        """The agent loop: reason -> act -> observe, bounded by the budget."""
        while True:
            budget_stop = state.budget_stop_reason()
            if budget_stop is not None:
                state.stop_reason = budget_stop
                return

            state.iteration += 1
            step = await self._llm.complete_with_tools(
                endpoint=Endpoint.RESEARCH_AGENT,
                request_id=request_id,
                messages=messages,
                tools=self._tools,
                temperature=self._settings.AGENT_TEMPERATURE,
            )
            state.record_call(tokens_used=step.tokens_used, model=step.model)

            # No tool calls means the model believes it is done.
            if not step.tool_calls:
                state.final_answer = step.content
                state.stop_reason = AgentStopReason.COMPLETED
                iterations.append(
                    AgentIterationResult(
                        iteration=state.iteration,
                        thought=step.content,
                        tool_calls=[],
                    )
                )
                return

            messages.append(assistant_tool_call_message(step))
            step_results: list[ToolExecutionResult] = []
            for call in step.tool_calls:
                execution = await self._tool_runner.execute(call, request_id=request_id)
                tool_executions.append(execution)
                step_results.append(execution)
                messages.append(tool_result_message(call.id, execution.result))

            iterations.append(
                AgentIterationResult(
                    iteration=state.iteration,
                    thought=step.content,
                    tool_calls=step_results,
                )
            )

    async def _force_final_report(
            self,
            messages: list[ChatCompletionMessageParam],
            state: AgentState,
            request_id: str,
    ) -> None:
        """Synthesize a best-effort report after a budget cut (no more tools)."""
        wrap_up: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": (
                "You have reached your research budget and must stop using tools. "
                "Write the final report now using only the information gathered so "
                "far. If the evidence is insufficient, state that explicitly rather "
                "than inventing details."
            ),
        }
        final = await self._llm.complete(
            endpoint=Endpoint.RESEARCH_AGENT,
            request_id=request_id,
            messages=[*messages, wrap_up],
            temperature=self._settings.AGENT_TEMPERATURE,
            max_tokens=self._settings.AGENT_REPORT_MAX_TOKENS,
        )
        state.record_call(tokens_used=final.tokens_used, model=final.model)
        state.final_answer = final.content

    async def _reflect(self, state: AgentState, request_id: str) -> None:
        """Self-critique the draft and optionally replace it with a revision."""
        if not state.final_answer:
            return
        prompt = get_prompts()[CRITIC_PROMPT_KEY]
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": prompt.system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": (
                f"Research topic:\n{state.topic}\n\n"
                f"Draft report:\n{state.final_answer}\n\n"
                "Critique the draft for factual support, relevance to the topic, "
                "and completeness. If it is good, set approved=true, leave issues "
                "empty, and leave revised_answer empty. If it can be improved, set "
                "approved=false, list concrete issues, and provide a corrected "
                "revised_answer."
            ),
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]
        result = await self._llm.complete(
            endpoint=Endpoint.RESEARCH_AGENT_CRITIC,
            request_id=request_id,
            messages=messages,
            temperature=0.0,
            max_tokens=self._settings.AGENT_CRITIC_MAX_TOKENS,
            response_format=AGENT_CRITIQUE_RESPONSE_FORMAT,
        )
        state.record_call(tokens_used=result.tokens_used, model=result.model)
        try:
            critique = AgentCritique.model_validate(parse_json_content(result.content))
        except (LLMServiceError, ValidationError) as exc:
            # Reflection is an optional polish step; a malformed critique must
            # not discard a valid report. Log and keep the draft as-is.
            logger.warning(
                "research_agent_critique_parse_failed request_id=%s error=%s raw=%r",
                request_id,
                exc,
                (result.content or "")[:PLAN_RAW_LOG_LIMIT],
            )
            return
        state.critique = (
            "; ".join(critique.issues) if critique.issues else "No issues found."
        )
        if not critique.approved and critique.revised_answer:
            state.final_answer = critique.revised_answer.strip()
