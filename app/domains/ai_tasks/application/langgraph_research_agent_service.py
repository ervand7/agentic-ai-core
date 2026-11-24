"""
LangGraph version of the research agent, built as a supervisor + workers graph.

Same planner, prompts, tools, and budgets as `ResearchAgentService`, but the
workflow is an explicit multi-agent graph whose state is checkpointed per
`thread_id`:

    supervisor -> planner | researcher | writer | critic | END
    every worker -> supervisor          (hub and spoke)
    researcher <-> tools                (agent loop)

The supervisor reads the shared state and hands off to the next worker:
planner -> researcher -> writer -> critic. A rejected report sends the
researcher back for another round; the writer then rewrites the report.

New compared to the plain loop:
- separate worker roles, with a dedicated writer producing the final report;
- the critic can send the report back for another research round;
- the run pauses (human-in-the-loop) before any tool that is not read-only and
  continues after `resume(thread_id, approved=...)`;
- LLM workers are retried on transient provider errors (`RetryPolicy`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# PyCharm's package-requirements inspection does not always see Poetry deps.
# noinspection PyPackageRequirements
from langgraph.checkpoint.base import BaseCheckpointSaver

# noinspection PyPackageRequirements
from langgraph.checkpoint.memory import InMemorySaver

# noinspection PyPackageRequirements
from langgraph.graph import END, START, StateGraph

# noinspection PyPackageRequirements
from langgraph.types import Command, RetryPolicy, StateSnapshot, interrupt

from app.domains.ai_tasks.api.schemas import (
    AgentIterationResult,
    LangGraphResearchAgentResponse,
    PendingToolCall,
    ToolExecutionResult,
)
from app.domains.ai_tasks.application.conversation import (
    assistant_tool_call_message,
    tool_result_message,
)
from app.domains.ai_tasks.application.research_agent_service import (
    AGENT_PROMPT_KEY,
    ResearchAgentService,
)
from app.domains.ai_tasks.application.tool_runner import ToolRunner
from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.agent import AgentBudget, AgentState, AgentStopReason
from app.domains.ai_tasks.domain.ports import LLMProvider, ToolCall
from app.domains.ai_tasks.domain.prompts import get_prompts
from app.domains.ai_tasks.domain.tools import TOOL_REGISTRY, ToolRisk
from app.domains.documents.application.search_documents_service import (
    SearchDocumentsService,
)
from app.shared.config import Settings
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMTemporaryError,
    LLMTimeoutError,
)
from app.shared.openai_types import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

SUPERVISOR_NODE = "supervisor"
PLANNER_NODE = "planner"
RESEARCHER_NODE = "researcher"
TOOLS_NODE = "tools"
WRITER_NODE = "writer"
CRITIC_NODE = "critic"
WORKER_NODES = (PLANNER_NODE, RESEARCHER_NODE, WRITER_NODE, CRITIC_NODE, END)

WRITER_PROMPT_KEY = "research_agent_writer_v1"
REJECTED_TOOL_RESULT = {"error": "Rejected by the human reviewer. Do not retry it."}
# Only transient provider failures are worth re-running a whole node for.
TRANSIENT_LLM_ERRORS = (LLMTimeoutError, LLMRateLimitError, LLMTemporaryError)


class ResearchRunNotFoundError(Exception):
    """Raised when resuming a thread that has no run paused for approval."""


@dataclass
class ResearchGraphState:
    """Checkpointed graph state. Kept to plain JSON-like values."""

    topic: str = ""
    max_iterations: int = 0
    plan: list[str] = field(default_factory=list)
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    iteration: int = 0
    total_tokens: int = 0
    model: str = ""
    findings: str = ""
    research_done: bool = False
    report: str = ""
    reviewed: bool = False
    critique: Optional[str] = None
    stop_reason: Optional[str] = None
    revisions: int = 0
    pending_thought: str = ""
    pending_tool_calls: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)


def _request_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("request_id", "")


def _needs_approval(call: ToolCall) -> bool:
    spec = TOOL_REGISTRY.get(call.name)
    return spec is not None and spec.risk != ToolRisk.READ_ONLY


class LangGraphResearchAgentService(ResearchAgentService):
    """Research agent orchestrated as a LangGraph supervisor + workers graph."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        settings: Settings,
        document_search: Optional[SearchDocumentsService] = None,
        tool_runner: Optional[ToolRunner] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            settings=settings,
            document_search=document_search,
            tool_runner=tool_runner,
            tools=tools,
        )
        self._graph: Any = self._build_graph().compile(
            checkpointer=checkpointer or InMemorySaver()
        )

    async def run(
        self,
        topic: str,
        request_id: str,
        *,
        max_iterations: Optional[int] = None,
    ) -> LangGraphResearchAgentResponse:
        """Start a new run. It may finish or pause awaiting human approval."""
        thread_id = uuid4().hex
        config = self._config(thread_id, request_id)
        initial_state = ResearchGraphState(
            topic=topic,
            max_iterations=max_iterations or self._settings.AGENT_MAX_ITERATIONS,
        )
        await self._graph.ainvoke(initial_state, config)
        return await self._response(thread_id, config)

    async def resume(
        self, thread_id: str, request_id: str, *, approved: bool
    ) -> LangGraphResearchAgentResponse:
        """Continue a paused run with the human approval decision."""
        config = self._config(thread_id, request_id)
        snapshot = await self._graph.aget_state(config)
        if not snapshot.interrupts:
            raise ResearchRunNotFoundError(
                f"No research run is waiting for approval with thread_id={thread_id}."
            )
        await self._graph.ainvoke(Command(resume=approved), config)
        return await self._response(thread_id, config)

    def _build_graph(self) -> Any:
        # Tools are not retried: they may have side effects and hold the interrupt.
        llm_retry = RetryPolicy(
            max_attempts=self._settings.AGENT_NODE_MAX_ATTEMPTS,
            initial_interval=self._settings.OPENAI_RETRY_BASE_DELAY_SECONDS,
            retry_on=TRANSIENT_LLM_ERRORS,
        )
        graph: Any = StateGraph(cast(Any, ResearchGraphState))
        graph.add_node(
            SUPERVISOR_NODE, self._supervisor_node, destinations=WORKER_NODES
        )
        graph.add_node(PLANNER_NODE, self._planner_node, retry_policy=llm_retry)
        graph.add_node(RESEARCHER_NODE, self._researcher_node, retry_policy=llm_retry)
        graph.add_node(TOOLS_NODE, self._tools_node)
        graph.add_node(WRITER_NODE, self._writer_node, retry_policy=llm_retry)
        graph.add_node(CRITIC_NODE, self._critic_node, retry_policy=llm_retry)

        graph.add_edge(START, SUPERVISOR_NODE)
        graph.add_edge(PLANNER_NODE, SUPERVISOR_NODE)
        graph.add_conditional_edges(
            RESEARCHER_NODE, self._after_researcher, [TOOLS_NODE, SUPERVISOR_NODE]
        )
        graph.add_edge(TOOLS_NODE, RESEARCHER_NODE)
        graph.add_edge(WRITER_NODE, SUPERVISOR_NODE)
        graph.add_edge(CRITIC_NODE, SUPERVISOR_NODE)
        return graph

    # ----------------------------------------------------------------- nodes

    def _supervisor_node(self, state: ResearchGraphState) -> Command:
        """Hand off to the next worker based on what the shared state still lacks."""
        if not state.plan:
            goto = PLANNER_NODE
        elif not state.research_done:
            goto = RESEARCHER_NODE
        elif not state.report:
            goto = WRITER_NODE
        elif self._settings.AGENT_ENABLE_REFLECTION and not state.reviewed:
            goto = CRITIC_NODE
        else:
            goto = END
        return Command(goto=goto)

    async def _planner_node(
        self, state: ResearchGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        agent = self._agent_state(state)
        plan = await self._plan(state.topic, _request_id(config), agent)
        return {
            **self._agent_updates(agent),
            "plan": plan.steps,
            "messages": self._initial_messages(state.topic, plan),
        }

    async def _researcher_node(
        self, state: ResearchGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """One reason step: the model either requests tools or reports findings."""
        agent = self._agent_state(state)
        budget_stop = agent.budget_stop_reason()
        if budget_stop is not None:
            return {"stop_reason": budget_stop.value, "research_done": True}

        agent.iteration += 1
        step = await self._llm.complete_with_tools(
            endpoint=Endpoint.RESEARCH_AGENT,
            request_id=_request_id(config),
            messages=state.messages,
            tools=self._tools,
            temperature=self._settings.AGENT_TEMPERATURE,
        )
        agent.record_call(tokens_used=step.tokens_used, model=step.model)

        if not step.tool_calls:
            agent.stop_reason = AgentStopReason.COMPLETED
            iteration = AgentIterationResult(
                iteration=agent.iteration, thought=step.content
            )
            return {
                **self._agent_updates(agent),
                "messages": [
                    *state.messages,
                    {"role": "assistant", "content": step.content},
                ],
                "findings": step.content,
                "research_done": True,
                "iterations": [*state.iterations, iteration.model_dump()],
            }

        agent.stop_reason = None
        return {
            **self._agent_updates(agent),
            "messages": [*state.messages, assistant_tool_call_message(step)],
            "pending_thought": step.content,
            "pending_tool_calls": [asdict(call) for call in step.tool_calls],
        }

    async def _tools_node(
        self, state: ResearchGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Run requested tools, pausing for human approval on risky ones."""
        calls = [ToolCall(**call) for call in state.pending_tool_calls]
        risky_ids = {call.id for call in calls if _needs_approval(call)}
        # `interrupt` pauses the graph here; on resume the node re-runs from the
        # top and `interrupt` returns the human decision instead of pausing.
        approved = bool(interrupt(state.pending_tool_calls)) if risky_ids else True

        messages = list(state.messages)
        executions: list[ToolExecutionResult] = []
        for call in calls:
            if call.id in risky_ids and not approved:
                execution = self._rejected_execution(call)
            else:
                execution = await self._tool_runner.execute(
                    call,
                    request_id=_request_id(config),
                    approved_tool_call_ids=risky_ids,
                )
            executions.append(execution)
            messages.append(tool_result_message(call.id, execution.result))

        iteration = AgentIterationResult(
            iteration=state.iteration,
            thought=state.pending_thought,
            tool_calls=executions,
        )
        return {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_calls": [
                *state.tool_calls,
                *(execution.model_dump() for execution in executions),
            ],
            "iterations": [*state.iterations, iteration.model_dump()],
        }

    async def _writer_node(
        self, state: ResearchGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Write the final report from the findings and the gathered evidence."""
        agent = self._agent_state(state)
        prompt = get_prompts()[WRITER_PROMPT_KEY]
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": prompt.system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": self._writer_brief(state),
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]
        result = await self._llm.complete(
            endpoint=Endpoint.RESEARCH_AGENT_WRITER,
            request_id=_request_id(config),
            messages=messages,
            temperature=self._settings.AGENT_TEMPERATURE,
            max_tokens=self._settings.AGENT_REPORT_MAX_TOKENS,
        )
        agent.record_call(tokens_used=result.tokens_used, model=result.model)
        agent.final_answer = result.content
        return {**self._agent_updates(agent), "reviewed": False}

    async def _critic_node(
        self, state: ResearchGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Review the report; send the researcher back if it is rejected."""
        agent = self._agent_state(state)
        critique = await self._reflect(agent, _request_id(config))

        can_revise = (
            state.stop_reason == AgentStopReason.COMPLETED.value
            and state.revisions < self._settings.AGENT_MAX_REVISIONS
        )
        if critique is None or critique.approved or not can_revise:
            return {**self._agent_updates(agent), "reviewed": True}

        feedback = (
            "A reviewer rejected the report written from your findings with these "
            "issues:\n"
            + "\n".join(f"- {issue}" for issue in critique.issues)
            + "\n\nResearch further with the tools if needed, then summarize your "
            "updated findings."
        )
        return {
            **self._agent_updates(agent),
            "messages": [*state.messages, {"role": "user", "content": feedback}],
            "report": "",
            "research_done": False,
            "revisions": state.revisions + 1,
        }

    # --------------------------------------------------------------- routing

    @staticmethod
    def _after_researcher(state: ResearchGraphState) -> str:
        return TOOLS_NODE if state.pending_tool_calls else SUPERVISOR_NODE

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _config(thread_id: str, request_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id, "request_id": request_id}}

    @staticmethod
    def _writer_brief(state: ResearchGraphState) -> str:
        plan_text = "\n".join(
            f"{index}. {step}" for index, step in enumerate(state.plan, start=1)
        )
        evidence = [
            {
                "tool": call["name"],
                "arguments": call["arguments"],
                "result": call["result"],
            }
            for call in state.tool_calls
            if call["status"] == "executed"
        ]
        findings = state.findings or "(none)"
        if state.stop_reason != AgentStopReason.COMPLETED.value:
            findings += (
                f"\n\nResearch was stopped early ({state.stop_reason}); write the "
                "best report possible from the evidence gathered so far."
            )
        return (
            f"Research topic:\n{state.topic}\n\n"
            f"Research plan:\n{plan_text}\n\n"
            f"Researcher findings:\n{findings}\n\n"
            f"Tool evidence (JSON):\n{json.dumps(evidence, ensure_ascii=False)}"
        )

    def _agent_state(self, state: ResearchGraphState) -> AgentState:
        """Bridge graph state onto the `AgentState` the base-class steps use."""
        stop_reason = state.stop_reason
        return AgentState(
            topic=state.topic,
            budget=AgentBudget(
                max_iterations=state.max_iterations,
                max_total_tokens=self._settings.AGENT_MAX_TOTAL_TOKENS,
            ),
            iteration=state.iteration,
            total_tokens=state.total_tokens,
            model=state.model,
            final_answer=state.report,
            critique=state.critique,
            stop_reason=AgentStopReason(stop_reason) if stop_reason else None,
        )

    @staticmethod
    def _agent_updates(agent: AgentState) -> dict[str, Any]:
        return {
            "iteration": agent.iteration,
            "total_tokens": agent.total_tokens,
            "model": agent.model,
            "report": agent.final_answer,
            "critique": agent.critique,
            "stop_reason": agent.stop_reason.value if agent.stop_reason else None,
        }

    @staticmethod
    def _rejected_execution(call: ToolCall) -> ToolExecutionResult:
        spec = TOOL_REGISTRY[call.name]
        return ToolExecutionResult(
            name=call.name,
            arguments={},
            result=REJECTED_TOOL_RESULT,
            tool_call_id=call.id,
            risk=spec.risk.value,
            requires_approval=True,
            approved=False,
            status="rejected",
            error="rejected_by_human",
        )

    async def _response(
        self, thread_id: str, config: RunnableConfig
    ) -> LangGraphResearchAgentResponse:
        snapshot: StateSnapshot = await self._graph.aget_state(config)
        values: dict[str, Any] = snapshot.values
        awaiting_approval = bool(snapshot.interrupts)
        status: Literal["completed", "awaiting_approval"] = (
            "awaiting_approval" if awaiting_approval else "completed"
        )

        logger.info(
            (
                "langgraph_research_agent_step request_id=%s thread_id=%s status=%s "
                "iterations=%s tokens=%s revisions=%s"
            ),
            _request_id(config),
            thread_id,
            status,
            values["iteration"],
            values["total_tokens"],
            values["revisions"],
        )

        return LangGraphResearchAgentResponse(
            thread_id=thread_id,
            status=status,
            topic=values["topic"],
            report="" if awaiting_approval else values["report"],
            plan=values.get("plan", []),
            iterations=[
                AgentIterationResult.model_validate(item)
                for item in values["iterations"]
            ],
            tool_calls=[
                ToolExecutionResult.model_validate(item)
                for item in values["tool_calls"]
            ],
            critique=values["critique"],
            stop_reason=values["stop_reason"] or status,
            iterations_used=values["iteration"],
            model=values["model"] or self._settings.OPENAI_MODEL,
            tokens_used=values["total_tokens"],
            prompt_version=get_prompts()[AGENT_PROMPT_KEY].prompt_version,
            revisions=values["revisions"],
            pending_tool_calls=[
                PendingToolCall.model_validate(call)
                for call in values["pending_tool_calls"]
            ]
            if awaiting_approval
            else [],
        )
