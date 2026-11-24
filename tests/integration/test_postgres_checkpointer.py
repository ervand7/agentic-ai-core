"""LangGraph research agent persisted in a real Postgres checkpointer."""

import json
import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.domains.ai_tasks.api import dependencies as ai_deps
from app.domains.ai_tasks.application.langgraph_research_agent_service import (
    LangGraphResearchAgentService,
)
from app.domains.ai_tasks.domain.ports import (
    CompletionResult,
    ToolCall,
    ToolCompletionResult,
)
from app.shared.config import Settings
from tests.conftest import async_mock_method, make_llm_provider_mock

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

POSTGRES_IMAGE = "postgres:17-alpine"


@pytest.fixture(scope="module")
def postgres_url():
    explicit = os.environ.get("LANGGRAPH_POSTGRES_TEST_URL")
    if explicit:
        yield explicit
        return

    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip(
            "testcontainers not installed and LANGGRAPH_POSTGRES_TEST_URL not set"
        )

    try:
        container = PostgresContainer(POSTGRES_IMAGE, driver=None)
        container.start()
    except Exception as exc:  # Docker not available / image pull failed.
        pytest.skip(f"Could not start Postgres container (Docker unavailable?): {exc}")

    try:
        yield container.get_connection_url()
    finally:
        container.stop()


async def test_lifespan_builds_agent_on_postgres_checkpointer(postgres_url):
    app = FastAPI()
    settings = Settings(LANGGRAPH_POSTGRES_URL=postgres_url)
    with patch.object(ai_deps, "get_settings", return_value=settings):
        async with ai_deps.langgraph_lifespan(app):
            service = app.state.langgraph_research_agent_service

            assert isinstance(service._graph.checkpointer, AsyncPostgresSaver)


async def test_paused_run_resumes_after_restart(postgres_url):
    settings = Settings(AGENT_ENABLE_REFLECTION=False)
    provider = make_llm_provider_mock()
    async_mock_method(provider, "complete").side_effect = [
        CompletionResult(
            content=json.dumps({"steps": ["a"]}), model="m", tokens_used=1
        ),
        CompletionResult(content="Report after restart", model="m", tokens_used=1),
    ]
    async_mock_method(provider, "complete_with_tools").side_effect = [
        ToolCompletionResult(
            content="",
            model="m",
            tokens_used=1,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="create_ticket",
                    arguments=json.dumps({"title": "Retries", "description": "Jitter"}),
                )
            ],
        ),
        ToolCompletionResult(
            content="Findings", model="m", tokens_used=1, tool_calls=[]
        ),
    ]

    async with AsyncPostgresSaver.from_conn_string(postgres_url) as checkpointer:
        await checkpointer.setup()
        before = LangGraphResearchAgentService(
            llm_provider=provider, settings=settings, checkpointer=checkpointer
        )
        paused = await before.run("topic", "req-1")

    async with AsyncPostgresSaver.from_conn_string(postgres_url) as checkpointer:
        after = LangGraphResearchAgentService(
            llm_provider=provider, settings=settings, checkpointer=checkpointer
        )
        resp = await after.resume(paused.thread_id, "req-2", approved=True)

    assert paused.status == "awaiting_approval"
    assert resp.status == "completed"
    assert resp.report == "Report after restart"
    assert resp.tool_calls[0].status == "executed"
