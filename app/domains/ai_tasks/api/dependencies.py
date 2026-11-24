"""FastAPI dependencies that wire concrete implementations into the API."""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.domains.ai_tasks.application.langgraph_research_agent_service import (
    LangGraphResearchAgentService,
)
from app.domains.ai_tasks.application.research_agent_service import ResearchAgentService
from app.domains.ai_tasks.application.ai_tasks_service import AITasksService
from app.domains.ai_tasks.infrastructure.langgraph_checkpointer import (
    open_checkpointer,
)
from app.domains.ai_tasks.infrastructure.openai_llm_provider import OpenAILLMProvider
from app.domains.documents.api.dependencies import get_search_documents_service
from app.shared.config import get_settings
from app.shared.infrastructure.openai_client import get_openai_client


@lru_cache
def get_ai_tasks_service() -> AITasksService:
    """Compose the AI tasks service with the OpenAI LLM provider."""
    return AITasksService(
        llm_provider=OpenAILLMProvider(get_openai_client()),
        settings=get_settings(),
        document_search=get_search_documents_service(),
    )


@lru_cache
def get_research_agent_service() -> ResearchAgentService:
    """Compose the research agent with the OpenAI provider and doc search tool."""
    return ResearchAgentService(
        llm_provider=OpenAILLMProvider(get_openai_client()),
        settings=get_settings(),
        document_search=get_search_documents_service(),
    )


@asynccontextmanager
async def langgraph_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the LangGraph agent on a checkpointer that stays open for the app's life."""
    settings = get_settings()
    async with open_checkpointer(settings.LANGGRAPH_POSTGRES_URL) as checkpointer:
        app.state.langgraph_research_agent_service = LangGraphResearchAgentService(
            llm_provider=OpenAILLMProvider(get_openai_client()),
            settings=settings,
            document_search=get_search_documents_service(),
            checkpointer=checkpointer,
        )
        yield


def get_langgraph_research_agent_service(
    request: Request,
) -> LangGraphResearchAgentService:
    """The LangGraph agent built at startup, so all requests share its checkpoints."""
    return request.app.state.langgraph_research_agent_service
