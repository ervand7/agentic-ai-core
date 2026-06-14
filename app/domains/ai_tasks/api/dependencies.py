"""FastAPI dependencies that wire concrete implementations into the API."""

from functools import lru_cache

from app.domains.ai_tasks.application.agent_service import ResearchAgentService
from app.domains.ai_tasks.application.services import AITasksService
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
