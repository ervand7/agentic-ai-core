"""FastAPI dependencies that wire concrete implementations into the API."""

from functools import lru_cache

from app.domains.ai_tasks.application.services import AITasksService
from app.domains.ai_tasks.infrastructure.openai_llm_provider import OpenAILLMProvider
from app.shared.config import get_settings
from app.shared.infrastructure.openai_client import get_openai_client


@lru_cache
def get_ai_tasks_service() -> AITasksService:
    """Compose the AI tasks service with the OpenAI LLM provider."""
    return AITasksService(OpenAILLMProvider(get_openai_client()), get_settings())
