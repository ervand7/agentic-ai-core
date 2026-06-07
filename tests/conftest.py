"""Shared pytest fixtures and mock factories.

Mock strategy (``unittest.mock.create_autospec``):
- Mocks are generated from port Protocols / classes so static checkers accept them
  as the real dependency types (no ``cast()`` needed at call sites).
- ``AsyncMock`` children (``.embed``, ``.generate``, ``.complete``) still support
  ``.assert_awaited_once()`` etc. at runtime.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.domains.ai_tasks.domain.ports import CompletionResult, LLMProvider  # noqa: E402
from app.domains.documents.domain.models import GeneratedAnswer  # noqa: E402
from app.domains.documents.domain.ports import (  # noqa: E402
    AnswerGenerator,
    EmbeddingProvider,
    VectorStore,
)
from app.domains.documents.infrastructure.in_memory_vector_store import (  # noqa: E402
    InMemoryVectorStore,
)
from app.shared.openai_types import (  # noqa: E402
    ChatCompletionMessageParam,
    ResponseFormat,
)


# ---------------------------------------------------------------------------
# Typed OpenAI request helpers (keep static checkers happy in client tests)
# ---------------------------------------------------------------------------
def user_messages(content: str = "hi") -> list[ChatCompletionMessageParam]:
    """A correctly typed single-user-message list for chat completion calls."""
    message: ChatCompletionMessageParam = {"role": "user", "content": content}
    return [message]


def json_object_format() -> ResponseFormat:
    """A correctly typed ``{"type": "json_object"}`` response format."""
    response_format: ResponseFormat = {"type": "json_object"}
    return response_format


# ---------------------------------------------------------------------------
# Mock factories (create_autospec from port Protocols)
# ---------------------------------------------------------------------------
def make_embedding_mock(
    default_vector: Optional[list[float]] = None,
) -> EmbeddingProvider:
    """Mock ``EmbeddingProvider`` with autospec."""
    mock = create_autospec(EmbeddingProvider, instance=True)
    mock.embed.return_value = (
        default_vector if default_vector is not None else [1.0, 0.0, 0.0]
    )
    return mock


def make_answer_generator_mock(
    content: str = "Generated answer [1].",
    model: str = "gpt-4o-mini",
    tokens_used: int = 17,
) -> AnswerGenerator:
    """Mock ``AnswerGenerator`` with autospec."""
    mock = create_autospec(AnswerGenerator, instance=True)
    mock.generate.return_value = GeneratedAnswer(
        content=content, model=model, tokens_used=tokens_used
    )
    return mock


def make_execute_service_mock(
    result=None,
    error: Optional[Exception] = None,
) -> AsyncMock:
    """Mock application services that expose ``async execute(**kwargs)``."""
    mock = AsyncMock()
    if error is not None:
        mock.execute.side_effect = error
    else:
        mock.execute.return_value = result
    return mock


def make_vector_store_mock(
    *,
    count: int = 0,
    search_results: Optional[list] = None,
) -> VectorStore:
    """Mock ``VectorStore`` with autospec."""
    mock = create_autospec(VectorStore, instance=True)
    mock.count.return_value = count
    mock.search.return_value = search_results if search_results is not None else []
    return mock


def async_mock_method(obj: object, method_name: str) -> AsyncMock:
    """Return the ``AsyncMock`` behind a ``create_autospec`` async method."""
    return cast(AsyncMock, getattr(obj, method_name))


def mock_method(obj: object, method_name: str) -> MagicMock:
    """Return the ``MagicMock`` behind a ``create_autospec`` sync method."""
    return cast(MagicMock, getattr(obj, method_name))


def make_llm_provider_mock(
    content: str = "result text",
    model: str = "gpt-4o-mini",
    tokens_used: int = 11,
) -> LLMProvider:
    """Mock ``LLMProvider`` with autospec."""
    mock = create_autospec(LLMProvider, instance=True)
    mock.complete.return_value = CompletionResult(
        content=content, model=model, tokens_used=tokens_used
    )
    return mock


@pytest.fixture
def mock_embeddings() -> EmbeddingProvider:
    return make_embedding_mock()


@pytest.fixture
def mock_answer_generator() -> AnswerGenerator:
    return make_answer_generator_mock()


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


# ---------------------------------------------------------------------------
# OpenAI SDK response factories (lightweight stand-ins, not service mocks)
# ---------------------------------------------------------------------------
def make_chat_completion(
    content: Optional[str] = "answer text",
    model: str = "gpt-4o-mini",
    total_tokens: Optional[int] = 42,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, delta=SimpleNamespace(content=content))
    usage = (
        SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def make_embedding_response(vector: list[float]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])


def make_stream_chunk(
    content: Optional[str] = None,
    model: Optional[str] = "gpt-4o-mini",
    total_tokens: Optional[int] = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    usage = (
        SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage)
