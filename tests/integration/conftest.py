from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.domains.ai_tasks.api.dependencies import get_ai_tasks_service
from app.domains.ai_tasks.application.services import AITasksService
from app.domains.documents.api.dependencies import (
    get_answer_question_service,
    get_ingest_document_service,
    get_search_documents_service,
)
from app.domains.documents.application.services import (
    AnswerQuestionService,
    IngestDocumentService,
    SearchDocumentsService,
)
from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)
from app.main import app
from app.shared.config import get_settings
from tests.integration.fakes import (
    FakeAnswerGenerator,
    FakeEmbeddingProvider,
    FakeLLMProvider,
)


@dataclass
class DocumentsHarness:
    """Bundles the test client with the fakes so tests can introspect them."""

    client: TestClient
    store: InMemoryVectorStore
    embeddings: FakeEmbeddingProvider
    answer_generator: FakeAnswerGenerator


@pytest.fixture
def documents_harness() -> Iterator[DocumentsHarness]:
    """Real documents services backed by InMemoryVectorStore + fake providers."""
    settings = get_settings()
    store = InMemoryVectorStore()
    embeddings = FakeEmbeddingProvider()
    answer_generator = FakeAnswerGenerator()

    def ingest_service() -> IngestDocumentService:
        return IngestDocumentService(
            embedding_provider=embeddings,
            vector_store=store,
            chunk_size=settings.DOCUMENT_CHUNK_SIZE,
            chunk_overlap=settings.DOCUMENT_CHUNK_OVERLAP,
        )

    def search_service() -> SearchDocumentsService:
        return SearchDocumentsService(
            embedding_provider=embeddings,
            vector_store=store,
        )

    def answer_service() -> AnswerQuestionService:
        return AnswerQuestionService(
            embedding_provider=embeddings,
            vector_store=store,
            answer_generator=answer_generator,
            system_prompt=settings.PROMPT_RAG_SYSTEM,
            top_k=settings.RAG_TOP_K,
            min_similarity=settings.RAG_MIN_SIMILARITY,
            temperature=settings.RAG_TEMPERATURE,
            max_tokens=settings.RAG_MAX_TOKENS,
        )

    app.dependency_overrides[get_ingest_document_service] = ingest_service
    app.dependency_overrides[get_search_documents_service] = search_service
    app.dependency_overrides[get_answer_question_service] = answer_service

    with TestClient(app) as client:
        yield DocumentsHarness(
            client=client,
            store=store,
            embeddings=embeddings,
            answer_generator=answer_generator,
        )

    app.dependency_overrides.clear()


@dataclass
class AITasksHarness:
    """Bundles the ai_tasks client with its fake LLM provider."""

    client: TestClient
    llm: FakeLLMProvider


@pytest.fixture
def ai_tasks_harness() -> Iterator[AITasksHarness]:
    """Real AITasksService backed by a deterministic fake LLM provider."""
    llm = FakeLLMProvider()

    def ai_tasks_service() -> AITasksService:
        return AITasksService(llm, get_settings())

    app.dependency_overrides[get_ai_tasks_service] = ai_tasks_service

    with TestClient(app) as client:
        yield AITasksHarness(client=client, llm=llm)

    app.dependency_overrides.clear()
