import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.domains.documents.api.dependencies import (
    get_answer_question_service,
    get_ingest_document_service,
)
from app.domains.documents.application.services import (
    AnswerQuestionService,
    IngestDocumentService,
)
from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)
from app.domains.documents.infrastructure.openai_answer_generator import (
    OpenAIAnswerGenerator,
)
from app.domains.documents.infrastructure.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from app.main import app
from app.shared.config import get_settings
from app.shared.infrastructure.openai_client import OpenAIClient, get_openai_client

_KEY = os.getenv("OPENAI_API_KEY", "")
LIVE_READY = os.getenv("RUN_LIVE_OPENAI") == "1" and _KEY and _KEY != "test-key"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_READY,
        reason="set RUN_LIVE_OPENAI=1 and a real OPENAI_API_KEY to run live tests",
    ),
]


@pytest.fixture
def live_documents_client() -> Iterator[TestClient]:
    """Documents RAG wired to the real OpenAI ports + in-memory store."""
    # Rebuild cached singletons so the real API key in the environment is used.
    get_settings.cache_clear()
    get_openai_client.cache_clear()

    settings = get_settings()
    client = OpenAIClient(settings)
    store = InMemoryVectorStore()
    embeddings = OpenAIEmbeddingProvider(client)
    answer_generator = OpenAIAnswerGenerator(client)

    def ingest_service() -> IngestDocumentService:
        return IngestDocumentService(
            embedding_provider=embeddings,
            vector_store=store,
            chunk_size=settings.DOCUMENT_CHUNK_SIZE,
            chunk_overlap=settings.DOCUMENT_CHUNK_OVERLAP,
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
    app.dependency_overrides[get_answer_question_service] = answer_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_live_rag_round_trip(live_documents_client):
    upload = live_documents_client.post(
        "/documents/upload",
        files={
            "file": (
                "facts.txt",
                b"The Eiffel Tower is a wrought-iron tower in Paris, France. "
                b"It was completed in 1889 and is 330 metres tall.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200

    ask = live_documents_client.post(
        "/documents/ask",
        json={"question": "In which city is the Eiffel Tower?"},
    )
    assert ask.status_code == 200
    body = ask.json()
    assert body["used_context"] is True
    assert body["answer"].strip()
    assert "paris" in body["answer"].lower()
    assert body["citations"]


def test_live_ai_task_summarize():
    """Uses the default DI wiring (real OpenAI LLM provider)."""
    get_settings.cache_clear()
    get_openai_client.cache_clear()
    with TestClient(app) as client:
        resp = client.post(
            "/summarize",
            json={
                "text": "FastAPI is a modern Python web framework for building APIs."
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"].strip()
    assert body["tokens_used"] > 0
