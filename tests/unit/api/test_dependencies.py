from unittest.mock import MagicMock, patch

import pytest

from app.domains.ai_tasks.api import dependencies as ai_deps
from app.domains.ai_tasks.application.services import AITasksService
from app.domains.documents.api import dependencies as doc_deps
from app.domains.documents.application.services import (
    AnswerQuestionService,
    IngestDocumentService,
    SearchDocumentsService,
)


@pytest.fixture(autouse=True)
def clear_caches():
    doc_deps.get_vector_store.cache_clear()
    doc_deps.get_embedding_provider.cache_clear()
    doc_deps.get_answer_generator.cache_clear()
    ai_deps.get_ai_tasks_service.cache_clear()
    yield
    doc_deps.get_vector_store.cache_clear()
    doc_deps.get_embedding_provider.cache_clear()
    doc_deps.get_answer_generator.cache_clear()
    ai_deps.get_ai_tasks_service.cache_clear()


class TestDocumentSingletons:
    def test_vector_store_is_cached_singleton(self):
        with patch.object(doc_deps, "QdrantVectorStore", return_value=MagicMock()):
            assert doc_deps.get_vector_store() is doc_deps.get_vector_store()

    def test_embedding_provider_is_cached_singleton(self):
        assert doc_deps.get_embedding_provider() is doc_deps.get_embedding_provider()

    def test_answer_generator_is_cached_singleton(self):
        assert doc_deps.get_answer_generator() is doc_deps.get_answer_generator()


class TestDocumentServices:
    def test_ingest_service_composed_with_settings(self):
        with patch.object(doc_deps, "QdrantVectorStore", return_value=MagicMock()):
            service = doc_deps.get_ingest_document_service()
        assert isinstance(service, IngestDocumentService)

    def test_search_service_composed(self):
        with patch.object(doc_deps, "QdrantVectorStore", return_value=MagicMock()):
            service = doc_deps.get_search_documents_service()
        assert isinstance(service, SearchDocumentsService)

    def test_answer_service_composed(self):
        with patch.object(doc_deps, "QdrantVectorStore", return_value=MagicMock()):
            service = doc_deps.get_answer_question_service()
        assert isinstance(service, AnswerQuestionService)


class TestAITasksDependency:
    def test_service_is_cached_singleton(self):
        first = ai_deps.get_ai_tasks_service()
        second = ai_deps.get_ai_tasks_service()
        assert first is second
        assert isinstance(first, AITasksService)
