"""FastAPI dependencies wiring concrete adapters into the documents API."""

from functools import lru_cache

from app.domains.documents.application.services import (
    IngestDocumentService,
    SearchDocumentsService,
)
from app.domains.documents.domain.ports import EmbeddingProvider, VectorStore
from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)
from app.domains.documents.infrastructure.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from app.shared.config import get_settings
from app.shared.infrastructure.openai_client import get_openai_client


@lru_cache
def get_vector_store() -> VectorStore:
    """Singleton in-memory vector store shared by ingest and search."""
    return InMemoryVectorStore()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Singleton embedding provider backed by the shared OpenAI client."""
    return OpenAIEmbeddingProvider(get_openai_client())


def get_ingest_document_service() -> IngestDocumentService:
    """Compose the ingest use case with its dependencies."""
    settings = get_settings()
    return IngestDocumentService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        chunk_size=settings.DOCUMENT_CHUNK_SIZE,
        chunk_overlap=settings.DOCUMENT_CHUNK_OVERLAP,
    )


def get_search_documents_service() -> SearchDocumentsService:
    """Compose the search use case with its dependencies."""
    return SearchDocumentsService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
    )
