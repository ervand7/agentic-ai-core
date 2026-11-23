import logging

from app.domains.documents.api.schemas import DocumentUploadResponse
from app.domains.documents.application.exceptions import DocumentValidationError
from app.domains.documents.domain.chunking import chunk_text
from app.domains.documents.domain.ports import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)


class IngestDocumentService:
    """Use case: ingest a `.txt` document into the vector store."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._embeddings = embedding_provider
        self._store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def execute(
        self,
        *,
        filename: str,
        content: str,
        request_id: str,
    ) -> DocumentUploadResponse:
        if not content.strip():
            raise DocumentValidationError("Uploaded file is empty.")

        chunks = chunk_text(
            content,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        logger.info(
            "document_chunking_completed request_id=%s filename=%s chunk_count=%s chunk_size=%s overlap=%s",
            request_id,
            filename,
            len(chunks),
            self._chunk_size,
            self._chunk_overlap,
        )

        embeddings: list[list[float]] = []
        for chunk in chunks:
            embeddings.append(await self._embeddings.embed(chunk, request_id))

        self._store.add_chunks(filename=filename, chunks=chunks, embeddings=embeddings)
        logger.info(
            "document_vectors_stored request_id=%s filename=%s chunks_stored=%s total_chunks_in_store=%s",
            request_id,
            filename,
            len(chunks),
            self._store.count(),
        )

        return DocumentUploadResponse(
            filename=filename,
            chunks_stored=len(chunks),
            total_characters=len(content),
        )
