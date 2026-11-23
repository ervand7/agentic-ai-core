import logging
import time
from typing import Optional

from app.domains.documents.api.schemas import DocumentSearchResponse, SearchResult
from app.domains.documents.application.exceptions import EmptyVectorStoreError
from app.domains.documents.domain.ports import EmbeddingProvider, VectorStore

logger = logging.getLogger(__name__)


class SearchDocumentsService:
    """Use case: run semantic search over the ingested chunks."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self._embeddings = embedding_provider
        self._store = vector_store

    async def execute(
        self,
        *,
        query: str,
        top_k: int,
        request_id: str,
        filename: Optional[str] = None,
        keyword: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> DocumentSearchResponse:
        if self._store.count() == 0:
            raise EmptyVectorStoreError(
                "No documents uploaded yet. Upload a .txt file first."
            )

        start_time = time.perf_counter()
        query_embedding = await self._embeddings.embed(query, request_id)
        hits = self._store.search(
            query_embedding,
            top_k=top_k,
            filename_filter=filename,
            keyword=keyword,
            min_similarity=min_similarity,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            (
                "semantic_search_completed request_id=%s query_length=%s top_k=%s "
                "filename_filter=%s keyword=%s min_similarity=%s latency_ms=%.2f"
            ),
            request_id,
            len(query),
            top_k,
            filename,
            keyword,
            min_similarity,
            latency_ms,
        )

        return DocumentSearchResponse(
            query=query,
            results=[
                SearchResult(
                    text=hit.chunk.text,
                    filename=hit.chunk.filename,
                    similarity=round(hit.similarity, 4),
                )
                for hit in hits
            ],
        )
