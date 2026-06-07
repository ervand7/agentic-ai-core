"""
Application services (use cases) for the documents bounded context.

Use cases orchestrate:
- pure domain operations (chunking, similarity)
- infrastructure ports (vector store, embedding provider)

There are two errors that are domain-meaningful and worth surfacing
explicitly so the API layer can map them to HTTP without leaking
infrastructure details.
"""

import logging
import time
from typing import Optional

from app.domains.documents.api.schemas import (
    DocumentSearchResponse,
    DocumentUploadResponse,
    SearchResult,
)
from app.domains.documents.domain.chunking import chunk_text
from app.domains.documents.domain.models import RagAnswer
from app.domains.documents.domain.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    VectorStore,
)
from app.domains.documents.domain.rag import (
    NO_CONTEXT_ANSWER,
    build_citations,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


class DocumentValidationError(Exception):
    """Raised when an uploaded document fails domain-level validation."""


class EmptyVectorStoreError(Exception):
    """Raised when a search is attempted before any document has been uploaded."""


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


class AnswerQuestionService:
    """
    Use case: Retrieval-Augmented Generation (RAG) over the ingested chunks.

    Pipeline: embed the question -> retrieve relevant chunks -> inject them as
    context -> ask the LLM to answer using only that context and cite sources.
    When nothing relevant is retrieved we abstain ("I don't know") instead of
    letting the model guess.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        system_prompt: str,
        top_k: int,
        min_similarity: float,
        temperature: float,
        max_tokens: int,
    ):
        self._embeddings = embedding_provider
        self._store = vector_store
        self._generator = answer_generator
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def execute(
        self,
        *,
        question: str,
        request_id: str,
        top_k: Optional[int] = None,
        filename: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> RagAnswer:
        if self._store.count() == 0:
            raise EmptyVectorStoreError(
                "No documents uploaded yet. Upload a document first."
            )

        effective_top_k = top_k if top_k is not None else self._top_k
        effective_min_sim = (
            min_similarity if min_similarity is not None else self._min_similarity
        )

        start_time = time.perf_counter()
        query_embedding = await self._embeddings.embed(question, request_id)
        hits = self._store.search(
            query_embedding,
            top_k=effective_top_k,
            filename_filter=filename,
            min_similarity=effective_min_sim,
        )

        # No relevant context -> abstain instead of calling the LLM.
        if not hits:
            logger.info(
                "rag_no_context request_id=%s question_length=%s top_k=%s min_similarity=%s",
                request_id,
                len(question),
                effective_top_k,
                effective_min_sim,
            )
            return RagAnswer(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                used_context=False,
                model=None,
                tokens_used=0,
            )

        user_prompt = build_user_prompt(question, hits)
        generated = await self._generator.generate(
            request_id=request_id,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            (
                "rag_answer_generated request_id=%s question_length=%s context_chunks=%s "
                "model=%s tokens_used=%s latency_ms=%.2f"
            ),
            request_id,
            len(question),
            len(hits),
            generated.model,
            generated.tokens_used,
            latency_ms,
        )

        return RagAnswer(
            answer=generated.content,
            citations=build_citations(hits),
            used_context=True,
            model=generated.model,
            tokens_used=generated.tokens_used,
        )
