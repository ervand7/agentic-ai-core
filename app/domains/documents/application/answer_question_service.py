import logging
import time
from typing import Optional

from app.domains.documents.application.exceptions import EmptyVectorStoreError
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
