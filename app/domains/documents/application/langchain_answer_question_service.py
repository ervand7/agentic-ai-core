import logging
import time
from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from pydantic import SkipValidation

from app.domains.documents.application.exceptions import EmptyVectorStoreError
from app.domains.documents.domain.models import RagAnswer, SearchHit, StoredChunk
from app.domains.documents.domain.ports import EmbeddingProvider, VectorStore
from app.domains.documents.domain.rag import (
    NO_CONTEXT_ANSWER,
    build_citations,
    build_user_prompt,
)
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# Variables are filled at invoke time, so braces inside document text are safe.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "{user_prompt}"),
    ]
)


class PortsRetriever(BaseRetriever):
    """LangChain retriever backed by the documents context's own ports."""

    # SkipValidation: ports are Protocols, which pydantic cannot isinstance-check.
    embedding_provider: SkipValidation[EmbeddingProvider]
    vector_store: SkipValidation[VectorStore]
    request_id: str
    top_k: int
    filename: Optional[str] = None
    min_similarity: Optional[float] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("PortsRetriever is async-only; use ainvoke().")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = await self.embedding_provider.embed(query, self.request_id)
        hits = self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
            filename_filter=self.filename,
            min_similarity=self.min_similarity,
        )
        return [
            Document(
                page_content=hit.chunk.text,
                metadata={"filename": hit.chunk.filename, "similarity": hit.similarity},
            )
            for hit in hits
        ]


def documents_to_hits(documents: list[Document]) -> list[SearchHit]:
    """Map LangChain documents back onto domain hits to reuse the RAG helpers."""
    return [
        SearchHit(
            chunk=StoredChunk(
                text=document.page_content,
                embedding=[],
                filename=document.metadata["filename"],
            ),
            similarity=document.metadata["similarity"],
        )
        for document in documents
    ]


class LangChainAnswerQuestionService:
    """
    Use case: the same RAG as `AnswerQuestionService`, orchestrated with LangChain.

    Same contract (abstain when nothing is retrieved, numbered citations, same
    prompts), rebuilt from LangChain building blocks for side-by-side comparison:
    `PortsRetriever` -> `ChatPromptTemplate` -> chat model, composed as an LCEL
    chain (`prompt | llm`).
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chat_model: BaseChatModel,
        system_prompt: str,
        top_k: int,
        min_similarity: float,
    ):
        self._embeddings = embedding_provider
        self._store = vector_store
        self._system_prompt = system_prompt
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._chain = RAG_PROMPT | chat_model

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
        retriever = PortsRetriever(
            embedding_provider=self._embeddings,
            vector_store=self._store,
            request_id=request_id,
            top_k=effective_top_k,
            filename=filename,
            min_similarity=effective_min_sim,
        )
        hits = documents_to_hits(await retriever.ainvoke(question))

        # No relevant context -> abstain instead of calling the LLM.
        if not hits:
            logger.info(
                "langchain_rag_no_context request_id=%s question_length=%s top_k=%s min_similarity=%s",
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

        message = await self._invoke_chain(
            system_prompt=self._system_prompt,
            user_prompt=build_user_prompt(question, hits),
        )
        model = message.response_metadata.get("model_name")
        tokens_used = (
            message.usage_metadata["total_tokens"] if message.usage_metadata else 0
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            (
                "langchain_rag_answer_generated request_id=%s question_length=%s "
                "context_chunks=%s model=%s tokens_used=%s latency_ms=%.2f"
            ),
            request_id,
            len(question),
            len(hits),
            model,
            tokens_used,
            latency_ms,
        )

        return RagAnswer(
            answer=message.text,
            citations=build_citations(hits),
            used_context=True,
            model=model,
            tokens_used=tokens_used,
        )

    async def _invoke_chain(self, *, system_prompt: str, user_prompt: str) -> AIMessage:
        """Run the LCEL chain, mapping OpenAI SDK errors onto the shared LLM errors."""
        try:
            message = await self._chain.ainvoke(
                {"system_prompt": system_prompt, "user_prompt": user_prompt}
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "The AI provider timed out. Please try again shortly."
            ) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(
                "Rate limit reached. Please wait a moment and retry."
            ) from exc
        except (APIConnectionError, InternalServerError) as exc:
            raise LLMTemporaryError(
                "Temporary AI provider issue. Please retry shortly."
            ) from exc
        except OpenAIError as exc:
            raise LLMServiceError("AI provider returned an unexpected error.") from exc

        if not isinstance(message, AIMessage):
            raise LLMServiceError("AI provider returned an unexpected error.")
        return message
