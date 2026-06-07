"""Unit tests for AnswerQuestionService (the RAG use case) with mocked ports."""

from typing import Optional

import pytest

from app.domains.documents.application.services import (
    AnswerQuestionService,
    EmptyVectorStoreError,
)
from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.domain.ports import (
    AnswerGenerator,
    EmbeddingProvider,
    VectorStore,
)
from app.domains.documents.domain.rag import NO_CONTEXT_ANSWER
from tests.conftest import (
    async_mock_method,
    make_answer_generator_mock,
    make_embedding_mock,
    make_vector_store_mock,
    mock_method,
)


def _service(
    *,
    embeddings: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    generator: Optional[AnswerGenerator] = None,
    system_prompt: str = "SYSTEM",
    top_k: int = 4,
    min_similarity: float = 0.2,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> AnswerQuestionService:
    return AnswerQuestionService(
        embedding_provider=embeddings or make_embedding_mock(),
        vector_store=store or make_vector_store_mock(),
        answer_generator=generator or make_answer_generator_mock(),
        system_prompt=system_prompt,
        top_k=top_k,
        min_similarity=min_similarity,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _hit(text="ctx", filename="a.txt", similarity=0.8):
    return SearchHit(
        chunk=StoredChunk(text=text, embedding=[0.1], filename=filename),
        similarity=similarity,
    )


async def test_empty_store_raises():
    store = make_vector_store_mock(count=0)
    service = _service(store=store)
    with pytest.raises(EmptyVectorStoreError, match="No documents uploaded"):
        await service.execute(question="q?", request_id="r")


class TestNoContext:
    async def test_no_hits_abstains_without_calling_llm(self):
        store = make_vector_store_mock(count=5, search_results=[])
        generator = make_answer_generator_mock()
        service = _service(store=store, generator=generator)

        result = await service.execute(question="q?", request_id="r")

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.used_context is False
        assert result.citations == []
        assert result.model is None
        assert result.tokens_used == 0
        async_mock_method(generator, "generate").assert_not_awaited()


class TestWithContext:
    async def test_generates_answer_and_citations(self):
        store = make_vector_store_mock(
            count=5, search_results=[_hit("ctx text", "doc.txt", 0.77)]
        )
        generator = make_answer_generator_mock(
            content="Answer [1].", model="m", tokens_used=12
        )
        service = _service(store=store, generator=generator)

        result = await service.execute(question="What?", request_id="r")

        assert result.answer == "Answer [1]."
        assert result.used_context is True
        assert result.model == "m"
        assert result.tokens_used == 12
        assert len(result.citations) == 1
        assert result.citations[0].filename == "doc.txt"
        assert result.citations[0].index == 1

    async def test_generator_receives_prompt_and_tuning_params(self):
        store = make_vector_store_mock(
            count=1, search_results=[_hit("snippet body", "f.txt", 0.9)]
        )
        generator = make_answer_generator_mock()
        service = _service(
            store=store,
            generator=generator,
            system_prompt="MY SYSTEM",
            temperature=0.42,
            max_tokens=123,
        )

        await service.execute(question="Why?", request_id="req-7")

        generate = async_mock_method(generator, "generate")
        generate.assert_awaited_once()
        call_kwargs = generate.await_args.kwargs
        assert call_kwargs["request_id"] == "req-7"
        assert call_kwargs["system_prompt"] == "MY SYSTEM"
        assert call_kwargs["temperature"] == 0.42
        assert call_kwargs["max_tokens"] == 123
        assert "snippet body" in call_kwargs["user_prompt"]
        assert "Question: Why?" in call_kwargs["user_prompt"]


class TestOverrides:
    async def test_request_top_k_and_min_similarity_override_defaults(self):
        store = make_vector_store_mock(count=1, search_results=[])
        service = _service(store=store, top_k=4, min_similarity=0.2)

        await service.execute(
            question="q?", request_id="r", top_k=9, min_similarity=0.55
        )

        kwargs = mock_method(store, "search").call_args.kwargs
        assert kwargs["top_k"] == 9
        assert kwargs["min_similarity"] == 0.55

    async def test_defaults_used_when_not_provided(self):
        store = make_vector_store_mock(count=1, search_results=[])
        service = _service(store=store, top_k=3, min_similarity=0.15)

        await service.execute(question="q?", request_id="r")

        kwargs = mock_method(store, "search").call_args.kwargs
        assert kwargs["top_k"] == 3
        assert kwargs["min_similarity"] == 0.15

    async def test_filename_filter_forwarded(self):
        store = make_vector_store_mock(count=1, search_results=[])
        service = _service(store=store)

        await service.execute(question="q?", request_id="r", filename="only.txt")
        assert (
            mock_method(store, "search").call_args.kwargs["filename_filter"]
            == "only.txt"
        )

    async def test_zero_is_a_valid_explicit_min_similarity(self):
        store = make_vector_store_mock(count=1, search_results=[])
        service = _service(store=store, min_similarity=0.2)

        await service.execute(question="q?", request_id="r", min_similarity=0.0)
        assert mock_method(store, "search").call_args.kwargs["min_similarity"] == 0.0
