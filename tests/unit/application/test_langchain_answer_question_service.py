"""Unit tests for LangChainAnswerQuestionService with mocked ports and a fake chat model."""

from typing import Optional

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from openai import APITimeoutError

from app.domains.documents.application.exceptions import EmptyVectorStoreError
from app.domains.documents.application.langchain_answer_question_service import (
    LangChainAnswerQuestionService,
)
from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.domain.ports import EmbeddingProvider, VectorStore
from app.domains.documents.domain.rag import NO_CONTEXT_ANSWER
from app.shared.exceptions import LLMTimeoutError
from tests.conftest import (
    async_mock_method,
    make_embedding_mock,
    make_vector_store_mock,
    mock_method,
)


class RecordingChatModel(GenericFakeChatModel):
    """Fake chat model that also records the prompt messages it received."""

    received: list = []

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return await super()._agenerate(messages, stop, run_manager, **kwargs)


def _chat_model(content="Answer [1].", model="m", tokens_used=12) -> RecordingChatModel:
    message = AIMessage(
        content=content,
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": tokens_used,
            "output_tokens": 0,
            "total_tokens": tokens_used,
        },
    )
    return RecordingChatModel(messages=iter([message]), received=[])


def _service(
    *,
    embeddings: Optional[EmbeddingProvider] = None,
    store: Optional[VectorStore] = None,
    chat_model: Optional[BaseChatModel] = None,
    system_prompt: str = "SYSTEM",
    top_k: int = 4,
    min_similarity: float = 0.2,
) -> LangChainAnswerQuestionService:
    return LangChainAnswerQuestionService(
        embedding_provider=embeddings or make_embedding_mock(),
        vector_store=store or make_vector_store_mock(),
        chat_model=chat_model or _chat_model(),
        system_prompt=system_prompt,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def _hit(text="ctx", filename="a.txt", similarity=0.8):
    return SearchHit(
        chunk=StoredChunk(text=text, embedding=[0.1], filename=filename),
        similarity=similarity,
    )


async def test_empty_store_raises():
    service = _service(store=make_vector_store_mock(count=0))
    with pytest.raises(EmptyVectorStoreError, match="No documents uploaded"):
        await service.execute(question="q?", request_id="r")


async def test_no_hits_abstains_without_calling_llm():
    chat_model = _chat_model()
    service = _service(
        store=make_vector_store_mock(count=5, search_results=[]),
        chat_model=chat_model,
    )

    result = await service.execute(question="q?", request_id="r")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.used_context is False
    assert result.citations == []
    assert result.model is None
    assert result.tokens_used == 0
    assert chat_model.received == []


async def test_generates_answer_citations_and_usage():
    store = make_vector_store_mock(
        count=5, search_results=[_hit("ctx text", "doc.txt", 0.77)]
    )
    service = _service(store=store)

    result = await service.execute(question="What?", request_id="r")

    assert result.answer == "Answer [1]."
    assert result.used_context is True
    assert result.model == "m"
    assert result.tokens_used == 12
    assert len(result.citations) == 1
    assert result.citations[0].filename == "doc.txt"
    assert result.citations[0].index == 1
    assert result.citations[0].similarity == 0.77


async def test_prompt_contains_system_prompt_context_and_question():
    store = make_vector_store_mock(
        count=1, search_results=[_hit("snippet {body}", "f.txt", 0.9)]
    )
    chat_model = _chat_model()
    service = _service(store=store, chat_model=chat_model, system_prompt="MY SYSTEM")

    await service.execute(question="Why?", request_id="req-7")

    system_message, human_message = chat_model.received[0]
    assert system_message.content == "MY SYSTEM"
    assert "[1] (source: f.txt)\nsnippet {body}" in human_message.content
    assert "Question: Why?" in human_message.content


async def test_retriever_uses_ports_with_overrides():
    embeddings = make_embedding_mock()
    store = make_vector_store_mock(count=1, search_results=[])
    service = _service(embeddings=embeddings, store=store, top_k=4, min_similarity=0.2)

    await service.execute(
        question="q?",
        request_id="req-1",
        top_k=9,
        filename="only.txt",
        min_similarity=0.0,
    )

    async_mock_method(embeddings, "embed").assert_awaited_once_with("q?", "req-1")
    kwargs = mock_method(store, "search").call_args.kwargs
    assert kwargs["top_k"] == 9
    assert kwargs["filename_filter"] == "only.txt"
    assert kwargs["min_similarity"] == 0.0


async def test_defaults_used_when_not_provided():
    store = make_vector_store_mock(count=1, search_results=[])
    service = _service(store=store, top_k=3, min_similarity=0.15)

    await service.execute(question="q?", request_id="r")

    kwargs = mock_method(store, "search").call_args.kwargs
    assert kwargs["top_k"] == 3
    assert kwargs["min_similarity"] == 0.15


async def test_openai_timeout_maps_to_llm_timeout_error():
    class TimeoutChatModel(GenericFakeChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            raise APITimeoutError(
                request=httpx.Request("POST", "https://api.openai.com")
            )

    service = _service(
        store=make_vector_store_mock(count=1, search_results=[_hit()]),
        chat_model=TimeoutChatModel(messages=iter([])),
    )

    with pytest.raises(LLMTimeoutError):
        await service.execute(question="q?", request_id="r")
