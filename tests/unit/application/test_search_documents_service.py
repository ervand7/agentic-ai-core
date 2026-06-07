"""Unit tests for SearchDocumentsService (mocked ports)."""

import pytest

from app.domains.documents.application.services import (
    EmptyVectorStoreError,
    SearchDocumentsService,
)
from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)
from tests.conftest import (
    async_mock_method,
    make_embedding_mock,
    make_vector_store_mock,
    mock_method,
)


def _service(embeddings, store):
    return SearchDocumentsService(embedding_provider=embeddings, vector_store=store)


async def test_empty_store_raises():
    service = _service(make_embedding_mock(), make_vector_store_mock(count=0))
    with pytest.raises(EmptyVectorStoreError, match="No documents uploaded"):
        await service.execute(query="hi", top_k=3, request_id="r")


async def test_returns_results_with_rounded_similarity():
    store = make_vector_store_mock(count=1)
    mock_method(store, "search").return_value = [
        SearchHit(
            chunk=StoredChunk(text="alpha", embedding=[0.1], filename="a.txt"),
            similarity=0.123456789,
        )
    ]
    embeddings = make_embedding_mock(default_vector=[1.0, 0.0])
    service = _service(embeddings, store)

    response = await service.execute(query="alpha?", top_k=3, request_id="r")

    assert response.query == "alpha?"
    assert len(response.results) == 1
    assert response.results[0].text == "alpha"
    assert response.results[0].filename == "a.txt"
    assert response.results[0].similarity == 0.1235


async def test_query_is_embedded_then_passed_to_store():
    store = make_vector_store_mock(count=5)
    search = mock_method(store, "search")
    search.return_value = []
    embeddings = make_embedding_mock(default_vector=[0.2, 0.4])
    service = _service(embeddings, store)

    await service.execute(
        query="my query",
        top_k=7,
        request_id="req-42",
        filename="only.txt",
        keyword="kw",
        min_similarity=0.3,
    )

    async_mock_method(embeddings, "embed").assert_awaited_once_with(
        "my query", "req-42"
    )
    search.assert_called_once()
    args, kwargs = search.call_args
    assert args[0] == [0.2, 0.4]
    assert kwargs["top_k"] == 7
    assert kwargs["filename_filter"] == "only.txt"
    assert kwargs["keyword"] == "kw"
    assert kwargs["min_similarity"] == 0.3


async def test_no_hits_returns_empty_results():
    store = make_vector_store_mock(count=3)
    mock_method(store, "search").return_value = []
    service = _service(make_embedding_mock(), store)

    response = await service.execute(query="nothing", top_k=3, request_id="r")
    assert response.results == []


async def test_end_to_end_with_in_memory_store():
    store = InMemoryVectorStore()
    store.add_chunks(
        filename="a.txt",
        chunks=["cats", "dogs"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    embeddings = make_embedding_mock(default_vector=[1.0, 0.0])
    service = _service(embeddings, store)

    response = await service.execute(query="feline", top_k=1, request_id="r")
    assert len(response.results) == 1
    assert response.results[0].text == "cats"
