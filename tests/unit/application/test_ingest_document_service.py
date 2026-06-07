"""Unit tests for IngestDocumentService (mocked ports)."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.domains.documents.application.services import (
    DocumentValidationError,
    IngestDocumentService,
)
from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)
from tests.conftest import async_mock_method, make_embedding_mock


def _service(embeddings, store, chunk_size=4, chunk_overlap=0):
    return IngestDocumentService(
        embedding_provider=embeddings,
        vector_store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


class TestIngestValidation:
    async def test_empty_content_raises(self):
        service = _service(make_embedding_mock(), InMemoryVectorStore())
        with pytest.raises(DocumentValidationError, match="empty"):
            await service.execute(filename="f.txt", content="", request_id="r1")

    async def test_whitespace_only_content_raises(self):
        service = _service(make_embedding_mock(), InMemoryVectorStore())
        with pytest.raises(DocumentValidationError):
            await service.execute(filename="f.txt", content="   \n ", request_id="r1")


class TestIngestHappyPath:
    async def test_stores_chunks_and_returns_response(self):
        embeddings = make_embedding_mock(default_vector=[1.0, 0.0])
        store = InMemoryVectorStore()
        service = _service(embeddings, store, chunk_size=4, chunk_overlap=0)

        response = await service.execute(
            filename="doc.txt", content="abcdefgh", request_id="req-1"
        )

        assert response.filename == "doc.txt"
        assert response.chunks_stored == 2
        assert response.total_characters == 8
        assert store.count() == 2

    async def test_embeds_each_chunk_once_with_request_id(self):
        embeddings = make_embedding_mock()
        store = InMemoryVectorStore()
        service = _service(embeddings, store, chunk_size=4, chunk_overlap=0)

        await service.execute(
            filename="doc.txt", content="aaaabbbb", request_id="req-9"
        )

        embed = async_mock_method(embeddings, "embed")
        assert embed.await_count == 2
        embed.assert_has_awaits([call("aaaa", "req-9"), call("bbbb", "req-9")])

    async def test_add_chunks_called_with_matching_lengths(self):
        embeddings = make_embedding_mock(default_vector=[0.5, 0.5])
        store = MagicMock()
        store.count.return_value = 2
        service = _service(embeddings, store, chunk_size=4, chunk_overlap=0)

        await service.execute(filename="doc.txt", content="abcdefgh", request_id="r")

        store.add_chunks.assert_called_once()
        kwargs = store.add_chunks.call_args.kwargs
        assert kwargs["filename"] == "doc.txt"
        assert len(kwargs["chunks"]) == len(kwargs["embeddings"]) == 2

    async def test_total_characters_counts_raw_content_not_chunks(self):
        embeddings = make_embedding_mock()
        store = InMemoryVectorStore()
        service = _service(embeddings, store, chunk_size=100, chunk_overlap=0)

        response = await service.execute(
            filename="doc.txt", content="  hello  ", request_id="r"
        )
        assert response.total_characters == len("  hello  ")
        assert response.chunks_stored == 1

    async def test_embed_failure_propagates(self):
        embeddings = AsyncMock()
        embeddings.embed.side_effect = RuntimeError("boom")
        store = InMemoryVectorStore()
        service = _service(embeddings, store, chunk_size=100, chunk_overlap=0)

        with pytest.raises(RuntimeError, match="boom"):
            await service.execute(filename="doc.txt", content="hello", request_id="r")
        assert store.count() == 0
