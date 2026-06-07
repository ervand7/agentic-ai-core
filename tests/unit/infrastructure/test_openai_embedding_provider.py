"""Unit tests for OpenAIEmbeddingProvider (OpenAIClient mocked)."""

from unittest.mock import AsyncMock

from app.domains.documents.infrastructure.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)


async def test_returns_vector_from_client():
    client = AsyncMock()
    client.create_embedding.return_value = [0.1, 0.2, 0.3]
    provider = OpenAIEmbeddingProvider(client)

    vector = await provider.embed("some text", "req-1")
    assert vector == [0.1, 0.2, 0.3]


async def test_forwards_endpoint_request_id_and_text():
    client = AsyncMock()
    client.create_embedding.return_value = [0.0]
    provider = OpenAIEmbeddingProvider(client)

    await provider.embed("hello world", "req-7")

    kwargs = client.create_embedding.call_args.kwargs
    assert kwargs["endpoint"] == "embeddings"
    assert kwargs["request_id"] == "req-7"
    assert kwargs["text"] == "hello world"
