"""
Ports (abstractions) for the documents context.

Application services depend on these protocols so the domain stays
free of HTTP, database, or provider-specific code.
"""

from typing import Protocol

from app.domains.documents.domain.models import SearchHit, StoredChunk


class VectorStore(Protocol):
    """Persistence port for ingested document chunks."""

    def add_chunks(
        self,
        *,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk text and matching embedding vectors."""

    def count(self) -> int:
        """Return the number of stored chunks."""

    def all(self) -> list[StoredChunk]:
        """Return every stored chunk (used by similarity search)."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        """Return the top_k chunks most similar to the query embedding."""


class EmbeddingProvider(Protocol):
    """Abstraction over an embedding provider."""

    async def embed(self, text: str, request_id: str) -> list[float]:
        """Generate one embedding vector for `text`."""
