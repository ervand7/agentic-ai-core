"""
Ports (abstractions) for the documents context.

Application services depend on these protocols so the domain stays
free of HTTP, database, or provider-specific code.
"""

from typing import Optional, Protocol

from app.domains.documents.domain.models import GeneratedAnswer, SearchHit, StoredChunk


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
        filename_filter: Optional[str] = None,
        keyword: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> list[SearchHit]:
        """Return the top_k chunks most similar to the query embedding."""


class EmbeddingProvider(Protocol):
    """Abstraction over an embedding provider."""

    async def embed(self, text: str, request_id: str) -> list[float]:
        """Generate one embedding vector for `text`."""


class AnswerGenerator(Protocol):
    """Abstraction over an LLM used to generate grounded RAG answers.

    Kept separate from the embedding provider so the RAG use case depends only
    on the capability it needs (turning a prompt into an answer), not on a
    concrete provider.
    """

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeneratedAnswer:
        """Generate one answer from a system + user prompt."""
