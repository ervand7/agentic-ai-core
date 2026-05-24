"""
In-memory implementation of the VectorStore port.

Simple process-memory storage suitable for demos and tests.
Replace with a real vector database in production.
"""

import logging

from app.domains.documents.domain.chunking import cosine_similarity
from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.domain.ports import VectorStore

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """Simple process-memory vector store (no database yet)."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def add_chunks(
        self,
        *,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk text and vectors together for later search."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk_text, embedding in zip(chunks, embeddings):
            self._chunks.append(
                StoredChunk(text=chunk_text, embedding=embedding, filename=filename)
            )

    def count(self) -> int:
        """Return number of stored chunks."""
        return len(self._chunks)

    def all(self) -> list[StoredChunk]:
        return list(self._chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        """Score every stored chunk using cosine similarity and return top hits."""
        scored: list[SearchHit] = []
        for chunk in self._chunks:
            similarity = cosine_similarity(query_embedding, chunk.embedding)
            logger.info(
                "semantic_similarity_score filename=%s similarity=%.4f text_preview=%s",
                chunk.filename,
                similarity,
                chunk.text[:80].replace("\n", " "),
            )
            scored.append(SearchHit(chunk=chunk, similarity=similarity))

        scored.sort(key=lambda hit: hit.similarity, reverse=True)
        return scored[:top_k]
