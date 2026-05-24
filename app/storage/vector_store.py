"""In-memory vector storage for beginner-friendly semantic search demos."""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    """One text chunk plus its embedding vector."""

    text: str
    embedding: list[float]
    filename: str


class InMemoryVectorStore:
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

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[StoredChunk, float]]:
        """Score every stored chunk using cosine similarity and return top hits."""
        scored: list[tuple[StoredChunk, float]] = []
        for chunk in self._chunks:
            similarity = cosine_similarity(query_embedding, chunk.embedding)
            logger.info(
                "semantic_similarity_score filename=%s similarity=%.4f text_preview=%s",
                chunk.filename,
                similarity,
                chunk.text[:80].replace("\n", " "),
            )
            scored.append((chunk, similarity))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity manually (no black-box similarity library).

    Mathematical idea:
    cos(theta) = (A · B) / (||A|| * ||B||)

    - A · B is the dot product (sum of pairwise multiplications)
    - ||A|| is the Euclidean norm (square root of sum of squares)
    - Result range is [-1, 1], where values closer to 1 mean vectors point
      in nearly the same direction, which usually means similar semantic meaning.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        dot_product += value_a * value_b
        norm_a_squared += value_a * value_a
        norm_b_squared += value_b * value_b

    norm_a = math.sqrt(norm_a_squared)
    norm_b = math.sqrt(norm_b_squared)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


@lru_cache
def get_vector_store() -> InMemoryVectorStore:
    """Singleton-like accessor for shared in-memory storage."""
    return InMemoryVectorStore()
