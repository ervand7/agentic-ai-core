"""Domain models (entities and value objects) for the documents context."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredChunk:
    """One ingested text chunk plus its embedding vector."""

    text: str
    embedding: list[float]
    filename: str


@dataclass(frozen=True)
class SearchHit:
    """A scored chunk returned from a similarity search."""

    chunk: StoredChunk
    similarity: float
