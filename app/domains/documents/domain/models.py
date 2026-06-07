"""Domain models (entities and value objects) for the documents context."""

from dataclasses import dataclass
from typing import Optional


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


@dataclass(frozen=True)
class GeneratedAnswer:
    """Provider-agnostic result of a single answer-generation call."""

    content: str
    model: str
    tokens_used: int


@dataclass(frozen=True)
class Citation:
    """A retrieved chunk offered as a source for a RAG answer.

    `index` is the 1-based number the LLM is asked to cite (e.g. `[1]`).
    """

    index: int
    filename: str
    text: str
    similarity: float


@dataclass(frozen=True)
class RagAnswer:
    """Result of a Retrieval-Augmented Generation answer.

    `used_context` is False when retrieval found nothing relevant, in which
    case we abstain ("I don't know") instead of calling the LLM.
    """

    answer: str
    citations: list[Citation]
    used_context: bool
    model: Optional[str]
    tokens_used: int
