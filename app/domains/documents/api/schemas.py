"""Request and response DTOs for the documents HTTP API."""

from typing import Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned after ingesting a text document."""

    filename: str
    chunks_stored: int
    total_characters: int


class DocumentSearchRequest(BaseModel):
    """Incoming request for semantic search."""

    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    filename: Optional[str] = Field(
        default=None,
        description="Optional exact filename filter (metadata filtering).",
    )
    keyword: Optional[str] = Field(
        default=None,
        description="Optional keyword query for hybrid search (vector + keyword).",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum vector similarity threshold (0..1).",
    )


class SearchResult(BaseModel):
    """Single semantic search hit."""

    text: str
    similarity: float


class DocumentSearchResponse(BaseModel):
    """Semantic search response format."""

    query: str
    results: list[SearchResult]
