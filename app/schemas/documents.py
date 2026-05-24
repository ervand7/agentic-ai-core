"""Schemas for semantic document upload and search."""

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


class SearchResult(BaseModel):
    """Single semantic search hit."""

    text: str
    similarity: float


class DocumentSearchResponse(BaseModel):
    """Semantic search response format."""

    query: str
    results: list[SearchResult]
