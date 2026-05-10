"""Request/response models for additional AI endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Incoming request for sentiment classification."""

    text: str = Field(..., min_length=1, description="Text to classify")


class ClassifyResponse(BaseModel):
    """Strict response format for /classify."""

    sentiment: Literal["positive", "negative", "neutral"]
    summary: str
    keywords: list[str]


class SummarizeRequest(BaseModel):
    """Incoming request for summarization."""

    text: str = Field(..., min_length=1, description="Text to summarize")


class SummarizeResponse(BaseModel):
    """Response for /summarize."""

    summary: str
    model: str
    tokens_used: int


class ExtractKeywordsRequest(BaseModel):
    """Incoming request for keyword extraction."""

    text: str = Field(..., min_length=1, description="Text to analyze")


class ExtractKeywordsResponse(BaseModel):
    """Response for /extract-keywords."""

    keywords: list[str]
    model: str
    tokens_used: int


class TranslateRequest(BaseModel):
    """Incoming request for translation."""

    text: str = Field(..., min_length=1, description="Text to translate")
    target_language: str = Field(
        ..., min_length=2, description="Target language (e.g., Spanish)"
    )


class TranslateResponse(BaseModel):
    """Response for /translate."""

    translation: str
    model: str
    tokens_used: int
