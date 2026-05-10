"""Request/response models for the ask endpoint."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Incoming request body for POST /ask."""

    question: str = Field(..., min_length=1, description="User question text")


class AskResponse(BaseModel):
    """Structured API response returned to clients."""

    answer: str
    model: str
    tokens_used: int
