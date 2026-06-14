"""Request and response DTOs for the AI tasks HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Incoming request body for POST /ask."""

    question: str = Field(..., min_length=1, description="User question text")


class AskResponse(BaseModel):
    """Structured API response returned to clients."""

    answer: str
    model: str
    tokens_used: int


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


class AnalyzeTextRequest(BaseModel):
    """Incoming request body for POST /analyze-text."""

    text: str = Field(..., min_length=1, description="Text to analyze")


class AnalyzeTextResponse(BaseModel):
    """Combined NLP response for the analyze endpoint."""

    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    keywords: list[str]
    language: str
    model: str
    tokens_used: int
    prompt_version: str


class ToolAssistantRequest(BaseModel):
    """Incoming request for the tool-calling assistant."""

    message: str = Field(..., min_length=1, description="User request")


class ToolExecutionResult(BaseModel):
    """One backend tool call made during a tool-assistant run."""

    name: str
    arguments: dict
    result: dict
    tool_call_id: str | None = None
    risk: str = "unknown"
    requires_approval: bool = False
    approved: bool = True
    status: str = "executed"
    error: str | None = None


class ToolAssistantResponse(BaseModel):
    """Final answer plus the tool calls used to produce it."""

    answer: str
    tool_calls: list[ToolExecutionResult]
    model: str
    tokens_used: int
    prompt_version: str


class ResearchAgentRequest(BaseModel):
    """Incoming request for the multistep research agent."""

    topic: str = Field(..., min_length=1, description="Topic to research")
    max_iterations: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional override for the agent loop iteration cap",
    )


class AgentIterationResult(BaseModel):
    """One step of the agent loop: the model's thought plus any tool calls."""

    iteration: int
    thought: str = ""
    tool_calls: list[ToolExecutionResult] = Field(default_factory=list)


class ResearchAgentResponse(BaseModel):
    """Final report plus a full, auditable trace of how it was produced."""

    topic: str
    report: str
    plan: list[str] = Field(default_factory=list)
    iterations: list[AgentIterationResult] = Field(default_factory=list)
    tool_calls: list[ToolExecutionResult] = Field(default_factory=list)
    critique: str | None = None
    stop_reason: str
    iterations_used: int
    model: str
    tokens_used: int
    prompt_version: str
