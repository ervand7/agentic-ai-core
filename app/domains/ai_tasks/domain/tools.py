"""Tool registry and argument schemas for the tool-calling assistant."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolRisk(str, Enum):
    """Risk level used by the tool policy layer."""

    READ_ONLY = "read_only"
    DRAFT = "draft"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ToolSpec:
    """Production-style contract for one model-callable tool."""

    name: str
    definition: dict[str, Any]
    risk: ToolRisk
    requires_approval: bool
    max_result_chars: int = 4_000


GET_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get a mock current weather report for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. Yerevan.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit.",
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}

SEARCH_DOCS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search uploaded documents by semantic meaning.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Number of snippets to return.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional exact filename filter.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional keyword filter.",
                },
                "min_similarity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Optional minimum similarity threshold.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CREATE_TICKET_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": "Create a support-ticket draft that requires human confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Ticket title."},
                "description": {
                    "type": "string",
                    "description": "Ticket details.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                },
            },
            "required": ["title", "description"],
            "additionalProperties": False,
        },
    },
}

SEND_EMAIL_DRAFT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_email_draft",
        "description": "Prepare an email draft. It does not send email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient address."},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
}

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_weather",
        definition=GET_WEATHER_TOOL,
        risk=ToolRisk.READ_ONLY,
        requires_approval=False,
    ),
    ToolSpec(
        name="search_docs",
        definition=SEARCH_DOCS_TOOL,
        risk=ToolRisk.READ_ONLY,
        requires_approval=False,
    ),
    ToolSpec(
        name="create_ticket",
        definition=CREATE_TICKET_TOOL,
        risk=ToolRisk.DRAFT,
        requires_approval=False,
    ),
    ToolSpec(
        name="send_email_draft",
        definition=SEND_EMAIL_DRAFT_TOOL,
        risk=ToolRisk.DRAFT,
        requires_approval=False,
    ),
)

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
TOOL_DEFINITIONS: list[dict[str, Any]] = [spec.definition for spec in TOOL_SPECS]


RESEARCH_TOOL_NAMES: tuple[str, ...] = ("search_docs", "create_ticket")
RESEARCH_TOOL_SPECS: tuple[ToolSpec, ...] = tuple(
    spec for spec in TOOL_SPECS if spec.name in RESEARCH_TOOL_NAMES
)
RESEARCH_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    spec.definition for spec in RESEARCH_TOOL_SPECS
]


class GetWeatherArgs(BaseModel):
    location: str = Field(..., min_length=1)
    unit: Literal["celsius", "fahrenheit"] = "celsius"


class SearchDocsArgs(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=5)
    filename: Optional[str] = None
    keyword: Optional[str] = None
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CreateTicketArgs(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class SendEmailDraftArgs(BaseModel):
    to: str = Field(..., min_length=3)
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


__all__ = [
    "CreateTicketArgs",
    "GetWeatherArgs",
    "RESEARCH_TOOL_DEFINITIONS",
    "RESEARCH_TOOL_NAMES",
    "RESEARCH_TOOL_SPECS",
    "SearchDocsArgs",
    "SendEmailDraftArgs",
    "TOOL_DEFINITIONS",
    "TOOL_REGISTRY",
    "TOOL_SPECS",
    "ToolRisk",
    "ToolSpec",
]
