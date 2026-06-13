"""Stable endpoint/task name constants for the AI tasks bounded context.

These string labels are shared between the HTTP routes and the application
services (where they are passed to the LLM provider for logging), so keeping
them in one place avoids drift between the two.
"""

from typing import Final


class Endpoint:
    """Canonical endpoint names used across the AI tasks context."""

    ASK: Final[str] = "ask"
    ASK_STREAM: Final[str] = "ask-stream"
    CLASSIFY: Final[str] = "classify"
    SUMMARIZE: Final[str] = "summarize"
    EXTRACT_KEYWORDS: Final[str] = "extract-keywords"
    TRANSLATE: Final[str] = "translate"
    ANALYZE_TEXT: Final[str] = "analyze-text"
    TOOL_ASSISTANT: Final[str] = "tool-assistant"


__all__ = ["Endpoint"]
