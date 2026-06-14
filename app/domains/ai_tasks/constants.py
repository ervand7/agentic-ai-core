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
    RESEARCH_AGENT: Final[str] = "research-agent"
    RESEARCH_AGENT_PLAN: Final[str] = "research-agent-plan"
    RESEARCH_AGENT_CRITIC: Final[str] = "research-agent-critic"


__all__ = ["Endpoint"]
