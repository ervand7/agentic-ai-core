"""
Prompt templates with explicit names and versions.

Prompt texts are loaded from environment variables (with safe defaults)
through `Settings`. This lets operators tune wording or A/B test prompts
without redeploying code, while still keeping prompt identifiers stable
in the codebase.
"""

from dataclasses import dataclass
from functools import lru_cache

from app.shared.config import Settings, get_settings


@dataclass(frozen=True)
class PromptTemplate:
    """Small container to keep prompt metadata together."""

    name: str
    version: str
    system_prompt: str

    @property
    def prompt_version(self) -> str:
        return f"{self.name}_{self.version}"


def _build_prompts(settings: Settings) -> dict[str, PromptTemplate]:
    """Build the prompt registry from settings.

    The dict keys are stable identifiers used by application services
    (`get_prompts()["ask_v1"]`, etc.). Versions stay in code for now;
    bump them deliberately when prompt text changes meaningfully.
    """
    return {
        "ask_v1": PromptTemplate(
            name="ask",
            version="v1",
            system_prompt=settings.PROMPT_ASK_SYSTEM,
        ),
        "ask_stream_v1": PromptTemplate(
            name="ask_stream",
            version="v1",
            system_prompt=settings.PROMPT_ASK_STREAM_SYSTEM,
        ),
        "classify_v1": PromptTemplate(
            name="classify",
            version="v1",
            system_prompt=settings.PROMPT_CLASSIFY_SYSTEM,
        ),
        "summarize_v1": PromptTemplate(
            name="summarize",
            version="v1",
            system_prompt=settings.PROMPT_SUMMARIZE_SYSTEM,
        ),
        "extract_keywords_v1": PromptTemplate(
            name="extract_keywords",
            version="v1",
            system_prompt=settings.PROMPT_EXTRACT_KEYWORDS_SYSTEM,
        ),
        "translate_v1": PromptTemplate(
            name="translate",
            version="v1",
            system_prompt=settings.PROMPT_TRANSLATE_SYSTEM,
        ),
        "analyze_text_v1": PromptTemplate(
            name="analyze_text",
            version="v1",
            system_prompt=settings.PROMPT_ANALYZE_TEXT_SYSTEM,
        ),
        "tool_assistant_v1": PromptTemplate(
            name="tool_assistant",
            version="v1",
            system_prompt=settings.PROMPT_TOOL_ASSISTANT_SYSTEM,
        ),
        "research_agent_planner_v1": PromptTemplate(
            name="research_agent_planner",
            version="v1",
            system_prompt=settings.PROMPT_RESEARCH_AGENT_PLANNER_SYSTEM,
        ),
        "research_agent_v1": PromptTemplate(
            name="research_agent",
            version="v1",
            system_prompt=settings.PROMPT_RESEARCH_AGENT_SYSTEM,
        ),
        "research_agent_critic_v1": PromptTemplate(
            name="research_agent_critic",
            version="v1",
            system_prompt=settings.PROMPT_RESEARCH_AGENT_CRITIC_SYSTEM,
        ),
    }


@lru_cache
def get_prompts() -> dict[str, PromptTemplate]:
    """Return the cached prompt registry, built from current settings."""
    return _build_prompts(get_settings())


__all__ = ["PromptTemplate", "get_prompts"]
