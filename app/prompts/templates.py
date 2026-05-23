"""Prompt templates with explicit names and versions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """Small container to keep prompt metadata together."""

    name: str
    version: str
    system_prompt: str

    @property
    def prompt_version(self) -> str:
        return f"{self.name}_{self.version}"


PROMPTS: dict[str, PromptTemplate] = {
    "ask_v1": PromptTemplate(
        name="ask",
        version="v1",
        system_prompt="You are a helpful assistant. Give clear and concise answers.",
    ),
    "ask_stream_v1": PromptTemplate(
        name="ask_stream",
        version="v1",
        system_prompt="You are a helpful assistant. Stream clear and concise answers.",
    ),
    "classify_v1": PromptTemplate(
        name="classify",
        version="v1",
        system_prompt=(
            "You are a precise sentiment classifier. "
            "Return only valid JSON that matches the schema."
        ),
    ),
    "summarize_v1": PromptTemplate(
        name="summarize",
        version="v1",
        system_prompt="You summarize text in 2-3 short, clear sentences.",
    ),
    "extract_keywords_v1": PromptTemplate(
        name="extract_keywords",
        version="v1",
        system_prompt=(
            "Extract the most relevant keywords from the text. "
            "Return only valid JSON that matches the schema."
        ),
    ),
    "translate_v1": PromptTemplate(
        name="translate",
        version="v1",
        system_prompt=(
            "You are a professional translator. "
            "Return only the translated text without extra commentary."
        ),
    ),
    "analyze_text_v1": PromptTemplate(
        name="analyze_text",
        version="v1",
        system_prompt=(
            "You are an NLP assistant. Analyze user text and return valid JSON. "
            "Include summary, sentiment (positive|negative|neutral), keywords list, "
            "and detected language."
        ),
    ),
}
