"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized app settings.

    pydantic-settings reads values from environment variables and `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str = Field(default="", min_length=1)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    OPENAI_MAX_TOKENS: int = Field(default=300, ge=1)

    OPENAI_TEMPERATURE_CLASSIFY: float = Field(default=0.0, ge=0.0, le=2.0)
    OPENAI_TEMPERATURE_SUMMARIZE: float = Field(default=0.2, ge=0.0, le=2.0)
    OPENAI_TEMPERATURE_EXTRACT_KEYWORDS: float = Field(default=0.0, ge=0.0, le=2.0)
    OPENAI_TEMPERATURE_TRANSLATE: float = Field(default=0.1, ge=0.0, le=2.0)
    OPENAI_TEMPERATURE_ANALYZE_TEXT: float = Field(default=0.0, ge=0.0, le=2.0)

    OPENAI_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0)
    OPENAI_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    OPENAI_RETRY_BASE_DELAY_SECONDS: float = Field(default=0.5, gt=0)
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    QDRANT_URL: str = Field(default="http://localhost:6333", min_length=1)
    QDRANT_COLLECTION_NAME: str = Field(default="documents", min_length=1)
    DOCUMENT_CHUNK_SIZE: int = Field(default=500, ge=50)
    DOCUMENT_CHUNK_OVERLAP: int = Field(default=100, ge=0)

    RAG_TOP_K: int = Field(default=4, ge=1, le=20)
    RAG_MIN_SIMILARITY: float = Field(default=0.2, ge=0.0, le=1.0)
    RAG_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    RAG_MAX_TOKENS: int = Field(default=500, ge=1)

    PROMPT_ASK_SYSTEM: str = Field(
        default="You are a helpful assistant. Give clear and concise answers.",
        min_length=1,
    )
    PROMPT_ASK_STREAM_SYSTEM: str = Field(
        default="You are a helpful assistant. Stream clear and concise answers.",
        min_length=1,
    )
    PROMPT_CLASSIFY_SYSTEM: str = Field(
        default=(
            "You are a precise sentiment classifier. "
            "Return only valid JSON that matches the schema."
        ),
        min_length=1,
    )
    PROMPT_SUMMARIZE_SYSTEM: str = Field(
        default="You summarize text in 2-3 short, clear sentences.",
        min_length=1,
    )
    PROMPT_EXTRACT_KEYWORDS_SYSTEM: str = Field(
        default=(
            "Extract the most relevant keywords from the text. "
            "Return only valid JSON that matches the schema."
        ),
        min_length=1,
    )
    PROMPT_TRANSLATE_SYSTEM: str = Field(
        default=(
            "You are a professional translator. "
            "Return only the translated text without extra commentary."
        ),
        min_length=1,
    )
    PROMPT_ANALYZE_TEXT_SYSTEM: str = Field(
        default=(
            "You are an NLP assistant. Analyze user text and return valid JSON. "
            "Include summary, sentiment (positive|negative|neutral), keywords list, "
            "and detected language."
        ),
        min_length=1,
    )
    PROMPT_TOOL_ASSISTANT_SYSTEM: str = Field(
        default=(
            "You are a safe tool-calling assistant. Use tools only when they help. "
            "Never claim that a ticket was filed or an email was sent; ticket and "
            "email tools only create drafts that require human confirmation. "
            "Summarize tool results clearly for the user."
        ),
        min_length=1,
    )
    PROMPT_RAG_SYSTEM: str = Field(
        default=(
            "You are a documentation assistant. Answer the user's question using ONLY "
            "the numbered context snippets provided. Cite the snippets you rely on with "
            "bracketed numbers like [1] or [2]. If the answer is not contained in the "
            "context, reply that you don't know based on the provided documents. "
            "Never invent facts that are not supported by the context."
        ),
        min_length=1,
    )


@lru_cache
def get_settings() -> Settings:
    """Cache settings so validation happens once per process."""
    return Settings()
