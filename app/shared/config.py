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

    OPENAI_API_KEY: str = Field(..., min_length=1)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    OPENAI_MAX_TOKENS: int = Field(default=300, ge=1)
    OPENAI_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0)
    OPENAI_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    OPENAI_RETRY_BASE_DELAY_SECONDS: float = Field(default=0.5, gt=0)
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    DOCUMENT_CHUNK_SIZE: int = Field(default=500, ge=50)
    DOCUMENT_CHUNK_OVERLAP: int = Field(default=100, ge=0)

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


@lru_cache
def get_settings() -> Settings:
    """Cache settings so validation happens once per process."""
    return Settings()
