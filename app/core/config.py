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


@lru_cache
def get_settings() -> Settings:
    """Cache settings so validation happens once per process."""
    return Settings()
