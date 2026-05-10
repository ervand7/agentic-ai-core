"""Application configuration module."""

import os

from dotenv import load_dotenv

# Load environment variables from a local .env file (if present).
load_dotenv()


class Settings:
    """Simple settings container for environment-based configuration."""

    # Read API key from environment for security (never hard-code secrets).
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model and generation settings are centralized here for easy updates.
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
    # Each endpoint has its own system prompt so behavior is isolated.
    OPENAI_SYSTEM_PROMPT_ASK: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_ASK",
        "You are a helpful assistant. Give clear and concise answers.",
    )
    OPENAI_SYSTEM_PROMPT_ASK_STREAM: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_ASK_STREAM",
        "You are a helpful assistant. Stream clear and concise answers.",
    )
    OPENAI_SYSTEM_PROMPT_CLASSIFY: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_CLASSIFY",
        "You are a precise text classifier. Follow the JSON schema exactly.",
    )
    OPENAI_SYSTEM_PROMPT_SUMMARIZE: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_SUMMARIZE",
        "You summarize text in 2-3 short, clear sentences.",
    )
    OPENAI_SYSTEM_PROMPT_EXTRACT_KEYWORDS: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_EXTRACT_KEYWORDS",
        "You extract the most relevant keywords from text.",
    )
    OPENAI_SYSTEM_PROMPT_TRANSLATE: str = os.getenv(
        "OPENAI_SYSTEM_PROMPT_TRANSLATE",
        "You are a professional translator and return accurate translations.",
    )


settings = Settings()
