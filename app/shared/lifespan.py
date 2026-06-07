"""Application lifespan hooks."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import ValidationError

from app.shared.config import get_settings

logger = logging.getLogger(__name__)


def validate_settings() -> None:
    """
    Validate environment configuration during startup.

    This fails fast when OPENAI_API_KEY is missing, instead of failing on first request.
    """
    try:
        get_settings()
    except ValidationError as exc:
        logger.error("startup_config_error error=%s", str(exc))
        raise RuntimeError(
            "Invalid configuration. Ensure OPENAI_API_KEY is set in .env."
        ) from exc


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    validate_settings()
    yield
