"""FastAPI entrypoint."""

import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.routers.ai import router as ai_router
from app.routers.documents import router as documents_router

setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="AI Backend", version="3.0.0")
app.include_router(ai_router)
app.include_router(documents_router)


@app.on_event("startup")
async def validate_settings_on_startup() -> None:
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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a request id so logs can be correlated across layers."""
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response

