import logging
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.domains.ai_tasks.api.router import router as ai_tasks_router
from app.domains.documents.api.router import router as documents_router
from app.domains.documents.application.services import (
    DocumentValidationError,
    EmptyVectorStoreError,
)
from app.shared.config import get_settings
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)
from app.shared.logging import setup_logging
from app.web.router import STATIC_DIR
from app.web.router import router as web_router

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Backend", version="3.0.0")
app.include_router(ai_tasks_router)
app.include_router(documents_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.exception_handler(LLMTimeoutError)
async def _handle_llm_timeout(_: Request, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": str(exc)},
    )


@app.exception_handler(LLMRateLimitError)
async def _handle_llm_rate_limit(_: Request, exc: LLMRateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
    )


@app.exception_handler(LLMTemporaryError)
async def _handle_llm_temporary(_: Request, exc: LLMTemporaryError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.exception_handler(LLMServiceError)
async def _handle_llm_service(_: Request, exc: LLMServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


@app.exception_handler(DocumentValidationError)
async def _handle_document_validation(
        _: Request, exc: DocumentValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(EmptyVectorStoreError)
async def _handle_empty_vector_store(
        _: Request, exc: EmptyVectorStoreError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
