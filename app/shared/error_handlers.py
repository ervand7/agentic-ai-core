"""Shared exception-to-HTTP mappings for provider-level errors."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)


async def handle_llm_timeout(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": str(exc)},
    )


async def handle_llm_rate_limit(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
    )


async def handle_llm_temporary(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


async def handle_llm_service(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(LLMTimeoutError, handle_llm_timeout)
    app.add_exception_handler(LLMRateLimitError, handle_llm_rate_limit)
    app.add_exception_handler(LLMTemporaryError, handle_llm_temporary)
    app.add_exception_handler(LLMServiceError, handle_llm_service)
