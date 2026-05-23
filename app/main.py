"""FastAPI entrypoint."""

import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)
from app.core.logging_config import setup_logging
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.tasks import (
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    ClassifyRequest,
    ClassifyResponse,
    ExtractKeywordsRequest,
    ExtractKeywordsResponse,
    SummarizeRequest,
    SummarizeResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.services.openai_service import (
    analyze_text,
    ask_openai,
    ask_openai_stream,
    classify_text,
    extract_keywords,
    summarize_text,
    translate_text,
)

setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="AI Backend", version="2.0.0")


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


def _raise_http_error_from_service(exc: Exception) -> None:
    """Map service-layer exceptions into clean API responses."""
    if isinstance(exc, LLMTimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, LLMRateLimitError):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    if isinstance(exc, LLMTemporaryError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, LLMServiceError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected server error.",
    ) from exc


@app.get("/health")
async def health_check() -> dict:
    """Small health endpoint useful for quick checks."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """Accept a question and return an AI answer."""
    try:
        return await ask_openai(payload.question, request.state.request_id)
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/ask-stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    """Stream generated text token-by-token to the client."""
    try:
        return StreamingResponse(
            ask_openai_stream(payload.question, request.state.request_id),
            media_type="text/plain; charset=utf-8",
        )
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
    """Classify user text into sentiment + summary + keywords."""
    try:
        return await classify_text(payload.text, request.state.request_id)
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest, request: Request) -> SummarizeResponse:
    """Summarize text into a short response."""
    try:
        return await summarize_text(payload.text, request.state.request_id)
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/extract-keywords", response_model=ExtractKeywordsResponse)
async def extract_keywords_endpoint(
    payload: ExtractKeywordsRequest,
    request: Request,
) -> ExtractKeywordsResponse:
    """Extract key terms from user text."""
    try:
        return await extract_keywords(payload.text, request.state.request_id)
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest, request: Request) -> TranslateResponse:
    """Translate input text to a target language."""
    try:
        return await translate_text(
            payload.text,
            payload.target_language,
            request.state.request_id,
        )
    except Exception as exc:
        _raise_http_error_from_service(exc)


@app.post("/analyze-text", response_model=AnalyzeTextResponse)
async def analyze_text_endpoint(
    payload: AnalyzeTextRequest,
    request: Request,
) -> AnalyzeTextResponse:
    """Combined analysis endpoint for Stage 2."""
    try:
        return await analyze_text(payload.text, request.state.request_id)
    except Exception as exc:
        _raise_http_error_from_service(exc)
