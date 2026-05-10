"""FastAPI entrypoint."""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.schemas.ask import AskRequest, AskResponse
from app.schemas.tasks import (
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
    OpenAIError,
    ask_openai,
    ask_openai_stream,
    classify_text,
    extract_keywords,
    summarize_text,
    translate_text,
)

app = FastAPI(title="AI Backend", version="1.0.0")
setup_logging()


def _ensure_api_key() -> None:
    """Fail early when API key is missing."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not set. Add it to your .env file.",
        )


@app.get("/health")
async def health_check() -> dict:
    """Small health endpoint useful for quick checks."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    """
    Accepts a question and returns an AI answer.
    This endpoint is async to support non-blocking I/O with OpenAI.
    """
    _ensure_api_key()

    try:
        return await ask_openai(payload.question)
    except OpenAIError as exc:
        # Convert provider-specific failures to API-friendly errors.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        # Catch unexpected issues to avoid leaking internals.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc


@app.post("/ask-stream")
async def ask_stream(payload: AskRequest) -> StreamingResponse:
    """
    Streams generated text token-by-token to the client.
    Useful for chat-like UX where users see output immediately.
    """
    _ensure_api_key()

    try:
        return StreamingResponse(
            ask_openai_stream(payload.question),
            media_type="text/plain; charset=utf-8",
        )
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    """Classify user text into sentiment + summary + keywords."""
    _ensure_api_key()

    try:
        return await classify_text(payload.text)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    """Summarize text into a short response."""
    _ensure_api_key()

    try:
        return await summarize_text(payload.text)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc


@app.post("/extract-keywords", response_model=ExtractKeywordsResponse)
async def extract_keywords_endpoint(
    payload: ExtractKeywordsRequest,
) -> ExtractKeywordsResponse:
    """Extract key terms from user text."""
    _ensure_api_key()

    try:
        return await extract_keywords(payload.text)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc


@app.post("/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest) -> TranslateResponse:
    """Translate input text to a target language."""
    _ensure_api_key()

    try:
        return await translate_text(payload.text, payload.target_language)
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI API error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected server error: {exc}",
        ) from exc
