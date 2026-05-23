"""Application-level AI service built on top of the OpenAI client wrapper."""

import json
import logging
import time
from collections.abc import AsyncGenerator
from json import JSONDecodeError

from app.core.exceptions import LLMServiceError
from app.prompts.templates import PROMPTS
from app.schemas.ask import AskResponse
from app.schemas.tasks import (
    AnalyzeTextResponse,
    ClassifyResponse,
    ExtractKeywordsResponse,
    SummarizeResponse,
    TranslateResponse,
)
from app.services.openai_client import get_openai_client

logger = logging.getLogger(__name__)

CLASSIFY_JSON_SCHEMA = {
    "name": "classify_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "summary": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sentiment", "summary", "keywords"],
        "additionalProperties": False,
    },
}

KEYWORDS_JSON_SCHEMA = {
    "name": "keywords_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
}

ANALYZE_TEXT_JSON_SCHEMA = {
    "name": "analyze_text_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "language": {"type": "string"},
        },
        "required": ["summary", "sentiment", "keywords", "language"],
        "additionalProperties": False,
    },
}


def _build_messages(user_input: str, prompt_key: str) -> list[dict[str, str]]:
    """Create system+user messages from the prompt registry."""
    prompt = PROMPTS[prompt_key]
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": user_input},
    ]


def _tokens_used(completion: object) -> int:
    usage = getattr(completion, "usage", None)
    return usage.total_tokens if usage else 0


def _parse_json_content(content: str) -> dict:
    """Protect routes from malformed model JSON output."""
    try:
        return json.loads(content or "{}")
    except JSONDecodeError as exc:
        raise LLMServiceError("Model returned invalid JSON output.") from exc


async def ask_openai(question: str, request_id: str) -> AskResponse:
    """Simple question answering endpoint service."""
    completion = await get_openai_client().create_chat_completion(
        endpoint="ask",
        request_id=request_id,
        messages=_build_messages(question, "ask_v1"),
    )
    answer = completion.choices[0].message.content or ""
    return AskResponse(
        answer=answer,
        model=completion.model,
        tokens_used=_tokens_used(completion),
    )


async def ask_openai_stream(question: str, request_id: str) -> AsyncGenerator[str, None]:
    """Stream tokens from OpenAI for chat-like UIs."""
    client_wrapper = get_openai_client()
    settings = client_wrapper.settings
    messages = _build_messages(question, "ask_stream_v1")
    start_time = time.perf_counter()
    tokens_used = 0
    model_name = settings.OPENAI_MODEL

    stream = await client_wrapper.create_chat_completion_stream(
        endpoint="ask-stream",
        request_id=request_id,
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    async for chunk in stream:
        if getattr(chunk, "model", None):
            model_name = chunk.model
        if chunk.usage and chunk.usage.total_tokens is not None:
            tokens_used = chunk.usage.total_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "openai_stream_success request_id=%s endpoint=%s model=%s latency_ms=%.2f tokens_used=%s",
        request_id,
        "ask-stream",
        model_name,
        latency_ms,
        tokens_used,
    )


async def classify_text(text: str, request_id: str) -> ClassifyResponse:
    """Classify sentiment and return JSON fields validated with Pydantic."""
    completion = await get_openai_client().create_chat_completion(
        endpoint="classify",
        request_id=request_id,
        messages=_build_messages(text, "classify_v1"),
        temperature=0,
        response_format={"type": "json_schema", "json_schema": CLASSIFY_JSON_SCHEMA},
    )
    raw = _parse_json_content(completion.choices[0].message.content or "")
    return ClassifyResponse.model_validate(raw)


async def summarize_text(text: str, request_id: str) -> SummarizeResponse:
    """Summarize text into a short answer."""
    completion = await get_openai_client().create_chat_completion(
        endpoint="summarize",
        request_id=request_id,
        messages=_build_messages(text, "summarize_v1"),
        temperature=0.2,
    )
    return SummarizeResponse(
        summary=completion.choices[0].message.content or "",
        model=completion.model,
        tokens_used=_tokens_used(completion),
    )


async def extract_keywords(text: str, request_id: str) -> ExtractKeywordsResponse:
    """Extract relevant terms as a JSON list."""
    completion = await get_openai_client().create_chat_completion(
        endpoint="extract-keywords",
        request_id=request_id,
        messages=_build_messages(text, "extract_keywords_v1"),
        temperature=0,
        response_format={"type": "json_schema", "json_schema": KEYWORDS_JSON_SCHEMA},
    )
    raw_data = _parse_json_content(completion.choices[0].message.content or "")
    return ExtractKeywordsResponse(
        keywords=raw_data.get("keywords", []),
        model=completion.model,
        tokens_used=_tokens_used(completion),
    )


async def translate_text(
    text: str,
    target_language: str,
    request_id: str,
) -> TranslateResponse:
    """Translate text to a target language."""
    prompt = PROMPTS["translate_v1"]
    messages = [
        {
            "role": "system",
            "content": (
                f"{prompt.system_prompt} Translate to {target_language}. "
                "Return only the translation."
            ),
        },
        {"role": "user", "content": text},
    ]
    completion = await get_openai_client().create_chat_completion(
        endpoint="translate",
        request_id=request_id,
        messages=messages,
        temperature=0.1,
    )
    return TranslateResponse(
        translation=completion.choices[0].message.content or "",
        model=completion.model,
        tokens_used=_tokens_used(completion),
    )


async def analyze_text(text: str, request_id: str) -> AnalyzeTextResponse:
    """
    Combined endpoint for summary, sentiment, keywords, and language.

    Uses one structured output call to keep latency and token usage lower.
    """
    completion = await get_openai_client().create_chat_completion(
        endpoint="analyze-text",
        request_id=request_id,
        messages=_build_messages(text, "analyze_text_v1"),
        temperature=0,
        response_format={"type": "json_schema", "json_schema": ANALYZE_TEXT_JSON_SCHEMA},
    )
    raw_data = _parse_json_content(completion.choices[0].message.content or "")
    parsed = AnalyzeTextResponse.model_validate(
        {
            **raw_data,
            "model": completion.model,
            "tokens_used": _tokens_used(completion),
            "prompt_version": PROMPTS["analyze_text_v1"].prompt_version,
        }
    )
    return parsed


__all__ = [
    "analyze_text",
    "ask_openai",
    "ask_openai_stream",
    "classify_text",
    "extract_keywords",
    "summarize_text",
    "translate_text",
]
