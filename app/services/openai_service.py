"""OpenAI interaction logic."""

import json
import logging
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI, OpenAIError

from app.core.config import settings
from app.schemas.ask import AskResponse
from app.schemas.tasks import (
    ClassifyResponse,
    ExtractKeywordsResponse,
    SummarizeResponse,
    TranslateResponse,
)

# Create one shared async client instance for the app.
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
logger = logging.getLogger(__name__)

# JSON Schema used for strict structured output in /classify.
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


def _build_messages(user_input: str, system_prompt: str) -> list[dict]:
    """Create consistent system + user message arrays."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


def _log_request(endpoint: str, model: str, messages: list[dict]) -> None:
    """Log full messages and model before sending the request."""
    logger.info(
        "openai_request endpoint=%s model=%s messages=%s",
        endpoint,
        model,
        json.dumps(messages, ensure_ascii=False),
    )


def _log_response(endpoint: str, model: str, tokens_used: int, latency_ms: float) -> None:
    """Log model, token usage, and latency after receiving a response."""
    logger.info(
        "openai_response endpoint=%s model=%s tokens_used=%s latency_ms=%.2f",
        endpoint,
        model,
        tokens_used,
        latency_ms,
    )


async def ask_openai(question: str) -> AskResponse:
    """Send system + user messages to OpenAI and return structured answer."""
    messages = _build_messages(question, settings.OPENAI_SYSTEM_PROMPT_ASK)
    _log_request("ask", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()

    completion = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    answer_text = completion.choices[0].message.content or ""
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("ask", completion.model, tokens_used, latency_ms)

    return AskResponse(answer=answer_text, model=completion.model, tokens_used=tokens_used)


async def ask_openai_stream(question: str) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens as they arrive from OpenAI.
    Each yielded chunk is sent immediately by FastAPI StreamingResponse.
    """
    messages = _build_messages(question, settings.OPENAI_SYSTEM_PROMPT_ASK_STREAM)
    _log_request("ask-stream", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()
    tokens_used = 0
    model_name = settings.OPENAI_MODEL

    stream = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        stream=True,
        # Ask API to include token usage in the final stream chunk.
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if getattr(chunk, "model", None):
            model_name = chunk.model

        if chunk.usage and chunk.usage.total_tokens is not None:
            tokens_used = chunk.usage.total_tokens

        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("ask-stream", model_name, tokens_used, latency_ms)


async def classify_text(text: str) -> ClassifyResponse:
    """Classify sentiment with strict JSON schema output."""
    messages = _build_messages(text, settings.OPENAI_SYSTEM_PROMPT_CLASSIFY)
    _log_request("classify", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()

    completion = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        # Structured output is configured here via strict JSON schema.
        response_format={"type": "json_schema", "json_schema": CLASSIFY_JSON_SCHEMA},
    )

    content = completion.choices[0].message.content or "{}"
    raw_data = json.loads(content)
    validated = ClassifyResponse.model_validate(raw_data)
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("classify", completion.model, tokens_used, latency_ms)
    return validated


async def summarize_text(text: str) -> SummarizeResponse:
    """Generate a concise summary."""
    messages = _build_messages(text, settings.OPENAI_SYSTEM_PROMPT_SUMMARIZE)
    _log_request("summarize", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()

    completion = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    output = completion.choices[0].message.content or ""
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("summarize", completion.model, tokens_used, latency_ms)
    return SummarizeResponse(summary=output, model=completion.model, tokens_used=tokens_used)


async def extract_keywords(text: str) -> ExtractKeywordsResponse:
    """Extract keywords and return them as a list."""
    messages = _build_messages(text, settings.OPENAI_SYSTEM_PROMPT_EXTRACT_KEYWORDS)
    _log_request("extract-keywords", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()

    completion = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "keywords_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
            },
        },
    )

    raw_data = json.loads(completion.choices[0].message.content or "{}")
    keywords = raw_data.get("keywords", [])
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("extract-keywords", completion.model, tokens_used, latency_ms)
    return ExtractKeywordsResponse(
        keywords=keywords,
        model=completion.model,
        tokens_used=tokens_used,
    )


async def translate_text(text: str, target_language: str) -> TranslateResponse:
    """Translate text into the requested language."""
    system_prompt = (
        f"{settings.OPENAI_SYSTEM_PROMPT_TRANSLATE} "
        f"Translate the user text to {target_language}. Return only the translation."
    )
    messages = _build_messages(text, system_prompt)
    _log_request("translate", settings.OPENAI_MODEL, messages)
    start = time.perf_counter()

    completion = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )

    output = completion.choices[0].message.content or ""
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    latency_ms = (time.perf_counter() - start) * 1000
    _log_response("translate", completion.model, tokens_used, latency_ms)
    return TranslateResponse(
        translation=output,
        model=completion.model,
        tokens_used=tokens_used,
    )


__all__ = [
    "OpenAIError",
    "ask_openai",
    "ask_openai_stream",
    "classify_text",
    "summarize_text",
    "extract_keywords",
    "translate_text",
]
