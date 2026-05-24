"""
Application services (use cases) for the AI tasks bounded context.

These services orchestrate the domain by calling the LLMProvider port.
They are deliberately ignorant of HTTP and of the concrete LLM provider.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from json import JSONDecodeError

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.domains.ai_tasks.api.schemas import (
    AnalyzeTextResponse,
    AskResponse,
    ClassifyResponse,
    ExtractKeywordsResponse,
    SummarizeResponse,
    TranslateResponse,
)
from app.domains.ai_tasks.domain.prompts import get_prompts
from app.domains.ai_tasks.domain.ports import LLMProvider
from app.domains.ai_tasks.domain.response_formats import (
    ANALYZE_TEXT_RESPONSE_FORMAT,
    CLASSIFY_RESPONSE_FORMAT,
    KEYWORDS_RESPONSE_FORMAT,
)
from app.shared.exceptions import LLMServiceError
from app.shared.openai_types import ChatCompletionMessageParam


def _build_messages(
    user_input: str, prompt_key: str
) -> list[ChatCompletionMessageParam]:
    """Create system+user messages from the prompt registry."""
    prompt = get_prompts()[prompt_key]
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": prompt.system_prompt,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_input,
    }
    return [system_message, user_message]


def _parse_json_content(content: str) -> dict:
    """Protect routes from malformed model JSON output."""
    try:
        return json.loads(content or "{}")
    except JSONDecodeError as exc:
        raise LLMServiceError("Model returned invalid JSON output.") from exc


class AITasksService:
    """All AI task use cases grouped behind a single application service."""

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    async def ask(self, question: str, request_id: str) -> AskResponse:
        """Simple question answering use case."""
        result = await self._llm.complete(
            endpoint="ask",
            request_id=request_id,
            messages=_build_messages(question, "ask_v1"),
        )
        return AskResponse(
            answer=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def ask_stream(
        self, question: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream tokens token-by-token for chat-like UIs."""
        async for chunk in self._llm.stream(
            endpoint="ask-stream",
            request_id=request_id,
            messages=_build_messages(question, "ask_stream_v1"),
        ):
            if chunk.content:
                yield chunk.content

    async def classify(self, text: str, request_id: str) -> ClassifyResponse:
        """Classify sentiment and return JSON validated with Pydantic."""
        result = await self._llm.complete(
            endpoint="classify",
            request_id=request_id,
            messages=_build_messages(text, "classify_v1"),
            temperature=0,
            response_format=CLASSIFY_RESPONSE_FORMAT,
        )
        raw = _parse_json_content(result.content)
        return ClassifyResponse.model_validate(raw)

    async def summarize(self, text: str, request_id: str) -> SummarizeResponse:
        """Summarize text into a short answer."""
        result = await self._llm.complete(
            endpoint="summarize",
            request_id=request_id,
            messages=_build_messages(text, "summarize_v1"),
            temperature=0.2,
        )
        return SummarizeResponse(
            summary=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def extract_keywords(
        self, text: str, request_id: str
    ) -> ExtractKeywordsResponse:
        """Extract relevant terms as a JSON list."""
        result = await self._llm.complete(
            endpoint="extract-keywords",
            request_id=request_id,
            messages=_build_messages(text, "extract_keywords_v1"),
            temperature=0,
            response_format=KEYWORDS_RESPONSE_FORMAT,
        )
        raw_data = _parse_json_content(result.content)
        return ExtractKeywordsResponse(
            keywords=raw_data.get("keywords", []),
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def translate(
        self, text: str, target_language: str, request_id: str
    ) -> TranslateResponse:
        """Translate text to a target language."""
        prompt = get_prompts()["translate_v1"]
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                f"{prompt.system_prompt} Translate to {target_language}. "
                "Return only the translation."
            ),
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": text,
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]
        result = await self._llm.complete(
            endpoint="translate",
            request_id=request_id,
            messages=messages,
            temperature=0.1,
        )
        return TranslateResponse(
            translation=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def analyze_text(self, text: str, request_id: str) -> AnalyzeTextResponse:
        """
        Combined use case for summary, sentiment, keywords, and language.

        Uses one structured-output call to keep latency and token usage lower.
        """
        result = await self._llm.complete(
            endpoint="analyze-text",
            request_id=request_id,
            messages=_build_messages(text, "analyze_text_v1"),
            temperature=0,
            response_format=ANALYZE_TEXT_RESPONSE_FORMAT,
        )
        raw_data = _parse_json_content(result.content)
        return AnalyzeTextResponse.model_validate(
            {
                **raw_data,
                "model": result.model,
                "tokens_used": result.tokens_used,
                "prompt_version": get_prompts()["analyze_text_v1"].prompt_version,
            }
        )
