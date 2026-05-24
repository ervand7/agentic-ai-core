"""OpenAI-backed implementation of the LLMProvider port."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Optional

from app.domains.ai_tasks.domain.ports import (
    CompletionResult,
    LLMProvider,
    StreamChunk,
)
from app.shared.infrastructure.openai_client import OpenAIClient
from app.shared.openai_types import ChatCompletionMessageParam, ResponseFormat

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """Adapter mapping OpenAI SDK responses onto our domain port."""

    def __init__(self, client: OpenAIClient):
        self._client = client

    async def complete(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[ChatCompletionMessageParam],
        response_format: Optional[ResponseFormat] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CompletionResult:
        completion = await self._client.create_chat_completion(
            endpoint=endpoint,
            request_id=request_id,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = completion.usage
        return CompletionResult(
            content=completion.choices[0].message.content or "",
            model=completion.model,
            tokens_used=usage.total_tokens if usage else 0,
        )

    async def stream(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        settings = self._client.settings
        start_time = time.perf_counter()
        tokens_used = 0
        model_name: Optional[str] = settings.OPENAI_MODEL

        stream = await self._client.create_chat_completion_stream(
            endpoint=endpoint,
            request_id=request_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async for chunk in stream:
            if chunk.model:
                model_name = chunk.model
            if chunk.usage and chunk.usage.total_tokens is not None:
                tokens_used = chunk.usage.total_tokens
            content_delta = ""
            if chunk.choices and chunk.choices[0].delta.content:
                content_delta = chunk.choices[0].delta.content
            if content_delta:
                yield StreamChunk(
                    content=content_delta,
                    model=model_name,
                    tokens_used=tokens_used or None,
                )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "openai_stream_success request_id=%s endpoint=%s model=%s latency_ms=%.2f tokens_used=%s",
            request_id,
            endpoint,
            model_name,
            latency_ms,
            tokens_used,
        )
