"""
Thin async OpenAI client wrapper with retries and timeout handling.

This module is part of the shared kernel because both the AI tasks and
documents bounded contexts need to talk to OpenAI (chat completions and
embeddings respectively). It exposes a single low-level adapter; each
bounded context wraps it behind its own port to keep domain code
provider-agnostic.
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Optional, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
    ResponseFormat,
)

from app.shared.config import Settings, get_settings
from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)
from app.shared.openai_types import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Central place for all direct OpenAI SDK calls."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        return (
            self.settings.OPENAI_TEMPERATURE
            if temperature is None
            else temperature
        )

    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> int:
        return (
            self.settings.OPENAI_MAX_TOKENS if max_tokens is None else max_tokens
        )

    def _build_non_streaming_params(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: Optional[ResponseFormat],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> CompletionCreateParamsNonStreaming:
        """Build a typed request body so static checkers resolve the non-stream overload."""
        params: CompletionCreateParamsNonStreaming = {
            "model": self.settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if response_format is not None:
            params["response_format"] = response_format
        return params

    def _build_streaming_params(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> CompletionCreateParamsStreaming:
        """Build a typed request body so static checkers resolve the stream overload."""
        stream_options: ChatCompletionStreamOptionsParam = {"include_usage": True}
        return {
            "model": self.settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": self._resolve_temperature(temperature),
            "max_tokens": self._resolve_max_tokens(max_tokens),
            "stream": True,
            "stream_options": stream_options,
        }

    async def create_chat_completion(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[ChatCompletionMessageParam],
        response_format: Optional[ResponseFormat] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """
        Execute a chat completion with retries and exponential backoff.

        Retries are intentionally simple and readable for beginner projects.
        """

        max_retries = self.settings.OPENAI_MAX_RETRIES
        base_delay = self.settings.OPENAI_RETRY_BASE_DELAY_SECONDS
        start_time = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                completion = cast(
                    ChatCompletion,
                    await self.client.chat.completions.create(
                        **self._build_non_streaming_params(
                            messages, response_format, temperature, max_tokens
                        ),
                    ),
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                tokens_used = completion.usage.total_tokens if completion.usage else 0
                logger.info(
                    "openai_success request_id=%s endpoint=%s model=%s latency_ms=%.2f tokens_used=%s",
                    request_id,
                    endpoint,
                    completion.model,
                    latency_ms,
                    tokens_used,
                )
                return completion
            except APITimeoutError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                logger.error(
                    "openai_timeout request_id=%s endpoint=%s model=%s error=%s",
                    request_id,
                    endpoint,
                    self.settings.OPENAI_MODEL,
                    str(exc),
                )
                raise LLMTimeoutError(
                    "The AI provider timed out. Please try again shortly."
                ) from exc
            except RateLimitError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                logger.error(
                    "openai_rate_limit request_id=%s endpoint=%s model=%s error=%s",
                    request_id,
                    endpoint,
                    self.settings.OPENAI_MODEL,
                    str(exc),
                )
                raise LLMRateLimitError(
                    "Rate limit reached. Please wait a moment and retry."
                ) from exc
            except (APIConnectionError, InternalServerError) as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                logger.error(
                    "openai_temporary_failure request_id=%s endpoint=%s model=%s error=%s",
                    request_id,
                    endpoint,
                    self.settings.OPENAI_MODEL,
                    str(exc),
                )
                raise LLMTemporaryError(
                    "Temporary AI provider issue. Please retry shortly."
                ) from exc
            except OpenAIError as exc:
                logger.error(
                    "openai_service_error request_id=%s endpoint=%s model=%s error=%s",
                    request_id,
                    endpoint,
                    self.settings.OPENAI_MODEL,
                    str(exc),
                )
                raise LLMServiceError("AI provider returned an unexpected error.") from exc

        raise LLMServiceError("AI provider returned an unexpected error.")

    async def create_chat_completion_stream(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncStream[ChatCompletionChunk]:
        """Create a streaming completion with the same retry behavior."""
        max_retries = self.settings.OPENAI_MAX_RETRIES
        base_delay = self.settings.OPENAI_RETRY_BASE_DELAY_SECONDS

        for attempt in range(max_retries + 1):
            try:
                return cast(
                    AsyncStream[ChatCompletionChunk],
                    await self.client.chat.completions.create(
                        **self._build_streaming_params(
                            messages, temperature, max_tokens
                        ),
                    ),
                )
            except APITimeoutError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMTimeoutError(
                    "The AI provider timed out. Please try again shortly."
                ) from exc
            except RateLimitError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMRateLimitError(
                    "Rate limit reached. Please wait a moment and retry."
                ) from exc
            except (APIConnectionError, InternalServerError) as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMTemporaryError(
                    "Temporary AI provider issue. Please retry shortly."
                ) from exc
            except OpenAIError as exc:
                raise LLMServiceError("AI provider returned an unexpected error.") from exc

        raise LLMServiceError("AI provider returned an unexpected error.")

    async def create_embedding(
        self,
        *,
        endpoint: str,
        request_id: str,
        text: str,
        model: Optional[str] = None,
    ) -> list[float]:
        """Generate one embedding vector with retries and friendly errors."""
        max_retries = self.settings.OPENAI_MAX_RETRIES
        base_delay = self.settings.OPENAI_RETRY_BASE_DELAY_SECONDS
        start_time = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.embeddings.create(
                    model=model or self.settings.OPENAI_EMBEDDING_MODEL,
                    input=text,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                vector = response.data[0].embedding
                if vector is None:
                    raise LLMServiceError(
                        "AI provider returned an embedding without vector data."
                    )
                logger.info(
                    "openai_embedding_success request_id=%s endpoint=%s model=%s latency_ms=%.2f vector_size=%s",
                    request_id,
                    endpoint,
                    model or self.settings.OPENAI_EMBEDDING_MODEL,
                    latency_ms,
                    len(vector),
                )
                return vector
            except APITimeoutError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMTimeoutError(
                    "The AI provider timed out. Please try again shortly."
                ) from exc
            except RateLimitError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMRateLimitError(
                    "Rate limit reached. Please wait a moment and retry."
                ) from exc
            except (APIConnectionError, InternalServerError) as exc:
                if attempt < max_retries:
                    await self._retry_sleep(
                        attempt, base_delay, request_id, endpoint, exc
                    )
                    continue
                raise LLMTemporaryError(
                    "Temporary AI provider issue. Please retry shortly."
                ) from exc
            except OpenAIError as exc:
                raise LLMServiceError("AI provider returned an unexpected error.") from exc

        raise LLMServiceError("AI provider returned an unexpected error.")

    @staticmethod
    async def _retry_sleep(
        attempt: int,
        base_delay: float,
        request_id: str,
        endpoint: str,
        error: Exception,
    ) -> None:
        """Sleep using exponential backoff: base, base*2, base*4, ..."""
        delay = base_delay * (2**attempt)
        logger.warning(
            "openai_retry request_id=%s endpoint=%s attempt=%s delay_s=%.2f error=%s",
            request_id,
            endpoint,
            attempt + 1,
            delay,
            str(error),
        )
        await asyncio.sleep(delay)


@lru_cache
def get_openai_client() -> OpenAIClient:
    """Singleton-style accessor so one async SDK client is reused."""
    return OpenAIClient(get_settings())
