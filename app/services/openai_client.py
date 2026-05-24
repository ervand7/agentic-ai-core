"""Thin async OpenAI client wrapper with retries and timeout handling."""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Central place for all direct OpenAI SDK calls."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

    async def create_chat_completion(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[dict[str, str]],
        response_format: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Execute a chat completion with retries and exponential backoff.

        Retries are intentionally simple and readable for beginner projects.
        """

        max_retries = self.settings.OPENAI_MAX_RETRIES
        base_delay = self.settings.OPENAI_RETRY_BASE_DELAY_SECONDS
        start_time = time.perf_counter()

        for attempt in range(max_retries + 1):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.settings.OPENAI_MODEL,
                    messages=messages,
                    temperature=(
                        self.settings.OPENAI_TEMPERATURE
                        if temperature is None
                        else temperature
                    ),
                    max_tokens=self.settings.OPENAI_MAX_TOKENS
                    if max_tokens is None
                    else max_tokens,
                    response_format=response_format,
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
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
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
                # 429 can happen during bursts. We retry briefly before failing cleanly.
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
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
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
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

    async def create_chat_completion_stream(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Create a streaming completion with the same retry behavior."""
        max_retries = self.settings.OPENAI_MAX_RETRIES
        base_delay = self.settings.OPENAI_RETRY_BASE_DELAY_SECONDS

        for attempt in range(max_retries + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self.settings.OPENAI_MODEL,
                    messages=messages,
                    temperature=(
                        self.settings.OPENAI_TEMPERATURE
                        if temperature is None
                        else temperature
                    ),
                    max_tokens=self.settings.OPENAI_MAX_TOKENS
                    if max_tokens is None
                    else max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except APITimeoutError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMTimeoutError(
                    "The AI provider timed out. Please try again shortly."
                ) from exc
            except RateLimitError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMRateLimitError(
                    "Rate limit reached. Please wait a moment and retry."
                ) from exc
            except (APIConnectionError, InternalServerError) as exc:
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMTemporaryError(
                    "Temporary AI provider issue. Please retry shortly."
                ) from exc
            except OpenAIError as exc:
                raise LLMServiceError("AI provider returned an unexpected error.") from exc

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
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMTimeoutError(
                    "The AI provider timed out. Please try again shortly."
                ) from exc
            except RateLimitError as exc:
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMRateLimitError(
                    "Rate limit reached. Please wait a moment and retry."
                ) from exc
            except (APIConnectionError, InternalServerError) as exc:
                if attempt < max_retries:
                    await self._retry_sleep(attempt, base_delay, request_id, endpoint, exc)
                    continue
                raise LLMTemporaryError(
                    "Temporary AI provider issue. Please retry shortly."
                ) from exc
            except OpenAIError as exc:
                raise LLMServiceError("AI provider returned an unexpected error.") from exc

    async def _retry_sleep(
        self,
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
