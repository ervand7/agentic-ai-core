"""
Domain ports for the AI tasks context.

Application services depend only on these abstractions, never on a
concrete LLM provider. This keeps the domain testable and lets us swap
providers (OpenAI, Anthropic, local model, fakes) in infrastructure.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional, Protocol

from app.shared.openai_types import ChatCompletionMessageParam, ResponseFormat


@dataclass(frozen=True)
class CompletionResult:
    """Provider-agnostic result of a single completion call."""

    content: str
    model: str
    tokens_used: int


@dataclass(frozen=True)
class StreamChunk:
    """Single token/delta yielded by a streaming completion."""

    content: str
    model: Optional[str]
    tokens_used: Optional[int]


class LLMProvider(Protocol):
    """Abstraction over an LLM provider used by AI task use cases."""

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
        """Run a non-streaming chat completion."""

    def stream(
        self,
        *,
        endpoint: str,
        request_id: str,
        messages: list[ChatCompletionMessageParam],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run a streaming chat completion."""
