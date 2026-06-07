"""In-memory fake adapters for integration tests.

These stand in for the OpenAI-backed ports so the *real* application services,
domain logic, routers, and vector store run end to end without any network
calls. They are intentionally deterministic so assertions stay stable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional

from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.ports import CompletionResult, StreamChunk
from app.domains.documents.domain.models import GeneratedAnswer
from app.shared.openai_types import ChatCompletionMessageParam, ResponseFormat

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUESTION_RE = re.compile(r"Question:\s*(.+?)\n", re.DOTALL)


def _stable_hash(token: str) -> int:
    """Process-stable hash (Python's ``hash`` is salted per run for strings)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


@dataclass
class FakeEmbeddingProvider:
    """Deterministic bag-of-words hashing embedder.

    Texts that share vocabulary produce vectors pointing in similar directions,
    so cosine similarity behaves like a crude-but-real semantic signal. That is
    enough to exercise retrieval, ranking, and similarity thresholds without
    calling OpenAI.
    """

    dim: int = 64
    calls: list[str] = field(default_factory=list)

    async def embed(self, text: str, request_id: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            vector[_stable_hash(token) % self.dim] += 1.0
        if not any(vector):
            # Avoid a zero vector so cosine similarity is well defined.
            vector[0] = 1.0
        return vector


@dataclass
class FakeAnswerGenerator:
    """Deterministic RAG answer generator.

    Echoes the question parsed out of the assembled user prompt and always
    cites ``[1]``, which lets tests assert that context injection and the RAG
    prompt assembly actually happened.
    """

    model: str = "fake-answer-model"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeneratedAnswer:
        self.calls.append(
            {
                "request_id": request_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        match = _QUESTION_RE.search(user_prompt)
        question = match.group(1).strip() if match else ""
        content = f"Based on the provided context [1], answering: {question}"
        return GeneratedAnswer(
            content=content,
            model=self.model,
            tokens_used=len(user_prompt.split()),
        )


def _last_user_text(messages: list[ChatCompletionMessageParam]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


@dataclass
class FakeLLMProvider:
    """Deterministic LLM provider for the ai_tasks use cases.

    Produces plain-text echoes for free-form endpoints and schema-valid JSON
    for the structured-output endpoints (classify / extract-keywords /
    analyze-text), so the real services can parse and validate results.
    """

    model: str = "fake-llm-model"
    calls: list[dict[str, Any]] = field(default_factory=list)

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
        self.calls.append({"endpoint": endpoint, "request_id": request_id})
        user_text = _last_user_text(messages)
        content = self._content_for(endpoint, user_text)
        return CompletionResult(
            content=content,
            model=self.model,
            tokens_used=max(1, len(user_text.split())),
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
        self.calls.append({"endpoint": endpoint, "request_id": request_id})
        for token in ("Hel", "lo ", "world"):
            yield StreamChunk(content=token, model=self.model, tokens_used=None)
        yield StreamChunk(content="", model=self.model, tokens_used=7)

    def _content_for(self, endpoint: str, user_text: str) -> str:
        keywords = _TOKEN_RE.findall(user_text.lower())[:3]
        if endpoint == Endpoint.CLASSIFY:
            return json.dumps(
                {
                    "sentiment": "positive",
                    "summary": f"summary of: {user_text}",
                    "keywords": keywords,
                }
            )
        if endpoint == Endpoint.EXTRACT_KEYWORDS:
            return json.dumps({"keywords": keywords})
        if endpoint == Endpoint.ANALYZE_TEXT:
            return json.dumps(
                {
                    "summary": f"summary of: {user_text}",
                    "sentiment": "neutral",
                    "keywords": keywords,
                    "language": "en",
                }
            )
        # Free-form endpoints (ask / summarize / translate) echo the input.
        return f"echo: {user_text}"
