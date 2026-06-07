"""OpenAI-backed implementation of the AnswerGenerator port."""

import logging
from typing import Optional

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.domains.documents.domain.models import GeneratedAnswer
from app.domains.documents.domain.ports import AnswerGenerator
from app.shared.infrastructure.openai_client import OpenAIClient
from app.shared.openai_types import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class OpenAIAnswerGenerator(AnswerGenerator):
    """Generate grounded answers via the shared OpenAI client.

    This is intentionally thin (like the embedding provider): it maps the
    domain's `system_prompt` + `user_prompt` onto chat messages and returns a
    provider-agnostic `GeneratedAnswer`.
    """

    def __init__(self, client: OpenAIClient):
        self._client = client

    async def generate(
        self,
        *,
        request_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GeneratedAnswer:
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_prompt,
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]

        completion = await self._client.create_chat_completion(
            endpoint="documents-ask",
            request_id=request_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = completion.usage
        return GeneratedAnswer(
            content=completion.choices[0].message.content or "",
            model=completion.model,
            tokens_used=usage.total_tokens if usage else 0,
        )
