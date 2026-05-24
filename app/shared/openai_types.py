"""OpenAI SDK type aliases reused by bounded contexts and infrastructure."""

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat

__all__ = ["ChatCompletionMessageParam", "ResponseFormat"]
