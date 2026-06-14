from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, cast

from app.domains.ai_tasks.domain.ports import ToolCompletionResult
from app.shared.exceptions import LLMServiceError
from app.shared.openai_types import ChatCompletionMessageParam


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse a model JSON response, guarding routes from malformed output."""
    try:
        data = json.loads(content or "{}")
    except JSONDecodeError as exc:
        raise LLMServiceError("Model returned invalid JSON output.") from exc
    if not isinstance(data, dict):
        raise LLMServiceError("Model returned non-object JSON output.")
    return data


def assistant_tool_call_message(
        result: ToolCompletionResult,
) -> ChatCompletionMessageParam:
    """Rebuild the assistant turn that requested tool calls.

    The provider needs the original ``assistant`` message (with its
    ``tool_calls``) echoed back before the matching ``tool`` results, or
    it cannot correlate the results with the calls it made.
    """
    message = {
        "role": "assistant",
        "content": result.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in result.tool_calls
        ],
    }
    return cast(ChatCompletionMessageParam, cast(object, message))


def tool_result_message(
        tool_call_id: str, result: dict[str, Any]
) -> ChatCompletionMessageParam:
    """Wrap one tool's result as a ``tool`` message for the next turn."""
    message = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result),
    }
    return cast(ChatCompletionMessageParam, cast(object, message))


__all__ = [
    "assistant_tool_call_message",
    "parse_json_content",
    "tool_result_message",
]
