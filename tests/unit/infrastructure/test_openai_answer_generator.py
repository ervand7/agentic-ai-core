"""Unit tests for OpenAIAnswerGenerator (OpenAIClient mocked)."""

from unittest.mock import AsyncMock

from app.domains.documents.infrastructure.openai_answer_generator import (
    OpenAIAnswerGenerator,
)
from tests.conftest import make_chat_completion


async def test_maps_completion_to_generated_answer():
    client = AsyncMock()
    client.create_chat_completion.return_value = make_chat_completion(
        content="grounded answer [1]", model="gpt-4o-mini", total_tokens=33
    )
    generator = OpenAIAnswerGenerator(client)

    result = await generator.generate(
        request_id="r1",
        system_prompt="SYS",
        user_prompt="USER",
        temperature=0.2,
        max_tokens=100,
    )

    assert result.content == "grounded answer [1]"
    assert result.model == "gpt-4o-mini"
    assert result.tokens_used == 33


async def test_builds_system_and_user_messages():
    client = AsyncMock()
    client.create_chat_completion.return_value = make_chat_completion()
    generator = OpenAIAnswerGenerator(client)

    await generator.generate(request_id="r1", system_prompt="SYS", user_prompt="USER")

    kwargs = client.create_chat_completion.call_args.kwargs
    assert kwargs["endpoint"] == "documents-ask"
    assert kwargs["request_id"] == "r1"
    messages = kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "USER"}


async def test_none_content_becomes_empty_string():
    client = AsyncMock()
    client.create_chat_completion.return_value = make_chat_completion(content=None)
    generator = OpenAIAnswerGenerator(client)

    result = await generator.generate(
        request_id="r", system_prompt="s", user_prompt="u"
    )
    assert result.content == ""


async def test_missing_usage_yields_zero_tokens():
    client = AsyncMock()
    client.create_chat_completion.return_value = make_chat_completion(total_tokens=None)
    generator = OpenAIAnswerGenerator(client)

    result = await generator.generate(
        request_id="r", system_prompt="s", user_prompt="u"
    )
    assert result.tokens_used == 0
