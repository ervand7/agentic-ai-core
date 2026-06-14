"""Unit tests for the ai_tasks structured-output response format schemas."""

from typing import Any, cast

from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

from app.domains.ai_tasks.domain.response_formats import (
    AGENT_CRITIQUE_RESPONSE_FORMAT,
    ANALYZE_TEXT_RESPONSE_FORMAT,
    CLASSIFY_RESPONSE_FORMAT,
    KEYWORDS_RESPONSE_FORMAT,
    RESEARCH_PLAN_RESPONSE_FORMAT,
)


def _object_schema(fmt: ResponseFormatJSONSchema) -> dict[str, Any]:
    """OpenAI stubs type nested JSON Schema fields as ``object``; cast for tests."""
    return fmt["json_schema"]["schema"]


def test_all_formats_are_json_schema_type():
    for fmt in (
        CLASSIFY_RESPONSE_FORMAT,
        KEYWORDS_RESPONSE_FORMAT,
        ANALYZE_TEXT_RESPONSE_FORMAT,
        RESEARCH_PLAN_RESPONSE_FORMAT,
        AGENT_CRITIQUE_RESPONSE_FORMAT,
    ):
        assert fmt["type"] == "json_schema"
        assert "json_schema" in fmt
        assert fmt["json_schema"]["strict"] is True


def test_research_plan_schema_shape():
    schema = _object_schema(RESEARCH_PLAN_RESPONSE_FORMAT)
    assert schema["required"] == ["steps"]
    properties = cast(dict[str, Any], schema["properties"])
    assert properties["steps"]["type"] == "array"


def test_agent_critique_schema_required_fields():
    schema = _object_schema(AGENT_CRITIQUE_RESPONSE_FORMAT)
    assert set(schema["required"]) == {"approved", "issues", "revised_answer"}
    assert schema["additionalProperties"] is False


def test_classify_schema_required_fields():
    schema = _object_schema(CLASSIFY_RESPONSE_FORMAT)
    assert set(schema["required"]) == {"sentiment", "summary", "keywords"}
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, Any], schema["properties"])
    assert properties["sentiment"]["enum"] == ["positive", "negative", "neutral"]


def test_keywords_schema_shape():
    schema = _object_schema(KEYWORDS_RESPONSE_FORMAT)
    assert schema["required"] == ["keywords"]
    properties = cast(dict[str, Any], schema["properties"])
    assert properties["keywords"]["type"] == "array"


def test_analyze_text_schema_required_fields():
    schema = _object_schema(ANALYZE_TEXT_RESPONSE_FORMAT)
    assert set(schema["required"]) == {"summary", "sentiment", "keywords", "language"}
