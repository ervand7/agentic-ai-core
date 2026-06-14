"""
Structured-output JSON schemas for AI tasks.

Each task that requires a strict JSON response defines its expected
shape here. The schemas are wrapped in OpenAI's `ResponseFormatJSONSchema`
envelope so application services can pass them straight to the LLM port.

Kept next to `prompts.py` because both describe the contract of a task:
prompts shape the input, response formats shape the output.
"""

from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

_CLASSIFY_JSON_SCHEMA: JSONSchema = {
    "name": "classify_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "summary": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sentiment", "summary", "keywords"],
        "additionalProperties": False,
    },
}

_KEYWORDS_JSON_SCHEMA: JSONSchema = {
    "name": "keywords_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
}

_RESEARCH_PLAN_JSON_SCHEMA: JSONSchema = {
    "name": "research_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
}

_AGENT_CRITIQUE_JSON_SCHEMA: JSONSchema = {
    "name": "agent_critique",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "revised_answer": {"type": "string"},
        },
        "required": ["approved", "issues", "revised_answer"],
        "additionalProperties": False,
    },
}

_ANALYZE_TEXT_JSON_SCHEMA: JSONSchema = {
    "name": "analyze_text_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "keywords": {"type": "array", "items": {"type": "string"}},
            "language": {"type": "string"},
        },
        "required": ["summary", "sentiment", "keywords", "language"],
        "additionalProperties": False,
    },
}

CLASSIFY_RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": _CLASSIFY_JSON_SCHEMA,
}

RESEARCH_PLAN_RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": _RESEARCH_PLAN_JSON_SCHEMA,
}

AGENT_CRITIQUE_RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": _AGENT_CRITIQUE_JSON_SCHEMA,
}

KEYWORDS_RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": _KEYWORDS_JSON_SCHEMA,
}

ANALYZE_TEXT_RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": _ANALYZE_TEXT_JSON_SCHEMA,
}

__all__ = [
    "AGENT_CRITIQUE_RESPONSE_FORMAT",
    "ANALYZE_TEXT_RESPONSE_FORMAT",
    "CLASSIFY_RESPONSE_FORMAT",
    "KEYWORDS_RESPONSE_FORMAT",
    "RESEARCH_PLAN_RESPONSE_FORMAT",
]
