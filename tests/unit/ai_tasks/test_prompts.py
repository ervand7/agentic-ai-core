"""Unit tests for the ai_tasks prompt registry."""

import dataclasses

import pytest

from app.domains.ai_tasks.domain.prompts import (
    PromptTemplate,
    get_prompts,
)


class TestPromptTemplate:
    def test_prompt_version_combines_name_and_version(self):
        tpl = PromptTemplate(name="ask", version="v1", system_prompt="hi")
        assert tpl.prompt_version == "ask_v1"

    def test_is_frozen(self):
        tpl = PromptTemplate(name="ask", version="v1", system_prompt="hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(tpl, "name", "other")


class TestGetPrompts:
    def test_contains_expected_keys(self):
        prompts = get_prompts()
        for key in [
            "ask_v1",
            "ask_stream_v1",
            "classify_v1",
            "summarize_v1",
            "extract_keywords_v1",
            "translate_v1",
            "analyze_text_v1",
        ]:
            assert key in prompts

    def test_each_prompt_has_non_empty_system_prompt(self):
        for tpl in get_prompts().values():
            assert tpl.system_prompt.strip() != ""

    def test_is_cached(self):
        assert get_prompts() is get_prompts()

    def test_analyze_text_prompt_version(self):
        assert get_prompts()["analyze_text_v1"].prompt_version == "analyze_text_v1"
