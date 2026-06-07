"""Unit tests for ai_tasks API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.domains.ai_tasks.api.schemas import (
    AskRequest,
    ClassifyResponse,
    TranslateRequest,
)


def test_ask_request_requires_non_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")
    assert AskRequest(question="hi").question == "hi"


class TestTranslateRequest:
    def test_valid(self):
        req = TranslateRequest(text="hello", target_language="Spanish")
        assert req.target_language == "Spanish"

    def test_target_language_min_length(self):
        with pytest.raises(ValidationError):
            TranslateRequest(text="hello", target_language="x")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            TranslateRequest(text="", target_language="Spanish")


class TestClassifyResponse:
    def test_valid_sentiment(self):
        resp = ClassifyResponse(sentiment="positive", summary="s", keywords=["a"])
        assert resp.sentiment == "positive"

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(ValidationError):
            ClassifyResponse.model_validate(
                {"sentiment": "happy", "summary": "s", "keywords": []}
            )
