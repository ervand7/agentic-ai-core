"""Unit tests for the shared LLM exception hierarchy."""

import pytest

from app.shared.exceptions import (
    LLMRateLimitError,
    LLMServiceError,
    LLMTemporaryError,
    LLMTimeoutError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [LLMTimeoutError, LLMRateLimitError, LLMTemporaryError],
)
def test_subclasses_of_llm_service_error(exc_cls):
    assert issubclass(exc_cls, LLMServiceError)


def test_service_error_is_exception():
    assert issubclass(LLMServiceError, Exception)


def test_message_is_preserved():
    err = LLMTimeoutError("timed out")
    assert str(err) == "timed out"


def test_can_be_caught_as_base():
    with pytest.raises(LLMServiceError):
        raise LLMRateLimitError("rate")
