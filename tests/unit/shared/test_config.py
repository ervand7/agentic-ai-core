"""Unit tests for application Settings and get_settings caching."""

import pytest
from pydantic import ValidationError

from app.shared.config import Settings, get_settings


class TestDefaults:
    def test_sensible_defaults(self):
        s = Settings(OPENAI_API_KEY="k")
        assert s.OPENAI_MODEL == "gpt-4o-mini"
        assert s.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"
        assert s.RAG_TOP_K == 4
        assert s.DOCUMENT_CHUNK_SIZE == 500
        assert s.QDRANT_COLLECTION_NAME == "documents"

    def test_prompt_defaults_non_empty(self):
        s = Settings(OPENAI_API_KEY="k")
        assert s.PROMPT_RAG_SYSTEM.strip() != ""
        assert s.PROMPT_ASK_SYSTEM.strip() != ""


class TestValidation:
    @pytest.mark.parametrize("temp", [-0.1, 2.1])
    def test_temperature_bounds(self, temp):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", OPENAI_TEMPERATURE=temp)

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", OPENAI_MAX_TOKENS=0)

    @pytest.mark.parametrize("k", [0, 21])
    def test_rag_top_k_bounds(self, k):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", RAG_TOP_K=k)

    def test_min_similarity_bounds(self):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", RAG_MIN_SIMILARITY=1.5)

    def test_chunk_size_minimum(self):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", DOCUMENT_CHUNK_SIZE=10)

    def test_max_retries_upper_bound(self):
        with pytest.raises(ValidationError):
            Settings(OPENAI_API_KEY="k", OPENAI_MAX_RETRIES=6)


def test_get_settings_is_cached():
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
