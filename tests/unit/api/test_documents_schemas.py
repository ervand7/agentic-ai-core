"""Unit tests for documents API Pydantic schemas (validation rules)."""

import pytest
from pydantic import ValidationError

from app.domains.documents.api.schemas import (
    DocumentAskRequest,
    DocumentSearchRequest,
    DocumentUploadResponse,
)


class TestDocumentSearchRequest:
    def test_defaults(self):
        req = DocumentSearchRequest(query="hi")
        assert req.top_k == 3
        assert req.filename is None
        assert req.keyword is None
        assert req.min_similarity is None

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="")

    @pytest.mark.parametrize("top_k", [0, 11])
    def test_top_k_bounds(self, top_k):
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="q", top_k=top_k)

    @pytest.mark.parametrize("sim", [-0.1, 1.1])
    def test_min_similarity_bounds(self, sim):
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="q", min_similarity=sim)


class TestDocumentAskRequest:
    def test_defaults_allow_none_top_k(self):
        req = DocumentAskRequest(question="why?")
        assert req.top_k is None
        assert req.min_similarity is None

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            DocumentAskRequest(question="")

    @pytest.mark.parametrize("top_k", [0, 21])
    def test_top_k_upper_bound(self, top_k):
        with pytest.raises(ValidationError):
            DocumentAskRequest(question="q", top_k=top_k)


def test_upload_response_roundtrip():
    resp = DocumentUploadResponse(
        filename="a.txt", chunks_stored=3, total_characters=120
    )
    assert resp.model_dump() == {
        "filename": "a.txt",
        "chunks_stored": 3,
        "total_characters": 120,
    }
