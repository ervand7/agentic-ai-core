"""API tests for the documents router using FastAPI dependency overrides."""

from typing import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.domains.documents.api.dependencies import (
    get_answer_question_service,
    get_ingest_document_service,
    get_search_documents_service,
)
from app.domains.documents.api.schemas import (
    DocumentSearchResponse,
    DocumentUploadResponse,
    SearchResult,
)
from app.domains.documents.application.services import (
    DocumentValidationError,
    EmptyVectorStoreError,
)
from app.domains.documents.domain.models import Citation, RagAnswer
from app.main import app
from tests.conftest import make_execute_service_mock


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestUpload:
    def test_rejects_unsupported_extension(self, client):
        app.dependency_overrides[get_ingest_document_service] = lambda: AsyncMock()
        resp = client.post(
            "/documents/upload",
            files={"file": ("notes.docx", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "supported" in resp.json()["detail"].lower()

    def test_rejects_invalid_utf8_txt(self, client):
        app.dependency_overrides[get_ingest_document_service] = lambda: AsyncMock()
        resp = client.post(
            "/documents/upload",
            files={"file": ("bad.txt", b"\xff\xfe\x00", "text/plain")},
        )
        assert resp.status_code == 400
        assert "UTF-8" in resp.json()["detail"]

    def test_successful_upload(self, client):
        service = make_execute_service_mock(
            result=DocumentUploadResponse(
                filename="a.txt", chunks_stored=2, total_characters=11
            )
        )
        app.dependency_overrides[get_ingest_document_service] = lambda: service
        resp = client.post(
            "/documents/upload",
            files={"file": ("a.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "filename": "a.txt",
            "chunks_stored": 2,
            "total_characters": 11,
        }
        service.execute.assert_awaited_once()
        call_kwargs = service.execute.await_args.kwargs
        assert call_kwargs["filename"] == "a.txt"
        assert call_kwargs["content"] == "hello world"

    def test_validation_error_maps_to_400(self, client):
        service = make_execute_service_mock(
            error=DocumentValidationError("Uploaded file is empty.")
        )
        app.dependency_overrides[get_ingest_document_service] = lambda: service
        resp = client.post(
            "/documents/upload",
            files={"file": ("a.txt", b"   ", "text/plain")},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Uploaded file is empty."

    def test_request_id_passed_through(self, client):
        service = make_execute_service_mock(
            result=DocumentUploadResponse(
                filename="a.txt", chunks_stored=0, total_characters=0
            )
        )
        app.dependency_overrides[get_ingest_document_service] = lambda: service
        client.post(
            "/documents/upload",
            files={"file": ("a.txt", b"hi", "text/plain")},
            headers={"x-request-id": "trace-123"},
        )
        assert service.execute.await_args.kwargs["request_id"] == "trace-123"


class TestSearch:
    def test_returns_results(self, client):
        service = make_execute_service_mock(
            result=DocumentSearchResponse(
                query="cats",
                results=[
                    SearchResult(
                        text="cats are great", filename="a.txt", similarity=0.9
                    )
                ],
            )
        )
        app.dependency_overrides[get_search_documents_service] = lambda: service
        resp = client.post("/documents/search", json={"query": "cats", "top_k": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "cats"
        assert body["results"][0]["filename"] == "a.txt"
        assert service.execute.await_args.kwargs["top_k"] == 2

    def test_empty_query_rejected_by_validation(self, client):
        app.dependency_overrides[get_search_documents_service] = lambda: AsyncMock()
        resp = client.post("/documents/search", json={"query": ""})
        assert resp.status_code == 422

    def test_empty_store_maps_to_400(self, client):
        service = make_execute_service_mock(
            error=EmptyVectorStoreError("No documents uploaded yet.")
        )
        app.dependency_overrides[get_search_documents_service] = lambda: service
        resp = client.post("/documents/search", json={"query": "cats"})
        assert resp.status_code == 400
        assert "No documents" in resp.json()["detail"]


class TestAsk:
    def test_answer_with_citations(self, client):
        service = make_execute_service_mock(
            result=RagAnswer(
                answer="Cats are great [1].",
                citations=[
                    Citation(
                        index=1, filename="a.txt", text="cats", similarity=0.876543
                    )
                ],
                used_context=True,
                model="gpt-4o-mini",
                tokens_used=20,
            )
        )
        app.dependency_overrides[get_answer_question_service] = lambda: service
        resp = client.post("/documents/ask", json={"question": "Are cats great?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Cats are great [1]."
        assert body["used_context"] is True
        assert body["model"] == "gpt-4o-mini"
        assert body["tokens_used"] == 20
        assert body["citations"][0]["similarity"] == 0.8765

    def test_no_context_answer(self, client):
        service = make_execute_service_mock(
            result=RagAnswer(
                answer="I don't know.",
                citations=[],
                used_context=False,
                model=None,
                tokens_used=0,
            )
        )
        app.dependency_overrides[get_answer_question_service] = lambda: service
        resp = client.post("/documents/ask", json={"question": "Unknown?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["used_context"] is False
        assert body["citations"] == []
        assert body["model"] is None

    def test_empty_question_rejected(self, client):
        app.dependency_overrides[get_answer_question_service] = lambda: AsyncMock()
        resp = client.post("/documents/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_empty_store_maps_to_400(self, client):
        service = make_execute_service_mock(
            error=EmptyVectorStoreError("No documents uploaded yet.")
        )
        app.dependency_overrides[get_answer_question_service] = lambda: service
        resp = client.post("/documents/ask", json={"question": "hi"})
        assert resp.status_code == 400
