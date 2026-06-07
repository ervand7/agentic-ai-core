"""Smoke tests for app wiring (routes, static, web UI, error handlers)."""

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_app_metadata():
    assert app.title == "AI Backend"


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_openapi_lists_document_and_ai_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/documents/upload" in paths
    assert "/documents/search" in paths
    assert "/documents/ask" in paths
    assert "/ask" in paths
    assert "/classify" in paths
    assert "/health" in paths


def test_document_error_handlers_registered():
    from app.domains.documents.application.services import (
        DocumentValidationError,
        EmptyVectorStoreError,
    )

    assert DocumentValidationError in app.exception_handlers
    assert EmptyVectorStoreError in app.exception_handlers


def test_llm_error_handlers_registered():
    from app.shared.exceptions import LLMServiceError, LLMTimeoutError

    assert LLMServiceError in app.exception_handlers
    assert LLMTimeoutError in app.exception_handlers
