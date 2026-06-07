"""Tier 1 integration: full documents pipeline (upload -> search -> ask).

The real services, chunking, RAG prompt assembly, citation building, and the
real ``InMemoryVectorStore`` all run end to end through the HTTP layer. Only
the OpenAI embedding/answer ports are faked.
"""

import pytest

from app.domains.documents.domain.rag import NO_CONTEXT_ANSWER
from app.shared.config import get_settings

pytestmark = pytest.mark.integration


def _upload(harness, filename: str, text: str):
    return harness.client.post(
        "/documents/upload",
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )


class TestUploadThenSearch:
    def test_upload_persists_chunks_and_search_finds_them(self, documents_harness):
        resp = _upload(
            documents_harness,
            "python.txt",
            "Python is a programming language used for data science and web development.",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "python.txt"
        assert body["chunks_stored"] >= 1
        # Real store actually received the chunks.
        assert documents_harness.store.count() == body["chunks_stored"]
        # Real embedding port was exercised once per chunk.
        assert len(documents_harness.embeddings.calls) == body["chunks_stored"]

        search = documents_harness.client.post(
            "/documents/search",
            json={"query": "python programming language", "top_k": 5},
        )
        assert search.status_code == 200
        results = search.json()["results"]
        assert results, "expected at least one semantic hit"
        assert results[0]["filename"] == "python.txt"
        assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_search_ranks_more_relevant_document_first(self, documents_harness):
        _upload(
            documents_harness,
            "python.txt",
            "Python programming language tutorial for beginners.",
        )
        _upload(
            documents_harness,
            "cooking.txt",
            "Cooking pasta recipe with tomato sauce in the kitchen.",
        )

        search = documents_harness.client.post(
            "/documents/search",
            json={"query": "python programming tutorial", "top_k": 5},
        )
        results = search.json()["results"]
        assert results[0]["filename"] == "python.txt"

    def test_filename_filter_restricts_results(self, documents_harness):
        _upload(documents_harness, "a.txt", "alpha beta gamma delta epsilon")
        _upload(documents_harness, "b.txt", "alpha beta gamma delta epsilon")

        search = documents_harness.client.post(
            "/documents/search",
            json={"query": "alpha beta", "top_k": 5, "filename": "b.txt"},
        )
        results = search.json()["results"]
        assert results
        assert {r["filename"] for r in results} == {"b.txt"}

    def test_search_before_any_upload_returns_400(self, documents_harness):
        search = documents_harness.client.post(
            "/documents/search", json={"query": "anything"}
        )
        assert search.status_code == 400
        assert "No documents" in search.json()["detail"]


class TestAskFlow:
    def test_ask_uses_context_and_builds_citations(self, documents_harness):
        _upload(
            documents_harness,
            "facts.txt",
            "The Eiffel Tower is located in Paris, the capital of France.",
        )

        resp = documents_harness.client.post(
            "/documents/ask",
            json={
                "question": "Where is the Eiffel Tower located in Paris France?",
                "min_similarity": 0.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["used_context"] is True
        assert body["model"] == "fake-answer-model"
        assert "[1]" in body["answer"]
        assert body["citations"], "expected at least one citation"
        assert body["citations"][0]["index"] == 1
        assert body["citations"][0]["filename"] == "facts.txt"
        assert body["tokens_used"] > 0

        # The real RAG service passed the configured system prompt through.
        call = documents_harness.answer_generator.calls[-1]
        assert call["system_prompt"] == get_settings().PROMPT_RAG_SYSTEM
        assert "Eiffel Tower" in call["user_prompt"]

    def test_ask_abstains_when_nothing_relevant(self, documents_harness):
        _upload(documents_harness, "facts.txt", "Completely unrelated content here.")

        resp = documents_harness.client.post(
            "/documents/ask",
            # Impossible threshold => no hits => abstain, never calls the LLM.
            json={"question": "quantum chromodynamics", "min_similarity": 0.999},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["used_context"] is False
        assert body["answer"] == NO_CONTEXT_ANSWER
        assert body["citations"] == []
        assert body["model"] is None
        assert documents_harness.answer_generator.calls == []

    def test_ask_before_any_upload_returns_400(self, documents_harness):
        resp = documents_harness.client.post(
            "/documents/ask", json={"question": "hello?"}
        )
        assert resp.status_code == 400


class TestValidation:
    def test_unsupported_extension_rejected_before_service(self, documents_harness):
        resp = documents_harness.client.post(
            "/documents/upload",
            files={"file": ("notes.docx", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert documents_harness.store.count() == 0

    def test_empty_document_rejected_by_service(self, documents_harness):
        resp = _upload(documents_harness, "empty.txt", "   \n  ")
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()
