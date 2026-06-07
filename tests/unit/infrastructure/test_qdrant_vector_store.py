"""Unit tests for QdrantVectorStore with a fully mocked QdrantClient."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from app.domains.documents.infrastructure import qdrant_vector_store as qmod
from app.domains.documents.infrastructure.qdrant_vector_store import QdrantVectorStore


@pytest.fixture
def client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def store(client):
    with patch.object(qmod, "QdrantClient", return_value=client):
        s = QdrantVectorStore(url="http://localhost:6333", collection_name="docs")
    return s


def _point(text, filename, score, vector):
    return SimpleNamespace(
        payload={"text": text, "filename": filename}, score=score, vector=vector
    )


class TestAddChunks:
    def test_mismatched_lengths_raise(self, store):
        with pytest.raises(ValueError, match="same length"):
            store.add_chunks(filename="f", chunks=["a"], embeddings=[[1.0], [2.0]])

    def test_empty_chunks_is_noop(self, store, client):
        store.add_chunks(filename="f", chunks=[], embeddings=[])
        client.upsert.assert_not_called()
        client.create_collection.assert_not_called()

    def test_empty_vectors_raise(self, store):
        with pytest.raises(ValueError, match="must not be empty"):
            store.add_chunks(filename="f", chunks=["a"], embeddings=[[]])

    def test_creates_collection_when_missing_then_upserts(self, store, client):
        client.collection_exists.return_value = False
        store.add_chunks(
            filename="doc.txt",
            chunks=["a", "b"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        client.create_collection.assert_called_once()
        client.upsert.assert_called_once()
        kwargs = client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "docs"
        assert len(kwargs["points"]) == 2

    def test_skips_create_when_collection_exists(self, store, client):
        client.collection_exists.return_value = True
        store.add_chunks(filename="d", chunks=["a"], embeddings=[[1.0, 2.0]])
        client.create_collection.assert_not_called()
        client.upsert.assert_called_once()


class TestCount:
    def test_returns_zero_when_collection_missing(self, store, client):
        client.collection_exists.return_value = False
        assert store.count() == 0
        client.count.assert_not_called()

    def test_returns_client_count(self, store, client):
        client.collection_exists.return_value = True
        client.count.return_value = SimpleNamespace(count=7)
        assert store.count() == 7

    def test_collection_exists_error_treated_as_missing(self, store, client):
        client.collection_exists.side_effect = ValueError("nope")
        assert store.count() == 0


class TestAll:
    def test_empty_when_collection_missing(self, store, client):
        client.collection_exists.return_value = False
        assert store.all() == []

    def test_maps_records_to_stored_chunks(self, store, client):
        client.collection_exists.return_value = True
        client.scroll.return_value = (
            [
                SimpleNamespace(
                    payload={"text": "hello", "filename": "a.txt"},
                    vector=[0.1, 0.2],
                )
            ],
            None,
        )
        chunks = store.all()
        assert len(chunks) == 1
        assert chunks[0].text == "hello"
        assert chunks[0].filename == "a.txt"
        assert chunks[0].embedding == [0.1, 0.2]


class TestSearch:
    def test_non_positive_top_k_returns_empty(self, store, client):
        client.collection_exists.return_value = True
        assert store.search([0.1], top_k=0) == []
        client.query_points.assert_not_called()

    def test_missing_collection_returns_empty(self, store, client):
        client.collection_exists.return_value = False
        assert store.search([0.1], top_k=5) == []

    def test_returns_sorted_hits(self, store, client):
        client.collection_exists.return_value = True
        client.query_points.return_value = SimpleNamespace(
            points=[
                _point("low", "a.txt", 0.3, [0.1]),
                _point("high", "a.txt", 0.9, [0.2]),
            ]
        )
        hits = store.search([0.1], top_k=5)
        assert [h.chunk.text for h in hits] == ["high", "low"]
        assert hits[0].similarity == pytest.approx(0.9)

    def test_min_similarity_filters_vector_score(self, store, client):
        client.collection_exists.return_value = True
        client.query_points.return_value = SimpleNamespace(
            points=[
                _point("keep", "a.txt", 0.8, [0.1]),
                _point("drop", "a.txt", 0.1, [0.2]),
            ]
        )
        hits = store.search([0.1], top_k=5, min_similarity=0.5)
        assert [h.chunk.text for h in hits] == ["keep"]

    def test_keyword_drops_non_matching_and_blends_score(self, store, client):
        client.collection_exists.return_value = True
        client.query_points.return_value = SimpleNamespace(
            points=[
                _point("the cat sat", "a.txt", 1.0, [0.1]),
                _point("a dog ran", "a.txt", 1.0, [0.2]),
            ]
        )
        hits = store.search([0.1], top_k=5, keyword="cat")
        assert len(hits) == 1
        assert hits[0].chunk.text == "the cat sat"
        # hybrid: 0.8 * vector(1.0) + 0.2 * keyword(1.0) = 1.0
        assert hits[0].similarity == pytest.approx(1.0)

    def test_keyword_overfetch_candidate_limit(self, store, client):
        client.collection_exists.return_value = True
        client.query_points.return_value = SimpleNamespace(points=[])
        store.search([0.1], top_k=10, keyword="x")
        # top_k * KEYWORD_CANDIDATE_MULTIPLIER (5) = 50, capped at 200
        assert client.query_points.call_args.kwargs["limit"] == 50

    def test_no_keyword_uses_top_k_as_limit(self, store, client):
        client.collection_exists.return_value = True
        client.query_points.return_value = SimpleNamespace(points=[])
        store.search([0.1], top_k=3)
        assert client.query_points.call_args.kwargs["limit"] == 3


class TestStaticHelpers:
    def test_build_filename_filter_none_for_blank(self):
        assert QdrantVectorStore._build_filename_filter(None) is None
        assert QdrantVectorStore._build_filename_filter("   ") is None

    def test_build_filename_filter_builds_condition(self):
        flt = QdrantVectorStore._build_filename_filter("a.txt")
        assert flt is not None
        conditions = cast(list[models.FieldCondition], flt.must)
        field = conditions[0]
        match = cast(models.MatchValue, field.match)
        assert field.key == "filename"
        assert match.value == "a.txt"

    def test_keyword_score_zero_for_empty(self):
        assert QdrantVectorStore._keyword_score(text="anything", keyword="") == 0.0

    def test_keyword_score_phrase_and_tokens(self):
        # phrase present + all tokens present -> capped at MAX_SCORE (1.0)
        score = QdrantVectorStore._keyword_score(
            text="the quick brown fox", keyword="quick brown"
        )
        assert score == pytest.approx(1.0)

    def test_keyword_score_partial_tokens(self):
        # phrase "quick zzz" not present (0.0) + 1/2 tokens present
        score = QdrantVectorStore._keyword_score(
            text="the quick fox", keyword="quick zzz"
        )
        # 0.6*0 + 0.4*0.5 = 0.2
        assert score == pytest.approx(0.2)

    def test_hybrid_score_passthrough_when_no_keyword(self):
        assert (
            QdrantVectorStore._hybrid_score(
                vector_score=0.7, keyword_score=0.9, keyword_used=False
            )
            == 0.7
        )

    def test_hybrid_score_blend(self):
        result = QdrantVectorStore._hybrid_score(
            vector_score=1.0, keyword_score=0.5, keyword_used=True
        )
        assert result == pytest.approx(0.8 * 1.0 + 0.2 * 0.5)

    def test_payload_str_handles_missing_and_non_str(self):
        assert QdrantVectorStore._payload_str(None, "text") == ""
        assert QdrantVectorStore._payload_str({"text": 123}, "text") == ""
        assert QdrantVectorStore._payload_str({"text": "ok"}, "text") == "ok"

    def test_vector_list_handles_non_list(self):
        assert QdrantVectorStore._vector_list(None) == []
        assert QdrantVectorStore._vector_list([1, 2]) == [1.0, 2.0]

    def test_keyword_score_whitespace_keyword_returns_phrase_present(self):
        # Non-empty but token-less keyword falls back to the phrase score.
        assert QdrantVectorStore._keyword_score(text="abc", keyword=" ") == 0.0


class TestEnsurePayloadIndexes:
    def test_idempotent_when_already_ready(self, store, client):
        store._payload_indexes_ready = True
        store._ensure_payload_indexes()
        client.create_payload_index.assert_not_called()

    def test_swallows_errors_and_stays_not_ready(self, store, client):
        client.create_payload_index.side_effect = ValueError("collection missing")
        store._ensure_payload_indexes()  # must not raise
        assert store._payload_indexes_ready is False

    def test_marks_ready_on_success(self, store, client):
        store._ensure_payload_indexes()
        assert store._payload_indexes_ready is True
        client.create_payload_index.assert_called_once()
