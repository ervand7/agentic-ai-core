"""Unit tests for the InMemoryVectorStore adapter (real cosine search)."""

import pytest

from app.domains.documents.infrastructure.in_memory_vector_store import (
    InMemoryVectorStore,
)


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore()
    s.add_chunks(
        filename="a.txt",
        chunks=["cats are great", "dogs are loyal"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    s.add_chunks(
        filename="b.txt",
        chunks=["birds can fly"],
        embeddings=[[0.7, 0.7]],
    )
    return s


class TestAddAndCount:
    def test_count_reflects_added_chunks(self, store):
        assert store.count() == 3

    def test_empty_store_count_zero(self):
        assert InMemoryVectorStore().count() == 0

    def test_mismatched_lengths_raise(self):
        s = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            s.add_chunks(filename="f.txt", chunks=["a", "b"], embeddings=[[1.0]])

    def test_all_returns_copy_not_internal_list(self, store):
        chunks = store.all()
        assert len(chunks) == 3
        chunks.clear()
        assert store.count() == 3  # internal state untouched


class TestSearch:
    def test_returns_most_similar_first(self, store):
        hits = store.search([1.0, 0.0], top_k=3)
        assert hits[0].chunk.text == "cats are great"  # perfectly aligned
        # scores are sorted descending
        scores = [h.similarity for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, store):
        hits = store.search([1.0, 0.0], top_k=1)
        assert len(hits) == 1

    def test_filename_filter(self, store):
        hits = store.search([1.0, 0.0], top_k=10, filename_filter="b.txt")
        assert {h.chunk.filename for h in hits} == {"b.txt"}

    def test_filename_filter_is_stripped(self, store):
        hits = store.search([1.0, 0.0], top_k=10, filename_filter="  b.txt  ")
        assert {h.chunk.filename for h in hits} == {"b.txt"}

    def test_keyword_filter_substring_case_insensitive(self, store):
        hits = store.search([1.0, 0.0], top_k=10, keyword="DOGS")
        assert len(hits) == 1
        assert hits[0].chunk.text == "dogs are loyal"

    def test_keyword_filter_no_match_returns_empty(self, store):
        hits = store.search([1.0, 0.0], top_k=10, keyword="elephant")
        assert hits == []

    def test_min_similarity_filters_low_scores(self, store):
        # Query aligned to "cats" ([1,0]); "dogs" ([0,1]) has similarity 0.
        hits = store.search([1.0, 0.0], top_k=10, min_similarity=0.5)
        texts = {h.chunk.text for h in hits}
        assert "dogs are loyal" not in texts
        assert "cats are great" in texts

    def test_search_empty_store_returns_empty(self):
        assert InMemoryVectorStore().search([1.0, 0.0], top_k=5) == []

    def test_combined_filename_and_keyword(self, store):
        hits = store.search(
            [1.0, 0.0], top_k=10, filename_filter="a.txt", keyword="cats"
        )
        assert len(hits) == 1
        assert hits[0].chunk.filename == "a.txt"
        assert hits[0].chunk.text == "cats are great"
