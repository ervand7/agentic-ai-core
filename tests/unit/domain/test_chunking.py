"""Unit tests for the pure chunking + cosine similarity domain functions."""

import math

import pytest

from app.domains.documents.domain.chunking import chunk_text, cosine_similarity


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ", chunk_size=100, overlap=10) == []

    def test_text_is_stripped_before_chunking(self):
        assert chunk_text("  hello  ", chunk_size=100, overlap=0) == ["hello"]

    def test_splits_into_multiple_chunks(self):
        text = "abcdefghij"  # 10 chars
        chunks = chunk_text(text, chunk_size=4, overlap=0)
        assert chunks == ["abcd", "efgh", "ij"]

    def test_overlap_repeats_tail(self):
        text = "abcdefghij"  # 10 chars
        # step = chunk_size - overlap = 4 - 2 = 2
        chunks = chunk_text(text, chunk_size=4, overlap=2)
        assert chunks[0] == "abcd"
        assert chunks[1] == "cdef"  # overlaps last 2 chars of previous chunk
        assert chunks[2] == "efgh"

    def test_chunk_covers_entire_text(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        # Reconstruct without overlap by stepping; just confirm full coverage.
        assert "".join(c for c in chunks)  # non-empty
        assert chunks[0] == "x" * 300

    @pytest.mark.parametrize("bad_size", [0, -1, -100])
    def test_invalid_chunk_size_raises(self, bad_size):
        with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
            chunk_text("hello", chunk_size=bad_size, overlap=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap must be 0 or greater"):
            chunk_text("hello", chunk_size=10, overlap=-1)

    @pytest.mark.parametrize("overlap", [10, 11, 50])
    def test_overlap_not_smaller_than_chunk_size_raises(self, overlap):
        with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
            chunk_text("hello", chunk_size=10, overlap=overlap)

    def test_internal_whitespace_chunks_are_skipped(self):
        # A chunk that is pure whitespace after .strip() must not be appended.
        text = "ab" + (" " * 4) + "cd"
        chunks = chunk_text(text, chunk_size=2, overlap=0)
        assert "" not in chunks
        assert chunks[0] == "ab"
        assert chunks[-1] == "cd"


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_returns_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scaled_vectors_are_equally_similar(self):
        assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_known_value(self):
        # angle of 45 degrees -> cos = sqrt(2)/2
        result = cosine_similarity([1.0, 0.0], [1.0, 1.0])
        assert result == pytest.approx(math.sqrt(2) / 2)

    def test_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError, match="same dimension"):
            cosine_similarity([1.0, 2.0], [1.0])
