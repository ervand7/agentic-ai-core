"""Unit tests for the documents domain value objects."""

import dataclasses

import pytest

from app.domains.documents.domain.models import (
    Citation,
    GeneratedAnswer,
    RagAnswer,
    SearchHit,
    StoredChunk,
)


def test_stored_chunk_holds_text_embedding_filename():
    chunk = StoredChunk(text="hi", embedding=[0.1, 0.2], filename="f.txt")
    assert chunk.text == "hi"
    assert chunk.embedding == [0.1, 0.2]
    assert chunk.filename == "f.txt"


def test_models_are_frozen():
    chunk = StoredChunk(text="hi", embedding=[0.1], filename="f.txt")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(chunk, "text", "changed")


def test_search_hit_wraps_chunk_and_similarity():
    chunk = StoredChunk(text="hi", embedding=[0.1], filename="f.txt")
    hit = SearchHit(chunk=chunk, similarity=0.5)
    assert hit.chunk is chunk
    assert hit.similarity == 0.5


def test_generated_answer_fields():
    answer = GeneratedAnswer(content="hello", model="gpt", tokens_used=3)
    assert (answer.content, answer.model, answer.tokens_used) == ("hello", "gpt", 3)


def test_citation_fields():
    citation = Citation(index=1, filename="f.txt", text="t", similarity=0.9)
    assert citation.index == 1
    assert citation.filename == "f.txt"


def test_rag_answer_allows_none_model_when_abstaining():
    answer = RagAnswer(
        answer="I don't know",
        citations=[],
        used_context=False,
        model=None,
        tokens_used=0,
    )
    assert answer.used_context is False
    assert answer.model is None
    assert answer.citations == []
