"""Unit tests for the pure RAG prompt/citation builders."""

from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.domain.rag import (
    NO_CONTEXT_ANSWER,
    build_citations,
    build_context_block,
    build_user_prompt,
)


def _hit(text: str, filename: str, similarity: float) -> SearchHit:
    return SearchHit(
        chunk=StoredChunk(text=text, embedding=[0.0], filename=filename),
        similarity=similarity,
    )


class TestBuildContextBlock:
    def test_empty_hits_returns_empty_string(self):
        assert build_context_block([]) == ""

    def test_single_hit_is_numbered_from_one(self):
        block = build_context_block([_hit("the cat sat", "a.txt", 0.9)])
        assert block == "[1] (source: a.txt)\nthe cat sat"

    def test_multiple_hits_are_separated_by_blank_line(self):
        block = build_context_block(
            [_hit("first", "a.txt", 0.9), _hit("second", "b.txt", 0.8)]
        )
        assert block == "[1] (source: a.txt)\nfirst\n\n[2] (source: b.txt)\nsecond"


class TestBuildUserPrompt:
    def test_contains_context_question_and_instruction(self):
        prompt = build_user_prompt("What is X?", [_hit("X is a thing", "doc.txt", 0.7)])
        assert "Context snippets:" in prompt
        assert "[1] (source: doc.txt)\nX is a thing" in prompt
        assert "Question: What is X?" in prompt
        assert "cite the snippet numbers" in prompt

    def test_works_with_no_hits(self):
        prompt = build_user_prompt("Anything?", [])
        assert "Question: Anything?" in prompt


class TestBuildCitations:
    def test_empty_hits_returns_empty_list(self):
        assert build_citations([]) == []

    def test_indexes_are_one_based_and_fields_preserved(self):
        hits = [_hit("alpha", "a.txt", 0.91), _hit("beta", "b.txt", 0.42)]
        citations = build_citations(hits)

        assert [c.index for c in citations] == [1, 2]
        assert citations[0].filename == "a.txt"
        assert citations[0].text == "alpha"
        assert citations[0].similarity == 0.91
        assert citations[1].index == 2
        assert citations[1].similarity == 0.42


def test_no_context_answer_is_a_non_empty_abstention_message():
    assert "don't know" in NO_CONTEXT_ANSWER.lower()
    assert NO_CONTEXT_ANSWER.strip() != ""
