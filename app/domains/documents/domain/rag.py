"""
Pure domain logic for Retrieval-Augmented Generation (RAG).

This module only knows how to turn retrieved chunks into:
- a context block + user prompt to send to the LLM ("context injection"), and
- a list of citations to return to the caller.

It has no dependency on OpenAI, FastAPI, or the vector store, which keeps the
core RAG rules easy to read and unit-test.
"""

from app.domains.documents.domain.models import Citation, SearchHit

# Returned verbatim when retrieval finds nothing relevant. Abstaining here (the
# "I don't know" behavior) avoids paying for an LLM call that could only
# hallucinate, since there is no grounding context to answer from.
NO_CONTEXT_ANSWER = (
    "I don't know based on the uploaded documents. "
    "I couldn't find anything relevant to that question."
)


def build_context_block(hits: list[SearchHit]) -> str:
    """
    Render retrieved chunks as a numbered context block.

    Each source is prefixed with the bracket number the LLM is asked to cite,
    e.g. `[1]`, plus its filename so answers can point back to a real document.
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(f"[{index}] (source: {hit.chunk.filename})\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    """
    Inject the retrieved context into a single user message.

    The system prompt (loaded from settings) carries the behavior rules
    (cite sources, say "I don't know"); here we only assemble context + question.
    """
    context_block = build_context_block(hits)
    return (
        "Context snippets:\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite the snippet numbers you use."
    )


def build_citations(hits: list[SearchHit]) -> list[Citation]:
    """Map retrieved hits onto citation value objects (1-based numbering)."""
    return [
        Citation(
            index=index,
            filename=hit.chunk.filename,
            text=hit.chunk.text,
            similarity=hit.similarity,
        )
        for index, hit in enumerate(hits, start=1)
    ]
