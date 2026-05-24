"""Utilities for splitting long documents into overlapping chunks."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of roughly `chunk_size` characters.

    Overlap repeats a small tail of one chunk in the next chunk.
    This matters because important meaning often sits near boundaries:
    if one sentence starts at the end of chunk N and continues in chunk N+1,
    overlap helps both chunks keep enough context for better embeddings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
