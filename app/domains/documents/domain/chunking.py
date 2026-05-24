"""
Pure domain functions: text chunking and cosine similarity.

These functions encode the core rules of how we slice text and how we
compare embeddings. They do not depend on any framework or provider.
"""

import math


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


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity manually (no black-box similarity library).

    Mathematical idea:
    cos(theta) = (A . B) / (||A|| * ||B||)

    - A . B is the dot product (sum of pairwise multiplications)
    - ||A|| is the Euclidean norm (square root of sum of squares)
    - Result range is [-1, 1], where values closer to 1 mean vectors point
      in nearly the same direction, which usually means similar semantic meaning.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        dot_product += value_a * value_b
        norm_a_squared += value_a * value_a
        norm_b_squared += value_b * value_b

    norm_a = math.sqrt(norm_a_squared)
    norm_b = math.sqrt(norm_b_squared)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
