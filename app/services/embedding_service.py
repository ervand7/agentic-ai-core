"""Service for generating vector embeddings from text."""

import logging

from app.services.openai_client import get_openai_client

logger = logging.getLogger(__name__)


async def generate_embedding(text: str, request_id: str) -> list[float]:
    """
    Generate an embedding vector for text using OpenAI.

    An embedding is a list of numbers (a vector) that represents meaning.
    Texts with similar meaning get vectors that point in similar directions.
    That is why embeddings are a practical base for semantic search.
    """
    logger.info(
        "embedding_generation_started request_id=%s text_length=%s",
        request_id,
        len(text),
    )
    vector = await get_openai_client().create_embedding(
        endpoint="embeddings",
        request_id=request_id,
        text=text,
    )
    logger.info(
        "embedding_generation_finished request_id=%s vector_size=%s",
        request_id,
        len(vector),
    )
    return vector
