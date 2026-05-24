"""OpenAI-backed implementation of the EmbeddingProvider port."""

import logging

from app.domains.documents.domain.ports import EmbeddingProvider
from app.shared.infrastructure.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings via the shared OpenAI client."""

    def __init__(self, client: OpenAIClient):
        self._client = client

    async def embed(self, text: str, request_id: str) -> list[float]:
        """
        Generate an embedding vector for `text` using OpenAI.

        An embedding is a list of numbers (a vector) that represents meaning.
        Texts with similar meaning get vectors that point in similar directions.
        """
        logger.info(
            "embedding_generation_started request_id=%s text_length=%s",
            request_id,
            len(text),
        )
        vector = await self._client.create_embedding(
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
