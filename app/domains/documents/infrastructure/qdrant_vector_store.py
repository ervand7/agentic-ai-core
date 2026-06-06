"""Qdrant-backed implementation of the VectorStore port."""

import logging
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.domains.documents.domain.models import SearchHit, StoredChunk
from app.domains.documents.domain.ports import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Vector store backed by a Qdrant collection."""

    def __init__(self, *, url: str, collection_name: str) -> None:
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._collection_name = collection_name
        self._payload_indexes_ready = False

    def add_chunks(
        self,
        *,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk text and vectors in Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        vector_size = len(embeddings[0])
        if vector_size == 0:
            raise ValueError("embedding vectors must not be empty")

        self._ensure_collection(vector_size)

        points = [
            models.PointStruct(
                id=uuid4().hex,
                vector=embedding,
                payload={"text": chunk_text, "filename": filename},
            )
            for chunk_text, embedding in zip(chunks, embeddings)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)

    def count(self) -> int:
        """Return number of vectors stored in the Qdrant collection."""
        if not self._collection_exists():
            return 0

        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )
        return result.count

    def all(self) -> list[StoredChunk]:
        """
        Return all stored chunks.

        This is not used by the application flow with Qdrant but remains
        implemented to satisfy the VectorStore protocol.
        """
        if not self._collection_exists():
            return []

        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=True,
        )
        return [
            StoredChunk(
                text=self._payload_str(record.payload, "text"),
                filename=self._payload_str(record.payload, "filename"),
                embedding=self._vector_list(record.vector),
            )
            for record in records
        ]

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filename_filter: Optional[str] = None,
        keyword: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> list[SearchHit]:
        """Run vector similarity search in Qdrant and return top hits."""
        if top_k <= 0 or not self._collection_exists():
            return []

        self._ensure_payload_indexes()
        query_filter = self._build_filename_filter(filename_filter)
        keyword_query = (keyword or "").strip().lower()
        # Over-fetch when keyword ranking is enabled so lexical scoring can rerank.
        candidate_limit = min(200, top_k * 5) if keyword_query else top_k

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
            with_vectors=True,
        )

        scored_hits: list[SearchHit] = []
        for result in response.points:
            text = self._payload_str(result.payload, "text")
            filename = self._payload_str(result.payload, "filename")
            vector_score = float(result.score)
            if min_similarity is not None and vector_score < min_similarity:
                continue

            keyword_score = self._keyword_score(text=text, keyword=keyword_query)
            if keyword_query and keyword_score == 0.0:
                continue
            final_score = self._hybrid_score(
                vector_score=vector_score,
                keyword_score=keyword_score,
                keyword_used=bool(keyword_query),
            )

            logger.info(
                (
                    "semantic_similarity_score filename=%s vector_score=%.4f "
                    "keyword_score=%.4f hybrid_score=%.4f text_preview=%s"
                ),
                filename,
                vector_score,
                keyword_score,
                final_score,
                text[:80].replace("\n", " "),
            )
            scored_hits.append(
                SearchHit(
                    chunk=StoredChunk(
                        text=text,
                        filename=filename,
                        embedding=self._vector_list(result.vector),
                    ),
                    similarity=final_score,
                )
            )

        scored_hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return scored_hits[:top_k]

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            self._ensure_payload_indexes()
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10_000,
                ),
            ),
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        if self._payload_indexes_ready:
            return
        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._payload_indexes_ready = True
        except (UnexpectedResponse, ValueError):
            # Collection may not exist yet, or index may already exist.
            pass

    def _collection_exists(self) -> bool:
        try:
            return self._client.collection_exists(self._collection_name)
        except (UnexpectedResponse, ValueError):
            return False

    @staticmethod
    def _build_filename_filter(filename_filter: Optional[str]) -> Optional[models.Filter]:
        normalized = (filename_filter or "").strip()
        if not normalized:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=normalized),
                )
            ]
        )

    @staticmethod
    def _keyword_score(*, text: str, keyword: str) -> float:
        if not keyword:
            return 0.0

        normalized_text = text.lower()
        phrase_present = 1.0 if keyword in normalized_text else 0.0

        keyword_tokens = [token for token in keyword.split() if token]
        if not keyword_tokens:
            return phrase_present

        token_matches = sum(1 for token in keyword_tokens if token in normalized_text)
        token_ratio = token_matches / len(keyword_tokens)
        return min(1.0, 0.6 * phrase_present + 0.4 * token_ratio)

    @staticmethod
    def _hybrid_score(
        *,
        vector_score: float,
        keyword_score: float,
        keyword_used: bool,
    ) -> float:
        if not keyword_used:
            return vector_score
        # Weighted blend: semantic similarity dominates, keyword adds lexical signal.
        return 0.8 * vector_score + 0.2 * keyword_score

    @staticmethod
    def _payload_str(payload: Optional[models.Payload], key: str) -> str:
        if not payload:
            return ""
        value = payload.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _vector_list(vector: Optional[models.VectorStruct]) -> list[float]:
        if isinstance(vector, list):
            return [float(value) for value in vector]
        return []
