import os
import time
from uuid import uuid4

import pytest

from app.domains.documents.infrastructure.qdrant_vector_store import QdrantVectorStore

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]

QDRANT_IMAGE = "qdrant/qdrant:v1.16.1"


def _qdrant_ready(url: str, timeout: float = 30.0) -> bool:
    from qdrant_client import QdrantClient

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            QdrantClient(url=url, check_compatibility=False).get_collections()
            return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def qdrant_url():
    explicit = os.environ.get("QDRANT_TEST_URL")
    if explicit:
        if not _qdrant_ready(explicit, timeout=10.0):
            pytest.skip(f"QDRANT_TEST_URL set but Qdrant unreachable at {explicit}")
        yield explicit
        return

    try:
        from testcontainers.core.container import DockerContainer
    except ImportError:
        pytest.skip("testcontainers not installed and QDRANT_TEST_URL not set")

    try:
        container = DockerContainer(QDRANT_IMAGE).with_exposed_ports(6333)
        container.start()
    except Exception as exc:  # Docker not available / image pull failed.
        pytest.skip(f"Could not start Qdrant container (Docker unavailable?): {exc}")

    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        url = f"http://{host}:{port}"
        if not _qdrant_ready(url):
            pytest.skip("Qdrant container did not become ready in time")
        yield url
    finally:
        container.stop()


@pytest.fixture
def store(qdrant_url):
    collection = f"itest_{uuid4().hex}"
    vector_store = QdrantVectorStore(url=qdrant_url, collection_name=collection)
    yield vector_store
    try:
        vector_store._client.delete_collection(collection)
    except Exception:
        pass


class TestQdrantStore:
    def test_count_is_zero_before_any_upload(self, store):
        assert store.count() == 0

    def test_add_chunks_then_count(self, store):
        store.add_chunks(
            filename="a.txt",
            chunks=["hello world", "goodbye world"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        assert store.count() == 2

    def test_search_returns_relevant_hit_with_vectors(self, store):
        store.add_chunks(
            filename="a.txt",
            chunks=["red apple", "blue ocean"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        hits = store.search([0.95, 0.05, 0.0], top_k=1)
        assert len(hits) == 1
        assert hits[0].chunk.text == "red apple"
        assert hits[0].chunk.filename == "a.txt"
        # Vectors round-trip back out (exercises _vector_list on real payloads).
        assert hits[0].chunk.embedding
        assert 0.0 <= hits[0].similarity <= 1.0

    def test_filename_filter(self, store):
        store.add_chunks(
            filename="a.txt", chunks=["alpha"], embeddings=[[1.0, 0.0, 0.0]]
        )
        store.add_chunks(
            filename="b.txt", chunks=["alpha"], embeddings=[[1.0, 0.0, 0.0]]
        )
        hits = store.search([1.0, 0.0, 0.0], top_k=5, filename_filter="b.txt")
        assert hits
        assert {hit.chunk.filename for hit in hits} == {"b.txt"}

    def test_min_similarity_filters_out_low_scores(self, store):
        store.add_chunks(
            filename="a.txt",
            chunks=["match", "orthogonal"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        hits = store.search([1.0, 0.0, 0.0], top_k=5, min_similarity=0.9)
        assert [hit.chunk.text for hit in hits] == ["match"]

    def test_all_returns_every_stored_chunk(self, store):
        store.add_chunks(
            filename="a.txt",
            chunks=["one", "two"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        stored = store.all()
        assert {chunk.text for chunk in stored} == {"one", "two"}
        assert all(chunk.embedding for chunk in stored)

    def test_search_on_missing_collection_returns_empty(self, qdrant_url):
        empty = QdrantVectorStore(
            url=qdrant_url, collection_name=f"missing_{uuid4().hex}"
        )
        assert empty.search([1.0, 0.0, 0.0], top_k=3) == []
        assert empty.count() == 0
