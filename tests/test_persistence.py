"""Tests for persistent on-disk Qdrant round-trip (offline, fake embeddings)."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.retrieval.engine import RetrievalEngine


def test_client_kwargs_precedence_cloud_over_path_over_memory():
    embeddings = DeterministicFakeEmbedding(size=8)

    cloud = RetrievalEngine(
        embeddings,
        qdrant_cloud_url="https://x.cloud.qdrant.io:6333",
        qdrant_api_key="secret",
        qdrant_path="/tmp/local",  # cloud must win over path
    )
    assert cloud._client_kwargs() == {
        "url": "https://x.cloud.qdrant.io:6333",
        "api_key": "secret",
    }

    # cloud_url without api_key does NOT activate cloud (falls back to path)
    no_key = RetrievalEngine(
        embeddings, qdrant_cloud_url="https://x.cloud.qdrant.io:6333", qdrant_path="/tmp/local"
    )
    assert no_key._client_kwargs() == {"path": "/tmp/local"}

    assert RetrievalEngine(embeddings, qdrant_path="/tmp/local")._client_kwargs() == {
        "path": "/tmp/local"
    }
    assert RetrievalEngine(embeddings)._client_kwargs() == {"location": ":memory:"}


def test_persistent_qdrant_round_trip(tmp_path):
    path = str(tmp_path / "qdrant")
    embeddings = DeterministicFakeEmbedding(size=8)

    writer = RetrievalEngine(embeddings, collection_name="persist_test", qdrant_path=path)
    writer.index([Document(page_content="paged attention kv cache", metadata={"source": "a"})])
    writer.close()

    reader = RetrievalEngine(embeddings, collection_name="persist_test", qdrant_path=path)
    reader.connect_existing()
    out = reader.query("paged attention kv cache", top_k=1)
    reader.close()

    assert out[0].metadata["source"] == "a"
