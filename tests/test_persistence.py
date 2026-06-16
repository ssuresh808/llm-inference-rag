"""Tests for persistent on-disk Qdrant round-trip (offline, fake embeddings)."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.retrieval.engine import RetrievalEngine


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
