"""Tests for the retrieval engine and embedding factory.

Embeddings are faked with ``DeterministicFakeEmbedding`` (deterministic and
offline), so Qdrant insertion/retrieval is exercised without any network call or
model download. Querying the exact text of an indexed chunk yields an identical
vector, so that chunk is guaranteed to rank first.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config.settings import Settings
from src.retrieval.embeddings import build_embeddings
from src.retrieval.engine import RetrievalEngine

EMBED_DIM = 8


def _engine() -> RetrievalEngine:
    return RetrievalEngine(
        DeterministicFakeEmbedding(size=EMBED_DIM),
        collection_name="test_collection",
    )


def test_index_returns_chunk_count():
    engine = _engine()
    docs = [
        Document(page_content="PagedAttention manages the KV cache"),
        Document(page_content="Continuous batching raises throughput"),
    ]
    assert engine.index(docs) == 2


def test_query_returns_exact_match_first():
    engine = _engine()
    target = "Speculative decoding reduces decoding latency"
    engine.index(
        [
            Document(page_content=target),
            Document(page_content="FP8 quantization shrinks the model"),
        ]
    )

    results = engine.query(target, top_k=2)

    assert results[0].page_content == target


def test_query_respects_top_k():
    engine = _engine()
    engine.index([Document(page_content=f"doc {i} about gpu serving") for i in range(5)])

    assert len(engine.query("gpu serving", top_k=3)) == 3


def test_metadata_survives_round_trip():
    engine = _engine()
    target = "Tensor parallelism shards weights across GPUs"
    engine.index([Document(page_content=target, metadata={"source": "tp.md"})])

    [hit] = engine.query(target, top_k=1)

    assert hit.metadata["source"] == "tp.md"


def test_query_before_index_raises():
    with pytest.raises(RuntimeError):
        _engine().query("anything")


def test_index_empty_raises():
    with pytest.raises(ValueError):
        _engine().index([])


def test_build_embeddings_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_embeddings(Settings(_env_file=None, embedding_provider="bogus"))


def test_build_embeddings_dispatches_to_openai():
    from langchain_openai import OpenAIEmbeddings

    emb = build_embeddings(
        Settings(
            _env_file=None,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key="sk-test",
        )
    )

    assert isinstance(emb, OpenAIEmbeddings)


def test_build_embeddings_dispatches_to_voyage(monkeypatch):
    # Mock the Voyage class: its real constructor fetches a tokenizer from the
    # HF Hub (network), which the zero-network unit-test rule forbids.
    import langchain_voyageai

    sentinel = object()
    monkeypatch.setattr(langchain_voyageai, "VoyageAIEmbeddings", lambda **kwargs: sentinel)

    emb = build_embeddings(
        Settings(
            _env_file=None,
            embedding_provider="voyage",
            embedding_model="voyage-3.5-lite",
            voyage_api_key="pa-test",
        )
    )

    assert emb is sentinel
