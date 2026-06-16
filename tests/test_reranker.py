"""Tests for the cross-encoder reranker and engine integration (offline)."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.retrieval.engine import RetrievalEngine
from src.retrieval.reranker import CrossEncoderReranker


class _FakeModel:
    """Scores each (query, text) pair by text length (longer = more relevant)."""

    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


def test_reranker_orders_by_score_and_caps_top_k():
    reranker = CrossEncoderReranker(model=_FakeModel())
    docs = [
        Document(page_content="short"),
        Document(page_content="a much longer passage than the others"),
        Document(page_content="medium length text"),
    ]

    out = reranker.rerank("q", docs, top_k=2)

    assert len(out) == 2
    assert out[0].page_content == "a much longer passage than the others"


def test_reranker_handles_empty():
    assert CrossEncoderReranker(model=_FakeModel()).rerank("q", [], top_k=3) == []


class _PinReranker:
    """Test reranker that always pins one source to the front."""

    def __init__(self, pin: str) -> None:
        self.pin = pin

    def rerank(self, query, documents, *, top_k):
        return sorted(documents, key=lambda d: d.metadata.get("source") != self.pin)[:top_k]


def test_engine_applies_reranker():
    engine = RetrievalEngine(
        DeterministicFakeEmbedding(size=8),
        collection_name="rerank_test",
        reranker=_PinReranker("c.md"),
        rerank_fetch_k=10,
    )
    engine.index(
        [
            Document(page_content="alpha", metadata={"source": "a.md"}),
            Document(page_content="bravo", metadata={"source": "b.md"}),
            Document(page_content="charlie", metadata={"source": "c.md"}),
        ]
    )

    out = engine.query("anything", top_k=1)

    assert out[0].metadata["source"] == "c.md"  # reranker pinned c.md to the front
