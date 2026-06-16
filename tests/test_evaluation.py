"""Tests for retrieval metrics and the evaluation harness (fully offline)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from src.evaluation.retrieval_eval import RetrievalMetrics, evaluate_retrieval
from src.retrieval.engine import RetrievalEngine


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert hit_at_k(["a"], {"z"}, 1) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_evaluate_retrieval_perfect_on_exact_match():
    engine = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="eval_test")
    engine.index(
        [
            Document(page_content="alpha question text", metadata={"source": "a.md"}),
            Document(page_content="beta question text", metadata={"source": "b.md"}),
        ]
    )
    gold = [
        {"question": "alpha question text", "relevant_sources": ["a.md"]},
        {"question": "beta question text", "relevant_sources": ["b.md"]},
    ]

    metrics = evaluate_retrieval(engine, gold, top_k=2)

    assert isinstance(metrics, RetrievalMetrics)
    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.n_questions == 2


def test_evaluate_retrieval_empty_gold_raises():
    engine = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="eval_empty")
    with pytest.raises(ValueError):
        evaluate_retrieval(engine, [], top_k=2)
