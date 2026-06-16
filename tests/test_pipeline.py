"""Offline integration test for the end-to-end pipeline.

Runs the real ingestion path over the committed sample corpus, but with a
deterministic fake embedding, so the load -> chunk -> gate -> index -> query flow
is exercised end-to-end without any network call or model download.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.pipeline import run_pipeline
from src.retrieval.engine import RetrievalEngine

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_run_pipeline_over_sample_corpus():
    engine = RetrievalEngine(
        DeterministicFakeEmbedding(size=8),
        collection_name="pipeline_test",
    )

    results = run_pipeline(SAMPLE_DIR, "paged attention kv cache", engine=engine, top_k=2)

    assert 1 <= len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)
    assert all(doc.metadata.get("source") for doc in results)
