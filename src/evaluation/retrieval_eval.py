"""Retrieval evaluation harness.

Runs a gold question set through the retrieval engine and reports aggregate
Hit@k, MRR, and Recall@k. The metrics are deterministic (no LLM judge), giving
fast, reproducible, defensible retrieval numbers and a yardstick for later
optimizations (hybrid search, reranking, embedding/chunking ablations).

Run as a CLI::

    uv run python -m src.evaluation.retrieval_eval --corpus data/sample
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document

from src.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from src.ingestion.loader import ingest_directory
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_GOLD_PATH = Path("data/eval/retrieval_gold.json")


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval metrics over a gold question set."""

    hit_at_k: float
    mrr: float
    recall_at_k: float
    k: int
    n_questions: int


def load_gold_set(path: Path | str = DEFAULT_GOLD_PATH) -> list[dict]:
    """Load the gold question set from a JSON file.

    Args:
        path: Path to a JSON list of ``{"question", "relevant_sources"}`` items.

    Returns:
        The parsed gold set.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ranked_sources(documents: list[Document]) -> list[str]:
    """Return distinct source ids from documents, preserving rank order.

    Args:
        documents: Retrieved documents, most relevant first.

    Returns:
        Distinct source identifiers in first-seen (rank) order.
    """
    ranked: list[str] = []
    for document in documents:
        source = document.metadata.get("source")
        if source and source not in ranked:
            ranked.append(source)
    return ranked


def evaluate_retrieval(
    engine: RetrievalEngine,
    gold: list[dict],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalMetrics:
    """Evaluate retrieval quality over a gold set.

    Args:
        engine: An indexed retrieval engine.
        gold: Items of ``{"question", "relevant_sources"}``.
        top_k: Number of chunks to retrieve per question.

    Returns:
        Aggregate ``RetrievalMetrics`` (means over all questions).

    Raises:
        ValueError: If ``gold`` is empty.
    """
    if not gold:
        raise ValueError("Gold set is empty.")

    hits: list[float] = []
    rrs: list[float] = []
    recalls: list[float] = []
    for item in gold:
        ranked = _ranked_sources(engine.query(item["question"], top_k=top_k))
        relevant = set(item["relevant_sources"])
        hits.append(hit_at_k(ranked, relevant, top_k))
        rrs.append(reciprocal_rank(ranked, relevant))
        recalls.append(recall_at_k(ranked, relevant, top_k))

    return RetrievalMetrics(
        hit_at_k=mean(hits),
        mrr=mean(rrs),
        recall_at_k=mean(recalls),
        k=top_k,
        n_questions=len(gold),
    )


def main() -> None:
    """CLI: index the corpus and print retrieval metrics over the gold set."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--corpus", default="data/sample", help="Corpus directory.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH), help="Gold set JSON.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks per query.")
    parser.add_argument("--log-level", default="WARNING", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    chunks = ingest_directory(args.corpus)
    engine = build_engine()
    engine.index(chunks)
    metrics = evaluate_retrieval(engine, load_gold_set(args.gold), top_k=args.top_k)

    print(f"\nRetrieval evaluation ({metrics.n_questions} questions, k={metrics.k})")
    print(f"  Hit@{metrics.k}    : {metrics.hit_at_k:.3f}")
    print(f"  MRR        : {metrics.mrr:.3f}")
    print(f"  Recall@{metrics.k} : {metrics.recall_at_k:.3f}\n")


if __name__ == "__main__":
    main()
