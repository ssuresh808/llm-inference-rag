"""End-to-end retrieval pipeline: ingest a corpus, index it, and query it.

Ties the ingestion (Step 2) and retrieval (Step 3) layers into one runnable entry
point. Run it as a CLI::

    uv run python -m src.pipeline --corpus data/sample --query "..." --top-k 3
"""

import argparse
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.loader import ingest_directory
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)


def run_pipeline(
    corpus_dir: Path | str,
    query: str,
    *,
    engine: RetrievalEngine | None = None,
    top_k: int = 5,
) -> list[Document]:
    """Ingest ``corpus_dir``, index it, and return the top-k hits for ``query``.

    Args:
        corpus_dir: Directory of source documents.
        query: Natural-language query.
        engine: Retrieval engine to use. Defaults to ``build_engine()``, which
            uses the configured embedding provider.
        top_k: Number of chunks to return.

    Returns:
        The retrieved chunks, most similar first.

    Raises:
        ValueError: If the corpus yields no indexable chunks.
    """
    chunks = ingest_directory(corpus_dir)
    if not chunks:
        raise ValueError(f"No indexable chunks found in {corpus_dir}.")

    engine = engine or build_engine()
    engine.index(chunks)
    return engine.query(query, top_k=top_k)


def main() -> None:
    """CLI entry point: run the pipeline and print the retrieved chunks."""
    parser = argparse.ArgumentParser(description="Run the RAG retrieval pipeline.")
    parser.add_argument("--corpus", default="data/sample", help="Corpus directory.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = run_pipeline(args.corpus, args.query, top_k=args.top_k)
    print(f"\nTop {len(results)} result(s) for: {args.query!r}\n")
    for rank, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "?")
        snippet = doc.page_content[:240].strip()
        print(f"[{rank}] source={source}\n    {snippet}...\n")


if __name__ == "__main__":
    main()
