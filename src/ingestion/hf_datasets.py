"""Ingest a Hugging Face dataset as a retrieval corpus.

Streams a public HF dataset, optionally filters rows to the project's domain by
keyword, and converts them to LangChain ``Document`` objects. The row-to-document
transform is a pure function, so it is unit-tested without any network access.

Default target: ``CShorten/ML-ArXiv-Papers`` (title + abstract), filtered to
LLM/ML inference-optimization topics (ADR-012).

CLI::

    uv run python -m src.ingestion.hf_datasets --limit 3000 --query "..."
"""

import argparse
import logging
from collections.abc import Iterable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "CShorten/ML-ArXiv-Papers"
DEFAULT_TEXT_COLUMNS = ("title", "abstract")
DEFAULT_TITLE_COLUMN = "title"

# Keywords that keep an abstract on the inference-optimization domain (ADR-000).
DEFAULT_KEYWORDS = (
    "inference",
    "latency",
    "throughput",
    "quantization",
    "quantize",
    "kv cache",
    "kv-cache",
    "attention",
    "serving",
    "gpu",
    "speculative",
    "batching",
    "parallelism",
    "distillation",
    "pruning",
    "efficient",
    "low-bit",
    "fp8",
    "int8",
    "decoding",
    "memory",
)


def rows_to_documents(
    rows: Iterable[dict],
    *,
    dataset_id: str,
    text_columns: tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
    title_column: str | None = DEFAULT_TITLE_COLUMN,
    keywords: tuple[str, ...] | None = DEFAULT_KEYWORDS,
    limit: int | None = None,
) -> list[Document]:
    """Convert dataset rows into documents, optionally filtering by keyword.

    Args:
        rows: Iterable of dataset rows (dicts).
        dataset_id: Identifier used to build the ``source`` metadata.
        text_columns: Columns concatenated (in order) into document content.
        title_column: Column stored as ``title`` metadata, if present.
        keywords: If given, keep only rows whose text contains one of these
            (case-insensitive). Pass ``None`` to keep every non-empty row.
        limit: Maximum number of documents to return.

    Returns:
        The converted documents (filtered, capped at ``limit``).
    """
    lowered = tuple(k.lower() for k in keywords) if keywords else ()
    documents: list[Document] = []
    for index, row in enumerate(rows):
        parts = [str(row[c]).strip() for c in text_columns if row.get(c)]
        text = "\n\n".join(part for part in parts if part)
        if not text:
            continue
        if lowered and not any(keyword in text.lower() for keyword in lowered):
            continue
        metadata = {"source": f"{dataset_id}#{index}"}
        if title_column and row.get(title_column):
            metadata["title"] = str(row[title_column]).strip()
        documents.append(Document(page_content=text, metadata=metadata))
        if limit is not None and len(documents) >= limit:
            break
    logger.info("Built %d document(s) from %s", len(documents), dataset_id)
    return documents


def load_hf_corpus(
    dataset_id: str = DEFAULT_DATASET,
    *,
    split: str = "train",
    streaming: bool = True,
    **kwargs,
) -> list[Document]:
    """Load a HF dataset and convert it to documents.

    Args:
        dataset_id: HF dataset repo id.
        split: Dataset split to read.
        streaming: Stream rows instead of downloading the whole dataset.
        **kwargs: Forwarded to :func:`rows_to_documents`.

    Returns:
        The corpus as documents.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset_id, split=split, streaming=streaming)
    return rows_to_documents(rows, dataset_id=dataset_id, **kwargs)


def main() -> None:
    """CLI: index a HF dataset corpus and run a sample query (reports timing)."""
    parser = argparse.ArgumentParser(description="Index a Hugging Face dataset corpus.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id.")
    parser.add_argument("--limit", type=int, default=2000, help="Max documents.")
    parser.add_argument("--top-k", type=int, default=5, help="Results to show.")
    parser.add_argument(
        "--query",
        default="quantization for low-latency LLM inference",
        help="Sample query to run after indexing.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    import time

    from src.ingestion.loader import apply_quality_gate, chunk_documents
    from src.retrieval.engine import build_engine

    t0 = time.perf_counter()
    docs = load_hf_corpus(args.dataset, limit=args.limit)
    chunks = apply_quality_gate(chunk_documents(docs))
    t1 = time.perf_counter()

    engine = build_engine()
    engine.index(chunks)
    t2 = time.perf_counter()

    results = engine.query(args.query, top_k=args.top_k)
    t3 = time.perf_counter()

    print(f"\nDataset: {args.dataset}")
    print(f"Docs loaded: {len(docs)} | chunks indexed: {len(chunks)}")
    print(
        f"Timing: load+chunk {t1 - t0:.1f}s | "
        f"index {t2 - t1:.1f}s | query {t3 - t2:.2f}s"
    )
    print(f"\nTop {len(results)} for: {args.query!r}\n")
    for rank, doc in enumerate(results, start=1):
        label = doc.metadata.get("title", doc.metadata.get("source", "?"))
        print(f"[{rank}] {label}\n    {doc.page_content[:160].strip()}...\n")


if __name__ == "__main__":
    main()
