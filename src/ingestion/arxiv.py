"""On-domain corpus ingestion from arXiv metadata (Phase 2b, ADR-015).

Streams ``librarian-bots/arxiv-metadata-snapshot`` and selects recent
LLM-inference-optimization papers by category, date, and a seed vocabulary, then
emits ``Document`` objects (title + abstract as content; id + authors as
metadata). The selection transform is pure, so it is unit-tested against fake
rows with no network access. Streaming + a document cap keep memory bounded.
"""

import argparse
import logging
from collections.abc import Iterable
from datetime import date, datetime

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_ARXIV_DATASET = "librarian-bots/arxiv-metadata-snapshot"
DOMAIN_CATEGORIES = ("cs.CL", "cs.AI")
SINCE_DEFAULT = "2023-01-01"
DEFAULT_MAX_DOCS = 500

# A paper must contain at least one of these (case-insensitive) in title/abstract.
# Hyphen/space variants are included so e.g. "KV cache" and "KV-cache" both match.
SEED_VOCAB = (
    "vllm",
    "tensorrt-llm",
    "pagedattention",
    "paged attention",
    "kv-cache",
    "kv cache",
    "speculative decoding",
    "continuous batching",
    "flashattention",
    "flash attention",
    "quantization",
    "gpu serving",
)


def _iso_date(value: object) -> str:
    """Coerce an ``update_date`` value to an ISO ``YYYY-MM-DD`` string.

    Handles both string and ``date``/``datetime`` types for safe comparison.

    Args:
        value: The raw ``update_date`` field (string, date, datetime, or None).

    Returns:
        The date as ``YYYY-MM-DD``, or ``""`` if missing.
    """
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)[:10]


def arxiv_rows_to_documents(
    rows: Iterable[dict],
    *,
    categories: tuple[str, ...] = DOMAIN_CATEGORIES,
    since: str = SINCE_DEFAULT,
    keywords: tuple[str, ...] = SEED_VOCAB,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_scan: int | None = None,
) -> list[Document]:
    """Select on-domain arXiv rows and convert them to documents.

    A row is kept only if it (1) is in one of ``categories``, (2) has
    ``update_date >= since``, and (3) matches a seed-vocabulary keyword in its
    title/abstract.

    Args:
        rows: Iterable of arXiv metadata rows (dicts).
        categories: Keep rows whose ``categories`` field contains one of these.
        since: Minimum ``update_date`` (ISO ``YYYY-MM-DD``); ``""`` disables it.
        keywords: Case-insensitive seed terms; one must appear in the text.
        max_docs: Stop after this many selected documents (fail-safe).
        max_scan: Stop after scanning this many rows, regardless of matches.

    Returns:
        The selected documents (title + abstract content; id/authors metadata).
    """
    wanted_categories = set(categories)
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    documents: list[Document] = []

    for scanned, row in enumerate(rows, start=1):
        if max_scan is not None and scanned > max_scan:
            break

        row_categories = set(str(row.get("categories", "")).split())
        if wanted_categories and row_categories.isdisjoint(wanted_categories):
            continue
        if since and _iso_date(row.get("update_date")) < since:
            continue

        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            continue
        if lowered_keywords and not any(kw in text.lower() for kw in lowered_keywords):
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"arxiv:{row.get('id', '?')}",
                    "title": title,
                    "authors": str(row.get("authors", "")).strip(),
                },
            )
        )
        if len(documents) >= max_docs:
            break

    logger.info("Selected %d on-domain arXiv document(s)", len(documents))
    return documents


def load_arxiv_domain_abstracts(
    dataset_id: str = DEFAULT_ARXIV_DATASET,
    *,
    split: str = "train",
    max_docs: int = DEFAULT_MAX_DOCS,
    max_scan: int | None = None,
    **kwargs,
) -> list[Document]:
    """Stream the arXiv snapshot and select on-domain abstracts.

    Args:
        dataset_id: HF dataset id.
        split: Dataset split to stream.
        max_docs: Fail-safe cap on selected documents.
        max_scan: Optional cap on rows scanned.
        **kwargs: Forwarded to :func:`arxiv_rows_to_documents`.

    Returns:
        The selected documents.
    """
    from datasets import load_dataset

    rows = load_dataset(dataset_id, split=split, streaming=True)
    return arxiv_rows_to_documents(rows, max_docs=max_docs, max_scan=max_scan, **kwargs)


def main() -> None:
    """CLI: stream the arXiv corpus into the isolated domain collection."""
    parser = argparse.ArgumentParser(description="Ingest the on-domain arXiv corpus.")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Doc cap.")
    parser.add_argument("--max-scan", type=int, default=200_000, help="Row scan cap.")
    parser.add_argument(
        "--query",
        default="PagedAttention KV cache memory for LLM serving",
        help="Sample query to run after indexing.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    import time

    from src.config.settings import get_settings
    from src.ingestion.loader import apply_quality_gate, chunk_documents
    from src.retrieval.embeddings import build_embeddings
    from src.retrieval.engine import RetrievalEngine

    settings = get_settings()

    t0 = time.perf_counter()
    docs = load_arxiv_domain_abstracts(
        settings.arxiv_dataset, max_docs=args.max_docs, max_scan=args.max_scan
    )
    chunks = apply_quality_gate(chunk_documents(docs))
    t1 = time.perf_counter()

    if not chunks:
        print("No on-domain documents found within the scan limit.")
        return

    engine = RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.arxiv_collection,
        qdrant_url=settings.qdrant_url,
    )
    engine.index(chunks)
    t2 = time.perf_counter()

    results = engine.query(args.query, top_k=5)
    t3 = time.perf_counter()

    print(f"\nCollection: {settings.arxiv_collection}")
    print(f"Docs selected: {len(docs)} | chunks indexed: {len(chunks)}")
    print(
        f"Timing: load+chunk {t1 - t0:.1f}s | index {t2 - t1:.1f}s | query {t3 - t2:.2f}s"
    )
    print(f"\nTop {len(results)} for: {args.query!r}\n")
    for rank, doc in enumerate(results, start=1):
        print(f"[{rank}] {doc.metadata.get('title', doc.metadata.get('source'))}")
        print(f"    {doc.page_content[:150].strip()}...\n")


if __name__ == "__main__":
    main()
