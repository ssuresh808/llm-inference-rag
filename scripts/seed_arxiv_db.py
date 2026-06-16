"""One-off: seed the on-domain arXiv corpus into a persistent on-disk Qdrant DB.

Runs the Phase-2b ingestion once and writes the chunks to ``./.qdrant_storage``
(or ``QDRANT_PATH``) so evaluation/RAGAS scripts can read the collection without
re-ingesting on every run. Safe to re-run: the collection is recreated.

    uv run python -m scripts.seed_arxiv_db
"""

import argparse
import logging
import time

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.ingestion.arxiv import load_arxiv_domain_abstracts
from src.ingestion.loader import apply_quality_gate, chunk_documents
from src.retrieval.embeddings import build_embeddings
from src.retrieval.engine import RetrievalEngine

DEFAULT_PATH = ".qdrant_storage"


def _drop_collection(path: str, collection: str) -> None:
    """Delete the collection if it already exists (idempotent re-seed)."""
    client = QdrantClient(path=path)
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 - missing collection is fine
        pass
    finally:
        client.close()


def main() -> None:
    """Ingest the arXiv corpus and persist it to the on-disk Qdrant collection."""
    parser = argparse.ArgumentParser(description="Seed the persistent arXiv DB.")
    parser.add_argument("--max-docs", type=int, default=500, help="Document cap.")
    parser.add_argument("--max-scan", type=int, default=200_000, help="Row scan cap.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    path = settings.qdrant_path or DEFAULT_PATH

    t0 = time.perf_counter()
    docs = load_arxiv_domain_abstracts(
        settings.arxiv_dataset, max_docs=args.max_docs, max_scan=args.max_scan
    )
    chunks = apply_quality_gate(chunk_documents(docs))
    if not chunks:
        print("No on-domain documents found; nothing seeded.")
        return

    _drop_collection(path, settings.arxiv_collection)
    engine = RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.arxiv_collection,
        qdrant_path=path,
    )
    engine.index(chunks)
    engine.close()

    print(
        f"Seeded {len(chunks)} chunks ({len(docs)} papers) into "
        f"'{settings.arxiv_collection}' at {path} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
