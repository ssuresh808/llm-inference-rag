"""One-off: seed the on-domain arXiv corpus into Qdrant (cloud or local).

Runs the Phase-2b ingestion once and writes the chunks to the configured Qdrant
target. If ``QDRANT_CLOUD_URL`` + ``QDRANT_API_KEY`` are set, vectors are routed
to the remote Qdrant Cloud cluster; otherwise they persist to the on-disk
``./.qdrant_storage`` (or ``QDRANT_PATH``). Safe to re-run: the collection is
recreated.

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


def _drop_collection(collection: str, *, path: str = "", url: str = "", api_key: str = "") -> None:
    """Delete the collection if it exists (idempotent re-seed), cloud or local.

    Args:
        collection: Collection name to drop.
        path: Local on-disk Qdrant path (used when no cloud URL is given).
        url: Qdrant Cloud cluster URL.
        api_key: Qdrant Cloud API key.
    """
    if url and api_key:
        client = QdrantClient(url=url, api_key=api_key)
    else:
        client = QdrantClient(path=path)
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 - missing collection is fine
        pass
    finally:
        client.close()


def main() -> None:
    """Ingest the arXiv corpus and persist it to the configured Qdrant target."""
    parser = argparse.ArgumentParser(description="Seed the arXiv corpus into Qdrant.")
    parser.add_argument("--max-docs", type=int, default=500, help="Document cap.")
    parser.add_argument("--max-scan", type=int, default=200_000, help="Row scan cap.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    use_cloud = bool(settings.qdrant_cloud_url and settings.qdrant_api_key)
    if use_cloud:
        engine_kwargs = {
            "qdrant_cloud_url": settings.qdrant_cloud_url,
            "qdrant_api_key": settings.qdrant_api_key,
        }
        drop_kwargs = {"url": settings.qdrant_cloud_url, "api_key": settings.qdrant_api_key}
        target = "Qdrant Cloud"
    else:
        path = settings.qdrant_path or DEFAULT_PATH
        engine_kwargs = {"qdrant_path": path}
        drop_kwargs = {"path": path}
        target = f"local {path}"

    t0 = time.perf_counter()
    docs = load_arxiv_domain_abstracts(
        settings.arxiv_dataset, max_docs=args.max_docs, max_scan=args.max_scan
    )
    chunks = apply_quality_gate(chunk_documents(docs))
    if not chunks:
        print("No on-domain documents found; nothing seeded.")
        return

    _drop_collection(settings.arxiv_collection, **drop_kwargs)
    engine = RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.arxiv_collection,
        **engine_kwargs,
    )
    engine.index(chunks)
    engine.close()

    print(
        f"Seeded {len(chunks)} chunks ({len(docs)} papers) into "
        f"'{settings.arxiv_collection}' on {target} in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
