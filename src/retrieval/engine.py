"""Vector indexing and retrieval over Qdrant.

``RetrievalEngine`` wraps a Qdrant vector store and an embeddings backend, with
optional sparse (BM25) **hybrid** retrieval and optional cross-encoder
**reranking**. Collaborators are injected, so tests can supply deterministic
fakes and exercise the index/query path without network access or downloads.
"""

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import build_embeddings, build_sparse_embeddings

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_RERANK_FETCH_K = 20


class RetrievalEngine:
    """Index document chunks into Qdrant and retrieve the most similar ones."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        collection_name: str = "rag_portfolio",
        qdrant_url: str = ":memory:",
        qdrant_path: str = "",
        qdrant_cloud_url: str = "",
        qdrant_api_key: str = "",
        sparse_embeddings: object | None = None,
        reranker: object | None = None,
        rerank_fetch_k: int = DEFAULT_RERANK_FETCH_K,
    ) -> None:
        """Initialize the engine.

        Args:
            embeddings: Dense embeddings backend for indexing and queries.
            collection_name: Qdrant collection to write to / read from.
            qdrant_url: ``":memory:"`` for a local store, else an ``http`` URL.
            qdrant_path: On-disk Qdrant storage path; when set, takes precedence
                over ``qdrant_url`` and persists the collection between runs.
            qdrant_cloud_url: Qdrant Cloud cluster URL; with ``qdrant_api_key``
                set, takes precedence over all local options.
            qdrant_api_key: API key for the Qdrant Cloud cluster.
            sparse_embeddings: Optional sparse (BM25) embedding; when provided,
                indexing/retrieval use Qdrant hybrid mode.
            reranker: Optional object with ``rerank(query, docs, top_k)``; when
                set, queries fetch more candidates and rerank them.
            rerank_fetch_k: Candidates fetched before reranking.
        """
        self._embeddings = embeddings
        self._sparse_embeddings = sparse_embeddings
        self.reranker = reranker
        self._rerank_fetch_k = rerank_fetch_k
        self._collection_name = collection_name
        self._qdrant_url = qdrant_url
        self._qdrant_path = qdrant_path
        self._qdrant_cloud_url = qdrant_cloud_url
        self._qdrant_api_key = qdrant_api_key
        self._vector_store: QdrantVectorStore | None = None

    def _client_kwargs(self) -> dict:
        """Return Qdrant client kwargs for the configured storage location.

        Precedence: Qdrant Cloud (url + api_key) > on-disk path > in-memory >
        plain URL.
        """
        if self._qdrant_cloud_url and self._qdrant_api_key:
            return {"url": self._qdrant_cloud_url, "api_key": self._qdrant_api_key}
        if self._qdrant_path:
            return {"path": self._qdrant_path}
        if self._qdrant_url == ":memory:":
            return {"location": ":memory:"}
        return {"url": self._qdrant_url}

    def _hybrid_kwargs(self) -> dict:
        """Return hybrid-mode kwargs when a sparse embedding is configured."""
        if self._sparse_embeddings is None:
            return {}
        return {
            "sparse_embedding": self._sparse_embeddings,
            "retrieval_mode": RetrievalMode.HYBRID,
        }

    def connect_existing(self) -> None:
        """Attach to an already-indexed collection without re-indexing.

        Used to read a persistent (on-disk/server) Qdrant collection that was
        seeded by a separate process.
        """
        self._vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self._embeddings,
            collection_name=self._collection_name,
            **self._client_kwargs(),
            **self._hybrid_kwargs(),
        )

    def close(self) -> None:
        """Close the underlying Qdrant client (releases any on-disk lock)."""
        if self._vector_store is not None:
            self._vector_store.client.close()
            self._vector_store = None

    def index(self, chunks: list[Document]) -> int:
        """Index chunks, creating the collection on first call.

        Uses Qdrant hybrid mode when a sparse embedding was supplied.

        Args:
            chunks: Non-empty list of documents to index.

        Returns:
            The number of chunks indexed in this call.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("index() requires at least one chunk.")

        if self._vector_store is None:
            self._vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                collection_name=self._collection_name,
                **self._client_kwargs(),
                **self._hybrid_kwargs(),
            )
            count = len(chunks)
        else:
            count = len(self._vector_store.add_documents(chunks))

        mode = "hybrid" if self._sparse_embeddings is not None else "dense"
        logger.info("Indexed %d chunk(s) into '%s' (%s)", count, self._collection_name, mode)
        return count

    def query(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[Document]:
        """Return the ``top_k`` chunks most relevant to ``text``.

        Retrieves ``rerank_fetch_k`` candidates and reranks them when a reranker
        is configured; otherwise returns the top-k dense/hybrid hits directly.

        Args:
            text: Natural-language query.
            top_k: Maximum number of chunks to return.

        Returns:
            The matching chunks, most relevant first.

        Raises:
            RuntimeError: If called before anything has been indexed.
        """
        if self._vector_store is None:
            raise RuntimeError("Index is empty; call index() before query().")

        fetch_k = max(top_k, self._rerank_fetch_k) if self.reranker is not None else top_k
        candidates = self._vector_store.similarity_search(text, k=fetch_k)
        if self.reranker is not None:
            candidates = self.reranker.rerank(text, candidates, top_k=top_k)
        return candidates[:top_k]


def build_engine(settings: Settings | None = None) -> RetrievalEngine:
    """Construct a ``RetrievalEngine`` from configuration.

    Wires the embedding provider (ADR-003), optional hybrid sparse retrieval and
    optional reranking (ADR-014), and the Qdrant location.

    Args:
        settings: Configuration to read from. Defaults to ``get_settings()``.

    Returns:
        A ready-to-use ``RetrievalEngine``.
    """
    settings = settings or get_settings()
    sparse = build_sparse_embeddings() if settings.hybrid else None
    reranker = None
    if settings.rerank:
        from src.retrieval.reranker import build_reranker

        reranker = build_reranker(settings)
    return RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
        qdrant_path=settings.qdrant_path,
        qdrant_cloud_url=settings.qdrant_cloud_url,
        qdrant_api_key=settings.qdrant_api_key,
        sparse_embeddings=sparse,
        reranker=reranker,
        rerank_fetch_k=settings.rerank_fetch_k,
    )
