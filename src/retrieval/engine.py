"""Vector indexing and retrieval over Qdrant.

``RetrievalEngine`` wraps a Qdrant vector store and an embeddings backend. The
embeddings are injected, so tests can supply a deterministic fake and exercise
the index/query path without any network access or model download.
"""

import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import build_embeddings

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


class RetrievalEngine:
    """Index document chunks into Qdrant and retrieve the most similar ones."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        collection_name: str = "rag_portfolio",
        qdrant_url: str = ":memory:",
    ) -> None:
        """Initialize the engine.

        Args:
            embeddings: Embeddings backend used for both indexing and queries.
            collection_name: Qdrant collection to write to / read from.
            qdrant_url: ``":memory:"`` for a local in-memory store, otherwise an
                ``http(s)://host:port`` URL of a running Qdrant instance.
        """
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._qdrant_url = qdrant_url
        self._vector_store: QdrantVectorStore | None = None

    def index(self, chunks: list[Document]) -> int:
        """Index chunks, creating the collection on first call.

        The vector dimension is inferred from the embeddings, so the collection
        matches whichever provider is configured.

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
            location_kwarg = (
                {"location": ":memory:"}
                if self._qdrant_url == ":memory:"
                else {"url": self._qdrant_url}
            )
            self._vector_store = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=self._embeddings,
                collection_name=self._collection_name,
                **location_kwarg,
            )
            count = len(chunks)
        else:
            count = len(self._vector_store.add_documents(chunks))

        logger.info("Indexed %d chunk(s) into '%s'", count, self._collection_name)
        return count

    def query(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[Document]:
        """Return the ``top_k`` chunks most similar to ``text``.

        Args:
            text: Natural-language query.
            top_k: Maximum number of chunks to return.

        Returns:
            The matching chunks, most similar first.

        Raises:
            RuntimeError: If called before anything has been indexed.
        """
        if self._vector_store is None:
            raise RuntimeError("Index is empty; call index() before query().")
        return self._vector_store.similarity_search(text, k=top_k)


def build_engine(settings: Settings | None = None) -> RetrievalEngine:
    """Construct a ``RetrievalEngine`` from configuration.

    Wires the configured embedding provider (ADR-003) and Qdrant location.

    Args:
        settings: Configuration to read from. Defaults to ``get_settings()``.

    Returns:
        A ready-to-use ``RetrievalEngine``.
    """
    settings = settings or get_settings()
    return RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.qdrant_collection,
        qdrant_url=settings.qdrant_url,
    )
