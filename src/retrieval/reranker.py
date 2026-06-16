"""Cross-encoder reranking of retrieved candidates (ADR-014).

A bi-encoder (the embedding model) retrieves candidates cheaply; a cross-encoder
then re-scores each (query, passage) pair jointly for a sharper final ordering.
The model is injectable, so the ranking logic is unit-tested without downloads.
"""

import logging

from langchain_core.documents import Document

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank candidate documents with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str | None = None,
        model: object | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model id (used when ``model`` is None).
            device: Torch device for the model (e.g. ``"mps"``).
            model: A pre-built model with a ``predict(pairs)`` method. When
                provided, ``model_name``/``device`` are ignored (used in tests).
        """
        if model is not None:
            self._model = model
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[Document], *, top_k: int) -> list[Document]:
        """Return the ``top_k`` documents most relevant to ``query``.

        Args:
            query: The query text.
            documents: Candidate documents to re-score.
            top_k: Number of documents to return.

        Returns:
            The reranked documents, most relevant first (at most ``top_k``).
        """
        if not documents:
            return []
        scores = self._model.predict([(query, doc.page_content) for doc in documents])
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [documents[i] for i in order[:top_k]]


def build_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    """Construct a cross-encoder reranker from configuration.

    Args:
        settings: Configuration to read from. Defaults to ``get_settings()``.

    Returns:
        A ``CrossEncoderReranker`` placed on the configured local device.
    """
    settings = settings or get_settings()
    from src.retrieval.embeddings import _resolve_device

    device = _resolve_device(settings.embedding_device)
    logger.info("Building reranker %s on device=%s", settings.reranker_model, device)
    return CrossEncoderReranker(settings.reranker_model, device=device)
