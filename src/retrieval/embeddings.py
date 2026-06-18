"""Embedding-provider factory.

Selects the embedding backend from configuration so the rest of the system is
provider-agnostic (ADR-003). Provider integration packages are imported lazily,
so only the provider you actually use needs to be installed. Switching providers
is a config change (``EMBEDDING_PROVIDER``), not a code change.
"""

import logging

from langchain_core.embeddings import Embeddings

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _resolve_device(preference: str) -> str:
    """Resolve the torch device for local embeddings.

    Args:
        preference: ``"auto"``, ``"mps"``, ``"cuda"``, or ``"cpu"``.

    Returns:
        The concrete device string. ``"auto"`` prefers MPS, then CUDA, else CPU.
    """
    if preference != "auto":
        return preference
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the embeddings backend named by ``settings.embedding_provider``.

    Supported providers: ``"huggingface"`` (local, free - default),
    ``"voyage"`` (hosted, light), ``"openai"``, and ``"nvidia"``. The matching
    integration package is imported lazily, and a clear error is raised if it is
    not installed.

    Args:
        settings: Configuration to read from. Defaults to ``get_settings()``.

    Returns:
        A LangChain ``Embeddings`` instance for the configured provider.

    Raises:
        ValueError: If ``embedding_provider`` is not a recognized value.
        ImportError: If the selected provider's integration package is missing.
    """
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    logger.info(
        "Building embeddings (provider=%s, model=%s)",
        provider,
        settings.embedding_model,
    )

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - install guard
            raise ImportError(
                "The 'huggingface' provider needs langchain-huggingface + "
                "sentence-transformers. Install them or set EMBEDDING_PROVIDER."
            ) from exc
        device = _resolve_device(settings.embedding_device)
        logger.info(
            "HuggingFace embeddings on device=%s (batch_size=%d)",
            device,
            settings.embedding_batch_size,
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": settings.embedding_batch_size,
                "normalize_embeddings": True,
            },
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
        )

    if provider == "voyage":
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:  # pragma: no cover - install guard
            raise ImportError(
                "The 'voyage' provider needs langchain-voyageai. "
                "Install it (and set VOYAGE_API_KEY) or set EMBEDDING_PROVIDER."
            ) from exc
        return VoyageAIEmbeddings(
            model=settings.embedding_model,
            voyage_api_key=settings.voyage_api_key or None,
            batch_size=128,  # fewer, larger requests for fast bulk seeding
        )

    if provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        except ImportError as exc:  # pragma: no cover - install guard
            raise ImportError(
                "The 'nvidia' provider needs langchain-nvidia-ai-endpoints. "
                "Install it (and set NVIDIA_API_KEY) or set EMBEDDING_PROVIDER."
            ) from exc
        return NVIDIAEmbeddings(
            model=settings.embedding_model,
            api_key=settings.nvidia_api_key or None,
        )

    raise ValueError(
        f"Unknown embedding_provider '{settings.embedding_provider}'. "
        "Expected one of: huggingface, voyage, openai, nvidia."
    )


def build_sparse_embeddings(model_name: str = "Qdrant/bm25"):
    """Build a sparse (BM25) embedding for hybrid retrieval (ADR-014).

    Args:
        model_name: FastEmbed sparse model id.

    Returns:
        A ``FastEmbedSparse`` instance for use as Qdrant's sparse vector.

    Raises:
        ImportError: If ``fastembed`` / ``FastEmbedSparse`` is not available.
    """
    try:
        from langchain_qdrant import FastEmbedSparse
    except ImportError as exc:  # pragma: no cover - install guard
        raise ImportError(
            "Hybrid retrieval needs fastembed (and langchain-qdrant's "
            "FastEmbedSparse). Install fastembed or set HYBRID=false."
        ) from exc
    return FastEmbedSparse(model_name=model_name)
