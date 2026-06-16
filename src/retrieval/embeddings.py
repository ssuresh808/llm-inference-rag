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


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the embeddings backend named by ``settings.embedding_provider``.

    Supported providers: ``"huggingface"`` (local, free — default),
    ``"openai"``, and ``"nvidia"``. The matching integration package is imported
    lazily, and a clear error is raised if it is not installed.

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
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
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
        "Expected one of: huggingface, openai, nvidia."
    )
