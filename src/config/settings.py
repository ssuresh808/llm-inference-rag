"""Application configuration.

All settings load from environment variables (and a local ``.env`` file) via
``pydantic-settings``. Secrets must never be hardcoded or committed; see
``.env.example`` for the expected keys.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, validated configuration for the RAG system.

    Attributes:
        embedding_provider: Embedding backend to use. One of ``"huggingface"``
            (local, free, no key — Phase 1 default), ``"nvidia"``, or
            ``"openai"``. Swappable without code changes (see ADR-003).
        embedding_model: Model identifier for the active provider (a
            sentence-transformers repo id by default).
        embedding_device: Torch device for local embeddings: "auto" (prefer
            MPS/CUDA, else CPU) or an explicit "mps"/"cuda"/"cpu".
        embedding_batch_size: Batch size for local embedding encoding.
        openai_api_key: API key for the OpenAI provider. Optional; only needed
            when ``embedding_provider == "openai"``.
        nvidia_api_key: API key for NVIDIA NeMo Retriever (hosted). Optional;
            only needed when ``embedding_provider == "nvidia"``.
        qdrant_url: Qdrant location. Use ``":memory:"`` for the local in-memory
            store (Phase 1 default) or an ``http(s)://host:port`` URL for a
            running Qdrant instance.
        qdrant_collection: Name of the Qdrant collection holding the corpus.
        llm_provider: LLM backend. One of ``"ollama"`` (local, free — default),
            ``"anthropic"``, or ``"openai"``. Swappable via config (ADR-006).
        llm_model: Model identifier for the active LLM provider.
        ollama_base_url: Base URL of the local Ollama server.
        anthropic_api_key: API key for the Anthropic provider. Optional; only
            needed when ``llm_provider == "anthropic"``.
        corpus_dir: Directory of source documents the API indexes on first use.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    embedding_provider: str = Field(
        default="huggingface",
        description="Embedding backend: 'huggingface' (local/free), 'nvidia', or 'openai'.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Model id for the active provider (HuggingFace repo id by default).",
    )
    embedding_device: str = Field(
        default="auto",
        description="Torch device for local embeddings: 'auto', 'mps', 'cuda', or 'cpu'.",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Batch size for local embedding encoding.",
    )
    openai_api_key: str = Field(
        default="", description="API key for the OpenAI provider (optional)."
    )
    nvidia_api_key: str = Field(
        default="", description="API key for NVIDIA NeMo Retriever, hosted (optional)."
    )
    qdrant_url: str = Field(
        default=":memory:",
        description="Qdrant URL, or ':memory:' for the local in-memory store.",
    )
    qdrant_collection: str = Field(
        default="rag_portfolio",
        description="Qdrant collection name for the indexed corpus.",
    )
    llm_provider: str = Field(
        default="ollama",
        description="LLM backend: 'ollama' (local/free), 'anthropic', or 'openai'.",
    )
    llm_model: str = Field(
        default="llama3.2",
        description="Model id for the active LLM provider.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    anthropic_api_key: str = Field(
        default="", description="API key for the Anthropic provider (optional)."
    )
    corpus_dir: str = Field(
        default="data/sample",
        description="Directory of source documents the API indexes on first use.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance.

    Returns:
        The singleton ``Settings`` object, parsed once and reused. Call
        ``get_settings.cache_clear()`` in tests that mutate the environment.
    """
    return Settings()
