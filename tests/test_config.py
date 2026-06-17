"""Tests for application configuration loading (src/config/settings.py)."""

from src.config.settings import Settings, get_settings


def test_settings_reads_from_env(monkeypatch):
    """Settings should pick up values from the environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    settings = Settings(_env_file=None)  # ignore any developer .env on disk

    assert settings.openai_api_key == "sk-test-123"
    assert settings.qdrant_url == "http://localhost:6333"


def test_settings_defaults_without_env(monkeypatch):
    """Sensible defaults apply when no env vars are set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == ""
    assert settings.nvidia_api_key == ""
    assert settings.qdrant_url == ":memory:"
    assert settings.embedding_provider == "huggingface"
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"
    assert settings.embedding_device == "auto"
    assert settings.embedding_batch_size == 64
    assert settings.qdrant_collection == "rag_portfolio"
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "qwen2.5:14b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.corpus_dir == "data/sample"
    assert settings.hybrid is False
    assert settings.rerank is False
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.rerank_fetch_k == 20
    assert settings.arxiv_dataset == "librarian-bots/arxiv-metadata-snapshot"
    assert settings.arxiv_collection == "llm_optimization_domain"
    assert settings.arxiv_max_docs == 500
    assert settings.ragas_judge_provider == "ollama"
    assert settings.ragas_judge_model == ""
    assert settings.enable_wandb is False
    assert settings.wandb_project == "llm-rag-inference"
    assert settings.wandb_entity == ""
    assert settings.generation_mode == "single"


def test_get_settings_is_cached():
    """get_settings returns the same cached instance across calls."""
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()

    assert first is second
