"""LLM-provider factory.

Mirrors the embeddings factory (ADR-006): the chat model is chosen from config
(``LLM_PROVIDER``) so the system stays provider-agnostic. Provider integration
packages are imported lazily, so only the provider you use must be installed.
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Return the chat model named by ``settings.llm_provider``.

    Supported providers: ``"ollama"`` (local, free - default), ``"anthropic"``,
    and ``"openai"``. The matching integration package is imported lazily.

    Args:
        settings: Configuration to read from. Defaults to ``get_settings()``.

    Returns:
        A LangChain ``BaseChatModel`` for the configured provider.

    Raises:
        ValueError: If ``llm_provider`` is not a recognized value.
        ImportError: If the selected provider's integration package is missing.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    logger.info("Building LLM (provider=%s, model=%s)", provider, settings.llm_model)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - install guard
            raise ImportError(
                "The 'ollama' provider needs langchain-ollama and a running Ollama "
                "server. Install them or set LLM_PROVIDER."
            ) from exc
        return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - install guard
            raise ImportError(
                "The 'anthropic' provider needs langchain-anthropic. Install it "
                "(and set ANTHROPIC_API_KEY) or set LLM_PROVIDER."
            ) from exc
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key or None,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or None,
        )

    raise ValueError(
        f"Unknown llm_provider '{settings.llm_provider}'. "
        "Expected one of: ollama, anthropic, openai."
    )
