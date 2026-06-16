"""Tests for the LLM factory and the RAG answer function (fully offline).

The LLM is a ``FakeListChatModel`` and embeddings are deterministic fakes, so the
retrieve -> prompt -> generate path is exercised without any network or model.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config.settings import Settings
from src.generation.llm import build_llm
from src.generation.rag import answer_question
from src.retrieval.engine import RetrievalEngine


def _indexed_engine() -> RetrievalEngine:
    engine = RetrievalEngine(
        DeterministicFakeEmbedding(size=8),
        collection_name="gen_test",
    )
    engine.index(
        [
            Document(
                page_content="PagedAttention manages the KV cache in vLLM",
                metadata={"source": "paged-attention.md"},
            ),
            Document(
                page_content="Continuous batching raises GPU throughput",
                metadata={"source": "continuous-batching.md"},
            ),
        ]
    )
    return engine


def test_answer_question_uses_llm_and_returns_sources():
    engine = _indexed_engine()
    fake_llm = FakeListChatModel(responses=["PagedAttention pages the KV cache."])

    result = answer_question(
        "How does paged attention work?",
        engine=engine,
        llm=fake_llm,
        top_k=2,
    )

    assert result.answer == "PagedAttention pages the KV cache."
    assert "paged-attention.md" in result.sources
    assert len(result.chunks) == 2


def test_build_llm_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_llm(Settings(_env_file=None, llm_provider="bogus"))


def test_build_llm_dispatches_to_openai():
    from langchain_openai import ChatOpenAI

    llm = build_llm(
        Settings(
            _env_file=None,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            openai_api_key="sk-test",
        )
    )

    assert isinstance(llm, ChatOpenAI)
