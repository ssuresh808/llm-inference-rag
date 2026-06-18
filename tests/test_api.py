"""Tests for the FastAPI app (src/api/main.py).

The retrieval engine and LLM are overridden with fakes, so the API is exercised
end-to-end (including the RAG endpoint) without any network call or model.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.api.main import app, get_engine, get_llm
from src.config.settings import Settings
from src.retrieval.engine import RetrievalEngine


def test_get_engine_attaches_to_persistent_collection(monkeypatch):
    from src.api import main as api_main

    class _FakeEngine:
        def __init__(self):
            self.connected = False
            self.indexed = False

        def connect_existing(self):
            self.connected = True

        def index(self, chunks):
            self.indexed = True

    fake = _FakeEngine()
    monkeypatch.setattr(api_main, "build_engine", lambda: fake)
    monkeypatch.setattr(
        api_main, "get_settings", lambda: Settings(_env_file=None, qdrant_path="/tmp/seeded")
    )
    api_main.get_engine.cache_clear()
    try:
        engine = api_main.get_engine()
    finally:
        api_main.get_engine.cache_clear()

    assert engine is fake
    assert fake.connected is True
    assert fake.indexed is False  # persistent DB is not re-indexed


def test_get_engine_attaches_to_cloud_collection(monkeypatch):
    from src.api import main as api_main

    class _FakeEngine:
        def __init__(self):
            self.connected = False
            self.indexed = False

        def connect_existing(self):
            self.connected = True

        def index(self, chunks):
            self.indexed = True

    fake = _FakeEngine()
    monkeypatch.setattr(api_main, "build_engine", lambda: fake)
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            qdrant_cloud_url="https://x.cloud.qdrant.io:6333",
            qdrant_api_key="k",
        ),
    )
    api_main.get_engine.cache_clear()
    try:
        engine = api_main.get_engine()
    finally:
        api_main.get_engine.cache_clear()

    assert engine is fake
    assert fake.connected is True  # cloud creds -> attach to seeded collection
    assert fake.indexed is False


def _indexed_engine() -> RetrievalEngine:
    engine = RetrievalEngine(
        DeterministicFakeEmbedding(size=8),
        collection_name="api_test",
    )
    engine.index(
        [
            Document(
                page_content="PagedAttention manages the KV cache",
                metadata={"source": "vllm.md"},
            ),
            Document(
                page_content="Speculative decoding reduces latency",
                metadata={"source": "spec.md"},
            ),
        ]
    )
    return engine


@pytest.fixture
def client():
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_health_boots_without_engine(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_returns_matching_chunk(client):
    engine = _indexed_engine()
    app.dependency_overrides[get_engine] = lambda: engine
    target = "PagedAttention manages the KV cache"

    resp = client.post("/api/v1/query", json={"text": target, "top_k": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["text"] == target
    assert body["results"][0]["source"] == "vllm.md"


def test_query_empty_index_returns_empty_list(client):
    empty = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="empty")
    app.dependency_overrides[get_engine] = lambda: empty

    resp = client.post("/api/v1/query", json={"text": "anything"})

    assert resp.status_code == 200
    assert resp.json() == {"results": []}


def test_query_rejects_empty_text(client):
    resp = client.post("/api/v1/query", json={"text": ""})
    assert resp.status_code == 422


def test_query_rejects_out_of_range_top_k(client):
    resp = client.post("/api/v1/query", json={"text": "ok", "top_k": 0})
    assert resp.status_code == 422


def test_answer_returns_grounded_answer(client):
    engine = _indexed_engine()
    reply = "PagedAttention pages the KV cache into blocks."
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_llm] = lambda: FakeListChatModel(responses=[reply])

    resp = client.post(
        "/api/v1/answer",
        json={"text": "how does paged attention work?", "top_k": 2},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == reply
    assert "vllm.md" in body["sources"]
    assert len(body["results"]) == 2


def test_answer_rejects_empty_text(client):
    resp = client.post("/api/v1/answer", json={"text": ""})
    assert resp.status_code == 422


def test_answer_empty_index_is_graceful(client):
    empty = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="empty2")
    app.dependency_overrides[get_engine] = lambda: empty
    app.dependency_overrides[get_llm] = lambda: FakeListChatModel(responses=["unused"])

    resp = client.post("/api/v1/answer", json={"text": "anything"})

    assert resp.status_code == 200
    assert resp.json()["results"] == []
