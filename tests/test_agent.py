"""Tests for the agentic generation path (offline; langgraph + LLM not loaded)."""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from src.generation.agent import _make_search_tool, agentic_answer_question
from src.retrieval.engine import RetrievalEngine


def _engine() -> RetrievalEngine:
    engine = RetrievalEngine(DeterministicFakeEmbedding(size=8), collection_name="agent_test")
    engine.index(
        [
            Document(
                page_content="PagedAttention manages the KV cache in vLLM.",
                metadata={"source": "paged.md"},
            ),
            Document(
                page_content="Speculative decoding reduces decode latency.",
                metadata={"source": "spec.md"},
            ),
        ]
    )
    return engine


def test_search_tool_returns_passages_and_records_sources():
    collected: list[Document] = []
    search = _make_search_tool(_engine(), top_k=2, collected=collected)

    out = search.invoke({"query": "paged attention kv cache"})

    assert isinstance(out, str) and out
    assert len(collected) == 2  # tool records what it retrieved for citations


class _FakeAgent:
    """Stand-in deep agent that returns a fixed final message (no LLM/tool)."""

    def __init__(self, messages):
        self.messages = messages

    def invoke(self, state):
        self._state = state
        return {"messages": self.messages}


def test_agentic_answer_extracts_final_dict_message():
    agent = _FakeAgent(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Agentic synthesis."}]
    )
    out = agentic_answer_question(
        "How does paged attention work?", engine=_engine(), agent=agent, top_k=2
    )
    assert out.answer == "Agentic synthesis."


def test_agentic_answer_extracts_final_aimessage_object():
    agent = _FakeAgent([AIMessage(content="From an AIMessage.")])
    out = agentic_answer_question("q", engine=_engine(), agent=agent, top_k=1)
    assert out.answer == "From an AIMessage."


def test_agentic_answer_passes_question_to_agent():
    agent = _FakeAgent([AIMessage(content="ok")])
    agentic_answer_question("explain KV cache", engine=_engine(), agent=agent, top_k=1)
    assert agent._state["messages"][0]["content"] == "explain KV cache"
