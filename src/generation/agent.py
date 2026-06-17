"""Agentic generation via a LangGraph ReAct agent (ADR-018, behind a flag).

Wraps the Qdrant retrieval engine as a ``search_arxiv_literature`` tool and lets
a ReAct agent plan + actively search to synthesize an answer. This path is
config-gated by ``GENERATION_MODE``; the single-shot :func:`answer_question`
remains the default, RAGAS-measured baseline.

``langgraph`` is imported lazily inside :func:`build_react_agent`, so this module
and its tests load without constructing an agent and make no network/GPU calls.
"""

import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

from src.config.settings import Settings, get_settings
from src.generation.rag import RagAnswer
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)

AGENT_INSTRUCTIONS = (
    "You are an expert assistant on LLM inference optimization (vLLM, "
    "TensorRT-LLM, quantization, KV-cache, continuous batching, speculative "
    "decoding, GPU serving cost/latency).\n\n"
    "You have NO prior knowledge of this corpus. You MUST call the "
    "`search_arxiv_literature` tool at least once to gather evidence BEFORE you "
    "write any answer. Never answer from memory. If the question has multiple "
    "parts, search for each part. Base your synthesis ONLY on the passages the "
    "tool returns, and cite the source ids you used. If the tool returns nothing "
    "relevant, say so plainly rather than guessing."
)


def _make_search_tool(
    engine: RetrievalEngine, top_k: int, collected: list[Document]
) -> BaseTool:
    """Build a retrieval tool bound to ``engine`` that records what it returns.

    Args:
        engine: The retrieval engine to search.
        top_k: Passages to return per call.
        collected: List mutated in place with every retrieved document, so the
            caller can assemble citations after the agent finishes.

    Returns:
        A LangChain tool the agent can call with a ``query`` string.
    """

    @tool
    def search_arxiv_literature(query: str) -> str:
        """Search the arXiv LLM-inference-optimization corpus for passages.

        Args:
            query: A focused natural-language search query.

        Returns:
            Retrieved passages, each prefixed with its source id.
        """
        documents = engine.query(query, top_k=top_k)
        collected.extend(documents)
        if not documents:
            return "No relevant passages found."
        return "\n\n".join(
            f"[{doc.metadata.get('source', '?')}] {doc.page_content}" for doc in documents
        )

    return search_arxiv_literature


def build_react_agent(tools: list[BaseTool], settings: Settings | None = None) -> object:
    """Create a LangGraph ReAct agent driven by the configured local/hosted LLM.

    A lightweight tool-calling harness (chosen over deepagents for reliable
    tool use on local ~14B models — see ADR-018).

    Args:
        tools: Tools the agent may call (e.g. the retrieval tool).
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A compiled LangGraph agent, invoked with ``{"messages": [...]}``.
    """
    settings = settings or get_settings()
    from langgraph.prebuilt import create_react_agent

    from src.generation.llm import build_llm

    return create_react_agent(build_llm(settings), tools, prompt=AGENT_INSTRUCTIONS)


def _extract_final_text(result: object) -> str:
    """Pull the final assistant message text out of an agent result state."""
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
    return content if isinstance(content, str) else str(content)


def agentic_answer_question(
    question: str,
    *,
    engine: RetrievalEngine | None = None,
    agent: object | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> RagAnswer:
    """Answer ``question`` with the deep agent, returning the baseline schema.

    The agent decides when/what to search; retrieved documents are captured for
    citations so the output matches :class:`RagAnswer` exactly (answer + sources
    + chunks), keeping the API response schema identical across modes.

    Args:
        question: The user's question.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        agent: Pre-built agent (injected in tests). Defaults to a fresh agent.
        top_k: Passages per tool call.
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A ``RagAnswer`` with the agent's synthesis, cited sources, and chunks.
    """
    settings = settings or get_settings()
    engine = engine or build_engine()
    collected: list[Document] = []
    search_tool = _make_search_tool(engine, top_k, collected)
    if agent is None:
        agent = build_react_agent([search_tool], settings)

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer_text = _extract_final_text(result)
    sources = list(
        dict.fromkeys(
            doc.metadata.get("source") for doc in collected if doc.metadata.get("source")
        )
    )
    return RagAnswer(answer=answer_text, sources=sources, chunks=collected)
