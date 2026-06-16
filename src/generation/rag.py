"""Retrieval-augmented generation: retrieve context, then generate an answer.

Combines the retrieval engine (Step 3) with an LLM (ADR-006). The LLM is built
via the provider factory, defaulting to a local Ollama model (free), and is
injectable so tests can supply a fake chat model and stay offline.

Run as a CLI (needs a running Ollama server for the default provider)::

    uv run python -m src.generation.rag --query "..."
"""

import argparse
import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.generation.llm import build_llm
from src.ingestion.loader import ingest_directory
from src.retrieval.engine import RetrievalEngine, build_engine

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise technical assistant. Answer the question using ONLY "
            "the provided context. If the context is insufficient, say you don't "
            "know. Cite the source filename(s) you used.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class RagAnswer:
    """The result of a RAG query."""

    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)


def _format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks into a labelled prompt context block.

    Args:
        chunks: Retrieved documents.

    Returns:
        A single string with each chunk prefixed by its source filename.
    """
    return "\n\n".join(
        f"[{chunk.metadata.get('source', '?')}]\n{chunk.page_content}" for chunk in chunks
    )


def answer_question(
    question: str,
    *,
    engine: RetrievalEngine | None = None,
    llm: BaseChatModel | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> RagAnswer:
    """Answer ``question`` from retrieved context using an LLM.

    Args:
        question: The user's question.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        llm: Chat model. Defaults to ``build_llm()`` (Ollama by default).
        top_k: Number of chunks to retrieve as context.

    Returns:
        A ``RagAnswer`` with the generated answer, the source filenames used, and
        the retrieved chunks.
    """
    engine = engine or build_engine()
    llm = llm or build_llm()

    chunks = engine.query(question, top_k=top_k)
    messages = _PROMPT.format_messages(context=_format_context(chunks), question=question)
    response = llm.invoke(messages)

    sources = list(
        dict.fromkeys(
            chunk.metadata.get("source") for chunk in chunks if chunk.metadata.get("source")
        )
    )
    content = response.content
    answer_text = content if isinstance(content, str) else str(content)
    return RagAnswer(answer=answer_text, sources=sources, chunks=chunks)


def main() -> None:
    """CLI entry point: ingest a corpus, index it, and answer a question."""
    parser = argparse.ArgumentParser(description="Run retrieval-augmented generation.")
    parser.add_argument("--corpus", default="data/sample", help="Corpus directory.")
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Context chunks.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chunks = ingest_directory(args.corpus)
    engine = build_engine()
    engine.index(chunks)
    result = answer_question(args.query, engine=engine, top_k=args.top_k)

    print(f"\nQ: {args.query}\n")
    print(f"A: {result.answer}\n")
    print(f"Sources: {', '.join(result.sources) or '(none)'}\n")


if __name__ == "__main__":
    main()
