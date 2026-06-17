"""Route a question to the configured generation path (``GENERATION_MODE``).

``single`` (default) uses the RAGAS-measured single-shot baseline; ``agent`` uses
the LangGraph ReAct path (ADR-018). Both return the identical :class:`RagAnswer`
schema, so the API response is the same regardless of mode. The agent module is
imported lazily so the default path never loads the agent stack.
"""

from src.config.settings import Settings, get_settings
from src.generation.rag import RagAnswer, answer_question


def generate_answer(
    question: str,
    *,
    engine: object | None = None,
    llm: object | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> RagAnswer:
    """Generate an answer using the configured generation mode.

    Args:
        question: The user's question.
        engine: Retrieval engine (overridable in tests).
        llm: Chat model for the single-shot path (overridable in tests).
        top_k: Context chunks / passages to use.
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A ``RagAnswer`` produced by the selected generation path.
    """
    settings = settings or get_settings()
    if settings.generation_mode == "agent":
        from src.generation.agent import agentic_answer_question

        return agentic_answer_question(question, engine=engine, top_k=top_k, settings=settings)
    return answer_question(question, engine=engine, llm=llm, top_k=top_k)
