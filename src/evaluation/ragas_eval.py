"""Local RAGAS generation evaluation (faithfulness + answer_relevancy).

Measures generation quality with a **local** judge by default (qwen2.5:14b via
Ollama) and local bge-large embeddings — no OpenAI. Because local models often
emit malformed JSON that RAGAS parsers reject, the loop is NaN-tolerant: a parse
failure is logged and returned as NaN rather than crashing. A hosted judge can be
swapped in via ``RAGAS_JUDGE_PROVIDER`` for cleaner numbers (ADR-016).

RAGAS/ragas imports are lazy, so this module (and its tests) load without the
``ragas`` package or any network/GPU dependency.
"""

import logging

from src.config.settings import Settings, get_settings
from src.generation.rag import answer_question

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy")


def _ensure_ragas_importable() -> None:
    """Shim a removed module that ragas 0.4.x eagerly imports but we never use.

    ragas 0.4.3 imports ``langchain_community.chat_models.vertexai`` at load time;
    that submodule was removed in recent langchain-community. Since we only use
    ragas with our own (Ollama) judge, inject a harmless stand-in so the import
    succeeds (ADR-016).
    """
    import importlib.util
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules and importlib.util.find_spec(name) is None:
        shim = types.ModuleType(name)
        shim.ChatVertexAI = object
        sys.modules[name] = shim


def _judge_settings(settings: Settings) -> Settings:
    """Return settings overridden to use the RAGAS judge provider/model.

    Args:
        settings: Base application settings.

    Returns:
        A copy whose ``llm_provider``/``llm_model`` point at the RAGAS judge
        (``ragas_judge_model`` falls back to ``llm_model`` when empty).
    """
    return settings.model_copy(
        update={
            "llm_provider": settings.ragas_judge_provider,
            "llm_model": settings.ragas_judge_model or settings.llm_model,
        }
    )


def build_judge(settings: Settings | None = None) -> object:
    """Build the RAGAS judge LLM via the existing provider factory.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A RAGAS-compatible LLM wrapper (no external API unless the configured
        judge provider is hosted).
    """
    settings = settings or get_settings()
    _ensure_ragas_importable()
    from ragas.llms import LangchainLLMWrapper

    from src.generation.llm import build_llm

    return LangchainLLMWrapper(build_llm(_judge_settings(settings)))


def build_judge_embeddings(settings: Settings | None = None) -> object:
    """Build RAGAS embeddings from the local HuggingFace embedding backend.

    Args:
        settings: Configuration. Defaults to ``get_settings()``.

    Returns:
        A RAGAS-compatible embeddings wrapper.
    """
    settings = settings or get_settings()
    _ensure_ragas_importable()
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from src.retrieval.embeddings import build_embeddings

    return LangchainEmbeddingsWrapper(build_embeddings(settings))


def _to_dataset(rows: list[dict]) -> object:
    """Build a HuggingFace ``Dataset`` from question/answer/contexts rows."""
    from datasets import Dataset

    return Dataset.from_list(rows)


def _run_ragas(dataset: object, *, judge: object, embeddings: object) -> object:
    """Run RAGAS ``evaluate`` with faithfulness + answer_relevancy.

    Maps our ``question``/``answer``/``contexts`` columns to the names ragas
    0.4.x expects (``user_input``/``response``/``retrieved_contexts``).
    """
    _ensure_ragas_importable()
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    ragas_dataset = dataset.rename_columns(
        {
            "question": "user_input",
            "answer": "response",
            "contexts": "retrieved_contexts",
        }
    )
    return evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
    )


def _nan_scores(error: str) -> dict:
    """Return a NaN-filled score dict carrying the failure reason."""
    scores: dict = {name: float("nan") for name in METRIC_NAMES}
    scores["error"] = error
    return scores


def evaluate_generation(
    questions: list[str],
    *,
    engine: object | None = None,
    llm: object | None = None,
    judge: object | None = None,
    embeddings: object | None = None,
    top_k: int = 5,
) -> object:
    """Evaluate generation quality over ``questions`` with RAGAS.

    Retrieves + generates an answer per question, formats them as a RAGAS
    dataset (``question``/``answer``/``contexts``), and scores faithfulness and
    answer_relevancy with a local-by-default judge. Failures from a flaky local
    judge are caught and returned as NaN (logged), not raised.

    Args:
        questions: Questions to evaluate.
        engine: Retrieval engine. Defaults to ``build_engine()``.
        llm: Generator LLM. Defaults to ``answer_question``'s own default.
        judge: RAGAS judge LLM. Defaults to ``build_judge()``.
        embeddings: RAGAS embeddings. Defaults to ``build_judge_embeddings()``.
        top_k: Number of context chunks per question.

    Returns:
        The RAGAS result (scores), or a NaN-filled dict with an ``error`` key.
    """
    if engine is None:
        from src.retrieval.engine import build_engine

        engine = build_engine()

    rows: list[dict] = []
    for question in questions:
        result = answer_question(question, engine=engine, llm=llm, top_k=top_k)
        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "contexts": [chunk.page_content for chunk in result.chunks],
            }
        )
    dataset = _to_dataset(rows)

    judge = judge or build_judge()
    embeddings = embeddings or build_judge_embeddings()

    try:
        return _run_ragas(dataset, judge=judge, embeddings=embeddings)
    except Exception as exc:  # noqa: BLE001 - local-judge JSON formatting is fragile
        logger.warning("RAGAS evaluation failed (likely local-judge formatting): %s", exc)
        return _nan_scores(str(exc))
