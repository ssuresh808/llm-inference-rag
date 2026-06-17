"""Run the local RAGAS generation eval on a few on-domain questions.

Reads the persistent ``llm_optimization_domain`` collection (seed it first with
``scripts/seed_arxiv_db.py``) and scores faithfulness + answer_relevancy with the
local judge (qwen2.5:14b by default). NaN-tolerant: prints whatever the local
judge produces. Swap to a hosted judge with ``RAGAS_JUDGE_PROVIDER`` (ADR-016).

    uv run python -m scripts.run_ragas_sample
"""

import logging

from src.config.settings import get_settings
from src.evaluation.ragas_eval import evaluate_generation
from src.evaluation.tracking import log_evaluation, summarize_result
from src.retrieval.embeddings import build_embeddings
from src.retrieval.engine import RetrievalEngine

QUESTIONS = [
    "How does PagedAttention improve vLLM serving efficiency?",
    "What is speculative decoding and how does it reduce latency?",
    "How does KV-cache quantization reduce GPU memory use?",
]
TOP_K = 5


def main() -> None:
    """Evaluate the sample questions against the persistent arXiv collection."""
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    path = settings.qdrant_path or ".qdrant_storage"

    engine = RetrievalEngine(
        build_embeddings(settings),
        collection_name=settings.arxiv_collection,
        qdrant_path=path,
    )
    engine.connect_existing()

    print(
        f"Evaluating {len(QUESTIONS)} questions against "
        f"'{settings.arxiv_collection}' (judge={settings.ragas_judge_provider})...\n"
    )
    result = evaluate_generation(QUESTIONS, engine=engine, top_k=TOP_K)
    engine.close()

    print("RAGAS result:")
    print(result)

    scores, failures = summarize_result(result, QUESTIONS)
    config = {
        "retriever_top_k": TOP_K,
        "embedding_model": settings.embedding_model,
        "llm_judge_model": settings.ragas_judge_model or settings.llm_model,
        "collection_name": settings.arxiv_collection,
    }
    log_evaluation(scores, failures, config)
    if settings.enable_wandb:
        print(f"\nLogged to W&B project '{settings.wandb_project}'.")


if __name__ == "__main__":
    main()
