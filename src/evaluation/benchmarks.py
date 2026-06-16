"""Benchmark dataset loaders + gold-set construction for retrieval evaluation.

Supports ``rag-datasets/rag-mini-wikipedia``, which bundles a passage corpus and
a QA set. The QA set has **no explicit passage labels**, so gold relevance is
derived by **answer containment**: a passage is relevant to a question if it
contains the (non yes/no) answer string. This yields standard answer-recall@k /
MRR retrieval metrics over real ground truth, and reuses the same
``evaluate_retrieval`` harness (matching on ``source`` metadata).
"""

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MINI_WIKI = "rag-datasets/rag-mini-wikipedia"


def load_mini_wiki_corpus() -> list[Document]:
    """Load the mini-wikipedia passage corpus as documents.

    Returns:
        One ``Document`` per passage, with ``metadata['source']`` set to
        ``"mini-wiki#<id>"``.
    """
    from datasets import load_dataset

    passages = load_dataset(MINI_WIKI, "text-corpus")["passages"]
    documents = [
        Document(
            page_content=str(row["passage"]).strip(),
            metadata={"source": f"mini-wiki#{row['id']}"},
        )
        for row in passages
        if str(row.get("passage", "")).strip()
    ]
    logger.info("Loaded %d mini-wiki passages", len(documents))
    return documents


def load_mini_wiki_qa() -> list[dict]:
    """Load the mini-wikipedia QA set.

    Returns:
        Items of ``{"question", "answer"}``.
    """
    from datasets import load_dataset

    qa = load_dataset(MINI_WIKI, "question-answer")["test"]
    return [
        {"question": str(row["question"]).strip(), "answer": str(row["answer"]).strip()}
        for row in qa
    ]


def build_answer_containment_gold(
    corpus: list[Document],
    qa_items: list[dict],
    *,
    exclude_yes_no: bool = True,
    min_answer_len: int = 3,
) -> list[dict]:
    """Build a retrieval gold set via answer containment.

    For each QA item, ``relevant_sources`` is the list of corpus passage ids
    whose text contains the answer (case-insensitive). Yes/no and too-short
    answers are skipped (no extractable span); questions with no containing
    passage are dropped (not answerable from the corpus).

    Args:
        corpus: The passage documents (with ``source`` metadata).
        qa_items: Items of ``{"question", "answer"}``.
        exclude_yes_no: Drop questions whose answer is exactly "yes"/"no".
        min_answer_len: Drop questions whose answer is shorter than this.

    Returns:
        Gold items of ``{"question", "relevant_sources"}`` (harness-compatible).
    """
    lowered = [(doc.metadata["source"], doc.page_content.lower()) for doc in corpus]
    gold: list[dict] = []
    for item in qa_items:
        answer = item["answer"].strip()
        answer_low = answer.lower()
        if exclude_yes_no and answer_low in {"yes", "no"}:
            continue
        if len(answer) < min_answer_len:
            continue
        relevant = [source for source, text in lowered if answer_low in text]
        if not relevant:
            continue
        gold.append({"question": item["question"], "relevant_sources": relevant})

    logger.info("Built gold set: %d / %d questions answerable", len(gold), len(qa_items))
    return gold
