"""Tests for benchmark gold-set construction (offline, no network)."""

from langchain_core.documents import Document

from src.evaluation.benchmarks import build_answer_containment_gold

CORPUS = [
    Document(page_content="The capital of France is Paris.", metadata={"source": "c#1"}),
    Document(page_content="Abraham Lincoln was born in 1809.", metadata={"source": "c#2"}),
    Document(page_content="Completely unrelated filler text.", metadata={"source": "c#3"}),
]
QA = [
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "Birth year?", "answer": "1809"},
    {"question": "True?", "answer": "yes"},  # excluded (yes/no)
    {"question": "Where is Atlantis?", "answer": "Atlantis"},  # unanswerable -> dropped
    {"question": "Short?", "answer": "a"},  # below min_answer_len -> dropped
]


def test_gold_built_by_answer_containment():
    gold = build_answer_containment_gold(CORPUS, QA)
    by_q = {g["question"]: g["relevant_sources"] for g in gold}
    assert by_q["Capital of France?"] == ["c#1"]
    assert by_q["Birth year?"] == ["c#2"]
    assert len(gold) == 2


def test_gold_excludes_yes_no_unanswerable_and_short():
    qs = {g["question"] for g in build_answer_containment_gold(CORPUS, QA)}
    assert "True?" not in qs  # yes/no answer
    assert "Where is Atlantis?" not in qs  # answer not in any passage
    assert "Short?" not in qs  # answer shorter than min_answer_len
