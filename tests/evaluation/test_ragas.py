"""Tests for the RAGAS generation-evaluation loop (fully mocked, no network/GPU).

The generator (answer_question), the retriever (engine), and the RAGAS scoring
call (_run_ragas) are all mocked, so these run instantly and offline.
"""

import math

from langchain_core.documents import Document

from src.config.settings import Settings
from src.evaluation import ragas_eval
from src.generation.rag import RagAnswer


def _fake_answer(question, **kwargs):
    return RagAnswer(
        answer=f"answer to {question}",
        sources=["arxiv:1"],
        chunks=[Document(page_content="ctx chunk", metadata={"source": "arxiv:1"})],
    )


def test_evaluate_generation_builds_ragas_dataset_and_returns_scores(monkeypatch):
    monkeypatch.setattr(ragas_eval, "answer_question", _fake_answer)
    captured = {}

    def fake_run(dataset, *, judge, embeddings):
        captured["dataset"] = dataset
        captured["judge"] = judge
        return {"faithfulness": 0.8, "answer_relevancy": 0.9}

    monkeypatch.setattr(ragas_eval, "_run_ragas", fake_run)

    out = ragas_eval.evaluate_generation(
        ["Q1", "Q2"], engine=object(), judge="JUDGE", embeddings="EMB"
    )

    assert out == {"faithfulness": 0.8, "answer_relevancy": 0.9}
    dataset = captured["dataset"]
    assert set(dataset.column_names) == {"question", "answer", "contexts"}
    assert len(dataset) == 2
    assert dataset[0]["contexts"] == ["ctx chunk"]
    assert captured["judge"] == "JUDGE"  # injected judge is used (no OpenAI default)


def test_evaluate_generation_is_nan_tolerant(monkeypatch):
    monkeypatch.setattr(ragas_eval, "answer_question", _fake_answer)

    def boom(dataset, *, judge, embeddings):
        raise ValueError("could not parse JSON from local judge")

    monkeypatch.setattr(ragas_eval, "_run_ragas", boom)

    out = ragas_eval.evaluate_generation(["Q"], engine=object(), judge="J", embeddings="E")

    assert math.isnan(out["faithfulness"])
    assert math.isnan(out["answer_relevancy"])
    assert "could not parse JSON" in out["error"]


def test_judge_settings_uses_ragas_provider_and_model():
    settings = Settings(
        _env_file=None,
        ragas_judge_provider="openai",
        ragas_judge_model="gpt-4o",
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    )
    judge_settings = ragas_eval._judge_settings(settings)
    assert judge_settings.llm_provider == "openai"
    assert judge_settings.llm_model == "gpt-4o"


def test_judge_settings_falls_back_to_llm_model():
    settings = Settings(
        _env_file=None,
        ragas_judge_provider="ollama",
        ragas_judge_model="",
        llm_model="qwen2.5:14b",
    )
    judge_settings = ragas_eval._judge_settings(settings)
    assert judge_settings.llm_provider == "ollama"
    assert judge_settings.llm_model == "qwen2.5:14b"
