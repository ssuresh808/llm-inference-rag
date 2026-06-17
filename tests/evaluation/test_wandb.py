"""Tests for the W&B tracking wrapper (fully mocked, offline, no API key).

``wandb`` is injected via ``tracking._get_wandb`` so these tests never import
the real library or touch the network.
"""

import math

import pandas as pd

from src.config.settings import Settings
from src.evaluation import tracking


class _FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *row):
        self.rows.append(row)


class _FakeWandb:
    def __init__(self):
        self.init_kwargs = None
        self.logged = []
        self.finished = False
        self.Table = _FakeTable

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def log(self, data):
        self.logged.append(data)

    def finish(self):
        self.finished = True


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_no_wandb_calls_when_disabled(monkeypatch):
    touched = []
    monkeypatch.setattr(tracking, "_get_wandb", lambda: touched.append(1) or _FakeWandb())

    tracking.log_evaluation(
        {"faithfulness": 0.5}, [], {"k": 1}, settings=_settings(enable_wandb=False)
    )

    assert touched == []  # _get_wandb (and thus wandb) is never invoked


def test_logs_scores_config_and_finishes_when_enabled(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setattr(tracking, "_get_wandb", lambda: fake)

    tracking.log_evaluation(
        {"faithfulness": 0.48, "answer_relevancy": 0.65},
        [],
        {"retriever_top_k": 5, "collection_name": "c"},
        settings=_settings(enable_wandb=True, wandb_project="proj"),
    )

    assert fake.init_kwargs["project"] == "proj"
    assert fake.init_kwargs["config"] == {"retriever_top_k": 5, "collection_name": "c"}
    metric_log = fake.logged[0]
    assert metric_log["faithfulness"] == 0.48
    assert metric_log["answer_relevancy"] == 0.65
    assert metric_log["eval/judge_parser_failures"] == 0
    assert fake.finished is True


def test_nan_filtered_failures_counted_and_table_logged(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setattr(tracking, "_get_wandb", lambda: fake)

    scores = {"faithfulness": float("nan"), "answer_relevancy": 0.65}
    failures = [
        {
            "question": "Q1",
            "generated_answer": "A1",
            "raw_judge_output": "bad json",
            "failed_metric": "faithfulness",
        }
    ]

    tracking.log_evaluation(scores, failures, {}, settings=_settings(enable_wandb=True))

    metric_log = fake.logged[0]
    assert "faithfulness" not in metric_log  # NaN filtered: no bad float logged
    assert metric_log["answer_relevancy"] == 0.65
    assert metric_log["eval/judge_parser_failures"] == 1

    table_logs = [d for d in fake.logged if "Evaluation Failures" in d]
    assert table_logs, "failure debug table should be logged"
    table = table_logs[0]["Evaluation Failures"]
    assert table.rows[0][0] == "Q1"
    assert table.rows[0][3] == "faithfulness"
    assert fake.finished is True


def test_summarize_result_from_dict_marks_all_failed():
    scores, failures = tracking.summarize_result(
        {"faithfulness": float("nan"), "answer_relevancy": float("nan"), "error": "boom"},
        ["q1", "q2"],
    )
    assert math.isnan(scores["faithfulness"])
    assert len(failures) == 2
    assert failures[0]["failed_metric"] == "all"
    assert failures[0]["raw_judge_output"] == "boom"


def test_summarize_result_from_dataframe_means_and_failures():
    df = pd.DataFrame(
        {
            "user_input": ["q1", "q2"],
            "response": ["a1", "a2"],
            "faithfulness": [0.8, float("nan")],
            "answer_relevancy": [0.6, 0.4],
        }
    )

    class _Result:
        def to_pandas(self):
            return df

    scores, failures = tracking.summarize_result(_Result(), ["q1", "q2"])

    assert scores["faithfulness"] == 0.8  # mean over non-NaN rows only
    assert abs(scores["answer_relevancy"] - 0.5) < 1e-9
    assert len(failures) == 1
    assert failures[0]["question"] == "q2"
    assert failures[0]["failed_metric"] == "faithfulness"
