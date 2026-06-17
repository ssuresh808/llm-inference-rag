"""Tests for generation-mode routing (single vs agent)."""

from src.config.settings import Settings
from src.generation import dispatch
from src.generation.rag import RagAnswer


def test_routes_to_single_by_default(monkeypatch):
    monkeypatch.setattr(dispatch, "answer_question", lambda q, **kw: RagAnswer(answer="SINGLE"))

    out = dispatch.generate_answer("q", settings=Settings(_env_file=None, generation_mode="single"))

    assert out.answer == "SINGLE"


def test_routes_to_agent_when_configured(monkeypatch):
    import src.generation.agent as agent_mod

    monkeypatch.setattr(
        agent_mod, "agentic_answer_question", lambda q, **kw: RagAnswer(answer="AGENT")
    )

    out = dispatch.generate_answer(
        "q", settings=Settings(_env_file=None, generation_mode="agent")
    )

    assert out.answer == "AGENT"


def test_single_path_never_invokes_agent(monkeypatch):
    import src.generation.agent as agent_mod

    touched = {"agent": False}

    def _spy(question, **kwargs):
        touched["agent"] = True
        return RagAnswer(answer="AGENT")

    monkeypatch.setattr(agent_mod, "agentic_answer_question", _spy)
    monkeypatch.setattr(dispatch, "answer_question", lambda q, **kw: RagAnswer(answer="SINGLE"))

    dispatch.generate_answer("q", settings=Settings(_env_file=None, generation_mode="single"))

    assert touched["agent"] is False
