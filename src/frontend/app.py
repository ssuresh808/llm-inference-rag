"""Streamlit frontend for the RAG API: a clean, technical retrieval tool.

Connects to the FastAPI ``/api/v1/answer`` endpoint and renders the grounded
answer beside source cards (arXiv title + id + hyperlink), so retrieval
provenance is the visible proof of work. Styling lives in ``style.css``. Run::

    streamlit run src/frontend/app.py

Set ``RAG_API_URL`` to point at the backend (default ``http://localhost:8000``;
``http://backend:8000`` under docker-compose).
"""

import html
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0
_CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

_SUBTITLE = (
    "Grounded answers over a curated arXiv corpus on vLLM, KV-cache, "
    "quantization, speculative decoding, and continuous batching. Every claim "
    "links back to the paper it came from."
)


def _arxiv_url(source: str | None) -> str:
    """Build an arXiv abstract URL from an ``arxiv:<id>`` source string.

    Args:
        source: A source id such as ``"arxiv:2605.22850"``.

    Returns:
        The ``https://arxiv.org/abs/<id>`` URL (best-effort if unprefixed).
    """
    if not source:
        return "https://arxiv.org"
    arxiv_id = source.split(":", 1)[1] if source.startswith("arxiv:") else source
    return f"https://arxiv.org/abs/{arxiv_id}"


def _dedupe_sources(results: list[dict]) -> list[dict]:
    """Drop duplicate sources (multiple chunks per paper), preserving order."""
    seen: set[str] = set()
    unique: list[dict] = []
    for record in results:
        source = record.get("source") or ""
        if source in seen:
            continue
        seen.add(source)
        unique.append(record)
    return unique


def _render_answer(answer: str) -> None:
    """Render the synthesized answer inside a card."""
    st.markdown('<div class="col-label">Synthesis</div>', unsafe_allow_html=True)
    body = html.escape(answer) if answer else "No answer was generated."
    st.markdown(f'<div class="card answer">{body}</div>', unsafe_allow_html=True)


def _render_sources(results: list[dict]) -> None:
    """Render retrieved sources as cards: rank, arXiv title, id, and link."""
    st.markdown('<div class="col-label">Sources</div>', unsafe_allow_html=True)
    sources = _dedupe_sources(results)
    if not sources:
        st.markdown('<p class="empty">No sources retrieved.</p>', unsafe_allow_html=True)
        return

    cards = []
    for index, record in enumerate(sources, start=1):
        title = html.escape(record.get("title") or "Untitled")
        source = html.escape(record.get("source") or "n/a")
        cards.append(
            f'<a class="src-card" href="{_arxiv_url(record.get("source"))}" '
            f'target="_blank" rel="noopener">'
            f'<span class="src-rank">{index:02d}</span>'
            f'<span class="src-title">{title}</span>'
            f'<span class="src-foot"><span class="src-id">{source}</span>'
            f'<span class="src-link">arxiv.org &rarr;</span></span></a>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def _request_answer(question: str, top_k: int) -> dict:
    """POST a question to the RAG API and return the parsed JSON response.

    Args:
        question: The user's question.
        top_k: Number of context passages to retrieve.

    Returns:
        The decoded ``/api/v1/answer`` response.

    Raises:
        httpx.HTTPError: On connection failure or a non-2xx response.
    """
    response = httpx.post(
        f"{API_URL}/api/v1/answer",
        json={"text": question, "top_k": top_k},
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
    )
    response.raise_for_status()
    return response.json()


def _render_output() -> None:
    """Render the current result, error, or the initial invitation."""
    error = st.session_state.get("error")
    result = st.session_state.get("result")

    if error:
        st.markdown(
            f'<div class="notice">Can\'t reach the backend at {html.escape(API_URL)}. '
            f"Start it, then search again.<br>{html.escape(error)}</div>",
            unsafe_allow_html=True,
        )
        return
    if not result:
        st.markdown(
            '<p class="empty">Ask a question to query the corpus.</p>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
    answer_col, sources_col = st.columns([60, 40], gap="large")
    with answer_col:
        _render_answer(result.get("answer", ""))
    with sources_col:
        _render_sources(result.get("results", []))


def _render_header() -> None:
    """Render the page header (eyebrow, title, subtitle)."""
    st.markdown(
        '<div class="eyebrow">arXiv &middot; LLM inference optimization</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="app-title">Ask the inference-optimization literature.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="app-sub">{_SUBTITLE}</p>', unsafe_allow_html=True)


def _render_search() -> tuple[bool, str, int]:
    """Render the search form and return (submitted, question, top_k)."""
    with st.form("ask"):
        question = st.text_input(
            "Question",
            placeholder="How does continuous batching improve GPU throughput?",
            label_visibility="collapsed",
        )
        field, action = st.columns([4, 1], gap="medium")
        top_k = field.number_input(
            "Passages", min_value=1, max_value=20, value=5, label_visibility="collapsed"
        )
        submitted = action.form_submit_button("Search")
    # Defensive clamp: guarantee a valid top_k even if widget state is corrupted.
    return submitted, question, max(1, min(int(top_k), 20))


def main() -> None:
    """Compose and render the single-page Streamlit application."""
    st.set_page_config(
        page_title="Inference Atlas",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    _render_header()
    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
    submitted, question, top_k = _render_search()

    if submitted and question.strip():
        with st.spinner("Retrieving and synthesizing"):
            try:
                st.session_state["result"] = _request_answer(question.strip(), top_k)
                st.session_state["error"] = None
            except httpx.HTTPError as exc:
                st.session_state["result"] = None
                st.session_state["error"] = str(exc)

    _render_output()


main()
