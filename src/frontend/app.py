"""Streamlit frontend for the RAG API — a stark research instrument.

Connects to the FastAPI ``/api/v1/answer`` endpoint and renders the grounded
answer beside a citation ledger (arXiv title + id + hyperlink) so the retrieval
provenance is the visible proof of work. Run with::

    streamlit run src/frontend/app.py

Set ``RAG_API_URL`` to point at the backend (default ``http://localhost:8000``;
``http://backend:8000`` under docker-compose).
"""

import html
import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180.0

_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --paper:#FCFCFB; --ink:#121212; --muted:#6F6F6F; --line:#E4E3DE; --accent:#2438D9;
}
.stApp { background: var(--paper); }
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 4rem; }

.eyebrow {
  font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted);
}
.hero-title {
  font-family:'Newsreader',serif; font-weight:500; font-size:clamp(34px,5vw,54px);
  line-height:1.03; letter-spacing:-.01em; color:var(--ink); margin:.5rem 0 .6rem;
}
.hero-sub {
  font-family:'Inter',sans-serif; font-size:16px; line-height:1.55;
  color:var(--muted); max-width:50ch;
}
hr.rule { border:0; border-top:1px solid var(--line); margin:1.7rem 0; }

.col-label {
  font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); margin-bottom:.7rem;
}
.answer {
  font-family:'Newsreader',serif; font-size:18px; line-height:1.62;
  color:var(--ink); white-space:pre-wrap;
}

.ledger { border-top:1px solid var(--ink); }
.src { display:block; padding:.85rem 0; border-bottom:1px solid var(--line); text-decoration:none; }
.src .idx { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--accent); }
.src .ttl {
  font-family:'Inter',sans-serif; font-size:14px; font-weight:500; color:var(--ink);
  line-height:1.35; display:block; margin:.15rem 0 .25rem;
}
.src .meta { font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--muted); }
.src:hover .ttl { color:var(--accent); }

.notice {
  font-family:'JetBrains Mono',monospace; font-size:13px; line-height:1.5; color:#8a1f1f;
  border:1px solid #e7c9c9; background:#fbf3f3; padding:.85rem 1rem;
}

.stTextInput input {
  font-family:'Inter',sans-serif; font-size:16px; border-radius:0;
  border:1px solid var(--ink); padding:.7rem .8rem; background:#fff;
}
.stTextInput input:focus { box-shadow:none; border-color:var(--accent); }
.stFormSubmitButton button {
  border-radius:0; border:1px solid var(--ink); background:var(--ink); color:var(--paper);
  font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.1em;
  text-transform:uppercase; padding:.65rem 1.6rem;
}
.stFormSubmitButton button:hover { background:var(--accent); border-color:var(--accent); color:#fff; }
</style>
"""

_HERO_SUB = (
    '<p class="hero-sub">Grounded answers over a curated corpus of arXiv work on '
    "vLLM, quantization, KV-cache, continuous batching, and speculative decoding "
    "— every claim traced back to its source.</p>"
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
    """Render the synthesized answer with reading-optimized typography."""
    st.markdown(f'<div class="answer">{html.escape(answer)}</div>', unsafe_allow_html=True)


def _render_sources(results: list[dict]) -> None:
    """Render the citation ledger: numbered arXiv title + id + hyperlink."""
    sources = _dedupe_sources(results)
    if not sources:
        st.markdown(
            '<p class="hero-sub">No sources were retrieved for this query.</p>',
            unsafe_allow_html=True,
        )
        return

    rows = []
    for index, record in enumerate(sources, start=1):
        title = html.escape(record.get("title") or "Untitled")
        source = html.escape(record.get("source") or "—")
        rows.append(
            f'<a class="src" href="{_arxiv_url(record.get("source"))}" '
            f'target="_blank" rel="noopener">'
            f'<span class="idx">[{index:02d}]</span>'
            f'<span class="ttl">{title}</span>'
            f'<span class="meta">{source} &nbsp;&rarr;&nbsp; arxiv.org</span></a>'
        )
    st.markdown(f'<div class="ledger">{"".join(rows)}</div>', unsafe_allow_html=True)


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
            f'<div class="notice">Cannot reach the API at {html.escape(API_URL)}. '
            f"Start the backend, then search again.<br>{html.escape(error)}</div>",
            unsafe_allow_html=True,
        )
        return
    if not result:
        st.markdown(
            '<p class="hero-sub">Enter a question to query the corpus.</p>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)
    answer_col, sources_col = st.columns([62, 38], gap="large")
    with answer_col:
        st.markdown('<div class="col-label">Synthesis</div>', unsafe_allow_html=True)
        _render_answer(result.get("answer", ""))
    with sources_col:
        st.markdown('<div class="col-label">Sources</div>', unsafe_allow_html=True)
        _render_sources(result.get("results", []))


def main() -> None:
    """Compose and render the single-page Streamlit application."""
    st.set_page_config(
        page_title="Inference Atlas",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_STYLES, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">arXiv · LLM inference optimization</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Ask the inference literature.</h1>', unsafe_allow_html=True)
    st.markdown(_HERO_SUB, unsafe_allow_html=True)
    st.markdown('<hr class="rule"/>', unsafe_allow_html=True)

    with st.form("ask"):
        question = st.text_input(
            "Question",
            placeholder="How does continuous batching improve GPU throughput?",
            label_visibility="collapsed",
        )
        field, action = st.columns([4, 1])
        top_k = field.number_input(
            "Passages", min_value=1, max_value=20, value=5, label_visibility="collapsed"
        )
        submitted = action.form_submit_button("Search")

    if submitted and question.strip():
        with st.spinner("Searching the corpus"):
            try:
                st.session_state["result"] = _request_answer(question.strip(), int(top_k))
                st.session_state["error"] = None
            except httpx.HTTPError as exc:
                st.session_state["result"] = None
                st.session_state["error"] = str(exc)

    _render_output()


main()
