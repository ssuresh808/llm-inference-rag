# Architecture Decision Records (ADRs)

Decisions for the RAG Portfolio project. Each record states the choice, the
alternatives considered, the trade-off, and the consequence. The goal is that
every choice is defensible in an interview.

> Status legend: **Accepted** = locked for the current phase · **Provisional**
> = default chosen but unconfirmed · **Deferred** = decided when its phase
> arrives.

---

## ADR-000: Domain / corpus

- **Status:** Accepted
- **Decision:** The RAG system retrieves over a curated corpus of **LLM
  inference-optimization** material — vLLM, TensorRT-LLM, quantization
  (GPTQ/AWQ/FP8), KV-cache management, continuous batching, speculative
  decoding, and GPU serving cost/latency trade-offs (~20–50 high-signal public
  docs).
- **Why:** It is the ML-systems layer that infra teams at FAANG / AI labs live
  in. A retrieval system that is itself an expert on making LLMs cheaper and
  faster signals understanding *beneath* the model layer, not just prompting.
  It is also self-consistent with the project framing ("LLM Cost, Latency & GPU
  Optimization").
- **Alternatives:** (a) arXiv ML papers — easy to source but generic and
  over-used in portfolios; (b) personal resume/projects — memorable to
  recruiters but a corpus too small to support a credible retrieval/eval story.
- **Consequence:** Corpus is technical, English, text-first (no OCR/licensing
  mess). Eval questions are concrete and verifiable, which strengthens the
  Phase 2 RAGAS numbers.

---

## ADR-001: RAG framework — LangChain over LlamaIndex

- **Status:** Accepted
- **Decision:** Standardize the whole project on **LangChain**.
- **Why:** The end-goal includes an agentic layer (`deepagents`), which is
  LangChain-native. Choosing LangChain now keeps ingestion → retrieval → LCEL →
  agents in a single ecosystem instead of bolting two frameworks together
  later. LangChain also has first-party integrations for every other component
  in this stack (`langchain-qdrant`, `langchain-openai`).
- **Alternatives:** LlamaIndex has cleaner, more opinionated pure-retrieval
  primitives (nodes, `SemanticSplitterNodeParser`) and was the originally
  proposed stack. The deciding factor was ecosystem coherence with the agentic
  end-goal, not Phase-1 ergonomics (where the two are roughly equivalent).
- **Trade-off:** LangChain's abstractions are heavier and its API moves faster;
  we pin versions and keep our own thin wrappers around it to limit churn.
- **Consequence:** The original LlamaIndex-based ingestion/retrieval steps are
  re-expressed with `langchain-text-splitters`, `langchain-qdrant`, and
  `langchain-openai`.

---

## ADR-002: Vector DB — Qdrant (local, in-memory for Phase 1)

- **Status:** Accepted
- **Decision:** **Qdrant** in `:memory:` / local mode for Phase 1; same engine
  runs as a container/cloud in later phases (no rewrite).
- **Why:** Free local dev with zero external dependency, production-capable,
  rich filtering + hybrid search (needed in Phase 2), and an official
  `langchain-qdrant` integration.
- **Alternatives:** Pinecone (managed, but cloud-only + paid + account setup
  friction for Phase 1); pgvector (fine, but weaker hybrid/rerank story for a
  retrieval-focused portfolio).
- **Consequence:** Phase 1 runs entirely offline. The Phase 2 hybrid-search
  work stays on the same engine.

---

## ADR-003: Embeddings — local HuggingFace default, provider-swappable

- **Status:** Accepted
- **Decision:** Default to **local HuggingFace / sentence-transformers**
  embeddings (`BAAI/bge-small-en-v1.5`, 384-dim) via `langchain-huggingface`.
  The backend is selected by one config value (`EMBEDDING_PROVIDER`) behind a
  factory, so it can be swapped to a paid/hosted provider later with **no code
  change** — just an env var (plus a re-index; see consequence).
- **Why:** The requirement is "free, no cost, no API key, swappable later."
  Local sentence-transformers is genuinely free, needs no key or quota, runs on
  CPU/Mac, and works offline (great for tests + reproducibility). bge-small is
  strong for its size and fast to iterate on.
- **Why not NVIDIA NeMo Retriever as the default (the earlier pick):** It is not
  actually free — the hosted NVIDIA API catalog gives limited free *credits*
  then requires payment + a key, and self-hosted NIM needs an NVIDIA GPU (a Mac
  has none). So NeMo Retriever becomes the **headline "when funded" upgrade and
  a Phase-2 ablation contender**, wired as a drop-in provider.
- **Swap path (when funded):** set `EMBEDDING_PROVIDER=nvidia` (+ `NVIDIA_API_KEY`)
  or `=openai` (+ `OPENAI_API_KEY`). The factory returns the matching LangChain
  `Embeddings` object; nothing else changes.
- **Consequence:** Providers emit different vector dimensions (bge-small 384,
  bge-base 768, OpenAI 3-small 1536, NeMo ~1024). **Switching providers requires
  re-embedding the corpus into a fresh Qdrant collection** — embeddings are not
  hot-swappable. Step 3 adds `langchain-huggingface` + `sentence-transformers`
  (pulls torch — a large install). Tests mock the embedding call, so none of
  this affects the suite.

---

## ADR-004: Chunking — RecursiveCharacterTextSplitter for v1

- **Status:** Accepted
- **Decision:** **`RecursiveCharacterTextSplitter`** for the v1 pipeline.
- **Why:** Deterministic, free, and trivially unit-testable. Ship a correct,
  measurable v1 before optimizing.
- **Alternatives:** `SemanticChunker` produces better boundaries but makes
  **real embedding API calls at chunk time** (cost + nondeterminism +
  test-mocking burden). It is deferred to the Phase 2 ablation, where its
  retrieval quality can be measured against the deterministic baseline.
- **Consequence:** Plus a "Quality Gate" (reject < 50 chars, strip excessive
  whitespace, log processed-vs-rejected) implemented in Step 2.

---

## ADR-005: Python 3.13, managed by uv

- **Status:** Accepted
- **Decision:** Pin **Python 3.13** via uv; `requires-python = ">=3.11,<3.14"`.
- **Why:** 3.14 is too new for reliable ML wheel availability; 3.13 is current
  and already present locally. uv gives fast, deterministic, lockfile-backed
  installs.
- **Consequence:** `uv.lock` is committed for reproducibility.

---

## Deferred (decided when their phase arrives)

- **ADR-006 LLM / generation:** Implemented (`src/generation/`). Default to
  **local Ollama** (free, no key — e.g. Llama 3.2 / Qwen) via `build_llm()`,
  behind the same provider-factory pattern as embeddings, swappable to
  **Claude** or OpenAI by config. Requires a running Ollama server for real
  generation; `answer_question()` retrieves context then generates a grounded,
  source-cited answer.
- **ADR-007 Frontend:** Deferred to Phase 3 (Streamlit vs Next.js).
- **ADR-008 Deployment:** Deferred to Phase 3 (Modal vs Railway vs HF Spaces).
- **ADR-009 Evaluation:** RAGAS, set up in Phase 2.
- **ADR-010 Experiment tracking:** Weights & Biases, first run in Phase 2.

---

## ADR-011: Agentic RAG — deferred to Phase 2 as a measured ablation

- **Status:** Deferred — scoped now, built and evaluated in Phase 2, and
  promoted to the default path *only if* it beats the baseline on a metric.
- **Decision:** Phase 1 stays **plain single-shot RAG** (retrieve → generate).
  Introduce an **agentic retrieval loop** in Phase 2 as an *ablation arm*, not
  the default. Candidate pattern: **Corrective RAG (CRAG) / Self-RAG** — grade
  the retrieved chunks and, on low confidence, rewrite the query and re-retrieve
  (or abstain) instead of answering from weak context.
- **Why:** The corpus has a real failure mode single-shot RAG handles badly —
  multi-hop questions ("how does continuous batching interact with speculative
  decoding's latency?") and retrieved-but-irrelevant chunks. An agentic loop
  targets exactly that, and it realizes the agentic end-goal already committed
  to in ADR-001 (`deepagents`), giving a coherent retrieval → agentic story.
- **Why not now:** Phase 1's job is a correct, measurable baseline. You cannot
  show an agent *helps* without a baseline + eval set to beat, and adding the
  loop now would confound the Phase-2 embedding × chunk ablation. (Project rule:
  ship working v1 before optimizing.)
- **Substrate:** Built on **LangGraph** — the state-machine layer `deepagents`
  itself sits on — so it stays inside the LangChain ecosystem from ADR-001 with
  no new framework decision.
- **Trade-off (the interview answer):** Agentic loops multiply LLM/embedding
  calls per query. On the ADR-006 default (local Ollama) that cost is mainly
  **latency/throughput**; on a paid provider it is **latency + dollars**. Either
  way it adds failure modes (retry loops, latency spikes) that hurt a live demo
  — a reliable 1.5 s single-shot answer beats a flaky 8 s agentic one. The win
  must be *earned*, not assumed.
- **Success criterion:** Measured against the Phase-1 baseline on the RAGAS
  suite (ADR-009). Promote only on a defensible lift on multi-hop / low-context
  queries (e.g. faithfulness ↑, fewer ungrounded answers), reported with the
  latency/cost delta stated. If there is no lift, "agentic RAG didn't beat the
  baseline here, and here's why" is itself an honest, strong portfolio finding.
- **Consequence:** Zero Phase-1 code or dependency impact. Phase 2 adds
  `langgraph` + a doc-grading step whose generation calls run through the
  ADR-006 provider factory. Logged now so the deferral is deliberate and
  defensible, not an omission.

**Rejected for this project — NVIDIA NemoClaw:** an enterprise stack for running
always-on agents sandboxed in NVIDIA OpenShell. It is a *host-level
agent-sandboxing runtime*, not a RAG/retrieval or embeddings component, so it
adds nothing to a solo retrieval portfolio. Recorded here to close the loop:
NemoClaw ≠ NeMo Retriever (ADR-003) — different product, similar "Nemo" naming.
