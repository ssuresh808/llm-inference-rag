# Roadmap & Goals

_Living document — reflects current reality, not the original plan._

## Goal

A production-minded RAG system that demonstrates end-to-end ML-engineering
judgment — ingestion, retrieval quality, **measured evaluation**, generation,
and deployment — strong enough to anchor a job-search portfolio. Primary signal:
**defensible retrieval/generation numbers**, not a flashy demo.

## Current status

**Phase 1 — Foundation: ✅ complete**
- uv project (Python 3.13), Ruff/Black/pytest, pydantic-settings config.
- Ingestion: Markdown/PDF + HF datasets, chunking, quality gate.
- Retrieval: Qdrant + `bge-large` embeddings on Apple MPS; provider-swappable.
- Generation: local Ollama `qwen2.5:14b`; grounded, source-cited answers.
- API: FastAPI `/health`, `/api/v1/query`, `/api/v1/answer`.
- 46 tests (offline), CLIs, ADRs, architecture + mindmap diagrams.

**Phase 2 — Quality: 🚧 in progress**
- ✅ Evaluation harness (Hit@k / MRR / Recall@k).
- ✅ Real benchmark: rag-mini-wikipedia + answer-containment gold set.
- ✅ Hybrid (BM25) retrieval + cross-encoder reranking — **measured**
  ([experiments.md](experiments.md)): hybrid +0.045 MRR; reranking −0.119 MRR
  (metric mismatch, documented).
- ⬜ Swap to an on-domain corpus (recent LLM-optimization papers).
- ⬜ RAGAS generation metrics (faithfulness, answer/context relevancy) — also the
  fair way to re-judge reranking.
- ⬜ Experiment tracking (Weights & Biases).

**Phase 3 — Ship: ⬜ not started**
- Frontend (chat UI), Docker, CI (GitHub Actions: ruff + pytest), deployment to a
  public URL, README polish, demo video, blog post.

## Immediate next steps (priority order)

1. **On-domain corpus (Phase 2b).** Replace the off-domain mini-wiki *demo*
   corpus with recent LLM-optimization papers via `Cornell-University/arxiv`
   (filter `cs.CL`/`cs.AI` + `update_date >= 2023`) or `cosmopedia`; use modern
   seed vocabulary (vLLM, PagedAttention, KV-cache, speculative decoding,
   FlashAttention). Keeps mini-wiki as the eval benchmark.
2. **RAGAS generation metrics** (LLM-judge via Ollama) — defensible answer-quality
   numbers + a fair reranking re-evaluation.
3. **W&B** experiment tracking for the ablations.
4. **Phase 3 deploy.** See open decision below.
5. **Publish to GitHub** (currently local-only — see below).

## Open decisions

- **Deployment vs the free-local stack.** The fully-local stack (14B Ollama +
  `bge-large`) won't fit a free serverless host. For a public demo we either (a)
  flip the provider config to hosted embeddings/LLM (the swappability is built
  for exactly this), or (b) deploy retrieval-only. Decide at Phase 3.
- **GitHub publish.** No remote exists yet; nothing is pushed. When ready, the
  repo can be created and pushed; `CLAUDE.md`/session-log/`.claude` stay private
  via `.git/info/exclude`.

## Definition of done

1. Deployed demo with a public URL.
2. Clean README that converts in ~90 seconds.
3. Evaluation suite with real, defensible numbers. _(partially met — Phase 2)_
4. A blog post explaining the build and key decisions.
5. A short demo video.
