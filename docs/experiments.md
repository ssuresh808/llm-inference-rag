# Experiments

## Retrieval ablation - rag-mini-wikipedia (Phase 2)

**Setup:** 3,200 Wikipedia passages indexed; 300 factoid QA questions (yes/no and
too-short answers excluded). Relevance = answer-string containment
(answer-recall@k). Embeddings: `BAAI/bge-large-en-v1.5` on Apple MPS. Sparse:
BM25 (`Qdrant/bm25` via fastembed). Reranker: `BAAI/bge-reranker-base`
cross-encoder, fetch 20 → top 5.

| Config | Hit@5 | MRR | Recall@5 |
|---|---|---|---|
| Dense (bge-large) | 0.697 | 0.594 | 0.487 |
| **+ Hybrid (BM25)** | **0.717** | **0.639** | **0.496** |
| + Hybrid + Rerank | 0.690 | 0.520 | 0.470 |

### Findings

- **Hybrid helps** (+0.045 MRR, +0.020 Hit@5 over dense). BM25's exact-term
  matching catches entity/number questions (dates, names) that dense embeddings
  blur - precisely where lexical signal matters.
- **Reranking hurt** on this metric (−0.119 MRR vs hybrid). The cross-encoder
  optimizes *semantic* relevance, but the proxy metric rewards *answer-string
  presence*. The reranker promotes topically-relevant-but-answerless passages
  above answer-containing ones, lowering answer-recall. This is an
  objective/metric mismatch, not a code defect.

### Takeaways

- Ship **hybrid** as the retrieval default for real corpora.
- Reranking needs a relevance-aware metric to be judged fairly (LLM-judge /
  RAGAS faithfulness, or gold passage labels). Answer-containment under-credits
  it. Tracked for the RAGAS work (ADR-009).

> Reproduce: hybrid/rerank are config-gated (`HYBRID`, `RERANK`). Numbers are
> deterministic given the same corpus, gold sample, and models.

## Generation eval - RAGAS, local judge (Phase 2c)

**Setup:** on-domain arXiv corpus (`llm_optimization_domain`), 3 sample
questions. Generator = `qwen2.5:14b` (Ollama); judge = `qwen2.5:14b` (Ollama);
embeddings = `bge-large`. Fully local, no OpenAI.

| Metric | Score |
|---|---|
| faithfulness | 0.48 |
| answer_relevancy | 0.65 |

**Notes:**
- Real local-judge numbers. `answer_relevancy` is unstable with a local 14B judge
  (one question scored 0.0 in isolation): RAGAS's metric prompts demand strict
  JSON adherence that local models satisfy unreliably.
- The loop is **NaN-tolerant**, and a hosted-judge fallback
  (`RAGAS_JUDGE_PROVIDER`) exists for cleaner, defensible numbers without making
  the app depend on a paid API (ADR-016).
