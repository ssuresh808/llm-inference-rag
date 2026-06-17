# Building a domain-isolated, agentic RAG system - and being honest about the numbers

*A build log for an LLM inference-optimization retrieval system: hybrid retrieval,
a LangGraph ReAct agent, local-first evaluation, and a clean path to the cloud.
Written for engineers who care more about trade-offs than demos.*

---

## The challenge: RAG over dense, narrow ML literature

General-purpose RAG demos retrieve over Wikipedia and call it a day. This system
does something harder: it answers questions over a **narrow, technical corpus** - 
recent arXiv work on LLM *inference optimization* (vLLM, quantization, KV-cache
management, continuous batching, speculative decoding, GPU serving economics).

That domain changes the engineering problem:

- **Vocabulary is dense and overloaded.** "KV-cache," "paging," "speculative" - 
  the discriminating signal is often an exact term or a number, not fuzzy
  topicality.
- **Questions are multi-hop.** "How does continuous batching interact with
  speculative decoding's latency?" needs evidence assembled from more than one
  passage.
- **Hallucination is expensive.** A confident, wrong answer about a serving
  optimization is worse than "the corpus doesn't cover this."

The corpus is built by streaming a 500-paper slice of an arXiv metadata snapshot,
filtering to the domain, and indexing ~1,477 chunks behind a quality gate
(minimum length, whitespace normalization) into an isolated Qdrant collection,
`llm_optimization_domain`.

## The architecture

```
Question → FastAPI → [single-shot RAG | LangGraph ReAct agent] → Qdrant (dense + BM25 hybrid) → LLM → answer + citations
```

**Hybrid dense/sparse retrieval.** `bge-large` dense embeddings capture semantic
similarity; BM25 (sparse) captures exact-term matches. Qdrant fuses them. A
cross-encoder reranker is available as an optional third stage.

**A LangGraph ReAct agent (opt-in).** Behind a `GENERATION_MODE` flag, the system
swaps the single-shot pipeline for an agent that *decides when to search*, can
issue multiple queries for multi-part questions, and **cites the passages it
used**. It also carries two capabilities worth calling out:

- **Dynamic Markdown skills.** A sandboxed `read_agent_skill` tool lets the agent
  load reusable analysis playbooks (e.g. `critical_analysis.md`) at runtime, so
  evaluative questions get a consistent reasoning scaffold without bloating the
  system prompt.
- **Checkpointing.** The graph compiles with a `MemorySaver` checkpointer and a
  thread id, making runs checkpoint-capable. (Honest scope: a fresh saver is
  created per request today, so this is within-run durability; true multi-turn
  memory needs a cached agent + stable thread - a deliberate next step, not a
  claim.)

**Docker orchestration.** A multi-stage `uv` Dockerfile and `docker-compose`
run the FastAPI backend and Streamlit frontend; the backend mounts the persistent
Qdrant volume and reaches host-native Ollama via `host.docker.internal`. This was
verified with a live containerized integration test, not just asserted.

## The empirical truth

Numbers a hiring manager could probe, with the caveats that make them defensible.

### Retrieval (rag-mini-wikipedia, 300 factoid QA, answer-recall@k)

| Config | Hit@5 | MRR | Recall@5 |
|---|---|---|---|
| Dense (`bge-large`) | 0.697 | 0.594 | 0.487 |
| **+ Hybrid (BM25)** | **0.717** | **0.639** | **0.496** |
| + Hybrid + Rerank | 0.690 | 0.520 | 0.470 |

**Hybrid beat dense** (+0.045 MRR). BM25's exact-term matching wins precisely on
the entity/number questions - dates, names, specific quantities - where dense
embeddings blur the discriminating token.

**Reranking *hurt* on this metric (0.520 vs 0.639 MRR) - and that result is the
interesting one.** The cross-encoder optimizes *semantic* relevance, but the
evaluation proxy here is *answer-string containment*. The reranker confidently
promotes passages that are on-topic but don't contain the answer string above
passages that do, which lowers answer-recall. This is an **objective/metric
mismatch, not a code defect**: judged by a relevance-aware metric (LLM-judge or
gold passage labels), the reranker would likely look very different. The lesson - 
*the metric you choose decides which technique "wins"* - is the kind of thing
worth surfacing rather than hiding.

### Generation (RAGAS, fully local judge: `qwen2.5:14b`)

| Metric | Score |
|---|---|
| faithfulness | 0.48 |
| answer_relevancy | 0.65 |

These are **real local-judge numbers**, run with no OpenAI dependency. The honest
caveat: a local 14B judge is unstable for RAGAS, whose metric prompts demand
strict JSON-schema adherence that small models satisfy unreliably (one question
scored 0.0 in isolation). So the evaluation loop is **NaN-tolerant** - a parse
failure is logged and recorded, never crashes the suite - and a hosted-judge
fallback (`RAGAS_JUDGE_PROVIDER`) exists to produce clean, defensible numbers
without making the *application* depend on a paid API.

## The pivot: deepagents → LangGraph ReAct

The agent was first built on `deepagents`. Unit tests (mocked) passed. Then a
**live smoke test** told the truth: driven by local `qwen2.5:14b`, the agent
**never called the search tool** and produced fluent, off-topic output - answering
from parametric memory instead of the corpus. That is the exact failure a RAG
system must not have.

A controlled test isolated the cause: the same model calls the same tool reliably
in a *lightweight* harness. deepagents' heavy built-in middleware (todo planning,
filesystem, sub-agents) overwhelms a 14B model's limited tool-calling budget. So I
migrated to LangGraph's `create_react_agent` - a lighter harness - and re-ran the
live test: the agent now searches, cites, and stays grounded (it even declines to
answer when the corpus lacks the specifics). The decision and its evidence are
recorded in **ADR-018**. The win that mattered most: this kept the system
**local-first and privacy-preserving** - no hosted model required for the agent to
behave.

The meta-point for senior readers: **mocked tests prove wiring; only live tests
prove behavior.** The framework that demos best is not always the one that works
on the model you can actually afford to run.

## The cloud strategy: unplug the local stack

The local stack is deliberately heavy - `bge-large` + a 14B model produce a ~9.5GB
image and need real memory. That's right for a free, private local demo and wrong
for a small serverless instance. The architecture was built so the heavy parts
**unplug cleanly**:

- **Providers are factories behind config.** `EMBEDDING_PROVIDER` and
  `LLM_PROVIDER` select the backend; cloud flips them to hosted APIs
  (`text-embedding-3-small`, a hosted chat model) so no model runs in-process.
- **Storage falls back by config.** When `QDRANT_CLOUD_URL` + `QDRANT_API_KEY` are
  present the engine connects to a managed Qdrant Cloud cluster; otherwise it uses
  the local persistent volume. One branch, no code path duplication.
- **Same image, different env.** A `render.yaml` Blueprint deploys the two Docker
  services to Render; secrets are injected at deploy time (`sync: false`). The
  only difference between laptop and cloud is environment variables.

Honest note: routing *inference* to hosted APIs shrinks the runtime footprint, but
the image still bundles the ML dependencies. A truly minimal cloud image would
split those into optional dependency groups - a clear, scoped next step.

## What this project is really about

Not the demo. The judgment: choosing hybrid retrieval for the right reason,
reading an adverse reranker result correctly instead of burying it, catching an
agent framework that failed on a live model, keeping evaluation honest when the
local judge is noisy, and designing the provider boundary so "free and local"
becomes "hosted and scalable" by configuration alone.

*Architecture decisions and their trade-offs are recorded as ADRs in the repo's
`docs/decisions.md`; retrieval and generation numbers (and how to reproduce them)
are in `docs/experiments.md`.*
