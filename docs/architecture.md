# Architecture

The system has two execution paths — an **indexing** path (offline, one-time per
corpus) and a **query** path (per request) — plus an **evaluation** harness that
measures retrieval quality against a gold set. Embeddings and the LLM are chosen
by configuration behind factories, so providers are swappable without code
changes.

## End-to-end workflow

```mermaid
flowchart TD
    subgraph INGEST["Ingestion — offline, one-time"]
        A["Corpus: Markdown / PDF / HF datasets"] --> B["load_documents()"]
        B --> C["RecursiveCharacterTextSplitter"]
        C --> D{"Quality gate: cleaned length >= 50?"}
        D -->|"reject"| X["Dropped (logged: processed / accepted / rejected)"]
        D -->|"accept"| E["Clean chunks"]
    end

    subgraph INDEX["Indexing"]
        E --> F["Embeddings factory (EMBEDDING_PROVIDER)"]
        F -->|"HuggingFace bge-large on MPS — default, free"| G[("Qdrant vector store")]
        F -.->|"swap via env: NVIDIA / OpenAI"| G
    end

    subgraph QUERY["Query time — per request"]
        Q["User question"] --> H["FastAPI: POST /api/v1/answer"]
        H --> I["Embed query + similarity search (top-k)"]
        G --> I
        I --> J["Assemble cited context"]
        J --> K["LLM factory (LLM_PROVIDER)"]
        K -->|"Ollama llama3.2 — default, free"| L["Grounded answer + sources"]
        K -.->|"swap via env: Claude / OpenAI"| L
    end

    subgraph EVAL["Evaluation harness"]
        GOLD["Gold question set (data/eval)"] --> EV["evaluate_retrieval()"]
        G --> EV
        EV --> M["Hit@k / MRR / Recall@k"]
    end
```

## Request lifecycle: `POST /api/v1/answer`

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI (/api/v1/answer)
    participant Eng as RetrievalEngine
    participant Emb as Embeddings (bge-large, MPS)
    participant DB as Qdrant
    participant LLM as LLM (Ollama llama3.2)

    User->>API: POST { text, top_k }
    API->>Eng: query(text, top_k)
    Eng->>Emb: embed_query(text)
    Emb-->>Eng: query vector
    Eng->>DB: similarity_search(vector, k)
    DB-->>Eng: top-k chunks
    Eng-->>API: chunks
    API->>LLM: prompt(context + question)
    LLM-->>API: grounded answer
    API-->>User: { answer, sources, results }
```

## Components

| Module | Responsibility |
|---|---|
| `src/config/settings.py` | Pydantic settings: provider selection, model ids, API keys, Qdrant + corpus config (from `.env`). |
| `src/ingestion/loader.py` | Load Markdown/PDF, recursive chunking, quality gate with logging. |
| `src/retrieval/embeddings.py` | Embeddings provider factory (HuggingFace / NVIDIA / OpenAI), lazy imports. |
| `src/retrieval/engine.py` | `RetrievalEngine` over Qdrant: `index()` and `query()`; `build_engine()` wires config. |
| `src/generation/llm.py` | LLM provider factory (Ollama / Anthropic / OpenAI), lazy imports. |
| `src/generation/rag.py` | `answer_question()`: retrieve context, then generate a grounded, source-cited answer. |
| `src/api/main.py` | FastAPI app: `/health`, `/api/v1/query`, `/api/v1/answer`; lazy engine/LLM; corpus auto-indexed on first use. |
| `src/evaluation/` | Deterministic metrics (Hit@k, MRR, Recall@k) + `evaluate_retrieval()` harness. |
| `src/pipeline.py` | Retrieval CLI tying ingestion → indexing → query. |

## Provider swappability

Both embeddings and the LLM are selected by a single environment variable and
built behind a factory. The default stack is fully local and free; switching to a
hosted provider is a config change, not a code change.

| Concern | Env var | Default (free) | Swap options |
|---|---|---|---|
| Embeddings | `EMBEDDING_PROVIDER` | `huggingface` (`bge-large`, MPS) | `nvidia`, `openai` |
| LLM | `LLM_PROVIDER` | `ollama` (`llama3.2`) | `anthropic`, `openai` |

> Note: embedding providers emit different vector dimensions, so switching the
> embedding provider requires re-indexing the corpus into a fresh collection.
