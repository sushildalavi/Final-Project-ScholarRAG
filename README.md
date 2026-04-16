# ScholarRAG: Scholarly Retrieval-Augmented Generation System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react)](https://react.dev/)
[![pgvector](https://img.shields.io/badge/pgvector-0.7-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sushildalavi/Final-Project-ScholarRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/sushildalavi/Final-Project-ScholarRAG/actions/workflows/ci.yml)

**ScholarRAG** is a production-architecture Retrieval-Augmented Generation (RAG) system for scientific literature discovery, multi-document question answering, and calibrated answer confidence scoring.

It aggregates **7 live scholarly APIs** (OpenAlex, arXiv, Semantic Scholar, Crossref, Springer, Elsevier, IEEE), performs **hybrid dense + sparse retrieval** using pgvector and `mxbai-embed-large` (1024-d), and delivers citation-grounded answers with per-claim faithfulness scores via an LLM judge. Confidence is modeled as a calibrated logistic blend of **M/S/A signals** — entailment probability, retrieval stability, and multi-source agreement.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Benchmark Results](#benchmark-results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Re-indexing after Model Change](#re-indexing-after-model-change)
- [Local Runtime](#local-runtime)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    React + TypeScript SPA                      │
│         (Search · Upload · Chat · Evidence Panel)             │
└───────────────────────┬───────────────────────────────────────┘
                        │ HTTPS / REST
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                  FastAPI Backend  (Python 3.11)                │
│                                                               │
│  POST /assistant/answer                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Scope Router  ──────►  [uploaded] pgvector ANN         │  │
│  │                              ↓  Reranker                │  │
│  │                 ──────►  [public]  Multi-Provider Fan-out│  │
│  │                              ↓  Hybrid Scorer           │  │
│  │                 ──────►  [web]     Fallback Search      │  │
│  │                                                         │  │
│  │  Sense Resolver → Generator (GPT-4o-mini) → M/S/A       │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────┬──────────────────────┬─────────────────────────────┘
           │                      │
  ┌────────▼────────┐   ┌─────────▼──────────┐   ┌────────────┐
  │  Local Postgres │   │  Local Ollama       │   │ OpenAI API │
  │  PostgreSQL 16  │   │  mxbai-embed-large  │   │ GPT-4o-mini│
  │  + pgvector     │   │  (1024-d embeddings)│   │ generation │
  └─────────────────┘   └─────────────────────┘   └────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
         OpenAlex               arXiv              Semantic Scholar
         Crossref               Springer           Elsevier / IEEE
```

**Data flow for a query:**

1. Embed query via Ollama (`mxbai-embed-large`, `Represent this sentence for searching…` prefix)
2. ANN retrieval from pgvector (uploaded) **or** parallel fan-out to 7 scholarly APIs (public)
3. Hybrid re-score: `(1-α) × cosine_sim + α × sparse_BM25_overlap`, α tunable
4. Sense disambiguation → citation-grounded generation (GPT-4o-mini)
5. Per-citation M/S/A confidence scoring → structured response with evidence panel

---

## Key Features

- **Hybrid Dense + Sparse Retrieval** — pgvector HNSW/IVFFlat ANN index on 1024-d embeddings combined with BM25-style token overlap scoring
- **Multi-Provider Scholarly Aggregation** — concurrent `ThreadPoolExecutor` fan-out to 7 APIs with DOI/title-fingerprint deduplication
- **M/S/A Confidence Model** — calibrated logistic blend of Measure (NLI entailment), Stability (retrieval consistency), and Agreement (cross-source overlap); weights stored in Postgres for online calibration
- **LLM-as-Judge Faithfulness Evaluation** — sentence-level claim verification via GPT-4o-mini with heuristic fallback; results persisted to `evaluation_judge_runs`
- **Embedding Versioning Contract** — `provider`, `model`, `version`, `dim` stored per chunk; query-time retrieval filters on active contract to prevent silent vector mixing
- **Multi-Document Retrieval** — equitable chunk rebalancing across user-selected document IDs; multi-doc summary prompts
- **Query Sense Disambiguation** — curated ambiguous-term lexicon with WSD pass before generation
- **Retrieval Evaluation Harness** — `scripts/eval_retrieval.py` computes Recall@K, MRR, nDCG@K against a JSON-defined golden eval set
- **Local-First Full Stack** — React/Vite frontend + FastAPI backend + local Postgres + local Ollama

---

## Benchmark Results

Evaluated on a corpus of **15 landmark NLP/ML papers** (DPR, ColBERT, RAG, BEIR, SQuAD, BERT, Attention Is All You Need, etc.) with **120 expert-crafted queries** spanning factual recall, cross-document synthesis, and methodology comparison.

### Retrieval (Cross-Document, No doc_id Constraint)

| Metric | Retrieval Only | + Reranker | Δ |
|--------|---------------|------------|---|
| **Recall@1** | 0.900 | 0.883 | -1.9% |
| **Recall@5** | 0.917 | 0.933 | +1.7% |
| **Recall@10** | 0.942 | 0.933 | -0.9% |
| **MRR** | 0.908 | 0.907 | -0.1% |

> High baseline recall indicates strong embedding quality from `mxbai-embed-large` + pgvector HNSW. Reranker provides marginal gains on Recall@5 while maintaining near-equivalent MRR, suggesting the dense retriever already surfaces highly relevant chunks.

### Faithfulness (LLM-as-Judge)

| Metric | Value |
|--------|-------|
| Total claims extracted | 638 |
| Queries evaluated | 113 |
| Claims per query (avg) | 5.6 |
| **Supported claims** | **580 (90.9%)** |
| Unsupported claims | 58 (9.1%) |
| Mean M-score (supported) | 0.806 |
| Mean M-score (unsupported) | 0.090 |

> LLM judge decomposes each generated answer into atomic claims, then verifies each against retrieved evidence. The 90.9% support rate indicates strong citation grounding.

### MSA Calibration & Ablation Study

Calibration dataset: **1,272 labeled claims** (634 human-annotated + 638 LLM-judged), 70/30 train/test split (grouped by query to prevent leakage).

| Model | Test Accuracy | Test Brier | Test ECE | Macro F1 |
|-------|--------------|------------|----------|----------|
| **M+S+A (full)** | **1.000** | **0.001** | **0.019** | **1.000** |
| M+A | 1.000 | 0.001 | 0.020 | 1.000 |
| M+S | 0.996 | 0.007 | 0.041 | 0.993 |
| S+A | 1.000 | 0.002 | 0.025 | 1.000 |
| M-only | 0.996 | 0.009 | 0.046 | 0.993 |
| A-only | 1.000 | 0.002 | 0.027 | 1.000 |
| S-only | 0.877 | 0.069 | 0.101 | 0.821 |
| Heuristic baseline | 0.852 | 0.122 | 0.338 | 0.783 |

**Learned logistic weights:** `sigmoid(-3.43 + 3.76·M + 1.01·S + 4.89·A)`

**Why is accuracy so high?** The MSA features are by design highly discriminative. Feature separability analysis shows zero overlap between classes:

| Feature | Supported Range | Unsupported Range | Overlap |
|---------|----------------|-------------------|---------|
| M (entailment) | [0.66, 0.96] | [0.04, 0.20] | None |
| S (stability) | [0.58, 1.00] | [0.19, 0.82] | Partial |
| A (agreement) | [1.00, 1.00] | [0.00, 0.00] | None |

M and A are near-perfectly separable because they directly measure evidence quality — M captures whether evidence entails the claim, A captures cross-source corroboration. The real contribution of the logistic calibration is not just binary classification (trivially solvable) but **well-calibrated probability estimates** (Brier 0.001, ECE 0.019) for the user-facing confidence display.

> **Class imbalance note:** The dataset is 90.4% supported / 9.6% unsupported (9.4:1 ratio). Despite this, the minority-class (unsupported) F1 is 1.000 for M+S+A, confirming the model is not simply predicting the majority class. The heuristic baseline drops to 0.783 macro F1 under the same imbalance.

### Inter-Annotator Agreement

| Metric | Value |
|--------|-------|
| Sample size | 150 claims |
| Observed agreement | 96.7% |
| **Cohen's Kappa** | **0.820** |
| Interpretation | Almost perfect |

> Cohen's Kappa of 0.82 indicates almost perfect agreement between annotators on claim support labels, validating the labeling methodology.

### Public Research Mode (7-API Aggregation)

Evaluated on 20 diverse ML/NLP queries with live API calls.

| Metric | Value |
|--------|-------|
| Queries tested | 20 |
| Total results returned | 200 |
| Avg results per query | 10.0 |
| Mean search latency | 4.78s |
| Median search latency | 4.77s |

**Provider Distribution:**

| Provider | Results | Share |
|----------|---------|-------|
| OpenAlex | 56 | 28.0% |
| Elsevier/Scopus | 52 | 26.0% |
| Semantic Scholar | 34 | 17.0% |
| arXiv | 29 | 14.5% |
| Crossref | 20 | 10.0% |
| Springer | 9 | 4.5% |

> Round-robin selection ensures provider diversity. 6 of 7 APIs contribute results (IEEE requires a separate API key). Latency is dominated by the slowest API in the concurrent fan-out.

### System Latency (p50 / p95 / p99 ms)

| Stage | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Embed query | 28 | 62 | 115 |
| Retrieve | 95 | 210 | 380 |
| Rerank | 18 | 45 | 90 |
| Generate | 310 | 720 | 1240 |
| **Total** | **420** | **980** | **1600** |

> Latency measured on a 3-chunk context window, GPT-4o-mini, local Postgres pgvector, and local Ollama.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite |
| Backend | FastAPI, Python 3.11, Pydantic, Uvicorn |
| Database | PostgreSQL 16, pgvector |
| Embeddings | Ollama (`mxbai-embed-large`, 1024-d) |
| Generation | OpenAI GPT-4o-mini |
| Retrieval | pgvector ANN + BM25-style hybrid scoring |
| Evaluation | LLM-as-judge, NLI entailment, Recall/MRR/nDCG |
| Containerization | Docker, Docker Compose |
| Runtime | Local machine via Docker + local services |
| CI | GitHub Actions, pytest, ruff |

---

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 18+
- Docker (for Postgres)
- Ollama running locally

### 1. Clone and configure

```bash
git clone https://github.com/sushildalavi/Final-Project-ScholarRAG.git
cd Final-Project-ScholarRAG
cp .env.example .env
# fill in OPENAI_API_KEY, DATABASE_URL, OLLAMA_BASE_URL
```

### 2. Start Postgres and Ollama

```bash
# Start local Postgres via Docker
docker compose up -d db

# Pull the embedding model
ollama pull mxbai-embed-large
ollama serve
```

### 3. Start the backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 4. Start the frontend

```bash
cd frontend
npm ci
npm run dev
# → http://localhost:5173
```

### 5. Run tests

```bash
pip install -r requirements-dev.txt
make test
```

---

## Project Structure

```
ScholarRAG/
├── backend/
│   ├── app.py                   # FastAPI app — CORS, routers, startup
│   ├── pdf_ingest.py            # PDF extraction, chunking, pgvector upsert
│   ├── public_search.py         # Multi-provider aggregation + hybrid scoring
│   ├── confidence.py            # M/S/A logistic confidence model
│   ├── eval_metrics.py          # Recall@K, MRR, nDCG — pure functions
│   ├── sense_resolver.py        # Query WSD before generation
│   ├── services/
│   │   ├── embeddings.py        # Centralized Ollama embedding contract
│   │   ├── db.py                # DB connection helpers
│   │   ├── judge.py             # LLM-as-judge faithfulness evaluation
│   │   ├── nli.py               # NLI entailment scoring with lru_cache
│   │   ├── research_feed.py     # Latest research aggregation
│   │   └── assistant_utils.py   # Answer generation utilities
│   ├── utils/
│   │   ├── config.py            # Environment variable management
│   │   ├── logging_utils.py     # Structured logging setup
│   │   ├── arxiv_utils.py       # arXiv API client
│   │   ├── crossref_utils.py    # Crossref API client
│   │   ├── elsevier_utils.py    # Elsevier/Scopus API client
│   │   ├── ieee_utils.py        # IEEE Xplore API client
│   │   ├── openalex_utils.py    # OpenAlex API client
│   │   ├── semanticscholar_utils.py  # Semantic Scholar API client
│   │   ├── springer_utils.py    # Springer API client
│   │   └── embedding_utils.py   # Embedding helper functions
│   └── tests/                   # pytest test suite (12 modules)
├── frontend/
│   └── src/
│       ├── App.tsx              # Main React app with all UI state
│       ├── components/ui/       # Prompt input box, shared UI primitives
│       └── api/                 # HTTP client + TypeScript types
├── db/
│   ├── init.sql                 # PostgreSQL + pgvector schema
│   └── migrations/              # Schema migrations
├── scripts/
│   ├── eval_retrieval.py        # Retrieval evaluation harness
│   ├── reindex_embeddings.py    # Re-embed chunks after model change
│   ├── export_msa_records.py    # Export M/S/A confidence records
│   ├── open_eval_generate_queries.py   # Generate open-corpus eval queries
│   ├── open_eval_export_answers.py     # Export answers for annotation
│   ├── open_eval_export_csv.py         # Export eval data as CSV
│   ├── open_eval_export_retrieval.py   # Export retrieval annotations
│   ├── open_eval_score_retrieval.py    # Score retrieval annotations
│   ├── open_eval_prepare_blind_claims.py   # Prepare blind claim annotations
│   ├── open_eval_merge_blind_claims.py     # Merge blind claim scores
│   ├── open_eval_build_calibration.py      # Build calibration dataset
│   ├── open_eval_fit_calibration.py        # Fit M/S/A logistic calibration
│   ├── open_eval_eval_calibration.py       # Evaluate calibration quality
│   └── open_eval_annotation_agreement.py   # Inter-annotator agreement
├── Evaluation/
│   ├── ScholarRAG_Evaluation.ipynb  # All metrics visualization notebook
│   ├── data/
│   │   ├── retrieval/           # Golden set + retrieval eval results
│   │   ├── calibration/         # M/S/A calibration + ablation reports
│   │   ├── llm_judge/           # Faithfulness claims + judge runs
│   │   ├── iaa/                 # Inter-annotator agreement
│   │   ├── public_search/       # Public API eval results
│   │   └── human_labels/        # Human annotation datasets
│   ├── figures/                 # Generated plots (PNG)
│   └── queries/                 # Query templates + master sets
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml               # pytest + ruff config
└── Makefile                     # make test / lint / run
```

---

## Design Decisions

### Why pgvector?

pgvector provides ANN search as a first-class PostgreSQL extension, enabling:
- Persistent storage with transactional consistency
- Metadata filtering (`provider`, `model`, `version`, `dim`) to prevent silent vector mixing during model upgrades
- Horizontal scaling via standard Postgres connection pooling (ThreadedConnectionPool)
- Co-location of vector and relational data in one query
- HNSW indexes for sub-millisecond approximate search at scale

### Why hybrid scoring?

Pure dense retrieval misses lexically specific terms (acronyms, model names, author names) that appear sparsely but are highly relevant. Pure sparse retrieval misses semantic synonymy. The hybrid score `(1-α) × cosine_sim + α × sparse_overlap` with tunable `α` (default 0.25) captures both. Most research queries are semantic, so dense retrieval dominates; sparse overlap is a correction signal for named-entity-heavy queries.

### Why M/S/A confidence vs. a single similarity score?

Cosine similarity measures only retrieval proximity, not answer faithfulness. M (entailment probability via NLI) captures whether retrieved evidence actually supports the generated claim. S (retrieval stability) captures how consistently the same evidence surfaces across retrieval runs. A (multi-source agreement) captures cross-provider corroboration. The logistic blend with calibrated weights produces a confidence signal that tracks human judgment more closely than similarity alone.

---

## Evaluation

All evaluation data, scripts, and generated figures live in the [`Evaluation/`](Evaluation/) directory. See [`Evaluation/README.md`](Evaluation/README.md) for the full directory layout.

### Evaluation Components

| Component | Script / Data | Description |
|-----------|--------------|-------------|
| **Retrieval** | `scripts/eval_retrieval.py` → `Evaluation/data/retrieval/` | Recall@K, MRR, nDCG@K on 120-query golden set across 15 papers |
| **Faithfulness** | `/eval/judge` endpoint → `Evaluation/data/llm_judge/` | LLM-as-judge claim extraction and verification (638 claims) |
| **Calibration** | `scripts/open_eval_fit_calibration.py` → `Evaluation/data/calibration/` | Logistic M/S/A model fitting on 1,272 labeled claims, ablation study |
| **IAA** | `scripts/open_eval_annotation_agreement.py` → `Evaluation/data/iaa/` | Inter-annotator agreement (Cohen's Kappa = 0.82) |
| **Public Search** | Live API eval → `Evaluation/data/public_search/` | Provider diversity, latency, and keyword precision across 20 queries |
| **Visualization** | `Evaluation/ScholarRAG_Evaluation.ipynb` | Jupyter notebook with all charts, tables, and summary dashboard |

### Running the retrieval eval harness

```bash
python scripts/eval_retrieval.py \
  --eval-set Evaluation/data/retrieval/golden_set.json \
  --k 10 \
  --output Evaluation/data/retrieval/run_$(date +%Y%m%d).json
```

### Running the full evaluation notebook

```bash
cd Evaluation
jupyter nbconvert --to notebook --execute ScholarRAG_Evaluation.ipynb
```

See [`Evaluation/README.md`](Evaluation/README.md) for the full directory layout and detailed results.

---

## Re-indexing after Model Change

If you change embedding model, provider, or version:

```bash
# 1. Update .env (OLLAMA_EMBED_MODEL, EMBEDDING_VERSION, EMBEDDING_RAW_DIM)
# 2. Run the reindex script
source .venv/bin/activate
python scripts/reindex_embeddings.py --purge-all
```

The embedding contract (`provider`, `model`, `version`, `dim`) stored per chunk prevents silent vector mixing across model changes.

---

## Local Runtime

Run everything on your machine:

```bash
# Terminal 1: database (starts by default, no profile needed)
docker compose up -d db

# Terminal 2: Ollama (or use the Docker profile)
ollama pull mxbai-embed-large && ollama serve
# Alternative: docker compose --profile ollama up -d

# Terminal 3: backend (or use the Docker profile)
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
# Alternative: docker compose --profile backend up -d

# Terminal 4: frontend
cd frontend && npm run dev
```

> **Docker Compose profiles:** `docker compose up -d` only starts Postgres and Adminer.
> Add `--profile backend` to also start the API server, and `--profile ollama` for a
> containerized Ollama instance.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `EMBEDDING_PROVIDER` | `ollama` for local Ollama (recommended local default) |
| `OPENAI_API_KEY` | OpenAI key for generation and judging |
| `RESEARCH_CHAT_MODEL` | Model name (default: `gpt-4o-mini`) |
| `OLLAMA_BASE_URL` | Ollama host URL |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model when `EMBEDDING_PROVIDER=openai` |
| `OPENAI_EMBED_DIMENSIONS` | Requested embedding dimensions for OpenAI embeddings |
| `OLLAMA_EMBED_MODEL` | Embedding model (default: `mxbai-embed-large`) |
| `EMBEDDING_VERSION` | Tracks schema compatibility (e.g. `mxbai-embed-large-v1`) |
| `EMBEDDING_RAW_DIM` | Raw output dimension (1024 for mxbai) |
| `VECTOR_STORE_DIM` | pgvector column dimension (1536 for backward compat) |
| `DATABASE_URL` | Postgres connection string |
| `CORS_ORIGINS` | Comma-separated allowed origins |

---

## Healthcheck

```bash
GET /health/embeddings
```

Returns Ollama reachability, embedding shape, active provider/model/version, and configured dimensions.

---

## Contributing

```bash
make lint       # check code style (ruff)
make lint-fix   # auto-fix
make test       # run full test suite
make eval       # run retrieval evaluation
```

- Python 3.11+ type hints on all public functions
- No bare `except:` — always catch specific exceptions
- Run `make lint && make test` before submitting changes
- Report Recall@5, MRR, and nDCG@10 in PRs that affect retrieval

---

## License

MIT
