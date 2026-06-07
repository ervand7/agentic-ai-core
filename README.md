# Agentic AI Core

> A production-style AI backend built with **FastAPI** and **OpenAI** — async LLM calls, a clean Domain-Driven design, and a full **RAG pipeline** powered by **Qdrant** vector search. Ships with a built-in web UI, so you can try every feature in your browser.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM%20%2B%20Embeddings-412991?logo=openai&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-DDD%20%2B%20Hexagonal-555)

---

## Why this project is interesting

- **Two bounded contexts, one clean codebase.** `ai_tasks` (LLM text tools) and `documents` (semantic search + RAG) are fully isolated behind ports & adapters — swap OpenAI or Qdrant without touching domain logic.
- **A complete RAG pipeline, not a toy.** Upload → chunk (with overlap) → embed → store → retrieve → ground → answer **with citations**, and it **abstains** ("I don't know") when nothing relevant is found.
- **Hybrid & filtered search.** Combine vector similarity with keyword matching, filter by filename, and enforce a minimum-similarity threshold.
- **Real-time streaming.** `/ask-stream` returns tokens as they're generated; the web UI renders them live.
- **Operational maturity.** Startup config validation, retries with exponential backoff, friendly provider-error mapping, and structured logging with a per-request `request_id`.
- **Batteries-included UI.** A single-page app served at `/` exercises every endpoint — no separate frontend server.

---

## Feature overview

| Area | What you get |
| --- | --- |
| **LLM text tools** | Ask, streaming Ask, classify (sentiment), summarize, extract keywords, translate, full text analysis |
| **Semantic search** | Embed-and-search over uploaded docs, similarity scores, top-K control |
| **Hybrid search** | Vector + keyword matching, filename filtering, min-similarity threshold |
| **RAG** | Grounded answers with numbered citations + graceful abstention |
| **Document ingestion** | `.txt` and text-based `.pdf` upload, chunking with overlap, vector storage |
| **Web UI** | Built-in single-page app covering every endpoint, live health indicator |
| **Reliability** | Retries, backoff, timeouts, provider-error → HTTP mapping |
| **Observability** | Structured logs: request id, model, latency, tokens, similarity, chunk counts |
| **Configurability** | Per-task temperatures, tunable prompts via env vars, RAG knobs |

---

## API reference

### LLM text tools (`ai_tasks`)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET`  | `/health` | Liveness check |
| `POST` | `/ask` | Ask a question, get a single answer |
| `POST` | `/ask-stream` | Same as `/ask`, streamed token-by-token |
| `POST` | `/classify` | Sentiment classification (structured JSON) |
| `POST` | `/summarize` | 2–3 sentence summary |
| `POST` | `/extract-keywords` | Key terms as structured JSON |
| `POST` | `/translate` | Translate text to a target language |
| `POST` | `/analyze-text` | Combined summary + sentiment + keywords + language |

### Semantic search & RAG (`documents`)

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/documents/upload` | Upload `.txt`/`.pdf`, chunk, embed, store in Qdrant |
| `POST` | `/documents/search` | Vector (+ optional keyword) search with filters |
| `POST` | `/documents/ask` | RAG answer grounded in your docs, with citations |

Interactive OpenAPI docs are available at `/docs`. Ready-to-run `curl` snippets for every endpoint live in [`doc/3. API_CURL_EXAMPLES.md`](doc/3.%20API_CURL_EXAMPLES.md).

---

## Quick start

### 1. Install

Dependencies are managed with [Poetry](https://python-poetry.org/).

```bash
poetry install            # runtime + dev dependencies
# or: poetry install --only main   # runtime only
```

Run commands inside the Poetry environment with `poetry run ...`, or open a shell with `poetry shell`.

### 2. Configure

```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

> `.env` is git-ignored. Required settings are validated at startup — the app fails fast if `OPENAI_API_KEY` is missing.

### 3. Start Qdrant (needed for semantic search / RAG)

```bash
docker compose up -d qdrant
```

### 4. Run

```bash
poetry run uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Web UI: `http://127.0.0.1:8000/`
- Docs: `http://127.0.0.1:8000/docs`

### Run everything in Docker (app + Qdrant)

```bash
docker compose up --build -d   # start
```

```bash
docker compose down            # stop
```
---

## Configuration

All settings are read from environment variables / `.env` and validated by `pydantic-settings` (see `app/shared/config.py`). Highlights:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI credentials (required) | — |
| `OPENAI_MODEL` | Chat model | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_TEMPERATURE_*` | Per-task temperatures (classify, summarize, …) | task-specific |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` | Reliability knobs | `20` / `2` |
| `QDRANT_URL` / `QDRANT_COLLECTION_NAME` | Vector store target | `localhost:6333` / `documents` |
| `DOCUMENT_CHUNK_SIZE` / `DOCUMENT_CHUNK_OVERLAP` | Chunking behaviour | `500` / `100` |
| `RAG_TOP_K` / `RAG_MIN_SIMILARITY` | Retrieval tuning | `4` / `0.2` |
| `RAG_TEMPERATURE` / `RAG_MAX_TOKENS` | Answer generation | `0.1` / `500` |
| `PROMPT_*_SYSTEM` | Tunable system prompts per task | sensible defaults |

> Prompts are versioned in `app/domains/ai_tasks/domain/prompts.py`. Bump a prompt's `version` when you change its wording meaningfully. `python-multipart` (file uploads) and `pypdf` (PDF parsing) are included in `pyproject.toml`.

---

## Observability

Every request is logged with structured fields:

- `request_id` (also returned in the `x-request-id` response header)
- endpoint name, model name, latency, token usage
- chunk counts for uploads
- similarity scores and search latency
- errors

Sensitive values such as `OPENAI_API_KEY` are never logged.

---

## Current limitations

- Qdrant must be running for upload / search / ask.
- Upload supports `.txt` and text-based `.pdf` only (no OCR for scanned PDFs).
