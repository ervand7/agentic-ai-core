# Agentic AI Core

<div>

**AI backend for LLM tools, semantic search, RAG, and bounded research agents.**

Built with **FastAPI**, **OpenAI**, **Qdrant**, and a clean **Domain-Driven + Hexagonal** architecture.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM%20%2B%20Embeddings-412991?logo=openai&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-DDD%20%2B%20Hexagonal-555)

</div>

---

## What It Is

Agentic AI Core is an async AI backend that packages the pieces most production AI apps need:
LLM text tools, tool calling, document ingestion, vector search, grounded RAG answers,
and a multi-step research agent.

It also ships with a built-in single-page web UI, so every endpoint can be tested directly
from the browser with no separate frontend server.

---

## Why It Stands Out

| Strength | What it means |
| --- | --- |
| **Clean architecture** | Two bounded contexts, `ai_tasks` and `documents`, stay isolated behind ports and adapters. Swap OpenAI or Qdrant without touching domain logic. |
| **Real RAG pipeline** | Upload → chunk with overlap → embed → store → retrieve → ground → answer with citations. If nothing relevant is found, it gracefully abstains with "I don't know". |
| **Hybrid retrieval** | Combine vector similarity with keyword matching, filename filters, and a minimum-similarity threshold. |
| **Streaming UX** | `/ask-stream` returns tokens as they are generated, and the web UI renders them live. |
| **Operational maturity** | Startup config validation, retries with exponential backoff, timeouts, friendly provider-error mapping, and structured request logs. |
| **Batteries included** | A browser UI served from `/` exercises every feature and includes a live health indicator. |

---

## Feature Map

| Area | What you get |
| --- | --- |
| **LLM text tools** | Ask, streaming Ask, classify sentiment, summarize, extract keywords, translate, and full text analysis |
| **Tool calling** | Model-selected tools for mock weather, document search, ticket drafts, and email drafts |
| **Research agent** | Goal-driven `plan → act → observe` loop with hard iteration/token budgets, self-critique, and a full execution trace |
| **Semantic search** | Embed-and-search over uploaded docs, similarity scores, and top-K control |
| **Hybrid search** | Vector + keyword matching, filename filtering, and min-similarity threshold |
| **RAG** | Grounded answers with numbered citations and graceful abstention |
| **Document ingestion** | `.txt` and text-based `.pdf` upload, chunking with overlap, and vector storage |
| **Web UI** | Built-in single-page app covering every endpoint, plus a live health indicator |
| **Reliability** | Retries, backoff, timeouts, and provider-error → HTTP mapping |
| **Observability** | Structured logs with request id, model, latency, tokens, similarity, and chunk counts |
| **Configurability** | Per-task temperatures, tunable prompts via environment variables, and RAG knobs |

---

## Quick Start

### 1. Install

Dependencies are managed with [Poetry](https://python-poetry.org/).

```bash
poetry install                  # runtime + dev dependencies
# or: poetry install --only main # runtime only
```

Use `poetry run ...` for commands inside the environment, or open a shell with:

```bash
poetry shell
```

### 2. Configure

```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

`.env` is git-ignored. Required settings are validated at startup, so the app fails fast
if `OPENAI_API_KEY` is missing.

### 3. Start Qdrant

Qdrant is required for semantic search and RAG.

```bash
docker compose up -d qdrant
```

### 4. Run the API

```bash
poetry run uvicorn app.main:app --reload
```

| Surface | URL |
| --- | --- |
| API | `http://127.0.0.1:8000` |
| Web UI | `http://127.0.0.1:8000/` |
| OpenAPI docs | `http://127.0.0.1:8000/docs` |

### Run Everything in Docker

Start the app and Qdrant together:

```bash
docker compose up --build -d
```

Stop them:

```bash
docker compose down
```

---

## API Reference

Interactive OpenAPI docs are available at `/docs`.
Ready-to-run `curl` examples for every endpoint live in
[`doc/3. API_CURL_EXAMPLES.md`](doc/3.%20API_CURL_EXAMPLES.md).

### LLM Text Tools (`ai_tasks`)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `POST` | `/ask` | Ask a question and get a single answer |
| `POST` | `/ask-stream` | Same as `/ask`, streamed token-by-token |
| `POST` | `/classify` | Sentiment classification as structured JSON |
| `POST` | `/summarize` | 2-3 sentence summary |
| `POST` | `/extract-keywords` | Key terms as structured JSON |
| `POST` | `/translate` | Translate text to a target language |
| `POST` | `/analyze-text` | Combined summary, sentiment, keywords, and language |
| `POST` | `/tool-assistant` | Assistant that can call safe backend tools |
| `POST` | `/research-agent` | Multi-step research agent that returns a report plus its full trace |

### Semantic Search & RAG (`documents`)

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/documents/upload` | Upload `.txt`/`.pdf`, chunk, embed, and store in Qdrant |
| `POST` | `/documents/search` | Vector search with optional keyword search and filters |
| `POST` | `/documents/ask` | RAG answer grounded in your docs, with citations |

---

## Configuration

All settings are read from environment variables or `.env` and validated by
`pydantic-settings`. See `app/shared/config.py` for the full source of truth.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI credentials, required | - |
| `OPENAI_MODEL` | Chat model | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_TEMPERATURE_*` | Per-task temperatures for classify, summarize, and other tasks | task-specific |
| `OPENAI_TIMEOUT_SECONDS` / `OPENAI_MAX_RETRIES` | Reliability knobs | `20` / `2` |
| `QDRANT_URL` / `QDRANT_COLLECTION_NAME` | Vector store target | `localhost:6333` / `documents` |
| `DOCUMENT_CHUNK_SIZE` / `DOCUMENT_CHUNK_OVERLAP` | Chunking behaviour | `500` / `100` |
| `RAG_TOP_K` / `RAG_MIN_SIMILARITY` | Retrieval tuning | `4` / `0.2` |
| `RAG_TEMPERATURE` / `RAG_MAX_TOKENS` | Answer generation | `0.1` / `500` |
| `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOTAL_TOKENS` | Research-agent safety budgets | `6` / `12000` |
| `AGENT_TEMPERATURE` / `AGENT_REPORT_MAX_TOKENS` | Agent sampling and final report size | `0.1` / `800` |
| `AGENT_ENABLE_REFLECTION` | Run the agent's self-critique pass | `true` |
| `PROMPT_*_SYSTEM` | Tunable system prompts per task | sensible defaults |

Prompts are versioned in `app/domains/ai_tasks/domain/prompts.py`. Bump a prompt's
`version` when you change its wording meaningfully.

`python-multipart` powers file uploads, and `pypdf` handles PDF parsing. Both are included
in `pyproject.toml`.

---

## Observability

Every request is logged with structured fields that make debugging and tracing easier:

| Signal | Details |
| --- | --- |
| Request identity | `request_id`, also returned in the `x-request-id` response header |
| Model activity | endpoint name, model name, latency, and token usage |
| Document ingestion | chunk counts for uploads |
| Search quality | similarity scores and search latency |
| Failures | errors |

Sensitive values such as `OPENAI_API_KEY` are never logged.

---

## Current Limitations

- Qdrant must be running for upload, search, and ask.
- Upload supports `.txt` and text-based `.pdf` only. Scanned PDFs are not OCR'd.
