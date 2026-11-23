<!-- ═══════════════════════════════ HERO ═══════════════════════════════ -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=230&color=0:7F00FF,50:E100FF,100:00D4FF&text=Agentic%20AI%20Core&fontColor=ffffff&fontSize=62&fontAlignY=38&desc=LLM%20tools%20%E2%80%A2%20RAG%20%E2%80%A2%20Tool%20Calling%20%E2%80%A2%20Research%20Agents&descAlignY=58&descSize=18&animation=fadeIn" width="100%" alt="Agentic AI Core"/>

<a href="#-quick-start">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=2800&pause=700&color=B14CFF&center=true&vCenter=true&width=720&lines=Async+AI+backend+built+on+FastAPI+%2B+OpenAI;Grounded+RAG+with+citations+over+Qdrant;Agents+that+plan%2C+act%2C+critique+and+revise;Human-in-the-loop+approvals+with+LangGraph" alt="Typing animation"/>
</a>

<br/>

<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-async-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenAI-LLM%20%2B%20Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=for-the-badge&logoColor=white"/>
<br/>
<img src="https://img.shields.io/badge/LangChain-RAG%20chain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-Agent%20graph-FF6F00?style=for-the-badge&logo=langgraph&logoColor=white"/>
<img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<br/>
<img src="https://img.shields.io/badge/tests-338%20passing-2EA44F?style=flat-square&logo=pytest&logoColor=white"/>
<img src="https://img.shields.io/badge/mypy-clean-2A6DB2?style=flat-square"/>
<img src="https://img.shields.io/badge/ruff-lint%20%2B%20format-D7FF64?style=flat-square&logo=ruff&logoColor=black"/>
<img src="https://img.shields.io/badge/architecture-DDD%20%2B%20Hexagonal-8A2BE2?style=flat-square"/>

<br/><br/>

**[✨ Features](#-features) · [🏛 Architecture](#-architecture) · [🚀 Quick Start](#-quick-start) · [📡 API](#-api-reference) · [⚙️ Config](#%EF%B8%8F-configuration) · [📚 Docs](#-deep-dive-docs)**

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 💡 What It Is

**Agentic AI Core** is an async AI backend that packages the pieces most production AI apps need —
LLM text tools, tool calling, document ingestion, vector search, grounded RAG answers,
and multi-step research agents — behind a clean **Domain-Driven + Hexagonal** architecture.

It ships with a **built-in single-page web UI**, so every endpoint can be tried straight from the browser.
No separate frontend server needed.

> [!TIP]
> Several features exist in **two flavors side by side** — hand-written and framework-based —
> so you can compare plain Python against **LangChain** and **LangGraph** on the same data.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ✨ Features

<table>
<tr>
<td width="33%" valign="top">

### 🧠 LLM Text Tools
Ask, **streaming** ask, sentiment classification, summarize, keyword extraction,
translation, and combined text analysis — all with **structured JSON** output.

</td>
<td width="33%" valign="top">

### 🔍 Hybrid Search
Semantic vector search over Qdrant, blended with **keyword matching**,
filename filters, and a minimum-similarity threshold.

</td>
<td width="33%" valign="top">

### 📚 Grounded RAG
Answers built **only** from your documents with numbered **citations**.
Nothing relevant found? It honestly says **"I don't know"**.

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 🛠 Safe Tool Calling
The model picks tools (weather, doc search, ticket & email **drafts**).
Every call is validated, risk-tagged, size-limited, and audited.

</td>
<td width="33%" valign="top">

### 🤖 Research Agent
`plan → act → observe → reflect` loop with hard **iteration and token budgets**,
self-critique, and a full step-by-step trace.

</td>
<td width="33%" valign="top">

### 🕸 LangGraph Agent
The same agent as a **graph**: the critic can send reports back for revision,
and runs **pause for human approval** before risky tools.

</td>
</tr>
<tr>
<td width="33%" valign="top">

### ⛓ LangChain RAG
The RAG pipeline rebuilt as an LCEL chain —
`retriever → prompt → LLM` — reading the same Qdrant data.

</td>
<td width="33%" valign="top">

### 🛡 Reliability
Retries with **exponential backoff**, timeouts, fail-fast config validation,
and provider errors mapped to clean HTTP codes.

</td>
<td width="33%" valign="top">

### 📈 Observability
Structured logs with `request_id`, model, latency, tokens,
similarity scores, and chunk counts — secrets never logged.

</td>
</tr>
</table>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🏛 Architecture

Two bounded contexts — **`ai_tasks`** and **`documents`** — isolated behind ports and adapters.
Swap OpenAI or Qdrant without touching domain logic.

```mermaid
flowchart LR
    UI["🖥 Web UI<br/>(single-page app)"]:::ui --> API

    subgraph API["⚡ FastAPI"]
        R1["ai_tasks router"]:::api
        R2["documents router"]:::api
    end

    subgraph APP["🧩 Application · use cases"]
        S1["AI tasks service"]:::app
        S2["Research agent<br/>(plain loop)"]:::app
        S3["Research agent<br/>(LangGraph)"]:::lg
        S4["Ingest · Search · Ask"]:::app
        S5["Ask (LangChain)"]:::lc
    end

    subgraph DOMAIN["💎 Domain · ports"]
        P1["LLMProvider"]:::dom
        P2["EmbeddingProvider"]:::dom
        P3["VectorStore"]:::dom
    end

    subgraph INFRA["🔌 Infrastructure · adapters"]
        O["OpenAI client<br/>retries · timeouts"]:::infra
        Q["Qdrant store<br/>hybrid scoring"]:::infra
    end

    R1 --> S1 & S2 & S3
    R2 --> S4 & S5
    S1 & S2 & S3 --> P1
    S4 & S5 --> P2 & P3
    P1 & P2 --> O
    P3 --> Q

    classDef ui fill:#00D4FF,stroke:#0090b3,color:#002b36
    classDef api fill:#009688,stroke:#00695c,color:#fff
    classDef app fill:#7F00FF,stroke:#5a00b3,color:#fff
    classDef lg fill:#FF6F00,stroke:#c25400,color:#fff
    classDef lc fill:#1C3C3C,stroke:#0d1f1f,color:#fff
    classDef dom fill:#E100FF,stroke:#a000b8,color:#fff
    classDef infra fill:#DC244C,stroke:#a01836,color:#fff
```

<details>
<summary><b>📚 RAG pipeline — click to expand</b></summary>
<br/>

```mermaid
sequenceDiagram
    autonumber
    actor U as 👤 User
    participant A as ⚡ API
    participant E as 🧬 Embeddings
    participant Q as 🗄 Qdrant
    participant L as 🧠 LLM

    U->>A: POST /documents/ask
    A->>E: embed(question)
    E-->>A: vector
    A->>Q: top-k similar chunks
    Q-->>A: hits + similarity
    alt nothing relevant
        A-->>U: "I don't know" (no LLM call)
    else context found
        A->>L: system rules + numbered snippets + question
        L-->>A: answer citing [1], [2]
        A-->>U: answer + citations
    end
```

</details>

<details>
<summary><b>🕸 LangGraph research agent — click to expand</b></summary>
<br/>

```mermaid
flowchart TD
    START(["▶ start"]):::edge --> plan["🗺 plan"]:::node
    plan --> research["🔎 research"]:::node
    research -->|tools requested| tools["🛠 tools"]:::node
    tools -->|"✋ pause for approval<br/>(non-read-only tools)"| research
    research -->|budget hit| force["⏱ force report"]:::warn
    research -->|report written| critic["🧐 critic"]:::node
    force --> critic
    critic -->|rejected · revisions left| research
    critic -->|approved| END(["🏁 end"]):::edge

    classDef node fill:#7F00FF,stroke:#E100FF,color:#fff
    classDef warn fill:#FF6F00,stroke:#c25400,color:#fff
    classDef edge fill:#00D4FF,stroke:#0090b3,color:#002b36
```

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 🚀 Quick Start

```bash
# 1️⃣  Install (Poetry)
poetry install

# 2️⃣  Configure — then set OPENAI_API_KEY in .env
cp .env.example .env

# 3️⃣  Start Qdrant (needed for search & RAG)
docker compose up -d qdrant

# 4️⃣  Run the API
poetry run uvicorn app.main:app --reload
```

<div align="center">

| 🌐 Surface | 🔗 URL |
| :---: | :---: |
| **Web UI** | `http://127.0.0.1:8000/` |
| **API** | `http://127.0.0.1:8000` |
| **OpenAPI docs** | `http://127.0.0.1:8000/docs` |

</div>

> [!NOTE]
> `.env` is git-ignored. Settings are validated at startup, so the app **fails fast** if `OPENAI_API_KEY` is missing.

<details>
<summary><b>🐳 Run everything in Docker</b></summary>
<br/>

```bash
docker compose up --build -d   # app + Qdrant
docker compose down            # stop
```

</details>

<details>
<summary><b>🧪 Tests, lint & types</b></summary>
<br/>

```bash
make test-unit     # unit tests
make test-api      # API router tests
make test          # full suite (unit + api + integration)
make lint          # ruff
make typecheck     # mypy
make check         # read-only CI gate: lint + format check + types + coverage
```

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📡 API Reference

Interactive docs live at **`/docs`**. Ready-to-run `curl` examples for every endpoint:
[`doc/1. API_CURL_EXAMPLES.md`](doc/1.%20API_CURL_EXAMPLES.md).

### 🧠 AI Tasks

| Method | Endpoint | Description |
| :---: | --- | --- |
| ![GET](https://img.shields.io/badge/GET-2EA44F?style=flat-square) | `/health` | Liveness check |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/ask` | Ask a question and get a single answer |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/ask-stream` | Same as `/ask`, streamed token-by-token |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/classify` | Sentiment classification as structured JSON |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/summarize` | 2–3 sentence summary |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/extract-keywords` | Key terms as structured JSON |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/translate` | Translate text to a target language |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/analyze-text` | Combined summary, sentiment, keywords, and language |
| ![POST](https://img.shields.io/badge/POST-7F00FF?style=flat-square) | `/tool-assistant` | Assistant that can call safe backend tools |

### 🤖 Agents

| Method | Endpoint | Description |
| :---: | --- | --- |
| ![POST](https://img.shields.io/badge/POST-FF6F00?style=flat-square) | `/research-agent` | Multi-step research agent: report plus full trace |
| ![POST](https://img.shields.io/badge/POST-FF6F00?style=flat-square) | `/research-agent-langgraph` | LangGraph agent: critic revision loop, pauses for human approval |
| ![POST](https://img.shields.io/badge/POST-FF6F00?style=flat-square) | `/research-agent-langgraph/resume` | Approve or reject a paused run's tool calls and continue it |

### 📚 Documents · Search & RAG

| Method | Endpoint | Description |
| :---: | --- | --- |
| ![POST](https://img.shields.io/badge/POST-DC244C?style=flat-square) | `/documents/upload` | Upload `.txt` / `.pdf`, chunk, embed, and store in Qdrant |
| ![POST](https://img.shields.io/badge/POST-DC244C?style=flat-square) | `/documents/search` | Vector search with optional keyword search and filters |
| ![POST](https://img.shields.io/badge/POST-DC244C?style=flat-square) | `/documents/ask` | RAG answer grounded in your docs, with citations |
| ![POST](https://img.shields.io/badge/POST-DC244C?style=flat-square) | `/documents/ask-langchain` | Same RAG answer, pipeline built with LangChain |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## ⚙️ Configuration

All settings come from environment variables or `.env`, validated by `pydantic-settings`.
Source of truth: [`app/shared/config.py`](app/shared/config.py).

<details>
<summary><b>🔧 Show all settings</b></summary>
<br/>

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI credentials, **required** | – |
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
| `AGENT_MAX_REVISIONS` | Extra research rounds the LangGraph critic may request | `1` |
| `PROMPT_*_SYSTEM` | Tunable system prompts per task | sensible defaults |

Prompts are versioned in `app/domains/ai_tasks/domain/prompts.py` — bump a prompt's
`version` when you change its wording meaningfully.

</details>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📈 Observability

| Signal | Details |
| --- | --- |
| 🆔 **Request identity** | `request_id`, also returned in the `x-request-id` response header |
| 🧠 **Model activity** | endpoint name, model name, latency, and token usage |
| 📄 **Document ingestion** | chunk counts for uploads |
| 🎯 **Search quality** | similarity scores and search latency |
| 🚨 **Failures** | errors, mapped to clean HTTP status codes |

> [!IMPORTANT]
> Sensitive values such as `OPENAI_API_KEY` are **never** logged.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📚 Deep-Dive Docs

| | Topic |
| :---: | --- |
| 📖 | [Glossary](doc/0.%20GLOSSARY.md) |
| 🧪 | [API curl examples](doc/1.%20API_CURL_EXAMPLES.md) |
| 🛠 | [Tool calling](doc/2.%20TOOL_CALLING.md) |
| 🔍 | [Semantic search](doc/3.%20DOCUMENTS_SEMANTIC_SEARCH.md) |
| 🗄 | [Qdrant](doc/4.%20Qdrant.md) |
| 📚 | [RAG pipeline](doc/5.%20RAG_PIPELINE.md) |
| ⛓ | [LangChain](doc/6.%20LANGCHAIN.md) |
| 🕸 | [LangGraph](doc/7.%20LANGGRAPH.md) |

## ⚠️ Current Limitations

- Qdrant must be running for upload, search, and ask.
- Upload supports `.txt` and text-based `.pdf` only — scanned PDFs are not OCR'd.
- Paused LangGraph runs are kept in memory and are lost on restart.

<!-- ═══════════════════════════════ FOOTER ═══════════════════════════════ -->

<div align="center">

<br/>

**Built with ❤️ by [Ervand Agadzhanyan](https://github.com/ervand7)**

<img src="https://capsule-render.vercel.app/api?type=waving&height=120&section=footer&color=0:00D4FF,50:E100FF,100:7F00FF" width="100%"/>

</div>
