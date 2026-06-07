# FastAPI + OpenAI

Production-style AI backend with async OpenAI calls, clean service boundaries, and semantic search powered by Qdrant.

## Current Features

### Core AI endpoints

- `GET /health`
- `POST /ask`
- `POST /ask-stream`
- `POST /classify`
- `POST /summarize`
- `POST /extract-keywords`
- `POST /translate`
- `POST /analyze-text`

### Semantic search endpoints

- `POST /documents/upload` (upload `.txt`, chunk text, embed chunks, store vectors in Qdrant)
- `POST /documents/search` (embed query and search nearest chunks in Qdrant)

### Web UI

- A built-in single-page UI that covers every endpoint above
- Served by the app at `/` (no separate frontend server needed)

### Operational features

- Startup validation for required settings (`OPENAI_API_KEY`)
- Retries with exponential backoff for transient OpenAI errors
- Friendly provider error mapping (`429`, `503`, `504`, `502`)
- Structured logging with `request_id`

## Semantic Search Concepts

### What are embeddings?

Embeddings are numeric vectors that represent text meaning.  
Texts with similar meaning are mapped to nearby directions in vector space.

### What is cosine similarity?

Higher values (closer to `1`) mean stronger semantic similarity.

### Why chunking with overlap?

Long documents are split into smaller chunks before embedding.  
Overlap preserves context at chunk boundaries so meaning is less likely to be lost.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Qdrant (required for semantic search):

```bash
docker compose up -d qdrant
```

Create `.env` from `.env.example`:

Notes:

- `python-multipart` is required for file uploads and is included in `requirements.txt`.
- Prompt texts are loaded from `PROMPT_*_SYSTEM` env vars; defaults live in `app/shared/config.py` and the registry is built in `app/domains/ai_tasks/domain/prompts.py`. Bump the `version` in `prompts.py` when changing a prompt's wording meaningfully.
- `.env` is git-ignored.

## Run

```bash
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

## Run With Docker (App + Qdrant)

Use one command to start both services:

```bash
docker compose up --build -d
```

Stop everything:

```bash
docker compose down
```

## Web UI

The project ships with a built-in browser UI that exercises every endpoint
(Ask, Ask-Stream, Classify, Summarize, Extract Keywords, Translate, Analyze Text,
Document Upload, Semantic Search, and Health).

Once the app is running, just open:

```
http://127.0.0.1:8000/
```

- Pick a tool from the sidebar, fill the form, and press **Run**.
- `Ask (Stream)` renders the answer token-by-token as it streams.
- Search results show a similarity bar; every response has a raw-JSON view.
- A live health indicator polls `/health` in the top-right corner.

### Where it lives (architecture)

The UI is a **presentation-only delivery layer** and is intentionally kept out of the
domain code, so the DDD boundaries stay clean:

- `app/web/router.py` — serves the single page at `/`
- `app/web/static/` — `index.html`, `styles.css`, `app.js`

It talks to the existing domain HTTP endpoints over the same origin, so there is no
extra coupling to the `ai_tasks` or `documents` bounded contexts.

## Curl Examples

See [`doc/3. API_CURL_EXAMPLES.md`](doc/3.%20API_CURL_EXAMPLES.md) for ready-to-run `curl` examples for every endpoint.

## Error Handling

- Missing/invalid required settings at startup -> app fails fast
- Rate limit errors -> `429`
- Provider timeout -> `504`
- Temporary provider failures -> `503`
- Other provider errors -> `502`
- Invalid input payload -> `422`
- Unexpected server errors -> `500`

## Logging

The backend logs:

- `request_id` (returned in `x-request-id`)
- endpoint name
- model name
- latency
- token usage
- chunk counts for uploads
- similarity scores and search latency
- errors

Sensitive values such as `OPENAI_API_KEY` are not logged.

## Limitations (Current Stage)

- Qdrant must be running for document upload/search
- Upload supports `.txt` only
