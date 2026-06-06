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

## Curl Examples

### Health

```bash
curl "http://127.0.0.1:8000/health"
```

### Ask

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain async/await in Python in simple words."}'
```

### Ask Stream

```bash
curl -N -X POST "http://127.0.0.1:8000/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Write a short paragraph about FastAPI streaming."}'
```

### Classify

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"The product is good but shipping was slow."}'
```

### Summarize

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Paste a longer paragraph here..."}'
```

### Extract Keywords

```bash
curl -X POST "http://127.0.0.1:8000/extract-keywords" \
  -H "Content-Type: application/json" \
  -d '{"text":"FastAPI is often used for building async APIs in Python."}'
```

### Translate

```bash
curl -X POST "http://127.0.0.1:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text":"Good morning, how are you?","target_language":"Spanish"}'
```

### Analyze Text (combined endpoint)

```bash
curl -X POST "http://127.0.0.1:8000/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{"text":"FastAPI is great for building APIs quickly, but onboarding juniors needs better docs."}'
```

### Upload document

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@./sample.txt"
```

Example response:

```json
{
  "filename": "sample.txt",
  "chunks_stored": 4,
  "total_characters": 1672
}
```

### Semantic search

```bash
curl -X POST "http://127.0.0.1:8000/documents/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does retry logic work?","top_k":3}'
```

Example response:

```json
{
  "query": "How does retry logic work?",
  "results": [
    {
      "text": "Retries use exponential backoff to reduce pressure...",
      "similarity": 0.9241
    },
    {
      "text": "Timeout handling returns a friendly API message...",
      "similarity": 0.8819
    }
  ]
}
```

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
