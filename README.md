# FastAPI + OpenAI (Stage 2)

Beginner-friendly AI backend with async OpenAI calls, clean service boundaries, retries, timeouts, and prompt versioning.

## Current Features

- `POST /ask` - standard Q&A response
- `POST /ask-stream` - token streaming response
- `POST /classify` - sentiment + summary + keywords (strict JSON schema)
- `POST /summarize` - concise summary
- `POST /extract-keywords` - keyword extraction
- `POST /translate` - translation to target language
- `POST /analyze-text` - combined summary/sentiment/keywords/language
- Startup validation for `OPENAI_API_KEY`
- Exponential backoff retries for temporary OpenAI failures
- Friendly timeout and rate-limit errors
- Structured logging (`request_id`, endpoint, model, latency, tokens, errors)

## Project Structure

```text
.
├── app/
│   ├── core/
│   │   ├── config.py            # pydantic-settings config
│   │   ├── exceptions.py        # service-layer exception types
│   │   └── logging_config.py
│   ├── prompts/
│   │   └── templates.py         # prompt registry + versions (e.g. classify_v1)
│   ├── schemas/
│   │   ├── ask.py
│   │   └── tasks.py
│   ├── services/
│   │   ├── openai_client.py     # direct OpenAI SDK wrapper
│   │   └── openai_service.py    # endpoint workflows and output parsing
│   └── main.py                  # FastAPI routes
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=300
OPENAI_TIMEOUT_SECONDS=20
OPENAI_MAX_RETRIES=2
OPENAI_RETRY_BASE_DELAY_SECONDS=0.5
```

> Prompts are no longer loaded from env vars. Prompt text lives in `app/prompts/templates.py` with explicit versions (`ask_v1`, `classify_v1`, `analyze_text_v1`, etc.).

`.env` is git-ignored.

## Run

```bash
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

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

Expected response shape:

```json
{
  "summary": "string",
  "sentiment": "positive|negative|neutral",
  "keywords": ["string"],
  "language": "string",
  "model": "string",
  "tokens_used": 123,
  "prompt_version": "analyze_text_v1"
}
```

## Error Handling (actual behavior)

- Missing/invalid `OPENAI_API_KEY` at startup -> app fails to start
- Rate limit errors -> `429`
- Provider timeout -> `504`
- Temporary provider failures -> `503`
- Other provider errors -> `502`
- Invalid input payload -> `422` (FastAPI/Pydantic validation)
- Unexpected server errors -> `500`

## Logging

The backend logs operational metadata for observability:

- `request_id` (also returned in `x-request-id` response header)
- endpoint name
- model name
- latency
- token usage
- errors

Sensitive data such as `OPENAI_API_KEY` is not logged.
