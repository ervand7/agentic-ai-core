# Minimal AI Backend (FastAPI + OpenAI)

Beginner-friendly, production-style FastAPI backend with async OpenAI calls.

## Features

- `POST /ask` - standard Q&A response
- `POST /ask-stream` - token streaming response
- `POST /classify` - strict JSON schema output
- `POST /summarize` - concise summary
- `POST /extract-keywords` - keyword extraction
- `POST /translate` - translation to target language
- Logging for prompts/messages, model, token usage, and latency
- Endpoint-specific system prompts (no single global prompt)

## Project Structure

```text
.
├── app/
│   ├── core/
│   │   ├── config.py          # Env-based settings
│   │   └── logging_config.py  # Logging setup
│   ├── schemas/
│   │   ├── ask.py             # /ask and /ask-stream models
│   │   └── tasks.py           # classify/summarize/keywords/translate models
│   ├── services/
│   │   └── openai_service.py  # OpenAI calls + parsing + logging
│   └── main.py                # FastAPI endpoints
├── .env.example
├── requirements.txt
└── README.md
```

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=300
OPENAI_SYSTEM_PROMPT_ASK=You are a helpful assistant. Give clear and concise answers.
OPENAI_SYSTEM_PROMPT_ASK_STREAM=You are a helpful assistant. Stream clear and concise answers.
OPENAI_SYSTEM_PROMPT_CLASSIFY=You are a precise text classifier. Follow the JSON schema exactly.
OPENAI_SYSTEM_PROMPT_SUMMARIZE=You summarize text in 2-3 short, clear sentences.
OPENAI_SYSTEM_PROMPT_EXTRACT_KEYWORDS=You extract the most relevant keywords from text.
OPENAI_SYSTEM_PROMPT_TRANSLATE=You are a professional translator and return accurate translations.
```

## 2) Run

```bash
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

## 3) Endpoint Examples

### Ask

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain async/await in Python in simple words."}'
```

### Ask Stream (token streaming)

Use `-N` so curl prints chunks as they arrive:

```bash
curl -N -X POST "http://127.0.0.1:8000/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Write a short paragraph about FastAPI streaming."}'
```

### Classify (strict JSON schema)

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"The product is good but shipping was slow."}'
```

Expected format:

```json
{
  "sentiment": "neutral",
  "summary": "Mixed feedback: positive on product quality, negative on shipping speed.",
  "keywords": ["product", "shipping", "slow"]
}
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

## 4) How Streaming Works Internally

1. `/ask-stream` calls `ask_openai_stream(...)` in `app/services/openai_service.py`.
2. OpenAI is called with `stream=True`.
3. The SDK returns chunks incrementally.
4. The async generator `yield`s each token chunk.
5. `StreamingResponse` sends chunks to the client immediately.
6. After stream completion, token usage and latency are logged.

This gives a faster UX because users see output while generation is still running.

## 5) Where Structured Output Is Configured

- JSON schema is defined in `app/services/openai_service.py` as `CLASSIFY_JSON_SCHEMA`.
- It is passed to OpenAI with:
  - `response_format={"type": "json_schema", "json_schema": ...}`
- The returned JSON string is parsed and validated again with Pydantic:
  - `ClassifyResponse.model_validate(...)`

Using both model-side schema constraints and server-side validation keeps output reliable.

## 6) Logging in AI Systems (Why It Matters)

This project logs:

- Full messages sent to the model
- Model name
- Token usage
- Latency

Why this is important:

- Debugging: inspect prompts/messages when output quality drops
- Cost tracking: token usage directly maps to API cost
- Performance monitoring: latency helps detect slow requests
- Reliability: makes incidents easier to investigate in production

## 7) Error Handling

- Missing `OPENAI_API_KEY` -> `500`
- OpenAI API errors -> `502`
- Unexpected internal errors -> `500`
