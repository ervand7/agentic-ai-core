# Documents Semantic Search — Full Flow

This guide walks through how `/documents/upload` and `/documents/search` work end to end, with concrete request/response examples and a clear picture of **what gets saved**, **what gets compared**, and **where OpenAI is involved**.

---

## Sample document

Suppose you upload a small file `coffee.txt`:

```
To brew espresso, press the button twice.
To clean the milk frother, remove the nozzle and rinse with warm water.
The device warranty is 2 years.
```

With the default `DOCUMENT_CHUNK_SIZE=500`, a short file like this may produce 1–3 chunks. For clarity, the examples below assume three chunks.

---

# Part 1: UPLOAD

## Client request

```http
POST /documents/upload
Content-Type: multipart/form-data

file: coffee.txt
```

---

## What the server does (step by step)

### Step 1 — Read the file

```
content  = "To brew espresso...\nTo clean the milk frother...\nThe device warranty..."
filename = "coffee.txt"
```

### Step 2 — `chunk_text()` splits the text

```python
chunks = [
  "To brew espresso, press the button twice.",
  "To clean the milk frother, remove the nozzle and rinse with warm water.",
  "The device warranty is 2 years.",
]
```

### Step 3 — Embed each chunk via OpenAI

**Request 1 to OpenAI** (simplified):

```json
{
  "input": "To brew espresso, press the button twice.",
  "model": "text-embedding-3-small"
}
```

**OpenAI response:**

```json
{
  "data": [{
    "embedding": [0.12, 0.85, 0.03, 0.44, "..."]
  }]
}
```

**Request 2 to OpenAI:**

```json
{ "input": "To clean the milk frother, remove the nozzle and rinse with warm water." }
```

**Response:**

```json
{ "embedding": [0.21, 0.88, 0.05, 0.31, "..."] }
```

**Request 3 to OpenAI:**

```json
{ "input": "The device warranty is 2 years." }
```

**Response:**

```json
{ "embedding": [0.05, 0.10, 0.92, 0.11, "..."] }
```

OpenAI does **not** store this data for you. It returns the vectors and moves on.

---

### Step 4 — Save into Qdrant (vector DB)

Inside Qdrant (collection from `QDRANT_COLLECTION_NAME`):

```
┌─────────────────────────────────────────────────────────────────┐
│  Qdrant collection (persistent vector database)                 │
├─────────────────────────────────────────────────────────────────┤
│  [0] Point:                                                     │
│      payload.filename = "coffee.txt"                            │
│      payload.text     = "To brew espresso, press the..."        │
│      vector           = [0.12, 0.85, 0.03, 0.44, ...]           │
│                                                                 │
│  [1] Point:                                                     │
│      payload.filename = "coffee.txt"                            │
│      payload.text     = "To clean the milk frother, remove..."  │
│      vector           = [0.21, 0.88, 0.05, 0.31, ...]           │
│                                                                 │
│  [2] Point:                                                     │
│      payload.filename = "coffee.txt"                            │
│      payload.text     = "The device warranty is 2 years."       │
│      vector           = [0.05, 0.10, 0.92, 0.11, ...]           │
└─────────────────────────────────────────────────────────────────┘
```

Relevant code:

- Chunking: `app/domains/documents/domain/chunking.py`
- Ingest use case: `app/domains/documents/application/services.py` → `IngestDocumentService`
- Storage: `app/domains/documents/infrastructure/qdrant_vector_store.py`

---

## Upload response to the client

```json
{
  "filename": "coffee.txt",
  "chunks_stored": 3,
  "total_characters": 142
}
```

Vectors are **not** returned to the client — only ingestion stats.

---

# Part 2: SEARCH

## Client request

```http
POST /documents/search
Content-Type: application/json

{
  "query": "how do I clean the milk frother",
  "top_k": 2
}
```

---

## What the server does (step by step)

### Step 1 — Check the store

```python
store.count()  # → 3  ✅ something to search
```

If the count is `0`, the API returns an error: *"No documents uploaded yet. Upload a .txt file first."*

### Step 2 — Embed the question via OpenAI

**Request to OpenAI:**

```json
{
  "input": "how do I clean the milk frother",
  "model": "text-embedding-3-small"
}
```

**OpenAI response:**

```json
{
  "embedding": [0.20, 0.87, 0.06, 0.30, "..."]
}
```

⚠️ **OpenAI does not see your file.** Only the query string is sent.

---

### Step 3 — Search in Qdrant by vector similarity

Your code sends the query vector to Qdrant (`query_points`), and Qdrant returns the top matching chunks by similarity score:

Example ranking:

```
1. similarity 0.94 → "To clean the milk frother, remove the nozzle..."
2. similarity 0.41 → "To brew espresso, press the button twice."
```

Relevant code:

- Search use case: `app/domains/documents/application/services.py` → `SearchDocumentsService`
- Similarity + ranking: `app/domains/documents/infrastructure/qdrant_vector_store.py`

---

## Search response to the client

```json
{
  "query": "how do I clean the milk frother",
  "results": [
    {
      "text": "To clean the milk frother, remove the nozzle and rinse with warm water.",
      "similarity": 0.9401
    },
    {
      "text": "To brew espresso, press the button twice.",
      "similarity": 0.4123
    }
  ]
}
```

This returns **text snippets from your uploaded file**, not a ChatGPT-generated answer.
