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

The body is the raw `.txt` file.

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

Nothing is persisted yet. This is just a list of strings in process memory.

---

### Step 3 — Embed each chunk via OpenAI

**Request 1 to OpenAI** (simplified):

```json
POST https://api.openai.com/v1/embeddings
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

### Step 4 — Save into `InMemoryVectorStore`

Inside the application (`self._chunks`):

```
┌─────────────────────────────────────────────────────────────────┐
│  InMemoryVectorStore (RAM — one shared instance per process)    │
├─────────────────────────────────────────────────────────────────┤
│  [0] StoredChunk:                                               │
│      filename  = "coffee.txt"                                   │
│      text      = "To brew espresso, press the button twice."    │
│      embedding = [0.12, 0.85, 0.03, 0.44, ...]                  │
│                                                                 │
│  [1] StoredChunk:                                               │
│      filename  = "coffee.txt"                                   │
│      text      = "To clean the milk frother, remove the..."     │
│      embedding = [0.21, 0.88, 0.05, 0.31, ...]                  │
│                                                                 │
│  [2] StoredChunk:                                               │
│      filename  = "coffee.txt"                                   │
│      text      = "The device warranty is 2 years."              │
│      embedding = [0.05, 0.10, 0.92, 0.11, ...]                  │
└─────────────────────────────────────────────────────────────────┘
```

**Saved:** chunk text + its vector + source filename.  
**Not saved separately:** the original file as a whole — only the chunks.

Relevant code:

- Chunking: `app/domains/documents/domain/chunking.py`
- Ingest use case: `app/domains/documents/application/services.py` → `IngestDocumentService`
- Storage: `app/domains/documents/infrastructure/in_memory_vector_store.py`

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

- `query` — natural-language question
- `top_k` — how many best-matching chunks to return (default `3`, max `10`)

---

## What the server does (step by step)

### Step 1 — Check the store

```python
store.count()  # → 3  ✅ something to search
```

If the count is `0`, the API returns an error: *"No documents uploaded yet. Upload a .txt file first."*

---

### Step 2 — Embed the question via OpenAI

**Request to OpenAI:**

```json
POST https://api.openai.com/v1/embeddings
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

### Step 3 — Compare locally in `InMemoryVectorStore`

Your code loops over **all stored chunks** and scores each one with `cosine_similarity()`:

```
query_vector = [0.20, 0.87, 0.06, 0.30, ...]

compare with [0] embedding [0.12, 0.85, ...]  → similarity = 0.41
compare with [1] embedding [0.21, 0.88, ...]  → similarity = 0.94  ← best match
compare with [2] embedding [0.05, 0.10, ...]  → similarity = 0.18
```

This is **local math** in `chunking.py`. OpenAI is not involved in this step.

Results are sorted by similarity (highest first). With `top_k=2`:

```
1. similarity 0.94 → "To clean the milk frother, remove the nozzle..."
2. similarity 0.41 → "To brew espresso, press the button twice."
```

Relevant code:

- Search use case: `app/domains/documents/application/services.py` → `SearchDocumentsService`
- Similarity + ranking: `app/domains/documents/infrastructure/in_memory_vector_store.py`

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
