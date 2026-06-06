## 1) Indexes

- Qdrant collection now uses explicit HNSW config on create:
  - `m=16`
  - `ef_construct=100`
  - `full_scan_threshold=10000`
- Added payload index for metadata field `filename` (`KEYWORD` schema).

## 2) Metadata filtering

- `POST /documents/search` now supports optional `filename`.
- If provided, search is filtered to chunks where payload `filename` matches exactly.

## 3) Hybrid search (keyword + vector)

- `POST /documents/search` now supports optional `keyword`.
- Search now does hybrid ranking:
  - vector similarity from Qdrant
  - lexical keyword score from chunk text
  - final score = `0.8 * vector + 0.2 * keyword`
- When `keyword` is provided, only chunks with keyword signal are kept.

## 4) Similarity thresholds

- `POST /documents/search` now supports optional `min_similarity` (`0..1`).
- Chunks below this vector similarity are dropped before final ranking.

## 5) Vector DB limitations (documented briefly)

Current limitations still relevant:

- Hybrid lexical scoring is simple (token/phrase heuristics), not BM25.
- Large-scale production tuning still needed (`m`, `ef_search`, quantization, shard strategy).
- Exact thresholds can hide relevant results if set too aggressively.
- Metadata filtering is exact-match only for `filename` right now.

## API shape added

`DocumentSearchRequest` new optional fields:

- `filename: str`
- `keyword: str`
- `min_similarity: float` (0..1)
