"""Document routes for Stage 3 semantic search."""

import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.config import get_settings
from app.schemas.documents import (
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentUploadResponse,
    SearchResult,
)
from app.services.chunking_service import chunk_text
from app.services.embedding_service import generate_embedding
from app.storage.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload one .txt file, chunk it, embed chunks, and store everything in memory."""
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt files are supported for now.",
        )

    raw_bytes = await file.read()
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text.",
        ) from exc

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    settings = get_settings()
    chunks = chunk_text(
        content,
        chunk_size=settings.DOCUMENT_CHUNK_SIZE,
        overlap=settings.DOCUMENT_CHUNK_OVERLAP,
    )
    logger.info(
        "document_chunking_completed request_id=%s filename=%s chunk_count=%s chunk_size=%s overlap=%s",
        request.state.request_id,
        file.filename,
        len(chunks),
        settings.DOCUMENT_CHUNK_SIZE,
        settings.DOCUMENT_CHUNK_OVERLAP,
    )

    embeddings: list[list[float]] = []
    for chunk in chunks:
        embeddings.append(await generate_embedding(chunk, request.state.request_id))

    vector_store = get_vector_store()
    vector_store.add_chunks(filename=file.filename, chunks=chunks, embeddings=embeddings)
    logger.info(
        "document_vectors_stored request_id=%s filename=%s chunks_stored=%s total_chunks_in_store=%s",
        request.state.request_id,
        file.filename,
        len(chunks),
        vector_store.count(),
    )

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_stored=len(chunks),
        total_characters=len(content),
    )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    payload: DocumentSearchRequest,
    request: Request,
) -> DocumentSearchResponse:
    """Search uploaded document chunks by semantic meaning using vector similarity."""
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents uploaded yet. Upload a .txt file first.",
        )

    start_time = time.perf_counter()
    query_embedding = await generate_embedding(payload.query, request.state.request_id)
    matches = vector_store.search(query_embedding, top_k=payload.top_k)
    latency_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "semantic_search_completed request_id=%s query_length=%s top_k=%s latency_ms=%.2f",
        request.state.request_id,
        len(payload.query),
        payload.top_k,
        latency_ms,
    )

    return DocumentSearchResponse(
        query=payload.query,
        results=[
            SearchResult(
                text=stored_chunk.text,
                similarity=round(similarity, 4),
            )
            for stored_chunk, similarity in matches
        ],
    )
