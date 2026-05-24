"""HTTP routes for the documents bounded context."""

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.domains.documents.api.dependencies import (
    get_ingest_document_service,
    get_search_documents_service,
)
from app.domains.documents.api.schemas import (
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentUploadResponse,
)
from app.domains.documents.application.services import (
    IngestDocumentService,
    SearchDocumentsService,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    service: IngestDocumentService = Depends(get_ingest_document_service),
) -> DocumentUploadResponse:
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

    return await service.execute(
        filename=file.filename,
        content=content,
        request_id=request.state.request_id,
    )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    payload: DocumentSearchRequest,
    request: Request,
    service: SearchDocumentsService = Depends(get_search_documents_service),
) -> DocumentSearchResponse:
    """Search uploaded document chunks by semantic meaning."""
    return await service.execute(
        query=payload.query,
        top_k=payload.top_k,
        request_id=request.state.request_id,
    )
