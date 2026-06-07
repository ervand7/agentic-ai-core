"""HTTP routes for the documents bounded context."""

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.domains.documents.api.dependencies import (
    get_answer_question_service,
    get_ingest_document_service,
    get_search_documents_service,
)
from app.domains.documents.api.schemas import (
    CitationResult,
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentUploadResponse,
)
from app.domains.documents.application.services import (
    AnswerQuestionService,
    IngestDocumentService,
    SearchDocumentsService,
)
from app.domains.documents.infrastructure.document_loader import (
    DocumentDecodeError,
    UnsupportedDocumentError,
    extract_text,
    is_supported,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    service: IngestDocumentService = Depends(get_ingest_document_service),
) -> DocumentUploadResponse:
    """Upload a .txt or .pdf file, chunk it, embed chunks, and store vectors in Qdrant."""
    if not file.filename or not is_supported(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt and .pdf files are supported for now.",
        )

    raw_bytes = await file.read()
    try:
        content = extract_text(filename=file.filename, raw_bytes=raw_bytes)
    except (UnsupportedDocumentError, DocumentDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
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
        filename=payload.filename,
        keyword=payload.keyword,
        min_similarity=payload.min_similarity,
        request_id=request.state.request_id,
    )


@router.post("/ask", response_model=DocumentAskResponse)
async def ask_documents(
    payload: DocumentAskRequest,
    request: Request,
    service: AnswerQuestionService = Depends(get_answer_question_service),
) -> DocumentAskResponse:
    """Answer a question using RAG over the uploaded documents, with citations."""
    result = await service.execute(
        question=payload.question,
        top_k=payload.top_k,
        filename=payload.filename,
        min_similarity=payload.min_similarity,
        request_id=request.state.request_id,
    )
    return DocumentAskResponse(
        question=payload.question,
        answer=result.answer,
        used_context=result.used_context,
        citations=[
            CitationResult(
                index=citation.index,
                filename=citation.filename,
                text=citation.text,
                similarity=round(citation.similarity, 4),
            )
            for citation in result.citations
        ],
        model=result.model,
        tokens_used=result.tokens_used,
    )
