"""Documents domain exception-to-HTTP mappings."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domains.documents.application.services import (
    DocumentValidationError,
    EmptyVectorStoreError,
)


async def handle_document_validation(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


async def handle_empty_vector_store(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DocumentValidationError, handle_document_validation)
    app.add_exception_handler(EmptyVectorStoreError, handle_empty_vector_store)
