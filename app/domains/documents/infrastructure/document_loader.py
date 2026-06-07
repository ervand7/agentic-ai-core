"""
Document loading: turn an uploaded file's raw bytes into plain text.

This is the "document loading" step of the RAG pipeline. It lives in
infrastructure because parsing formats like PDF depends on third-party
libraries; the rest of the pipeline only ever sees clean text.
"""

import io

SUPPORTED_EXTENSIONS = (".txt", ".pdf")


class UnsupportedDocumentError(Exception):
    """Raised when an uploaded file type cannot be loaded as text."""


class DocumentDecodeError(Exception):
    """Raised when a supported file type cannot be decoded/parsed."""


def is_supported(filename: str) -> bool:
    """Return True when the filename has a supported extension."""
    lowered = filename.lower()
    return lowered.endswith(SUPPORTED_EXTENSIONS)


def extract_text(*, filename: str, raw_bytes: bytes) -> str:
    """
    Extract plain text from an uploaded file.

    Supports UTF-8 `.txt` and `.pdf` (text-based). Scanned/image-only PDFs
    will yield little or no text because no OCR is performed.
    """
    lowered = filename.lower()
    if lowered.endswith(".txt"):
        return _extract_txt(raw_bytes)
    if lowered.endswith(".pdf"):
        return _extract_pdf(raw_bytes)
    raise UnsupportedDocumentError(
        "Only .txt and .pdf files are supported for now."
    )


def _extract_txt(raw_bytes: bytes) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentDecodeError("File must be UTF-8 encoded text.") from exc


def _extract_pdf(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise DocumentDecodeError(
            "PDF support requires the 'pypdf' package. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - surface any parser failure cleanly
        raise DocumentDecodeError("Could not read the PDF file.") from exc

    return "\n".join(pages)
