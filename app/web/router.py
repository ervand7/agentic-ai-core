"""Routes that serve the browser UI (presentation layer only)."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["web"])

STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")
