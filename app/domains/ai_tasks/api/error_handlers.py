"""AI tasks domain exception-to-HTTP mappings."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domains.ai_tasks.application.langgraph_research_agent_service import (
    ResearchRunNotFoundError,
)


async def handle_research_run_not_found(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ResearchRunNotFoundError, handle_research_run_not_found)
