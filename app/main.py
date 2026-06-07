from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.domains.ai_tasks.api.router import router as ai_tasks_router
from app.domains.documents.api import error_handlers as documents_error_handlers
from app.domains.documents.api.router import router as documents_router
from app.shared.error_handlers import register_error_handlers
from app.shared.lifespan import lifespan
from app.shared.logging import setup_logging
from app.shared.middleware import register_middleware
from app.web.router import STATIC_DIR
from app.web.router import router as web_router

setup_logging()

app = FastAPI(title="AI Backend", version="3.0.0", lifespan=lifespan)
app.include_router(ai_tasks_router)
app.include_router(documents_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

register_middleware(app)
register_error_handlers(app)
documents_error_handlers.register_error_handlers(app)
