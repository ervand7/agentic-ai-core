"""HTTP routes for the AI tasks bounded context."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.domains.ai_tasks.api.dependencies import (
    get_ai_tasks_service,
    get_research_agent_service,
)
from app.domains.ai_tasks.api.schemas import (
    AnalyzeTextRequest,
    AnalyzeTextResponse,
    AskRequest,
    AskResponse,
    ClassifyRequest,
    ClassifyResponse,
    ExtractKeywordsRequest,
    ExtractKeywordsResponse,
    ResearchAgentRequest,
    ResearchAgentResponse,
    SummarizeRequest,
    SummarizeResponse,
    ToolAssistantRequest,
    ToolAssistantResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.domains.ai_tasks.application.agent_service import ResearchAgentService
from app.domains.ai_tasks.application.services import AITasksService

router = APIRouter(tags=["ai"])


@router.get("/health")
async def health_check() -> dict:
    """Small health endpoint useful for quick checks."""
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    payload: AskRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> AskResponse:
    """Accept a question and return an AI answer."""
    return await service.ask(payload.question, request.state.request_id)


@router.post("/ask-stream")
async def ask_stream(
    payload: AskRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> StreamingResponse:
    """Stream generated text token-by-token to the client."""
    return StreamingResponse(
        service.ask_stream(payload.question, request.state.request_id),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify(
    payload: ClassifyRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> ClassifyResponse:
    """Classify user text into sentiment + summary + keywords."""
    return await service.classify(payload.text, request.state.request_id)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    payload: SummarizeRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> SummarizeResponse:
    """Summarize text into a short response."""
    return await service.summarize(payload.text, request.state.request_id)


@router.post("/extract-keywords", response_model=ExtractKeywordsResponse)
async def extract_keywords_endpoint(
    payload: ExtractKeywordsRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> ExtractKeywordsResponse:
    """Extract key terms from user text."""
    return await service.extract_keywords(payload.text, request.state.request_id)


@router.post("/translate", response_model=TranslateResponse)
async def translate(
    payload: TranslateRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> TranslateResponse:
    """Translate input text to a target language."""
    return await service.translate(
        payload.text,
        payload.target_language,
        request.state.request_id,
    )


@router.post("/analyze-text", response_model=AnalyzeTextResponse)
async def analyze_text_endpoint(
    payload: AnalyzeTextRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> AnalyzeTextResponse:
    """Combined analysis endpoint."""
    return await service.analyze_text(payload.text, request.state.request_id)


@router.post("/tool-assistant", response_model=ToolAssistantResponse)
async def tool_assistant(
    payload: ToolAssistantRequest,
    request: Request,
    service: AITasksService = Depends(get_ai_tasks_service),
) -> ToolAssistantResponse:
    """Let the model choose and call safe backend tools."""
    return await service.tool_assistant(payload.message, request.state.request_id)


@router.post("/research-agent", response_model=ResearchAgentResponse)
async def research_agent(
    payload: ResearchAgentRequest,
    request: Request,
    service: ResearchAgentService = Depends(get_research_agent_service),
) -> ResearchAgentResponse:
    """Run the multistep research agent and return a report plus its trace."""
    return await service.run(
        payload.topic,
        request.state.request_id,
        max_iterations=payload.max_iterations,
    )
