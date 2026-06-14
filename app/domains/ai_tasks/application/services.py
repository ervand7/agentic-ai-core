from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Optional

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.domains.ai_tasks.api.schemas import (
    AnalyzeTextResponse,
    AskResponse,
    ClassifyResponse,
    ExtractKeywordsResponse,
    SummarizeResponse,
    ToolAssistantResponse,
    ToolExecutionResult,
    TranslateResponse,
)
from app.domains.ai_tasks.application.conversation import (
    assistant_tool_call_message,
    parse_json_content,
    tool_result_message,
)
from app.domains.ai_tasks.application.tool_runner import ToolRunner
from app.domains.ai_tasks.constants import Endpoint
from app.domains.ai_tasks.domain.prompts import get_prompts
from app.domains.ai_tasks.domain.ports import LLMProvider
from app.domains.ai_tasks.domain.response_formats import (
    ANALYZE_TEXT_RESPONSE_FORMAT,
    CLASSIFY_RESPONSE_FORMAT,
    KEYWORDS_RESPONSE_FORMAT,
)
from app.domains.ai_tasks.domain.tools import TOOL_DEFINITIONS
from app.domains.documents.application.services import SearchDocumentsService
from app.shared.config import Settings
from app.shared.openai_types import ChatCompletionMessageParam


def _build_messages(
        user_input: str, prompt_key: str
) -> list[ChatCompletionMessageParam]:
    """Create system+user messages from the prompt registry."""
    prompt = get_prompts()[prompt_key]
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": prompt.system_prompt,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_input,
    }
    return [system_message, user_message]


class AITasksService:
    """All AI task use cases grouped behind a single application service."""

    def __init__(
            self,
            llm_provider: LLMProvider,
            settings: Settings,
            document_search: Optional[SearchDocumentsService] = None,
            tool_runner: Optional[ToolRunner] = None,
    ):
        self._llm = llm_provider
        self._settings = settings
        self._tool_runner = tool_runner or ToolRunner(document_search=document_search)

    async def ask(self, question: str, request_id: str) -> AskResponse:
        """Simple question answering use case."""
        result = await self._llm.complete(
            endpoint=Endpoint.ASK,
            request_id=request_id,
            messages=_build_messages(question, "ask_v1"),
        )
        return AskResponse(
            answer=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def ask_stream(
            self, question: str, request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream tokens token-by-token for chat-like UIs."""
        async for chunk in self._llm.stream(
                endpoint=Endpoint.ASK_STREAM,
                request_id=request_id,
                messages=_build_messages(question, "ask_stream_v1"),
        ):
            if chunk.content:
                yield chunk.content

    async def classify(self, text: str, request_id: str) -> ClassifyResponse:
        """Classify sentiment and return JSON validated with Pydantic."""
        result = await self._llm.complete(
            endpoint=Endpoint.CLASSIFY,
            request_id=request_id,
            messages=_build_messages(text, "classify_v1"),
            temperature=self._settings.OPENAI_TEMPERATURE_CLASSIFY,
            response_format=CLASSIFY_RESPONSE_FORMAT,
        )
        raw = parse_json_content(result.content)
        return ClassifyResponse.model_validate(raw)

    async def summarize(self, text: str, request_id: str) -> SummarizeResponse:
        """Summarize text into a short answer."""
        result = await self._llm.complete(
            endpoint=Endpoint.SUMMARIZE,
            request_id=request_id,
            messages=_build_messages(text, "summarize_v1"),
            temperature=self._settings.OPENAI_TEMPERATURE_SUMMARIZE,
        )
        return SummarizeResponse(
            summary=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def extract_keywords(
            self, text: str, request_id: str
    ) -> ExtractKeywordsResponse:
        """Extract relevant terms as a JSON list."""
        result = await self._llm.complete(
            endpoint=Endpoint.EXTRACT_KEYWORDS,
            request_id=request_id,
            messages=_build_messages(text, "extract_keywords_v1"),
            temperature=self._settings.OPENAI_TEMPERATURE_EXTRACT_KEYWORDS,
            response_format=KEYWORDS_RESPONSE_FORMAT,
        )
        raw_data = parse_json_content(result.content)
        return ExtractKeywordsResponse(
            keywords=raw_data.get("keywords", []),
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def translate(
            self, text: str, target_language: str, request_id: str
    ) -> TranslateResponse:
        """Translate text to a target language."""
        prompt = get_prompts()["translate_v1"]
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": (
                f"{prompt.system_prompt} Translate to {target_language}. "
                "Return only the translation."
            ),
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": text,
        }
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]
        result = await self._llm.complete(
            endpoint=Endpoint.TRANSLATE,
            request_id=request_id,
            messages=messages,
            temperature=self._settings.OPENAI_TEMPERATURE_TRANSLATE,
        )
        return TranslateResponse(
            translation=result.content,
            model=result.model,
            tokens_used=result.tokens_used,
        )

    async def analyze_text(self, text: str, request_id: str) -> AnalyzeTextResponse:
        """
        Combined use case for summary, sentiment, keywords, and language.

        Uses one structured-output call to keep latency and token usage lower.
        """
        result = await self._llm.complete(
            endpoint=Endpoint.ANALYZE_TEXT,
            request_id=request_id,
            messages=_build_messages(text, "analyze_text_v1"),
            temperature=self._settings.OPENAI_TEMPERATURE_ANALYZE_TEXT,
            response_format=ANALYZE_TEXT_RESPONSE_FORMAT,
        )
        raw_data = parse_json_content(result.content)
        return AnalyzeTextResponse.model_validate(
            {
                **raw_data,
                "model": result.model,
                "tokens_used": result.tokens_used,
                "prompt_version": get_prompts()["analyze_text_v1"].prompt_version,
            }
        )

    async def tool_assistant(
            self, message: str, request_id: str
    ) -> ToolAssistantResponse:
        """Let the model choose safe backend tools, then summarize their results."""
        messages = _build_messages(message, "tool_assistant_v1")
        first = await self._llm.complete_with_tools(
            endpoint=Endpoint.TOOL_ASSISTANT,
            request_id=request_id,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=0.0,
        )

        if not first.tool_calls:
            return ToolAssistantResponse(
                answer=first.content,
                tool_calls=[],
                model=first.model,
                tokens_used=first.tokens_used,
                prompt_version=get_prompts()["tool_assistant_v1"].prompt_version,
            )

        executed_tools: list[ToolExecutionResult] = []
        follow_up_messages: list[ChatCompletionMessageParam] = [
            *messages,
            assistant_tool_call_message(first),
        ]

        for call in first.tool_calls:
            execution = await self._tool_runner.execute(call, request_id=request_id)
            executed_tools.append(execution)
            follow_up_messages.append(tool_result_message(call.id, execution.result))

        final = await self._llm.complete(
            endpoint=Endpoint.TOOL_ASSISTANT,
            request_id=request_id,
            messages=follow_up_messages,
            temperature=0.0,
        )
        return ToolAssistantResponse(
            answer=final.content,
            tool_calls=executed_tools,
            model=final.model,
            tokens_used=first.tokens_used + final.tokens_used,
            prompt_version=get_prompts()["tool_assistant_v1"].prompt_version,
        )
