"""Production-style runner for model-requested tool calls."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Optional

from pydantic import ValidationError

from app.domains.ai_tasks.api.schemas import ToolExecutionResult
from app.domains.ai_tasks.domain.ports import ToolCall
from app.domains.ai_tasks.domain.tools import (
    TOOL_REGISTRY,
    CreateTicketArgs,
    GetWeatherArgs,
    SearchDocsArgs,
    SendEmailDraftArgs,
    ToolSpec,
)
from app.domains.documents.application.services import (
    EmptyVectorStoreError,
    SearchDocumentsService,
)
from app.shared.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

GET_WEATHER_TOOL_NAME = "get_weather"
SEARCH_DOCS_TOOL_NAME = "search_docs"
CREATE_TICKET_TOOL_NAME = "create_ticket"
SEND_EMAIL_DRAFT_TOOL_NAME = "send_email_draft"
HASH_HEX_BASE = 16
WEATHER_BASE_CELSIUS = 8
WEATHER_VARIATION_RANGE = 24
FAHRENHEIT_MULTIPLIER = 9
FAHRENHEIT_DIVISOR = 5
FAHRENHEIT_OFFSET = 32
TEMPERATURE_DECIMAL_PLACES = 1
DRAFT_ID_MODULUS = 100_000
DRAFT_ID_WIDTH = 5


@dataclass(frozen=True)
class ToolAuditRecord:
    """Structured audit data for one tool execution attempt."""

    request_id: str
    tool_call_id: str
    tool_name: str
    risk: str
    requires_approval: bool
    approved: bool
    status: str
    error: Optional[str] = None


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        raw = json.loads(arguments or "{}")
    except JSONDecodeError as exc:
        raise LLMServiceError("Model returned invalid tool arguments.") from exc
    if not isinstance(raw, dict):
        raise LLMServiceError("Model returned non-object tool arguments.")
    return raw


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), HASH_HEX_BASE)


def _limit_result(result: dict[str, Any], max_chars: int) -> dict[str, Any]:
    encoded = json.dumps(result)
    if len(encoded) <= max_chars:
        return result
    return {
        "truncated": True,
        "max_chars": max_chars,
        "preview": encoded[:max_chars],
    }


class ToolRunner:
    """Validate, authorize, execute, and audit tool calls."""

    def __init__(
            self,
            *,
            document_search: Optional[SearchDocumentsService] = None,
            registry: Optional[dict[str, ToolSpec]] = None,
    ):
        self._document_search = document_search
        self._registry = registry or TOOL_REGISTRY

    async def execute(
            self,
            call: ToolCall,
            *,
            request_id: str,
            approved_tool_call_ids: Optional[set[str]] = None,
    ) -> ToolExecutionResult:
        spec = self._registry.get(call.name)
        if spec is None:
            return self._audit_result(
                call=call,
                request_id=request_id,
                arguments={},
                result={"error": f"Unknown tool: {call.name}"},
                risk="unknown",
                requires_approval=False,
                approved=False,
                status="rejected",
                error=f"Unknown tool: {call.name}",
            )

        approved = not spec.requires_approval or (
                approved_tool_call_ids is not None and call.id in approved_tool_call_ids
        )
        if not approved:
            return self._audit_result(
                call=call,
                request_id=request_id,
                arguments={},
                result={
                    "error": "Tool requires human approval before execution.",
                    "approval_required": True,
                },
                risk=spec.risk.value,
                requires_approval=spec.requires_approval,
                approved=False,
                status="approval_required",
                error="approval_required",
            )

        arguments: dict[str, Any] = {}
        try:
            arguments = _parse_tool_arguments(call.arguments)
            result = await self._run(call.name, arguments, request_id)
            result = _limit_result(result, spec.max_result_chars)
            return self._audit_result(
                call=call,
                request_id=request_id,
                arguments=arguments,
                result=result,
                risk=spec.risk.value,
                requires_approval=spec.requires_approval,
                approved=True,
                status="executed",
            )
        except (LLMServiceError, ValidationError) as exc:
            return self._audit_result(
                call=call,
                request_id=request_id,
                arguments=arguments,
                result={"error": str(exc)},
                risk=spec.risk.value,
                requires_approval=spec.requires_approval,
                approved=True,
                status="failed",
                error=str(exc),
            )

    async def _run(
            self, name: str, arguments: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        if name == GET_WEATHER_TOOL_NAME:
            weather_args = GetWeatherArgs.model_validate(arguments)
            base_celsius = WEATHER_BASE_CELSIUS + (
                    _stable_int(weather_args.location.lower()) % WEATHER_VARIATION_RANGE
            )
            temperature = (
                base_celsius
                if weather_args.unit == "celsius"
                else round(
                    (base_celsius * FAHRENHEIT_MULTIPLIER / FAHRENHEIT_DIVISOR)
                    + FAHRENHEIT_OFFSET,
                    TEMPERATURE_DECIMAL_PLACES,
                )
            )
            return {
                "location": weather_args.location,
                "temperature": temperature,
                "unit": weather_args.unit,
                "condition": "partly cloudy",
                "source": "mock_weather_tool",
            }

        if name == SEARCH_DOCS_TOOL_NAME:
            search_args = SearchDocsArgs.model_validate(arguments)
            if self._document_search is None:
                return {
                    "error": "Document search service is not configured.",
                    "results": [],
                }
            try:
                response = await self._document_search.execute(
                    query=search_args.query,
                    top_k=search_args.top_k,
                    filename=search_args.filename,
                    keyword=search_args.keyword,
                    min_similarity=search_args.min_similarity,
                    request_id=request_id,
                )
            except EmptyVectorStoreError as exc:
                return {"error": str(exc), "results": []}
            return {
                "query": response.query,
                "results": [
                    {
                        "text": result.text,
                        "filename": result.filename,
                        "similarity": result.similarity,
                    }
                    for result in response.results
                ],
            }

        if name == CREATE_TICKET_TOOL_NAME:
            ticket_args = CreateTicketArgs.model_validate(arguments)
            draft_number = _stable_int(ticket_args.title) % DRAFT_ID_MODULUS
            draft_id = f"TICKET-DRAFT-{draft_number:0{DRAFT_ID_WIDTH}d}"
            return {
                "draft_id": draft_id,
                "title": ticket_args.title,
                "description": ticket_args.description,
                "priority": ticket_args.priority,
                "status": "draft_created",
                "requires_human_confirmation": True,
            }

        if name == SEND_EMAIL_DRAFT_TOOL_NAME:
            email_args = SendEmailDraftArgs.model_validate(arguments)
            draft_number = (
                _stable_int(email_args.to + email_args.subject) % DRAFT_ID_MODULUS
            )
            draft_id = f"EMAIL-DRAFT-{draft_number:0{DRAFT_ID_WIDTH}d}"
            return {
                "draft_id": draft_id,
                "to": email_args.to,
                "subject": email_args.subject,
                "body": email_args.body,
                "status": "draft_created",
                "sent": False,
                "requires_human_confirmation": True,
            }

        return {"error": f"Unknown tool: {name}"}

    def _audit_result(
            self,
            *,
            call: ToolCall,
            request_id: str,
            arguments: dict[str, Any],
            result: dict[str, Any],
            risk: str,
            requires_approval: bool,
            approved: bool,
            status: str,
            error: Optional[str] = None,
    ) -> ToolExecutionResult:
        audit = ToolAuditRecord(
            request_id=request_id,
            tool_call_id=call.id,
            tool_name=call.name,
            risk=risk,
            requires_approval=requires_approval,
            approved=approved,
            status=status,
            error=error,
        )
        logger.info("tool_execution_audit %s", audit)
        return ToolExecutionResult(
            name=call.name,
            arguments=arguments,
            result=result,
            tool_call_id=call.id,
            risk=risk,
            requires_approval=requires_approval,
            approved=approved,
            status=status,
            error=error,
        )
