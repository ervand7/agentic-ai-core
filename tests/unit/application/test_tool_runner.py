"""Unit tests for production-style tool execution policy."""

import json

from app.domains.ai_tasks.application.tool_runner import ToolRunner
from app.domains.ai_tasks.domain.ports import ToolCall
from app.domains.ai_tasks.domain.tools import ToolRisk, ToolSpec


class TestToolRunner:
    async def test_executes_registered_tool_with_policy_metadata(self):
        runner = ToolRunner()
        result = await runner.execute(
            ToolCall(
                id="call_1",
                name="get_weather",
                arguments=json.dumps({"location": "Yerevan"}),
            ),
            request_id="req-1",
        )

        assert result.name == "get_weather"
        assert result.tool_call_id == "call_1"
        assert result.risk == "read_only"
        assert result.status == "executed"
        assert result.approved is True
        assert result.result["location"] == "Yerevan"

    async def test_rejects_unknown_tool(self):
        runner = ToolRunner()
        result = await runner.execute(
            ToolCall(id="call_1", name="delete_everything", arguments="{}"),
            request_id="req-1",
        )

        assert result.status == "rejected"
        assert result.approved is False
        assert result.error == "Unknown tool: delete_everything"

    async def test_blocks_approval_required_tool_until_approved(self):
        registry = {
            "get_weather": ToolSpec(
                name="get_weather",
                definition={},
                risk=ToolRisk.DANGEROUS,
                requires_approval=True,
            )
        }
        runner = ToolRunner(registry=registry)
        result = await runner.execute(
            ToolCall(
                id="call_1",
                name="get_weather",
                arguments=json.dumps({"location": "Yerevan"}),
            ),
            request_id="req-1",
        )

        assert result.status == "approval_required"
        assert result.requires_approval is True
        assert result.approved is False
        assert result.result["approval_required"] is True
