import asyncio

from agent.mcp_client import open_mcp_tools
from eval.cases import (
    LIT_TOOLS,
    ToolCount,
    ToolMatches,
    ToolNumber,
    load_cases,
    normalize_request,
)
from eval.checks import check_tool, field_value
from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR


async def test_eval_http_paths_replay_through_stdio(monkeypatch):
    monkeypatch.setenv("LIT_MCP_MODE", "live")
    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", "/not-the-recordings")
    replay_env = {
        "LIT_MCP_MODE": "replay",
        "LIT_MCP_RECORDINGS_DIR": str(DEFAULT_RECORDINGS_DIR),
        "NCBI_API_KEY": "",
    }
    cases_with_http = set()
    async with asyncio.timeout(30), open_mcp_tools(lit_env=replay_env) as tools:
        by_name = {tool.name: tool for tool in tools}
        for case in load_cases().cases:
            for path_index, path in enumerate(case.accepted_paths):
                evidence = {}
                for step in path:
                    if step.tool not in LIT_TOOLS or step.expected_error is not None:
                        continue
                    references = {
                        field: field_value(evidence[ref.step], ref.path)
                        for field, ref in step.from_results.items()
                    }
                    request = {**step.request, **references}
                    assert (
                        normalize_request(step.tool, request, None)
                        == step.normalized_request
                    )
                    cases_with_http.add(case.id)
                    result = await by_name[step.tool].ainvoke(
                        {
                            "type": "tool_call",
                            "name": step.tool,
                            "id": f"{case.id}-{path_index}-{step.id}",
                            "args": {"request": request},
                        }
                    )
                    assert result.status == "success", (case.id, step.id, result.text)
                    evidence[step.id] = result.artifact["structured_content"]
                for check in case.checks:
                    if (
                        isinstance(check, (ToolMatches, ToolCount, ToolNumber))
                        and check.step in evidence
                    ):
                        assert (failure := check_tool(check, evidence)) is None, (
                            case.id,
                            failure,
                        )

    assert cases_with_http == {"S03", "S04", "S05", "S06", "M01", "M02", "M03", "E03"}
