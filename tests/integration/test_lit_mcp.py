import asyncio
import logging
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from fastmcp import Client
from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR
from mcp_servers.lit.http import UpstreamTransport, make_client
from mcp_servers.lit.models import LabelSectionResult, TrialAdverseEventsResult
from mcp_servers.lit.server import create_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def test_http_status_error_reaches_mcp_client_without_api_key(caplog):
    fake_key = "pharmaflow-rejected fake key/+only-for-tests"
    caplog.set_level(logging.INFO)

    def reject(request):
        assert request.url.params["api_key"] == fake_key
        return httpx.Response(
            401,
            json={
                "reason": "NCBI rejected the supplied credential",
                "url": str(request.url),
                "credential": fake_key,
            },
        )

    mcp = create_server(
        lambda: make_client(
            UpstreamTransport(httpx.MockTransport(reject), api_key=fake_key)
        )
    )
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_pubmed", {"request": {"query": "KRAS"}}, raise_on_error=False
        )

    assert result.is_error
    message = result.content[0].text
    assert "401 Unauthorized" in message
    assert "esearch.fcgi" in message
    assert "NCBI rejected the supplied credential" in message
    assert "[redacted]" in message
    for secret in (fake_key, quote_plus(fake_key), "api_key"):
        assert secret not in message
        assert secret not in caplog.text


def _stdio_client(cwd: Path, recordings_dir: Path) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "lit": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_servers.lit.server"],
                "cwd": str(cwd),
                "env": {
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "FASTMCP_ENV_FILE": os.devnull,
                    "LIT_MCP_MODE": "replay",
                    "LIT_MCP_RECORDINGS_DIR": str(recordings_dir),
                    "NCBI_API_KEY": "",
                },
            }
        }
    )


async def test_codebreak_200_serious_events_over_stdio(tmp_path):
    client = _stdio_client(tmp_path, DEFAULT_RECORDINGS_DIR)
    term = "ALANINE aminotransferase increased"
    async with asyncio.timeout(30), client.session("lit") as session:
        tools = await session.list_tools()
        assert {tool.name for tool in tools.tools} == {
            "search_pubmed",
            "search_trials",
            "get_trial_adverse_events",
            "get_label_section",
        }
        for tool in tools.tools:
            assert tool.inputSchema["required"] == ["request"]
            assert list(tool.inputSchema["properties"]) == ["request"]
            assert tool.outputSchema

        result = await session.call_tool(
            "get_trial_adverse_events",
            {"request": {"nct_id": "NCT04303780", "category": "serious"}},
        )
        filtered = await session.call_tool(
            "get_trial_adverse_events",
            {"request": {"nct_id": "NCT04303780", "category": "serious", "term": term}},
        )

    assert not result.isError, result.content
    assert result.structuredContent["adverse_events"]["frequencyThreshold"] == "5"
    assert "timeFrame" in result.structuredContent["adverse_events"]
    events = TrialAdverseEventsResult.model_validate(result.structuredContent)
    assert events.nct_id == "NCT04303780"
    assert events.has_results
    assert events.adverse_events.serious.total_count == 156
    assert events.adverse_events.serious.returned_count == 50
    assert events.adverse_events.groups.results[0].serious_num_at_risk == 151
    assert events.adverse_events.other is None
    assert not filtered.isError, filtered.content
    filtered_events = TrialAdverseEventsResult.model_validate(
        filtered.structuredContent
    )
    assert filtered_events.term == term
    serious = filtered_events.adverse_events.serious
    assert (serious.total_count, serious.matched_count, serious.returned_count) == (
        156,
        1,
        1,
    )
    assert [event.term for event in serious.results] == [
        "Alanine aminotransferase increased"
    ]
    assert filtered_events.adverse_events.other is None


async def test_sotorasib_label_tables_over_stdio(tmp_path):
    client = _stdio_client(tmp_path, DEFAULT_RECORDINGS_DIR)
    async with asyncio.timeout(30), client.session("lit") as session:
        result = await session.call_tool(
            "get_label_section",
            {
                "request": {
                    "drug": "sotorasib",
                    "fields": ["adverse_reactions_table", "boxed_warning"],
                }
            },
        )

    assert not result.isError, result.content
    assert (
        result.structuredContent["results"][0]["sections"]["results"][1]["content"]
        is None
    )
    (label,) = LabelSectionResult.model_validate(result.structuredContent).results
    tables, warning = label.sections.results
    assert tables.status == "present"
    assert tables.content.returned_count == 4
    assert tables.content.results[0].text.startswith("<table")
    assert not tables.content.results[0].truncated
    assert warning.status == "missing"


async def test_replay_miss_reaches_stdio_client(tmp_path):
    client = _stdio_client(tmp_path, tmp_path)
    async with asyncio.timeout(30), client.session("lit") as session:
        result = await session.call_tool(
            "get_trial_adverse_events", {"request": {"nct_id": "NCT04303780"}}
        )

    assert result.isError
    assert "回放未命中: GET" in result.content[0].text
    assert (
        "https://clinicaltrials.gov/api/v2/studies/NCT04303780"
        in result.content[0].text
    )
