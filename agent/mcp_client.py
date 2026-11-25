"""Keep the two stdio MCP sessions open for the lifetime of the agent."""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import PROJECT_ROOT


@asynccontextmanager
async def open_mcp_tools() -> AsyncIterator[list[BaseTool]]:
    stdio_env = {
        "PATH": os.pathsep.join(os.get_exec_path()),
        "PYTHONPATH": str(PROJECT_ROOT),
        "FASTMCP_ENV_FILE": os.devnull,
    }
    lit_env = {
        name: os.environ[name]
        for name in ("LIT_MCP_MODE", "LIT_MCP_RECORDINGS_DIR", "NCBI_API_KEY")
        if name in os.environ
    }
    client = MultiServerMCPClient(
        {
            "chem": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_servers.chem.server"],
                "cwd": str(PROJECT_ROOT),
                "env": stdio_env,
            },
            "lit": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_servers.lit.server"],
                "cwd": str(PROJECT_ROOT),
                "env": {**stdio_env, **lit_env},
            },
        }
    )
    async with (
        client.session("chem") as chem_session,
        client.session("lit") as lit_session,
    ):
        chem_tools = await load_mcp_tools(chem_session)
        lit_tools = await load_mcp_tools(lit_session)
        yield [*chem_tools, *lit_tools]
