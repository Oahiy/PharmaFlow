import asyncio
from contextlib import AsyncExitStack
from uuid import uuid4

import httpx
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import SecretStr

from agent.graph import build_graph
from agent.mcp_client import open_mcp_tools
from agent.persistence import open_connection_pool, setup_checkpointer
from api.main import router
from api.uploads import UploadSettings
from mcp_servers.chem.models import PropertiesResult


async def test_workers_reject_overlapping_runs_and_release_lock_on_cancel(
    database_url, run_config, fake_chat_model
):
    entered, finish = asyncio.Event(), asyncio.Event()

    @tool
    async def wait_for_result() -> str:
        """Wait until the test allows the tool to finish."""
        entered.set()
        await finish.wait()
        return "结果"

    first_reply = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "wait_for_result",
                "args": {},
                "id": "wait-1",
            }
        ],
    )
    url = f"/api/threads/{run_config['configurable']['thread_id']}"
    async with asyncio.timeout(15), AsyncExitStack() as stack:
        clients = []
        for replies in (
            [first_reply],
            [AIMessage(content=t) for t in ("独立完成", "恢复完成", "新轮完成")],
        ):
            pool = await stack.enter_async_context(open_connection_pool(database_url))
            worker = FastAPI()
            worker.include_router(router)
            worker.state.database_url = SecretStr(database_url)
            worker.state.graph = build_graph(
                fake_chat_model(responses=replies),
                [wait_for_result],
                await setup_checkpointer(pool),
            ).with_config(
                {"configurable": {"user_id": run_config["configurable"]["user_id"]}}
            )
            clients.append(
                await stack.enter_async_context(
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=worker),
                        base_url="http://test",
                    )
                )
            )
        first, second = clients
        async with asyncio.TaskGroup() as tasks:
            active = tasks.create_task(
                first.post(url + "/runs", json={"message": "开始"})
            )
            await entered.wait()
            conflict = await second.post(url + "/runs", json={"message": "冲突"})
            assert conflict.status_code == 409
            assert "正在运行" in conflict.json()["detail"]
            other = await second.post(
                f"/api/threads/{uuid4()}/runs", json={"message": "独立"}
            )
            assert "event: done" in other.raise_for_status().text
            active.cancel()

        pending = (await second.get(url)).raise_for_status().json()
        assert pending["next_nodes"] == ["tools"]
        finish.set()
        resumed = await second.post(url + "/runs", json={"message": "继续"})
        assert "event: done" in resumed.raise_for_status().text
        state = (await second.get(url)).json()
        assert state["next_nodes"] == []
        assert [m["content"] for m in state["messages"] if m["role"] == "human"] == [
            "开始",
            "继续",
        ]
        assert [m["role"] for m in state["messages"]] == [
            "human",
            "ai",
            "tool",
            "ai",
            "human",
            "ai",
        ]


async def test_upload_readable_in_existing_stdio_session(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("LIT_MCP_MODE", "replay")
    worker = FastAPI()
    worker.include_router(router)
    worker.state.upload_settings = UploadSettings.from_env()
    async with (
        asyncio.timeout(30),
        open_mcp_tools() as tools,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=worker), base_url="http://test"
        ) as client,
    ):
        upload = await client.post(
            "/api/datasets/upload",
            files={"file": ("mols.csv", b"smiles\nCCO\n", "text/csv")},
        )
        dataset_id = upload.raise_for_status().json()["dataset_id"]
        compute = next(t for t in tools if t.name == "compute_properties")
        result = await compute.ainvoke(
            {
                "type": "tool_call",
                "name": compute.name,
                "id": "upload-test",
                "args": {"request": {"dataset_id": dataset_id}},
            }
        )
        assert result.status == "success"
        rows = PropertiesResult.model_validate(
            result.artifact["structured_content"]
        ).results
        assert [(row.smiles, round(row.mw, 3)) for row in rows] == [("CCO", 46.069)]
