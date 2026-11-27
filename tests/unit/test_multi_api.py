import json
from contextlib import aclosing
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest
from test_multi_graph import config_for, request, task, turn

from agent.config import GraphSettings
from agent.graph import build_graph
from api.main import app, get_agent_graph
from api.schemas import RunRequest
from api.streaming import stream_run
from sandbox import artifacts
from sandbox.executor import PythonExecution
from sandbox.tool import create_run_python_tool

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


async def frames(graph, thread, message):
    async with aclosing(
        stream_run(graph, thread, RunRequest(message=message), trace_for_thread=None)
    ) as response:
        return [
            (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
            async for frame in response
            for event, data, *_ in [frame.splitlines()]
        ]


async def test_sse_streams_reporter_only_and_correlates_parallel_tool_results(
    task8_model, plan_reply, mcp_tool
):
    @mcp_tool("lit")
    def search_trials():
        """Return trials."""
        return "试验工具结果"

    @mcp_tool("chem")
    def compute_properties():
        """Return molecular properties."""
        return "分子工具结果"

    model = task8_model(
        disable_streaming=False,
        replies={
            "Planner": [
                plan_reply(
                    task("lit", "search_trials"), task("mol", "compute_properties")
                )
            ],
            "LitResearcher": [
                request(("search_trials", "same", {})),
                AIMessage(content="文献中间结论"),
            ],
            "MolAnalyst": [
                request(("compute_properties", "same", {})),
                AIMessage(content="分子中间结论"),
            ],
            "Reporter": [AIMessage(content="用户最终回答")],
        },
    )
    graph = build_graph(
        model,
        [search_trials, compute_properties],
        InMemorySaver(),
        settings=GraphSettings(mode="multi"),
    )
    thread = uuid4()
    events = await frames(graph, thread, "分别查证据和算性质")
    assert events[0] == ("node", {"node": "Planner"})
    assert [data["content"] for kind, data in events if kind == "token"] == [
        "用户最终回答"
    ]
    assert events[-1] == (
        "done",
        {"thread_id": str(thread), "final_response": "用户最终回答"},
    )
    assert [
        data["node"]
        for kind, data in events
        if kind == "node" and "subtask_id" not in data
    ] == ["Planner", "Critic", "Reporter"]

    for branch, node, tool_name, output in (
        ("lit", "LitResearcher", "search_trials", "试验工具结果"),
        ("mol", "MolAnalyst", "compute_properties", "分子工具结果"),
    ):
        own = [
            (kind, data) for kind, data in events if data.get("subtask_id") == branch
        ]
        assert [(kind, data.get("node")) for kind, data in own] == [
            ("node", node),
            ("node", "react"),
            ("tool_call", None),
            ("node", "tools"),
            ("tool_result", None),
            ("node", "react"),
            ("node", "finalize"),
        ]
        assert own[2][1] == {
            "subtask_id": branch,
            "tool_name": tool_name,
            "tool_call_id": "same",
            "args": {},
        }
        assert own[4][1] == {
            "subtask_id": branch,
            "tool_name": tool_name,
            "tool_call_id": "same",
            "status": "success",
            "output": output,
        }
        assert events.index(own[-1]) < events.index(("node", {"node": "Critic"}))


@pytest.fixture
async def history_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.parametrize("reporter_error", [False, True])
async def test_pending_or_failed_report_shows_current_user_and_ordered_available_tools(
    task8_model,
    plan_reply,
    mcp_tool,
    history_client,
    monkeypatch,
    reporter_error,
):
    @mcp_tool("lit")
    def search_trials():
        """Return a trial."""
        return "已保存文献结果"

    @mcp_tool("chem")
    def compute_properties():
        """Return a descriptor."""
        return "已保存分子结果"

    model = task8_model(
        replies={
            "Planner": [
                plan_reply(),
                plan_reply(
                    task("mol", "compute_properties"), task("lit", "search_trials")
                ),
            ],
            "LitResearcher": [
                request(("search_trials", "lit", {})),
                AIMessage(content="隐藏的文献结论"),
            ],
            "MolAnalyst": [
                request(("compute_properties", "mol", {})),
                AIMessage(content="隐藏的分子结论"),
            ],
            "Reporter": [
                AIMessage(content="之前的报告"),
                None,
            ],
        }
    )
    graph = build_graph(
        model,
        [search_trials, compute_properties],
        InMemorySaver(),
        settings=GraphSettings(mode="multi"),
    )
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    config = config_for(graph)
    thread_url = "/api/threads/" + config["configurable"]["thread_id"]
    await graph.ainvoke(turn("第一问"), config)
    if reporter_error:
        with pytest.raises(RuntimeError, match="Reporter 中断"):
            await graph.ainvoke(turn("第二问"), config)
    else:
        updates = graph.astream(
            turn("第二问"), config, stream_mode="updates", durability="sync"
        )
        async with aclosing(updates):
            async for update in updates:
                if "Critic" in update:
                    break

    history = (await history_client.get(thread_url)).raise_for_status().json()
    assert history["next_nodes"] == ["Reporter"]
    assert history["incomplete_turn"]
    assert history["step_count"] == 4
    assert [
        (message["role"], message["content"]) for message in history["messages"]
    ] == [
        ("human", "第一问"),
        ("ai", "之前的报告"),
        ("human", "第二问"),
        ("ai", ""),
        ("tool", "已保存分子结果"),
        ("ai", ""),
        ("tool", "已保存文献结果"),
    ]


async def test_completed_first_turn_chart_renders_after_second_turn_resets_results(
    task8_model,
    plan_reply,
    python_call,
    history_client,
    monkeypatch,
    tmp_path,
    png_bytes,
):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "artifacts")

    def execute(request, call_id):
        return PythonExecution("已绘图\n", "", 0, files={"first-turn.png": png_bytes})

    model = task8_model(
        replies={
            "Planner": [plan_reply(task("mol", "run_python")), plan_reply()],
            "MolAnalyst": [
                AIMessage(content="", tool_calls=[python_call("chart", "pass")]),
                AIMessage(content="分支图表已生成"),
            ],
            "Reporter": [
                AIMessage(content="第一轮报告"),
                AIMessage(content="第二轮报告"),
            ],
        }
    )
    graph = build_graph(
        model,
        [create_run_python_tool(execute)],
        InMemorySaver(),
        settings=GraphSettings(mode="multi"),
    )
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    config = config_for(graph)
    thread = config["configurable"]["thread_id"]
    first = await frames(graph, UUID(thread), "画一张图")
    (file,) = next(data["files"] for kind, data in first if kind == "tool_result")
    second = await graph.ainvoke(turn("继续对话"), config)
    assert second["molecule_results"] == []

    url = f"/api/threads/{thread}"
    history = (await history_client.get(url)).raise_for_status().json()
    artifact_url = f"{url}/artifacts/{file['artifact_id']}"
    contents = (await history_client.get(artifact_url)).raise_for_status().content
    assert contents == png_bytes
    assert not history["incomplete_turn"]
    (tool_record,) = [item for item in history["messages"] if item["role"] == "tool"]
    assert tool_record["files"] == [file]
    assert [
        message["content"]
        for message in history["messages"]
        if message["role"] == "human"
    ] == ["画一张图", "继续对话"]

    requested = []

    def api(request):
        path = request.url.path
        requested.append(path)
        if path == url:
            return httpx.Response(200, json=history)
        assert path == artifact_url
        return httpx.Response(
            200, content=contents, headers={"content-type": "image/png"}
        )

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(api))
    )
    page = AppTest.from_file(str(APP))
    page.query_params["thread_id"] = thread
    page.run()
    assert not page.exception
    assert requested == [url, artifact_url]
    (image,) = page.image
    assert image.captions == ["first-turn.png"]
    assert image.value[0]
