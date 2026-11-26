import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import _msgpack
from langgraph.graph import END, START, StateGraph
from mcp.client import stdio
from pydantic import BaseModel

from agent import cli, llm, mcp_client
from agent.config import AgentSettings, LLMSettings
from agent.graph import build_graph
from agent.mcp_client import open_mcp_tools
from agent.persistence import open_connection_pool, setup_checkpointer
from mcp_servers.chem.models import PropertiesResult
from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR
from mcp_servers.lit.models import TrialAdverseEventsResult


async def test_conversation_survives_new_pool_and_graph(
    database_url, run_config, fake_chat_model
):
    @tool
    def echo(text: str) -> str:
        """Return the supplied text."""
        return text

    first_model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "echo", "args": {"text": "蓝色"}, "id": "echo-1"}],
            ),
            AIMessage(content="我记住了蓝色。"),
        ]
    )
    async with open_connection_pool(database_url) as pool:
        graph = build_graph(first_model, [echo], await setup_checkpointer(pool))
        first = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="记住蓝色。")],
                "step_count": 0,
                "max_steps": 10,
            },
            run_config,
        )
        assert first["step_count"] == 2

    second_model = fake_chat_model(responses=[AIMessage(content="你说的是蓝色。")])
    async with open_connection_pool(database_url) as pool:
        graph = build_graph(second_model, [echo], await setup_checkpointer(pool))
        restored = await graph.aget_state(run_config)
        assert restored.values["messages"] == first["messages"]
        assert restored.next == ()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="我让你记住了什么颜色？")],
                "step_count": 0,
                "max_steps": 10,
            },
            run_config,
        )

    assert second_model.received[0][1:-1] == first["messages"]
    assert result["messages"][:-2] == first["messages"]
    assert result["messages"][-1].content == "你说的是蓝色。"
    assert result["step_count"] == 1


@pytest.mark.parametrize("transport_recovers", [True, False])
async def test_cli_resumes_pending_tool_before_reading_input(
    database_url, run_config, fake_chat_model, monkeypatch, capsys, transport_recovers
):
    connected = False
    executed = []

    @tool
    def echo(text: str) -> str:
        """Return the supplied text."""
        if not connected:
            raise ConnectionError("MCP transport disconnected")
        executed.append(text)
        return text

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "echo", "args": {"text": text}, "id": f"echo-{index}"}
            ],
        )
        for index, text in enumerate(("蓝色", "绿色", "红色"), start=1)
    ]
    first_model = fake_chat_model(responses=responses[:1])
    async with open_connection_pool(database_url) as pool:
        graph = build_graph(first_model, [echo], await setup_checkpointer(pool))
        with pytest.raises(ConnectionError, match="MCP transport disconnected"):
            await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="记住这三种颜色。")],
                    "step_count": 0,
                    "max_steps": 6,
                },
                run_config,
            )
        pending = await graph.aget_state(run_config)
        assert pending.next == ("tools",)
        assert pending.values["step_count"] == 1

    connected = transport_recovers
    resumed_model = fake_chat_model(
        responses=[*responses[1:], AIMessage(content="我记住了这三种颜色。")]
    )

    @asynccontextmanager
    async def fake_mcp_tools():
        yield [echo]

    prompts = []

    def end_input(prompt):
        prompts.append(prompt)
        assert executed == ["蓝色", "绿色", "红色"]
        raise EOFError

    monkeypatch.setattr(llm, "create_llm", lambda settings: resumed_model)
    monkeypatch.setattr(mcp_client, "open_mcp_tools", fake_mcp_tools)
    monkeypatch.setattr("builtins.input", end_input)
    settings = AgentSettings(
        database_url=database_url,
        llm=LLMSettings(
            base_url="https://example.invalid", api_key="test", model="mock"
        ),
    )
    thread_id = run_config["configurable"]["thread_id"]

    if transport_recovers:
        await cli.chat(settings, thread_id, max_steps=1)
        assert prompts == ["你> "]
        assert "我记住了这三种颜色。" in capsys.readouterr().out
    else:
        with pytest.raises(ConnectionError, match="MCP transport disconnected"):
            await cli.chat(settings, thread_id, max_steps=1)
        assert prompts == []
        assert resumed_model.received == []

    async with open_connection_pool(database_url) as pool:
        graph = build_graph(resumed_model, [echo], await setup_checkpointer(pool))
        latest = await graph.aget_state(run_config)
    if transport_recovers:
        assert latest.next == ()
        assert latest.values["max_steps"] == 6
        assert latest.values["step_count"] == 4
        assert latest.values["messages"][-1].content == "我记住了这三种颜色。"
    else:
        assert latest.next == ("tools",)
        assert latest.values["messages"] == pending.values["messages"]


class ParentPayload(BaseModel):
    label: str


class ChildPayload(BaseModel):
    count: int


class ParentState(TypedDict):
    parent: ParentPayload


class ChildState(ParentState):
    child: ChildPayload


def write_child(state: ChildState) -> dict[str, ChildPayload]:
    return {"child": ChildPayload(count=len(state["parent"].label))}


async def test_strict_msgpack_restores_typed_parent_and_child(database_url, run_config):
    assert os.environ["LANGGRAPH_STRICT_MSGPACK"] == "true"
    assert _msgpack.STRICT_MSGPACK_ENABLED
    child = StateGraph(ChildState)
    child.add_node("write", write_child)
    child.add_edge(START, "write")
    child.add_edge("write", END)

    parent = StateGraph(ParentState)
    parent.add_node("child_graph", child.compile(checkpointer=True))
    parent.add_edge(START, "child_graph")
    parent.add_edge("child_graph", END)

    payload = ParentPayload(label="test")
    async with open_connection_pool(database_url) as pool:
        graph = parent.compile(checkpointer=await setup_checkpointer(pool))
        await graph.ainvoke({"parent": payload}, run_config)

    async with open_connection_pool(database_url) as pool:
        graph = parent.compile(checkpointer=await setup_checkpointer(pool))
        parent_state = await graph.aget_state(run_config)
        child_state = await graph.aget_state(
            {
                "configurable": {
                    **run_config["configurable"],
                    "checkpoint_ns": "child_graph",
                }
            }
        )

    restored_parent = parent_state.values["parent"]
    restored_child = child_state.values["child"]
    assert isinstance(restored_parent, ParentPayload)
    assert restored_parent == payload
    assert isinstance(restored_child, ChildPayload)
    assert restored_child == ChildPayload(count=4)


@pytest.mark.parametrize("explicit_env", [True, False])
async def test_mcp_graph_reuses_stdio_processes_and_replays(
    fake_chat_model, monkeypatch, tmp_path, run_config, explicit_env
):
    lit_variables = {"LIT_MCP_MODE", "LIT_MCP_RECORDINGS_DIR", "NCBI_API_KEY"}
    for name in lit_variables:
        monkeypatch.delenv(name, raising=False)
    if explicit_env:
        recordings_dir = tmp_path / "recordings"
        shutil.copytree(DEFAULT_RECORDINGS_DIR, recordings_dir)
        monkeypatch.setenv("LIT_MCP_MODE", "replay")
        monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(recordings_dir))
        monkeypatch.setenv("NCBI_API_KEY", "")

    starts: list[tuple[str, int]] = []
    start_process = stdio._create_platform_compatible_process

    async def track_process(*, command, args, env, errlog, cwd):
        if args[-1] == "mcp_servers.lit.server":
            if explicit_env:
                assert env["LIT_MCP_MODE"] == "replay"
                assert env["LIT_MCP_RECORDINGS_DIR"] == str(recordings_dir)
                assert env["NCBI_API_KEY"] == ""
            else:
                assert lit_variables.isdisjoint(env)
        process = await start_process(
            command=command, args=args, env=env, errlog=errlog, cwd=cwd
        )
        starts.append((args[-1], process.pid))
        return process

    monkeypatch.setattr(stdio, "_create_platform_compatible_process", track_process)

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "compute_properties",
                        "args": {"request": {"smiles": ["CCO"]}},
                        "id": f"chem-{index}",
                    },
                    {
                        "name": "get_trial_adverse_events",
                        "args": {
                            "request": {
                                "nct_id": "NCT04303780",
                                "category": "serious",
                                "term": "Alanine aminotransferase increased",
                            }
                        },
                        "id": f"lit-{index}",
                    },
                ],
            )
            for index in (1, 2)
        ]
        + [AIMessage(content="已查询试验记录和分子性质。")]
    )

    async with asyncio.timeout(30), open_mcp_tools() as tools:
        graph = build_graph(model, tools, InMemorySaver())
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="查询试验记录和分子性质。")],
                "step_count": 0,
                "max_steps": 5,
            },
            run_config,
        )

    assert [name for name, _ in starts] == [
        "mcp_servers.chem.server",
        "mcp_servers.lit.server",
    ]
    assert len({pid for _, pid in starts}) == 2
    tool_results = {
        msg.tool_call_id: msg
        for msg in result["messages"]
        if isinstance(msg, ToolMessage)
    }
    assert set(tool_results) == {"chem-1", "chem-2", "lit-1", "lit-2"}
    assert all(msg.status == "success" for msg in tool_results.values())
    for index in (1, 2):
        properties = PropertiesResult.model_validate(
            tool_results[f"chem-{index}"].artifact["structured_content"]
        )
        assert properties.results[0].mw == pytest.approx(46.069, abs=0.01)
        events = TrialAdverseEventsResult.model_validate(
            tool_results[f"lit-{index}"].artifact["structured_content"]
        )
        assert events.nct_id == "NCT04303780"
        assert events.adverse_events.serious.total_count == 156
        assert events.adverse_events.serious.matched_count == 1
    assert result["messages"][-1].content == "已查询试验记录和分子性质。"
