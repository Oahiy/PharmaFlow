import json
from contextlib import aclosing

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde import _msgpack

from agent.config import GraphSettings
from agent.graph import build_graph, resume_config, turn_recursion_limit
from agent.persistence import open_connection_pool, setup_checkpointer
from agent.state import HistorySummary, Subtask, WorkerResult


async def test_pending_multi_turn_and_typed_results_survive_new_postgres_pool(
    database_url,
    run_config,
    task8_model,
    plan_reply,
    mcp_tool,
):
    @mcp_tool("lit", response_format="content_and_artifact")
    def search_evidence(index: int):
        """Return replayed excerpts."""
        payload = {
            "results": [
                {
                    "pmid": "36764316",
                    "text": "较长的文章原文。" * 900
                    if index == 0
                    else "试验见 NCT04303780。",
                }
            ]
        }
        return json.dumps(payload, ensure_ascii=False), {"structured_content": payload}

    async def build(pool, model, summary_model):
        return build_graph(
            model,
            [search_evidence],
            await setup_checkpointer(pool),
            settings=GraphSettings(mode="multi", summarize_history=True),
            summary_model=summary_model,
        )

    assert _msgpack.STRICT_MSGPACK_ENABLED
    plan = plan_reply(
        {
            "id": "research",
            "kind": "lit",
            "goal": "核对证据",
            "tools": ["search_evidence"],
        },
        {"id": "molecules", "kind": "mol", "goal": "记录分子需求", "tools": []},
    )
    async with open_connection_pool(database_url) as pool:
        graph = await build(
            pool, task8_model(replies={"Planner": [plan]}), task8_model(replies={})
        )
        config = {**run_config, "recursion_limit": turn_recursion_limit(graph, 4)}
        stream = graph.astream(
            {
                "messages": [HumanMessage(content="核查证据，并评估候选集。")],
                "max_steps": 4,
                "step_count": 0,
            },
            config,
            stream_mode="updates",
            durability="sync",
        )
        async with aclosing(stream):
            assert "Planner" in await anext(stream)

    research_replies = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": f"source-{index}",
                    "name": "search_evidence",
                    "args": {"index": index},
                }
            ],
        )
        for index in range(3)
    ]
    model = task8_model(
        replies={
            "LitResearcher": [
                *research_replies,
                AIMessage(content="PMID 36764316 和 NCT04303780。"),
            ],
            "MolAnalyst": [AIMessage(content="已记录候选集。")],
            "Reporter": [AIMessage(content="证据和分子结果。")],
        }
    )
    summary = task8_model(replies={"summary": [AIMessage(content="保留文献出处。")]})
    async with open_connection_pool(database_url) as pool:
        graph = await build(pool, model, summary)
        pending = await graph.aget_state(run_config)
        assert set(pending.next) == {"LitResearcher", "MolAnalyst"}
        assert all(type(task) is Subtask for task in pending.values["tasks"])
        state = await graph.ainvoke(
            None, await resume_config(graph, run_config, pending)
        )
        assert state["step_count"] == 5
        assert state["issues"] == []
        assert len(summary.received["summary"]) == 1

    async with open_connection_pool(database_url) as pool:
        graph = await build(pool, task8_model(replies={}), task8_model(replies={}))
        restored = await graph.aget_state(run_config)
    assert restored.next == ()
    (result,) = restored.values["evidence_results"]
    assert type(result) is WorkerResult
    assert type(result.summary) is HistorySummary
    assert result.summary.ids == ["PMID:36764316"]
    assert [type(message) for message in result.messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert result == state["evidence_results"][0]
    assert type(restored.values["molecule_results"][0]) is WorkerResult
    assert restored.values["messages"] == state["messages"]
