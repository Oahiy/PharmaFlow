import asyncio
import json
from contextlib import aclosing
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.config import GraphSettings
from agent.graph import (
    PARALLEL_SANDBOX_MESSAGE,
    STEP_LIMIT_MESSAGE,
    build_graph,
    resume_config,
    turn_recursion_limit,
)
from agent.references import observed_ids
from agent.state import ordered_results
from sandbox.executor import PythonExecution
from sandbox.models import MAX_REPAIR_ROUNDS, REPAIR_LIMIT_MESSAGE, SandboxError
from sandbox.tool import create_run_python_tool


def task(kind, *tools, task_id=None):
    return {
        "id": kind if task_id is None else task_id,
        "kind": kind,
        "goal": "执行该部分",
        "tools": list(tools),
    }


def request(*calls):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "id": identifier, "args": args}
            for name, identifier, args in calls
        ],
    )


def turn(text, steps=10):
    return {
        "messages": [HumanMessage(content=text)],
        "max_steps": steps,
        "step_count": 0,
    }


def payload(system, kind):
    return json.loads(
        system.text.split(f'<data kind="{kind}">', 1)[1].split("</data>", 1)[0]
    )


def config_for(graph, steps=10):
    return {
        "configurable": {"thread_id": str(uuid4()), "user_id": str(uuid4())},
        "recursion_limit": turn_recursion_limit(graph, steps),
    }


async def test_parallel_workers_share_results_and_reset_next_turn(
    task8_model, plan_reply, mcp_tool
):
    lit_started, mol_started = asyncio.Event(), asyncio.Event()
    evidence = {
        "request": {"query": "NCT00000000"},
        "results": [
            {
                "pmid": "36764316",
                "citation_id": "pubmed:36764316:1",
                "source_id": "36764316",
                "text": "registered at ClinicalTrials.gov, NCT04303780",
            }
        ],
    }

    @mcp_tool("lit", response_format="content_and_artifact")
    async def search_evidence():
        """Look up one evidence passage."""
        lit_started.set()
        await asyncio.wait_for(mol_started.wait(), 2)
        return json.dumps(evidence), {"structured_content": evidence}

    @mcp_tool("chem")
    async def compute_properties():
        """Compute descriptors."""
        mol_started.set()
        await asyncio.wait_for(lit_started.wait(), 2)
        return "TPSA 20.23"

    @mcp_tool("chem")
    def other():
        """This tool should not be bound."""
        raise AssertionError("not selected")

    model = task8_model(
        replies={
            "Planner": [
                plan_reply(
                    task("mol", "compute_properties"), task("lit", "search_evidence")
                ),
                plan_reply(),
            ],
            "LitResearcher": [
                request(("search_evidence", "hit", {})),
                AIMessage(content="PMID 36764316，pubmed:36764316:1，NCT04303780。"),
            ],
            "MolAnalyst": [
                request(("compute_properties", "props", {})),
                AIMessage(content="TPSA 20.23，与 NCT04303780 分开报告。"),
            ],
            "Reporter": [
                AIMessage(content="两类结果。"),
                AIMessage(content="新一轮。"),
            ],
        }
    )
    graph = build_graph(
        model,
        [search_evidence, compute_properties, other],
        InMemorySaver(),
        settings=GraphSettings(mode="multi"),
    )
    config = config_for(graph)
    output = await graph.ainvoke(turn("并行核查文献和计算 TPSA。"), config)

    assert output["critique_passes"] == 1
    assert output["issues"] == []
    assert output["step_count"] == 4
    assert [result.subtask_id for result in ordered_results(output)] == ["mol", "lit"]
    (lit,) = output["evidence_results"]
    assert lit.observed_ids == ["NCT04303780", "PMID:36764316", "pubmed:36764316:1"]
    assert [bound for _, bound in model.received["MolAnalyst"]] == [
        ("compute_properties",)
    ] * 2
    assert [bound for _, bound in model.received["LitResearcher"]] == [
        ("search_evidence",)
    ] * 2
    assert [m.text for m in output["messages"]] == [
        "并行核查文献和计算 TPSA。",
        "两类结果。",
    ]

    next_turn = await graph.ainvoke(turn("再会。"), config)
    assert (
        next_turn["tasks"]
        == next_turn["evidence_results"]
        == next_turn["molecule_results"]
        == []
    )
    assert next_turn["step_count"] == next_turn["critique_passes"] == 0


@pytest.mark.parametrize(
    ("conclusion", "kind", "bad_id"),
    [
        ("", "missing_result", None),
        ("论文 PMID: 987654321。", "unverified_ids", "PMID:987654321"),
        ("试验 NCT00000000。", "unverified_ids", "NCT00000000"),
        ("引用 pubmed:36764316:99。", "unverified_ids", "pubmed:36764316:99"),
    ],
)
async def test_critic_retries_only_affected_worker_and_reports_final_issues(
    task8_model, plan_reply, conclusion, kind, bad_id
):
    model = task8_model(
        replies={
            "Planner": [plan_reply(task("lit"), task("mol"))],
            "LitResearcher": [AIMessage(content=conclusion)] * 3,
            "MolAnalyst": [AIMessage(content="分子结论。")],
            "Reporter": [AIMessage(content="报告正文。")],
        }
    )
    graph = build_graph(
        model, [], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    assert turn_recursion_limit(graph, 1) == 9
    output = await graph.ainvoke(turn("问题", steps=1), config_for(graph, 1))
    assert output["step_count"] == 4
    assert output["critique_passes"] == 3
    (issue,) = output["issues"]
    assert issue.kind == kind and issue.subtask_id == "lit"
    assert issue.ids == ([] if bad_id is None else [bad_id])
    assert len(model.received["MolAnalyst"]) == 1
    for attempt, (messages, _) in enumerate(model.received["LitResearcher"][1:], 1):
        system, question, *previous, feedback = messages
        assert isinstance(question, HumanMessage) and question.text == "问题"
        assert len(previous) == attempt
        assert all(
            isinstance(message, AIMessage) and message.text == conclusion
            for message in previous
        )
        assert isinstance(feedback, HumanMessage)
        assert 'kind="critic_feedback"' not in system.text
        instructions = feedback.text.split('<data kind="critic_feedback">', 1)[0]
        assert (
            "补充缺失的结论" if bad_id is None else "核实或删除下列"
        ) in instructions
        assert payload(feedback, "critic_feedback") == [
            {"subtask_id": "lit", "ids": issue.ids}
        ]
    (worker,) = output["evidence_results"]
    assert [
        message.text for message in worker.messages if isinstance(message, HumanMessage)
    ] == ["问题"]
    assert payload(model.received["Reporter"][0][0][0], "critic_issues") == [
        issue.model_dump()
    ]
    assert "仍未通过核查" in output["messages"][-1].text
    assert ("缺少结论" if bad_id is None else bad_id) in output["messages"][-1].text


def test_observed_citations_include_original_text_but_skip_request_and_errors():
    data = {
        "request": {"nct_id": "NCT00000000"},
        "results": [
            {"set_id": "c80a362c-7ac3-4894-a076-0691e68ef8c1"},
            {
                "citation_id": "pubmed:36764316:1",
                "source_id": "36764316",
                "text": "registered at ClinicalTrials.gov, NCT04303780",
            },
        ],
    }
    success = ToolMessage(
        content=json.dumps(data),
        tool_call_id="a",
        artifact={"structured_content": data},
    )
    error = ToolMessage(content="bad NCT11111111", tool_call_id="b", status="error")
    assert observed_ids([success, error]) == {
        "c80a362c-7ac3-4894-a076-0691e68ef8c1",
        "pubmed:36764316:1",
        "PMID:36764316",
        "NCT04303780",
    }


@pytest.mark.parametrize(
    ("tasks", "problem"),
    [
        ([task("lit", "unknown")], "unknown"),
        ([task("mol", "search_trials")], "search_trials"),
        ([task("mol"), task("mol", task_id="second")], "最多允许一个 mol"),
        ([task("mol"), task("lit", task_id="mol")], "子任务 ID 重复"),
    ],
)
async def test_planner_validates_once_and_reports_raw_reply(
    task8_model, plan_reply, mcp_tool, tasks, problem
):
    @mcp_tool("lit")
    def search_trials():
        """Search trial records."""
        raise AssertionError("invalid plan must stop before tools")

    model = task8_model(replies={"Planner": [plan_reply(*tasks)]})
    graph = build_graph(
        model, [search_trials], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    with pytest.raises(ValueError, match=problem) as raised:
        await graph.ainvoke(turn("规划"), config_for(graph))
    assert "规划原始回复" in str(raised.value)
    assert len(model.received["Planner"]) == 1


@pytest.mark.parametrize("call_ids", [(None,), ("same", "same")])
async def test_worker_rejects_missing_or_duplicate_tool_call_ids(
    task8_model, plan_reply, call_ids
):
    model = task8_model(
        replies={
            "Planner": [plan_reply(task("lit"))],
            "LitResearcher": [
                request(*(("tool", identifier, {}) for identifier in call_ids))
            ],
        }
    )
    graph = build_graph(
        model, [], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    with pytest.raises(ValueError, match="工具调用 ID 缺失或重复"):
        await graph.ainvoke(turn("问题"), config_for(graph))


async def test_worker_limit_closes_batch_before_redispatch(task8_model, plan_reply):
    model = task8_model(
        replies={
            "Planner": [plan_reply(task("lit"))],
            "LitResearcher": [
                request(("search_trials", "same", {})),
                AIMessage(content="已说明执行限制。"),
            ],
            "Reporter": [AIMessage(content="报告。")],
        }
    )
    graph = build_graph(
        model, [], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    result = await graph.ainvoke(turn("问题", steps=1), config_for(graph, 1))
    messages, _ = model.received["LitResearcher"][1]
    call, skipped, feedback = messages[-3:]
    assert call.tool_calls[0]["id"] == skipped.tool_call_id == "same"
    assert payload(skipped, "tool_result") == {
        "content": STEP_LIMIT_MESSAGE,
        "status": "error",
    }
    assert isinstance(feedback, HumanMessage)
    (worker,) = result["evidence_results"]
    assert worker.messages[2].text == STEP_LIMIT_MESSAGE
    assert result["critique_passes"] == 2


async def test_sandbox_repairs_do_not_reset_on_critic_redispatch(
    task8_model, plan_reply, python_call
):
    executions = []

    def execute(request, call_id):
        executions.append(call_id)
        return PythonExecution(
            "", "traceback", 1, SandboxError(kind="code", message="执行失败")
        )

    ids = [f"failure-{index}" for index in range(MAX_REPAIR_ROUNDS + 1)]
    attempts = [
        AIMessage(
            content="", tool_calls=[python_call(identifier, "raise ValueError()")]
        )
        for identifier in ids
    ]
    attempts[0].tool_calls.append(python_call("parallel", "raise ValueError()"))
    model = task8_model(
        replies={
            "Planner": [plan_reply(task("mol", "run_python"))],
            "MolAnalyst": [
                *attempts,
                AIMessage(content="引用伪造 PMID 314159。"),
                AIMessage(content="", tool_calls=[python_call("after-retry", "pass")]),
                AIMessage(content="脚本未能执行成功。"),
            ],
            "Reporter": [AIMessage(content="脚本分析失败。")],
        }
    )
    graph = build_graph(
        model,
        [create_run_python_tool(execute)],
        InMemorySaver(),
        settings=GraphSettings(mode="multi"),
    )
    result = await graph.ainvoke(turn("运行 Python"), config_for(graph))
    assert executions == ids
    (worker,) = result["molecule_results"]
    assert sum(isinstance(message, HumanMessage) for message in worker.messages) == 1
    errors = {
        message.tool_call_id: message
        for message in worker.messages
        if isinstance(message, ToolMessage)
    }
    assert errors["parallel"].text == PARALLEL_SANDBOX_MESSAGE
    assert REPAIR_LIMIT_MESSAGE in errors[ids[-1]].text
    assert errors["after-retry"].text == REPAIR_LIMIT_MESSAGE
    assert result["critique_passes"] == 2


async def test_worker_can_run_more_supersteps_than_parent_and_resume_with_original_budget(
    task8_model, plan_reply, mcp_tool
):
    executed = []

    @mcp_tool("lit")
    def search_trials(query: str):
        """Record a query."""
        executed.append(query)
        return "已搜到。"

    model = task8_model(
        replies={
            "Planner": [plan_reply(task("lit", "search_trials"))],
            "LitResearcher": [
                *(
                    request(("search_trials", "same", {"query": str(index)}))
                    for index in range(6)
                ),
                AIMessage(content="完成六项检索。"),
            ],
            "Reporter": [AIMessage(content="最终报告。")],
        }
    )
    graph = build_graph(
        model, [search_trials], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    config = config_for(graph)
    updates = graph.astream(
        turn("长任务", steps=7), config, stream_mode="updates", durability="sync"
    )
    async with aclosing(updates):
        assert "Planner" in await anext(updates)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("LitResearcher",)
    recovery = await resume_config(graph, {**config, "recursion_limit": 1}, snapshot)
    output = await graph.ainvoke(None, recovery)
    assert output["max_steps"] == output["step_count"] == 7
    assert executed == list(map(str, range(6)))
    assert len(model.received["Planner"]) == 1
