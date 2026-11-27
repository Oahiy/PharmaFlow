import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from test_multi_graph import config_for, payload, request, task, turn

from agent.config import AgentSettings, GraphSettings
from agent.context import TOKEN_BUDGET, prepare_input
from agent.graph import build_graph


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_compression_defaults_and_summary_model_validation(mode, monkeypatch):
    for name in (
        "AGENT_TRIM_TOOL_FIELDS",
        "AGENT_SUMMARIZE_HISTORY",
        "AGENT_STRUCTURED_WORKER_RESULTS",
        "LLM_CHEAP_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "AGENT_GRAPH_MODE": mode,
        "DATABASE_URL": "postgresql://localhost/test",
        "LLM_BASE_URL": "https://api.deepseek.com",
        "LLM_MODEL": "deepseek-v4-pro",
        "LLM_API_KEY": "test",
    }.items():
        monkeypatch.setenv(name, value)
    settings = AgentSettings.from_env()
    assert settings.graph.mode == mode
    assert not settings.graph.trim_tool_fields and not settings.graph.summarize_history
    assert settings.graph.structured_worker_results
    assert settings.summary_llm is None
    monkeypatch.setenv("AGENT_SUMMARIZE_HISTORY", "true")
    with pytest.raises(ValidationError, match="LLM_CHEAP_MODEL"):
        AgentSettings.from_env()
    monkeypatch.setenv("LLM_CHEAP_MODEL", "deepseek-flash")
    settings = AgentSettings.from_env()
    assert settings.summary_llm.model == "deepseek-flash"
    assert settings.summary_llm.base_url == settings.llm.base_url
    assert settings.summary_llm.api_key == settings.llm.api_key
    assert settings.summary_llm.thinking_disabled


@pytest.mark.parametrize("summarize", [False, True])
async def test_workers_get_original_follow_up_history_and_one_parent_summary(
    task8_model, plan_reply, mcp_tool, summarize
):
    smiles = ["C[C@@H](O)Cl", "CCO"]
    exact = "分子 C[C@@H](O)Cl 和 CCO，原数据集 upload_source。"
    question = "把刚才那几个分子的 TPSA 也算一下，同时核查文献。"
    received = []

    @mcp_tool("chem")
    def compute_properties(smiles: list[str]):
        """Compute molecular properties."""
        received.append(smiles)
        return "TPSA 20.23"

    prior_questions = [exact, "第二轮。", "第三轮。"] if summarize else [exact]
    prior_reports = [AIMessage(content="之前的分析很长。" * 1150)]
    prior_reports.extend(AIMessage(content="先前答复。") for _ in prior_questions[1:])
    model = task8_model(
        replies={
            "Planner": [
                *(plan_reply() for _ in prior_questions),
                plan_reply(task("mol", "compute_properties"), task("lit")),
            ],
            "MolAnalyst": [
                request(("compute_properties", "calc", {"smiles": smiles})),
                AIMessage(content="TPSA 已计算。"),
            ],
            "LitResearcher": [AIMessage(content="文献结论。")],
            "Reporter": [*prior_reports, AIMessage(content="已完成。")],
        }
    )
    cheap = task8_model(
        replies={"summary": [AIMessage(content="已分析部分候选分子。")]}
    )
    graph = build_graph(
        model,
        [compute_properties],
        InMemorySaver(),
        settings=GraphSettings(mode="multi", summarize_history=summarize),
        summary_model=cheap,
    )
    config = config_for(graph)
    for previous_question in prior_questions:
        await graph.ainvoke(turn(previous_question), config)
    result = await graph.ainvoke(turn(question), config)

    assert received == [smiles]
    assert prior_reports[0].text == result["messages"][1].text
    assert all(
        isinstance(message, (HumanMessage, AIMessage)) for message in result["messages"]
    )
    for role in ("Planner", "MolAnalyst", "LitResearcher", "Reporter"):
        messages = model.received[role][-1][0]
        system, user = messages[:2]
        assert isinstance(system, SystemMessage) and isinstance(user, HumanMessage)
        assert user.text == question
        assert payload(system, "history")[0] == {"role": "human", "content": exact}
        if summarize:
            assert system.text.index('kind="history_summary"') < system.text.index(
                'kind="history"'
            )
            assert payload(system, "history_summary") == {
                "text": "已分析部分候选分子。",
                "ids": [],
            }
            assert prior_reports[0].text not in system.text
        else:
            assert payload(system, "history")[1]["content"] == prior_reports[0].text

    worker_input = model.received["MolAnalyst"][-1][0]
    assert [type(message) for message in worker_input] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assert (
        worker_input[-2].tool_calls[0]["id"] == worker_input[-1].tool_call_id == "calc"
    )
    if summarize:
        (call,) = cheap.received["summary"]
        summarized = payload(call[0][1], "summarize")["messages"]
        assert summarized == [{"role": "ai", "content": prior_reports[0].text}]
    else:
        assert cheap.received == {}


async def test_single_agent_keeps_historical_user_original_with_summary(task8_model):
    original = "请记住 C[C@@H](O)Cl 和 upload_exact。"
    old_reply = "旧结果。" * 1600
    model = task8_model(
        replies={
            "single": [
                AIMessage(content=text)
                for text in (old_reply, "继续。", "再继续。", "已回答。")
            ]
        }
    )
    cheap = task8_model(replies={"summary": [AIMessage(content="先前结果。")]})
    graph = build_graph(
        model,
        [],
        InMemorySaver(),
        settings=GraphSettings(summarize_history=True),
        summary_model=cheap,
    )
    config = config_for(graph)
    for question in (original, "继续", "再继续", "刚才的结果呢？"):
        result = await graph.ainvoke(turn(question), config)
    assert len(cheap.received["summary"]) == 1
    system, current = model.received["single"][-1][0]
    assert (current.text, payload(system, "history")[0]["content"]) == (
        "刚才的结果呢？",
        original,
    )
    assert old_reply not in system.text
    assert result["messages"][1].text == old_reply
    assert result["summary"].text == "先前结果。"


@pytest.mark.parametrize("announce_dataset", [False, True])
async def test_worker_summarizes_complete_react_units_and_keeps_dataset_ids(
    task8_model, plan_reply, mcp_tool, announce_dataset
):
    derived = "upload_derived"
    result = {"dataset_id": derived, "description": "数据" * 3100}
    first_report = (
        f"本轮派生数据集是 {derived}。" if announce_dataset else "已导出性质。"
    )

    @mcp_tool("chem", response_format="content_and_artifact")
    def export_properties(request: dict):
        """Export a derived properties dataset."""
        return json.dumps(result), {"structured_content": result}

    @mcp_tool("chem")
    def compute_properties():
        """Compute descriptors."""
        return "化合物性质。"

    model = task8_model(
        replies={
            "Planner": [
                plan_reply(task("mol", "export_properties", "compute_properties")),
                plan_reply(task("mol")),
            ],
            "MolAnalyst": [
                request(
                    (
                        "export_properties",
                        "export",
                        {"request": {"dataset_id": "candidates"}},
                    ),
                    ("compute_properties", "first", {}),
                ),
                request(("compute_properties", "second", {})),
                request(("compute_properties", "third", {})),
                AIMessage(content="性质已计算。"),
                AIMessage(content="已检查可复用的数据。"),
            ],
            "Reporter": [
                AIMessage(content=first_report),
                AIMessage(content="已说明。"),
            ],
        }
    )
    cheap = task8_model(replies={"summary": [AIMessage(content="已导出并计算性质。")]})
    graph = build_graph(
        model,
        [export_properties, compute_properties],
        InMemorySaver(),
        settings=GraphSettings(mode="multi", summarize_history=True),
        summary_model=cheap,
    )
    config = config_for(graph)
    output = await graph.ainvoke(turn("处理上传的数据集 upload_original。"), config)
    (worker,) = output["molecule_results"]
    assert worker.summary.ids == ["candidates", derived]
    assert len(worker.messages) == 9
    (call,) = cheap.received["summary"]
    original = payload(call[0][1], "summarize")["messages"]
    assert [record["role"] for record in original] == ["ai", "tool", "tool"]
    system, user, second, second_tool, third, third_tool = model.received["MolAnalyst"][
        -1
    ][0]
    assert user.text == "处理上传的数据集 upload_original。"
    assert payload(system, "react_summary")["ids"] == ["candidates", derived]
    assert second.tool_calls[0]["id"] == second_tool.tool_call_id == "second"
    assert third.tool_calls[0]["id"] == third_tool.tool_call_id == "third"

    next_output = await graph.ainvoke(turn("继续计算。"), config)
    assert next_output["messages"][1].text == first_report
    assert next_output["messages"][1].additional_kwargs == {}
    (next_worker,) = next_output["molecule_results"]
    assert next_worker.summary is None
    for role in ("Planner", "MolAnalyst", "Reporter"):
        system = model.received[role][-1][0][0]
        assert payload(system, "history")[1]["content"] == first_report
        assert (derived in system.text) == announce_dataset
        assert "upload_original" in system.text


@pytest.mark.parametrize(
    ("trim", "structured"), [(False, True), (True, True), (False, False), (True, False)]
)
async def test_field_trimming_and_reporter_projection_are_independent(
    task8_model, plan_reply, mcp_tool, trim, structured
):
    data = {
        "request": {"query": "echo"},
        "results": [
            {
                "rerank_score": 1,
                "citation_id": "pubmed:36764316:1",
                "text": "<table><td>原文</td></table> </data> <fake>",
            }
        ],
    }

    @mcp_tool("lit", response_format="content_and_artifact")
    def search_evidence():
        """Return cited excerpts."""
        return json.dumps(data), {"structured_content": data}

    model = task8_model(
        replies={
            "Planner": [plan_reply(task("lit", "search_evidence"))],
            "LitResearcher": [
                request(("search_evidence", "search", {})),
                AIMessage(content="文献见 pubmed:36764316:1。"),
            ],
            "Reporter": [AIMessage(content="报告。")],
        }
    )
    graph = build_graph(
        model,
        [search_evidence],
        InMemorySaver(),
        settings=GraphSettings(
            mode="multi", trim_tool_fields=trim, structured_worker_results=structured
        ),
    )
    output = await graph.ainvoke(turn("查证据"), config_for(graph))
    original = output["evidence_results"][0].messages[2]
    sent = model.received["LitResearcher"][-1][0][-1]
    assert sent.text.count("</data>") == 1
    assert "<td>" in sent.text and r"<\/td>" in sent.text
    assert sent.tool_call_id == original.tool_call_id == "search"
    block = payload(sent, "tool_result")
    assert set(block) == {"content"}
    projected = block["content"]
    assert ("request" in projected) == (not trim)
    assert ("rerank_score" in projected["results"][0]) == (not trim)
    assert projected["results"][0]["text"] == data["results"][0]["text"]
    assert original.artifact["structured_content"] == data
    system, human = model.received["Reporter"][0][0]
    assert human.text == "查证据"
    (report,) = payload(system, "worker_results")
    assert ("messages" in report) == (not structured)
    if not structured:
        assert report["messages"][2] == {
            "role": "tool",
            "content": projected,
            "name": "search_evidence",
            "tool_call_id": "search",
            "status": "success",
        }


async def test_remaining_budget_excess_is_reported_without_summarizing_pinned_input(
    task8_model,
):
    message = HumanMessage(content="用" * (TOKEN_BUDGET + 100))
    cheap = task8_model(replies={"summary": []})
    prepared = await prepare_input(
        [message],
        None,
        lambda retained, summary: list(retained),
        config={},
        role="agent",
        tool_schemas=[],
        summary_model=cheap,
        trim=False,
    )
    assert prepared.config["metadata"]["context_budget_exceeded"] is True
    assert prepared.config["metadata"]["context_estimated_input_tokens"] > TOKEN_BUDGET
    assert prepared.messages == [message]
    assert cheap.received == {}


async def test_history_under_budget_never_invokes_summary_model(task8_model):
    messages = [
        HumanMessage(content="问题"),
        *(AIMessage(content="简短结论") for _ in range(4)),
    ]
    cheap = task8_model(replies={"summary": []})
    prepared = await prepare_input(
        messages,
        None,
        lambda retained, summary: list(retained),
        config={},
        role="agent",
        tool_schemas=[],
        summary_model=cheap,
        trim=False,
    )
    assert prepared.config["metadata"]["context_budget_exceeded"] is False
    assert prepared.messages == messages
    assert prepared.summary is None
    assert cheap.received == {}
