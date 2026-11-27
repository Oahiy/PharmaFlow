import json
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_multi_graph import config_for, request, task, turn

from agent.config import AgentSettings, GraphSettings, LLMSettings
from agent.graph import build_graph
from eval.cases import CASE_DIRECTORY, EvalCase, load_cases
from eval.compare import GRAPH_CONFIG_FIELDS, read_comparison
from eval.provenance import graph_config
from eval.runner import execute_case, run_suite
from eval.scoring import multi_score_messages, observed_calls, score_case
from eval.usage import AgentUsage


async def unexpected_judge(*args):
    raise AssertionError("judge should not run")


def settings_for(mode, *, summarize_history=False, trim_tool_fields=False):
    return AgentSettings(
        database_url="postgresql://localhost/test",
        llm=LLMSettings(
            base_url="https://example.invalid/v1", api_key="test", model="strong"
        ),
        graph=GraphSettings(
            mode=mode,
            summarize_history=summarize_history,
            trim_tool_fields=trim_tool_fields,
        ),
        cheap_model="cheap" if summarize_history else None,
    )


async def test_merged_branch_records_score_like_single_agent(
    task8_model, plan_reply, mcp_tool
):
    @mcp_tool("lit", response_format="content_and_artifact")
    def search_pubmed(request: dict):
        """Return evidence rows."""
        result = {"results": [{"pmid": "36764316"}]}
        return json.dumps(result), {"structured_content": result}

    @mcp_tool("chem", response_format="content_and_artifact")
    def compute_properties(request: dict):
        """Return descriptors for the inputs."""
        result = {"results": [{"id": "result", "smiles": request["smiles"][0]}]}
        return json.dumps(result), {"structured_content": result}

    case = EvalCase.model_validate(
        {
            "id": "fixture",
            "category": "multi_hop",
            "question": "计算 CCO，并分别查 PubMed。",
            "accepted_paths": [
                [
                    {
                        "id": "properties",
                        "tool": "compute_properties",
                        "request": {"smiles": ["CCO"]},
                    },
                    {
                        "id": "paper",
                        "tool": "search_pubmed",
                        "request": {"query": "KRAS"},
                    },
                ]
            ],
            "checks": [{"type": "answer_regex", "pattern": "CCO"}],
        }
    )
    calls = [
        request((step.tool, "same", {"request": step.request}))
        for step in case.accepted_paths[0]
    ]
    answer = AIMessage(content="已计算 CCO，并查到 PMID 36764316。")
    tools = [search_pubmed, compute_properties]
    single_model = task8_model(replies={"single": [*calls, answer]})
    single = build_graph(single_model, tools, InMemorySaver())
    single_state = await single.ainvoke(turn(case.question), config_for(single))

    multi_model = task8_model(
        replies={
            "Planner": [
                plan_reply(
                    task("mol", "compute_properties"), task("lit", "search_pubmed")
                )
            ],
            "MolAnalyst": [calls[0], AIMessage(content="已计算。")],
            "LitResearcher": [calls[1], AIMessage(content="已查到 PMID 36764316。")],
            "Reporter": [answer],
        }
    )
    multi = build_graph(
        multi_model, tools, InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    multi_state = await multi.ainvoke(turn(case.question), config_for(multi))
    raw = multi_score_messages(multi_state)
    observed = observed_calls(raw)
    assert [call.name for call in observed] == ["compute_properties", "search_pubmed"]
    assert observed[0].message_index < observed[1].message_index
    expected = score_case(case, single_state["messages"][1:])
    assert expected.failure_type is None
    assert score_case(case, raw) == expected


async def test_e02_reporter_diagnostics_survive_scoring_and_account_for_models(
    task8_model, plan_reply
):
    case = next(case for case in load_cases().cases if case.id == "E02")
    planner = plan_reply().model_copy(
        update={
            "usage_metadata": {
                "input_tokens": 12,
                "output_tokens": 1,
                "total_tokens": 13,
            }
        }
    )
    report = AIMessage(
        content="NCT123 格式错误，需要 NCT 加 8 位数字。NCT87654321。",
        usage_metadata={"input_tokens": 23, "output_tokens": 1, "total_tokens": 24},
    )
    model = task8_model(replies={"Planner": [planner], "Reporter": [report]})
    graph = build_graph(
        model, [], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    usage = AgentUsage()
    config = {**config_for(graph), "callbacks": [usage]}
    result = await execute_case(
        case,
        graph=graph,
        judge=unexpected_judge,
        thread_id=UUID(config["configurable"]["thread_id"]),
        trace_id=uuid4().hex,
        config=config,
        usage=usage,
        tools_by_name={},
    )
    assert result.input_tokens == 35
    assert result.failure_type == "rule_failed"
    assert result.error_detail["reporter_unverified_ids"] == ["NCT87654321"]
    snapshot = await graph.aget_state(config)
    assert snapshot.values["messages"][-1].text.endswith(
        "以下引用未核验：NCT87654321。"
    )


async def test_reporter_allows_user_provided_ids_and_ignores_bare_numbers(
    task8_model, plan_reply
):
    model = task8_model(
        replies={
            "Planner": [plan_reply()],
            "Reporter": [
                AIMessage(content="原始编号 NCT04303780，PMID 123，数值 35000000。")
            ],
        }
    )
    graph = build_graph(
        model, [], InMemorySaver(), settings=GraphSettings(mode="multi")
    )
    state = await graph.ainvoke(
        turn("我给的编号是 NCT04303780 和 PMID:123。"), config_for(graph)
    )
    assert state["reporter_unverified_ids"] == []
    assert (
        state["messages"][-1].text == "原始编号 NCT04303780，PMID 123，数值 35000000。"
    )


async def test_multi_self_healing_suite_is_rejected_before_start(task8_model):
    with pytest.raises(ValueError, match="多智能体模式暂不支持自愈"):
        await run_suite(
            settings_for("multi"),
            tag="fixture",
            model=task8_model(replies={}),
            judge=unexpected_judge,
            callback_factory=lambda *args: None,
            case_directory=CASE_DIRECTORY.parent / "cases_self_healing",
        )


@pytest.mark.parametrize(
    ("trim", "summarize"), [(False, False), (True, False), (False, True), (True, True)]
)
async def test_single_agent_prompt_digest_matches_the_enabled_template(
    task8_model, trim, summarize
):
    settings = settings_for(
        "single", summarize_history=summarize, trim_tool_fields=trim
    )
    model = task8_model(replies={"single": [AIMessage(content="完成。")]})
    graph = build_graph(
        model,
        [],
        InMemorySaver(),
        settings=settings.graph,
        summary_model=task8_model(replies={}) if summarize else None,
    )
    await graph.ainvoke(turn("测试"), config_for(graph))
    sent_system_prompt = model.received["single"][0][0][0].text
    recorded = graph_config(settings, [])
    prompt_hash = recorded["system_prompt_sha256"]
    assert prompt_hash == sha256(sent_system_prompt.encode("utf-8")).hexdigest()
    assert prompt_hash == recorded["node_system_prompt_sha256"]["agent"]
    baseline_hash = graph_config(settings_for("single"), [])["system_prompt_sha256"]
    assert (prompt_hash == baseline_hash) == (not (trim or summarize))


async def test_new_graph_provenance_compare_uses_separate_fields(monkeypatch, mcp_tool):
    @mcp_tool("lit")
    def search_trials():
        """Return trials."""
        return []

    first_id, second_id = uuid4(), uuid4()
    digests = {"cases_sha256": "cases", "recordings_sha256": "recordings"}
    first = {**graph_config(settings_for("single"), [search_trials]), **digests}
    second = {
        **graph_config(settings_for("multi", summarize_history=True), [search_trials]),
        **digests,
    }
    assert (
        first["system_prompt_sha256"]
        == "7f59ebb151d74990cfbe633cf68681c7fb6015fcc5e97b2b893740cf28d2c604"
    )
    assert second["node_models"]["Critic"] is None
    assert second["node_models"]["summary"]["model"] == "cheap"
    assert second["compression"] == {
        "trim_tool_fields": False,
        "structured_worker_results": True,
        "summarize_history": True,
        "token_budget": 6144,
        "keep_units": 2,
    }
    assert set(second["node_system_prompt_sha256"]) == {
        "Planner",
        "LitResearcher",
        "MolAnalyst",
        "Reporter",
        "summary",
    }
    assert set(second["worker_tool_schema_sha256"]) == {"LitResearcher", "MolAnalyst"}

    async def coverage(conn, run_id):
        return {
            "finished_at": True,
            "planned_count": 1,
            "stored_count": 1,
            "matched_count": 1,
            "config": first if run_id == first_id else second,
        }

    class Connection:
        async def execute(self, sql, params):
            assert params == (first_id, second_id)
            return self

        async def fetchall(self):
            return []

    monkeypatch.setattr("eval.compare.run_coverage", coverage)
    comparison = await read_comparison(Connection(), first_id, second_id)
    assert comparison.model_input_matches["system_prompt_sha256"] is None
    assert comparison.graph_config_matches == dict.fromkeys(GRAPH_CONFIG_FIELDS, False)
