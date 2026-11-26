from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import PARALLEL_SANDBOX_MESSAGE, build_graph
from sandbox import artifacts
from sandbox.executor import DatasetInputError, PythonExecution
from sandbox.models import REPAIR_LIMIT_MESSAGE, RunPythonResult, SandboxError
from sandbox.tool import create_run_python_tool, is_repairable_failure


async def test_named_cli_thread_repairs_bad_output_after_correcting_dataset(
    fake_chat_model, python_call, png_bytes, tmp_path, monkeypatch
):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)

    def execute(request, call_id):
        if call_id == "missing":
            raise DatasetInputError("数据集不存在：unknown")
        if call_id == "bad-output":
            return PythonExecution(
                "", "", 0, SandboxError(kind="output", message="输出只支持 PNG")
            )
        return PythonExecution("", "", 0, files={"chart.png": png_bytes})

    model = fake_chat_model(
        responses=[
            *(
                AIMessage(content="", tool_calls=[python_call(call_id, "pass")])
                for call_id in ("missing", "bad-output", "fixed")
            ),
            AIMessage(content="完成"),
        ]
    )
    thread = "named-cli-thread"
    graph = build_graph(model, [create_run_python_tool(execute)], InMemorySaver())
    result = await graph.ainvoke(
        {"messages": [HumanMessage("画图")], "step_count": 0, "max_steps": 5},
        {"configurable": {"thread_id": thread}},
    )
    missing, bad_output, fixed = [
        msg for msg in result["messages"] if isinstance(msg, ToolMessage)
    ]
    assert not is_repairable_failure(missing)
    assert is_repairable_failure(bad_output)
    assert fixed.status == "success"
    namespace = uuid5(NAMESPACE_URL, f"pharmaflow:cli:{thread}")
    assert (tmp_path / str(namespace)).is_dir()


async def test_budget_counts_failures_from_latest_human_and_allows_one_call_per_reply(
    fake_chat_model, python_call
):
    executed = []
    failures = iter(
        [
            ("timeout", "请缩小数据或计算规模后再提交脚本。"),
            ("oom", "请缩小数据规模或分批处理后再提交脚本。"),
            ("code", "请根据 traceback 修正脚本。"),
            ("code", "请根据 traceback 修正脚本。"),
        ]
    )

    def execute(request, call_id):
        executed.append(call_id)
        if call_id == "new-turn":
            return PythonExecution("4\n", "", 0)
        kind, hint = next(failures)
        return PythonExecution(
            "", "KeyError: 'QED'\n", 1, SandboxError(kind=kind, message=hint)
        )

    missing_dataset_ids = python_call("missing-dataset-ids", "print(1)")
    del missing_dataset_ids["args"]["request"]["dataset_ids"]
    replies = [
        AIMessage(
            content="",
            tool_calls=[
                python_call("first", "print(1)"),
                python_call("parallel", "print(0)"),
            ],
        ),
        AIMessage(content="", tool_calls=[missing_dataset_ids]),
        *(
            # The budget check allows another attempt with the same source.
            AIMessage(content="", tool_calls=[python_call(call_id, "print(1)")])
            for call_id in ("fix-1", "fix-2", "fix-3", "over-budget")
        ),
        AIMessage(content="已说明剩余问题。"),
        AIMessage(content="", tool_calls=[python_call("new-turn", "print(4)")]),
        AIMessage(content="4"),
    ]
    graph = build_graph(
        fake_chat_model(responses=replies),
        [create_run_python_tool(execute)],
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    first = await graph.ainvoke(
        {"messages": [HumanMessage("处理数据")], "step_count": 0, "max_steps": 10},
        config,
    )
    assert executed == ["first", "fix-1", "fix-2", "fix-3"]
    results = {
        m.tool_call_id: m for m in first["messages"] if isinstance(m, ToolMessage)
    }
    assert results["parallel"].text == PARALLEL_SANDBOX_MESSAGE
    assert results["over-budget"].text == REPAIR_LIMIT_MESSAGE
    omitted = results["missing-dataset-ids"]
    assert "request.dataset_ids: Field required" in omitted.text
    assert omitted.artifact is None
    assert not is_repairable_failure(omitted)
    assert not is_repairable_failure(results["parallel"])
    assert not is_repairable_failure(results["over-budget"])
    timeout = RunPythonResult.model_validate_json(results["first"].text)
    oom = RunPythonResult.model_validate_json(results["fix-1"].text)
    last = RunPythonResult.model_validate_json(results["fix-3"].text)
    assert timeout.error.kind == "timeout" and "缩小" in timeout.error.message
    assert oom.error.kind == "oom" and "分批" in oom.error.message
    assert last.error.kind == "code" and REPAIR_LIMIT_MESSAGE in last.error.message
    assert last.stderr == "KeyError: 'QED'\n"
    assert all(result.status == "error" for result in results.values())
    second = await graph.ainvoke(
        {"messages": [HumanMessage("继续")], "step_count": 0, "max_steps": 3}, config
    )
    assert executed[-1] == "new-turn"
    assert second["messages"][-2].status == "success"
