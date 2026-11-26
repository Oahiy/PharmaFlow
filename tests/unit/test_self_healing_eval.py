import csv
from pathlib import Path
from uuid import uuid4

import pytest
from docker.errors import DockerException
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from requests.exceptions import ConnectionError

from agent.graph import STEP_LIMIT_MESSAGE, build_graph
from eval.cases import CASE_DIRECTORY, load_cases
from eval.runner import execute_case, infrastructure_error
from eval.scoring import score_case
from eval.usage import AgentUsage
from mcp_servers.chem.datasets import resolve_dataset
from mcp_servers.chem.models import ExportPropertiesInput
from mcp_servers.chem.tools import export_properties
from sandbox import artifacts
from sandbox.executor import PythonExecution
from sandbox.models import SandboxError
from sandbox.tool import create_run_python_tool

SELF_HEALING_DIR = Path(CASE_DIRECTORY).with_name("cases_self_healing")


@pytest.mark.parametrize("error", [DockerException, ConnectionError])
def test_sandbox_connection_error_aborts_evaluation(error):
    assert infrastructure_error(error("infrastructure unavailable"))


@pytest.mark.parametrize(
    ("fixes", "failure_type"),
    [(1, None), (3, None), (4, "tool_error"), (0, "wrong_tool")],
)
async def test_self_healing_eval_scores_only_model_attempts(
    fake_chat_model, python_call, png_bytes, tmp_path, monkeypatch, fixes, failure_type
):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    suite = load_cases(SELF_HEALING_DIR)
    case = next(c for c in suite.cases if c.id == "H01")
    assert suite.sha256 != load_cases().sha256
    executed = []

    def execute(request, call_id):
        executed.append((request, call_id))
        if call_id == f"fix-{fixes}":
            return PythonExecution(
                "rows=50\nmin_ic50_nm=1.70\n", "", 0, files={"ic50.png": png_bytes}
            )
        return PythonExecution(
            "",
            "KeyError: 'IC50_NM'\n",
            1,
            SandboxError(kind="code", message="检查列名。"),
        )

    model = fake_chat_model(
        responses=[
            *(
                AIMessage(
                    content="",
                    tool_calls=[
                        python_call(f"fix-{i}", f"# revised {i}\npass", ["candidates"])
                    ],
                )
                for i in range(1, fixes + 1)
            ),
            AIMessage(content="rows=50\nmin_ic50_nm=1.70\n图已生成。"),
        ]
    )
    run_python = create_run_python_tool(execute)
    graph = build_graph(model, [run_python], InMemorySaver())
    thread_id, user_id = uuid4(), uuid4()
    config = {"configurable": {"thread_id": str(thread_id), "user_id": str(user_id)}}

    async def judge(*_):
        pytest.fail("the self-healing case should be evaluated by rules")

    result = await execute_case(
        case,
        graph=graph,
        judge=judge,
        thread_id=thread_id,
        trace_id=uuid4().hex,
        config=config,
        usage=AgentUsage(),
        tools_by_name={run_python.name: run_python},
    )
    assert (result.failure_type, result.success) == (failure_type, failure_type is None)
    assert result.repair_attempts == min(fixes, 3)
    assert executed[0][0] == case.initial_run_python
    assert len(executed) == 1 + min(fixes, 3)
    assert model.received[0][-1].status == "error"
    assert "KeyError: 'IC50_NM'" in model.received[0][-1].text


def test_only_dataset_ids_are_declared_as_python_argument_constraints():
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H01")
    (step,) = case.accepted_paths[0]
    assert step.normalized_request == {"dataset_ids": ["candidates"]}


async def _evaluate(case, model, tools):
    thread_id = uuid4()

    async def judge(*_):
        pytest.fail("self-healing cases use rule scoring")

    return await execute_case(
        case,
        graph=build_graph(model, tools, InMemorySaver()),
        judge=judge,
        thread_id=thread_id,
        trace_id=uuid4().hex,
        usage=AgentUsage(),
        config={"configurable": {"thread_id": str(thread_id), "user_id": str(uuid4())}},
        tools_by_name={tool.name: tool for tool in tools},
    )


@pytest.mark.parametrize(
    ("steps", "failure", "attempts"),
    [
        (["inspect", "finish"], None, 2),
        (["inspect", "code_error", "finish"], None, 3),
        (["inspect"], "rule_failed", 1),
        (["wrong_dataset", "finish"], "invalid_args", 2),
        (["inspect", "inspect", "inspect", "finish"], "repair_budget_exceeded", 4),
    ],
)
async def test_h01_scores_diagnostics_within_three_executions(
    fake_chat_model,
    python_call,
    png_bytes,
    tmp_path,
    monkeypatch,
    steps,
    failure,
    attempts,
):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H01")
    stdout = "rows=50\nmin_ic50_nm=1.70\n"
    columns = "['molecule_chembl_id', 'name', 'ic50_nm']\n"

    def execute(request, call_id):
        if request == case.initial_run_python or request.code == "code_error":
            return PythonExecution(
                "",
                "KeyError: 'IC50_NM'",
                1,
                SandboxError(kind="code", message="检查列名"),
            )
        if request.code == "finish":
            return PythonExecution(stdout, "", 0, files={"ic50.png": png_bytes})
        return PythonExecution(columns, "", 0)

    calls = [
        python_call(
            str(i), step, ["other"] if step == "wrong_dataset" else ["candidates"]
        )
        for i, step in enumerate(steps)
    ]
    model = fake_chat_model(
        responses=[
            *(AIMessage(content="", tool_calls=[call]) for call in calls),
            AIMessage(content=stdout + "图已生成。"),
        ]
    )
    result = await _evaluate(case, model, [create_run_python_tool(execute)])
    assert result.failure_type == failure
    assert result.repair_attempts == attempts
    if failure == "rule_failed":
        stdout_rule = next(
            r for r in result.error_detail["rules"] if r["rule"]["path"] == "stdout"
        )
        assert columns in stdout_rule["detail"]


def test_self_healing_model_step_limit_has_its_own_failure_type(python_call):
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H01")
    result = score_case(
        case,
        [
            AIMessage(
                content="", tool_calls=[python_call("fix", "pass", ["candidates"])]
            ),
            ToolMessage(
                content=STEP_LIMIT_MESSAGE,
                name="run_python",
                tool_call_id="fix",
                status="error",
            ),
        ],
    )

    assert result.failure_type == "step_limit"
    assert result.repair_attempts == 0


@pytest.mark.parametrize("kind", ["timeout", "code"])
async def test_h02_requires_genuine_timeout_then_scores_repair(
    fake_chat_model, python_call, kind
):
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H02")
    calls = []

    def execute(request, call_id):
        calls.append(request)
        if len(calls) == 1:
            return PythonExecution(
                "", "", None, SandboxError(kind=kind, message="缩小计算规模")
            )
        return PythonExecution("rows=50\nmedian_ic50_nm=77.50\n", "", 0)

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    python_call("fix", "print(values.median())", ["candidates"])
                ],
            ),
            AIMessage(content="共 50 条记录，中位数为 77.50 nM。"),
        ]
    )
    run = create_run_python_tool(execute)
    if kind == "code":
        with pytest.raises(RuntimeError, match="H02.*预期失败.*timeout"):
            await _evaluate(case, model, [run])
        assert not model.received
    else:
        result = await _evaluate(case, model, [run])
        assert result.success and result.repair_attempts == 1
        assert "缩小计算规模" in model.received[0][-1].text
    assert calls[0] == case.initial_run_python


@pytest.mark.parametrize("use_exported_id", [True, False])
async def test_h03_repairs_real_property_export_using_its_runtime_id(
    fake_chat_model, python_call, tmp_path, png_bytes, monkeypatch, use_exported_id
):
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H03")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "artifacts")
    exports = []

    @tool("export_properties", response_format="content_and_artifact")
    def export(request: ExportPropertiesInput):
        """Expose the real chem export with the MCP result shape."""
        result = export_properties(request)
        exports.append(result.dataset_id)
        return result.model_dump_json(), {
            "structured_content": result.model_dump(mode="json")
        }

    ranking = []

    def execute(request, call_id):
        if not ranking:
            assert request.dataset_ids == exports
            assert '["QED"]' in request.code
            with resolve_dataset(exports[0]).open(
                encoding="utf-8", newline=""
            ) as source:
                rows = sorted(
                    csv.DictReader(source),
                    key=lambda row: (-float(row["qed"]), row["molecule_chembl_id"]),
                )
            ranking.extend(
                (r["molecule_chembl_id"], float(r["qed"])) for r in rows[:10]
            )
            return PythonExecution(
                "",
                "KeyError: 'QED'\n",
                1,
                SandboxError(kind="code", message="检查列名"),
            )
        # Give even the wrong ID the correct result to isolate argument scoring.
        stdout = "".join(f"{key} {qed:.4f}\n" for key, qed in ranking)
        return PythonExecution(stdout, "", 0, files={"mw-logp.png": png_bytes})

    class RepairModel(fake_chat_model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            last = messages[-1]
            assert isinstance(last, ToolMessage)
            if last.status == "error":
                seeded_run = messages[-2].tool_calls[0]
                exported_id = seeded_run["args"]["request"]["dataset_ids"][0]
                ids = [exported_id] if use_exported_id else ["candidates"]
                answer = AIMessage(
                    content="",
                    tool_calls=[python_call("repair", "print(data.qed)", ids)],
                )
            else:
                answer = AIMessage(content="\n".join(key for key, _ in ranking))
            self.responses = [answer]
            return super()._generate(messages, stop, run_manager, **kwargs)

    model = RepairModel(responses=[])
    result = await _evaluate(case, model, [export, create_run_python_tool(execute)])
    assert len(exports) == 1 and exports[0].startswith("upload_")
    assert [m.name for m in model.received[0] if isinstance(m, ToolMessage)] == [
        "export_properties",
        "run_python",
    ]
    assert result.repair_attempts == 1 and result.tools_correct
    assert result.success is use_exported_id
    assert result.failure_type == (None if use_exported_id else "invalid_args")


@pytest.mark.parametrize(
    ("extra_line", "top_qed", "failure"),
    [
        ("saved", "0.7868", None),
        ("CHEMBL9999999 0.1000", "0.7868", "rule_failed"),
        ("saved", "0.78680", "rule_failed"),
        ("saved", "0.7869", "rule_failed"),
    ],
)
def test_h03_stdout_checks_ranked_rows_and_accepts_status_line(
    python_call, extra_line, top_qed, failure
):
    case = next(c for c in load_cases(SELF_HEALING_DIR).cases if c.id == "H03")
    dataset_id = "upload_derived"
    case = case.bind_property_export(dataset_id)
    rows = [
        f"CHEMBL5397175 {top_qed}",
        "CHEMBL5439404 0.7822",
        "CHEMBL5434277 0.7817",
        "CHEMBL5423732 0.7544",
        "CHEMBL5411433 0.7526",
        "CHEMBL5436538 0.7306",
        "CHEMBL5425389 0.7196",
        "CHEMBL5431716 0.7020",
        "CHEMBL5399669 0.6861",
        "CHEMBL5417255 0.6382",
    ]
    stdout = "\n".join([*rows, extra_line, ""])
    result = score_case(
        case,
        [
            AIMessage(
                content="", tool_calls=[python_call("fix", "pass", [dataset_id])]
            ),
            ToolMessage(
                content="",
                name="run_python",
                tool_call_id="fix",
                artifact={
                    "structured_content": {
                        "stdout": stdout,
                        "files": [
                            {"filename": "mw-logp.png", "media_type": "image/png"}
                        ],
                    }
                },
            ),
            AIMessage(content="\n".join(row.split()[0] for row in rows)),
        ],
    )
    assert result.failure_type == failure
    if failure:
        (rule_failure,) = result.error_detail["rules"]
        assert rule_failure["rule"]["path"] == "stdout"
        assert stdout in rule_failure["detail"]
