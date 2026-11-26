import json
import shutil
from uuid import UUID

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from openai import APIConnectionError, RateLimitError

from agent.config import AgentSettings, LLMSettings
from agent.persistence import open_connection_pool
from eval.cases import CASE_DIRECTORY, load_cases
from eval.judge import JudgeParseError, JudgeVerdict
from eval.provenance import recordings_digest
from eval.runner import run_suite
from eval.storage import read_summary
from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR


def reply(content, input_tokens, calls=()):
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 1,
            "total_tokens": input_tokens + 1,
        },
        tool_calls=[
            {"id": step.id, "name": step.tool, "args": {"request": step.request}}
            for step in calls
        ],
    )


@pytest.mark.parametrize(
    "first_failure",
    [None, "agent_error", "judge_parse_error", "judge_below_threshold"],
)
async def test_runner_records_two_cases_and_isolates_checkpoints(
    database_url, tmp_path, monkeypatch, fake_chat_model, first_failure
):
    # The ready-to-run suite must use its explicit recordings even with a live API environment.
    monkeypatch.setenv("LIT_MCP_MODE", "live")
    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(tmp_path / "wrong-path"))
    for case_id in ("E03", "S01"):
        shutil.copyfile(
            CASE_DIRECTORY / f"{case_id}.yaml", tmp_path / f"{case_id}.yaml"
        )
    suite = load_cases(tmp_path)
    empty_results, properties = suite.cases
    secret = "dummy-llm-credential-only-for-eval-test"
    fail_first_model_call = first_failure == "agent_error"
    settings = AgentSettings(
        database_url=database_url,
        llm=LLMSettings(
            model="offline-test-model",
            base_url="https://example.invalid/v1?key=dummy-url-credential",
            api_key=secret,
        ),
    )

    class FixtureModel(fake_chat_model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if fail_first_model_call and messages[-1].text == empty_results.question:
                raise RuntimeError(f"模拟模型错误 {secret}")
            return super()._generate(messages, stop, run_manager, **kwargs)

    responses = [
        reply("", 19, properties.accepted_paths[0]),
        reply("CCO 的分子量为 46.07 Da，HBD 1，HBA 1。", 25),
    ]
    if not fail_first_model_call:
        responses[:0] = [
            reply("", 100, empty_results.accepted_paths[0]),
            reply("本次 PubMed 没有匹配文献，openFDA 也没有匹配说明书。", 125),
        ]
    model = FixtureModel(responses=responses)
    traces = {}
    judged = []

    class TraceRecorder(BaseCallbackHandler):
        def __init__(self):
            self.metadata = []

        def on_chain_start(
            self, serialized, inputs, *, parent_run_id, metadata, **kwargs
        ):
            if parent_run_id is None:
                self.metadata.append(metadata)

    def make_callback(trace_id):
        callback = TraceRecorder()
        traces[trace_id] = callback
        return callback

    async def judge(case, answer, evidence):
        judged.append(case.id)
        assert "没有匹配文献" in answer
        assert {key: value["total_count"] for key, value in evidence.items()} == {
            "articles": 0,
            "label": 0,
        }
        if first_failure == "judge_parse_error":
            raise JudgeParseError("模拟解析错误 dummy-url-credential")
        return JudgeVerdict(
            coverage=True,
            grounded=True,
            correct_scope=first_failure != "judge_below_threshold",
            reason="已检查零结果。",
        )

    run_id, summary = await run_suite(
        settings,
        tag="v1",
        model=model,
        judge=judge,
        callback_factory=make_callback,
        case_directory=tmp_path,
    )
    assert summary[0]["complete"]
    assert summary[0]["stored_count"] == 2
    assert summary[0]["success_rate"] == (1 if first_failure is None else 0.5)

    async with open_connection_pool(database_url) as pool, pool.connection() as conn:
        run = await (
            await conn.execute("SELECT * FROM eval.runs WHERE run_id = %s", (run_id,))
        ).fetchone()
        rows = await (
            await conn.execute(
                "SELECT * FROM eval.results WHERE run_id = %s ORDER BY case_id",
                (run_id,),
            )
        ).fetchall()
        thread_ids = [str(row["thread_id"]) for row in rows]
        checkpoints = await (
            await conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id = ANY(%s)",
                (thread_ids,),
            )
        ).fetchall()

    assert isinstance(run["user_id"], UUID)
    assert run["tag"] == "v1" and run["finished_at"] >= run["started_at"]
    recorded = run["config"]
    assert recorded["git_commit"] == run["git_commit"]
    assert isinstance(recorded["git_dirty"], bool)
    assert recorded["base_url"] == "https://example.invalid/v1"
    assert recorded["cases_sha256"] == suite.sha256
    assert recorded["recordings_sha256"] == recordings_digest(DEFAULT_RECORDINGS_DIR)
    assert recorded["lit_mode"] == "replay"
    assert recorded["recordings_dir"] == str(DEFAULT_RECORDINGS_DIR)
    assert recorded["judge"]["method"] == "function_calling"
    assert recorded["judge"]["thresholds"] == {"E03": 3}
    assert secret not in json.dumps(recorded)
    assert "dummy-url-credential" not in json.dumps(recorded)
    assert database_url not in json.dumps(recorded)
    assert len(set(thread_ids)) == 2
    assert {row["thread_id"] for row in checkpoints} == set(thread_ids)
    for row in rows:
        (metadata,) = traces[row["trace_id"]].metadata
        assert metadata["langfuse_session_id"] == str(row["thread_id"])
        assert metadata["langfuse_user_id"] == str(run["user_id"])
        assert metadata["langfuse_tags"] == ["v1", run["git_commit"]]
        assert row["elapsed_ms"] >= 0
    e03, s01 = rows
    assert s01["case_id"] == "S01" and s01["success"]
    assert s01["input_tokens"] == 44 and s01["judge_score"] is None
    assert e03["case_id"] == "E03"
    assert e03["failure_type"] == first_failure
    assert e03["success"] is (first_failure is None)
    if fail_first_model_call:
        assert e03["input_tokens"] is None and e03["params_valid"] is None
        assert e03["error_detail"]["exception"]["type"] == "RuntimeError"
        assert e03["error_detail"]["exception"]["message"] == "模拟模型错误 [redacted]"
        assert "RuntimeError" in e03["error_detail"]["exception"]["traceback"]
        assert secret not in json.dumps(e03["error_detail"])
        assert judged == []
    else:
        assert e03["tools_correct"] and e03["params_valid"]
        assert e03["input_tokens"] == 225
        assert judged == ["E03"]
        if first_failure == "judge_parse_error":
            assert e03["judge_score"] is None
            assert e03["error_detail"]["exception"]["type"] == "JudgeParseError"
            assert (
                e03["error_detail"]["exception"]["message"] == "模拟解析错误 [redacted]"
            )
        elif first_failure == "judge_below_threshold":
            assert e03["judge_score"] == 2
            assert not e03["error_detail"]["judge"]["correct_scope"]
        else:
            assert e03["judge_score"] == 3


@pytest.mark.parametrize("stage", ["agent", "judge"])
async def test_model_service_error_leaves_partial_run(
    database_url, tmp_path, capsys, fake_chat_model, stage
):
    for case_id in ("E02", "E03"):
        shutil.copyfile(
            CASE_DIRECTORY / f"{case_id}.yaml", tmp_path / f"{case_id}.yaml"
        )
    _, empty_results = load_cases(tmp_path).cases
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")

    class FailingModel(fake_chat_model):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if stage == "agent" and messages[-1].text == empty_results.question:
                raise RateLimitError(
                    "服务达到调用限额",
                    response=httpx.Response(429, request=request),
                    body=None,
                )
            return super()._generate(messages, stop, run_manager, **kwargs)

    async def judge(case, answer, evidence):
        assert stage == "judge"
        raise APIConnectionError(request=request)

    settings = AgentSettings(
        database_url=database_url,
        llm=LLMSettings(
            model="offline-test-model",
            base_url="https://example.invalid",
            api_key="offline-key",
        ),
    )
    model = FailingModel(
        responses=[
            reply("编号格式错误。请提供 NCT 加 8 位数字的编号。", 10),
            reply("", 20, empty_results.accepted_paths[0]),
            reply("本次 PubMed 和 openFDA 都没有找到匹配结果。", 25),
        ]
    )
    expected_error = RateLimitError if stage == "agent" else APIConnectionError
    with pytest.raises(expected_error):
        await run_suite(
            settings,
            tag="test-interrupted",
            model=model,
            judge=judge,
            callback_factory=lambda trace_id: BaseCallbackHandler(),
            case_directory=tmp_path,
        )

    (line,) = capsys.readouterr().out.splitlines()
    run_id = UUID(line.removeprefix("run_id "))
    async with open_connection_pool(database_url) as pool, pool.connection() as conn:
        summary, *_ = await read_summary(conn, run_id)
        rows = await (
            await conn.execute(
                "SELECT case_id, success FROM eval.results WHERE run_id = %s",
                (run_id,),
            )
        ).fetchall()
    assert rows == [{"case_id": "E02", "success": True}]
    assert not summary["complete"]
    assert (summary["planned_count"], summary["stored_count"]) == (2, 1)
