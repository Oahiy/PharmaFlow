from uuid import uuid4

import pytest

from agent.persistence import open_connection_pool
from eval.compare import read_comparison
from eval.results import CaseResult
from eval.storage import finish_run, save_result, setup_schema, start_run


@pytest.mark.parametrize(
    "difference",
    [
        None,
        "cases_sha256",
        "recordings_sha256",
        "system_prompt_sha256",
        "tool_schema_sha256",
        "corpus_sha256",
        "unfinished",
        "missing_result",
    ],
)
async def test_comparison_checks_inputs_and_derives_per_case_deltas(
    database_url, difference
):
    first_id, second_id = uuid4(), uuid4()
    prefix = uuid4().hex
    a, b = f"{prefix}_a", f"{prefix}_b"
    plan = [
        {"case_id": a, "category": "multi_hop"},
        {"case_id": b, "category": "single_tool"},
    ]
    config = {"cases_sha256": "cases", "recordings_sha256": "recordings"}
    input_digests = {
        "system_prompt_sha256": "prompt",
        "tool_schema_sha256": "tools",
        "corpus_sha256": "corpus",
    }

    async with open_connection_pool(database_url) as pool, pool.connection() as conn:
        await setup_schema(conn)
        async with conn.transaction(force_rollback=True):
            for run_id, results in (
                (
                    first_id,
                    [(a, 3, 60, 120, None), (b, None, None, 100, "replay_miss")],
                ),
                (
                    second_id,
                    [(a, 1, 75, 95, "judge_below_threshold"), (b, None, 30, 110, None)],
                ),
            ):
                run_config = config.copy()
                if difference in input_digests or (
                    run_id == second_id and difference is None
                ):
                    run_config.update(input_digests)
                if run_id == second_id and difference in run_config:
                    run_config[difference] = "other"
                await start_run(
                    conn,
                    run_id=run_id,
                    user_id=uuid4(),
                    tag="v1",
                    git_commit="abc",
                    config=run_config,
                    cases=plan,
                )
                for case, score, tokens, ms, failure in results:
                    if (
                        run_id == second_id
                        and case == b
                        and difference == "missing_result"
                    ):
                        continue
                    await save_result(
                        conn,
                        run_id,
                        CaseResult(
                            case_id=case,
                            thread_id=uuid4(),
                            trace_id=uuid4().hex,
                            tools_correct=True,
                            params_valid=True,
                            input_tokens=tokens,
                            elapsed_ms=ms,
                            judge_score=score,
                            failure_type=failure,
                        ),
                    )
                if run_id == second_id and difference == "missing_result":
                    await conn.execute(
                        "UPDATE eval.runs SET finished_at = clock_timestamp() WHERE run_id = %s",
                        (run_id,),
                    )
                elif run_id != second_id or difference != "unfinished":
                    await finish_run(conn, run_id)

            if difference is not None and difference not in input_digests:
                expected = (
                    difference if difference in config else f"{second_id} 未完整结束"
                )
                with pytest.raises(ValueError, match=expected):
                    await read_comparison(conn, first_id, second_id)
                return

            comparison = await read_comparison(conn, first_id, second_id)
            assert comparison.model_input_matches == {
                field: None if difference is None else field != difference
                for field in input_digests
            }
            primary, unscorable = comparison.rows
            assert primary == {
                "case_id": a,
                "first_success": True,
                "second_success": False,
                "success_delta": -1,
                "first_failure_type": None,
                "second_failure_type": "judge_below_threshold",
                "first_judge_score": 3,
                "second_judge_score": 1,
                "judge_delta": -2,
                "first_input_tokens": 60,
                "second_input_tokens": 75,
                "input_tokens_delta": 15,
                "first_elapsed_ms": 120,
                "second_elapsed_ms": 95,
                "elapsed_ms_delta": -25,
            }
            assert unscorable["case_id"] == b
            assert unscorable["first_failure_type"] == "replay_miss"
            assert unscorable["success_delta"] is None
            assert unscorable["input_tokens_delta"] is None
            assert unscorable["judge_delta"] is None
            assert unscorable["elapsed_ms_delta"] == 10
