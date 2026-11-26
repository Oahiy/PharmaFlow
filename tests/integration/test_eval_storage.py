from decimal import Decimal
from uuid import uuid4

import pytest

from agent.persistence import open_connection_pool
from eval.results import CaseResult
from eval.storage import finish_run, read_summary, save_result, setup_schema, start_run


async def test_summary_counts_partial_runs_and_null_denominators(database_url):
    run_id, user_id = uuid4(), uuid4()
    suffix = uuid4().hex
    a, b, c, d = (f"{name}_{suffix}" for name in "abcd")
    plan = [
        {"case_id": a, "category": "single_tool"},
        {"case_id": b, "category": "single_tool"},
        {"case_id": c, "category": "exception"},
        {"case_id": d, "category": "multi_hop"},
    ]

    async with open_connection_pool(database_url) as pool, pool.connection() as conn:
        await setup_schema(conn)
        await setup_schema(conn)
        async with conn.transaction(force_rollback=True):
            await start_run(
                conn,
                run_id=run_id,
                user_id=user_id,
                tag="v1",
                git_commit="abc",
                config={},
                cases=plan,
            )
            for case_id, tools, params, tokens, ms, failure in (
                (a, True, True, 120, 200, None),
                (b, True, True, 60, 100, "replay_miss"),
                (c, False, None, None, 300, "wrong_tool"),
            ):
                await save_result(
                    conn,
                    run_id,
                    CaseResult(
                        case_id=case_id,
                        thread_id=uuid4(),
                        trace_id=uuid4().hex,
                        tools_correct=tools,
                        params_valid=params,
                        input_tokens=tokens,
                        elapsed_ms=ms,
                        failure_type=failure,
                    ),
                )
            # Syncing a later case revision must not change this run's categories.
            await start_run(
                conn,
                run_id=uuid4(),
                user_id=uuid4(),
                tag="v2",
                git_commit="abc",
                config={},
                cases=[{"case_id": a, "category": "multi_hop"}],
            )

            groups = {row["category"]: row for row in await read_summary(conn, run_id)}
            assert groups[None] == {
                "run_id": run_id,
                "complete": False,
                "category": None,
                "planned_count": 4,
                "stored_count": 3,
                "scorable_count": 2,
                "replay_miss_count": 1,
                "missing_tokens_count": 1,
                "tool_accuracy": pytest.approx(Decimal(2) / 3),
                "parameter_accuracy": Decimal(1),
                "success_rate": Decimal("0.5"),
                "average_input_tokens": Decimal(90),
                "average_elapsed_ms": Decimal(200),
            }
            single = groups["single_tool"]
            assert (
                single["planned_count"],
                single["stored_count"],
                single["scorable_count"],
            ) == (2, 2, 1)
            assert single["success_rate"] == 1
            assert groups["exception"]["parameter_accuracy"] is None
            empty = groups["multi_hop"]
            assert (
                empty["planned_count"],
                empty["stored_count"],
                empty["scorable_count"],
            ) == (1, 0, 0)
            assert empty["missing_tokens_count"] == 0
            assert (
                empty["tool_accuracy"]
                is empty["success_rate"]
                is empty["average_elapsed_ms"]
                is None
            )

            with pytest.raises(ValueError, match=r"计划 4，已入库 3，匹配 3"):
                await finish_run(conn, run_id)
            await save_result(
                conn,
                run_id,
                CaseResult(
                    case_id=d,
                    thread_id=uuid4(),
                    trace_id=uuid4().hex,
                    tools_correct=True,
                    params_valid=None,
                    input_tokens=240,
                    elapsed_ms=400,
                ),
            )
            await finish_run(conn, run_id)
            complete, *_ = await read_summary(conn, run_id)
            assert complete["complete"]
            assert complete["stored_count"] == complete["planned_count"] == 4
            assert complete["success_rate"] == pytest.approx(Decimal(2) / 3)
