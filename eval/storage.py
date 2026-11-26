"""Persist one evaluation at a time and aggregate its results in PostgreSQL."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from pydantic import JsonValue

from .results import CaseResult, Category

_SQL_DIR = Path(__file__).resolve().parent

# Used at finalization and when checking a finished run before comparison.
COVERAGE_SQL = """
    SELECT
        (SELECT COUNT(*) FROM jsonb_to_recordset(run.config->'cases')
            AS planned(case_id text)) AS planned_count,
        (SELECT COUNT(*) FROM eval.results AS result
            WHERE result.run_id = run.run_id) AS stored_count,
        (SELECT COUNT(*) FROM jsonb_to_recordset(run.config->'cases')
            AS planned(case_id text)
            JOIN eval.results AS result
                ON result.run_id = run.run_id AND result.case_id = planned.case_id
        ) AS matched_count
"""


class PlannedCase(TypedDict):
    case_id: str
    category: Category


async def setup_schema(conn: AsyncConnection[DictRow]) -> None:
    async with conn.transaction():
        # A fresh database can be initialized by concurrent evaluation runs.
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))", ("pharmaflow.eval.schema",)
        )
        await conn.execute((_SQL_DIR / "schema.sql").read_text(encoding="utf-8"))


async def start_run(
    conn: AsyncConnection[DictRow],
    *,
    run_id: UUID,
    user_id: UUID,
    tag: str,
    git_commit: str,
    config: dict[str, JsonValue],
    cases: Sequence[PlannedCase],
) -> None:
    async with conn.transaction(), conn.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO eval.cases (case_id, category) VALUES (%(case_id)s, %(category)s)
            ON CONFLICT (case_id) DO UPDATE SET category = EXCLUDED.category
            """,
            sorted(cases, key=lambda case: case["case_id"]),
        )
        await cursor.execute(
            """
            INSERT INTO eval.runs (run_id, user_id, tag, git_commit, config)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, user_id, tag, git_commit, Jsonb({**config, "cases": list(cases)})),
        )


async def save_result(
    conn: AsyncConnection[DictRow], run_id: UUID, result: CaseResult
) -> None:
    await conn.execute(
        """
        INSERT INTO eval.results (
            run_id, case_id, thread_id, trace_id, success, tools_correct, params_valid,
            input_tokens, elapsed_ms, judge_score, failure_type, error_detail, repair_attempts
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            result.case_id,
            result.thread_id,
            result.trace_id,
            result.success,
            result.tools_correct,
            result.params_valid,
            result.input_tokens,
            result.elapsed_ms,
            result.judge_score,
            result.failure_type,
            Jsonb(result.error_detail) if result.error_detail is not None else None,
            result.repair_attempts,
        ),
    )


async def run_coverage(conn: AsyncConnection[DictRow], run_id: UUID) -> DictRow:
    cursor = await conn.execute(
        f"""
        SELECT run.finished_at, run.config, coverage.*
        FROM eval.runs AS run CROSS JOIN LATERAL ({COVERAGE_SQL}) AS coverage
        WHERE run.run_id = %s
        """,
        (run_id,),
    )
    run = await cursor.fetchone()
    if run is None:
        raise ValueError(f"评测运行 {run_id} 不存在。")
    return run


def has_full_coverage(run: DictRow) -> bool:
    return run["planned_count"] == run["stored_count"] == run["matched_count"]


async def finish_run(conn: AsyncConnection[DictRow], run_id: UUID) -> None:
    async with conn.transaction():
        run = await run_coverage(conn, run_id)
        if not has_full_coverage(run):
            raise ValueError(
                f"评测运行 {run_id} 的结果未覆盖计划，"
                f"计划 {run['planned_count']}，已入库 {run['stored_count']}，"
                f"匹配 {run['matched_count']}。"
            )
        await conn.execute(
            "UPDATE eval.runs SET finished_at = clock_timestamp() "
            "WHERE run_id = %s AND finished_at IS NULL",
            (run_id,),
        )


async def read_summary(conn: AsyncConnection[DictRow], run_id: UUID) -> list[DictRow]:
    sql = (_SQL_DIR / "summary.sql").read_text(encoding="utf-8")
    return await (await conn.execute(sql, (run_id,))).fetchall()
