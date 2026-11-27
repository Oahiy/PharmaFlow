"""Persist retrieval runs and read their PostgreSQL reports."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from pydantic import JsonValue

from rag.retrieval import RetrievalMethod

from .evaluation import RankingScore


@dataclass(frozen=True)
class RetrievalReport:
    finished: bool
    summary: list[DictRow]
    details: list[DictRow]


async def start_run(
    conn: AsyncConnection[DictRow],
    run_id: UUID,
    git_commit: str,
    config: dict[str, JsonValue],
) -> None:
    await conn.execute(
        "INSERT INTO eval.retrieval_runs (run_id, git_commit, config) VALUES (%s, %s, %s)",
        (run_id, git_commit, Jsonb(config)),
    )


async def save_result(
    conn: AsyncConnection[DictRow],
    run_id: UUID,
    query_id: str,
    method: RetrievalMethod,
    score: RankingScore,
) -> None:
    await conn.execute(
        """
        INSERT INTO eval.retrieval_results (
            run_id, query_id, method, first_relevant_rank, hit_count, relevant_count, citation_ids
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            run_id,
            query_id,
            method,
            score.first_relevant_rank,
            score.hit_count,
            score.relevant_count,
            Jsonb(score.citation_ids),
        ),
    )


async def finish_run(conn: AsyncConnection[DictRow], run_id: UUID) -> None:
    await conn.execute(
        "UPDATE eval.retrieval_runs SET finished_at = clock_timestamp() WHERE run_id = %s",
        (run_id,),
    )


async def read_report(conn: AsyncConnection[DictRow], run_id: UUID) -> RetrievalReport:
    run = await (
        await conn.execute(
            "SELECT finished_at FROM eval.retrieval_runs WHERE run_id = %s", (run_id,)
        )
    ).fetchone()
    if run is None:
        raise ValueError(f"检索评测运行 {run_id} 不存在。")
    sql = Path(__file__).with_name("summary.sql").read_text(encoding="utf-8")
    summary = await (await conn.execute(sql, (run_id,))).fetchall()
    details = await (
        await conn.execute(
            """
        SELECT query_id, method, first_relevant_rank, hit_count, relevant_count, citation_ids
        FROM eval.retrieval_results WHERE run_id = %s ORDER BY query_id, method
        """,
            (run_id,),
        )
    ).fetchall()
    return RetrievalReport(run["finished_at"] is not None, summary, details)
