"""Compare corresponding cases from two completed evaluation runs."""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow

from agent.config import DatabaseSettings, load_environment

from .report import print_table
from .storage import has_full_coverage, run_coverage

MODEL_INPUT_DIGESTS = {
    "system_prompt_sha256": "系统提示词",
    "tool_schema_sha256": "工具定义",
    "corpus_sha256": "检索语料",
}


@dataclass(frozen=True)
class Comparison:
    rows: list[DictRow]
    model_input_matches: dict[str, bool | None]


async def read_comparison(
    conn: AsyncConnection[DictRow], first_id: UUID, second_id: UUID
) -> Comparison:
    first = await run_coverage(conn, first_id)
    second = await run_coverage(conn, second_id)
    for run_id, run in ((first_id, first), (second_id, second)):
        if run["finished_at"] is None or not has_full_coverage(run):
            raise ValueError(
                f"评测运行 {run_id} 未完整结束。"
                f"计划 {run['planned_count']}，已入库 {run['stored_count']}，"
                f"匹配 {run['matched_count']}。"
            )
    first_config, second_config = first["config"], second["config"]
    differences = [
        field
        for field in ("cases_sha256", "recordings_sha256")
        if first_config[field] != second_config[field]
    ]
    if differences:
        raise ValueError(
            f"运行 {first_id} 与 {second_id} 的摘要不一致（{', '.join(differences)}）。"
        )
    model_input_matches: dict[str, bool | None] = {}
    for field in MODEL_INPUT_DIGESTS:
        before, after = first_config.get(field), second_config.get(field)
        model_input_matches[field] = (
            before == after if before is not None and after is not None else None
        )
    sql = Path(__file__).with_suffix(".sql").read_text(encoding="utf-8")
    rows = await (await conn.execute(sql, (first_id, second_id))).fetchall()
    return Comparison(rows=rows, model_input_matches=model_input_matches)


COLUMNS = {
    "case_id": "用例",
    "first_success": "成功 1",
    "second_success": "成功 2",
    "success_delta": "Δ成功",
    "first_failure_type": "失败类型 1",
    "second_failure_type": "失败类型 2",
    "first_judge_score": "Judge 1",
    "second_judge_score": "Judge 2",
    "judge_delta": "ΔJudge",
    "first_input_tokens": "输入 token 1",
    "second_input_tokens": "输入 token 2",
    "input_tokens_delta": "Δtoken",
    "first_elapsed_ms": "耗时 1 (ms)",
    "second_elapsed_ms": "耗时 2 (ms)",
    "elapsed_ms_delta": "Δ耗时 (ms)",
}


async def compare(first_id: UUID, second_id: UUID) -> None:
    from agent.persistence import open_connection_pool

    settings = DatabaseSettings.from_env()
    async with (
        open_connection_pool(settings.database_url.get_secret_value()) as pool,
        pool.connection() as conn,
    ):
        comparison = await read_comparison(conn, first_id, second_id)
    print(f"第 1 轮 {first_id}\n第 2 轮 {second_id}")
    statuses = {True: "一致", False: "不同", None: "缺少记录"}
    print(
        "，".join(
            f"{label}摘要 {statuses[comparison.model_input_matches[field]]}"
            for field, label in MODEL_INPUT_DIGESTS.items()
        )
    )
    print_table(
        tuple(COLUMNS.values()),
        (tuple(row[key] for key in COLUMNS) for row in comparison.rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_id", type=UUID)
    parser.add_argument("second_id", type=UUID)
    options = parser.parse_args()
    load_environment()
    try:
        asyncio.run(compare(options.first_id, options.second_id))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
