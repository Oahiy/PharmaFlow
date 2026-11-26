"""Render results that have already been computed by PostgreSQL."""

from collections.abc import Iterable, Sequence

from psycopg.rows import DictRow


def print_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join("NULL" if value is None else str(value) for value in row)
            + " |"
        )


def print_summary(rows: list[DictRow]) -> None:
    if not rows:
        raise ValueError("没有找到评测运行。")
    print(f"运行状态 {'完整' if rows[0]['complete'] else '未完成'}")
    print_table(
        (
            "类别",
            "计划",
            "已入库",
            "可评分",
            "回放未命中",
            "缺 token",
            "工具选择",
            "参数合法",
            "任务成功",
            "平均输入 token",
            "平均耗时 (ms)",
        ),
        (
            (
                row["category"] if row["category"] is not None else "全部",
                row["planned_count"],
                row["stored_count"],
                row["scorable_count"],
                row["replay_miss_count"],
                row["missing_tokens_count"],
                *(
                    f"{row[key]:.1%}" if row[key] is not None else None
                    for key in ("tool_accuracy", "parameter_accuracy", "success_rate")
                ),
                *(
                    f"{row[key]:.1f}" if row[key] is not None else None
                    for key in ("average_input_tokens", "average_elapsed_ms")
                ),
            )
            for row in rows
        ),
    )
