"""Start a replay evaluation or print the stored summary of a run."""

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from agent.config import AgentSettings, DatabaseSettings, load_environment


async def evaluate(tag: str, *, case_directory: Path | None = None) -> None:
    from agent.llm import create_llm
    from agent.tracing import langfuse_callback, open_langfuse

    from .cases import CASE_DIRECTORY
    from .judge import build_judge
    from .report import print_summary
    from .runner import MODEL_SERVICE_ERRORS, redact_text, run_suite, secret_values

    async with open_langfuse(required=True):
        settings = AgentSettings.from_env()
        try:
            _, summary = await run_suite(
                settings,
                tag=tag,
                model=create_llm(settings.llm),
                judge=build_judge(create_llm(settings.llm)),
                callback_factory=langfuse_callback,
                case_directory=CASE_DIRECTORY
                if case_directory is None
                else case_directory,
            )
        except MODEL_SERVICE_ERRORS as exc:
            detail = redact_text(str(exc), secret_values(settings))
            raise SystemExit(
                f"模型服务不可用，评测中止。{type(exc).__name__} {detail}"
            ) from exc
    print_summary(summary)


async def summarize(run_id: UUID) -> None:
    from agent.persistence import open_connection_pool

    from .report import print_summary
    from .storage import read_summary

    settings = DatabaseSettings.from_env()
    async with (
        open_connection_pool(settings.database_url.get_secret_value()) as pool,
        pool.connection() as conn,
    ):
        rows = await read_summary(conn, run_id)
    if not rows:
        raise SystemExit(f"评测运行 {run_id} 不存在。")
    print_summary(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tag", help="评测运行的标签，例如 v1。")
    mode.add_argument(
        "--summary", type=UUID, metavar="RUN_ID", help="查询已入库的运行汇总。"
    )
    parser.add_argument("--cases", type=Path, help="指定独立的 YAML 用例目录。")
    options = parser.parse_args()
    if options.summary is not None and options.cases is not None:
        parser.error("--cases 仅与 --tag 一起使用。")
    # Set strict checkpoint deserialization before importing LangGraph.
    load_environment()
    if options.summary is not None:
        asyncio.run(summarize(options.summary))
    else:
        if not options.tag.strip():
            parser.error("--tag 不能为空。")
        asyncio.run(evaluate(options.tag, case_directory=options.cases))


if __name__ == "__main__":
    main()
