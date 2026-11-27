"""Generate a blinded candidate pool, run retrieval evaluation, or show a run."""

import argparse
import asyncio
from contextlib import closing
from hashlib import sha256
from uuid import UUID, uuid4

from pydantic import JsonValue

from agent.config import DatabaseSettings, git_commit, load_environment
from agent.persistence import open_connection_pool
from eval.report import print_table
from eval.storage import setup_schema
from rag.chunks import parse_chunks
from rag.config import CHUNKS_PATH, RagSettings
from rag.embedding import MODEL_NAMES
from rag.retrieval import (
    ALIAS_GROUPS,
    METHODS,
    PREFETCH_LIMIT,
    RESULT_LIMIT,
    RRF_LIMIT,
    load_retriever,
)

from .dataset import POOL_PATH, QRELS_PATH, load_qrels, load_queries
from .evaluation import RankingScore, score_ranking, write_pool
from .storage import RetrievalReport, finish_run, read_report, save_result, start_run


def build_pool() -> None:
    suite = load_queries()
    corpus_sha256 = sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    with closing(load_retriever(RagSettings.from_env(), corpus_sha256)) as retriever:
        count = write_pool(POOL_PATH, suite.queries, retriever)
    print(f"已写入 {count} 条候选片段至 {POOL_PATH}")


def print_report(run_id: UUID, report: RetrievalReport) -> None:
    print(f"运行 {run_id} {'完整' if report.finished else '未完成'}")
    print_table(
        ("方法", "查询数", "Recall@8", "MRR"),
        (
            (
                row["method"],
                row["query_count"],
                f"{row['recall_at_8']:.4f}",
                f"{row['mrr']:.4f}",
            )
            for row in report.summary
        ),
    )

    details = []
    for row in report.details:
        score = RankingScore(
            row["first_relevant_rank"],
            row["hit_count"],
            row["relevant_count"],
            row["citation_ids"],
        )
        details.append(
            (
                row["query_id"],
                row["method"],
                score.first_relevant_rank,
                f"{score.hit_count}/{score.relevant_count}",
                f"{score.recall_at_8:.4f}",
                f"{score.reciprocal_rank:.4f}",
                ", ".join(score.citation_ids),
            )
        )
    print()
    print_table(
        ("查询", "方法", "首个相关排名", "相关单元", "Recall@8", "倒数排名", "引用 ID"),
        details,
    )


async def evaluate() -> None:
    suite = load_queries()
    corpus = CHUNKS_PATH.read_bytes()
    judgments = load_qrels(QRELS_PATH, suite.queries, parse_chunks(corpus))
    corpus_sha256 = sha256(corpus).hexdigest()
    config: dict[str, JsonValue] = {
        "corpus_sha256": corpus_sha256,
        "queries_sha256": suite.sha256,
        "qrels_sha256": judgments.sha256,
        **MODEL_NAMES,
        "prefetch_limit": PREFETCH_LIMIT,
        "rrf_limit": RRF_LIMIT,
        "rrf_tie_break": "point_id_ascending",
        "result_limit": RESULT_LIMIT,
        "alias_groups": [list(group) for group in ALIAS_GROUPS],
    }
    database_url = DatabaseSettings.from_env().database_url.get_secret_value()
    run_id, commit = uuid4(), git_commit()
    retriever = await asyncio.to_thread(
        load_retriever, RagSettings.from_env(), corpus_sha256
    )
    try:
        async with (
            open_connection_pool(database_url) as pool,
            pool.connection() as conn,
        ):
            await setup_schema(conn)
            await start_run(conn, run_id, commit, config)
            print(f"run_id {run_id}", flush=True)
            for query in suite.queries:
                relevant = judgments.units_by_query[query.id]
                for method in METHODS:
                    result = await asyncio.to_thread(
                        retriever.search, query.query_en, method
                    )
                    score = score_ranking((hit.chunk for hit in result.hits), relevant)
                    await save_result(conn, run_id, query.id, method, score)
            await finish_run(conn, run_id)
            report = await read_report(conn, run_id)
    finally:
        await asyncio.to_thread(retriever.close)
    print_report(run_id, report)


async def summarize(run_id: UUID) -> None:
    database_url = DatabaseSettings.from_env().database_url.get_secret_value()
    async with open_connection_pool(database_url) as pool, pool.connection() as conn:
        report = await read_report(conn, run_id)
    print_report(run_id, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "pool", help="Generate pool.jsonl from the three retrieval methods."
    )
    commands.add_parser(
        "run", help="Evaluate queries against qrels.yaml and store the results."
    )
    summary = commands.add_parser(
        "summary", help="Show the metrics and ranked citations for one run."
    )
    summary.add_argument("run_id", type=UUID)
    args = parser.parse_args()

    if args.command == "pool":
        build_pool()
        return
    load_environment()
    if args.command == "run":
        asyncio.run(evaluate())
    else:
        asyncio.run(summarize(args.run_id))


if __name__ == "__main__":
    main()
