from decimal import Decimal
from uuid import uuid4

from agent.persistence import open_connection_pool
from eval.retrieval.evaluation import RankingScore
from eval.retrieval.storage import finish_run, read_report, save_result, start_run
from eval.storage import setup_schema


async def test_retrieval_report_averages_query_recall_and_mrr_per_method(database_url):
    run_id = uuid4()
    first, second = uuid4().hex, uuid4().hex
    citations = ["pubmed:1:0", "pubmed:1:1", "pubmed:2:0"]
    async with (
        open_connection_pool(database_url) as pool,
        pool.connection() as connection,
    ):
        await setup_schema(connection)
        async with connection.transaction(force_rollback=True):
            await start_run(connection, run_id, "fixture-commit", {"result_limit": 8})
            for method, first_score, second_score in (
                ("dense", (2, 1), (None, 0)),
                ("hybrid", (1, 2), (None, 0)),
                ("hybrid_rerank", (1, 1), (2, 1)),
            ):
                await save_result(
                    connection,
                    run_id,
                    first,
                    method,
                    RankingScore(*first_score, 2, citations),
                )
                await save_result(
                    connection,
                    run_id,
                    second,
                    method,
                    RankingScore(*second_score, 1, citations),
                )
            await finish_run(connection, run_id)
            report = await read_report(connection, run_id)

    assert report.finished
    assert [
        (row["method"], row["query_count"], row["recall_at_8"], row["mrr"])
        for row in report.summary
    ] == [
        ("dense", 2, Decimal("0.25"), Decimal("0.25")),
        ("hybrid", 2, Decimal("0.50"), Decimal("0.50")),
        ("hybrid_rerank", 2, Decimal("0.75"), Decimal("0.75")),
    ]
    assert len(report.details) == 6
    assert {row["query_id"] for row in report.details} == {first, second}
    assert all(row["citation_ids"] == citations for row in report.details)
