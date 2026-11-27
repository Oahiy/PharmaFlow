import json
import re
from collections import Counter
from hashlib import sha256
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from eval.retrieval.dataset import RetrievalQuery, load_qrels, load_queries
from eval.retrieval.evaluation import score_ranking, write_pool
from rag.chunks import Chunk, parse_chunks
from rag.config import CHUNKS_PATH
from rag.retrieval import RetrievalResult, Retriever, SearchHit


def chunk(source_id: int, part: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"pubmed:{source_id}:{part}",
        source_type="pubmed",
        source_id=str(source_id),
        title="A fixture abstract",
        year=None,
        section="Abstract",
        text="Fixture evidence.",
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{source_id}/",
    )


@pytest.fixture
def query():
    return RetrievalQuery(
        id="RX",
        category="pubmed",
        question_zh="文献怎么说？",
        query_en="trial safety outcomes",
    )


def test_retrieval_query_categories():
    suite = load_queries()

    assert len(suite.queries) == 15
    assert Counter(query.category for query in suite.queries) == {
        "pubmed": 5,
        "trial": 4,
        "label": 4,
        "alias": 2,
    }


def _sixgrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[^\W_]+", text.casefold())
    return {tuple(words[i : i + 6]) for i in range(len(words) - 5)}


def test_english_queries_share_no_six_consecutive_words_with_corpus():
    by_query = {query.id: _sixgrams(query.query_en) for query in load_queries().queries}
    for evidence in parse_chunks(CHUNKS_PATH.read_bytes()):
        in_chunk = _sixgrams(evidence.text)
        for query_id, in_query in by_query.items():
            overlaps = in_query & in_chunk
            assert not overlaps, (
                f"{query_id} shares {sorted(overlaps)} with {evidence.chunk_id}"
            )


def test_recall_counts_each_unit_once_and_mrr_uses_the_first_matching_rank():
    ranking = [chunk(10), chunk(1), chunk(1, 1), chunk(3)]
    score = score_ranking(ranking, frozenset({"pubmed:1", "pubmed:2", "pubmed:3"}))

    assert score.first_relevant_rank == 2
    assert score.hit_count == 2 and score.relevant_count == 3
    assert score.recall_at_8 == pytest.approx(2 / 3)
    assert score.reciprocal_rank == 0.5


def test_metrics_are_zero_if_first_relevant_result_falls_past_rank_eight():
    ranking = [chunk(i) for i in range(1, 10)]
    score = score_ranking(ranking, frozenset({"pubmed:9"}))

    assert score.first_relevant_rank is None
    assert score.recall_at_8 == score.reciprocal_rank == 0
    assert score.citation_ids == [f"pubmed:{i}:0" for i in range(1, 9)]


def test_load_qrels_resolves_units_and_records_input_digest(tmp_path, query):
    path = tmp_path / "qrels.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "RX": [
                    {
                        "unit_id": "pubmed:1",
                        "reason": "Reports both arms",
                        "source": "pool",
                    },
                    {
                        "unit_id": "pubmed:2",
                        "reason": "Supports the unpooled endpoint",
                        "source": "corpus",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    judgments = load_qrels(path, [query], [chunk(1), chunk(1, 1), chunk(2)])

    assert judgments.sha256 == sha256(path.read_bytes()).hexdigest()
    assert judgments.units_by_query == {"RX": frozenset({"pubmed:1", "pubmed:2"})}


@pytest.mark.parametrize(
    ("contents", "error", "message"),
    [
        ({}, ValueError, r"missing \['RX'\]"),
        ({"RX": []}, ValidationError, "at least 1 item"),
        (
            {"RX": [{"unit_id": "pubmed:9", "reason": "Example", "source": "pool"}]},
            ValueError,
            "unknown corpus units",
        ),
        (
            {"RX": [{"unit_id": "pubmed:1:0", "reason": "Example", "source": "pool"}]},
            ValueError,
            "unknown corpus units",
        ),
        (
            {"RX": [{"unit_id": "pubmed:1", "reason": "Example"}]},
            ValidationError,
            "source",
        ),
        (
            {
                "RX": [{"unit_id": "pubmed:1", "reason": "Example", "source": "pool"}]
                * 2
            },
            ValueError,
            "duplicate relevant units",
        ),
    ],
)
def test_qrels_reject_incomplete_or_invalid_judgments(
    tmp_path, query, contents, error, message
):
    path = tmp_path / "qrels.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")

    with pytest.raises(error, match=message):
        load_qrels(path, [query], [chunk(1)])


def test_pool_deduplicates_chunks_and_hides_methods_in_a_reproducible_order(
    tmp_path, query
):
    fixtures = {
        "dense": (chunk(1), chunk(1, 1)),
        "hybrid": (chunk(1), chunk(2)),
        "hybrid_rerank": (chunk(3), chunk(1)),
    }

    def search(text, method, *, limit):
        assert limit == 10
        return RetrievalResult(
            text, text, [SearchHit(evidence, 1.0) for evidence in fixtures[method]]
        )

    retriever = Mock(spec=Retriever)
    retriever.search.side_effect = search
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"

    assert write_pool(first, [query], retriever) == 4
    assert write_pool(second, [query], retriever) == 4
    assert first.read_bytes() == second.read_bytes()
    assert [call.args for call in retriever.search.call_args_list] == [
        (query.query_en, method) for method in fixtures
    ] * 2
    rows = [json.loads(line) for line in first.read_bytes().splitlines()]
    assert {row["chunk"]["chunk_id"] for row in rows} == {
        "pubmed:1:0",
        "pubmed:1:1",
        "pubmed:2:0",
        "pubmed:3:0",
    }
    assert all(
        set(row) == {"query_id", "relevance_unit", "chunk"} and row["query_id"] == "RX"
        for row in rows
    )
    assert all(
        row["relevance_unit"] == row["chunk"]["chunk_id"].rsplit(":", 1)[0]
        for row in rows
    )
