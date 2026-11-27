"""Load retrieval queries and relevance judgments from YAML snapshots."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mcp_servers.lit.models import NonBlank
from rag.chunks import Chunk

_DIRECTORY = Path(__file__).resolve().parent
QUERIES_PATH = _DIRECTORY / "queries.yaml"
QRELS_PATH = _DIRECTORY / "qrels.yaml"
POOL_PATH = _DIRECTORY / "pool.jsonl"


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: NonBlank
    category: Literal["pubmed", "trial", "label", "alias"]
    question_zh: NonBlank
    query_en: NonBlank


class RelevantUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    unit_id: NonBlank
    reason: NonBlank
    source: Literal["pool", "corpus"]


_QUERIES = TypeAdapter(Annotated[list[RetrievalQuery], Field(min_length=1)])
_QRELS = TypeAdapter(dict[str, Annotated[list[RelevantUnit], Field(min_length=1)]])


@dataclass(frozen=True)
class QuerySuite:
    sha256: str
    queries: list[RetrievalQuery]


@dataclass(frozen=True)
class Judgments:
    sha256: str
    units_by_query: dict[str, frozenset[str]]


def load_queries(path: Path = QUERIES_PATH) -> QuerySuite:
    contents = path.read_bytes()
    queries = _QUERIES.validate_python(yaml.safe_load(contents))
    duplicates = [
        query_id
        for query_id, count in Counter(query.id for query in queries).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate retrieval query IDs: {duplicates}")
    return QuerySuite(sha256(contents).hexdigest(), queries)


def load_qrels(
    path: Path,
    queries: Sequence[RetrievalQuery],
    chunks: Sequence[Chunk],
) -> Judgments:
    contents = path.read_bytes()
    qrels = _QRELS.validate_python(yaml.safe_load(contents))
    expected = {query.id for query in queries}
    actual = qrels.keys()
    if expected != actual:
        raise ValueError(
            f"Qrels query IDs: missing {sorted(expected - actual)}, unknown {sorted(actual - expected)}"
        )

    corpus_units = {chunk.relevance_unit for chunk in chunks}
    units_by_query = {}
    for query_id, records in qrels.items():
        units = frozenset(record.unit_id for record in records)
        if len(units) != len(records):
            raise ValueError(f"Qrels for {query_id} contain duplicate relevant units")
        unknown = units - corpus_units
        if unknown:
            raise ValueError(
                f"Qrels for {query_id} reference unknown corpus units: {sorted(unknown)}"
            )
        units_by_query[query_id] = units
    return Judgments(sha256(contents).hexdigest(), units_by_query)
