"""Pool candidate chunks and measure relevance at the top eight ranks."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from rag.chunks import Chunk
from rag.retrieval import METHODS, RESULT_LIMIT, Retriever

from .dataset import RetrievalQuery

POOL_LIMIT = 10
POOL_SEED = 7


class PoolCandidate(BaseModel):
    query_id: str
    relevance_unit: str
    chunk: Chunk


def write_pool(
    path: Path, queries: Sequence[RetrievalQuery], retriever: Retriever
) -> int:
    rng = Random(POOL_SEED)
    rows = []
    for query in queries:
        candidates = {}
        for method in METHODS:
            result = retriever.search(query.query_en, method, limit=POOL_LIMIT)
            candidates.update((hit.chunk.chunk_id, hit.chunk) for hit in result.hits)
        shuffled = sorted(candidates.values(), key=lambda chunk: chunk.chunk_id)
        rng.shuffle(shuffled)
        rows.extend(
            PoolCandidate(
                query_id=query.id, relevance_unit=chunk.relevance_unit, chunk=chunk
            )
            for chunk in shuffled
        )

    contents = "".join(f"{row.model_dump_json()}\n" for row in rows)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    return len(rows)


@dataclass(frozen=True)
class RankingScore:
    first_relevant_rank: int | None
    hit_count: int
    relevant_count: int
    citation_ids: list[str]

    @property
    def recall_at_8(self) -> float:
        return self.hit_count / self.relevant_count

    @property
    def reciprocal_rank(self) -> float:
        return (
            1 / self.first_relevant_rank
            if self.first_relevant_rank is not None
            else 0.0
        )


def score_ranking(chunks: Iterable[Chunk], relevant: frozenset[str]) -> RankingScore:
    ranked = list(islice(chunks, RESULT_LIMIT))
    units = [chunk.relevance_unit for chunk in ranked]
    first_rank = next(
        (rank for rank, unit in enumerate(units, 1) if unit in relevant), None
    )
    return RankingScore(
        first_relevant_rank=first_rank,
        hit_count=len(relevant.intersection(units)),
        relevant_count=len(relevant),
        citation_ids=[chunk.chunk_id for chunk in ranked],
    )
