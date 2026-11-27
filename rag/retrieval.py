"""Search the Qdrant corpus using dense, hybrid, or reranked retrieval."""

import re
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from .chunks import Chunk
from .config import RagSettings
from .embedding import BM25_MODEL, DENSE_MODEL, RERANK_MODEL
from .index import (
    BM25_VECTOR,
    COLLECTION_NAME,
    DENSE_VECTOR,
    sparse_vector,
    verify_index,
)

type RetrievalMethod = Literal["dense", "hybrid", "hybrid_rerank"]
METHODS: tuple[RetrievalMethod, ...] = ("dense", "hybrid", "hybrid_rerank")

PREFETCH_LIMIT = 30
RRF_LIMIT = 30
RESULT_LIMIT = 8
ALIAS_GROUPS = (
    ("sotorasib", "AMG 510", "LUMAKRAS"),
    ("adagrasib", "MRTX849", "KRAZATI"),
)


def _alias_pattern(name: str) -> re.Pattern[str]:
    parts = re.findall(r"[a-zA-Z]+|[0-9]+", name)
    return re.compile(r"\b" + r"[\s‐‑-]*".join(parts) + r"\b", re.IGNORECASE)


_ALIAS_PATTERNS = tuple(
    tuple((name, _alias_pattern(name)) for name in group) for group in ALIAS_GROUPS
)


def expand_aliases(query: str) -> str:
    additions = []
    for group in _ALIAS_PATTERNS:
        missing = [name for name, pattern in group if pattern.search(query) is None]
        if len(missing) < len(group):
            additions.extend(missing)
    return " ".join((query, *additions))


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    expanded_query: str
    hits: list[SearchHit]


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        dense: TextEmbedding,
        bm25: SparseTextEmbedding,
        reranker: TextCrossEncoder,
        *,
        collection_name: str = COLLECTION_NAME,
    ):
        self._client = client
        self._dense = dense
        self._bm25 = bm25
        self._reranker = reranker
        self._collection_name = collection_name

    def close(self) -> None:
        self._client.close()

    def search(
        self,
        query: str,
        method: RetrievalMethod = "hybrid_rerank",
        *,
        limit: int = RESULT_LIMIT,
    ) -> RetrievalResult:
        expanded = expand_aliases(query)
        (dense_vector,) = self._dense.query_embed(expanded)
        if method == "dense":
            response = self._client.query_points(
                self._collection_name,
                query=dense_vector.tolist(),
                using=DENSE_VECTOR,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:
            (bm25_vector,) = self._bm25.query_embed(expanded)
            response = self._client.query_points(
                self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector.tolist(),
                        using=DENSE_VECTOR,
                        limit=PREFETCH_LIMIT,
                    ),
                    models.Prefetch(
                        query=sparse_vector(bm25_vector),
                        using=BM25_VECTOR,
                        limit=PREFETCH_LIMIT,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                # Both prefetches together contain at most this many points.
                # Resolve score ties before cutting the fused ranking to RRF_LIMIT.
                limit=2 * PREFETCH_LIMIT,
                with_payload=True,
            )
            points = sorted(
                response.points, key=lambda point: (-point.score, str(point.id))
            )[:RRF_LIMIT]

        ranked = [
            SearchHit(Chunk.model_validate(point.payload), point.score)
            for point in points
        ]
        if method == "hybrid_rerank" and ranked:
            scores = self._reranker.rerank(expanded, [hit.chunk.text for hit in ranked])
            ranked = sorted(
                (
                    SearchHit(hit.chunk, float(score))
                    for hit, score in zip(ranked, scores, strict=True)
                ),
                key=lambda hit: hit.score,
                reverse=True,
            )
        return RetrievalResult(query, expanded, ranked[:limit])


def load_retriever(settings: RagSettings, corpus_sha256: str) -> Retriever:
    """Load each model from the cache and open a validated Qdrant collection."""
    model_options = {"cache_dir": str(settings.cache_path), "local_files_only": True}
    dense = TextEmbedding(DENSE_MODEL, **model_options)
    bm25 = SparseTextEmbedding(BM25_MODEL, **model_options)
    reranker = TextCrossEncoder(RERANK_MODEL, **model_options)
    with ExitStack() as cleanup:
        client = cleanup.enter_context(
            closing(QdrantClient(url=str(settings.qdrant_url)))
        )
        verify_index(client, corpus_sha256)
        retriever = Retriever(client, dense, bm25, reranker)
        cleanup.pop_all()
    return retriever
