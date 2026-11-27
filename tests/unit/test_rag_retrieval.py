import logging
from contextlib import closing, nullcontext
from threading import get_ident
from unittest.mock import Mock
from uuid import UUID, uuid5

import httpx
import numpy as np
import pytest
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastmcp import Client
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import QueryResponse

from agent.mcp_client import lit_env_from_process
from mcp_servers.lit import server as lit_server
from mcp_servers.lit.models import SearchEvidenceResult
from mcp_servers.lit.server import create_server
from rag import retrieval
from rag.chunks import Chunk
from rag.config import RagSettings
from rag.embedding import BM25_MODEL, DENSE_MODEL, MODEL_NAMES, RERANK_MODEL
from rag.index import COLLECTION_NAME, POINT_NAMESPACE
from rag.retrieval import Retriever, expand_aliases

CORPUS_SHA256 = "1" * 64


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("lUmAkRaS hepatotoxicity", "lUmAkRaS hepatotoxicity sotorasib AMG 510"),
        ("AMG-510 rash", "AMG-510 rash sotorasib LUMAKRAS"),
        (
            "amg  510 plus KRAZATI",
            "amg  510 plus KRAZATI sotorasib LUMAKRAS adagrasib MRTX849",
        ),
        ("mrtx-849 QT interval", "mrtx-849 QT interval adagrasib KRAZATI"),
        ("LUMAKRAS amg-510 Sotorasib", "LUMAKRAS amg-510 Sotorasib"),
        ("prelumakras or MRTX8490", "prelumakras or MRTX8490"),
    ],
)
def test_alias_expansion(query, expected):
    assert expand_aliases(query) == expected


def create_collection(client: QdrantClient, **overrides):
    client.create_collection(
        COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata={
            **MODEL_NAMES,
            "corpus_sha256": CORPUS_SHA256,
            "chunk_count": 3,
            "bm25_avg_len": 3.0,
            **overrides,
        },
    )


@pytest.fixture
def index():
    with closing(QdrantClient(":memory:")) as client:
        create_collection(client)
        points = []
        for pmid, dense, bm25 in (
            ("1", [1.0, 0.0], 0.1),
            ("2", [0.8, 0.6], 0.9),
            ("3", [0.0, 1.0], 0.4),
        ):
            chunk = Chunk(
                chunk_id=f"pubmed:{pmid}:0",
                source_type="pubmed",
                source_id=pmid,
                title=f"Article {pmid}",
                year=None,
                section="Abstract",
                text=f"Evidence {pmid}",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
            points.append(
                models.PointStruct(
                    id=str(uuid5(POINT_NAMESPACE, chunk.chunk_id)),
                    vector={
                        "dense": dense,
                        "bm25": models.SparseVector(indices=[11], values=[bm25]),
                    },
                    payload=chunk.model_dump(mode="json"),
                )
            )
        client.upsert(COLLECTION_NAME, points, wait=True)
        yield client


@pytest.fixture
def models_for_query():
    dense = Mock(spec=TextEmbedding)
    dense.query_embed.return_value = [np.array([1.0, 0.0])]
    bm25 = Mock(spec=SparseTextEmbedding)
    bm25.query_embed.return_value = [SparseEmbedding.from_dict({11: 1.0})]
    reranker = Mock(spec=TextCrossEncoder)
    reranker.rerank.return_value = [3.0, 5.0, 9.0]
    return dense, bm25, reranker


def test_retrieval_uses_dense_rrf_and_cross_encoder_in_order(
    index, models_for_query, monkeypatch
):
    query = "AMG-510 toxicity"
    expanded = "AMG-510 toxicity sotorasib LUMAKRAS"
    query_points = Mock(wraps=index.query_points)
    monkeypatch.setattr(index, "query_points", query_points)
    retriever = Retriever(index, *models_for_query)

    dense = retriever.search(query, "dense", limit=2)
    assert [hit.chunk.source_id for hit in dense.hits] == ["1", "2"]
    assert query_points.call_args.kwargs["using"] == "dense"
    assert query_points.call_args.kwargs["limit"] == 2
    models_for_query[1].query_embed.assert_not_called()

    hybrid = retriever.search(query, "hybrid", limit=2)
    assert [hit.chunk.source_id for hit in hybrid.hits] == ["2", "1"]
    kwargs = query_points.call_args.kwargs
    assert [(entry.using, entry.limit) for entry in kwargs["prefetch"]] == [
        ("dense", 30),
        ("bm25", 30),
    ]
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert kwargs["limit"] == 60
    models_for_query[2].rerank.assert_not_called()

    reranked = retriever.search(query, limit=2)
    assert reranked.query == query and reranked.expanded_query == expanded
    assert [hit.chunk.source_id for hit in reranked.hits] == ["3", "1"]
    assert [hit.score for hit in reranked.hits] == [9.0, 5.0]
    models_for_query[2].rerank.assert_called_once_with(
        expanded, ["Evidence 2", "Evidence 1", "Evidence 3"]
    )
    assert [call.args for call in models_for_query[0].query_embed.call_args_list] == [
        (expanded,)
    ] * 3
    assert [call.args for call in models_for_query[1].query_embed.call_args_list] == [
        (expanded,)
    ] * 2


def test_hybrid_resolves_equal_rrf_scores_before_cutting_the_candidate_list(
    index, models_for_query, monkeypatch
):
    points = []
    for point_number in range(1, 33):
        pmid = str(100 - point_number)
        chunk = Chunk(
            chunk_id=f"pubmed:{pmid}:0",
            source_type="pubmed",
            source_id=pmid,
            title=f"Article {pmid}",
            year=None,
            section="Abstract",
            text=f"Evidence {pmid}",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )
        points.append(
            models.ScoredPoint(
                id=str(UUID(int=point_number)),
                version=0,
                score=0.5,
                payload=chunk.model_dump(mode="json"),
            )
        )
    first_response = QueryResponse(points=points)
    second_response = QueryResponse(points=list(reversed(points)))
    monkeypatch.setattr(
        index, "query_points", Mock(side_effect=[first_response, second_response] * 2)
    )
    reranker = models_for_query[2]
    reranker.rerank.side_effect = lambda _query, texts: [1.0] * len(texts)
    retriever = Retriever(index, *models_for_query)

    query = "toxicity"
    expected = [f"pubmed:{pmid}:0" for pmid in range(99, 91, -1)]
    for method in ("hybrid", "hybrid_rerank"):
        for _ in range(2):
            result = retriever.search(query, method)
            assert [hit.chunk.chunk_id for hit in result.hits] == expected
    expected_inputs = [f"Evidence {pmid}" for pmid in range(99, 69, -1)]
    assert [call.args for call in reranker.rerank.call_args_list] == [
        (query, expected_inputs),
        (query, expected_inputs),
    ]


@pytest.mark.parametrize("failure", ["qdrant", "dense", "bm25", "rerank"])
def test_retrieval_propagates_qdrant_and_model_errors(
    index, models_for_query, failure, monkeypatch
):
    cause = RuntimeError(f"original {failure} failure")
    dense, bm25, reranker = models_for_query
    if failure == "qdrant":
        monkeypatch.setattr(index, "query_points", Mock(side_effect=cause))
    elif failure == "rerank":
        reranker.rerank.side_effect = cause
    else:
        encoder = dense if failure == "dense" else bm25
        encoder.query_embed.side_effect = cause

    with pytest.raises(RuntimeError) as raised:
        Retriever(index, *models_for_query).search("sotorasib")
    assert raised.value is cause


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (None, "does not exist"),
        ({"corpus_sha256": "2" * 64}, "corpus_sha256"),
        ({}, "0 points, expected chunk_count=3"),
        ({"rerank_model": "wrong-model"}, "rerank_model='wrong-model'"),
    ],
)
def test_startup_rejects_invalid_index(overrides, message):
    with closing(QdrantClient(":memory:")) as client:
        if overrides is not None:
            create_collection(client, **overrides)
        with pytest.raises(ValueError, match=message):
            retrieval.verify_index(client, CORPUS_SHA256)


def test_load_retriever_uses_cached_models_in_sequence(index, monkeypatch):
    loaded = []

    def create_model(name, *, cache_dir, local_files_only):
        assert local_files_only
        assert cache_dir == str(settings.cache_path)
        loaded.append(name)
        return Mock()

    settings = RagSettings(qdrant_url="http://vector-store.invalid:6333")
    open_client = Mock(return_value=index)
    monkeypatch.setattr(retrieval, "TextEmbedding", create_model)
    monkeypatch.setattr(retrieval, "SparseTextEmbedding", create_model)
    monkeypatch.setattr(retrieval, "TextCrossEncoder", create_model)
    monkeypatch.setattr(retrieval, "QdrantClient", open_client)

    with closing(retrieval.load_retriever(settings, CORPUS_SHA256)):
        assert loaded == [DENSE_MODEL, BM25_MODEL, RERANK_MODEL]
        open_client.assert_called_once_with(url=str(settings.qdrant_url))


def test_lit_env_passes_qdrant_and_fastembed_settings(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://vector-store.invalid:6333")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", ".cache/test-models")

    env = lit_env_from_process()

    assert env["QDRANT_URL"] == "http://vector-store.invalid:6333"
    assert env["FASTEMBED_CACHE_PATH"] == ".cache/test-models"


async def test_lifespan_loads_retriever_once_off_the_event_loop(monkeypatch, caplog):
    event_loop_thread = get_ident()
    fake = Mock(spec=Retriever)

    def load():
        assert get_ident() != event_loop_thread
        return fake

    factory = Mock(side_effect=load)
    monkeypatch.setattr(lit_server, "_load_retriever_from_env", factory)
    caplog.set_level(logging.INFO, logger=lit_server.__name__)

    async with Client(create_server(client_factory=httpx.AsyncClient)) as client:
        await client.list_tools()
        await client.list_tools()
        factory.assert_called_once_with()
        fake.close.assert_not_called()

    fake.close.assert_called_once_with()
    assert (
        sum(message.startswith("lit-mcp ready in ") for message in caplog.messages) == 1
    )


async def test_search_evidence_validates_query_and_retrieves_off_the_event_loop(
    index, models_for_query
):
    event_loop_thread = get_ident()
    retriever = Retriever(index, *models_for_query)

    def search(query):
        assert get_ident() != event_loop_thread
        return retriever.search(query)

    fake = Mock(spec=Retriever)
    fake.search.side_effect = search
    server = create_server(
        client_factory=httpx.AsyncClient,
        retriever_factory=lambda: nullcontext(fake),
    )
    async with Client(server) as client:
        invalid = await client.call_tool(
            "search_evidence", {"request": {"query": "  "}}, raise_on_error=False
        )
        assert invalid.is_error
        fake.search.assert_not_called()
        result = await client.call_tool(
            "search_evidence", {"request": {"query": "AMG-510 toxicity"}}
        )
        fake.search.side_effect = RuntimeError("original cross-encoder failure")
        error = await client.call_tool(
            "search_evidence",
            {"request": {"query": "AMG-510 toxicity"}},
            raise_on_error=False,
        )

    output = SearchEvidenceResult.model_validate(result.structured_content)
    assert output.query == "AMG-510 toxicity"
    assert output.expanded_query == "AMG-510 toxicity sotorasib LUMAKRAS"
    assert [hit.citation_id for hit in output.results] == [
        "pubmed:3:0",
        "pubmed:1:0",
        "pubmed:2:0",
    ]
    assert [hit.rerank_score for hit in output.results] == [9.0, 5.0, 3.0]
    assert error.is_error
    assert "original cross-encoder failure" in error.content[0].text
