from contextlib import closing
from hashlib import sha256
from uuid import uuid4, uuid5

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from rag.chunks import Chunk
from rag.config import RagSettings
from rag.embedding import BM25_MODEL, DENSE_MODEL, RERANK_MODEL
from rag.index import POINT_NAMESPACE, bm25_doc_length, build_index, verify_index
from rag.retrieval import Retriever


def test_bm25_document_length_matches_the_tokens_used_by_term_frequency(monkeypatch):
    settings = RagSettings.from_env()
    bm25 = SparseTextEmbedding(
        BM25_MODEL, cache_dir=str(settings.cache_path), local_files_only=True
    ).model
    term_frequency = bm25._term_frequency
    observed = []

    def capture_doc_len(tokens):
        observed.append(len(tokens))
        return term_frequency(tokens)

    monkeypatch.setattr(bm25, "_term_frequency", capture_doc_len)
    text = "The therapies, running; and cats cats. " + "z" * 41
    bm25.raw_embed([text])

    assert observed == [4]
    assert bm25_doc_length(bm25, text) == observed[0]
    assert bm25.token_count(text) > observed[0]


def test_index_build_and_rerank_with_real_models():
    query = "Which city is the capital of France?"
    chunks = [
        Chunk(
            chunk_id=f"pubmed:{pmid}:0",
            source_type="pubmed",
            source_id=str(pmid),
            title=f"Fixture {pmid}",
            year=2026,
            section="Abstract",
            text=f"Fixture {pmid} (2026)\n{text}",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )
        for pmid, text in enumerate(
            (query, "Paris is France's capital.", "Bananas are a tropical fruit."),
            start=1,
        )
    ]
    corpus = "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks).encode()
    settings = RagSettings.from_env()
    collection_name = f"test_kras_evidence_{uuid4().hex}"
    options = {"cache_dir": str(settings.cache_path), "local_files_only": True}

    with closing(QdrantClient(url=str(settings.qdrant_url))) as client:
        try:
            metadata = build_index(
                client, corpus, settings.cache_path, collection_name=collection_name
            )
            assert (
                verify_index(client, sha256(corpus).hexdigest(), collection_name)
                == metadata
            )
            assert metadata.chunk_count == len(chunks)
            info = client.get_collection(collection_name)
            assert (
                info.config.params.sparse_vectors["bm25"].modifier
                == models.Modifier.IDF
            )
            ids = [str(uuid5(POINT_NAMESPACE, chunk.chunk_id)) for chunk in chunks]
            records = client.retrieve(collection_name, ids, with_payload=True)
            assert {
                record.id: Chunk.model_validate(record.payload) for record in records
            } == dict(zip(ids, chunks, strict=True))

            retriever = Retriever(
                client,
                TextEmbedding(DENSE_MODEL, **options),
                SparseTextEmbedding(BM25_MODEL, **options),
                TextCrossEncoder(RERANK_MODEL, **options),
                collection_name=collection_name,
            )
            hybrid = retriever.search(query, "hybrid")
            reranked = retriever.search(query)
        finally:
            client.delete_collection(collection_name)

    assert hybrid.hits[0].chunk.source_id == "1"
    assert "2" in {hit.chunk.source_id for hit in hybrid.hits}
    assert reranked.hits[0].chunk.source_id == "2"
