"""Build the named dense and BM25 vectors and validate the corpus snapshot."""

import argparse
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Annotated
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from fastembed.common.utils import remove_non_alphanumeric
from fastembed.sparse.bm25 import Bm25
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints
from qdrant_client import QdrantClient, models

from .chunks import parse_chunks
from .config import CHUNKS_PATH, RagSettings
from .embedding import BM25_MODEL, DENSE_MODEL, MODEL_NAMES

COLLECTION_NAME = "kras_evidence"
DENSE_VECTOR = "dense"
BM25_VECTOR = "bm25"
POINT_NAMESPACE = uuid5(NAMESPACE_URL, "urn:pharmaflow:kras-evidence")


class IndexMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    corpus_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    chunk_count: PositiveInt
    dense_model: str
    bm25_model: str
    rerank_model: str
    bm25_avg_len: float = Field(gt=0, allow_inf_nan=False)


def bm25_doc_length(model: Bm25, text: str) -> int:
    # FastEmbed 0.9.1 internals: raw_embed stems exactly these tokens before
    # _term_frequency assigns len(stemmed_tokens) to doc_len.
    tokens = model.tokenizer.tokenize(remove_non_alphanumeric(text))
    return len(model._stem(tokens))


def sparse_vector(embedding: SparseEmbedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


def _check_point_count(
    client: QdrantClient, collection_name: str, expected_count: int
) -> None:
    actual_count = client.count(collection_name, exact=True).count
    if actual_count != expected_count:
        raise ValueError(
            f"Qdrant collection {collection_name!r} has {actual_count} points, "
            f"expected chunk_count={expected_count}"
        )


def verify_index(
    client: QdrantClient, corpus_sha256: str, collection_name: str = COLLECTION_NAME
) -> IndexMetadata:
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection {collection_name!r} does not exist")
    info = client.get_collection(collection_name)
    if info.config.metadata is None:
        raise ValueError(f"Qdrant collection {collection_name!r} has no metadata")
    metadata = IndexMetadata.model_validate(info.config.metadata)
    values = metadata.model_dump()
    for key, expected in {**MODEL_NAMES, "corpus_sha256": corpus_sha256}.items():
        if values[key] != expected:
            raise ValueError(
                f"Qdrant collection {collection_name!r} has {key}={values[key]!r}, "
                f"expected {expected!r}"
            )
    _check_point_count(client, collection_name, metadata.chunk_count)
    return metadata


def build_index(
    client: QdrantClient,
    corpus: bytes,
    cache_path: Path,
    *,
    collection_name: str = COLLECTION_NAME,
) -> IndexMetadata:
    chunks = parse_chunks(corpus)
    texts = [chunk.text for chunk in chunks]
    model_options = {"cache_dir": str(cache_path), "local_files_only": True}
    bm25 = SparseTextEmbedding(BM25_MODEL, **model_options)
    avg_len = mean(bm25_doc_length(bm25.model, text) for text in texts)
    bm25 = SparseTextEmbedding(BM25_MODEL, avg_len=avg_len, **model_options)
    dense = TextEmbedding(DENSE_MODEL, **model_options)
    metadata = IndexMetadata(
        corpus_sha256=sha256(corpus).hexdigest(),
        chunk_count=len(chunks),
        bm25_avg_len=avg_len,
        **MODEL_NAMES,
    )
    points = [
        models.PointStruct(
            id=str(uuid5(POINT_NAMESPACE, chunk.chunk_id)),
            vector={DENSE_VECTOR: vector.tolist(), BM25_VECTOR: sparse_vector(sparse)},
            payload=chunk.model_dump(mode="json"),
        )
        for chunk, vector, sparse in zip(
            chunks, dense.embed(texts, batch_size=32), bm25.embed(texts), strict=True
        )
    ]

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dense.embedding_size, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
        metadata=metadata.model_dump(mode="json"),
    )
    client.upsert(collection_name, points, wait=True)
    _check_point_count(client, collection_name, metadata.chunk_count)
    return metadata


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    settings = RagSettings.from_env()
    corpus = CHUNKS_PATH.read_bytes()
    with closing(QdrantClient(url=str(settings.qdrant_url))) as client:
        metadata = build_index(client, corpus, settings.cache_path)
    print(metadata.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
