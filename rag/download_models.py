"""Download the three FastEmbed models into the configured cache."""

import argparse

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import RagSettings
from .embedding import BM25_MODEL, DENSE_MODEL, RERANK_MODEL


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cache_dir = str(RagSettings.from_env().cache_path)
    TextEmbedding(DENSE_MODEL, cache_dir=cache_dir, lazy_load=True)
    SparseTextEmbedding(BM25_MODEL, cache_dir=cache_dir)
    TextCrossEncoder(RERANK_MODEL, cache_dir=cache_dir, lazy_load=True)
    print(f"FastEmbed models cached in {cache_dir}")


if __name__ == "__main__":
    main()
