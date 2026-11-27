"""FastEmbed model names and the dense model's tokenizer for corpus builds."""

from pathlib import Path

from fastembed import TextEmbedding

from .chunks import TokenCounter

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BM25_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODEL_NAMES = {
    "dense_model": DENSE_MODEL,
    "bm25_model": BM25_MODEL,
    "rerank_model": RERANK_MODEL,
}


def load_chunk_token_counter(cache_path: Path) -> TokenCounter:
    model = TextEmbedding(
        DENSE_MODEL, cache_dir=str(cache_path), lazy_load=True, local_files_only=True
    ).model
    # FastEmbed 0.9.1 can initialize the ONNX backend's cached tokenizer without
    # creating an inference session. Count the entire input, including special tokens.
    model._ensure_tokenizer()
    tokenizer = model.tokenizer
    assert tokenizer is not None
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda text: len(tokenizer.encode(text).ids)
