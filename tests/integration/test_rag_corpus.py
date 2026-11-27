import pytest

from rag.build_corpus import build_corpus
from rag.chunks import parse_chunks
from rag.config import CHUNKS_PATH, CORPUS_RECORDINGS_DIR, RagSettings
from rag.embedding import load_chunk_token_counter


@pytest.fixture(scope="module")
def count_tokens():
    return load_chunk_token_counter(RagSettings.from_env().cache_path)


def test_committed_chunks_fit_the_token_budget(count_tokens):
    for chunk in parse_chunks(CHUNKS_PATH.read_bytes()):
        size = count_tokens(chunk.text)
        assert size <= 448, f"{chunk.chunk_id}: {size} tokens"


async def test_replay_rebuild_matches_the_committed_corpus(count_tokens, tmp_path):
    rebuilt = tmp_path / "chunks.jsonl"
    await build_corpus(CORPUS_RECORDINGS_DIR, rebuilt, count_tokens)

    assert rebuilt.read_bytes() == CHUNKS_PATH.read_bytes()
