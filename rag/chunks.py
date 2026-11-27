"""Represent and split corpus chunks before embedding."""

import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, HttpUrl, model_validator

from mcp_servers.lit.models import NonBlank, SourceType

MAX_CHUNK_TOKENS = 448
type TokenCounter = Callable[[str], int]

_CHUNK_IDS = {
    "pubmed": re.compile(r"pubmed:[1-9][0-9]*:[0-9]+"),
    "trial": re.compile(
        r"trial:NCT[0-9]{8}:(?:overview:0|(?:serious|other):[\w-]+:[0-9]+)"
    ),
    "label": re.compile(
        r"label:[^:\s]+:(?:(?:warnings_and_cautions|adverse_reactions):(?:intro|[0-9]+(?:\.[0-9]+)+)"
        r"|adverse_reactions_table:table-[1-9][0-9]*):[0-9]+"
    ),
}
_SENTENCE_BREAK = re.compile(
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bet al\.)(?<!\bvs\.)(?<!\bU\.S\.)(?<=[.!?])\s+",
    re.IGNORECASE,
)


class Chunk(BaseModel):
    """One citable evidence fragment, including the context line in its text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: NonBlank
    source_type: SourceType
    source_id: NonBlank
    title: str | None
    year: int | None
    section: NonBlank
    text: NonBlank
    source_url: HttpUrl

    @model_validator(mode="after")
    def check_chunk_id(self) -> Self:
        prefix = f"{self.source_type}:{self.source_id}:"
        if not self.chunk_id.startswith(prefix) or not _CHUNK_IDS[
            self.source_type
        ].fullmatch(self.chunk_id):
            raise ValueError(
                f"Invalid chunk ID {self.chunk_id!r} for {self.source_type} source {self.source_id!r}"
            )
        return self

    @property
    def relevance_unit(self) -> str:
        return self.chunk_id.rsplit(":", 1)[0]


def _pack(
    prefix: str,
    units: Iterable[str],
    separator: str,
    count_tokens: TokenCounter,
    unit_id: str,
) -> Iterator[str]:
    text = None
    for unit in units:
        first = f"{prefix}\n{unit}"
        candidate = first if text is None else f"{text}{separator}{unit}"
        token_count = count_tokens(candidate)
        if token_count <= MAX_CHUNK_TOKENS:
            text = candidate
            continue
        if text is not None:
            yield text
            token_count = count_tokens(first)
            if token_count <= MAX_CHUNK_TOKENS:
                text = first
                continue
        raise ValueError(
            f"{unit_id} has a sentence or row with {token_count} tokens including context "
            f"(limit {MAX_CHUNK_TOKENS}): {unit[:120]!r}"
        )
    if text is not None:
        yield text


def split_text(
    context: str, text: str, count_tokens: TokenCounter, unit_id: str
) -> list[str]:
    full_text = f"{context}\n{text}"
    if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
        return [full_text]
    return list(_pack(context, _SENTENCE_BREAK.split(text), " ", count_tokens, unit_id))


def split_rows(
    context: str, rows: Iterable[str], count_tokens: TokenCounter, unit_id: str
) -> list[str]:
    return list(_pack(context, rows, "\n", count_tokens, unit_id))


def _check_unique_ids(chunks: Sequence[Chunk]) -> None:
    counts = Counter(chunk.chunk_id for chunk in chunks)
    duplicates = [chunk_id for chunk_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs: {duplicates}")


def parse_chunks(contents: bytes) -> list[Chunk]:
    chunks = [Chunk.model_validate_json(line) for line in contents.splitlines()]
    _check_unique_ids(chunks)
    return chunks


def write_chunks(path: Path, chunks: Sequence[Chunk]) -> str:
    """Publish validated JSONL atomically and return the SHA-256 of its bytes."""
    _check_unique_ids(chunks)
    contents = "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temp_path = Path(directory) / path.name
        temp_path.write_bytes(contents)
        temp_path.replace(path)
    return sha256(contents).hexdigest()
