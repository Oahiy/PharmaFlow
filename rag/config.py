"""Local corpus paths and validated settings for FastEmbed and Qdrant."""

import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_RECORDINGS_DIR = PROJECT_ROOT / "data/corpus_recordings"
CHUNKS_PATH = PROJECT_ROOT / "data/corpus/chunks.jsonl"
DEFAULT_FASTEMBED_CACHE = PROJECT_ROOT / ".cache/fastembed"
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


class RagSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, validate_default=True)

    cache_path: Path = DEFAULT_FASTEMBED_CACHE
    qdrant_url: HttpUrl = DEFAULT_QDRANT_URL

    @field_validator("cache_path")
    @classmethod
    def resolve_cache_path(cls, cache_path: Path) -> Path:
        return (PROJECT_ROOT / cache_path).resolve()

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                "cache_path": os.getenv(
                    "FASTEMBED_CACHE_PATH", str(DEFAULT_FASTEMBED_CACHE)
                ),
                "qdrant_url": os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL),
            }
        )
