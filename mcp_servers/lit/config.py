"""Read and validate lit-mcp configuration once at startup."""

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "data/lit_recordings"


class LitSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    mode: Literal["live", "record", "replay"] = "replay"
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    ncbi_api_key: SecretStr | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_recordings_dir(self) -> Self:
        if self.mode == "replay" and not self.recordings_dir.is_dir():
            raise ValueError(
                f"LIT_MCP_RECORDINGS_DIR is not an existing directory: {self.recordings_dir}"
            )
        return self

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                "mode": os.getenv("LIT_MCP_MODE", "replay"),
                "recordings_dir": os.getenv(
                    "LIT_MCP_RECORDINGS_DIR", str(DEFAULT_RECORDINGS_DIR)
                ),
                "ncbi_api_key": os.getenv("NCBI_API_KEY") or None,
            }
        )
