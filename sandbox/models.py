"""Input and output contracts for the local run_python tool."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from mcp_servers.chem.models import DatasetId

RUN_PYTHON = "run_python"
MAX_REPAIR_ROUNDS = 3
REPAIR_LIMIT_MESSAGE = (
    f"本轮的 {MAX_REPAIR_ROUNDS} 次沙盒代码修正机会已用完，请说明剩余问题。"
)


class SandboxDatasets(BaseModel):
    """CSV datasets made available to the script at /work/input."""

    model_config = ConfigDict(extra="forbid")

    dataset_ids: list[DatasetId] = Field(
        description=(
            "IDs of all CSVs read by the script, without .csv. Supply them on each "
            "call, including diagnostics and repairs. Use [] for scripts without "
            "CSV input. For the built-in CSV, "
            'set dataset_ids to ["candidates"].'
        ),
    )


class RunPythonRequest(SandboxDatasets):
    """A standalone Python script and its input dataset IDs."""

    code: str = Field(
        min_length=1, pattern=r"\S", description="Python 3.12 source code."
    )


class GeneratedFile(BaseModel):
    """Metadata for a generated PNG, visible to the model."""

    filename: str
    media_type: Literal["image/png"] = "image/png"
    size_bytes: PositiveInt


class ArtifactReference(GeneratedFile):
    """Metadata and identifier used by the API to serve a chart."""

    artifact_id: UUID


class SandboxError(BaseModel):
    """Script execution or input/output validation failure."""

    kind: Literal["code", "timeout", "oom", "input", "output"]
    message: str


class RunPythonResult(BaseModel):
    """Bounded stdout/stderr, execution status, and generated file metadata."""

    stdout: str
    stderr: str
    exit_code: int | None
    files: list[GeneratedFile]
    error: SandboxError | None
