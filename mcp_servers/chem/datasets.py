"""Dataset locations and the CSV contract shared by uploads and chem-mcp."""

import csv
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_PREFIX = "upload_"


def upload_directory() -> Path:
    return (PROJECT_ROOT / os.environ.get("UPLOAD_DIR", "data/uploads")).resolve()


def dataset_path(directory: Path, dataset_id: str) -> Path:
    directory = directory.resolve()
    path = (directory / f"{dataset_id}.csv").resolve()
    if path.parent != directory:
        raise ValueError(
            f"Dataset ID must resolve within {directory.name}/: {dataset_id}"
        )
    return path


def resolve_dataset(
    dataset_id: str, data_directory: Path = PROJECT_ROOT / "data"
) -> Path:
    directory = (
        upload_directory() if dataset_id.startswith(UPLOAD_PREFIX) else data_directory
    )
    return dataset_path(directory, dataset_id)


def publish_dataset(directory: Path, payload: bytes) -> str:
    dataset_id = f"{UPLOAD_PREFIX}{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=True)
    target = dataset_path(directory, dataset_id)
    with NamedTemporaryFile(dir=directory) as staging:
        staging.write(payload)
        staging.flush()
        os.link(staging.name, target)
    return dataset_id


def read_dataset_csv(
    source: TextIO, label: str, *, max_rows: int | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(source, strict=True)
    columns = reader.fieldnames
    if not columns or "smiles" not in columns:
        raise ValueError(f"Dataset must contain one smiles column: {label}")
    for name, count in Counter(columns).items():
        if count > 1:
            raise ValueError(f"Dataset {label}: duplicate {name} column")

    rows = []
    for row in reader:
        if max_rows is not None and len(rows) == max_rows:
            raise ValueError(f"CSV 数据行数超过上限 {max_rows}")
        if None in row:
            raise ValueError(
                f"Dataset {label}, line {reader.line_num}: "
                f"extra fields: {len(row[None])} (expected {len(columns)} fields)"
            )
        missing = [name for name, value in row.items() if value is None]
        if missing:
            raise ValueError(
                f"Dataset {label}, line {reader.line_num}: "
                f"missing {', '.join(missing)} (expected {len(columns)} fields)"
            )
        rows.append(row)
    return list(columns), rows
