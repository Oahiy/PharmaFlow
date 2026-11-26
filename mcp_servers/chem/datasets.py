"""Dataset locations and the CSV contract shared by uploads and chem-mcp."""

import csv
import os
from collections import Counter
from pathlib import Path
from typing import TextIO

from .models import MoleculeRecord

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


def read_dataset_csv(
    source: TextIO, label: str, *, max_rows: int | None = None
) -> tuple[list[str], list[MoleculeRecord]]:
    reader = csv.DictReader(source, strict=True)
    columns = reader.fieldnames
    if not columns or "smiles" not in columns:
        raise ValueError(f"Dataset must contain one smiles column: {label}")
    for name, count in Counter(columns).items():
        if count > 1:
            raise ValueError(f"Dataset {label}: duplicate {name} column")

    records = []
    for row in reader:
        if max_rows is not None and len(records) == max_rows:
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
        records.append(
            MoleculeRecord(
                smiles=row["smiles"],
                molecule_chembl_id=row.get("molecule_chembl_id") or None,
            )
        )
    return list(columns), records
