"""Calculate RDKit molecular descriptors and screen for drug-likeness."""

import csv
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .models import (
    DruglikeInput,
    DruglikeMolecule,
    DruglikeResult,
    InvalidMolecule,
    LipinskiProperties,
    MoleculeInput,
    MoleculeProperties,
    MoleculeRecord,
    PropertiesResult,
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_catalog_params = FilterCatalogParams()
_catalog_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_catalog_params)


def _load_dataset(dataset_id: str) -> list[MoleculeRecord]:
    data_dir = _DATA_DIR.resolve()
    path = (data_dir / f"{dataset_id}.csv").resolve()
    if path.parent != data_dir:
        raise ValueError(f"Dataset ID must resolve within data/: {dataset_id}")

    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            fieldnames = reader.fieldnames
            if not fieldnames or "smiles" not in fieldnames:
                raise ValueError(
                    f"Dataset must contain one smiles column: {dataset_id}"
                )
            for name, count in Counter(fieldnames).items():
                if count > 1:
                    raise ValueError(f"Dataset {dataset_id}: duplicate {name} column")

            records = []
            for row in reader:
                if None in row:
                    raise ValueError(
                        f"Dataset {dataset_id}, line {reader.line_num}: "
                        f"extra fields: {len(row[None])} (expected {len(fieldnames)} fields)"
                    )
                missing = [name for name, value in row.items() if value is None]
                if missing:
                    raise ValueError(
                        f"Dataset {dataset_id}, line {reader.line_num}: "
                        f"missing {', '.join(missing)} (expected {len(fieldnames)} fields)"
                    )
                records.append(
                    MoleculeRecord(
                        smiles=row["smiles"],
                        molecule_chembl_id=row.get("molecule_chembl_id") or None,
                    )
                )
            return records
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown dataset ID: {dataset_id}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV dataset {dataset_id}: {exc}") from exc


def _molecules(
    request: MoleculeInput,
) -> Iterator[tuple[MoleculeRecord, Chem.Mol | None]]:
    if request.dataset_id is not None:
        records = _load_dataset(request.dataset_id)
    else:
        assert request.smiles is not None
        records = [MoleculeRecord(smiles=smiles) for smiles in request.smiles]

    params = Chem.SmilesParserParams()
    params.parseName = False
    for record in records:
        cleaned = record.smiles.strip()
        # RDKit can stop at an embedded newline and return a valid prefix.
        single_line = "\n" not in cleaned and "\r" not in cleaned
        mol = Chem.MolFromSmiles(cleaned, params) if single_line else None
        if mol is not None and mol.GetNumAtoms() == 0:
            mol = None
        yield record, mol


def _lipinski_properties(mol: Chem.Mol) -> LipinskiProperties:
    return LipinskiProperties(
        mw=Descriptors.MolWt(mol),
        logp=Crippen.MolLogP(mol),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
    )


def compute_properties(request: MoleculeInput) -> PropertiesResult:
    """Calculate RDKit molecular descriptors for a batch.

    Args:
        request: One non-empty list of SMILES or the ID of a CSV with a smiles column.

    Returns:
        MW (Da), Crippen LogP, TPSA (Å²), HBD, HBA, and QED for each molecule.
        HBD and HBA use RDKit Lipinski.NumHDonors and Lipinski.NumHAcceptors.
        All rows preserve the input order and optional molecule_chembl_id.
        Invalid SMILES have an error entry at the corresponding input position.
    """
    results: list[MoleculeProperties | InvalidMolecule] = []
    for record, mol in _molecules(request):
        if mol is None:
            results.append(InvalidMolecule(**record.model_dump()))
            continue

        results.append(
            MoleculeProperties(
                **record.model_dump(),
                tpsa=rdMolDescriptors.CalcTPSA(mol),
                qed=QED.qed(mol),
                **_lipinski_properties(mol).model_dump(),
            )
        )
    return PropertiesResult(results=results)


def filter_druglike(request: DruglikeInput) -> DruglikeResult:
    """Screen a batch using configurable Lipinski thresholds and RDKit PAINS alerts.

    Args:
        request: One non-empty list of SMILES or a CSV dataset ID, with optional
            thresholds and a maximum number of allowed Lipinski violations.

    Returns:
        Each molecule's Lipinski violations, catalog alerts, pass status, and
        optional molecule_chembl_id. HBD and HBA use RDKit Lipinski.NumHDonors
        and Lipinski.NumHAcceptors.
        A molecule passes the combined filter when it passes Lipinski and has no
        PAINS alerts. Invalid SMILES have an error entry in input order.
    """
    thresholds = request.thresholds
    results: list[DruglikeMolecule | InvalidMolecule] = []
    for record, mol in _molecules(request):
        if mol is None:
            results.append(InvalidMolecule(**record.model_dump()))
            continue

        values = _lipinski_properties(mol)
        checks: tuple[tuple[Literal["mw", "logp", "hbd", "hba"], bool], ...] = (
            ("mw", values.mw > thresholds.max_mw),
            ("logp", values.logp > thresholds.max_logp),
            ("hbd", values.hbd > thresholds.max_hbd),
            ("hba", values.hba > thresholds.max_hba),
        )
        violations = [name for name, violated in checks if violated]
        pains_alerts = [
            entry.GetDescription() for entry in _PAINS_CATALOG.GetMatches(mol)
        ]
        passes_lipinski = len(violations) <= thresholds.max_violations
        results.append(
            DruglikeMolecule(
                **record.model_dump(),
                lipinski_violations=violations,
                passes_lipinski=passes_lipinski,
                pains_alerts=pains_alerts,
                passes_filter=passes_lipinski and not pains_alerts,
            )
        )
    return DruglikeResult(results=results)
