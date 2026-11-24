import pytest
from pydantic import ValidationError

from mcp_servers.chem import tools
from mcp_servers.chem.models import (
    DruglikeInput,
    DruglikeMolecule,
    DruglikeResult,
    InvalidMolecule,
    LipinskiThresholds,
    MoleculeInput,
    MoleculeProperties,
    PropertiesResult,
)

# ChEMBL CHEMBL4535757
SOTORASIB = (
    "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@@H](C)C1"
)
CATECHOL = "Oc1ccccc1O"
ETHANOL = "CCO"


def test_sotorasib_descriptors():
    (row,) = tools.compute_properties(MoleculeInput(smiles=[SOTORASIB])).results

    assert isinstance(row, MoleculeProperties)
    assert row.mw == pytest.approx(560.605, abs=0.01)
    assert row.logp == pytest.approx(4.4814, abs=0.01)
    assert row.tpsa == pytest.approx(104.45, abs=0.01)
    assert (row.hbd, row.hba) == (1, 7)
    assert row.qed == pytest.approx(0.3591, abs=0.001)


def test_pains_catechol_fails_combined_filter():
    catechol, ethanol = tools.filter_druglike(
        DruglikeInput(smiles=[CATECHOL, ETHANOL])
    ).results

    assert isinstance(catechol, DruglikeMolecule)
    assert catechol.passes_lipinski
    assert any(alert.startswith("catechol_") for alert in catechol.pains_alerts)
    assert not catechol.passes_filter
    assert isinstance(ethanol, DruglikeMolecule)
    assert ethanol.pains_alerts == []
    assert ethanol.passes_filter


@pytest.mark.parametrize(
    ("tool", "request_type"),
    [(tools.compute_properties, MoleculeInput), (tools.filter_druglike, DruglikeInput)],
)
def test_invalid_smiles_are_reported_in_order(tool, request_type):
    batch = [SOTORASIB, "not-a-smiles", "", "CCO ethanol", ETHANOL]
    results = tool(request_type(smiles=batch)).results
    valid_results = tool(request_type(smiles=[SOTORASIB, ETHANOL])).results

    assert [row.smiles for row in results] == batch
    assert results[0] == valid_results[0]
    assert results[4] == valid_results[1]
    assert results[1:4] == [InvalidMolecule(smiles=smiles) for smiles in batch[1:4]]


@pytest.mark.parametrize(
    ("tool", "request_type", "result_type", "field"),
    [
        (tools.compute_properties, MoleculeInput, PropertiesResult, "mw"),
        (tools.filter_druglike, DruglikeInput, DruglikeResult, "passes_filter"),
    ],
)
def test_client_rejects_malformed_result_row(tool, request_type, result_type, field):
    payload = tool(request_type(smiles=[ETHANOL])).model_dump()
    payload["results"][0][field] = "abc"

    with pytest.raises(ValidationError):
        result_type.model_validate(payload)


@pytest.mark.parametrize("smiles", ["CCO\nC1", "CCO\rC1"])
def test_multiline_smiles_is_rejected(smiles):
    result = tools.compute_properties(MoleculeInput(smiles=[smiles]))

    assert result.results == [InvalidMolecule(smiles=smiles)]


@pytest.mark.parametrize(
    ("thresholds", "violations", "passes"),
    [
        (LipinskiThresholds(), ["mw"], False),
        (LipinskiThresholds(max_mw=600), [], True),
        (LipinskiThresholds(max_violations=1), ["mw"], True),
        (
            LipinskiThresholds(max_mw=550, max_logp=4, max_hbd=0, max_hba=6),
            ["mw", "logp", "hbd", "hba"],
            False,
        ),
    ],
)
def test_configurable_lipinski_thresholds(thresholds, violations, passes):
    (row,) = tools.filter_druglike(
        DruglikeInput(smiles=[SOTORASIB], thresholds=thresholds)
    ).results

    assert isinstance(row, DruglikeMolecule)
    assert row.lipinski_violations == violations
    assert row.passes_lipinski is passes
    assert row.passes_filter is passes


def test_lipinski_thresholds_are_inclusive():
    (properties,) = tools.compute_properties(MoleculeInput(smiles=[ETHANOL])).results
    assert isinstance(properties, MoleculeProperties)
    thresholds = LipinskiThresholds(
        max_mw=properties.mw,
        max_logp=properties.logp,
        max_hbd=properties.hbd,
        max_hba=properties.hba,
    )

    (row,) = tools.filter_druglike(
        DruglikeInput(smiles=[ETHANOL], thresholds=thresholds)
    ).results
    assert isinstance(row, DruglikeMolecule)
    assert row.lipinski_violations == []
    assert row.passes_filter


@pytest.mark.parametrize(
    ("tool", "request_type"),
    [(tools.compute_properties, MoleculeInput), (tools.filter_druglike, DruglikeInput)],
)
def test_dataset_reads_smiles_column_in_csv_order(
    monkeypatch, tmp_path, tool, request_type
):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "test_set.csv").write_text(
        f"name,smiles\nfirst,{ETHANOL}\nblank,\nlast,{CATECHOL}\n", encoding="utf-8-sig"
    )

    assert tool(request_type(dataset_id="test_set")) == tool(
        request_type(smiles=[ETHANOL, "", CATECHOL])
    )


def test_csv_row_missing_smiles_reports_dataset_and_line(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "short_row.csv").write_text(
        "name,smiles\nfirst,CCO\nonly_one_field\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"short_row.*line 3.*missing smiles"):
        tools.compute_properties(MoleculeInput(dataset_id="short_row"))


@pytest.mark.parametrize(
    ("csv_content", "reason"),
    [
        ("name,smiles\nbatch,C,CCO\n", "extra fields: 1"),
        ("smiles,molecule_chembl_id\nCCC\n", "missing molecule_chembl_id"),
        ("smiles,name,molecule_chembl_id\nCCC\n", "missing name, molecule_chembl_id"),
    ],
)
def test_csv_rejects_incorrect_row_width(monkeypatch, tmp_path, csv_content, reason):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "wrong_width.csv").write_text(csv_content, encoding="utf-8")

    with pytest.raises(ValueError, match=rf"wrong_width.*line 2.*{reason}"):
        tools.compute_properties(MoleculeInput(dataset_id="wrong_width"))


@pytest.mark.parametrize(
    ("tool", "request_type"),
    [(tools.compute_properties, MoleculeInput), (tools.filter_druglike, DruglikeInput)],
)
def test_csv_results_preserve_optional_chembl_id(
    monkeypatch, tmp_path, tool, request_type
):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "identified.csv").write_text(
        "molecule_chembl_id,smiles\nCHEMBL1,CCO\nCHEMBL2,\n,CCO\n",
        encoding="utf-8",
    )

    valid, invalid, anonymous = tool(request_type(dataset_id="identified")).results
    assert valid.molecule_chembl_id == "CHEMBL1"
    assert invalid.error == "invalid_smiles"
    assert invalid.molecule_chembl_id == "CHEMBL2"
    assert anonymous.molecule_chembl_id is None


@pytest.mark.parametrize("column", ["molecule_chembl_id", "name"])
def test_csv_rejects_duplicate_columns(monkeypatch, tmp_path, column):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "ambiguous.csv").write_text(
        f"{column},smiles,{column}\nCHEMBL1,CCO,CHEMBL2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"ambiguous.*duplicate {column}"):
        tools.compute_properties(MoleculeInput(dataset_id="ambiguous"))


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"smiles": []},
        {"smiles": [ETHANOL], "dataset_id": "candidates"},
        {"dataset_id": "../candidates"},
        {"dataset_id": "candidates.csv"},
    ],
)
def test_invalid_molecule_sources_are_rejected(source):
    with pytest.raises(ValidationError):
        MoleculeInput(**source)


def test_unknown_dataset_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)

    with pytest.raises(ValueError, match=r"^Unknown dataset ID: missing$"):
        tools.compute_properties(MoleculeInput(dataset_id="missing"))


def test_dataset_must_have_a_smiles_column(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DATA_DIR", tmp_path)
    (tmp_path / "test_set.csv").write_text(
        "name,ic50_nm\nexample,10\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must contain one smiles column"):
        tools.compute_properties(MoleculeInput(dataset_id="test_set"))


def test_dataset_symlink_cannot_access_outside_data(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(tools, "_DATA_DIR", data_dir)
    outside = tmp_path / "outside.csv"
    outside.write_text(f"smiles\n{ETHANOL}\n", encoding="utf-8")
    (data_dir / "linked.csv").symlink_to(outside)

    with pytest.raises(ValueError, match="Dataset ID must resolve within data/"):
        tools.compute_properties(MoleculeInput(dataset_id="linked"))
