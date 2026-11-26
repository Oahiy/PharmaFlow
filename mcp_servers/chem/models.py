"""Input and structured output models for chem-mcp."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    model_validator,
)

DatasetId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]


class MoleculeInput(BaseModel):
    """Exactly one source of molecules to analyze."""

    model_config = ConfigDict(extra="forbid")

    smiles: list[str] | None = Field(
        default=None, min_length=1, description="SMILES in the desired result order."
    )
    dataset_id: DatasetId | None = Field(
        default=None,
        description="Dataset ID without .csv, e.g. candidates or an upload_ ID from the API.",
    )

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if (self.smiles is None) == (self.dataset_id is None):
            raise ValueError("Provide exactly one of smiles or dataset_id.")
        return self


class LipinskiThresholds(BaseModel):
    """Inclusive thresholds for the four Lipinski properties."""

    model_config = ConfigDict(extra="forbid")

    max_mw: FiniteFloat = Field(
        default=500, gt=0, description="Molecular weight in Da."
    )
    max_logp: FiniteFloat = Field(default=5, description="RDKit Crippen LogP.")
    max_hbd: int = Field(default=5, ge=0, description="Hydrogen-bond donors.")
    max_hba: int = Field(default=10, ge=0, description="Hydrogen-bond acceptors.")
    max_violations: int = Field(default=0, ge=0, le=4)


class DruglikeInput(MoleculeInput):
    """Molecule source and optional Lipinski thresholds."""

    thresholds: LipinskiThresholds = Field(default_factory=LipinskiThresholds)


class MoleculeRecord(BaseModel):
    """Input SMILES and its ChEMBL identifier when supplied by the dataset."""

    smiles: str
    molecule_chembl_id: str | None = None


class InvalidMolecule(MoleculeRecord):
    """A row that RDKit could not parse as a molecule."""

    model_config = ConfigDict(extra="forbid")

    error: Literal["invalid_smiles"] = "invalid_smiles"


class LipinskiProperties(BaseModel):
    """The four properties used by the Lipinski filter."""

    mw: FiniteFloat
    logp: FiniteFloat
    hbd: int
    hba: int


class MoleculeProperties(MoleculeRecord, LipinskiProperties):
    """RDKit descriptors for a single SMILES."""

    tpsa: FiniteFloat
    qed: FiniteFloat


class PropertiesResult(BaseModel):
    """One property result or parse error for each input molecule."""

    results: list[MoleculeProperties | InvalidMolecule]


class ExportPropertiesInput(BaseModel):
    """Source CSV for a batch of descriptors to use in the sandbox."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: DatasetId = Field(description="Source CSV dataset ID without .csv.")


class InvalidPropertyRow(BaseModel):
    """An excluded row and its position among data rows, starting at 1."""

    row: PositiveInt
    error: Literal["invalid_smiles"] = "invalid_smiles"


class ExportPropertiesResult(BaseModel):
    """Derived CSV metadata and the positions of any excluded molecules."""

    dataset_id: DatasetId | None
    columns: list[str]
    row_count: int = Field(ge=0)
    invalid_rows: list[InvalidPropertyRow]


class DruglikeMolecule(MoleculeRecord):
    """Lipinski and PAINS screening results for a single SMILES."""

    lipinski_violations: list[Literal["mw", "logp", "hbd", "hba"]]
    passes_lipinski: bool
    pains_alerts: list[str]
    passes_filter: bool


class DruglikeResult(BaseModel):
    """One screening result or parse error for each input molecule."""

    results: list[DruglikeMolecule | InvalidMolecule]
