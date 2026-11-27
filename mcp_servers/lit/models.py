"""Input and structured output models for lit-mcp."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    HttpUrl,
    NonNegativeInt,
    StringConstraints,
)

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NctId = Annotated[str, StringConstraints(pattern=r"^NCT[0-9]{8}$")]
TrialPhase = Literal["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
EventCategory = Literal["all", "serious", "other"]
type SourceType = Literal["pubmed", "trial", "label"]
LabelField = Literal[
    "adverse_reactions",
    "adverse_reactions_table",
    "boxed_warning",
    "warnings_and_cautions",
    "contraindications",
    "clinical_studies",
    "dosage_and_administration",
    "indications_and_usage",
]


class PubMedSearchInput(BaseModel):
    """A PubMed ESearch expression and a bounded number of abstracts."""

    model_config = ConfigDict(extra="forbid")

    query: NonBlank = Field(description="PubMed search expression.")
    max_results: int = Field(default=5, ge=1, le=10)


class TrialSearchInput(BaseModel):
    """ClinicalTrials.gov search by condition, intervention, phase, or target keywords."""

    model_config = ConfigDict(extra="forbid")

    condition: NonBlank = Field(description="Condition or disease, e.g. NSCLC.")
    drug: NonBlank | None = Field(
        default=None, description="Intervention or drug name."
    )
    phase: TrialPhase | None = None
    query: NonBlank | None = Field(
        default=None, description="Additional trial keywords."
    )
    max_results: int = Field(default=5, ge=1, le=10)


class TrialAdverseEventsInput(BaseModel):
    """One trial, event categories, and an optional event term filter."""

    model_config = ConfigDict(extra="forbid")

    nct_id: NctId
    category: EventCategory = "all"
    term: str | None = Field(
        default=None,
        pattern=r"\S",
        description="Case-insensitive substring of the source event term, for example Alanine aminotransferase increased.",
    )


class LabelSectionInput(BaseModel):
    """A generic drug name and the requested openFDA label sections."""

    model_config = ConfigDict(extra="forbid")

    drug: NonBlank = Field(description="Exact generic drug name, e.g. sotorasib.")
    fields: list[LabelField] = Field(
        default_factory=lambda: ["adverse_reactions", "adverse_reactions_table"],
        min_length=1,
        max_length=9,
    )


class ResultList[T](BaseModel):
    """Source row count and the rows included in this result."""

    model_config = ConfigDict(extra="forbid")

    total_count: NonNegativeInt
    returned_count: NonNegativeInt
    results: list[T]

    @classmethod
    def from_rows(cls, rows: list[T], *, total_count: int) -> Self:
        return cls(total_count=total_count, returned_count=len(rows), results=rows)


class TextExcerpt(BaseModel):
    """Text with its character count before any shortening."""

    model_config = ConfigDict(extra="forbid")

    text: str
    original_length: NonNegativeInt
    truncated: bool

    @classmethod
    def from_text(cls, text: str, *, limit: int | None = 4000) -> Self:
        excerpt = text[:limit]
        return cls(
            text=excerpt, original_length=len(text), truncated=len(excerpt) < len(text)
        )


class PubMedArticle(BaseModel):
    """A journal article or book entry with its PubMed citation and abstract."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    record_type: Literal["journal_article", "book_article"]
    title: str | None
    abstract: TextExcerpt | None


class PubMedSearchResult(ResultList[PubMedArticle]):
    """The PubMed search criteria, total hits, and fetched abstracts."""

    request: PubMedSearchInput


class TrialSummary(BaseModel):
    """The identifiers, searchable fields, and status of one trial."""

    model_config = ConfigDict(extra="forbid")

    nct_id: NctId
    title: str
    overall_status: str | None
    conditions: ResultList[str] | None
    interventions: ResultList[str] | None
    phases: ResultList[str] | None


class TrialSearchResult(ResultList[TrialSummary]):
    """The search criteria, total hits, and returned study summaries."""

    request: TrialSearchInput


class TrialEventGroup(BaseModel):
    """Trial arm and its reported affected and at-risk denominators."""

    model_config = ConfigDict(extra="forbid")

    nct_id: NctId
    group_id: str
    title: str
    deaths_num_affected: NonNegativeInt | None
    deaths_num_at_risk: NonNegativeInt | None
    serious_num_affected: NonNegativeInt | None
    serious_num_at_risk: NonNegativeInt | None
    other_num_affected: NonNegativeInt | None
    other_num_at_risk: NonNegativeInt | None


class TrialEventStats(BaseModel):
    """Counts for one event in one trial arm."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    num_affected: NonNegativeInt | None
    num_at_risk: NonNegativeInt | None
    num_events: NonNegativeInt | None


class TrialAdverseEvent(BaseModel):
    """Reported event term and associated arm-level counts."""

    model_config = ConfigDict(extra="forbid")

    nct_id: NctId
    term: str
    organ_system: str | None
    source_vocabulary: str | None
    assessment_type: str | None
    stats: ResultList[TrialEventStats] | None


class TrialEventList(ResultList[TrialAdverseEvent]):
    """Reported events counted before filtering, after filtering, and after clipping."""

    matched_count: NonNegativeInt


class TrialEventTables(BaseModel):
    """ClinicalTrials.gov adverse event categories and reporting context."""

    model_config = ConfigDict(
        extra="forbid", validate_by_name=True, serialize_by_alias=True
    )

    frequency_threshold: str | None = Field(alias="frequencyThreshold")
    time_frame: TextExcerpt | None = Field(alias="timeFrame")
    description: TextExcerpt | None
    groups: ResultList[TrialEventGroup] | None
    serious: TrialEventList | None
    other: TrialEventList | None


class TrialAdverseEventsResult(BaseModel):
    """The trial's results availability and its reported adverse events, if present."""

    model_config = ConfigDict(extra="forbid")

    nct_id: NctId
    category: EventCategory
    term: str | None
    has_results: bool
    adverse_events: TrialEventTables | None


class PresentLabelSection(BaseModel):
    """A requested section supplied by the selected label."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["present"] = "present"
    field: LabelField
    set_id: str
    content: ResultList[TextExcerpt]


class MissingLabelSection(BaseModel):
    """A requested section absent from the selected label."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["missing"] = "missing"
    field: LabelField
    set_id: str
    content: None = None


LabelSection = Annotated[
    PresentLabelSection | MissingLabelSection, Field(discriminator="status")
]


class DrugLabel(BaseModel):
    """The selected openFDA label, its effective date, and requested sections."""

    model_config = ConfigDict(extra="forbid")

    set_id: str
    effective_time: str | None
    sections: ResultList[LabelSection]


class LabelSectionResult(ResultList[DrugLabel]):
    """The label search criteria, match count, and selected label if available."""

    request: LabelSectionInput


class SearchEvidenceInput(BaseModel):
    """English query terms for the local evidence corpus."""

    model_config = ConfigDict(extra="forbid")

    query: NonBlank = Field(
        description="Semantic query in English. Translate Chinese questions before calling."
    )


class EvidenceMatch(BaseModel):
    """A reranked corpus excerpt with its citation and source metadata."""

    model_config = ConfigDict(extra="forbid")

    citation_id: NonBlank
    source_type: SourceType
    source_id: NonBlank
    title: str | None
    year: int | None
    section: NonBlank
    text: NonBlank
    rerank_score: FiniteFloat
    source_url: HttpUrl


class SearchEvidenceResult(BaseModel):
    """Original and expanded queries with evidence ranked by cross-encoder score."""

    model_config = ConfigDict(extra="forbid")

    query: NonBlank
    expanded_query: NonBlank
    results: list[EvidenceMatch]
