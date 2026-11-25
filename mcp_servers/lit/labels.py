"""Validate and project the selected openFDA SPL drug label."""

import httpx
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from .models import (
    DrugLabel,
    LabelField,
    LabelSection,
    LabelSectionInput,
    LabelSectionResult,
    MissingLabelSection,
    NonBlank,
    PresentLabelSection,
    ResultList,
    TextExcerpt,
)
from .requests import label_section_url

SECTION_BLOCK_LIMIT = 5


class _Label(BaseModel):
    model_config = ConfigDict(strict=True)

    set_id: NonBlank
    effective_time: str | None = Field(default=None, pattern=r"^[0-9]{8}$")
    adverse_reactions: list[str] | None = None
    adverse_reactions_table: list[str] | None = None
    boxed_warning: list[str] | None = None
    warnings_and_cautions: list[str] | None = None
    contraindications: list[str] | None = None
    clinical_studies: list[str] | None = None
    dosage_and_administration: list[str] | None = None
    indications_and_usage: list[str] | None = None


class _MatchInfo(BaseModel):
    model_config = ConfigDict(strict=True)

    total: PositiveInt


class _Meta(BaseModel):
    results: _MatchInfo


class _LabelResponse(BaseModel):
    meta: _Meta
    results: list[_Label] = Field(min_length=1, max_length=1)


def _section(label: _Label, field: LabelField) -> LabelSection:
    blocks: list[str] | None = getattr(label, field)
    if blocks is None:
        return MissingLabelSection(field=field, set_id=label.set_id)

    if field == "adverse_reactions_table":
        rows = [TextExcerpt.from_text(block, limit=None) for block in blocks]
    else:
        rows = [TextExcerpt.from_text(block) for block in blocks[:SECTION_BLOCK_LIMIT]]
    return PresentLabelSection(
        field=field,
        set_id=label.set_id,
        content=ResultList[TextExcerpt].from_rows(rows, total_count=len(blocks)),
    )


async def get_label_section(
    request: LabelSectionInput, http: httpx.AsyncClient
) -> LabelSectionResult:
    response = await http.get(label_section_url(request))
    if response.status_code == 404:
        return LabelSectionResult(
            request=request, total_count=0, returned_count=0, results=[]
        )

    source = _LabelResponse.model_validate_json(response.content)
    (label,) = source.results
    sections = [_section(label, field) for field in request.fields]
    row = DrugLabel(
        set_id=label.set_id,
        effective_time=label.effective_time,
        sections=ResultList[LabelSection].from_rows(
            sections, total_count=len(sections)
        ),
    )
    return LabelSectionResult(
        request=request,
        total_count=source.meta.results.total,
        returned_count=1,
        results=[row],
    )
