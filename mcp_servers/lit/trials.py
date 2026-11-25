"""Validate and project ClinicalTrials.gov trial responses."""

from typing import Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator
from pydantic.alias_generators import to_camel

from .models import (
    NctId,
    ResultList,
    TextExcerpt,
    TrialAdverseEvent,
    TrialAdverseEventsInput,
    TrialAdverseEventsResult,
    TrialEventGroup,
    TrialEventList,
    TrialEventStats,
    TrialEventTables,
    TrialSearchInput,
    TrialSearchResult,
    TrialSummary,
)
from .requests import trial_adverse_events_url, trial_search_url

EVENT_LIMIT = 50
SUMMARY_FIELD_LIMIT = 20


class _TrialRecord(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, strict=True)


class _Identification(_TrialRecord):
    nct_id: NctId


class _Protocol(_TrialRecord):
    identification_module: _Identification


class _StudyIdentification(_Identification):
    brief_title: str


class _StudyStatus(_TrialRecord):
    overall_status: str | None = None


class _StudyConditions(_TrialRecord):
    conditions: list[str] | None = None


class _StudyDesign(_TrialRecord):
    phases: list[str] | None = None


class _Intervention(_TrialRecord):
    name: str


class _StudyInterventions(_TrialRecord):
    interventions: list[_Intervention] | None = None


class _StudyProtocol(_TrialRecord):
    identification_module: _StudyIdentification
    status_module: _StudyStatus | None = None
    conditions_module: _StudyConditions | None = None
    design_module: _StudyDesign | None = None
    arms_interventions_module: _StudyInterventions | None = None


class _Study(_TrialRecord):
    protocol_section: _StudyProtocol


class _TrialSearchResponse(_TrialRecord):
    total_count: NonNegativeInt
    studies: list[_Study] | None = None

    @model_validator(mode="after")
    def check_count(self) -> Self:
        if (self.total_count == 0) != (not self.studies) or (
            self.studies is not None and len(self.studies) > self.total_count
        ):
            raise ValueError("ClinicalTrials.gov totalCount and studies disagree")
        return self


class _EventGroup(_TrialRecord):
    group_id: str = Field(validation_alias="id")
    title: str
    deaths_num_affected: NonNegativeInt | None = None
    deaths_num_at_risk: NonNegativeInt | None = None
    serious_num_affected: NonNegativeInt | None = None
    serious_num_at_risk: NonNegativeInt | None = None
    other_num_affected: NonNegativeInt | None = None
    other_num_at_risk: NonNegativeInt | None = None


class _EventStats(_TrialRecord):
    group_id: str
    num_affected: NonNegativeInt | None = None
    num_at_risk: NonNegativeInt | None = None
    num_events: NonNegativeInt | None = None


class _Event(_TrialRecord):
    term: str
    organ_system: str | None = None
    source_vocabulary: str | None = None
    assessment_type: str | None = None
    stats: list[_EventStats] | None = None


class _AdverseEvents(_TrialRecord):
    frequency_threshold: str | None = None
    time_frame: str | None = None
    description: str | None = None
    event_groups: list[_EventGroup] | None = None
    serious_events: list[_Event] | None = None
    other_events: list[_Event] | None = None


class _TrialResults(_TrialRecord):
    adverse_events_module: _AdverseEvents | None = None


class _AdverseEventsResponse(_TrialRecord):
    protocol_section: _Protocol
    has_results: bool
    results_section: _TrialResults | None = None


def _summary_field(values: list[str] | None) -> ResultList[str] | None:
    if values is None:
        return None
    return ResultList[str].from_rows(
        values[:SUMMARY_FIELD_LIMIT], total_count=len(values)
    )


def _study_summary(protocol: _StudyProtocol) -> TrialSummary:
    identification = protocol.identification_module
    status = protocol.status_module
    conditions = protocol.conditions_module
    design = protocol.design_module
    interventions = None
    if protocol.arms_interventions_module is not None:
        entries = protocol.arms_interventions_module.interventions
        if entries is not None:
            interventions = [entry.name for entry in entries]

    return TrialSummary(
        nct_id=identification.nct_id,
        title=identification.brief_title,
        overall_status=status.overall_status if status is not None else None,
        conditions=_summary_field(
            conditions.conditions if conditions is not None else None
        ),
        interventions=_summary_field(interventions),
        phases=_summary_field(design.phases if design is not None else None),
    )


async def search_trials(
    request: TrialSearchInput, http: httpx.AsyncClient
) -> TrialSearchResult:
    response = await http.get(trial_search_url(request))
    source = _TrialSearchResponse.model_validate_json(response.content)
    rows = []
    if source.studies is not None:
        rows = [
            _study_summary(study.protocol_section)
            for study in source.studies[: request.max_results]
        ]
    return TrialSearchResult(
        request=request,
        total_count=source.total_count,
        returned_count=len(rows),
        results=rows,
    )


def _event_row(nct_id: NctId, event: _Event) -> TrialAdverseEvent:
    stats = None
    if event.stats is not None:
        rows = [TrialEventStats(**row.model_dump()) for row in event.stats]
        stats = ResultList[TrialEventStats].from_rows(rows, total_count=len(rows))
    return TrialAdverseEvent(
        nct_id=nct_id,
        term=event.term,
        organ_system=event.organ_system,
        source_vocabulary=event.source_vocabulary,
        assessment_type=event.assessment_type,
        stats=stats,
    )


def _event_table(
    nct_id: NctId, events: list[_Event] | None, *, term: str | None
) -> TrialEventList | None:
    if events is None:
        return None

    matched = events
    if term is not None:
        needle = term.casefold()
        matched = [event for event in events if needle in event.term.casefold()]

    rows = [_event_row(nct_id, event) for event in matched[:EVENT_LIMIT]]
    return TrialEventList(
        total_count=len(events),
        matched_count=len(matched),
        returned_count=len(rows),
        results=rows,
    )


def parse_trial_adverse_events(
    body: bytes, request: TrialAdverseEventsInput
) -> TrialAdverseEventsResult:
    source = _AdverseEventsResponse.model_validate_json(body)
    nct_id = source.protocol_section.identification_module.nct_id
    if nct_id != request.nct_id:
        raise ValueError(f"ClinicalTrials.gov returned {nct_id} for {request.nct_id}")

    tables = None
    module = (
        source.results_section.adverse_events_module
        if source.results_section is not None
        else None
    )
    if module is not None:
        if not source.has_results:
            raise ValueError(
                f"ClinicalTrials.gov returned adverse events for {nct_id} with hasResults=false"
            )
        groups = None
        if module.event_groups is not None:
            rows = [
                TrialEventGroup(nct_id=nct_id, **row.model_dump())
                for row in module.event_groups
            ]
            groups = ResultList[TrialEventGroup].from_rows(rows, total_count=len(rows))
        tables = TrialEventTables(
            frequency_threshold=module.frequency_threshold,
            time_frame=TextExcerpt.from_text(module.time_frame)
            if module.time_frame is not None
            else None,
            description=TextExcerpt.from_text(module.description)
            if module.description is not None
            else None,
            groups=groups,
            serious=_event_table(nct_id, module.serious_events, term=request.term)
            if request.category != "other"
            else None,
            other=_event_table(nct_id, module.other_events, term=request.term)
            if request.category != "serious"
            else None,
        )

    return TrialAdverseEventsResult(
        nct_id=nct_id,
        category=request.category,
        term=request.term,
        has_results=source.has_results,
        adverse_events=tables,
    )


async def get_trial_adverse_events(
    request: TrialAdverseEventsInput, http: httpx.AsyncClient
) -> TrialAdverseEventsResult:
    response = await http.get(trial_adverse_events_url(request))
    return parse_trial_adverse_events(response.content, request)
