import pytest
from pydantic import ValidationError

from mcp_servers.lit.models import (
    LabelSectionInput,
    PubMedSearchInput,
    TrialAdverseEventsInput,
    TrialSearchInput,
)
from mcp_servers.lit.record import recording_urls
from mcp_servers.lit.requests import (
    label_section_url,
    pubmed_fetch_url,
    pubmed_search_url,
    trial_adverse_events_url,
    trial_search_url,
)


def test_pubmed_bounds_search_and_uses_ordered_ids_for_fetch():
    request = PubMedSearchInput(query='"KRAS G12C"[Title/Abstract]', max_results=3)
    search = pubmed_search_url(request)
    assert search.path.endswith("/esearch.fcgi")
    assert search.params["term"] == request.query
    assert search.params["retmax"] == "3"
    assert search.params["retmode"] == "json"

    fetch = pubmed_fetch_url(["123", "456"])
    assert fetch.path.endswith("/efetch.fcgi")
    assert fetch.params["id"] == "123,456"
    assert fetch.params["retmode"] == "xml"


def test_trials_search_uses_separate_filters_and_requests_only_summary_fields():
    request = TrialSearchInput(
        condition="NSCLC",
        drug="sotorasib",
        phase="PHASE3",
        query="KRAS G12C",
        max_results=4,
    )
    url = trial_search_url(request)
    assert url.path == "/api/v2/studies"
    assert dict(url.params) == {
        "query.cond": "NSCLC",
        "query.intr": "sotorasib",
        "query.term": "KRAS G12C",
        "filter.advanced": "AREA[Phase]PHASE3",
        "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus",
        "countTotal": "true",
        "pageSize": "4",
    }


def test_trial_category_and_term_are_local_selections():
    all_events = trial_adverse_events_url(TrialAdverseEventsInput(nct_id="NCT04303780"))
    serious_events = trial_adverse_events_url(
        TrialAdverseEventsInput(nct_id="NCT04303780", category="serious")
    )
    assert all_events == serious_events
    filtered_events = trial_adverse_events_url(
        TrialAdverseEventsInput(
            nct_id="NCT04303780", category="serious", term="pneumonitis"
        )
    )
    assert serious_events == filtered_events
    assert serious_events.path == "/api/v2/studies/NCT04303780"
    assert serious_events.params["fields"] == "NCTId,HasResults,AdverseEventsModule"


def test_openfda_requests_latest_label_for_exact_generic_name():
    request = LabelSectionInput(drug="sotorasib", fields=["adverse_reactions_table"])
    url = label_section_url(request)
    assert url == label_section_url(LabelSectionInput(drug="SOTORASIB"))
    assert dict(url.params) == {
        "search": 'openfda.generic_name.exact:"SOTORASIB"',
        "sort": "effective_time:desc",
        "limit": "1",
    }


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (PubMedSearchInput, {"query": " "}),
        (PubMedSearchInput, {"query": "KRAS", "max_results": 11}),
        (TrialSearchInput, {"condition": "NSCLC", "max_results": 0}),
        (TrialAdverseEventsInput, {"nct_id": "../NCT04303780"}),
        (TrialAdverseEventsInput, {"nct_id": "NCT04303780", "term": " \t "}),
        (LabelSectionInput, {"drug": "sotorasib", "fields": ["unknown"]}),
        (
            LabelSectionInput,
            {"drug": "sotorasib", "fields": ["warnings_and_precautions"]},
        ),
    ],
)
def test_tool_inputs_reject_invalid_requests(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_recording_manifest_covers_both_searches_trials_and_labels():
    urls = recording_urls("all", pilot=False)
    assert len(urls) == 8
    assert len(set(urls)) == len(urls)
    for source, count in (("pubmed", 2), ("trials", 3), ("labels", 3)):
        source_urls = recording_urls(source, pilot=False)
        assert len(source_urls) == count
        assert recording_urls(source, pilot=True) == source_urls[:1]
