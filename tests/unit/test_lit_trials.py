import json

import pytest
from pydantic import ValidationError

from mcp_servers.lit.config import LitSettings
from mcp_servers.lit.http import open_http
from mcp_servers.lit.record import CODEBREAK_200
from mcp_servers.lit.requests import trial_adverse_events_url
from mcp_servers.lit.trials import parse_trial_adverse_events


@pytest.fixture
async def codebreak_response():
    async with open_http(LitSettings(mode="replay")) as client:
        return await client.get(trial_adverse_events_url(CODEBREAK_200))


def test_adverse_event_tables_preserve_counts_and_source_order(codebreak_response):
    payload = codebreak_response.json()["resultsSection"]["adverseEventsModule"]
    result = parse_trial_adverse_events(codebreak_response.content, CODEBREAK_200)

    assert result.has_results
    assert result.term is None
    tables = result.adverse_events
    assert tables.frequency_threshold == payload["frequencyThreshold"]
    assert tables.time_frame.text == payload["timeFrame"]
    assert tables.time_frame.original_length == len(payload["timeFrame"])
    assert not tables.time_frame.truncated
    assert tables.groups.total_count == tables.groups.returned_count == 3
    assert [group.title for group in tables.groups.results] == [
        "Docetaxel",
        "AMG 510",
        "Docetaxel, Then Switched to AMG 510",
    ]
    assert [group.serious_num_at_risk for group in tables.groups.results] == [
        151,
        169,
        46,
    ]

    assert tables.serious.total_count == 156
    assert tables.serious.matched_count == 156
    assert tables.serious.returned_count == 50
    assert [row.term for row in tables.serious.results] == [
        row["term"] for row in payload["seriousEvents"][:50]
    ]
    assert tables.other.total_count == tables.other.returned_count == 42
    assert tables.other.matched_count == 42
    assert [row.nct_id for row in tables.serious.results] == [CODEBREAK_200.nct_id] * 50
    first_serious = tables.serious.results[0]
    assert first_serious.term == "Anaemia"
    assert first_serious.source_vocabulary == "MedDRA 25.0"
    assert [arm.num_affected for arm in first_serious.stats.results] == [5, 1, 0]
    assert [arm.num_at_risk for arm in first_serious.stats.results] == [151, 169, 46]
    assert all(arm.num_events is None for arm in first_serious.stats.results)
    assert "description" not in tables.groups.results[0].model_dump()


def test_broad_event_filter_counts_all_matches_before_clipping(codebreak_response):
    request = CODEBREAK_200.model_copy(update={"category": "serious", "term": "i"})
    result = parse_trial_adverse_events(codebreak_response.content, request)

    assert result.term == "i"
    table = result.adverse_events.serious
    assert (table.total_count, table.matched_count, table.returned_count) == (
        156,
        142,
        50,
    )
    source_events = codebreak_response.json()["resultsSection"]["adverseEventsModule"][
        "seriousEvents"
    ]
    assert [row.term for row in table.results] == [
        row["term"] for row in source_events if "i" in row["term"].casefold()
    ][:50]


def test_missing_counts_stay_null_and_reported_zero_stays_zero(codebreak_response):
    payload = codebreak_response.json()
    module = payload["resultsSection"]["adverseEventsModule"]
    del module["eventGroups"][0]["seriousNumAtRisk"]
    stats = module["seriousEvents"][0]["stats"]
    del stats[0]["numAffected"]
    del stats[0]["numAtRisk"]
    model = parse_trial_adverse_events(json.dumps(payload).encode(), CODEBREAK_200)

    assert model.adverse_events.groups.results[0].serious_num_at_risk is None
    assert model.adverse_events.serious.results[0].stats.results[0].num_affected is None
    assert model.adverse_events.serious.results[0].stats.results[0].num_at_risk is None
    assert model.adverse_events.serious.results[0].stats.results[2].num_affected == 0


@pytest.mark.parametrize("term", [None, "pneumonitis"])
def test_trial_without_posted_results_has_no_event_tables(codebreak_response, term):
    payload = codebreak_response.json()
    payload["hasResults"] = False
    del payload["resultsSection"]

    request = CODEBREAK_200.model_copy(update={"term": term})
    result = parse_trial_adverse_events(json.dumps(payload).encode(), request)

    assert result.nct_id == CODEBREAK_200.nct_id
    assert not result.has_results
    assert result.adverse_events is None
    assert result.term == term


def test_malformed_adverse_event_count_fails_at_source_boundary(codebreak_response):
    payload = codebreak_response.json()
    payload["resultsSection"]["adverseEventsModule"]["seriousEvents"][0]["stats"][0][
        "numAtRisk"
    ] = "unknown"

    with pytest.raises(ValidationError, match="numAtRisk"):
        parse_trial_adverse_events(json.dumps(payload).encode(), CODEBREAK_200)
