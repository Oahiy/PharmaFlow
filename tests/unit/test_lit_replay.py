from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR, LitSettings
from mcp_servers.lit.http import open_http
from mcp_servers.lit.labels import get_label_section
from mcp_servers.lit.models import LabelSectionResult
from mcp_servers.lit.pubmed import search_pubmed
from mcp_servers.lit.record import (
    ADAGRASIB,
    KRAS_TRIALS,
    KRYSTAL_12,
    LABEL_NO_MATCH,
    PUBMED_KRAS,
    PUBMED_NO_MATCH,
    SOTORASIB,
    remaining_urls,
)
from mcp_servers.lit.requests import label_section_url, pubmed_fetch_url
from mcp_servers.lit.trials import get_trial_adverse_events, search_trials


@pytest.fixture
async def replay():
    async with open_http(LitSettings(mode="replay")) as client:
        yield client


async def test_pubmed_search_preserves_order_and_nonjournal_records(replay):
    result = await search_pubmed(PUBMED_KRAS, replay)
    pmids = ["31666701", "38072173", "32430388", "37782744", "37956237"]
    assert result.request == PUBMED_KRAS
    assert result.total_count == 1677
    assert result.returned_count == 5
    assert [row.pmid for row in result.results] == pmids
    assert [row.record_type for row in result.results] == ["journal_article"] * 3 + [
        "book_article"
    ] * 2
    assert (
        result.results[2].title
        == "EGFR Blockade Reverts Resistance to KRASG12C Inhibition in Colorectal Cancer."
    )
    assert [row.title for row in result.results[-2:]] == ["Sotorasib", "Adagrasib"]

    response = await replay.get(pubmed_fetch_url(pmids))
    xml = ElementTree.fromstring(response.content)
    nodes = xml.findall("./PubmedBookArticle/BookDocument/Abstract/AbstractText")
    assert [row.abstract.text for row in result.results[-2:]] == [
        "".join(node.itertext()).strip() for node in nodes
    ]
    assert all(not row.abstract.truncated for row in result.results)
    assert "authors" not in result.results[0].model_dump()


async def test_pubmed_zero_hits_skip_article_fetch(replay):
    result = await search_pubmed(PUBMED_NO_MATCH, replay)
    assert result.request == PUBMED_NO_MATCH
    assert result.total_count == result.returned_count == 0
    assert result.results == []


async def test_trial_search_returns_projected_summaries_and_total(replay):
    result = await search_trials(KRAS_TRIALS, replay)
    assert result.request == KRAS_TRIALS
    assert result.total_count == 25
    assert result.returned_count == 5
    codebreak, krystal, *_ = result.results
    assert (codebreak.nct_id, krystal.nct_id) == ("NCT04303780", "NCT04685135")
    assert codebreak.overall_status == "COMPLETED"
    assert codebreak.conditions.results == [
        "KRAS p, G12c Mutated /Advanced Metastatic NSCLC"
    ]
    assert codebreak.interventions.results == ["AMG 510", "Docetaxel"]
    assert codebreak.phases.results == ["PHASE3"]
    assert "nextPageToken" not in result.model_dump()


async def test_krystal_12_keeps_counts_for_the_requested_category(replay):
    query = KRYSTAL_12.model_copy(update={"category": "other"})
    result = await get_trial_adverse_events(query, replay)
    tables = result.adverse_events
    assert result.nct_id == "NCT04685135"
    assert tables.serious is None
    assert tables.other.total_count == 61
    assert tables.other.returned_count == 50
    assert [
        (group.title, group.other_num_at_risk) for group in tables.groups.results
    ] == [("Adagrasib", 298), ("Docetaxel", 140), ("Crossover Adagrasib", 44)]
    assert all(row.nct_id == result.nct_id for row in tables.other.results)


async def test_event_term_matches_both_source_phrases_and_keeps_zero_match_category(
    replay,
):
    request = KRYSTAL_12.model_copy(update={"term": "PNEUMONITIS"})
    result = await get_trial_adverse_events(request, replay)

    assert result.term == "PNEUMONITIS"
    assert result.has_results
    serious = result.adverse_events.serious
    assert (serious.total_count, serious.matched_count, serious.returned_count) == (
        171,
        2,
        2,
    )
    assert [row.term for row in serious.results] == [
        "Hypersensitivity pneumonitis",
        "Pneumonitis",
    ]
    other = result.adverse_events.other
    assert (other.total_count, other.matched_count, other.returned_count) == (61, 0, 0)
    assert other.results == []


@pytest.mark.parametrize(
    ("drug", "set_id", "effective_time", "table_count"),
    [
        (SOTORASIB.drug, "c80a362c-7ac3-4894-a076-0691e68ef8c1", "20261103", 4),
        (ADAGRASIB.drug, "0b8bf078-34c2-4f45-9012-38a8ac082b01", "20260831", 2),
    ],
)
async def test_label_returns_requested_sections_with_full_html(
    replay, drug, set_id, effective_time, table_count
):
    request = SOTORASIB.model_validate(
        {
            "drug": drug,
            "fields": [
                "adverse_reactions",
                "adverse_reactions_table",
                "boxed_warning",
                "warnings_and_cautions",
            ],
        }
    )
    result = await get_label_section(request, replay)
    source = (await replay.get(label_section_url(request))).json()["results"][0]
    assert result.request == request
    assert result.total_count == result.returned_count == 1
    (label,) = result.results
    assert (label.set_id, label.effective_time) == (set_id, effective_time)
    assert label.sections.total_count == label.sections.returned_count == 4
    assert [section.field for section in label.sections.results] == request.fields
    assert all(section.set_id == set_id for section in label.sections.results)

    adverse, tables, missing, warning = label.sections.results
    assert adverse.status == tables.status == warning.status == "present"
    (excerpt,) = adverse.content.results
    assert excerpt.text == source["adverse_reactions"][0][:4000]
    assert excerpt.original_length == len(source["adverse_reactions"][0])
    assert excerpt.truncated
    assert tables.content.total_count == tables.content.returned_count == table_count
    assert [block.text for block in tables.content.results] == source[
        "adverse_reactions_table"
    ]
    assert all(
        not block.truncated and block.original_length == len(block.text)
        for block in tables.content.results
    )
    assert missing.status == "missing"
    assert missing.content is None

    malformed = result.model_dump()
    malformed["results"][0]["sections"]["results"][0]["status"] = "missing"
    with pytest.raises(ValidationError):
        LabelSectionResult.model_validate(malformed)


async def test_confirmed_openfda_not_found_is_a_successful_empty_result(replay):
    result = await get_label_section(LABEL_NO_MATCH, replay)
    assert result.request == LABEL_NO_MATCH
    assert result.total_count == result.returned_count == 0
    assert result.results == []


async def test_all_required_requests_are_present_in_recordings():
    assert await remaining_urls(DEFAULT_RECORDINGS_DIR) == []
