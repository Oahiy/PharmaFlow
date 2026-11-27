import json
import re
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from mcp_servers.lit.config import LitSettings
from mcp_servers.lit.http import open_http
from mcp_servers.lit.labels import OpenFDALabel
from mcp_servers.lit.models import TrialSummary
from mcp_servers.lit.pubmed import parse_pubmed_articles
from mcp_servers.lit.trials import parse_adverse_events_response
from rag import embedding
from rag.build_corpus import build_corpus, label_chunks, pubmed_chunks, trial_chunks
from rag.chunks import MAX_CHUNK_TOKENS, parse_chunks, split_rows
from rag.config import PROJECT_ROOT, RagSettings
from rag.record_corpus import copy_shared_recordings, fetch_pubmed_corpus


def count_tokens(text: str) -> int:
    """Use synthetic token lengths, with two model special tokens."""
    return len(re.findall(r"\w+|[^\w\s]", text)) + 2


@pytest.fixture
def pubmed_xml():
    return b"""<PubmedArticleSet>
      <PubmedBookArticle><BookDocument><PMID>42</PMID>
        <Book><PubDate><MedlineDate>2018 Mar-Apr</MedlineDate></PubDate></Book>
        <ArticleTitle>A book entry</ArticleTitle>
        <Abstract><AbstractText>Unstructured abstract.</AbstractText></Abstract>
      </BookDocument></PubmedBookArticle>
      <PubmedArticle><MedlineCitation><PMID>41</PMID><Article>
        <Journal><JournalIssue><PubDate><Year>2024</Year><MedlineDate>2023 Jan</MedlineDate>
        </PubDate></JournalIssue></Journal>
        <ArticleTitle>Inhibition of <i>KRAS</i></ArticleTitle>
        <Abstract><AbstractText Label="METHODS">We followed patients.</AbstractText>
        <AbstractText Label="RESULTS">Patients reported symptoms.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>43</PMID><Article>
        <ArticleTitle>Article without an abstract</ArticleTitle>
      </Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>44</PMID><Article>
        <Journal><JournalIssue><PubDate><MedlineDate>Winter</MedlineDate></PubDate></JournalIssue></Journal>
        <ArticleTitle>Article with no publication year</ArticleTitle>
        <Abstract><AbstractText>Another abstract.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""


def test_pubmed_parses_full_sections_and_years(pubmed_xml):
    structured, unstructured, no_abstract, no_year = parse_pubmed_articles(
        pubmed_xml, ("41", "42", "43", "44")
    )
    assert [
        article.year for article in (structured, unstructured, no_abstract, no_year)
    ] == [2024, 2018, None, None]
    assert [section.label for section in structured.abstract_sections] == [
        "METHODS",
        "RESULTS",
    ]
    (chunk,) = pubmed_chunks(structured, count_tokens)
    assert chunk.chunk_id == "pubmed:41:0"
    assert chunk.relevance_unit == "pubmed:41"
    assert str(chunk.source_url) == "https://pubmed.ncbi.nlm.nih.gov/41/"
    assert chunk.text.splitlines()[0] == "Inhibition of KRAS (2024)"
    assert (
        "METHODS: We followed patients.\n\nRESULTS: Patients reported symptoms."
        in chunk.text
    )
    (chunk,) = pubmed_chunks(unstructured, count_tokens)
    assert chunk.section == "Abstract"
    assert chunk.text == "A book entry (2018)\nUnstructured abstract."
    (chunk,) = pubmed_chunks(no_year, count_tokens)
    assert chunk.year is None
    assert chunk.text.splitlines()[0].endswith("(year not reported)")
    assert no_abstract.abstract_sections is None
    assert pubmed_chunks(no_abstract, count_tokens) == []


def test_long_pubmed_abstract_splits_by_section_then_by_sentence(pubmed_xml):
    first = " ".join(["First"] * 250) + "."
    second = " ".join(["Second"] * 250) + "."
    xml = pubmed_xml.replace(b"We followed patients.", f"{first} {second}".encode())
    article = parse_pubmed_articles(xml, ("41", "42", "43", "44"))[0]

    chunks = pubmed_chunks(article, count_tokens)

    assert [chunk.chunk_id for chunk in chunks] == [f"pubmed:41:{i}" for i in range(3)]
    assert [chunk.section for chunk in chunks] == ["METHODS", "METHODS", "RESULTS"]
    assert [chunk.text.split("\n", 1)[1] for chunk in chunks] == [
        first,
        second,
        "Patients reported symptoms.",
    ]
    assert all(
        chunk.text.startswith("Inhibition of KRAS (2024) | ") for chunk in chunks
    )
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)


def test_trial_keeps_all_event_rows_and_splits_by_category_and_organ_system():
    nct_id = "NCT04303780"
    study = TrialSummary(
        nct_id=nct_id,
        title="CodeBreaK trial",
        overall_status=None,
        conditions=None,
        interventions=None,
        phases=None,
    )
    groups = [
        {
            "id": "A",
            "title": "Docetaxel",
            "seriousNumAtRisk": 151,
            "otherNumAtRisk": 157,
        },
        {"id": "B", "title": "AMG 510", "seriousNumAtRisk": 169, "otherNumAtRisk": 171},
    ]
    event = {
        "term": "Diarrhoea",
        "organSystem": "Gastrointestinal disorders",
        "stats": [
            {"groupId": "A", "numAffected": 1, "numAtRisk": 151},
            {"groupId": "B", "numAffected": 2, "numAtRisk": 169},
        ],
    }
    payload = {
        "hasResults": True,
        "protocolSection": {"identificationModule": {"nctId": nct_id}},
        "resultsSection": {
            "adverseEventsModule": {
                "frequencyThreshold": "5",
                "timeFrame": "Through week 30",
                "description": "Events as reported by study arms.",
                "eventGroups": groups,
                "seriousEvents": [
                    *({**event, "term": f"Diarrhoea {i}"} for i in range(52)),
                    {
                        "term": "Wheeze",
                        "organSystem": "Respiratory, thoracic disorders",
                    },
                ],
                "otherEvents": [
                    {**event, "stats": [{"groupId": "A", "numAtRisk": 157}]}
                ],
            }
        },
    }
    source = parse_adverse_events_response(json.dumps(payload).encode(), nct_id)
    chunks = trial_chunks(study, source.adverse_events, count_tokens)

    overview, *event_chunks = chunks
    assert overview.chunk_id == f"trial:{nct_id}:overview:0"
    assert overview.text.splitlines()[1:] == [
        "Reporting threshold (%): 5",
        "Time frame: Through week 30",
        "Description: Events as reported by study arms.",
    ]
    assert all(chunk.year is None for chunk in chunks)
    assert str(overview.source_url) == f"https://clinicaltrials.gov/study/{nct_id}"
    serious_id = f"trial:{nct_id}:serious:gastrointestinal-disorders"
    serious = [chunk for chunk in event_chunks if chunk.relevance_unit == serious_id]
    assert len(serious) > 1
    assert [chunk.chunk_id for chunk in serious] == [
        f"{serious_id}:{i}" for i in range(len(serious))
    ]
    assert {chunk.text.splitlines()[0] for chunk in serious} == {
        f"CodeBreaK trial | {nct_id} | Serious adverse events | Gastrointestinal disorders | At risk: Docetaxel 151; AMG 510 169"
    }
    assert [row for chunk in serious for row in chunk.text.splitlines()[1:]] == [
        f"Diarrhoea {i}: Docetaxel 1/151; AMG 510 2/169" for i in range(52)
    ]
    wheeze, other = event_chunks[-2:]
    assert wheeze.chunk_id == f"trial:{nct_id}:serious:respiratory-thoracic-disorders:0"
    assert wheeze.text.endswith("Wheeze: counts not reported")
    assert other.chunk_id == f"trial:{nct_id}:other:gastrointestinal-disorders:0"
    assert (
        "At risk: Docetaxel 157; AMG 510 171\nDiarrhoea: Docetaxel not reported/157"
        in other.text
    )
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)


def test_label_numbered_sections_and_html_rows_with_repeated_headers():
    values = [f"Adverse event {i} " + "finding " * 21 + "reported" for i in range(25)]
    rows = "".join(f"<tr><td>{value}</td><td>0</td><td/></tr>" for value in values)
    explicit_table = (
        "<table><caption>Table 1. Incidence &ge; 10%</caption>"
        "<thead><tr><th rowspan='2'>Event</th><th colspan='2'>Arms</th></tr>"
        "<tr><th>Active</th><th>Control</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    styled_table = """<table><caption>Table 2. Laboratory abnormalities</caption><tbody>
      <tr><td><content styleCode="bold">Event</content></td>
          <td colspan="2"><content styleCode="bold">Study arm</content></td></tr>
      <tr><td/><td><content styleCode="bold">All Grades (%)</content></td>
          <td><content styleCode="bold">Grade 3 or 4 (%)</content></td></tr>
      <tr><td colspan="3"><content styleCode="bold">Chemistry</content></td></tr>
      <tr><td>AST increased</td><td>22</td><td>1</td></tr>
    </tbody></table>"""
    label = OpenFDALabel(
        set_id="example-spl",
        effective_time="20261103",
        warnings_and_cautions=[
            (
                "5 WARNINGS AND PRECAUTIONS See (5.1) and (5.2 ). "
                "5.1 Hepatotoxicity Monitor liver function. "
                "5.2 Pneumonitis Monitor patients."
            )
        ],
        adverse_reactions=[
            "6 ADVERSE REACTIONS 6.1 Clinical Trials Experience The rate was 6.2 % in one arm."
        ],
        adverse_reactions_table=[explicit_table, styled_table],
    )

    chunks, missing = label_chunks("sotorasib", label, count_tokens)

    assert missing == ["boxed_warning"]
    prefix = "label:example-spl"
    assert [chunk.chunk_id for chunk in chunks[:5]] == [
        f"{prefix}:warnings_and_cautions:intro:0",
        f"{prefix}:warnings_and_cautions:5.1:0",
        f"{prefix}:warnings_and_cautions:5.2:0",
        f"{prefix}:adverse_reactions:intro:0",
        f"{prefix}:adverse_reactions:6.1:0",
    ]
    assert (
        chunks[1].text
        == "sotorasib | Warnings and Precautions (5.1)\n5.1 Hepatotoxicity Monitor liver function."
    )
    assert all(chunk.year == 2026 for chunk in chunks)
    assert (
        str(chunks[0].source_url)
        == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=example-spl"
    )
    first_table_id = f"{prefix}:adverse_reactions_table:table-1"
    first_table = [chunk for chunk in chunks if chunk.relevance_unit == first_table_id]
    assert len(first_table) > 1
    assert [chunk.chunk_id for chunk in first_table] == [
        f"{first_table_id}:{i}" for i in range(len(first_table))
    ]
    headers = [
        "sotorasib | Adverse Reactions Table (table-1)",
        "Table 1. Incidence ≥ 10%",
        "Event | Arms",
        "Active | Control",
    ]
    assert all(chunk.text.splitlines()[:4] == headers for chunk in first_table)
    assert [
        row.rstrip() for chunk in first_table for row in chunk.text.splitlines()[4:]
    ] == [f"{value} | 0 |" for value in values]
    assert chunks[-1].text.splitlines()[1:] == [
        "Table 2. Laboratory abnormalities",
        "Event | Study arm",
        " | All Grades (%) | Grade 3 or 4 (%)",
        "Chemistry",
        "AST increased | 22 | 1",
    ]
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)


def test_chunk_token_budget_includes_the_context_line():
    row = " ".join(["token"] * (MAX_CHUNK_TOKENS - 3))
    (chunk,) = split_rows("Context", [row], count_tokens, "test:row")
    assert count_tokens(chunk) == 448
    with pytest.raises(ValueError, match="test:row.*449 tokens including context"):
        split_rows("More context", [row], count_tokens, "test:row")


def test_jsonl_validation_rejects_mismatched_source_and_duplicate_ids(pubmed_xml):
    article = parse_pubmed_articles(pubmed_xml, ("41", "42", "43", "44"))[0]
    (chunk,) = pubmed_chunks(article, count_tokens)
    line = chunk.model_dump_json().encode() + b"\n"
    assert parse_chunks(line) == [chunk]
    with pytest.raises(ValidationError, match="Invalid chunk ID"):
        parse_chunks(line.replace(b"pubmed:41:0", b"pubmed:42:0"))
    with pytest.raises(ValueError, match="Duplicate chunk IDs.*pubmed:41:0"):
        parse_chunks(line + line)


async def test_fixture_replay_build_writes_deterministic_jsonl_and_missing_sections(
    pubmed_xml, tmp_path
):
    directory = tmp_path / "recordings"
    copy_shared_recordings(directory)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(
                200,
                json={
                    "esearchresult": {"count": "4", "idlist": ["41", "42", "43", "44"]}
                },
            )
        return httpx.Response(200, content=pubmed_xml)

    async with open_http(
        LitSettings(mode="record", recordings_dir=directory),
        transport=httpx.MockTransport(respond),
    ) as http:
        await fetch_pubmed_corpus(http)

    first_path, second_path = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first_report = await build_corpus(directory, first_path, count_tokens)
    second_report = await build_corpus(directory, second_path, count_tokens)

    contents = first_path.read_bytes()
    assert contents == second_path.read_bytes()
    assert first_report == second_report
    assert first_report.corpus_sha256 == sha256(contents).hexdigest()
    assert first_report.pubmed_without_abstract == ["43"]
    assert [
        (section.drug, section.field) for section in first_report.missing_label_sections
    ] == [("sotorasib", "boxed_warning"), ("adagrasib", "boxed_warning")]
    chunks = parse_chunks(contents)
    assert first_report.chunk_count == len(chunks)
    assert [chunk.source_id for chunk in chunks if chunk.source_type == "pubmed"] == [
        "41",
        "42",
        "44",
    ]
    assert {chunk.source_id for chunk in chunks if chunk.source_type == "trial"} == {
        "NCT04303780",
        "NCT04685135",
    }
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)


def test_tokenizer_uses_the_model_cache_without_padding_or_truncation(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", ".cache/test-models")
    tokenizer = Mock()
    tokenizer.encode.return_value = SimpleNamespace(ids=[101, 42, 102])
    model = Mock(tokenizer=tokenizer)
    dense = Mock(return_value=SimpleNamespace(model=model))
    monkeypatch.setattr(embedding, "TextEmbedding", dense)

    count = embedding.load_chunk_token_counter(RagSettings.from_env().cache_path)

    dense.assert_called_once_with(
        "BAAI/bge-small-en-v1.5",
        cache_dir=str(PROJECT_ROOT / ".cache/test-models"),
        lazy_load=True,
        local_files_only=True,
    )
    model._ensure_tokenizer.assert_called_once_with()
    tokenizer.no_truncation.assert_called_once_with()
    tokenizer.no_padding.assert_called_once_with()
    assert count("Context\nText") == 3
    tokenizer.encode.assert_called_once_with("Context\nText")
