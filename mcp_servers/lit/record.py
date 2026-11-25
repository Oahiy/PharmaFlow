"""Capture the Task 2 requests through the lit-mcp HTTP transport."""

import argparse
import asyncio
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import httpx

from .config import LitSettings
from .http import Recordings, open_http
from .models import (
    LabelSectionInput,
    PubMedSearchInput,
    TrialAdverseEventsInput,
    TrialSearchInput,
)
from .pubmed import PubMedSearchResponse
from .requests import (
    label_section_url,
    pubmed_fetch_url,
    pubmed_search_url,
    trial_adverse_events_url,
    trial_search_url,
)

PUBMED_KRAS = PubMedSearchInput(query='"KRAS G12C"[Title/Abstract]')
PUBMED_NO_MATCH = PubMedSearchInput(
    query='"pharmaflowtask2nomatchxq9z"[Title/Abstract]'
)
KRAS_TRIALS = TrialSearchInput(condition="NSCLC", query="KRAS G12C", phase="PHASE3")
CODEBREAK_200 = TrialAdverseEventsInput(nct_id="NCT04303780")
KRYSTAL_12 = TrialAdverseEventsInput(nct_id="NCT04685135")
SOTORASIB = LabelSectionInput(drug="sotorasib")
ADAGRASIB = LabelSectionInput(drug="adagrasib")
LABEL_NO_MATCH = LabelSectionInput(drug="pharmaflow-task2-no-match-xq9z")
_logger = logging.getLogger(__name__)


def recording_urls(source: str, *, pilot: bool) -> list[httpx.URL]:
    pubmed = [pubmed_search_url(PUBMED_KRAS), pubmed_search_url(PUBMED_NO_MATCH)]
    trials = [
        trial_adverse_events_url(CODEBREAK_200),
        trial_search_url(KRAS_TRIALS),
        trial_adverse_events_url(KRYSTAL_12),
    ]
    labels = [
        label_section_url(label) for label in (SOTORASIB, ADAGRASIB, LABEL_NO_MATCH)
    ]
    if pilot:
        return {
            "both": [pubmed[0], trials[0]],
            "pubmed": pubmed[:1],
            "trials": trials[:1],
            "labels": labels[:1],
        }[source]
    return {
        "all": pubmed + trials + labels,
        "pubmed": pubmed,
        "trials": trials,
        "labels": labels,
    }[source]


async def remaining_urls(directory: Path) -> list[httpx.URL]:
    """Use the pilot's PubMed IDs and plan only requests missing on disk."""
    search_url = pubmed_search_url(PUBMED_KRAS)
    async with open_http(
        LitSettings(mode="replay", recordings_dir=directory)
    ) as replay:
        pilot = await replay.get(search_url)
        await replay.get(trial_adverse_events_url(CODEBREAK_200))

    pmids = PubMedSearchResponse.model_validate_json(pilot.content).hits.pmids
    urls = recording_urls("all", pilot=False)
    if pmids:
        urls.insert(1, pubmed_fetch_url(pmids))
    recordings = Recordings(directory)
    return [url for url in urls if not recordings.contains("GET", url)]


async def record(
    settings: LitSettings,
    urls: Sequence[httpx.URL],
    *,
    fetch_pubmed_abstracts: bool = False,
) -> None:
    positive_pubmed_search = pubmed_search_url(PUBMED_KRAS)
    async with open_http(settings) as client:
        for url in urls:
            response = await client.get(url)
            if fetch_pubmed_abstracts and url == positive_pubmed_search:
                pmids = PubMedSearchResponse.model_validate_json(
                    response.content
                ).hits.pmids
                if pmids:
                    await client.get(pubmed_fetch_url(pmids))


async def record_remaining(settings: LitSettings) -> None:
    urls = await remaining_urls(settings.recordings_dir)
    if urls:
        await record(settings, urls)
    else:
        _logger.info("All Task 2 recordings are already present.")


def _pmid(value: str) -> str:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError(f"Invalid PMID: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--pilot",
        nargs="?",
        choices=("both", "pubmed", "trials", "labels"),
        const="both",
        help="Record the selected pilot requests.",
    )
    scope.add_argument(
        "--source",
        choices=("all", "pubmed", "trials", "labels"),
        default="all",
        help="Record PubMed searches and abstracts, trials, and/or labels (default: all).",
    )
    scope.add_argument(
        "--remaining",
        action="store_true",
        help="After the two pilot requests succeed, record only missing requests.",
    )
    scope.add_argument(
        "--pubmed-ids",
        nargs="+",
        type=_pmid,
        help="Record EFetch for the PMIDs from the positive ESearch, in result order.",
    )
    args = parser.parse_args()

    settings = LitSettings.from_env()
    if settings.mode != "record" or settings.ncbi_api_key is not None:
        parser.error("Set LIT_MCP_MODE=record and unset NCBI_API_KEY before recording.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.pubmed_ids is not None:
        if len(args.pubmed_ids) > PUBMED_KRAS.max_results:
            parser.error(
                f"--pubmed-ids accepts at most {PUBMED_KRAS.max_results} PMIDs."
            )
        asyncio.run(record(settings, [pubmed_fetch_url(args.pubmed_ids)]))
    elif args.pilot is not None:
        asyncio.run(record(settings, recording_urls(args.pilot, pilot=True)))
    elif args.remaining:
        asyncio.run(record_remaining(settings))
    else:
        asyncio.run(
            record(
                settings,
                recording_urls(args.source, pilot=False),
                fetch_pubmed_abstracts=True,
            )
        )


if __name__ == "__main__":
    main()
