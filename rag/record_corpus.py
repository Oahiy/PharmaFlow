"""Copy shared recordings and record the PubMed queries for the RAG corpus."""

import argparse
import asyncio
import logging
import os
from itertools import batched
from pathlib import Path
from shutil import copyfile

import httpx

from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR, LitSettings
from mcp_servers.lit.http import Recordings, open_http
from mcp_servers.lit.pubmed import (
    PubMedArticleRecord,
    PubMedSearchResponse,
    parse_pubmed_articles,
)
from mcp_servers.lit.record import (
    ADAGRASIB,
    CODEBREAK_200,
    KRAS_TRIALS,
    KRYSTAL_12,
    SOTORASIB,
)
from mcp_servers.lit.requests import (
    corpus_pubmed_search_url,
    label_section_url,
    pubmed_fetch_url,
    trial_adverse_events_url,
    trial_search_url,
)

from .config import CORPUS_RECORDINGS_DIR

TRIAL_REQUESTS = (CODEBREAK_200, KRYSTAL_12)
LABEL_REQUESTS = (SOTORASIB, ADAGRASIB)
PUBMED_BATCH_SIZE = 50


def shared_recording_urls() -> tuple[httpx.URL, ...]:
    return (
        trial_search_url(KRAS_TRIALS),
        *(trial_adverse_events_url(trial) for trial in TRIAL_REQUESTS),
        *(label_section_url(label) for label in LABEL_REQUESTS),
    )


def copy_shared_recordings(
    directory: Path = CORPUS_RECORDINGS_DIR,
) -> None:
    source = Recordings(DEFAULT_RECORDINGS_DIR)
    destination = Recordings(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for url in shared_recording_urls():
        copyfile(source.path_for("GET", url), destination.path_for("GET", url))


async def fetch_pubmed_corpus(http: httpx.AsyncClient) -> list[PubMedArticleRecord]:
    search = await http.get(corpus_pubmed_search_url())
    pmids = PubMedSearchResponse.model_validate_json(search.content).hits.pmids
    articles = []
    for batch in batched(pmids, PUBMED_BATCH_SIZE):
        response = await http.get(pubmed_fetch_url(batch))
        articles.extend(parse_pubmed_articles(response.content, batch))
    return articles


async def _record(settings: LitSettings) -> None:
    copy_shared_recordings(settings.recordings_dir)
    async with open_http(settings) as http:
        articles = await fetch_pubmed_corpus(http)
    print(f"Recorded {len(articles)} PubMed articles in {settings.recordings_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if os.getenv("LIT_MCP_MODE") != "record" or os.getenv("NCBI_API_KEY"):
        parser.error("Set LIT_MCP_MODE=record and unset NCBI_API_KEY before recording.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(
        _record(LitSettings(mode="record", recordings_dir=CORPUS_RECORDINGS_DIR))
    )


if __name__ == "__main__":
    main()
