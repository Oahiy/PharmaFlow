"""Expose the lit-mcp tools over stdio."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from hashlib import sha256
from time import perf_counter
from typing import TypedDict

_STARTED_AT = perf_counter()
os.environ.setdefault("FASTMCP_ENV_FILE", os.devnull)

import httpx
from fastmcp import Context, FastMCP

from rag.config import CHUNKS_PATH, RagSettings
from rag.retrieval import Retriever, load_retriever

from . import labels, pubmed, trials
from .config import LitSettings
from .http import open_http
from .models import (
    EvidenceMatch,
    LabelSectionInput,
    LabelSectionResult,
    PubMedSearchInput,
    PubMedSearchResult,
    SearchEvidenceInput,
    SearchEvidenceResult,
    TrialAdverseEventsInput,
    TrialAdverseEventsResult,
    TrialSearchInput,
    TrialSearchResult,
)

_logger = logging.getLogger(__name__)


class _ServerResources(TypedDict):
    http: httpx.AsyncClient
    retriever: Retriever


def _client_from_env() -> httpx.AsyncClient:
    return open_http(LitSettings.from_env())


def _load_retriever_from_env() -> Retriever:
    corpus_sha256 = sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    return load_retriever(RagSettings.from_env(), corpus_sha256)


@asynccontextmanager
async def _retriever_from_env() -> AsyncIterator[Retriever]:
    retriever = await asyncio.to_thread(_load_retriever_from_env)
    try:
        yield retriever
    finally:
        await asyncio.to_thread(retriever.close)


def create_server(
    client_factory: Callable[
        [], AbstractAsyncContextManager[httpx.AsyncClient]
    ] = _client_from_env,
    retriever_factory: Callable[
        [], AbstractAsyncContextManager[Retriever]
    ] = _retriever_from_env,
) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[_ServerResources]:
        started = _STARTED_AT if __name__ == "__main__" else perf_counter()
        async with client_factory() as http, retriever_factory() as retriever:
            _logger.info("lit-mcp ready in %.3f s", perf_counter() - started)
            yield {"http": http, "retriever": retriever}

    mcp = FastMCP("lit-mcp", lifespan=lifespan)
    mcp.tool(search_pubmed)
    mcp.tool(search_trials)
    mcp.tool(get_trial_adverse_events)
    mcp.tool(get_label_section)
    mcp.tool(search_evidence)
    return mcp


async def search_pubmed(request: PubMedSearchInput, ctx: Context) -> PubMedSearchResult:
    """Search PubMed and fetch abstracts in search-result order.

    Args:
        request: PubMed search expression and a maximum number of returned articles.

    Returns:
        The search criteria, total hits, and cited abstracts with truncation metadata.
    """
    return await pubmed.search_pubmed(request, ctx.lifespan_context["http"])


async def search_trials(request: TrialSearchInput, ctx: Context) -> TrialSearchResult:
    """Search ClinicalTrials.gov by disease, intervention, phase, or keywords.

    Args:
        request: Condition, optional drug, phase and keywords, and the result limit.

    Returns:
        The search criteria and total hits, together with trial IDs, titles,
        statuses, conditions, interventions, and phases for the returned page.
    """
    return await trials.search_trials(request, ctx.lifespan_context["http"])


async def get_trial_adverse_events(
    request: TrialAdverseEventsInput, ctx: Context
) -> TrialAdverseEventsResult:
    """Read reported adverse events for one ClinicalTrials.gov study.

    Args:
        request: NCT ID and optionally only the serious or other category.
            Use term for a case-insensitive substring of the ClinicalTrials.gov
            event term, such as "Alanine aminotransferase increased". Matching
            uses the source wording and runs before the 50-row limit per category.

    Returns:
        Report availability, reporting threshold and time frame, study arms and
        their denominators, and event tables with source, matched, and returned
        row counts. The result echoes the term. A match count of zero means the
        source terms did not match the filter.
    """
    return await trials.get_trial_adverse_events(request, ctx.lifespan_context["http"])


async def get_label_section(
    request: LabelSectionInput, ctx: Context
) -> LabelSectionResult:
    """Read selected sections from the latest matching openFDA drug label.

    Args:
        request: Generic drug name and the requested label fields.

    Returns:
        The matching label's set ID, effective date, and sections with availability
        and truncation metadata. Adverse reaction tables contain the original HTML.
    """
    return await labels.get_label_section(request, ctx.lifespan_context["http"])


async def search_evidence(
    request: SearchEvidenceInput, ctx: Context
) -> SearchEvidenceResult:
    """Search local PubMed, trial, and drug label evidence by meaning and keywords.

    Args:
        request: English search terms. Translate Chinese questions into English
            before calling. Drug aliases are expanded automatically.

    Returns:
        The original and expanded queries and up to eight matching excerpts in
        descending rerank score order, with citation IDs and source links.
    """
    retriever = ctx.lifespan_context["retriever"]
    found = await asyncio.to_thread(retriever.search, request.query)
    return SearchEvidenceResult(
        query=found.query,
        expanded_query=found.expanded_query,
        results=[
            EvidenceMatch(
                citation_id=hit.chunk.chunk_id,
                source_type=hit.chunk.source_type,
                source_id=hit.chunk.source_id,
                title=hit.chunk.title,
                year=hit.chunk.year,
                section=hit.chunk.section,
                text=hit.chunk.text,
                rerank_score=hit.score,
                source_url=hit.chunk.source_url,
            )
            for hit in found.hits
        ],
    )


mcp = create_server()


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s")
    _logger.setLevel(logging.INFO)
    mcp.run(transport="stdio", show_banner=False)
