"""Expose the lit-mcp tools over stdio."""

import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx

os.environ.setdefault("FASTMCP_ENV_FILE", os.devnull)

from fastmcp import Context, FastMCP

from . import labels, pubmed, trials
from .config import LitSettings
from .http import open_http
from .models import (
    LabelSectionInput,
    LabelSectionResult,
    PubMedSearchInput,
    PubMedSearchResult,
    TrialAdverseEventsInput,
    TrialAdverseEventsResult,
    TrialSearchInput,
    TrialSearchResult,
)


def _client_from_env() -> httpx.AsyncClient:
    return open_http(LitSettings.from_env())


def create_server(
    client_factory: Callable[
        [], AbstractAsyncContextManager[httpx.AsyncClient]
    ] = _client_from_env,
) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, httpx.AsyncClient]]:
        async with client_factory() as http:
            yield {"http": http}

    mcp = FastMCP("lit-mcp", lifespan=lifespan)
    mcp.tool(search_pubmed)
    mcp.tool(search_trials)
    mcp.tool(get_trial_adverse_events)
    mcp.tool(get_label_section)
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


mcp = create_server()


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
