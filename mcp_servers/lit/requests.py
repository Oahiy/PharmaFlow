"""Build the same upstream requests for the tools and the recorder."""

import json

import httpx

from .http import OPENFDA_HOST, OPENFDA_LABEL_PATH, PUBMED_HOST
from .models import (
    LabelSectionInput,
    PubMedSearchInput,
    TrialAdverseEventsInput,
    TrialSearchInput,
)

_PUBMED_URL = f"https://{PUBMED_HOST}/entrez/eutils"
_TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
_OPENFDA_URL = f"https://{OPENFDA_HOST}{OPENFDA_LABEL_PATH}"


def pubmed_search_url(request: PubMedSearchInput) -> httpx.URL:
    return httpx.URL(
        f"{_PUBMED_URL}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": request.query,
            "retmode": "json",
            "retmax": request.max_results,
            "sort": "relevance",
        },
    )


def pubmed_fetch_url(pmids: list[str]) -> httpx.URL:
    return httpx.URL(
        f"{_PUBMED_URL}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )


def trial_search_url(request: TrialSearchInput) -> httpx.URL:
    params = {
        "query.cond": request.condition,
        "countTotal": "true",
        "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase,OverallStatus",
        "pageSize": str(request.max_results),
    }
    if request.drug is not None:
        params["query.intr"] = request.drug
    if request.phase is not None:
        params["filter.advanced"] = f"AREA[Phase]{request.phase}"
    if request.query is not None:
        params["query.term"] = request.query
    return httpx.URL(_TRIALS_URL, params=params)


def trial_adverse_events_url(request: TrialAdverseEventsInput) -> httpx.URL:
    return httpx.URL(
        f"{_TRIALS_URL}/{request.nct_id}",
        params={"fields": "NCTId,HasResults,AdverseEventsModule"},
    )


def label_section_url(request: LabelSectionInput) -> httpx.URL:
    return httpx.URL(
        _OPENFDA_URL,
        params={
            "search": f"openfda.generic_name.exact:{json.dumps(request.drug.upper())}",
            "sort": "effective_time:desc",
            "limit": 1,
        },
    )
