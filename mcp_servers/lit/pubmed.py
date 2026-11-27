"""Validate PubMed E-utilities responses."""

import re
from collections.abc import Sequence
from typing import Annotated, Literal, Self
from xml.etree import ElementTree

import httpx
from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    StringConstraints,
    model_validator,
)

from .models import PubMedArticle, PubMedSearchInput, PubMedSearchResult, TextExcerpt
from .requests import pubmed_fetch_url, pubmed_search_url

Pmid = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*$")]


class PubMedHits(BaseModel):
    total_count: NonNegativeInt = Field(validation_alias="count")
    pmids: list[Pmid] = Field(validation_alias="idlist")

    @model_validator(mode="after")
    def check_count(self) -> Self:
        if (self.total_count == 0) != (not self.pmids) or len(
            self.pmids
        ) > self.total_count:
            raise ValueError("PubMed ESearch count and PMID list disagree")
        if len(set(self.pmids)) != len(self.pmids):
            raise ValueError("PubMed ESearch returned duplicate PMIDs")
        return self


class PubMedSearchResponse(BaseModel):
    hits: PubMedHits = Field(validation_alias="esearchresult")


class PubMedAbstractSection(BaseModel):
    label: str | None
    text: str

    @property
    def labeled_text(self) -> str:
        return f"{self.label}: {self.text}" if self.label is not None else self.text


class PubMedArticleRecord(BaseModel):
    """Source fields from EFetch before any abstract truncation."""

    pmid: Pmid
    record_type: Literal["journal_article", "book_article"]
    title: str | None
    year: int | None
    abstract_sections: list[PubMedAbstractSection] | None


def _publication_year(date: ElementTree.Element | None) -> int | None:
    if date is None:
        return None
    year = date.findtext("Year")
    if year is not None and re.fullmatch(r"[0-9]{4}", year.strip()):
        return int(year)
    medline_date = date.findtext("MedlineDate")
    if medline_date is not None and re.match(r"[0-9]{4}", medline_date):
        return int(medline_date[:4])
    return None


def _article(node: ElementTree.Element) -> PubMedArticleRecord:
    if node.tag == "PubmedArticle":
        record_type = "journal_article"
        pmid_path = "MedlineCitation/PMID"
        content_path = "MedlineCitation/Article"
        pubdate_path = "Journal/JournalIssue/PubDate"
    elif node.tag == "PubmedBookArticle":
        record_type = "book_article"
        pmid_path = "BookDocument/PMID"
        content_path = "BookDocument"
        pubdate_path = "Book/PubDate"
    else:
        raise ValueError(f"Unexpected PubMed EFetch record {node.tag}: {node.text!r}")

    title = node.find(f"{content_path}/ArticleTitle")
    sections = [
        PubMedAbstractSection(
            label=section.get("Label"), text="".join(section.itertext()).strip()
        )
        for section in node.findall(f"{content_path}/Abstract/AbstractText")
    ]
    return PubMedArticleRecord(
        pmid=node.findtext(pmid_path),
        record_type=record_type,
        title="".join(title.itertext()).strip() if title is not None else None,
        year=_publication_year(node.find(f"{content_path}/{pubdate_path}")),
        abstract_sections=sections if sections else None,
    )


def parse_pubmed_articles(
    body: bytes, pmids: Sequence[str]
) -> list[PubMedArticleRecord]:
    root = ElementTree.fromstring(body)
    if root.tag != "PubmedArticleSet":
        raise ValueError(f"Unexpected PubMed EFetch response {root.tag}: {root.text!r}")

    articles = [_article(node) for node in root]
    by_pmid = {article.pmid: article for article in articles}
    if len(articles) != len(pmids) or set(by_pmid) != set(pmids):
        actual = [article.pmid for article in articles]
        raise ValueError(f"PubMed EFetch returned PMIDs {actual} for {pmids}")
    return [by_pmid[pmid] for pmid in pmids]


def _tool_article(source: PubMedArticleRecord) -> PubMedArticle:
    abstract = None
    if source.abstract_sections is not None:
        abstract = TextExcerpt.from_text(
            "\n\n".join(section.labeled_text for section in source.abstract_sections)
        )
    return PubMedArticle(
        pmid=source.pmid,
        record_type=source.record_type,
        title=source.title,
        abstract=abstract,
    )


async def search_pubmed(
    request: PubMedSearchInput, http: httpx.AsyncClient
) -> PubMedSearchResult:
    search = await http.get(pubmed_search_url(request))
    hits = PubMedSearchResponse.model_validate_json(search.content).hits
    pmids = hits.pmids[: request.max_results]
    rows = []
    if pmids:
        response = await http.get(pubmed_fetch_url(pmids))
        rows = [
            _tool_article(article)
            for article in parse_pubmed_articles(response.content, pmids)
        ]
    return PubMedSearchResult(
        request=request,
        total_count=hits.total_count,
        returned_count=len(rows),
        results=rows,
    )
