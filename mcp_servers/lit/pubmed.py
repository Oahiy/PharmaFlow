"""Validate PubMed E-utilities responses."""

from typing import Annotated, Self
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


class _ArticleFields(BaseModel):
    pmid: Pmid
    title: str | None
    abstract: str | None


def _abstract_text(sections: list[ElementTree.Element]) -> str | None:
    if not sections:
        return None
    paragraphs = []
    for section in sections:
        text = "".join(section.itertext()).strip()
        label = section.get("Label")
        paragraphs.append(f"{label}: {text}" if label is not None else text)
    return "\n\n".join(paragraphs)


def _article(node: ElementTree.Element) -> PubMedArticle:
    if node.tag == "PubmedArticle":
        record_type = "journal_article"
        pmid_path = "MedlineCitation/PMID"
        content_path = "MedlineCitation/Article"
    elif node.tag == "PubmedBookArticle":
        record_type = "book_article"
        pmid_path = "BookDocument/PMID"
        content_path = "BookDocument"
    else:
        raise ValueError(f"Unexpected PubMed EFetch record {node.tag}: {node.text!r}")

    title = node.find(f"{content_path}/ArticleTitle")
    source = _ArticleFields(
        pmid=node.findtext(pmid_path),
        title="".join(title.itertext()).strip() if title is not None else None,
        abstract=_abstract_text(node.findall(f"{content_path}/Abstract/AbstractText")),
    )
    return PubMedArticle(
        pmid=source.pmid,
        record_type=record_type,
        title=source.title,
        abstract=TextExcerpt.from_text(source.abstract)
        if source.abstract is not None
        else None,
    )


def parse_pubmed_articles(body: bytes, pmids: list[str]) -> list[PubMedArticle]:
    root = ElementTree.fromstring(body)
    if root.tag != "PubmedArticleSet":
        raise ValueError(f"Unexpected PubMed EFetch response {root.tag}: {root.text!r}")

    articles = [_article(node) for node in root]
    by_pmid = {article.pmid: article for article in articles}
    if len(articles) != len(pmids) or set(by_pmid) != set(pmids):
        actual = [article.pmid for article in articles]
        raise ValueError(f"PubMed EFetch returned PMIDs {actual} for {pmids}")
    return [by_pmid[pmid] for pmid in pmids]


async def search_pubmed(
    request: PubMedSearchInput, http: httpx.AsyncClient
) -> PubMedSearchResult:
    search = await http.get(pubmed_search_url(request))
    hits = PubMedSearchResponse.model_validate_json(search.content).hits
    pmids = hits.pmids[: request.max_results]
    rows = []
    if pmids:
        response = await http.get(pubmed_fetch_url(pmids))
        rows = parse_pubmed_articles(response.content, pmids)
    return PubMedSearchResult(
        request=request,
        total_count=hits.total_count,
        returned_count=len(rows),
        results=rows,
    )
