"""Build the evidence chunk JSONL entirely from corpus recordings."""

import argparse
import asyncio
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

from pydantic import BaseModel

from mcp_servers.lit.config import LitSettings
from mcp_servers.lit.http import open_http
from mcp_servers.lit.labels import OpenFDALabel, fetch_label
from mcp_servers.lit.models import LabelField, SourceType, TrialSummary
from mcp_servers.lit.pubmed import PubMedArticleRecord
from mcp_servers.lit.record import KRAS_TRIALS
from mcp_servers.lit.requests import trial_adverse_events_url
from mcp_servers.lit.trials import (
    AdverseEvent,
    AdverseEvents,
    EventGroup,
    parse_adverse_events_response,
    search_trials,
)

from .chunks import (
    MAX_CHUNK_TOKENS,
    Chunk,
    TokenCounter,
    split_rows,
    split_text,
    write_chunks,
)
from .config import CHUNKS_PATH, CORPUS_RECORDINGS_DIR, RagSettings
from .embedding import load_chunk_token_counter
from .label_tables import parse_label_tables
from .record_corpus import LABEL_REQUESTS, TRIAL_REQUESTS, fetch_pubmed_corpus

_LABEL_TEXT_SECTIONS = (
    ("warnings_and_cautions", "Warnings and Precautions", 5),
    ("adverse_reactions", "Adverse Reactions", 6),
)


@dataclass(frozen=True)
class _Source:
    kind: SourceType
    identifier: str
    title: str | None
    year: int | None
    url: str

    @property
    def id_prefix(self) -> str:
        return f"{self.kind}:{self.identifier}"

    def chunks(
        self, unit_id: str, section: str, texts: Sequence[str], *, start: int = 0
    ) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{unit_id}:{index}",
                source_type=self.kind,
                source_id=self.identifier,
                title=self.title,
                year=self.year,
                section=section,
                text=text,
                source_url=self.url,
            )
            for index, text in enumerate(texts, start=start)
        ]


class MissingLabelSection(BaseModel):
    set_id: str
    drug: str
    field: LabelField


class BuildReport(BaseModel):
    chunk_count: int
    corpus_sha256: str
    pubmed_without_abstract: list[str]
    missing_label_sections: list[MissingLabelSection]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def pubmed_chunks(
    article: PubMedArticleRecord, count_tokens: TokenCounter
) -> list[Chunk]:
    if article.abstract_sections is None:
        return []
    sections = [section for section in article.abstract_sections if section.text]
    if not sections:
        return []

    source = _Source(
        "pubmed",
        article.pmid,
        article.title,
        article.year,
        f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/",
    )
    unit_id = source.id_prefix
    title = (
        _one_line(article.title)
        if article.title is not None
        else f"PubMed PMID {article.pmid}"
    )
    year = str(article.year) if article.year is not None else "year not reported"
    context = f"{title} ({year})"
    abstract = "\n\n".join(section.labeled_text for section in sections)
    full_text = f"{context}\n{abstract}"
    if count_tokens(full_text) <= MAX_CHUNK_TOKENS:
        section_name = (
            sections[0].label
            if len(sections) == 1 and sections[0].label
            else "Abstract"
        )
        return source.chunks(unit_id, section_name, [full_text])

    chunks = []
    for section in sections:
        section_name = section.label if section.label else "Abstract"
        section_context = f"{context} | {_one_line(section_name)}"
        texts = split_text(section_context, section.text, count_tokens, unit_id)
        chunks.extend(source.chunks(unit_id, section_name, texts, start=len(chunks)))
    return chunks


def _reported_count(value: int | None) -> str:
    return str(value) if value is not None else "not reported"


def _event_line(event: AdverseEvent, groups: dict[str, EventGroup]) -> str:
    term = _one_line(event.term)
    if not event.stats:
        return f"{term}: counts not reported"
    arms = "; ".join(
        f"{_one_line(groups[stat.group_id].title)} "
        f"{_reported_count(stat.num_affected)}/{_reported_count(stat.num_at_risk)}"
        for stat in event.stats
    )
    return f"{term}: {arms}"


def trial_chunks(
    study: TrialSummary, module: AdverseEvents, count_tokens: TokenCounter
) -> list[Chunk]:
    source = _Source(
        "trial",
        study.nct_id,
        study.title,
        None,
        f"https://clinicaltrials.gov/study/{study.nct_id}",
    )
    study_context = f"{_one_line(study.title)} | {study.nct_id}"
    overview_section = "Adverse event reporting overview"
    overview_id = f"{source.id_prefix}:overview"
    overview = "\n".join(
        f"{label}: {_one_line(value) if value is not None else 'not reported'}"
        for label, value in (
            ("Reporting threshold (%)", module.frequency_threshold),
            ("Time frame", module.time_frame),
            ("Description", module.description),
        )
    )
    texts = split_rows(
        f"{study_context} | {overview_section}", [overview], count_tokens, overview_id
    )
    chunks = source.chunks(overview_id, overview_section, texts)

    groups = (
        {group.group_id: group for group in module.event_groups}
        if module.event_groups is not None
        else {}
    )
    for category, events in (
        ("serious", module.serious_events),
        ("other", module.other_events),
    ):
        if events is None:
            continue
        at_risk = "; ".join(
            f"{_one_line(group.title)} "
            f"{_reported_count(group.serious_num_at_risk if category == 'serious' else group.other_num_at_risk)}"
            for group in groups.values()
        )
        if not at_risk:
            at_risk = "not reported"
        by_system: dict[str, list[str]] = defaultdict(list)
        for event in events:
            system = (
                _one_line(event.organ_system)
                if event.organ_system is not None
                else "Organ system not reported"
            )
            by_system[system].append(_event_line(event, groups))
        for system, rows in sorted(by_system.items()):
            slug = re.sub(r"[\W_]+", "-", system.lower())
            unit_id = f"{source.id_prefix}:{category}:{slug}"
            section = f"{category.capitalize()} adverse events | {system}"
            context = f"{study_context} | {section} | At risk: {at_risk}"
            texts = split_rows(context, rows, count_tokens, unit_id)
            chunks.extend(source.chunks(unit_id, section, texts))
    return chunks


def _label_parts(block: str, chapter: int) -> Iterator[tuple[str, str]]:
    # openFDA flattens SPL sections into one string. A label chapter prefix and
    # the heading's initial capital distinguish headings from citations and rates.
    pattern = rf"(?<![\w.])({chapter}\.[0-9]+(?:\.[0-9]+)*)\s+(?=[A-Z])"
    part, start = "intro", 0
    for match in re.finditer(pattern, block):
        text = block[start : match.start()].strip()
        if text:
            yield part, text
        part, start = match[1], match.start()
    text = block[start:].strip()
    if text:
        yield part, text


def label_chunks(
    drug: str, label: OpenFDALabel, count_tokens: TokenCounter
) -> tuple[list[Chunk], list[LabelField]]:
    source = _Source(
        "label",
        label.set_id,
        f"{drug} prescribing information",
        int(label.effective_time[:4]) if label.effective_time is not None else None,
        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?{urlencode({'setid': label.set_id})}",
    )
    chunks = []
    missing: list[LabelField] = []
    if label.boxed_warning is None:
        missing.append("boxed_warning")

    indices: Counter[str] = Counter()
    for field, section_name, chapter in _LABEL_TEXT_SECTIONS:
        blocks: list[str] | None = getattr(label, field)
        if blocks is None:
            missing.append(field)
            continue
        for block in blocks:
            for part, text in _label_parts(block, chapter):
                unit_id = f"{source.id_prefix}:{field}:{part}"
                section = f"{section_name} ({part})"
                texts = split_text(f"{drug} | {section}", text, count_tokens, unit_id)
                chunks.extend(
                    source.chunks(unit_id, section, texts, start=indices[unit_id])
                )
                indices[unit_id] += len(texts)

    field = "adverse_reactions_table"
    if label.adverse_reactions_table is None:
        missing.append(field)
    else:
        table_number = 0
        for block in label.adverse_reactions_table:
            tables = parse_label_tables(block)
            if not tables:
                raise ValueError(
                    f"{source.id_prefix}:{field} has no HTML table: {block[:120]!r}"
                )
            for table in tables:
                table_number += 1
                part = f"table-{table_number}"
                unit_id = f"{source.id_prefix}:{field}:{part}"
                section = f"Adverse Reactions Table ({part})"
                context = [f"{drug} | {section}"]
                if table.caption is not None:
                    context.append(table.caption)
                if table.rows:
                    context.extend(table.headers)
                rows = table.rows if table.rows else table.headers
                if not rows:
                    raise ValueError(f"{unit_id} has no HTML rows: {block[:120]!r}")
                texts = split_rows("\n".join(context), rows, count_tokens, unit_id)
                chunks.extend(source.chunks(unit_id, section, texts))
    return chunks, missing


async def build_corpus(
    recordings_dir: Path, chunks_path: Path, count_tokens: TokenCounter
) -> BuildReport:
    chunks = []
    pubmed_without_abstract = []
    missing_label_sections = []
    async with open_http(
        LitSettings(mode="replay", recordings_dir=recordings_dir)
    ) as http:
        for article in await fetch_pubmed_corpus(http):
            article_chunks = pubmed_chunks(article, count_tokens)
            if not article_chunks:
                pubmed_without_abstract.append(article.pmid)
            chunks.extend(article_chunks)

        search = await search_trials(KRAS_TRIALS, http)
        studies = {study.nct_id: study for study in search.results}
        for request in TRIAL_REQUESTS:
            response = await http.get(trial_adverse_events_url(request))
            module = parse_adverse_events_response(
                response.content, request.nct_id
            ).adverse_events
            if module is None:
                raise ValueError(
                    f"ClinicalTrials.gov recording has no adverse events module for {request.nct_id}"
                )
            chunks.extend(trial_chunks(studies[request.nct_id], module, count_tokens))

        for request in LABEL_REQUESTS:
            response = await fetch_label(request, http)
            if response is None:
                raise ValueError(
                    f"openFDA recording has no drug label for {request.drug}"
                )
            (label,) = response.results
            label_rows, missing = label_chunks(request.drug, label, count_tokens)
            chunks.extend(label_rows)
            missing_label_sections.extend(
                MissingLabelSection(set_id=label.set_id, drug=request.drug, field=field)
                for field in missing
            )

    return BuildReport(
        chunk_count=len(chunks),
        corpus_sha256=write_chunks(chunks_path, chunks),
        pubmed_without_abstract=pubmed_without_abstract,
        missing_label_sections=missing_label_sections,
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    count_tokens = load_chunk_token_counter(RagSettings.from_env().cache_path)
    report = asyncio.run(build_corpus(CORPUS_RECORDINGS_DIR, CHUNKS_PATH, count_tokens))
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
