"""Find explicit citation and dataset identifiers without trusting model metadata."""

import re
from collections.abc import Iterable

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import JsonValue

from .state import ChatMessage

_UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
_CHUNK = rf"(?:pubmed:\d+|trial:NCT\d{{8}}|label:{_UUID})(?::[a-z0-9_-]+)*:\d+"
_REFERENCES = re.compile(
    rf"(?<![a-z0-9_])(?:(?P<chunk>{_CHUNK})|(?P<nct>NCT\d{{8}})|"
    rf"(?P<set_id>{_UUID})|PMID[\s:*：#`-]*(?P<pmid>\d+)|"
    rf"citation_id[\s*`]*[:：=][\s*`]*(?P<named_chunk>[a-z0-9_-][a-z0-9_:-]*))(?![a-z0-9_-])",
    re.IGNORECASE | re.ASCII,
)
_DATASET = re.compile(
    r"(?<![A-Za-z0-9_-])(?:candidates|upload_[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])"
)


def _normalize_chunk(value: str) -> str:
    source, identifier, *suffix = value.lower().split(":")
    if source == "trial":
        identifier = identifier.upper()
    return ":".join((source, identifier, *suffix))


def _citation_id(value: str) -> str:
    return (
        _normalize_chunk(value) if re.fullmatch(_CHUNK, value, re.IGNORECASE) else value
    )


def cited_ids(text: str) -> set[str]:
    ids = set()
    for match in _REFERENCES.finditer(text):
        if match.lastgroup == "pmid":
            ids.add(f"PMID:{match['pmid']}")
        elif match.lastgroup in {"chunk", "named_chunk"}:
            ids.add(_citation_id(match[match.lastgroup]))
        elif match.lastgroup == "nct":
            ids.add(match["nct"].upper())
        else:
            ids.add(match["set_id"].lower())
    return ids


def structured_result(message: ToolMessage) -> JsonValue:
    return None if message.artifact is None else message.artifact["structured_content"]


def payload_ids(
    data: JsonValue, *, datasets: bool = False, skip_request: bool = True
) -> set[str]:
    ids: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if skip_request and key == "request":
                continue
            if isinstance(value, str):
                if (
                    key in {"pmid", "source_id"}
                    and value.isascii()
                    and value.isdecimal()
                ):
                    ids.add(f"PMID:{value}")
                elif key == "citation_id":
                    ids.add(_citation_id(value))
                elif key in {"set_id", "nct_id", "source_id"}:
                    ids.add(
                        value.upper()
                        if value.upper().startswith("NCT")
                        else value.lower()
                    )
            ids.update(payload_ids(value, datasets=datasets, skip_request=skip_request))
    elif isinstance(data, list):
        for item in data:
            ids.update(payload_ids(item, datasets=datasets, skip_request=skip_request))
    elif isinstance(data, str):
        ids.update(cited_ids(data))
        if datasets:
            ids.update(_DATASET.findall(data))
    return ids


def observed_ids(messages: Iterable[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.status == "success":
            payload = structured_result(message)
            ids.update(payload_ids(message.content if payload is None else payload))
    return ids


def history_ids(messages: Iterable[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        payload = (
            structured_result(message) if isinstance(message, ToolMessage) else None
        )
        ids.update(
            payload_ids(message.content if payload is None else payload, datasets=True)
        )
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                ids.update(payload_ids(call["args"], datasets=True, skip_request=False))
    return ids
