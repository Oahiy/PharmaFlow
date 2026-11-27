"""HTTP transport, upstream rate limits, and file-backed record/replay."""

import asyncio
import hashlib
import logging
import tempfile
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote, quote_plus

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .config import LitSettings

PUBMED_HOST = "eutils.ncbi.nlm.nih.gov"
OPENFDA_HOST = "api.fda.gov"
OPENFDA_LABEL_PATH = "/drug/label.json"
HTTP_TIMEOUT = 15.0
MAX_RETRIES = 3
BACKOFF_BASE = 1.0
ERROR_BODY_LIMIT = 2048

type Sleep = Callable[[float], Awaitable[None]]
type Clock = Callable[[], float]
_logger = logging.getLogger(__name__)


class ReplayMissError(FileNotFoundError):
    """The requested response is absent from the replay directory."""


class _Recording(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    url: str
    status_code: int
    content_type: str | None
    body: str


class _FDAError(BaseModel):
    code: Literal["NOT_FOUND"]


class _FDANotFound(BaseModel):
    error: _FDAError


def _canonical_url(url: httpx.URL) -> httpx.URL:
    params = sorted(
        (name, value)
        for name, value in url.params.multi_items()
        if name.lower() != "api_key"
    )
    return url.copy_with(params=params, fragment=None)


def _redact(text: str, api_key: str, *, preserve_width: bool = False) -> str:
    for value in (quote(api_key, safe=""), quote_plus(api_key), api_key):
        replacement = "[redacted]"
        if preserve_width:
            replacement = replacement.ljust(len(value), "*")
        text = text.replace(value, replacement)
    return text.replace("api_key", "redacted_key")


def _is_label_not_found(response: httpx.Response) -> bool:
    url = response.request.url
    return (
        response.status_code == 404
        and url.host == OPENFDA_HOST
        and url.path == OPENFDA_LABEL_PATH
    )


def _is_retryable_status(response: httpx.Response) -> bool:
    return response.status_code == 429 or response.is_server_error


def _limit_error_detail(detail: str, *, reached_limit: bool = False) -> str:
    if reached_limit or len(detail) > ERROR_BODY_LIMIT:
        return f"{detail[:ERROR_BODY_LIMIT]} [truncated]"
    return detail


def _check_status(response: httpx.Response) -> None:
    if _is_label_not_found(response):
        try:
            _FDANotFound.model_validate_json(response.content)
        except ValidationError as exc:
            detail = _limit_error_detail(response.text).strip()
            suffix = f"\nUpstream response: {detail}" if detail else ""
            raise httpx.HTTPStatusError(
                f"HTTP 404 for {response.request.url}: {exc}{suffix}",
                request=response.request,
                response=response,
            ) from exc
        return

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()
        if not detail:
            raise
        raise httpx.HTTPStatusError(
            f"{exc}\nUpstream response: {detail}",
            request=response.request,
            response=response,
        ) from exc


async def _read_error_detail(response: httpx.Response, key: str | None) -> str:
    # Read far enough to redact credentials before clipping the diagnostic prefix.
    lookahead = len(quote(key, safe="")) if key is not None else 0
    read_size = ERROR_BODY_LIMIT + lookahead + 1
    detail = ""
    try:
        async for chunk in response.aiter_text(chunk_size=read_size):
            detail = chunk
            break
    except httpx.RequestError as exc:
        # The status is already terminal even if reading its diagnostic body fails.
        detail = f"Could not read upstream error body: {type(exc).__name__}: {exc}"

    reached_limit = len(detail) >= read_size
    if key is not None:
        # A shorter replacement could pull a partial key from the read-ahead into the preview.
        detail = _redact(detail, key, preserve_width=True)
    return _limit_error_detail(detail, reached_limit=reached_limit)


class Recordings:
    """Store final responses in atomic, UTF-8 JSON files."""

    def __init__(self, directory: Path):
        self.directory = directory

    def _path(self, request: httpx.Request) -> Path:
        key = f"{request.method} {request.url}"
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def path_for(self, method: str, url: httpx.URL) -> Path:
        """Resolve the recording filename for an upstream request."""
        return self._path(httpx.Request(method, _canonical_url(url)))

    def contains(self, method: str, url: httpx.URL) -> bool:
        """Check for an existing recording without contacting the upstream."""
        return self.path_for(method, url).is_file()

    def write(self, response: httpx.Response) -> Path:
        request = response.request
        recording = _Recording(
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            body=response.content.decode("utf-8"),
        )
        path = self._path(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.directory) as temp_dir:
            temp_path = Path(temp_dir) / path.name
            temp_path.write_text(recording.model_dump_json(indent=2) + "\n", "utf-8")
            temp_path.replace(path)
        return path

    def read(self, request: httpx.Request) -> httpx.Response:
        path = self._path(request)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise ReplayMissError(
                f"回放未命中: {request.method} {request.url}"
            ) from exc
        recording = _Recording.model_validate_json(raw)
        if (recording.method, recording.url) != (request.method, str(request.url)):
            raise ValueError(f"Recording request does not match its filename: {path}")
        headers = {}
        if recording.content_type is not None:
            headers["content-type"] = recording.content_type
        return httpx.Response(
            recording.status_code,
            content=recording.body,
            headers=headers,
            request=request,
        )


class ReplayTransport(httpx.AsyncBaseTransport):
    """Resolve requests entirely against recorded responses."""

    def __init__(self, recordings: Recordings):
        self._recordings = recordings

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.url = _canonical_url(request.url)
        response = self._recordings.read(request)
        _check_status(response)
        return response


class _PubMedLimiter:
    def __init__(self, rate: int, sleep: Sleep, clock: Clock):
        self._rate = rate
        self._sleep = sleep
        self._clock = clock
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._starts and self._starts[0] + 1 <= now:
                    self._starts.popleft()
                if len(self._starts) < self._rate:
                    self._starts.append(now)
                    return
                await self._sleep(self._starts[0] + 1 - now)


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after is not None:
        retry_after = retry_after.strip()
        if retry_after.isascii() and retry_after.isdecimal():
            return float(retry_after)
        try:
            date = parsedate_to_datetime(retry_after)
        except ValueError:
            pass
        else:
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            return max(0.0, date.timestamp() - time.time())
    return BACKOFF_BASE * 2**attempt


class UpstreamTransport(httpx.AsyncBaseTransport):
    """Wrap a wire transport with retry, PubMed throttling, and optional recording."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        *,
        api_key: str | None = None,
        recordings: Recordings | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ):
        self._transport = transport
        self._api_key = api_key
        self._recordings = recordings
        self._sleep = sleep
        self._pubmed = _PubMedLimiter(10 if api_key is not None else 3, sleep, clock)

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def _send(
        self, request: httpx.Request, *, final_attempt: bool
    ) -> httpx.Response:
        key = None
        wire_request = request
        if request.url.host == PUBMED_HOST:
            await self._pubmed.acquire()
            key = self._api_key
            if key is not None:
                # HTTPX's client-level request log only sees the key-free request.
                wire_request = httpx.Request(
                    request.method,
                    request.url.copy_set_param("api_key", key),
                    headers=request.headers,
                    stream=request.stream,
                    extensions=request.extensions,
                )

        _logger.info("Requesting %s %s", request.method, request.url)
        try:
            raw = await self._transport.handle_async_request(wire_request)
            try:
                headers = {
                    name: raw.headers[name]
                    for name in ("content-type", "retry-after")
                    if name in raw.headers
                }
                if key is not None:
                    headers = {
                        name: _redact(value, key) for name, value in headers.items()
                    }
                response = httpx.Response(
                    raw.status_code, headers=headers, request=request
                )
                if raw.is_success or _is_label_not_found(response):
                    content = await raw.aread()
                    if key is not None:
                        content = _redact(content.decode("utf-8"), key).encode("utf-8")
                    response = httpx.Response(
                        raw.status_code,
                        content=content,
                        headers=headers,
                        request=request,
                    )
                elif final_attempt or not _is_retryable_status(response):
                    detail = await _read_error_detail(raw, key)
                    response = httpx.Response(
                        raw.status_code, text=detail, request=request
                    )
                return response
            finally:
                await raw.aclose()
        except httpx.RequestError as exc:
            if key is not None:
                exc.args = (_redact(str(exc), key),)
                exc.request = request
                # The original transport cause can contain the authenticated URL.
                exc.__cause__ = None
            raise

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.url = _canonical_url(request.url)
        attempt = 0
        while True:
            try:
                response = await self._send(
                    request, final_attempt=attempt == MAX_RETRIES
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"{request.method} {request.url} failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                delay = _retry_delay(None, attempt)
            else:
                if not _is_retryable_status(response) or attempt == MAX_RETRIES:
                    _check_status(response)
                    if self._recordings is not None:
                        path = self._recordings.write(response)
                        _logger.info(
                            "Recorded HTTP %d to %s (%d bytes)",
                            response.status_code,
                            path,
                            path.stat().st_size,
                        )
                    return response
                delay = _retry_delay(response.headers.get("retry-after"), attempt)

            await self._sleep(delay)
            attempt += 1


def make_client(transport: httpx.AsyncBaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, timeout=HTTP_TIMEOUT, trust_env=False)


def open_http(
    settings: LitSettings, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    if settings.mode == "replay":
        return make_client(ReplayTransport(Recordings(settings.recordings_dir)))

    wire = transport if transport is not None else httpx.AsyncHTTPTransport()
    api_key = (
        settings.ncbi_api_key.get_secret_value()
        if settings.ncbi_api_key is not None
        else None
    )
    recordings = (
        Recordings(settings.recordings_dir) if settings.mode == "record" else None
    )
    return make_client(UpstreamTransport(wire, api_key=api_key, recordings=recordings))
