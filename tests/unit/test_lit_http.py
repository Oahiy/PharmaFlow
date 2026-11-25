import asyncio
import json
import logging
import traceback
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from urllib.parse import quote_plus

import httpx
import pytest
from pydantic import ValidationError

from mcp_servers.lit.config import LitSettings
from mcp_servers.lit.http import (
    ERROR_BODY_LIMIT,
    OPENFDA_HOST,
    OPENFDA_LABEL_PATH,
    PUBMED_HOST,
    Recordings,
    ReplayMissError,
    UpstreamTransport,
    make_client,
    open_http,
)

TRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_URL = f"https://{PUBMED_HOST}/entrez/eutils/esearch.fcgi"
OPENFDA_URL = f"https://{OPENFDA_HOST}{OPENFDA_LABEL_PATH}"
FAKE_KEY = "pharmaflow-fake key/+only-for-tests"


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def mock_client(handler, time, **kwargs):
    return make_client(
        UpstreamTransport(
            httpx.MockTransport(handler),
            sleep=time.sleep,
            clock=time.monotonic,
            **kwargs,
        )
    )


def test_replay_configuration_validates_mode_and_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LIT_MCP_MODE", "invalid")
    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("NCBI_API_KEY", FAKE_KEY)
    with pytest.raises(ValidationError, match="live.*record.*replay") as invalid:
        LitSettings.from_env()
    assert FAKE_KEY not in str(invalid.value)

    monkeypatch.setenv("LIT_MCP_MODE", "replay")
    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(tmp_path / "missing"))
    with pytest.raises(ValidationError, match="LIT_MCP_RECORDINGS_DIR.*missing"):
        LitSettings.from_env()

    monkeypatch.setenv("LIT_MCP_RECORDINGS_DIR", str(tmp_path))
    settings = LitSettings.from_env()
    assert settings.recordings_dir == tmp_path
    assert settings.ncbi_api_key.get_secret_value() == FAKE_KEY


async def test_replay_miss_is_key_free_and_never_uses_upstream(tmp_path):
    def unexpected_request(request):
        pytest.fail(f"Replay contacted upstream: {request.url}")

    settings = LitSettings(
        mode="replay", recordings_dir=tmp_path, ncbi_api_key=FAKE_KEY
    )
    async with open_http(
        settings, transport=httpx.MockTransport(unexpected_request)
    ) as client:
        with pytest.raises(ReplayMissError) as error:
            await client.get(PUBMED_URL, params={"term": "KRAS", "api_key": FAKE_KEY})

    assert str(error.value) == f"回放未命中: GET {PUBMED_URL}?term=KRAS"
    assert list(tmp_path.iterdir()) == []


async def test_retries_429_and_503_and_captures_only_final_response(tmp_path):
    time = FakeTime()
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(503),
            httpx.Response(200, json={"result": "success"}),
        ]
    )

    def respond(request):
        assert list(tmp_path.iterdir()) == []
        return next(responses)

    async with mock_client(respond, time, recordings=Recordings(tmp_path)) as client:
        result = await client.get(TRIALS_URL)

    assert result.json() == {"result": "success"}
    assert time.waits == [5, 2]
    (path,) = tmp_path.iterdir()
    assert json.loads(path.read_text())["status_code"] == 200


async def test_retry_after_http_date():
    time = FakeTime()
    retry_at = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=60)
    responses = iter(
        [
            httpx.Response(
                429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}
            ),
            httpx.Response(200),
        ]
    )
    async with mock_client(lambda request: next(responses), time) as client:
        await client.get(TRIALS_URL)
    (delay,) = time.waits
    assert 57 <= delay <= 60


async def test_timeout_and_connect_error_are_retried():
    time = FakeTime()
    outcomes = iter([httpx.ReadTimeout, httpx.ConnectError, None])

    def respond(request):
        failure = next(outcomes)
        if failure is not None:
            raise failure("temporary failure", request=request)
        assert request.extensions["timeout"]["read"] == 15.0
        return httpx.Response(200, json={"result": "success"})

    async with mock_client(respond, time) as client:
        result = await client.get(TRIALS_URL)

    assert result.json() == {"result": "success"}
    assert time.waits == [1, 2]


async def test_final_timeout_preserves_last_error_and_writes_no_file(tmp_path):
    time = FakeTime()
    attempts = iter(range(1, 5))

    def respond(request):
        raise httpx.ReadTimeout(
            f"read timed out on call {next(attempts)}", request=request
        )

    async with mock_client(respond, time, recordings=Recordings(tmp_path)) as client:
        with pytest.raises(
            RuntimeError, match="ReadTimeout: read timed out on call 4"
        ) as error:
            await client.get(TRIALS_URL)

    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    assert time.waits == [1, 2, 4]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("status", "waits"), [(400, []), (404, []), (503, [1, 2, 4])])
async def test_unsuccessful_requests_keep_http_status_and_write_no_file(
    tmp_path, status, waits
):
    time = FakeTime()
    reason = f"upstream diagnostic for status {status}"
    async with mock_client(
        lambda request: httpx.Response(status, text=reason),
        time,
        recordings=Recordings(tmp_path),
    ) as client:
        with pytest.raises(httpx.HTTPStatusError, match=str(status)) as error:
            await client.get(TRIALS_URL)

    assert error.value.response.status_code == status
    assert reason in str(error.value)
    assert time.waits == waits
    assert list(tmp_path.iterdir()) == []


async def test_terminal_error_detail_is_redacted_before_clipping():
    time = FakeTime()
    secret_prefix = f"api_key={quote_plus(FAKE_KEY)}"
    body = "X" * (ERROR_BODY_LIMIT - 20) + secret_prefix + "X" * ERROR_BODY_LIMIT

    async with mock_client(
        lambda request: httpx.Response(400, text=body), time, api_key=FAKE_KEY
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.get(PUBMED_URL)

    detail = error.value.response.text
    assert "pharmaflow" not in detail
    assert "api_key" not in detail
    assert detail.endswith(" [truncated]")
    assert len(detail.removesuffix(" [truncated]")) == ERROR_BODY_LIMIT
    assert time.waits == []


async def test_repeated_key_echo_keeps_partial_second_key_out_of_error_detail():
    time = FakeTime()
    fake_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    second_echo_at = ERROR_BODY_LIMIT + 2
    gap = "x" * (second_echo_at - len(fake_key))
    body = f"{fake_key}{gap}{fake_key} end"

    async with mock_client(
        lambda request: httpx.Response(400, text=body), time, api_key=fake_key
    ) as client:
        with pytest.raises(httpx.HTTPStatusError, match="400 Bad Request") as error:
            await client.get(PUBMED_URL)

    detail = error.value.response.text
    assert "[redacted]" in detail
    assert fake_key[:8] not in detail
    assert fake_key[:8] not in str(error.value)
    assert detail.endswith(" [truncated]")


async def test_error_body_read_timeout_preserves_terminal_status():
    time = FakeTime()

    class IncompleteBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"partial detail"
            raise httpx.ReadTimeout("read deadline expired")

    async with mock_client(
        lambda request: httpx.Response(400, stream=IncompleteBody()), time
    ) as client:
        with pytest.raises(httpx.HTTPStatusError, match="400 Bad Request") as error:
            await client.get(TRIALS_URL)

    assert "ReadTimeout: read deadline expired" in str(error.value)
    assert time.waits == []


@pytest.mark.parametrize(("api_key", "rate"), [(None, 3), (FAKE_KEY, 10)])
async def test_pubmed_limits_concurrent_calls_and_uses_key_only_for_pubmed(
    api_key, rate
):
    time = FakeTime()
    starts = []

    async def respond(request):
        if request.url.host == PUBMED_HOST:
            assert request.url.params.get("api_key") == api_key
            starts.append(time.now)
        else:
            assert "api_key" not in request.url.params
        await asyncio.sleep(0)
        return httpx.Response(200)

    async with mock_client(respond, time, api_key=api_key) as client:
        await asyncio.gather(
            *(client.get(PUBMED_URL) for _ in range(2 * rate + 1)),
            client.get(TRIALS_URL),
        )

    assert starts == [0.0] * rate + [1.0] * rate + [2.0]
    assert time.waits == [1, 1]


async def test_recording_redacts_key_and_replays_with_reordered_parameters(
    tmp_path, caplog
):
    time = FakeTime()
    caplog.set_level(logging.INFO)

    def respond(request):
        assert request.url.params["api_key"] == FAKE_KEY
        return httpx.Response(200, json={"api_key": FAKE_KEY, "url": str(request.url)})

    async with mock_client(
        respond, time, api_key=FAKE_KEY, recordings=Recordings(tmp_path)
    ) as client:
        recorded = await client.get(
            PUBMED_URL, params=[("z", "2"), ("a", "x"), ("z", "1")]
        )

    (path,) = tmp_path.iterdir()
    captured = path.read_text()
    for secret in (FAKE_KEY, quote_plus(FAKE_KEY), "api_key"):
        assert secret not in captured
        assert secret not in path.name
        assert secret not in caplog.text
    assert json.loads(captured)["url"] == f"{PUBMED_URL}?a=x&z=1&z=2"

    settings = LitSettings(
        mode="replay", recordings_dir=tmp_path, ncbi_api_key="different-key"
    )
    async with open_http(settings) as client:
        replayed = await client.get(
            PUBMED_URL,
            params=[("a", "x"), ("z", 1), ("api_key", "different-key"), ("z", 2)],
        )
        with pytest.raises(ReplayMissError, match="POST"):
            await client.post(PUBMED_URL, params=[("a", "x"), ("z", 1), ("z", 2)])

    assert replayed.content == recorded.content


async def test_transport_error_redacts_key_even_in_chained_traceback():
    time = FakeTime()

    def respond(request):
        cause = ValueError(f"failed for {request.url}")
        raise httpx.ConnectError(
            f"cannot connect to {request.url}", request=request
        ) from cause

    async with mock_client(respond, time, api_key=FAKE_KEY) as client:
        with pytest.raises(
            RuntimeError, match="ConnectError: cannot connect to"
        ) as error:
            await client.get(PUBMED_URL)

    formatted = "".join(traceback.format_exception(error.value))
    assert FAKE_KEY not in formatted
    assert quote_plus(FAKE_KEY) not in formatted
    assert "?api_key=" not in formatted


@pytest.mark.parametrize("code", ["NOT_FOUND", "UNEXPECTED"])
async def test_only_expected_openfda_404_is_recordable(tmp_path, code):
    time = FakeTime()
    payload = {"error": {"code": code}}
    if code == "UNEXPECTED":
        payload["error"]["message"] = "Unknown label resource"
    async with mock_client(
        lambda request: httpx.Response(404, json=payload),
        time,
        recordings=Recordings(tmp_path),
    ) as client:
        if code == "NOT_FOUND":
            assert (await client.get(OPENFDA_URL)).status_code == 404
        else:
            with pytest.raises(httpx.HTTPStatusError, match="HTTP 404") as error:
                await client.get(OPENFDA_URL)
            assert "Unknown label resource" in str(error.value)
            assert list(tmp_path.iterdir()) == []
            return

    async with open_http(LitSettings(mode="replay", recordings_dir=tmp_path)) as client:
        result = await client.get(OPENFDA_URL)
    assert result.status_code == 404
    assert result.json() == {"error": {"code": "NOT_FOUND"}}
