from contextlib import asynccontextmanager

import httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from agent import llm, tracing
from agent.config import AgentSettings, LLMSettings
from eval import run, runner
from eval.runner import infrastructure_error, redact_text, secret_values


@pytest.mark.parametrize(
    ("kind", "status", "infrastructure"),
    [
        (AuthenticationError, 401, True),
        (PermissionDeniedError, 403, True),
        (InternalServerError, 503, True),
        (BadRequestError, 400, False),
        (UnprocessableEntityError, 422, False),
    ],
)
def test_model_error_kind_controls_abort(kind, status, infrastructure):
    request = httpx.Request("POST", "https://example.invalid")
    error = kind(
        "离线错误", response=httpx.Response(status, request=request), body=None
    )
    assert infrastructure_error(error) is infrastructure


def test_model_timeout_is_infrastructure_error():
    request = httpx.Request("POST", "https://example.invalid")
    assert infrastructure_error(APITimeoutError(request=request))


def test_redaction_preserves_numbers_outside_query_parameters(monkeypatch):
    for name in tracing.LANGFUSE_KEYS:
        monkeypatch.delenv(name, raising=False)
    settings = AgentSettings(
        database_url="postgresql://demo:offline-db-word@localhost/demo",
        llm=LLMSettings(
            model="offline",
            base_url="https://example.invalid/v1?v=1&api-key=offline%2Fquery%2Bkey",
            api_key="offline-model-key",
        ),
    )
    detail = (
        "attempt 1 HTTP 401 at https://example.invalid/v1/chat?v=1&api-key=offline%2Fquery%2Bkey "
        "echo offline/query+key"
    )
    redacted = redact_text(detail, secret_values(settings))
    assert "attempt 1 HTTP 401" in redacted
    assert "v=1" not in redacted
    assert "offline%2Fquery%2Bkey" not in redacted
    assert "offline/query+key" not in redacted


async def test_cli_reports_model_auth_error_without_printing_the_key(
    monkeypatch, fake_chat_model
):
    secret = "offline-only-model-credential"
    monkeypatch.setenv("DATABASE_URL", "postgresql://demo:test@localhost/demo")
    monkeypatch.setenv("LLM_MODEL", "offline")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1?v=1")
    monkeypatch.setenv("LLM_API_KEY", secret)
    request = httpx.Request("POST", "https://example.invalid")
    error = AuthenticationError(
        f"HTTP 401 rejected {secret}",
        response=httpx.Response(401, request=request),
        body=None,
    )

    @asynccontextmanager
    async def fake_langfuse(*, required):
        yield object()

    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(tracing, "open_langfuse", fake_langfuse)
    monkeypatch.setattr(
        llm, "create_llm", lambda settings: fake_chat_model(responses=[])
    )
    monkeypatch.setattr(runner, "run_suite", fail)
    with pytest.raises(
        SystemExit, match=r"AuthenticationError HTTP 401 rejected \[redacted\]"
    ) as failure:
        await run.evaluate("v2")
    assert failure.value.__cause__ is error
