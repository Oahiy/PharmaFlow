import re
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent import config, llm, mcp_client, persistence, tracing
from api import main


@pytest.fixture(autouse=True)
def no_local_credentials(monkeypatch):
    monkeypatch.setattr(main, "load_environment", lambda: None)
    for name in (
        *tracing.LANGFUSE_KEYS,
        "LANGFUSE_TRACING_ENABLED",
        "LANGFUSE_RELEASE",
    ):
        monkeypatch.delenv(name, raising=False)


async def test_api_lifespan_rejects_missing_langfuse_keys(monkeypatch):
    monkeypatch.setattr(
        tracing, "get_client", lambda: pytest.fail("initialized without keys")
    )
    with pytest.raises(RuntimeError, match=r"LANGFUSE_PUBLIC_KEY.*LANGFUSE_SECRET_KEY"):
        async with main.lifespan(FastAPI()):
            pytest.fail("API startup succeeded without credentials")


@pytest.mark.parametrize(
    ("enabled", "revision_source", "expected_tags"),
    [
        (True, "checkout", ["api-test-version", "test-git-commit"]),
        (False, "checkout", []),
        (True, "package", ["api-test-version"]),
        (True, "release", ["api-test-version", "build-git-commit"]),
    ],
)
async def test_api_lifespan_authentication_and_turn_callback(
    monkeypatch,
    tmp_path,
    fake_chat_model,
    enabled,
    revision_source,
    expected_tags,
):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/test")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LLM_API_KEY", "offline-model-key")
    monkeypatch.setenv("LLM_MODEL", "offline-model")

    if enabled:
        for name in tracing.LANGFUSE_KEYS:
            monkeypatch.setenv(name, "example-key")
    else:
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    if revision_source == "checkout":
        (tmp_path / ".git").mkdir()
    elif revision_source == "release":
        monkeypatch.setenv("LANGFUSE_RELEASE", "build-git-commit")

    def commit():
        assert enabled and revision_source == "checkout"
        return "test-git-commit"

    events = []
    metadata_seen = []
    trace_ids = []

    class FakeClient:
        def auth_check(self):
            events.append("auth")
            return True

        def flush(self):
            events.append("flush")

    class TraceRecorder(BaseCallbackHandler):
        def on_chain_start(
            self, serialized, inputs, *, parent_run_id, metadata, **kwargs
        ):
            if parent_run_id is None:
                metadata_seen.append(metadata)

    def default_client():
        assert enabled
        return FakeClient()

    def create_callback(trace_id):
        assert enabled
        trace_ids.append(trace_id)
        return TraceRecorder()

    @asynccontextmanager
    async def pool(database_url):
        yield object()

    async def checkpointer(pool):
        return InMemorySaver()

    @asynccontextmanager
    async def tools(*, lit_env):
        yield []

    monkeypatch.setattr(tracing, "get_client", default_client)
    monkeypatch.setattr(tracing, "langfuse_callback", create_callback)
    monkeypatch.setattr(config, "git_commit", commit)
    monkeypatch.setattr(persistence, "open_connection_pool", pool)
    monkeypatch.setattr(persistence, "setup_checkpointer", checkpointer)
    monkeypatch.setattr(mcp_client, "open_mcp_tools", tools)
    monkeypatch.setattr(
        llm,
        "create_llm",
        lambda settings: fake_chat_model(responses=[AIMessage(content="完成")]),
    )
    app = FastAPI(version="test-version")
    app.include_router(main.router)
    app.dependency_overrides[main.claim_thread] = lambda: None
    thread = uuid4()

    async with (
        main.lifespan(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        assert events == (["auth"] if enabled else [])
        response = await client.post(
            f"/api/threads/{thread}/runs", json={"message": "查询"}
        )
        assert response.status_code == 200
        assert "event: done" in response.text
        assert "完成" in response.text

    if enabled:
        assert events == ["auth", "flush"]
        (trace_id,) = trace_ids
        assert re.fullmatch(r"[a-f0-9]{32}", trace_id)
        (metadata,) = metadata_seen
        assert metadata["langfuse_session_id"] == str(thread)
        assert metadata["langfuse_tags"] == expected_tags
    else:
        assert events == trace_ids == metadata_seen == []
