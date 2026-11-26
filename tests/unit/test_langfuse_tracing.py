import pytest

from agent import tracing


@pytest.fixture(autouse=True)
def clear_langfuse_config(monkeypatch):
    for name in (*tracing.LANGFUSE_KEYS, "LANGFUSE_TRACING_ENABLED"):
        monkeypatch.delenv(name, raising=False)


async def test_eval_rejects_disabled_tracing_before_checking_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setattr(
        tracing, "get_client", lambda: pytest.fail("client initialized")
    )
    with pytest.raises(RuntimeError, match="LANGFUSE_TRACING_ENABLED=false") as error:
        async with tracing.open_langfuse(required=True):
            pytest.fail("evaluation entered without tracing")
    assert "LANGFUSE_PUBLIC_KEY" not in str(error.value)


async def test_eval_reports_only_missing_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "example-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", " ")
    monkeypatch.setattr(
        tracing, "get_client", lambda: pytest.fail("client initialized")
    )
    with pytest.raises(RuntimeError, match="LANGFUSE_SECRET_KEY") as error:
        async with tracing.open_langfuse(required=True):
            pytest.fail("evaluation entered with a blank secret key")
    assert "LANGFUSE_PUBLIC_KEY" not in str(error.value)
    assert "example-public-key" not in str(error.value)


@pytest.mark.parametrize(
    "auth_result", [True, False, RuntimeError("original auth error")]
)
async def test_auth_check_and_flush_use_the_same_client(monkeypatch, auth_result):
    for name in tracing.LANGFUSE_KEYS:
        monkeypatch.setenv(name, f"fake-{name}")
    events = []

    class FakeClient:
        def auth_check(self):
            events.append(("auth", self))
            if isinstance(auth_result, RuntimeError):
                raise auth_result
            return auth_result

        def flush(self):
            events.append(("flush", self))

    client = FakeClient()
    monkeypatch.setattr(tracing, "get_client", lambda: client)
    if auth_result is True:
        async with tracing.open_langfuse(required=True) as authenticated:
            assert authenticated is client
            assert events == [("auth", client)]
        assert events == [("auth", client), ("flush", client)]
    else:
        expected = "返回 False" if auth_result is False else "original auth error"
        with pytest.raises(RuntimeError, match=expected):
            async with tracing.open_langfuse(required=True):
                pytest.fail("evaluation entered after failed auth")
        assert events == [("auth", client)]
