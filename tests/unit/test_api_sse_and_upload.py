import asyncio
import json
from threading import Event
from uuid import UUID, uuid4

import httpx
import pytest
from docker.errors import ImageNotFound
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import STEP_LIMIT_MESSAGE, build_graph
from agent.tracing import trace_config
from api.main import (
    app,
    claim_thread,
    get_agent_graph,
    get_turn_tracer,
    get_upload_settings,
)
from api.schemas import RunRequest
from api.streaming import stream_run
from api.uploads import UploadSettings
from mcp_servers.chem.models import MoleculeInput
from mcp_servers.chem.tools import compute_properties
from sandbox import artifacts
from sandbox.executor import PythonExecution
from sandbox.models import RunPythonResult, SandboxError
from sandbox.tool import create_run_python_tool


@tool
def lookup(value: int) -> str:
    """Return a multi-line result for the streaming test."""
    return f'结果：\n"{value}"'


class StreamingModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "test-stream"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        raise AssertionError("The run must use the async streaming model")

    async def _astream(self, messages, **kwargs):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            if last.content == "引发异常":
                raise RuntimeError('模拟错误\n"模型"')
            if last.content == "继续":
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=messages[-2].text)
                )
            else:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": "lookup",
                                "args": '{"value":3}',
                                "id": "call-1",
                                "index": 0,
                            }
                        ],
                    )
                )
        else:
            for token in ["答", '案：\n"3"']:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))


@pytest.fixture
async def client(tmp_path, monkeypatch):
    graph = build_graph(StreamingModel(), [lookup], InMemorySaver())
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100")
    monkeypatch.setenv("MAX_UPLOAD_ROWS", "2")
    monkeypatch.setitem(app.dependency_overrides, claim_thread, lambda: None)
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    monkeypatch.setitem(app.dependency_overrides, get_turn_tracer, lambda: None)
    monkeypatch.setitem(
        app.dependency_overrides, get_upload_settings, UploadSettings.from_env
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as session:
        yield session


def events(response: httpx.Response):
    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.endswith("\n\n")
    result = []
    for frame in response.text.removesuffix("\n\n").split("\n\n"):
        event, data = frame.split("\n")
        assert event.startswith("event: ")
        assert data.startswith("data: ")
        name = event.removeprefix("event: ")
        assert name in {"node", "tool_call", "tool_result", "token", "done", "error"}
        result.append((name, json.loads(data.removeprefix("data: "))))
    return result


async def test_sandbox_repairs_then_serves_png_live_and_from_checkpoint(
    client, fake_chat_model, python_call, monkeypatch, tmp_path, png_bytes
):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "sandbox_artifacts")
    requests = []

    def execute(request, call_id):
        requests.append(request)
        if call_id == "initial":
            return PythonExecution(
                "",
                "Traceback (most recent call last):\nKeyError: 'QED'\n",
                1,
                SandboxError(kind="code", message="检查列名。"),
            )
        return PythonExecution("sorted\n", "", 0, files={"plot.png": png_bytes})

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[python_call("initial", "data['QED']", ["upload_demo"])],
            ),
            AIMessage(
                content="",
                tool_calls=[python_call("fixed", "data['qed']", ["upload_demo"])],
            ),
            AIMessage(content="已生成图表。"),
        ]
    )
    graph = build_graph(model, [create_run_python_tool(execute)], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    thread = uuid4()
    url = f"/api/threads/{thread}"
    trace = events(await client.post(url + "/runs", json={"message": "画散点图"}))
    calls = [data for kind, data in trace if kind == "tool_call"]
    assert [call["args"]["request"]["code"] for call in calls] == [
        "data['QED']",
        "data['qed']",
    ]
    assert [req.dataset_ids for req in requests] == [["upload_demo"], ["upload_demo"]]
    first, fixed = (data for kind, data in trace if kind == "tool_result")
    assert first["tool_call_id"] == "initial" and first["status"] == "error"
    assert (
        "KeyError: 'QED'" in RunPythonResult.model_validate_json(first["output"]).stderr
    )
    assert model.received[1][-1].status == "error"
    assert "KeyError" in model.received[1][-1].text
    assert fixed["tool_call_id"] == "fixed" and fixed["status"] == "success"
    (file,) = fixed["files"]
    artifact_id = UUID(file["artifact_id"])
    assert json.loads(fixed["output"])["files"] == [
        {
            "filename": "plot.png",
            "size_bytes": len(png_bytes),
            "media_type": "image/png",
        }
    ]
    assert trace[-1] == (
        "done",
        {"thread_id": str(thread), "final_response": "已生成图表。"},
    )
    history = (await client.get(url)).raise_for_status().json()["messages"]
    (saved,) = [
        message for message in history if message.get("tool_call_id") == "fixed"
    ]
    assert saved["files"] == fixed["files"]
    png = await client.get(f"{url}/artifacts/{artifact_id}")
    assert png.raise_for_status().headers["content-type"] == "image/png"
    assert png.content == png_bytes
    assert (
        await client.get(f"/api/threads/{uuid4()}/artifacts/{artifact_id}")
    ).status_code == 404
    assert (await client.get(f"{url}/artifacts/not-a-uuid")).status_code == 422
    (batch,) = list((artifacts.ARTIFACTS_DIR / str(thread)).iterdir())
    assert not batch.name.startswith(".")
    assert [child.name for child in batch.iterdir()] == [f"{artifact_id}.png"]


async def test_sandbox_infrastructure_failure_ends_sse(
    client, fake_chat_model, python_call, monkeypatch
):
    def execute(request, call_id):
        raise ImageNotFound("pharmaflow-sandbox:1")

    model = fake_chat_model(
        responses=[
            AIMessage(content="", tool_calls=[python_call("run", "print(1)")]),
        ]
    )
    graph = build_graph(model, [create_run_python_tool(execute)], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    trace = events(
        await client.post(f"/api/threads/{uuid4()}/runs", json={"message": "执行"})
    )
    assert trace[-1] == ("error", {"detail": "ImageNotFound: pharmaflow-sandbox:1"})
    assert not any(kind == "tool_result" for kind, _ in trace)
    assert len(model.received) == 1


@pytest.mark.parametrize("during_send", [False, True])
async def test_sandbox_sse_disconnect_waits_for_worker_cleanup(
    client, fake_chat_model, python_call, monkeypatch, during_send
):
    loop = asyncio.get_running_loop()
    started, disconnected, sending_tool_node, unlocked = (
        asyncio.Event() for _ in range(4)
    )
    release, cleaned = Event(), Event()

    def execute(request, call_id):
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(5), "test never released the worker"
            return PythonExecution("", "", 0)
        finally:
            cleaned.set()

    async def lock():
        try:
            yield
        finally:
            unlocked.set()

    model = fake_chat_model(
        responses=[
            AIMessage(content="", tool_calls=[python_call("run", "pass")]),
            AIMessage(content="完成"),
        ]
    )
    graph = build_graph(model, [create_run_python_tool(execute)], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    monkeypatch.setitem(app.dependency_overrides, claim_thread, lock)
    url = f"/api/threads/{uuid4()}"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": url + "/runs",
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    first = True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": b'{"message":"run"}'}
        await started.wait()
        if during_send:
            await sending_tool_node.wait()
        disconnected.set()
        return {"type": "http.disconnect"}

    async def send(message):
        if (
            during_send
            and message["type"] == "http.response.body"
            and b'{"node":"tools"}' in message["body"]
        ):
            sending_tool_node.set()
            await asyncio.Future()

    async with asyncio.timeout(5):
        request = asyncio.create_task(app(scope, receive, send))
        try:
            await disconnected.wait()
            assert not (await asyncio.wait({request}, timeout=0.05))[0]
            assert not unlocked.is_set()
        finally:
            release.set()
            await request
    assert cleaned.is_set() and unlocked.is_set()
    state = (await client.get(url)).raise_for_status().json()
    assert state["next_nodes"] == ["tools"]


def test_openapi_describes_sse_conflicts_and_multipart_uploads():
    paths = app.openapi()["paths"]
    responses = paths["/api/threads/{thread_id}/runs"]["post"]["responses"]
    assert "text/event-stream" in responses["200"]["content"]
    assert "409" in responses
    content = paths["/api/datasets/upload"]["post"]["requestBody"]["content"]
    assert content["multipart/form-data"]["schema"]["required"] == ["file"]


async def test_thread_stream_and_checkpoint_history(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    created = await client.post("/api/threads")
    assert created.status_code == 200
    thread = created.json()["thread_id"]
    assert UUID(thread).version == 4
    assert (await client.post("/api/threads")).json()["thread_id"] != thread
    url = f"/api/threads/{thread}"
    assert (await client.get(url)).json() == {
        "thread_id": thread,
        "exists": False,
        "next_nodes": [],
        "messages": [],
        "step_count": 0,
    }

    response = await client.post(
        f"{url}/runs", json={"message": "查询", "dataset_id": "candidates"}
    )
    trace = events(response)
    tokens = ["答", '案：\n"3"']
    final_text = "".join(tokens)
    assert trace == [
        ("node", {"node": "agent"}),
        (
            "tool_call",
            {"tool_name": "lookup", "tool_call_id": "call-1", "args": {"value": 3}},
        ),
        ("node", {"node": "tools"}),
        (
            "tool_result",
            {
                "tool_name": "lookup",
                "tool_call_id": "call-1",
                "output": '结果：\n"3"',
                "status": "success",
            },
        ),
        ("node", {"node": "agent"}),
        *(("token", {"content": token}) for token in tokens),
        ("done", {"thread_id": thread, "final_response": final_text}),
    ]

    state = (await client.get(url)).json()
    assert state["thread_id"] == thread
    assert state["exists"] is True
    assert state["step_count"] == 2
    history = state["messages"]
    assert [m["role"] for m in history] == ["human", "ai", "tool", "ai"]
    assert "查询" in history[0]["content"] and "candidates" in history[0]["content"]
    assert history[1]["tool_calls"] == [
        {"name": "lookup", "id": "call-1", "args": {"value": 3}, "type": "tool_call"}
    ]
    assert history[2] == {
        "role": "tool",
        "name": "lookup",
        "tool_call_id": "call-1",
        "content": '结果：\n"3"',
        "status": "success",
    }
    assert history[-1]["content"] == final_text

    follow_up = events(await client.post(f"{url}/runs", json={"message": "继续"}))
    assert follow_up[-1] == (
        "done",
        {"thread_id": thread, "final_response": final_text},
    )
    resumed = (await client.get(url)).json()
    assert resumed["step_count"] == 1
    assert resumed["messages"][:-2] == history


async def test_exception_ends_stream_with_json_error(client, caplog):
    trace = events(
        await client.post(f"/api/threads/{uuid4()}/runs", json={"message": "引发异常"})
    )
    assert trace == [
        ("node", {"node": "agent"}),
        ("error", {"detail": 'RuntimeError: 模拟错误\n"模型"'}),
    ]
    assert any(record.exc_info is not None for record in caplog.records)


async def test_step_limit_reports_reason_in_done(client):
    thread = str(uuid4())
    trace = events(
        await client.post(
            f"/api/threads/{thread}/runs", json={"message": "查询", "max_steps": 1}
        )
    )
    assert [kind for kind, _ in trace] == ["node", "tool_call", "tool_result", "done"]
    assert trace[-2][1] == {
        "tool_name": "lookup",
        "tool_call_id": "call-1",
        "output": STEP_LIMIT_MESSAGE,
        "status": "error",
    }
    assert trace[-1][1] == {"thread_id": thread, "final_response": STEP_LIMIT_MESSAGE}


@pytest.mark.parametrize(
    "payload",
    [
        {"message": " "},
        {"message": "查询", "max_steps": 0},
        {"message": "查询", "max_steps": 21},
        {"message": "查询", "dataset_id": "../outside"},
    ],
)
async def test_invalid_run_is_rejected_before_streaming(client, payload):
    assert (
        await client.post(f"/api/threads/{uuid4()}/runs", json=payload)
    ).status_code == 422


@pytest.mark.parametrize(
    ("name", "payload", "detail"),
    [
        ("data.txt", b"smiles\nCCO\n", ".csv"),
        ("data.csv", b"smiles\n\xff\n", "UTF-8"),
        ("data.csv", b"smiles\n" + b"C" * 94, "100"),
        ("data.csv", b"smiles\n\n", "至少"),
        ("data.csv", b"smiles\nC\nCO\nC\n", "行数"),
        ("data.csv", b"SMILES\nC\n", "smiles"),
        ("data.csv", b"smiles,smiles\nC,C\n", "duplicate"),
        ("data.csv", b"smiles,name\nC\n", "missing name"),
        ("data.csv", b'smiles\n"C', "unexpected end"),
    ],
)
async def test_invalid_upload_is_never_published(
    client, tmp_path, name, payload, detail
):
    response = await client.post(
        "/api/datasets/upload", files={"file": (name, payload, "text/csv")}
    )
    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


async def test_upload_at_limits_readable_by_chem_mcp(client, tmp_path):
    name = "../分子.csv"
    content = 'name,smiles\r\n"first,\r\nsample",CCO\r\nlast,C'.encode("utf-8-sig")
    content += b" " * (100 - len(content))
    files = {"file": (name, content, "text/csv")}
    response = await client.post("/api/datasets/upload", files=files)
    assert response.status_code == 200
    dataset_id = response.json()["dataset_id"]
    assert response.json() == {
        "dataset_id": dataset_id,
        "filename": name,
        "row_count": 2,
        "columns": ["name", "smiles"],
    }
    assert (tmp_path / f"{dataset_id}.csv").read_bytes() == content
    result = compute_properties(MoleculeInput(dataset_id=dataset_id)).results
    assert [row.mw for row in result] == pytest.approx([46.069, 16.043])
    second = await client.post("/api/datasets/upload", files=files)
    assert second.json()["dataset_id"] != dataset_id
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("during_send", [False, True])
async def test_disconnect_then_failed_and_successful_recovery(
    client, fake_chat_model, monkeypatch, during_send
):
    entered = asyncio.Event()
    stopped = asyncio.Event()
    sending = asyncio.Event()
    attempts = 0

    @tool
    async def slow() -> str:
        """Wait until the HTTP client disconnects on the first attempt."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
        if attempts == 2:
            raise ConnectionError("工具暂不可用")
        return "上轮结果"

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="", tool_calls=[{"id": "old", "name": "slow", "args": {}}]
            ),
            AIMessage(content="旧轮完成"),
            AIMessage(content="新轮完成"),
        ]
    )
    graph = build_graph(model, [slow], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    url = f"/api/threads/{uuid4()}"
    path = url + "/runs"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    first = True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {
                "type": "http.request",
                "body": b'{"message":"first","max_steps":4}',
            }
        await entered.wait()
        if during_send:
            await sending.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if (
            during_send
            and message["type"] == "http.response.body"
            and b'{"node":"tools"}' in message["body"]
        ):
            sending.set()
            await asyncio.Future()

    async with asyncio.timeout(5):
        await app(scope, receive, send)
    assert stopped.is_set()
    state = (await client.get(url)).raise_for_status().json()
    assert state["next_nodes"] == ["tools"]
    assert [m["role"] for m in state["messages"]] == ["human", "ai"]
    failure = events(await client.post(path, json={"message": "不得写入"}))
    assert failure == [
        ("node", {"node": "tools", "recovering": True}),
        ("error", {"detail": "ConnectionError: 工具暂不可用"}),
    ]
    assert (await client.get(url)).json() == state
    trace = events(await client.post(path, json={"message": "新问题", "max_steps": 1}))
    assert [data["final_response"] for kind, data in trace if kind == "done"] == [
        "新轮完成"
    ]
    assert all(
        "旧轮完成" not in data["content"] for kind, data in trace if kind == "token"
    )
    state = (await client.get(url)).json()
    assert state["next_nodes"] == []
    assert state["step_count"] == 1
    assert [m["role"] for m in state["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "human",
        "ai",
    ]
    assert [m["content"] for m in state["messages"]][2:] == [
        "上轮结果",
        "旧轮完成",
        "新问题",
        "新轮完成",
    ]
    assert isinstance(model.received[-1][-3], ToolMessage)
    assert model.received[-1][-3].tool_call_id == "old"


@pytest.mark.parametrize("max_steps", [1, 2])
async def test_mixed_tool_failures_emit_one_result_per_call(
    client, fake_chat_model, monkeypatch, max_steps
):
    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "valid", "name": "lookup", "args": {"value": 3}},
                    {"id": "unknown", "name": "missing", "args": {}},
                ],
                invalid_tool_calls=[
                    {
                        "id": "invalid",
                        "name": "lookup",
                        "args": "broken",
                        "error": "JSON error",
                    }
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    graph = build_graph(model, [lookup], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    trace = events(
        await client.post(
            f"/api/threads/{uuid4()}/runs",
            json={"message": "查询", "max_steps": max_steps},
        )
    )
    calls = [data for kind, data in trace if kind == "tool_call"]
    assert [call["tool_call_id"] for call in calls] == ["valid", "unknown", "invalid"]
    assert calls[-1] == {
        "tool_name": "lookup",
        "tool_call_id": "invalid",
        "raw_args": "broken",
    }
    results = [data for kind, data in trace if kind == "tool_result"]
    assert sorted((r["tool_call_id"], r["status"]) for r in results) == [
        ("invalid", "error"),
        ("unknown", "error"),
        ("valid", "success" if max_steps == 2 else "error"),
    ]
    assert trace[-1][0] == "done"


async def test_chunked_upload_stops_at_file_limit(client):
    async def chunks():
        yield b'--bound\r\nContent-Disposition: form-data; name="file"; filename="test.csv"\r\n\r\n'
        yield b"x" * 60
        yield b"x" * 60
        pytest.fail("超限后仍继续读取请求体")

    response = await client.post(
        "/api/datasets/upload",
        content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=bound"},
    )
    assert response.status_code == 422
    assert "100" in response.json()["detail"]


async def test_parallel_tool_result_arrives_while_another_tool_is_running(
    fake_chat_model,
):
    release = asyncio.Event()

    @tool
    async def slower() -> str:
        """Wait for the consumer to receive the other tool's result."""
        await release.wait()
        return "慢"

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "slow", "name": "slower", "args": {}},
                    {"id": "fast", "name": "lookup", "args": {"value": 3}},
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    graph = build_graph(model, [lookup, slower], InMemorySaver())
    ids = []
    async with asyncio.timeout(5):
        async for frame in stream_run(
            graph,
            uuid4(),
            RunRequest(message="并行查询", max_steps=20),
            trace_for_thread=None,
        ):
            if frame.startswith("event: tool_result\n"):
                data = json.loads(frame.splitlines()[1].removeprefix("data: "))
                ids.append(data["tool_call_id"])
                release.set()
    assert ids == ["fast", "slow"]


async def test_stream_only_model_resumes_without_mixing_answer_tokens(
    client, monkeypatch
):
    available = False

    @tool("lookup")
    def flaky_lookup(value: int) -> str:
        """Query a tool that becomes available between requests."""
        if not available:
            raise ConnectionError("未连接")
        return str(value)

    graph = build_graph(StreamingModel(), [flaky_lookup], InMemorySaver())
    monkeypatch.setitem(app.dependency_overrides, get_agent_graph, lambda: graph)
    metadata_seen = []

    class TraceRecorder(BaseCallbackHandler):
        def on_chain_start(
            self, serialized, inputs, *, metadata, parent_run_id, **kwargs
        ):
            if parent_run_id is None:
                metadata_seen.append(metadata)

    def trace_for_thread(thread_id):
        return trace_config([TraceRecorder()], thread_id=thread_id, tags=["api-test"])

    monkeypatch.setitem(
        app.dependency_overrides, get_turn_tracer, lambda: trace_for_thread
    )
    url = f"/api/threads/{uuid4()}"
    first = events(await client.post(url + "/runs", json={"message": "查询"}))
    assert first[-1] == ("error", {"detail": "ConnectionError: 未连接"})
    available = True
    trace = events(
        await client.post(url + "/runs", json={"message": "新查询", "max_steps": 1})
    )
    assert trace[:2] == [
        ("node", {"node": "tools", "recovering": True}),
        ("node", {"node": "agent", "recovering": True}),
    ]
    assert all(kind != "token" for kind, _ in trace)
    assert trace[-1][1]["final_response"] == STEP_LIMIT_MESSAGE
    history = (await client.get(url)).json()["messages"]
    assert history[3]["content"] == '答案：\n"3"'
    assert history[4]["content"] == "新查询"
    assert len(metadata_seen) == 3  # First attempt, recovery, then the new turn.
    assert {metadata["langfuse_session_id"] for metadata in metadata_seen} == {
        url.rsplit("/", 1)[-1]
    }
    assert all(metadata["langfuse_tags"] == ["api-test"] for metadata in metadata_seen)
