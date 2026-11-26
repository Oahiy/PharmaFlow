import json
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "app.py"


def test_refresh_restores_history_and_new_thread_replaces_url(monkeypatch):
    original, new = str(uuid4()), str(uuid4())
    calls = []

    def api(request):
        path = request.url.path
        calls.append((request.method, path))
        if path == "/api/threads":
            return httpx.Response(200, json={"thread_id": new})
        if path == f"/api/threads/{original}/runs":
            return httpx.Response(409, json={"detail": "此会话正在运行"})
        messages = []
        if path == f"/api/threads/{original}":
            messages = [
                {"role": "human", "content": "旧问题"},
                {
                    "role": "ai",
                    "content": "",
                    "tool_calls": [{"id": "1"}],
                    "invalid_tool_calls": [],
                },
                {"role": "tool", "tool_call_id": "1", "content": "工具结果"},
                {
                    "role": "ai",
                    "content": "旧回答",
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                },
            ]
        return httpx.Response(200, json={"messages": messages, "next_nodes": []})

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(api))
    )
    # A fresh Streamlit session has only the browser URL to identify the thread.
    ui = AppTest.from_file(str(APP))
    ui.query_params["thread_id"] = original
    ui.run()
    assert not ui.exception
    assert calls == [("GET", f"/api/threads/{original}")]
    assert [m.markdown[0].value for m in ui.chat_message] == ["旧问题", "旧回答"]
    ui.chat_input[0].set_value("新问题").run()
    assert not ui.exception
    assert ui.status[0].state == "error"
    assert ui.error[0].value == "此会话正在运行"
    ui.sidebar.button[0].click().run()
    assert not ui.exception
    assert ui.query_params["thread_id"] == new
    assert calls[-2:] == [("POST", "/api/threads"), ("GET", f"/api/threads/{new}")]
    assert not ui.chat_message


def test_refresh_shows_python_code_failure_and_chart(
    monkeypatch, png_bytes, python_call
):
    thread, artifact_id = str(uuid4()), str(uuid4())
    url = f"/api/threads/{thread}"
    failure_code, fixed_code = "values = data['QED']", "values = data['qed']"
    chart = {
        "artifact_id": artifact_id,
        "filename": "scatter.png",
        "size_bytes": len(png_bytes),
        "media_type": "image/png",
    }
    failure = {
        "stdout": "",
        "stderr": "KeyError: 'QED'\n",
        "exit_code": 1,
        "files": [],
        "error": {"kind": "code", "message": "检查列名。"},
    }
    success = {
        "stdout": "已排序\n",
        "stderr": "",
        "exit_code": 0,
        "error": None,
        "files": [{key: value for key, value in chart.items() if key != "artifact_id"}],
    }
    messages = [
        {"role": "human", "content": "画散点图"},
        {
            "role": "ai",
            "content": "",
            "invalid_tool_calls": [],
            "tool_calls": [python_call("failed", failure_code)],
        },
        {
            "role": "tool",
            "name": "run_python",
            "tool_call_id": "failed",
            "status": "error",
            "content": json.dumps(failure),
            "files": [],
        },
        {
            "role": "ai",
            "content": "",
            "invalid_tool_calls": [],
            "tool_calls": [python_call("fixed", fixed_code)],
        },
        {
            "role": "tool",
            "name": "run_python",
            "tool_call_id": "fixed",
            "status": "success",
            "content": json.dumps(success),
            "files": [chart],
        },
        {
            "role": "ai",
            "content": "已完成。",
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    ]
    paths = []

    def api(request):
        path = request.url.path
        paths.append(path)
        if path == url:
            return httpx.Response(200, json={"messages": messages, "next_nodes": []})
        assert path == f"{url}/artifacts/{artifact_id}"
        return httpx.Response(
            200, content=png_bytes, headers={"content-type": "image/png"}
        )

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(api))
    )
    ui = AppTest.from_file(str(APP))
    ui.query_params["thread_id"] = thread
    ui.run()
    assert not ui.exception
    assert paths == [url, f"{url}/artifacts/{artifact_id}"]
    shown_code = [code.value for code in ui.code]
    assert shown_code.index(failure_code) < shown_code.index(fixed_code)
    assert "KeyError: 'QED'" in shown_code
    assert [element.value for element in ui.error] == ["检查列名。"]
    (image,) = ui.image
    assert image.captions == ["scatter.png"]
    assert image.value[0]
