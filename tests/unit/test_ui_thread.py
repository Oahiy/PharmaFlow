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
