"""Streamlit client for the PharmaFlow HTTP API."""

import json
import os
from uuid import UUID

import httpx
import streamlit as st
from httpx_sse import SSEError, connect_sse

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
RUN_PYTHON = "run_python"


def error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 409:
            return exc.response.json()["detail"]
        return f"{exc.response.status_code} · {exc.response.text}"
    return str(exc)


def upload_changed() -> None:
    st.session_state.dataset = None
    st.session_state.upload_error = None
    st.session_state.upload_pending = True


def sidebar(client: httpx.Client) -> None:
    with st.sidebar:
        thread = st.query_params.get("thread_id")
        if st.button("新建会话") or thread is None:
            response = client.post("/api/threads")
            response.raise_for_status()
            thread = response.json()["thread_id"]
            st.query_params["thread_id"] = thread
        try:
            thread = str(UUID(thread))
        except ValueError as exc:
            st.error(f"会话 ID 无效：{exc}")
            st.stop()
        if thread != st.session_state.get("thread_id"):
            st.session_state.thread_id = thread
            st.session_state.dataset = None
            st.session_state.upload_error = None
            st.session_state.upload_pending = False
        st.caption("当前会话")
        st.code(st.session_state.thread_id, language=None)
        file = st.file_uploader(
            "上传候选分子 CSV",
            type="csv",
            help="表头须包含小写 smiles 列。",
            key=f"csv_{st.session_state.thread_id}",
            on_change=upload_changed,
        )
        if st.session_state.upload_pending:
            st.session_state.upload_pending = False
            if file is not None:
                try:
                    response = client.post(
                        "/api/datasets/upload",
                        files={"file": (file.name, file.getvalue(), "text/csv")},
                    )
                    response.raise_for_status()
                    st.session_state.dataset = response.json()
                except httpx.HTTPError as exc:
                    st.session_state.upload_error = error_detail(exc)
        if st.session_state.upload_error is not None:
            st.error(st.session_state.upload_error)
        dataset = st.session_state.dataset
        if dataset is not None:
            st.success(f"已上传 {dataset['row_count']} 行")
            st.code(dataset["dataset_id"], language=None)
            st.caption("列名：" + "、".join(dataset["columns"]))


def show_python_code(panel, args: dict) -> None:
    match args:
        case {"request": {"code": str(code), **inputs}}:
            if inputs:
                panel.json(inputs)
            panel.code(code, language="python")
        case _:
            panel.json(args)


def show_python_result(
    client: httpx.Client, details, charts, output: str, files: list[dict] | None
) -> None:
    if files is None:
        details.error(output)
        return
    result = json.loads(output)
    if result["error"] is not None:
        details.error(result["error"]["message"])
    for stream in ("stdout", "stderr"):
        if result[stream]:
            details.caption(stream)
            details.code(result[stream], language=None)
    with charts:
        for file in files:
            try:
                response = client.get(
                    f"/api/threads/{st.session_state.thread_id}/artifacts/{file['artifact_id']}"
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(error_detail(exc))
                continue
            st.image(response.content, caption=file["filename"])


def show_run(client: httpx.Client, message: str) -> str | None:
    activity = st.container()
    answer = st.empty()
    charts = st.container()
    text = ""
    dataset = st.session_state.dataset
    payload = {
        "message": message,
        "dataset_id": dataset["dataset_id"] if dataset else None,
    }
    with activity.status("正在执行...", expanded=True) as status:
        try:
            with connect_sse(
                client,
                "POST",
                f"/api/threads/{st.session_state.thread_id}/runs",
                json=payload,
                timeout=httpx.Timeout(10, read=None),
            ) as source:
                if not source.response.is_success:
                    source.response.read()
                    source.response.raise_for_status()
                for event in source.iter_sse():
                    data = event.json()
                    subtask = f" · {data['subtask_id']}" if "subtask_id" in data else ""
                    match event.event:
                        case "node":
                            phase = (
                                "恢复上一轮" if data.get("recovering") else "正在执行"
                            )
                            status.update(label=f"{phase} · {data['node']}{subtask}")
                            status.write(f"{phase} · {data['node']}{subtask}")
                        case "tool_call":
                            status.write(f"调用工具 · {data['tool_name']}{subtask}")
                            if "raw_args" in data:
                                status.text(f"无法解析的原始参数：{data['raw_args']!r}")
                            elif data["tool_name"] == RUN_PYTHON:
                                show_python_code(status, data["args"])
                            else:
                                status.json(data["args"])
                        case "tool_result":
                            output = data["output"]
                            status.write(
                                f"{data['tool_name']} · {data['status']}{subtask}"
                            )
                            if data["tool_name"] == RUN_PYTHON:
                                show_python_result(
                                    client, status, charts, output, data.get("files")
                                )
                            else:
                                summary = (
                                    output[:300] + "…" if len(output) > 300 else output
                                )
                                status.code(summary, language=None)
                        case "token":
                            text += data["content"]
                            answer.markdown(text)
                        case "done":
                            answer.markdown(data["final_response"])
                            status.update(
                                label="执行完成", state="complete", expanded=False
                            )
                            return data["final_response"]
                        case "error":
                            status.error(data["detail"])
                            status.update(
                                label="执行失败", state="error", expanded=True
                            )
                            return None
                raise SSEError("连接在收到结束事件前已关闭")
        except httpx.HTTPError as exc:
            status.error(error_detail(exc))
            status.update(label="请求失败", state="error", expanded=True)
            return None


def render_history(client: httpx.Client, messages: list[dict], pending: bool) -> None:
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "human":
            st.chat_message("user").write(message["content"])
        elif role == "ai":
            if not (message["tool_calls"] or message["invalid_tool_calls"]):
                st.chat_message("assistant").write(message["content"])
            for call in message["tool_calls"]:
                match call:
                    case {"name": "run_python", "args": args}:
                        with st.chat_message("assistant"):
                            show_python_code(st.expander("沙盒代码"), args)
        elif role == "tool":
            if message.get("name") == RUN_PYTHON:
                with st.chat_message("assistant"):
                    failed = message["status"] == "error"
                    label = "沙盒执行失败" if failed else "沙盒执行完成"
                    show_python_result(
                        client,
                        st.expander(label, expanded=failed),
                        st.container(),
                        message["content"],
                        message.get("files"),
                    )
                continue
            # A capped turn ends with the graph's explanatory ToolMessage.
            last_in_turn = (
                not pending
                if index + 1 == len(messages)
                else messages[index + 1]["role"] == "human"
            )
            if last_in_turn:
                st.chat_message("assistant").write(message["content"])


def main() -> None:
    st.set_page_config(page_title="PharmaFlow Agent")
    st.title("PharmaFlow Agent")
    with httpx.Client(base_url=API_BASE_URL) as client:
        try:
            sidebar(client)
            state = client.get(f"/api/threads/{st.session_state.thread_id}")
            state.raise_for_status()
            thread = state.json()
        except httpx.HTTPError as exc:
            st.sidebar.error(error_detail(exc))
            st.stop()
        pending = bool(thread["next_nodes"])
        incomplete = thread.get("incomplete_turn", False)
        render_history(client, thread["messages"], pending or incomplete)
        if pending:
            st.info("上一轮尚未结束，待执行节点：" + "、".join(thread["next_nodes"]))
        elif incomplete:
            st.info("上一轮未生成答复。")
        if message := st.chat_input("输入问题"):
            st.chat_message("user").markdown(message)
            with st.chat_message("assistant"):
                reply = show_run(client, message)
            if pending and reply is not None:
                st.rerun()  # Display the recovered reply in its original turn.


if __name__ == "__main__":
    main()
