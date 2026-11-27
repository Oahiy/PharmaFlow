"""Translate a single graph run's v2 callback stream to SSE frames."""

import json
import logging
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import aclosing
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from starlette.types import Receive, Scope, Send

from sandbox.tool import sandbox_worker_scope

from .schemas import (
    Done,
    InvalidToolCallEvent,
    Node,
    RunError,
    RunRequest,
    Token,
    ToolCallEvent,
    ToolResult,
    tool_artifacts,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from agent.state import TurnInput

logger = logging.getLogger(__name__)
type TurnTracer = Callable[[UUID], RunnableConfig]


class RunResponse(StreamingResponse):
    media_type = "text/event-stream"

    def __init__(self, frames: AsyncGenerator[str, None]):
        self.frames = frames
        super().__init__(
            frames, headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # A disconnect during send() leaves the generator suspended at yield.
        # Stop the graph before the request-scoped database lock is released.
        async with sandbox_worker_scope(), aclosing(self.frames):
            await super().__call__(scope, receive, send)


def is_graph_node(event: StreamEvent, subgraphs: set[str]) -> bool:
    # Exclude child chains, including the internal ToolNode also named "tools".
    parents = event["parent_ids"]
    return (
        len(parents) == 1 or (bool(parents) and parents[-1] in subgraphs)
    ) and event.get("metadata", {}).get("langgraph_node") == event["name"]


def node_tool_results(output: dict | list, key: str) -> Iterator[ToolMessage]:
    from langgraph.types import Command

    # ToolNode returns a list of state updates when a tool returns a Command.
    for update in output if isinstance(output, list) else [output]:
        values = update.update if isinstance(update, Command) else update
        if isinstance(values, dict) and key in values:
            yield from (m for m in values[key] if isinstance(m, ToolMessage))


def tool_result(
    message: ToolMessage, name: str | None, subtask_id: str | None
) -> ToolResult:
    content = message.content
    return ToolResult(
        subtask_id=subtask_id,
        tool_name=name,
        tool_call_id=message.tool_call_id,
        status=message.status,
        files=tool_artifacts(message),
        output=content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False),
    )


def tool_calls(reply: AIMessage, subtask_id: str | None) -> Iterator[str]:
    for call in reply.tool_calls:
        yield ToolCallEvent(
            subtask_id=subtask_id,
            tool_name=call["name"],
            tool_call_id=call["id"],
            args=call["args"],
        ).encode()
    for invalid in reply.invalid_tool_calls:
        yield InvalidToolCallEvent(
            subtask_id=subtask_id,
            tool_name=invalid["name"],
            tool_call_id=invalid["id"],
            raw_args=invalid["args"],
        ).encode()


def node_frame(
    event: StreamEvent, subgraphs: set[str], *, recovering: bool
) -> str | None:
    from agent.multi import WORKER_NODES

    name = event["name"]
    if name in {f"{node}ReAct" for node in WORKER_NODES.values()}:
        subgraphs.add(event["run_id"])
    if not is_graph_node(event, subgraphs):
        return None
    subtask_id = event["metadata"].get("subtask_id")
    if name in WORKER_NODES.values():
        subtask_id = event["data"]["input"]["subtask"].id
    details = {"recovering": True} if recovering else {}
    return Node(node=name, subtask_id=subtask_id, **details).encode()


async def stream_run(
    graph: "CompiledStateGraph",
    thread_id: UUID,
    request: RunRequest,
    *,
    trace_for_thread: TurnTracer | None,
) -> AsyncGenerator[str, None]:
    from agent.graph import is_multi_graph, resume_config, turn_recursion_limit

    multi = is_multi_graph(graph)
    message = request.message
    if request.dataset_id is not None:
        message += f"\n\n本轮候选分子数据集 dataset_id：{request.dataset_id}"
    turn: TurnInput = {
        "messages": [HumanMessage(content=message)],
        "step_count": 0,
        "max_steps": request.max_steps,
    }
    thread_config: RunnableConfig = {"configurable": {"thread_id": str(thread_id)}}
    config: RunnableConfig = {
        **thread_config,
        "recursion_limit": turn_recursion_limit(graph, request.max_steps),
    }
    try:
        snapshot = await graph.aget_state(thread_config)
        if snapshot.next:
            # The recovered reply belongs to the previous user message.
            recovery_config = await resume_config(graph, thread_config, snapshot)
            if trace_for_thread is not None:
                recovery_config.update(trace_for_thread(thread_id))
            recovery = graph.astream_events(
                None,
                recovery_config,
                version="v2",
                include_types=["chain"],
            )
            recovery_subgraphs: set[str] = set()
            async with aclosing(recovery):
                async for event in recovery:
                    if event["event"] == "on_chain_start":
                        frame = node_frame(event, recovery_subgraphs, recovering=True)
                        if frame is not None:
                            yield frame
        emitted_results: dict[tuple[str | None, int], set[str]] = defaultdict(set)
        subgraphs: set[str] = set()
        if trace_for_thread is not None:
            config.update(trace_for_thread(thread_id))
        events = graph.astream_events(turn, config, version="v2", stream_mode="values")
        async with aclosing(events):
            async for event in events:
                kind, name, data = event["event"], event["name"], event["data"]
                metadata = event.get("metadata", {})
                model_role = metadata.get("pharmaflow_model_role")
                subtask_id = metadata.get("subtask_id")
                scope = (subtask_id, metadata.get("subtask_attempt", 0))
                if kind == "on_chain_start":
                    frame = node_frame(event, subgraphs, recovering=False)
                    if frame is not None:
                        yield frame
                elif kind == "on_chat_model_stream":
                    if model_role == ("reporter" if multi else "agent"):
                        content = data["chunk"].text
                        if content:
                            yield Token(content=content).encode()
                elif (
                    kind == "on_chat_model_end" and not multi and model_role == "agent"
                ):
                    emitted_results[scope].clear()
                    for frame in tool_calls(data["output"], subtask_id):
                        yield frame
                elif kind == "on_tool_end" and isinstance(data["output"], ToolMessage):
                    result = data["output"]
                    if result.tool_call_id not in emitted_results[scope]:
                        emitted_results[scope].add(result.tool_call_id)
                        yield tool_result(result, name, subtask_id).encode()
                elif kind == "on_chain_end" and is_graph_node(event, subgraphs):
                    key = "worker_messages" if multi else "messages"
                    if multi and name == "react":
                        emitted_results[scope].clear()
                        for frame in tool_calls(data["output"][key][0], subtask_id):
                            yield frame
                    for result in node_tool_results(data["output"], key):
                        if result.tool_call_id not in emitted_results[scope]:
                            emitted_results[scope].add(result.tool_call_id)
                            yield tool_result(result, result.name, subtask_id).encode()
                elif kind == "on_chain_end" and not event["parent_ids"]:
                    last_message = data["output"]["messages"][-1]
                    yield Done(
                        thread_id=thread_id, final_response=last_message.text
                    ).encode()
    except Exception as exc:
        logger.exception("会话 %s 运行失败", thread_id)
        yield RunError(detail=f"{type(exc).__name__}: {exc}").encode()
