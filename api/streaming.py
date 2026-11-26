"""Translate a single graph run's v2 callback stream to SSE frames."""

import json
import logging
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent
from starlette.types import Receive, Scope, Send

from .schemas import (
    Done,
    InvalidToolCallEvent,
    Node,
    RunError,
    RunRequest,
    Token,
    ToolCallEvent,
    ToolResult,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from agent.state import AgentState

logger = logging.getLogger(__name__)


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
        async with aclosing(self.frames):
            await super().__call__(scope, receive, send)


def is_graph_node(event: StreamEvent) -> bool:
    # Exclude child chains, including the internal ToolNode also named "tools".
    return (
        len(event["parent_ids"]) == 1
        and event.get("metadata", {}).get("langgraph_node") == event["name"]
    )


def node_tool_results(output: dict | list) -> Iterator[ToolMessage]:
    from langgraph.types import Command

    # ToolNode returns a list of state updates when a tool returns a Command.
    for update in output if isinstance(output, list) else [output]:
        values = update.update if isinstance(update, Command) else update
        if isinstance(values, dict) and "messages" in values:
            yield from (m for m in values["messages"] if isinstance(m, ToolMessage))


def tool_result(message: ToolMessage, name: str | None) -> ToolResult:
    content = message.content
    return ToolResult(
        tool_name=name,
        tool_call_id=message.tool_call_id,
        status=message.status,
        output=content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False),
    )


async def stream_run(
    graph: "CompiledStateGraph", thread_id: UUID, request: RunRequest
) -> AsyncGenerator[str, None]:
    from agent.graph import resume_config

    message = request.message
    if request.dataset_id is not None:
        message += f"\n\n本轮候选分子数据集 dataset_id：{request.dataset_id}"
    turn: AgentState = {
        "messages": [HumanMessage(content=message)],
        "step_count": 0,
        "max_steps": request.max_steps,
    }
    thread_config: RunnableConfig = {"configurable": {"thread_id": str(thread_id)}}
    config: RunnableConfig = {
        **thread_config,
        "recursion_limit": 2 * request.max_steps + 1,
    }
    try:
        snapshot = await graph.aget_state(thread_config)
        if snapshot.next:
            # The recovered reply belongs to the previous user message.
            recovery = graph.astream_events(
                None,
                resume_config(thread_config, snapshot),
                version="v2",
                include_types=["chain"],
            )
            async with aclosing(recovery):
                async for event in recovery:
                    if event["event"] == "on_chain_start" and is_graph_node(event):
                        yield Node(node=event["name"], recovering=True).encode()
        emitted_results: set[str] = set()
        events = graph.astream_events(turn, config, version="v2", stream_mode="values")
        async with aclosing(events):
            async for event in events:
                kind, name, data = event["event"], event["name"], event["data"]
                if kind == "on_chain_start" and is_graph_node(event):
                    yield Node(node=name).encode()
                elif kind == "on_chat_model_stream":
                    content = data["chunk"].content
                    if isinstance(content, str) and content:
                        yield Token(content=content).encode()
                elif kind == "on_chat_model_end":
                    emitted_results.clear()
                    for call in data["output"].tool_calls:
                        yield ToolCallEvent(
                            tool_name=call["name"],
                            tool_call_id=call["id"],
                            args=call["args"],
                        ).encode()
                    for invalid in data["output"].invalid_tool_calls:
                        yield InvalidToolCallEvent(
                            tool_name=invalid["name"],
                            tool_call_id=invalid["id"],
                            raw_args=invalid["args"],
                        ).encode()
                elif kind == "on_tool_end" and isinstance(data["output"], ToolMessage):
                    result = data["output"]
                    emitted_results.add(result.tool_call_id)
                    yield tool_result(result, name).encode()
                elif kind == "on_chain_end" and is_graph_node(event):
                    for result in node_tool_results(data["output"]):
                        if result.tool_call_id not in emitted_results:
                            emitted_results.add(result.tool_call_id)
                            yield tool_result(result, result.name).encode()
                elif kind == "on_chain_end" and not event["parent_ids"]:
                    last_message = data["output"]["messages"][-1]
                    yield Done(
                        thread_id=thread_id, final_response=last_message.text
                    ).encode()
    except Exception as exc:
        logger.exception("会话 %s 运行失败", thread_id)
        yield RunError(detail=f"{type(exc).__name__}: {exc}").encode()
