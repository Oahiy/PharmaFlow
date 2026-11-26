"""Public request, checkpoint history, upload and SSE payloads."""

from typing import ClassVar, Literal
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from pydantic import BaseModel, Field

from mcp_servers.chem.models import DatasetId


class RunRequest(BaseModel):
    message: str = Field(min_length=1, pattern=r"\S")
    max_steps: int = Field(default=10, ge=1, le=20)
    dataset_id: DatasetId | None = None


class ThreadCreated(BaseModel):
    thread_id: UUID


class HistoryMessage(BaseModel):
    content: str | list[str | dict]


class HumanHistory(HistoryMessage):
    role: Literal["human"] = "human"


class AIHistory(HistoryMessage):
    role: Literal["ai"] = "ai"
    tool_calls: list[ToolCall]
    invalid_tool_calls: list[InvalidToolCall]


class ToolHistory(HistoryMessage):
    role: Literal["tool"] = "tool"
    name: str | None
    tool_call_id: str
    status: Literal["success", "error"]


def serialize_message(message: BaseMessage) -> HumanHistory | AIHistory | ToolHistory:
    if isinstance(message, HumanMessage):
        return HumanHistory(content=message.content)
    if isinstance(message, AIMessage):
        return AIHistory(
            content=message.content,
            tool_calls=message.tool_calls,
            invalid_tool_calls=message.invalid_tool_calls,
        )
    if isinstance(message, ToolMessage):
        return ToolHistory(
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
            status=message.status,
        )
    raise ValueError(f"未知会话消息类型：{message.type}")


class ThreadState(ThreadCreated):
    exists: bool
    next_nodes: tuple[str, ...]
    step_count: int
    messages: list[HumanHistory | AIHistory | ToolHistory]


class UploadedDataset(BaseModel):
    dataset_id: DatasetId
    filename: str
    row_count: int
    columns: list[str]


class SSEPayload(BaseModel):
    event: ClassVar[str]

    def encode(self) -> str:
        return (
            f"event: {self.event}\ndata: {self.model_dump_json(exclude_unset=True)}\n\n"
        )


class Node(SSEPayload):
    event = "node"
    node: str
    recovering: bool = False


class ToolCallEvent(SSEPayload):
    event = "tool_call"
    tool_name: str
    tool_call_id: str
    args: dict


class InvalidToolCallEvent(SSEPayload):
    event = "tool_call"
    tool_name: str | None
    tool_call_id: str
    raw_args: str | None


class ToolResult(SSEPayload):
    event = "tool_result"
    tool_name: str | None
    tool_call_id: str
    output: str
    status: Literal["success", "error"]


class Token(SSEPayload):
    event = "token"
    content: str


class RunError(SSEPayload):
    event = "error"
    detail: str


class Done(SSEPayload):
    event = "done"
    thread_id: UUID
    final_response: str
