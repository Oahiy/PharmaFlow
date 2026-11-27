import os
from collections.abc import Iterator
from io import BytesIO

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from PIL import Image
from pydantic import Field

os.environ["FASTMCP_ENV_FILE"] = os.devnull
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"


class FakeChatModel(FakeMessagesListChatModel):
    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.fixture
def fake_chat_model():
    return FakeChatModel


class Task8Model(BaseChatModel):
    replies: dict[str, list[AIMessage | None]]
    received: dict[str, list[tuple[list[BaseMessage], tuple[str, ...]]]] = Field(
        default_factory=dict
    )
    disable_streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "task-8-fake"

    def bind_tools(self, tools, **kwargs):
        names = tuple(
            tool["function"]["name"] if isinstance(tool, dict) else tool.name
            for tool in tools
        )
        return self.bind(tools=names)

    def _reply(self, messages, tools):
        text = messages[0].text
        role = next(
            (
                role
                for role in ("Planner", "LitResearcher", "MolAnalyst", "Reporter")
                if text.startswith(f"你是 PharmaFlow {role}")
            ),
            None,
        )
        if role is None:
            role = "summary" if text.startswith("用中文简要概括") else "single"
        calls = self.received.setdefault(role, [])
        reply = self.replies[role][len(calls)]
        calls.append((messages, tools))
        if reply is None:
            raise RuntimeError(f"{role} 中断")
        return reply.model_copy(deep=True)

    def _generate(self, messages, stop=None, run_manager=None, *, tools=(), **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=self._reply(messages, tools))]
        )

    def _stream(
        self, messages, stop=None, run_manager=None, *, tools=(), **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._reply(messages, tools)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=reply.content,
                tool_calls=reply.tool_calls,
                invalid_tool_calls=reply.invalid_tool_calls,
            )
        )


@pytest.fixture
def task8_model():
    return Task8Model


@pytest.fixture
def plan_reply():
    def reply(*tasks):
        return AIMessage(
            content="规划原始回复",
            tool_calls=[
                {
                    "id": "plan",
                    "name": "Plan",
                    "args": {"tasks": list(tasks)},
                }
            ],
        )

    return reply


@pytest.fixture
def mcp_tool():
    from agent.mcp_client import MCP_SERVER

    def for_server(server, **kwargs):
        def decorate(function):
            result = tool(**kwargs)(function)
            result.metadata = {MCP_SERVER: server}
            return result

        return decorate

    return for_server


@pytest.fixture
def png_bytes():
    with BytesIO() as output, Image.new("RGB", (2, 2), "white") as image:
        image.save(output, format="PNG")
        return output.getvalue()


@pytest.fixture
def python_call():
    def call(tool_call_id, code, dataset_ids=()):
        return {
            "name": "run_python",
            "id": tool_call_id,
            "args": {"request": {"code": code, "dataset_ids": list(dataset_ids)}},
        }

    return call
