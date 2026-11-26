import os
from io import BytesIO

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage
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
