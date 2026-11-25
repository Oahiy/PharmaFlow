import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage
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
