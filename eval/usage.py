"""Collect Agent input usage once per LangChain chat-model call."""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class AgentUsage(BaseCallbackHandler):
    run_inline = True
    raise_error = True

    def __init__(self) -> None:
        self._input_tokens: dict[UUID, int | None] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # An errored call has no on_llm_end event and keeps its missing usage.
        self._input_tokens[run_id] = None

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        assert isinstance(generation, ChatGeneration)
        message = generation.message
        assert isinstance(message, AIMessage)
        usage = message.usage_metadata
        self._input_tokens[run_id] = (
            usage["input_tokens"] if usage is not None else None
        )

    @property
    def input_tokens(self) -> int | None:
        if not self._input_tokens:
            return None
        total = 0
        for value in self._input_tokens.values():
            if value is None:
                return None
            total += value
        return total
