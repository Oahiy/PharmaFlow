"""Validate the default Langfuse client and configure LangChain callbacks."""

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
type CallbackFactory = Callable[[str], BaseCallbackHandler]


@asynccontextmanager
async def open_langfuse(*, required: bool) -> AsyncIterator[Langfuse | None]:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().casefold() == "false":
        if required:
            raise RuntimeError(
                "正式评测需要启用 Langfuse。LANGFUSE_TRACING_ENABLED=false。"
            )
        yield None
        return
    missing = [name for name in LANGFUSE_KEYS if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"缺少环境变量 {', '.join(missing)}。")

    client = get_client()
    if not await asyncio.to_thread(client.auth_check):
        raise RuntimeError("Langfuse auth_check() 返回 False。")
    try:
        yield client
    finally:
        await asyncio.to_thread(client.flush)


def langfuse_callback(trace_id: str) -> BaseCallbackHandler:
    return CallbackHandler(trace_context={"trace_id": trace_id})


def trace_config(
    callbacks: list[BaseCallbackHandler],
    *,
    thread_id: UUID,
    tags: list[str],
    user_id: UUID | None = None,
) -> RunnableConfig:
    metadata = {"langfuse_session_id": str(thread_id), "langfuse_tags": tags}
    if user_id is not None:
        metadata["langfuse_user_id"] = str(user_id)
    return {"callbacks": callbacks, "metadata": metadata}


def langfuse_turn_trace(thread_id: UUID, *, tags: list[str]) -> RunnableConfig:
    return trace_config(
        [langfuse_callback(Langfuse.create_trace_id())],
        thread_id=thread_id,
        tags=tags,
    )
