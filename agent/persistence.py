"""Open a shared Postgres pool and initialize the LangGraph checkpointer."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde import _msgpack
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


@asynccontextmanager
async def open_connection_pool(
    database_url: str,
) -> AsyncIterator[AsyncConnectionPool[AsyncConnection[DictRow]]]:
    async with AsyncConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        min_size=1,
        open=False,
    ) as pool:
        yield pool


async def setup_checkpointer(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
) -> AsyncPostgresSaver:
    if not _msgpack.STRICT_MSGPACK_ENABLED:
        raise RuntimeError(
            "LangGraph 严格反序列化未启用。请在导入 LangGraph 之前设置 "
            "LANGGRAPH_STRICT_MSGPACK=true。"
        )
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer
