"""FastAPI entry point for the persistent single-agent chat."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from langgraph.constants import START
from psycopg import AsyncConnection
from starlette.concurrency import run_in_threadpool

from agent.config import AgentSettings, load_environment

from .schemas import (
    RunRequest,
    ThreadCreated,
    ThreadState,
    UploadedDataset,
    serialize_message,
)
from .streaming import RunResponse, stream_run
from .uploads import UploadSettings, receive_upload, save_upload

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Strict checkpoint deserialization is configured at LangGraph import time.
    load_environment()
    from agent.graph import build_graph
    from agent.llm import create_llm
    from agent.mcp_client import open_mcp_tools
    from agent.persistence import open_connection_pool, setup_checkpointer

    settings = AgentSettings.from_env()
    app.state.database_url = settings.database_url
    app.state.upload_settings = UploadSettings.from_env()
    async with open_connection_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = await setup_checkpointer(pool)
        async with open_mcp_tools() as tools:
            app.state.graph = build_graph(create_llm(settings.llm), tools, checkpointer)
            yield


app = FastAPI(title="PharmaFlow Agent", lifespan=lifespan)
router = APIRouter(prefix="/api")


def get_agent_graph(request: Request) -> "CompiledStateGraph":
    return request.app.state.graph


def get_upload_settings(request: Request) -> UploadSettings:
    return request.app.state.upload_settings


GraphDependency = Annotated["CompiledStateGraph", Depends(get_agent_graph)]


async def claim_thread(thread_id: UUID, request: Request) -> AsyncIterator[None]:
    # This session owns the lock until the SSE response closes. Keep it separate
    # from the checkpoint pool so active runs cannot exhaust their own writers.
    async with await AsyncConnection.connect(
        request.app.state.database_url.get_secret_value(), autocommit=True
    ) as conn:
        cursor = await conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (str(thread_id),)
        )
        row = await cursor.fetchone()
        assert row is not None
        if not row[0]:
            raise HTTPException(409, "此会话正在运行，请等待当前运行结束后再发送。")
        yield


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/threads")
async def create_thread() -> ThreadCreated:
    return ThreadCreated(thread_id=uuid4())


@router.post(
    "/threads/{thread_id}/runs",
    response_class=RunResponse,
    status_code=200,
    responses={409: {"description": "同一会话已有运行中的请求"}},
    dependencies=[Depends(claim_thread, scope="request")],
)
async def run_thread(
    thread_id: UUID, request: RunRequest, graph: GraphDependency
) -> RunResponse:
    return RunResponse(stream_run(graph, thread_id, request))


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: UUID, graph: GraphDependency) -> ThreadState:
    snapshot = await graph.aget_state({"configurable": {"thread_id": str(thread_id)}})
    if snapshot.created_at is None:
        return ThreadState(
            thread_id=thread_id, exists=False, next_nodes=(), step_count=0, messages=[]
        )
    return ThreadState(
        thread_id=thread_id,
        exists=True,
        next_nodes=snapshot.next,
        # The initial input checkpoint precedes the first write to step_count.
        step_count=0 if snapshot.next == (START,) else snapshot.values["step_count"],
        messages=[
            serialize_message(message) for message in snapshot.values["messages"]
        ],
    )


@router.post(
    "/datasets/upload",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def upload_dataset(
    request: Request, settings: Annotated[UploadSettings, Depends(get_upload_settings)]
) -> UploadedDataset:
    async with receive_upload(request, settings.max_bytes) as file:
        return await run_in_threadpool(save_upload, file, settings)


app.include_router(router)
