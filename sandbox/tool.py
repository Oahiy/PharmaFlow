"""Expose sandbox execution and its repairable errors as a LangChain tool."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated
from uuid import NAMESPACE_URL, UUID, uuid5

from anyio import CancelScope
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool

from .artifacts import publish_artifacts
from .executor import DatasetInputError, PythonExecution, PythonExecutor, execute_python
from .models import (
    REPAIR_LIMIT_MESSAGE,
    RUN_PYTHON,
    RunPythonRequest,
    RunPythonResult,
    SandboxError,
)


@dataclass(slots=True)
class _WorkerScope:
    active: set[asyncio.Task[ToolMessage]] = field(default_factory=set)
    closing: bool = False


_worker_scope: ContextVar[_WorkerScope | None] = ContextVar(
    "sandbox_worker_scope", default=None
)


@asynccontextmanager
async def sandbox_worker_scope() -> AsyncIterator[None]:
    """Keep an SSE request alive until its Docker workers finish."""
    scope = _WorkerScope()
    token = _worker_scope.set(scope)
    try:
        yield
    finally:
        scope.closing = True
        _worker_scope.reset(token)
        # Closing the graph's event iterator can cancel its inner awaits before
        # they reach their own cleanup. Wait here, inside the request's lock.
        with CancelScope(shield=True):
            if scope.active:
                await asyncio.wait(scope.active)


def sandbox_config(
    config: RunnableConfig, message_id: str, *, final_repair: bool
) -> RunnableConfig:
    return {
        **config,
        "configurable": {
            **config["configurable"],
            "sandbox_message_id": message_id,
            "sandbox_final_repair": final_repair,
        },
    }


def is_repairable_failure(message: ToolMessage) -> bool:
    return (
        message.name == RUN_PYTHON
        and message.status == "error"
        and message.artifact is not None
        and message.artifact["repairable_failure"]
    )


def create_run_python_tool(executor: PythonExecutor = execute_python) -> BaseTool:
    @tool(RUN_PYTHON)
    async def run_python(
        request: RunPythonRequest,
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
    ) -> ToolMessage:
        """Run Python in a fresh container for custom sorting, statistics, and plotting.

        Args:
            request: Python code and dataset_ids for this call. The fresh
                container receives only the listed CSVs at
                /work/input/<dataset_id>.csv. Save only PNGs to /work/output/.
                For QED, MW, and LogP, use the dataset from export_properties.

        Returns:
            Stdout, stderr (the traceback tail if truncated), exit code, and
            PNG filenames, media types, and sizes. Execution failures include
            a diagnostic to guide the next attempt.
        """
        context = config["configurable"]
        thread_key = context["thread_id"]
        try:
            thread_id = UUID(thread_key)
        except ValueError:
            # The CLI also supports named threads created before Task 6.
            thread_id = uuid5(NAMESPACE_URL, f"pharmaflow:cli:{thread_key}")

        def execute_and_publish() -> ToolMessage:
            try:
                result = executor(request, tool_call_id)
            except DatasetInputError as exc:
                result = PythonExecution(
                    stdout="",
                    stderr="",
                    exit_code=None,
                    error=SandboxError(kind="input", message=str(exc)),
                )
            error = result.error
            # Input resolution precedes execution. All other typed failures
            # (including a script writing invalid output) consumed an attempt.
            repairable = error is not None and error.kind != "input"
            if repairable and context["sandbox_final_repair"]:
                assert error is not None
                error = SandboxError(
                    kind=error.kind, message=f"{error.message} {REPAIR_LIMIT_MESSAGE}"
                )
            artifacts = publish_artifacts(
                thread_id, context["sandbox_message_id"], tool_call_id, result.files
            )
            output = RunPythonResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                files=artifacts,
                error=error,
            )
            return ToolMessage(
                name=RUN_PYTHON,
                tool_call_id=tool_call_id,
                status="success" if error is None else "error",
                content=output.model_dump_json(),
                artifact={
                    "structured_content": output.model_dump(mode="json"),
                    "files": [file.model_dump(mode="json") for file in artifacts],
                    "repairable_failure": repairable,
                },
            )

        # A disconnected stream may cancel the graph, but the Docker worker
        # must finish cleanup before the API releases the conversation lock.
        scope = _worker_scope.get()
        if scope is not None and scope.closing:
            raise asyncio.CancelledError
        worker = asyncio.create_task(asyncio.to_thread(execute_and_publish))
        if scope is not None:
            scope.active.add(worker)
            worker.add_done_callback(scope.active.discard)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            with CancelScope(shield=True):
                await asyncio.shield(worker)
            raise

    return run_python
