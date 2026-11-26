"""Execute a replay suite serially with independent checkpoints and traces."""

import os
import traceback
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from urllib.parse import parse_qsl, quote, quote_plus, unquote, urlsplit
from uuid import UUID, uuid4

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langgraph.graph.state import CompiledStateGraph
from mcp.shared.exceptions import McpError
from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from psycopg import Error as DatabaseError
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow
from pydantic import JsonValue

from agent.config import AgentSettings, git_commit
from agent.graph import build_graph
from agent.mcp_client import open_mcp_tools
from agent.persistence import open_connection_pool, setup_checkpointer
from agent.state import AgentState
from agent.tracing import LANGFUSE_KEYS, CallbackFactory, trace_config
from mcp_servers.lit.config import DEFAULT_RECORDINGS_DIR, LitSettings

from .cases import CASE_DIRECTORY, EvalCase, load_cases
from .judge import Judge, JudgeParseError
from .provenance import run_config
from .results import CaseResult
from .scoring import ExceptionDetail, score_case
from .storage import finish_run, read_summary, save_result, setup_schema, start_run
from .usage import AgentUsage

MAX_STEPS = 10
MODEL_SERVICE_ERRORS = (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
_INFRASTRUCTURE_ERRORS = (
    DatabaseError,
    McpError,
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    *MODEL_SERVICE_ERRORS,
)
_CREDENTIAL_QUERY_FIELDS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "key",
        "password",
        "secret",
        "secret_key",
        "sig",
        "signature",
        "token",
    }
)


def infrastructure_error(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(infrastructure_error(error) for error in exc.exceptions)
    return isinstance(exc, _INFRASTRUCTURE_ERRORS)


def describe_exception(exc: Exception) -> ExceptionDetail:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def secret_values(settings: AgentSettings) -> tuple[str, ...]:
    database_url = settings.database_url.get_secret_value()
    endpoint = urlsplit(str(settings.llm.base_url))
    query = parse_qsl(endpoint.query)
    secrets = (
        database_url,
        conninfo_to_dict(database_url).get("password"),
        settings.llm.api_key.get_secret_value(),
        unquote(endpoint.password) if endpoint.password is not None else None,
        # Preserve unrelated occurrences of ordinary query values in diagnostics.
        *(f"{name}={value}" for name, value in query),
        *(part for part in endpoint.query.split("&") if "=" in part),
        *(
            value
            for name, value in query
            if name.casefold().replace("-", "_") in _CREDENTIAL_QUERY_FIELDS
        ),
        *(os.getenv(name) for name in LANGFUSE_KEYS),
    )
    variants = {
        encoded
        for secret in secrets
        if secret
        for encoded in (secret, quote(secret, safe=""), quote_plus(secret))
    }
    return tuple(sorted(variants, key=len, reverse=True))


def redact_text(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        value = value.replace(secret, "[redacted]")
    return value


def redact(value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {
            redact_text(key, secrets): redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    return value


async def execute_case(
    case: EvalCase,
    *,
    graph: CompiledStateGraph,
    judge: Judge,
    thread_id: UUID,
    trace_id: str,
    config: RunnableConfig,
    usage: AgentUsage,
) -> CaseResult:
    messages: list[AnyMessage] = [HumanMessage(content=case.question)]
    turn: AgentState = {"messages": messages, "step_count": 0, "max_steps": MAX_STEPS}
    agent_exception = None
    start = perf_counter_ns()
    try:
        async with aclosing(
            graph.astream(turn, config, stream_mode="values")
        ) as stream:
            async for state in stream:
                messages = state["messages"]
    except Exception as exc:
        if infrastructure_error(exc):
            raise
        agent_exception = exc
    elapsed_ms = (perf_counter_ns() - start) // 1_000_000

    assessment = score_case(
        case,
        messages,
        describe_exception(agent_exception) if agent_exception is not None else None,
    )
    result = CaseResult(
        case_id=case.id,
        thread_id=thread_id,
        trace_id=trace_id,
        elapsed_ms=elapsed_ms,
        input_tokens=usage.input_tokens,
        tools_correct=assessment.tools_correct,
        params_valid=assessment.params_valid,
        failure_type=assessment.failure_type,
        error_detail=assessment.error_detail,
    )
    if case.judge is None or not result.success:
        return result

    assert assessment.answer is not None
    try:
        verdict = await judge(case, assessment.answer, assessment.evidence)
    except JudgeParseError as exc:
        return replace(
            result,
            failure_type="judge_parse_error",
            error_detail={"exception": dict(describe_exception(exc))},
        )
    except Exception as exc:
        if infrastructure_error(exc):
            raise
        return replace(
            result,
            failure_type="judge_call_error",
            error_detail={"exception": dict(describe_exception(exc))},
        )
    score = verdict.score
    if score < case.judge.threshold:
        return replace(
            result,
            judge_score=score,
            failure_type="judge_below_threshold",
            error_detail={"judge": verdict.model_dump(mode="json")},
        )
    return replace(result, judge_score=score)


async def run_suite(
    settings: AgentSettings,
    *,
    tag: str,
    model: BaseChatModel,
    judge: Judge,
    callback_factory: CallbackFactory,
    case_directory: Path = CASE_DIRECTORY,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> tuple[UUID, list[DictRow]]:
    suite = load_cases(case_directory)
    lit = LitSettings(mode="replay", recordings_dir=recordings_dir.resolve())
    lit_env = {
        "LIT_MCP_MODE": lit.mode,
        "LIT_MCP_RECORDINGS_DIR": str(lit.recordings_dir),
        "NCBI_API_KEY": "",
    }
    run_id, user_id = uuid4(), uuid4()
    commit = git_commit()
    config = run_config(
        model=settings.llm, suite=suite, lit=lit, max_steps=MAX_STEPS, commit=commit
    )
    secrets = secret_values(settings)

    async with open_connection_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = await setup_checkpointer(pool)
        async with pool.connection() as conn:
            await setup_schema(conn)
        model_error = None
        async with open_mcp_tools(lit_env=lit_env) as tools:
            graph = build_graph(model, tools, checkpointer)
            async with pool.connection() as conn:
                await start_run(
                    conn,
                    run_id=run_id,
                    user_id=user_id,
                    tag=tag,
                    git_commit=commit,
                    config=config,
                    cases=[
                        {"case_id": case.id, "category": case.category}
                        for case in suite.cases
                    ],
                )
            print(f"run_id {run_id}", flush=True)

            for case in suite.cases:
                thread_id = uuid4()
                trace_id = Langfuse.create_trace_id()
                usage = AgentUsage()
                invocation = trace_config(
                    [callback_factory(trace_id), usage],
                    thread_id=thread_id,
                    user_id=user_id,
                    tags=[tag, commit],
                )
                invocation["configurable"] = {
                    "thread_id": str(thread_id),
                    "user_id": str(user_id),
                }
                invocation["recursion_limit"] = 2 * MAX_STEPS + 1
                try:
                    result = await execute_case(
                        case,
                        graph=graph,
                        judge=judge,
                        thread_id=thread_id,
                        trace_id=trace_id,
                        config=invocation,
                        usage=usage,
                    )
                except MODEL_SERVICE_ERRORS as exc:
                    # Close the MCP task groups before propagating the provider error.
                    model_error = exc
                    break
                if result.error_detail is not None:
                    result = replace(
                        result,
                        error_detail={
                            key: redact(value, secrets)
                            for key, value in result.error_detail.items()
                        },
                    )
                async with pool.connection() as conn:
                    await save_result(conn, run_id, result)

        if model_error is not None:
            raise model_error
        async with pool.connection() as conn:
            await finish_run(conn, run_id)
            return run_id, await read_summary(conn, run_id)
