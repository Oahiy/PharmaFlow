"""Build a display history from the completed parent checkpoints."""

from collections.abc import Iterator
from contextlib import aclosing

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot

from agent.graph import requests_tools, unstarted_turn
from agent.state import ChatMessage, MultiState, ordered_results


def _worker_tools(state: MultiState) -> Iterator[ChatMessage]:
    for result in ordered_results(state):
        for message in result.messages:
            if isinstance(message, ToolMessage) or requests_tools(message):
                yield message


def _completed_report(snapshot: StateSnapshot) -> str | None:
    if not snapshot.next and "turn_id" in snapshot.values:
        last = snapshot.values["messages"][-1]
        if isinstance(last, AIMessage) and not requests_tools(last):
            return last.id
    return None


async def multi_history(
    graph: CompiledStateGraph,
    thread_config: RunnableConfig,
    latest: StateSnapshot,
) -> tuple[list[ChatMessage], bool, int]:
    messages: list[ChatMessage] = list(latest.values.get("messages", []))
    if latest.next == (START,):
        messages.extend((await unstarted_turn(graph, latest))["messages"])

    reports = {
        message.id
        for message in messages
        if isinstance(message, AIMessage) and not requests_tools(message)
    }
    records: dict[str, list[ChatMessage]] = {}
    current = _completed_report(latest)
    if current is not None:
        records[current] = list(_worker_tools(latest.values))
        reports.remove(current)
    if reports:
        older = graph.aget_state_history(thread_config, before=latest.config)
        async with aclosing(older):
            async for snapshot in older:
                report = _completed_report(snapshot)
                if report is not None and report in reports:
                    records[report] = list(_worker_tools(snapshot.values))
                    reports.remove(report)
                    if not reports:
                        break

    history: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.id in records:
            history.extend(records[message.id])
        history.append(message)

    last = messages[-1]
    incomplete = isinstance(last, HumanMessage)
    if incomplete:
        if last.id is None or last.id != latest.values.get("turn_id"):
            return history, True, 0
        history.extend(_worker_tools(latest.values))
    return history, incomplete, latest.values["step_count"]
