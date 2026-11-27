"""Build a ReAct loop with explicit routing and a per-turn model call limit."""

from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, StateSnapshot

from sandbox.models import MAX_REPAIR_ROUNDS, REPAIR_LIMIT_MESSAGE, RUN_PYTHON
from sandbox.tool import is_repairable_failure, sandbox_config

from .config import GraphSettings
from .context import DATA_INSTRUCTIONS, prepare_input, render_prompt
from .state import AgentState, ChatMessage, HistorySummary, TurnInput

SYSTEM_PROMPT = SystemMessage(
    content=(
        "你是 PharmaFlow Agent，负责临床安全性证据核查和候选分子的类药性筛选。"
        "使用工具查询证据与计算分子性质，引用实际查到的文献、试验编号或药品说明书。"
        "跨文献、试验和说明书检索证据时，将问题改写为英文检索词，调用 search_evidence 并引用 citation_id。"
        "需要完整的不良事件表或说明书章节时，使用 get_trial_adverse_events 或 get_label_section。"
        "分子分析使用用户提供的 SMILES 或数据集 ID。项目自带的候选集 ID 为 candidates。"
        "性质计算、Lipinski 和 PAINS 筛选使用 chem-mcp。自定义排序、统计或画图才使用 run_python。"
        "需要按分子性质分析 CSV 时，先用 export_properties，再把派生数据集 ID 传给 run_python。"
        "导出性质时，报告被排除的非法 SMILES 行。派生数据集 ID 为空时结束这项分析。"
        "脚本读取 /work/input 下的数据，禁止手写 SMILES。图表保存为 /work/output 下的 PNG。"
        "每条回复最多调用一次 run_python。遇到代码错误，根据 traceback 修正。"
        "遇到超时或内存超限，缩小计算或数据规模后再提交。"
        "用户要求每条输入只列一行时，在同一行给出结果或解析错误，每条只列一次。"
    )
)
STEP_LIMIT_MESSAGE = "已达到本轮步数上限，跳过这次工具调用。"
PARALLEL_SANDBOX_MESSAGE = "每条回复只能执行一次 run_python，请等该次执行结束。"
MULTI_GRAPH_NAME = "PharmaFlowMulti"
MAX_CRITIC_RETURNS = 2
DEFAULT_GRAPH_SETTINGS = GraphSettings()

type ToolUpdates = (
    dict[str, list[ToolMessage]] | list[Command | dict[str, list[ToolMessage]]]
)


class ToolState(TypedDict):
    messages: list[ChatMessage]


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    if state["step_count"] >= state["max_steps"]:
        return END
    return "tools" if requests_tools(state["messages"][-1]) else END


def requests_tools(message: ChatMessage) -> bool:
    return isinstance(message, AIMessage) and bool(
        message.tool_calls or message.invalid_tool_calls
    )


def _invalid_tool_results(reply: AIMessage) -> list[ToolMessage]:
    results = []
    for call in reply.invalid_tool_calls:
        call_id = call["id"]
        assert call_id is not None
        detail = call["error"]
        if detail is None:
            detail = f"工具参数解析失败，原始参数为 {call['args']!r}。"
        results.append(
            ToolMessage(
                content=detail, name=call["name"], tool_call_id=call_id, status="error"
            )
        )
    return results


def _sandbox_failures_in_turn(messages: list[ChatMessage]) -> int:
    failures = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and is_repairable_failure(message):
            failures += 1
    return failures


def _append_tool_results(
    updates: ToolUpdates, results: list[ToolMessage]
) -> ToolUpdates:
    if not results:
        return updates
    if isinstance(updates, list):
        return [*updates, {"messages": results}]
    return {"messages": [*updates["messages"], *results]}


def is_multi_graph(graph: CompiledStateGraph) -> bool:
    return graph.name == MULTI_GRAPH_NAME


def react_recursion_limit(max_steps: int) -> int:
    return 2 * max_steps + 1


def turn_recursion_limit(graph: CompiledStateGraph, max_steps: int) -> int:
    if is_multi_graph(graph):
        # Planner, (workers + Critic) per round, Reporter, graph input.
        return 2 * (1 + MAX_CRITIC_RETURNS) + 3
    return react_recursion_limit(max_steps)


async def resume_config(
    graph: CompiledStateGraph, thread_config: RunnableConfig, snapshot: StateSnapshot
) -> RunnableConfig:
    """Use the pending turn's original model-call budget when resuming it."""
    if is_multi_graph(graph):
        return {**thread_config, "recursion_limit": turn_recursion_limit(graph, 0)}
    state = (
        await unstarted_turn(graph, snapshot)
        if snapshot.next == (START,)
        else snapshot.values
    )
    return {
        **thread_config,
        "recursion_limit": react_recursion_limit(state["max_steps"]),
    }


async def unstarted_turn(
    graph: CompiledStateGraph, snapshot: StateSnapshot
) -> TurnInput:
    """Read the saved input before LangGraph writes it to the State channels."""
    checkpointer = graph.checkpointer
    assert isinstance(checkpointer, BaseCheckpointSaver)
    saved = await checkpointer.aget_tuple(snapshot.config)
    assert saved is not None
    return saved.checkpoint["channel_values"][START]


def react_reply(reply: AIMessage, step_count: int, max_steps: int) -> list[ChatMessage]:
    call_ids = [call["id"] for call in (*reply.tool_calls, *reply.invalid_tool_calls)]
    if None in call_ids or "" in call_ids or len(set(call_ids)) != len(call_ids):
        raise ValueError(f"模型返回的工具调用 ID 缺失或重复：{call_ids!r}")
    messages: list[ChatMessage] = [reply]
    if step_count >= max_steps:
        # Close unexecuted tool calls so the next turn has valid chat history.
        for call in reply.tool_calls:
            assert call["id"] is not None
            messages.append(
                ToolMessage(
                    content=STEP_LIMIT_MESSAGE,
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            )
        messages.extend(_invalid_tool_results(reply))
    return messages


async def execute_tools(
    tool_node: ToolNode, state: ToolState, config: RunnableConfig
) -> ToolUpdates:
    messages = state["messages"]
    reply = messages[-1]
    assert isinstance(reply, AIMessage)
    errors = _invalid_tool_results(reply)
    if RUN_PYTHON not in tool_node.tools_by_name or not any(
        call["name"] == RUN_PYTHON for call in reply.tool_calls
    ):
        return _append_tool_results(await tool_node.ainvoke(state, config), errors)

    failures = _sandbox_failures_in_turn(messages)
    sandbox_call = None
    other_calls = []
    for call in reply.tool_calls:
        if call["name"] != RUN_PYTHON:
            other_calls.append(call)
            continue
        if failures > MAX_REPAIR_ROUNDS:
            reason = REPAIR_LIMIT_MESSAGE
        elif sandbox_call is not None:
            reason = PARALLEL_SANDBOX_MESSAGE
        else:
            sandbox_call = call
            continue
        assert call["id"] is not None
        errors.append(
            ToolMessage(
                name=RUN_PYTHON,
                tool_call_id=call["id"],
                status="error",
                content=reason,
            )
        )

    def select_calls(calls: list[ToolCall]) -> ToolState:
        selected = reply.model_copy(update={"tool_calls": calls})
        return {**state, "messages": [*messages[:-1], selected]}

    updates = {"messages": []}
    if other_calls:
        updates = await tool_node.ainvoke(select_calls(other_calls), config)
    if sandbox_call is not None:
        # Finish parallel MCP work before starting a sandbox worker.
        assert reply.id is not None
        sandbox_updates = await tool_node.ainvoke(
            select_calls([sandbox_call]),
            sandbox_config(
                config, reply.id, final_repair=failures == MAX_REPAIR_ROUNDS
            ),
        )
        updates = _append_tool_results(updates, sandbox_updates["messages"])
    return _append_tool_results(updates, errors)


def single_system_prompt(settings: GraphSettings) -> str:
    if settings.trim_tool_fields or settings.summarize_history:
        return f"{SYSTEM_PROMPT.text}\n\n{DATA_INSTRUCTIONS}"
    return SYSTEM_PROMPT.text


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    settings: GraphSettings = DEFAULT_GRAPH_SETTINGS,
    summary_model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    if settings.summarize_history:
        assert summary_model is not None
    else:
        summary_model = None
    if settings.mode == "multi":
        from .multi import build_multi_graph

        return build_multi_graph(model, tools, checkpointer, settings, summary_model)

    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)
    tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
    prompt = single_system_prompt(settings)

    def render(
        messages: list[ChatMessage], summary: HistorySummary | None
    ) -> list[BaseMessage]:
        if not (settings.trim_tool_fields or settings.summarize_history):
            return [SYSTEM_PROMPT, *messages]
        index = max(
            i for i, message in enumerate(messages) if isinstance(message, HumanMessage)
        )
        question = messages[index]
        assert isinstance(question, HumanMessage)
        return render_prompt(
            prompt,
            {"summary": summary, "messages": messages[:index]},
            question,
            messages[index + 1 :],
            trim=settings.trim_tool_fields,
        )

    async def call_agent(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, list[ChatMessage] | int | HistorySummary | None]:
        prepared = await prepare_input(
            state["messages"],
            state.get("summary"),
            render,
            config=config,
            role="agent",
            tool_schemas=tool_schemas,
            summary_model=summary_model,
            trim=settings.trim_tool_fields,
        )
        reply = await model_with_tools.ainvoke(prepared.messages, prepared.config)
        step_count = state["step_count"] + 1
        return {
            "messages": react_reply(reply, step_count, state["max_steps"]),
            "step_count": step_count,
            "summary": prepared.summary,
        }

    async def call_tools(state: AgentState, config: RunnableConfig) -> ToolUpdates:
        return await execute_tools(tool_node, state, config)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
