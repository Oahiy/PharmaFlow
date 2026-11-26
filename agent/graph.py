"""Build a ReAct loop with explicit routing and a per-turn model call limit."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, StateSnapshot

from .state import AgentState

SYSTEM_PROMPT = SystemMessage(
    content=(
        "你是 PharmaFlow Agent，负责临床安全性证据核查和候选分子的类药性筛选。"
        "使用工具查询证据与计算分子性质，引用实际查到的文献、试验编号或药品说明书。"
        "分子分析使用用户提供的 SMILES 或数据集 ID。项目自带的候选集 ID 为 candidates。"
    )
)
STEP_LIMIT_MESSAGE = "已达到本轮步数上限，跳过这次工具调用。"


def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    if state["step_count"] >= state["max_steps"]:
        return END
    reply = state["messages"][-1]
    assert isinstance(reply, AIMessage)
    if reply.tool_calls or reply.invalid_tool_calls:
        return "tools"
    return END


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


def resume_config(
    thread_config: RunnableConfig, snapshot: StateSnapshot
) -> RunnableConfig:
    """Use the pending turn's original model-call budget when resuming it."""
    config = thread_config.copy()
    # The initial input checkpoint can precede the write to max_steps.
    if snapshot.next != (START,):
        config["recursion_limit"] = 2 * snapshot.values["max_steps"] + 1
    return config


def build_graph(
    model: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def call_agent(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, list[AnyMessage] | int]:
        reply = await model_with_tools.ainvoke(
            [SYSTEM_PROMPT, *state["messages"]], config
        )
        call_ids = [
            call["id"] for call in (*reply.tool_calls, *reply.invalid_tool_calls)
        ]
        if None in call_ids or "" in call_ids or len(set(call_ids)) != len(call_ids):
            raise ValueError(f"模型返回的工具调用 ID 缺失或重复：{call_ids!r}")
        step_count = state["step_count"] + 1
        messages: list[AnyMessage] = [reply]
        if step_count >= state["max_steps"]:
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
        return {"messages": messages, "step_count": step_count}

    async def call_tools(
        state: AgentState, config: RunnableConfig
    ) -> dict[str, list[ToolMessage]] | list[Command | dict[str, list[ToolMessage]]]:
        reply = state["messages"][-1]
        assert isinstance(reply, AIMessage)
        updates = await tool_node.ainvoke(state, config)
        if not reply.invalid_tool_calls:
            return updates

        errors = _invalid_tool_results(reply)
        if isinstance(updates, list):
            return [*updates, {"messages": errors}]
        return {"messages": [*updates["messages"], *errors]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
