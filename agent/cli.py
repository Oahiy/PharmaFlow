"""Run the persistent ReAct baseline with `uv run python -m agent.cli`."""

import argparse
import asyncio
from uuid import uuid4

from .config import AgentSettings, load_environment


async def chat(settings: AgentSettings, thread_id: str, max_steps: int) -> None:
    # LangGraph reads LANGGRAPH_STRICT_MSGPACK at import time.
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig

    from .graph import STEP_LIMIT_MESSAGE, build_graph, resume_config
    from .llm import create_llm
    from .mcp_client import lit_env_from_process, open_mcp_tools
    from .persistence import open_connection_pool, setup_checkpointer
    from .state import AgentState

    def print_reply(state: AgentState) -> None:
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            print(f"PharmaFlow> {STEP_LIMIT_MESSAGE}")
        else:
            assert isinstance(last_message, AIMessage)
            print(f"PharmaFlow> {last_message.text}")

    model = create_llm(settings.llm)
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    config: RunnableConfig = {
        **thread_config,
        "recursion_limit": 2 * max_steps + 1,
    }
    async with open_connection_pool(settings.database_url.get_secret_value()) as pool:
        checkpointer = await setup_checkpointer(pool)
        async with open_mcp_tools(lit_env=lit_env_from_process()) as tools:
            graph = build_graph(model, tools, checkpointer)
            print(f"会话 ID：{thread_id}。下次可通过 --thread-id 续聊。")
            snapshot = await graph.aget_state(thread_config)
            if snapshot.next:
                print("正在从检查点恢复上一轮对话。")
                print_reply(
                    await graph.ainvoke(None, resume_config(thread_config, snapshot))
                )

            while True:
                try:
                    question = (await asyncio.to_thread(input, "你> ")).strip()
                except EOFError:
                    return
                if not question:
                    continue
                turn: AgentState = {
                    "messages": [HumanMessage(content=question)],
                    "step_count": 0,
                    "max_steps": max_steps,
                }
                print_reply(await graph.ainvoke(turn, config))


def main() -> None:
    parser = argparse.ArgumentParser(description="PharmaFlow ReAct 交互式会话")
    parser.add_argument("--thread-id", help="要继续的会话 ID")
    parser.add_argument(
        "--max-steps", type=int, default=10, help="每轮最多调用模型的次数"
    )
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps 必须大于 0")

    load_environment()
    thread_id = args.thread_id if args.thread_id is not None else str(uuid4())
    asyncio.run(chat(AgentSettings.from_env(), thread_id, args.max_steps))


if __name__ == "__main__":
    main()
