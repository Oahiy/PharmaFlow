import json
from typing import Annotated

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import STEP_LIMIT_MESSAGE, build_graph


async def test_agent_calls_tool_and_finishes(fake_chat_model):
    @tool
    def double(number: int) -> int:
        """Double a number."""
        return 2 * number

    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "double", "args": {"number": 3}, "id": "double-1"}
                ],
            ),
            AIMessage(content="结果是 6。"),
        ]
    )
    graph = build_graph(model, [double], InMemorySaver())
    config = {"configurable": {"thread_id": "tool-round"}}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="3 的两倍是多少？")],
            "step_count": 0,
            "max_steps": 5,
        },
        config,
    )

    question, call, tool_result, answer = state["messages"]
    assert isinstance(question, HumanMessage)
    assert isinstance(call, AIMessage)
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == call.tool_calls[0]["id"]
    assert tool_result.content == "6"
    assert tool_result.status == "success"
    assert isinstance(answer, AIMessage)
    assert answer.content == "结果是 6。"
    assert state["step_count"] == len(model.received) == 2
    assert model.received[1][-1] == tool_result
    assert (await graph.aget_state(config)).next == ()


async def test_agent_stops_at_max_steps_and_can_continue(fake_chat_model):
    executed = []

    @tool
    def record(number: int) -> str:
        """Record a number."""
        executed.append(number)
        return f"已记录 {number}"

    looping_model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record",
                        "args": {"number": number},
                        "id": f"record-{number}",
                    }
                ],
            )
            for number in (1, 2)
        ]
    )
    saver = InMemorySaver()
    graph = build_graph(looping_model, [record], saver)
    config = {"configurable": {"thread_id": "step-limit"}}
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="连续记录数字。")],
            "step_count": 0,
            "max_steps": 2,
        },
        config,
    )

    assert executed == [1]
    assert state["step_count"] == len(looping_model.received) == 2
    assert (await graph.aget_state(config)).next == ()
    capped_call, skipped = state["messages"][-2:]
    assert isinstance(capped_call, AIMessage)
    assert isinstance(skipped, ToolMessage)
    assert skipped.tool_call_id == capped_call.tool_calls[0]["id"] == "record-2"
    assert skipped.status == "error"
    assert skipped.content == STEP_LIMIT_MESSAGE

    next_model = fake_chat_model(responses=[AIMessage(content="已记录 1。")])
    resumed = build_graph(next_model, [record], saver)
    continued = await resumed.ainvoke(
        {
            "messages": [HumanMessage(content="记录了哪些数字？")],
            "step_count": 0,
            "max_steps": 2,
        },
        config,
    )

    assert continued["step_count"] == 1
    assert continued["messages"][:-2] == state["messages"]
    assert next_model.received[0][-3:-1] == [capped_call, skipped]


@pytest.mark.parametrize("valid_result", [None, "message", "command"])
@pytest.mark.parametrize("max_steps", [1, 2])
async def test_invalid_tool_calls_have_paired_results_on_the_wire(
    valid_result, max_steps
):
    executed = []

    @tool
    def double(
        number: int, call_id: Annotated[str, InjectedToolCallId]
    ) -> str | Command:
        """Double a number."""
        executed.append(number)
        content = str(2 * number)
        if valid_result == "command":
            return Command(
                update={
                    "messages": [ToolMessage(content=content, tool_call_id=call_id)]
                }
            )
        return content

    bad_arguments = '{"number":'
    calls = [
        {
            "id": "invalid",
            "type": "function",
            "function": {"name": "double", "arguments": bad_arguments},
        }
    ]
    if valid_result is not None:
        calls.insert(
            0,
            {
                "id": "valid",
                "type": "function",
                "function": {"name": "double", "arguments": '{"number": 3}'},
            },
        )
    requests = []

    def model_reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content)["messages"])
        message = {"role": "assistant", "content": "可以继续。"}
        finish_reason = "stop"
        if len(requests) == 1:
            message.update(content=None, tool_calls=calls)
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": f"mock-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(model_reply)) as client:
        model = ChatOpenAI(
            model="mock",
            api_key="test",
            base_url="https://example.invalid",
            http_async_client=client,
            use_responses_api=False,
            max_retries=0,
        )
        saver = InMemorySaver()
        graph = build_graph(model, [double], saver)
        config = {"configurable": {"thread_id": "invalid-call"}}
        first = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="计算 3 的两倍。")],
                "step_count": 0,
                "max_steps": max_steps,
            },
            config,
        )
        assert first["step_count"] == max_steps
        assert (await graph.aget_state(config)).next == ()
        errors = [
            msg
            for msg in first["messages"]
            if isinstance(msg, ToolMessage) and msg.tool_call_id == "invalid"
        ]
        (error,) = errors
        assert error.status == "error"
        assert bad_arguments in error.content

        resumed = build_graph(model, [double], saver)
        continued = await resumed.ainvoke(
            {
                "messages": [HumanMessage(content="继续。")],
                "step_count": 0,
                "max_steps": max_steps,
            },
            config,
        )
        assert continued["messages"][:-2] == first["messages"]
        assert continued["messages"][-1].content == "可以继续。"

    assert executed == ([3] if valid_result is not None and max_steps > 1 else [])
    assert len(requests) == max_steps + 1
    expected_ids = [call["id"] for call in calls]
    for messages in requests[1:]:
        assistant = messages[2]
        assert assistant["role"] == "assistant"
        assert [call["id"] for call in assistant["tool_calls"]] == expected_ids
        assert assistant["tool_calls"][-1]["function"]["arguments"] == bad_arguments
        results = messages[3 : 3 + len(expected_ids)]
        assert [msg["role"] for msg in results] == ["tool"] * len(expected_ids)
        assert [msg["tool_call_id"] for msg in results] == expected_ids
        assert bad_arguments in results[-1]["content"]
        if valid_result is not None:
            assert results[0]["content"] == (
                STEP_LIMIT_MESSAGE if max_steps == 1 else "6"
            )


@pytest.mark.parametrize("invalid_id", [None, "shared"])
async def test_unpairable_tool_call_ids_fail_before_checkpoint(
    fake_chat_model, invalid_id
):
    model = fake_chat_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "double", "args": {"number": 3}, "id": "shared"}],
                invalid_tool_calls=[
                    {
                        "name": "double",
                        "args": "{",
                        "id": invalid_id,
                        "error": "bad JSON",
                    }
                ],
            )
        ]
    )
    graph = build_graph(model, [], InMemorySaver())
    config = {"configurable": {"thread_id": "unpairable-id"}}
    with pytest.raises(ValueError, match="工具调用 ID 缺失或重复"):
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="计算 3 的两倍。")],
                "step_count": 0,
                "max_steps": 2,
            },
            config,
        )
    restored = await graph.aget_state(config)
    assert restored.next == ("agent",)
    assert len(restored.values["messages"]) == 1
    assert isinstance(restored.values["messages"][0], HumanMessage)
