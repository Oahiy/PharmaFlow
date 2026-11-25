"""Check an OpenAI-compatible model with `uv run python -m scripts.check_model`."""

import asyncio
import re
import sys
from collections import Counter
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from pydantic import BaseModel, ValidationError

from agent.config import LLMSettings, load_environment
from agent.llm import create_llm

TEMPERATURE_PROMPT = (
    "Choose any integer from 1 to 100, inclusive. Reply with only that integer."
)
TEMPERATURE_SAMPLE_SIZE = 8
MIN_LOW_MODE_COUNT = 6


class ProbeFailure(RuntimeError):
    """The model responded, but did not satisfy a compatibility check."""


class Readiness(BaseModel):
    status: Literal["ready"]


@tool
def get_test_number() -> str:
    """Return the fixed number used to test tool calling."""
    return "17"


async def check_tool_choice(model: ChatOpenAI) -> None:
    forced = model.bind_tools([get_test_number], tool_choice=get_test_number.name)
    reply = await forced.ainvoke("Reply with the word ready.")
    if [call["name"] for call in reply.tool_calls] != [get_test_number.name]:
        raise ProbeFailure("模型未按强制 tool_choice 调用指定工具一次。")


def parse_readiness(reply: AIMessage) -> Readiness:
    return Readiness.model_validate_json(reply.text)


async def check_structured_output(model: ChatOpenAI) -> None:
    structured = model.with_structured_output(Readiness, method="function_calling")
    try:
        result = await structured.ainvoke("Return the readiness status 'ready'.")
    except BadRequestError as exc:
        if "tool_choice" not in str(exc).casefold():
            raise
        # Some providers reject forced tool_choice even with thinking disabled.
        # JSON mode can still provide structured output with Pydantic validation.
        json_output = model.bind(response_format={"type": "json_object"})
        validated = (json_output | parse_readiness).with_retry(
            retry_if_exception_type=(ValidationError,), stop_after_attempt=3
        )
        try:
            await validated.ainvoke('Return a JSON object: {"status": "ready"}.')
        except (BadRequestError, ValidationError) as fallback_error:
            raise ProbeFailure(
                f"函数调用被拒绝（{exc}），JSON 模式验证也失败（{fallback_error}）"
            ) from fallback_error
        raise ProbeFailure(f"函数调用被拒绝（{exc}），JSON 模式验证通过。") from exc

    if result != Readiness(status="ready"):
        raise ProbeFailure("结构化输出未解析为预期的 Readiness 实例。")


async def check_temperature(model: ChatOpenAI) -> str:
    counts = {0: Counter[int](), 1.5: Counter[int]()}
    backends = {temperature: set() for temperature in counts}
    reference_errors = []
    for sample in range(1, TEMPERATURE_SAMPLE_SIZE + 1):
        for temperature, answers in counts.items():
            try:
                reply = await model.ainvoke(TEMPERATURE_PROMPT, temperature=temperature)
                metadata = reply.response_metadata
                backend = (
                    metadata.get("model_name"),
                    metadata.get("system_fingerprint"),
                )
                backends[temperature].add(backend)
                text = reply.text.strip()
                number = re.fullmatch(r"0*([1-9][0-9]?|100)", text)
                if number is None:
                    detail = (
                        f"期望一个 1–100 的整数，收到 {text!r}"
                        if text
                        else "采样返回空文本"
                    )
                    raise ProbeFailure(
                        f"temperature={temperature:g} 第 {sample} 次采样格式错误。"
                        f"{detail}。模型与指纹 {backend!r}。"
                    )
            except (BadRequestError, ProbeFailure) as exc:
                if temperature == 0:
                    raise
                reference_errors.append(
                    f"第 {sample} 次采样 {type(exc).__name__}（{exc}）"
                )
                continue
            answers[int(number[1])] += 1

    summaries = []
    for temperature, answers in counts.items():
        label = "低温判定" if temperature == 0 else "高温参考"
        if answers:
            value, frequency = answers.most_common(1)[0]
            result = f"众数 {value}（{frequency}/{TEMPERATURE_SAMPLE_SIZE}）"
        else:
            result = "无有效回答"
        summaries.append(
            f"{label} temperature={temperature:g} {result}，"
            f"分布 {dict(answers.most_common())!r}，"
            f"模型与指纹 {sorted(backends[temperature], key=repr)!r}"
        )
    if reference_errors:
        summaries.append(
            f"高温参考有 {len(reference_errors)}/{TEMPERATURE_SAMPLE_SIZE} 次采样异常。"
            + "。".join(reference_errors)
        )
    summary = "。".join(summaries)
    if max(counts[0].values()) < MIN_LOW_MODE_COUNT:
        raise ProbeFailure(
            f"低温稳定性未达标，众数至少应出现 {MIN_LOW_MODE_COUNT} 次。{summary}"
        )
    return summary


async def check_tool_conversation(model: ChatOpenAI) -> None:
    with_tools = model.bind_tools([get_test_number])
    human = HumanMessage(content="Call get_test_number once and reply with its result.")
    reply = await with_tools.ainvoke([human])
    if [call["name"] for call in reply.tool_calls] != [get_test_number.name]:
        raise ProbeFailure("带工具对话的首轮未调用指定工具一次。")
    call = reply.tool_calls[0]
    tool_result: ToolMessage = await get_test_number.ainvoke(call)
    final = await with_tools.ainvoke([human, reply, tool_result])
    if final.tool_calls or "17" not in final.text:
        raise ProbeFailure("模型未使用工具结果给出最终回答。")

    call_id = call["id"]
    assert call_id is not None
    bad_arguments = '{"value":'
    detail = f"工具参数不是有效的 JSON，原始参数为 {bad_arguments!r}。"
    invalid_reply = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": get_test_number.name,
                "args": bad_arguments,
                "id": call_id,
                "error": detail,
            }
        ],
    )
    error_result = ToolMessage(content=detail, tool_call_id=call_id, status="error")
    try:
        await with_tools.ainvoke([human, invalid_reply, error_result])
    except BadRequestError as exc:
        raise ProbeFailure(
            f"服务拒绝带非法工具参数及错误响应的历史（{exc}）。"
        ) from exc


async def main() -> int:
    load_environment()
    model = create_llm(LLMSettings.from_env())
    checks = (
        ("强制 tool_choice", check_tool_choice),
        ("Pydantic 结构化输出", check_structured_output),
        ("温度对照采样", check_temperature),
        ("带工具的多轮对话", check_tool_conversation),
    )
    failed = False
    for name, check in checks:
        try:
            details = await check(model)
        except (
            BadRequestError,
            OutputParserException,
            ProbeFailure,
            ValidationError,
        ) as exc:
            print(f"失败  {name}  {exc}", file=sys.stderr)
            failed = True
        else:
            suffix = f"  {details}" if details is not None else ""
            print(f"通过  {name}{suffix}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
