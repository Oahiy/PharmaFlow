import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.serde import _msgpack
from psycopg_pool import AsyncConnectionPool

from agent.config import LLMSettings
from agent.llm import create_llm
from agent.persistence import setup_checkpointer
from scripts import check_model

LOW_STABLE = ("42",) * 5 + (" 042\n", "1", "100")
LOW_UNSTABLE = ("42",) * 5 + ("11", "39", "57")
HIGH_DISPERSED = ("42",) * 4 + ("19", "37", "68", "75")
HIGH_CONCENTRATED = ("42",) * 5 + ("19", "37", "68")


@pytest.mark.parametrize(
    ("low", "high", "mode_counts", "verdict", "exit_code", "reference_error"),
    [
        (LOW_STABLE, HIGH_DISPERSED, (6, 4), "通过", 0, None),
        (LOW_UNSTABLE, HIGH_DISPERSED, (5, 4), "失败", 1, None),
        (LOW_STABLE, HIGH_CONCENTRATED, (6, 5), "通过", 0, None),
        (("42",) * 8, ("42",) * 6 + ("17", "37"), (8, 6), "通过", 0, None),
        (("42",) * 8, ("42",) * 7 + ("17",), (8, 7), "通过", 0, None),
        (("42",) * 8, ("42",) * 8, (8, 8), "通过", 0, None),
        (LOW_STABLE, ("42",) * 7 + ("I choose 42",), (6, 7), "通过", 0, "格式错误"),
        (LOW_STABLE, (None,) * 8, (6, None), "通过", 0, "unsupported temperature"),
    ],
)
async def test_model_check_reports_temperature_verdict(
    monkeypatch, capsys, low, high, mode_counts, verdict, exit_code, reference_error
):
    observed_temperatures = []
    observed_prompts = []
    samples = {0: iter(low), 1.5: iter(high)}

    def model_reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        temperature = body["temperature"]
        observed_temperatures.append(temperature)
        observed_prompts.append(body["messages"][-1]["content"])
        content = next(samples[temperature])
        if content is None:
            return httpx.Response(
                400, json={"error": {"message": "unsupported temperature"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock-response-model",
                "system_fingerprint": f"mock-fingerprint-{temperature:g}",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def other_probe(model):
        pass

    monkeypatch.setattr(check_model, "load_environment", lambda: None)
    monkeypatch.setattr(check_model, "check_tool_choice", other_probe)
    monkeypatch.setattr(check_model, "check_structured_output", other_probe)
    monkeypatch.setattr(check_model, "check_tool_conversation", other_probe)
    settings = LLMSettings(
        base_url="https://example.invalid", api_key="test", model="mock"
    )
    monkeypatch.setattr(check_model.LLMSettings, "from_env", lambda: settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(model_reply)) as client:
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=str(settings.base_url),
            temperature=0,
            http_async_client=client,
            use_responses_api=False,
            max_retries=0,
        )
        monkeypatch.setattr(check_model, "create_llm", lambda settings: model)
        assert await check_model.main() == exit_code

    assert observed_temperatures == [0, 1.5] * 8
    assert observed_prompts == [check_model.TEMPERATURE_PROMPT] * 16
    captured = capsys.readouterr()
    output = captured.out if exit_code == 0 else captured.err
    assert f"{verdict}  温度对照采样" in output
    assert "高温参考 temperature=1.5" in output
    for temperature, frequency in zip((0, 1.5), mode_counts, strict=True):
        if frequency is None:
            assert f"temperature={temperature:g} 无有效回答，分布 {{}}" in output
        else:
            assert (
                f"temperature={temperature:g} 众数 42（{frequency}/8），分布" in output
            )
            assert (
                f"('mock-response-model', 'mock-fingerprint-{temperature:g}')" in output
            )
    if reference_error is not None:
        assert "高温参考有" in output
        assert reference_error in output


async def test_temperature_probe_rejects_empty_output(fake_chat_model):
    model = fake_chat_model(responses=[AIMessage(content=" \n ")])
    with pytest.raises(check_model.ProbeFailure, match="采样返回空文本"):
        await check_model.check_temperature(model)


@pytest.mark.parametrize("content", ["0", "101", "-7", "42.", "I choose 42"])
async def test_temperature_probe_rejects_out_of_range_or_extra_text(
    fake_chat_model, content
):
    model = fake_chat_model(responses=[AIMessage(content=content)])
    with pytest.raises(check_model.ProbeFailure, match="采样格式错误"):
        await check_model.check_temperature(model)


@pytest.mark.parametrize("accepts_error_history", [True, False])
async def test_tool_conversation_probes_matched_error_history(accepts_error_history):
    requests = []

    def model_reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content)["messages"])
        if len(requests) == 3 and not accepts_error_history:
            return httpx.Response(400, json={"error": {"message": "bad tool history"}})
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "probe-call",
                        "type": "function",
                        "function": {"name": "get_test_number", "arguments": "{}"},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "17"}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": "mock",
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
        if accepts_error_history:
            await check_model.check_tool_conversation(model)
        else:
            with pytest.raises(check_model.ProbeFailure, match="bad tool history"):
                await check_model.check_tool_conversation(model)

    _, success, failure = requests
    assert success[-1]["content"] == "17"
    (invalid_call,) = failure[-2]["tool_calls"]
    assert invalid_call["function"]["arguments"] == '{"value":'
    assert failure[-1]["role"] == "tool"
    assert failure[-1]["tool_call_id"] == invalid_call["id"] == "probe-call"
    assert "原始参数" in failure[-1]["content"]


def test_create_llm_disables_deepseek_thinking_when_switch_is_false():
    model = create_llm(
        LLMSettings(
            base_url="https://api.deepseek.com",
            api_key="test",
            model="deepseek-v4-pro",
            disable_thinking=False,
        )
    )
    assert model.extra_body == {"thinking": {"type": "disabled"}}
    assert model.temperature == 0


async def test_setup_checkpointer_checks_effective_strict_mode(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", False)
    pool = AsyncConnectionPool(open=False)
    with pytest.raises(RuntimeError, match="在导入 LangGraph 之前设置"):
        await setup_checkpointer(pool)
