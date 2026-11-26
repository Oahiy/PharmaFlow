import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from eval.cases import load_cases
from eval.judge import JudgeParseError, build_judge


@pytest.fixture(scope="module")
def rated_case():
    return next(case for case in load_cases().cases if case.id == "E03")


def verdict_call(args):
    return AIMessage(
        content="", tool_calls=[{"name": "JudgeVerdict", "id": "rating", "args": args}]
    )


async def test_judge_scores_three_independent_criteria(fake_chat_model, rated_case):
    model = fake_chat_model(
        responses=[
            verdict_call(
                {
                    "coverage": True,
                    "grounded": True,
                    "correct_scope": False,
                    "reason": "超出证据范围",
                }
            )
        ]
    )
    evidence = {"articles": {"total_count": 0}, "label": {"total_count": 0}}
    answer = "两个来源都无匹配。该药安全。"
    verdict = await build_judge(model)(rated_case, answer, evidence)
    assert verdict.score == 2
    assert verdict.reason == "超出证据范围"
    assert json.loads(model.received[0][-1].text) == {
        "question": rated_case.question,
        "rubric": rated_case.judge.rubric.model_dump(),
        "tool_evidence": evidence,
        "answer": answer,
    }


@pytest.mark.parametrize(
    "args",
    [
        {"coverage": True, "grounded": True, "reason": "缺少一项"},
        {
            "coverage": "true",
            "grounded": True,
            "correct_scope": True,
            "reason": "类型错误",
        },
    ],
)
async def test_judge_keeps_pydantic_parse_error(fake_chat_model, rated_case, args):
    judge = build_judge(fake_chat_model(responses=[verdict_call(args)]))
    with pytest.raises(JudgeParseError, match="输出解析失败") as error:
        await judge(rated_case, "回答", {})
    assert isinstance(error.value.__cause__, ValidationError)


async def test_judge_rejects_reply_without_function_call(fake_chat_model, rated_case):
    judge = build_judge(fake_chat_model(responses=[AIMessage(content="没有判定")]))
    with pytest.raises(JudgeParseError, match="没有判定"):
        await judge(rated_case, "回答", {})
