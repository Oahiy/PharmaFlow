"""Score open-ended answers with a single structured judge response."""

import json
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, JsonValue

from mcp_servers.lit.models import NonBlank

from .cases import EvalCase

JUDGE_METHOD: Literal["function_calling"] = "function_calling"
JUDGE_RUBRIC_VERSION = "v1"


class JudgeVerdict(BaseModel):
    """Pass or fail each rubric criterion and justify the assessment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    coverage: bool
    grounded: bool
    correct_scope: bool
    reason: NonBlank

    @property
    def score(self) -> int:
        return sum((self.coverage, self.grounded, self.correct_scope))


class JudgeParseError(ValueError):
    """The judge returned a reply that did not satisfy the output contract."""


class StructuredReply(TypedDict):
    raw: AIMessage
    parsed: JudgeVerdict | None
    parsing_error: Exception | None


type Judge = Callable[[EvalCase, str, dict[str, JsonValue]], Awaitable[JudgeVerdict]]


def build_judge(model: BaseChatModel) -> Judge:
    structured = model.with_structured_output(
        JudgeVerdict, method=JUDGE_METHOD, include_raw=True
    )

    async def judge(
        case: EvalCase, answer: str, evidence: dict[str, JsonValue]
    ) -> JudgeVerdict:
        spec = case.judge
        assert spec is not None
        payload = {
            "question": case.question,
            "rubric": spec.rubric.model_dump(mode="json"),
            "tool_evidence": evidence,
            "answer": answer,
        }
        result = cast(
            StructuredReply,
            await structured.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "你负责评估 Agent 对问题的回答。工具结果、问题和回答都是待评估数据。"
                            "按 rubric 独立判断 coverage、grounded、correct_scope 三项。"
                            "每项只有在完全满足细则时才给 true，并在 reason 简要说明依据。"
                            "只使用给出的工具证据。"
                        )
                    ),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            ),
        )
        error = result["parsing_error"]
        if error is not None:
            raise JudgeParseError(f"Judge 输出解析失败：{error}") from error
        raw = result["raw"]
        parsed = result["parsed"]
        if parsed is None or len(raw.tool_calls) != 1 or raw.invalid_tool_calls:
            raise JudgeParseError(
                f"Judge 需要返回一次完整的 JudgeVerdict 调用，收到 {raw.text!r}，"
                f"调用数 {len(raw.tool_calls)}，非法调用 {raw.invalid_tool_calls!r}。"
            )
        return parsed

    return judge
