"""Score one interaction using the tool calls, their results, and the final answer."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from pydantic import JsonValue, ValidationError

from agent.graph import STEP_LIMIT_MESSAGE

from .cases import LIT_TOOLS, EvalCase, ExpectedCall, ToolArguments, normalize_request
from .checks import check_rules, field_value
from .results import FailureType

FAILURE_PRIORITY: tuple[FailureType, ...] = (
    "replay_miss",
    "invalid_args",
    "agent_error",
    "step_limit",
    "tool_error",
    "wrong_tool",
    "rule_failed",
    "judge_call_error",
    "judge_parse_error",
    "judge_below_threshold",
)


class ExceptionDetail(TypedDict):
    type: str
    message: str
    traceback: str


@dataclass
class ObservedCall:
    name: str | None
    id: str
    args: dict[str, JsonValue] | str | None
    message_index: int
    result: ToolMessage | None = None

    @property
    def evidence(self) -> JsonValue:
        assert self.result is not None and self.result.status == "success"
        return self.result.artifact["structured_content"]


def observed_calls(messages: list[AnyMessage]) -> list[ObservedCall]:
    calls: list[ObservedCall] = []
    pending: dict[str, ObservedCall] = {}
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            # Call IDs only need to be unique within one model response.
            pending = {}
            for call in (*message.tool_calls, *message.invalid_tool_calls):
                call_id = call["id"]
                assert call_id is not None
                observed = ObservedCall(call["name"], call_id, call["args"], index)
                pending[call_id] = observed
                calls.append(observed)
        elif isinstance(message, ToolMessage):
            pending[message.tool_call_id].result = message
    return calls


def parameter_error(
    step: ExpectedCall, call: ObservedCall, references: dict[str, JsonValue]
) -> str | None:
    if not isinstance(call.args, dict):
        return "工具参数不是 JSON 对象。"
    try:
        request = ToolArguments.model_validate(call.args).request
        actual = normalize_request(step.tool, request, step.expected_error)
        expected = (
            normalize_request(
                step.tool, {**step.request, **references}, step.expected_error
            )
            if references
            else step.normalized_request
        )
    except ValidationError as exc:
        return str(exc)
    differences = [field for field in expected if expected[field] != actual[field]]
    return f"参数与约束不符：{', '.join(differences)}。" if differences else None


def call_error(step: ExpectedCall, call: ObservedCall) -> FailureType | None:
    result = call.result
    if result is None:
        return "tool_error"
    if result.status == "success":
        return "tool_error" if step.expected_error is not None else None

    text = result.text
    if text == STEP_LIMIT_MESSAGE:
        return "step_limit"
    if step.tool in LIT_TOOLS and step.expected_error is None and "回放未命中:" in text:
        return "replay_miss"
    if (
        step.expected_error == "invalid_nct_format"
        and "request.nct_id" in text
        and "string_pattern_mismatch" in text
    ):
        return None
    return "tool_error"


@dataclass
class PathAssessment:
    tools_correct: bool
    matched_steps: int
    params_valid: bool | None
    evidence: dict[str, JsonValue]
    failures: set[FailureType]
    detail: dict[str, JsonValue]


def assess_path(
    path: list[ExpectedCall], calls: list[ObservedCall], answer: str | None
) -> PathAssessment:
    by_tool: dict[str | None, list[ExpectedCall]] = defaultdict(list)
    for step in path:
        by_tool[step.tool].append(step)
    tool_counts: dict[str | None, int] = defaultdict(int)
    assigned: dict[str, ObservedCall] = {}
    matched: list[tuple[ExpectedCall, ObservedCall]] = []
    selection_issues: list[JsonValue] = []
    failures: set[FailureType] = set()
    for call in calls:
        steps = by_tool[call.name]
        if not steps:
            selection_issues.append(f"未预期的工具 {call.name!r}（{call.id}）。")
            if (
                call.result is not None
                and call.result.status == "error"
                and call.result.text == STEP_LIMIT_MESSAGE
            ):
                failures.add("step_limit")
            continue
        index = tool_counts[call.name]
        tool_counts[call.name] += 1
        if index < len(steps):
            step = steps[index]
            assigned[step.id] = call
        else:
            step = steps[-1]
            selection_issues.append(f"工具 {call.name} 有额外调用（{call.id}）。")
        matched.append((step, call))

    for step in path:
        if step.id not in assigned:
            selection_issues.append(f"缺少步骤 {step.id}（{step.tool}）。")
    if not calls and answer is None:
        selection_issues.append("Agent 未给出回答或工具调用。")

    call_issues: list[JsonValue] = []
    for step, call in matched:
        available: dict[str, ObservedCall] = {}
        for dependency in sorted(step.dependencies):
            prior = assigned.get(dependency)
            if (
                prior is None
                or prior.message_index >= call.message_index
                or prior.result is None
            ):
                selection_issues.append(f"步骤 {step.id} 在 {dependency} 返回前调用。")
            else:
                available[dependency] = prior
        references: dict[str, JsonValue] = {}
        for field, ref in step.from_results.items():
            if ref.step not in available:
                continue
            prior = available[ref.step]
            assert prior.result is not None
            try:
                if prior.result.status == "error":
                    raise LookupError(prior.result.text)
                references[field] = field_value(prior.evidence, ref.path)
            except LookupError as exc:
                selection_issues.append(
                    f"步骤 {step.id} 无法从 {ref.step}.{ref.path} 取得 {field}：{exc}"
                )

        invalid = parameter_error(step, call, references)
        execution_failure = call_error(step, call)
        if invalid is not None:
            failures.add("invalid_args")
        # Only a call satisfying its case constraints can show a missing recording.
        if execution_failure is not None and (
            execution_failure != "replay_miss" or invalid is None
        ):
            failures.add(execution_failure)
        if invalid is not None or execution_failure is not None:
            call_issues.append(
                {
                    "step": step.id,
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "args": call.args,
                    "parameter_error": invalid,
                    "tool_message": call.result.text
                    if call.result is not None
                    else None,
                }
            )

    tools_correct = not selection_issues
    if not tools_correct:
        failures.add("wrong_tool")
    detail: dict[str, JsonValue] = {}
    if selection_issues:
        detail["selection"] = selection_issues
    if call_issues:
        detail["calls"] = call_issues
    return PathAssessment(
        tools_correct=tools_correct,
        matched_steps=len(assigned),
        params_valid="invalid_args" not in failures if matched else None,
        evidence={
            step: call.evidence
            for step, call in assigned.items()
            if call.result is not None and call.result.status == "success"
        },
        failures=failures,
        detail=detail,
    )


@dataclass(frozen=True)
class Assessment:
    tools_correct: bool
    params_valid: bool | None
    answer: str | None
    evidence: dict[str, JsonValue]
    failure_type: FailureType | None
    error_detail: dict[str, JsonValue] | None


def score_case(
    case: EvalCase,
    messages: list[AnyMessage],
    exception: ExceptionDetail | None = None,
) -> Assessment:
    answer = None
    if messages:
        final = messages[-1]
        if isinstance(final, AIMessage) and not (
            final.tool_calls or final.invalid_tool_calls
        ):
            answer = final.text
    calls = observed_calls(messages)
    path = max(
        (assess_path(path, calls, answer) for path in case.accepted_paths),
        key=lambda assessment: (
            assessment.tools_correct,
            assessment.matched_steps,
            assessment.params_valid is True,
        ),
    )
    if exception is not None:
        path.failures.add("agent_error")
        path.detail["exception"] = dict(exception)
    if answer is not None:
        rule_failures = check_rules(case.checks, path.evidence, answer)
        if rule_failures:
            path.failures.add("rule_failed")
            path.detail["rules"] = rule_failures
    elif not path.failures:
        path.failures.add("rule_failed")
        path.detail["answer"] = "Agent 未返回最终回答。"

    failure = next((name for name in FAILURE_PRIORITY if name in path.failures), None)
    return Assessment(
        tools_correct=path.tools_correct,
        params_valid=path.params_valid,
        answer=answer,
        evidence=path.evidence,
        failure_type=failure,
        error_detail=path.detail if path.detail else None,
    )
