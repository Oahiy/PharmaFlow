"""Deterministic rules for structured tool results and final answers."""

import re
from unicodedata import normalize

from pydantic import JsonValue

from .cases import (
    AnswerNumber,
    AnswerRegex,
    AnswerSequence,
    Check,
    SequencePattern,
    ToolCount,
    ToolField,
    ToolMatches,
    ToolNumber,
    ToolSequence,
)

type ToolCheck = ToolMatches | ToolCount | ToolNumber | ToolSequence


def field_value(value: JsonValue, path: str) -> JsonValue:
    for part in path.split(".") if path else ():
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdecimal():
            value = value[int(part)]
        else:
            raise LookupError(f"无法读取 {part!r}，所在路径为 {path!r}。")
    return value


def matches_value(actual: JsonValue, expected: JsonValue) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and matches_value(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                matches_value(item, target) for item, target in zip(actual, expected)
            )
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    return actual == expected


def check_sequence(rule: SequencePattern, text: str) -> bool:
    values = [match.group() for match in re.finditer(rule.pattern, text)]
    if rule.allow_trailing:
        values = values[: len(rule.values)]
    return values == rule.values


def check_tool(rule: ToolCheck, evidence: dict[str, JsonValue]) -> str | None:
    try:
        value = field_value(evidence[rule.step], rule.path)
    except LookupError as exc:
        return f"工具结果字段缺失：{exc}"

    match rule:
        case ToolMatches():
            passed = matches_value(value, rule.value)
        case ToolCount():
            passed = (
                isinstance(value, list)
                and sum(
                    not rule.where or matches_value(row, rule.where) for row in value
                )
                == rule.count
            )
        case ToolNumber():
            passed = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and rule.min <= value <= rule.max
            )
        case ToolSequence():
            passed = isinstance(value, str) and check_sequence(rule, value)
    if passed:
        return None
    detail = f"步骤 {rule.step} 的字段 {rule.path or '根'} 不满足规则。"
    if rule.path == "stdout":
        detail = f"{detail}\n实际 stdout\n{value}"
    return detail


def check_answer(rule: AnswerRegex | AnswerNumber | AnswerSequence, text: str) -> bool:
    if isinstance(rule, AnswerSequence):
        return check_sequence(rule, text)
    if isinstance(rule, AnswerRegex):
        found = re.search(rule.pattern, text, flags=re.IGNORECASE) is not None
        return found != rule.absent

    numbers = re.findall(rule.pattern, text, flags=re.IGNORECASE)
    try:
        return bool(numbers) and all(
            rule.min <= float(value.replace(",", "")) <= rule.max for value in numbers
        )
    except ValueError:
        return False


def check_rules(
    rules: list[Check], evidence: dict[str, JsonValue], answer: str
) -> list[JsonValue]:
    text = normalize("NFKC", answer).replace("**", "").replace("`", "")
    failures: list[JsonValue] = []
    for rule in rules:
        if isinstance(rule, ToolField):
            detail = check_tool(rule, evidence)
        else:
            detail = None if check_answer(rule, text) else "最终回答不满足规则。"
        if detail is not None:
            failures.append({"rule": rule.model_dump(mode="json"), "detail": detail})
    return failures
