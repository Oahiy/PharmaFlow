"""Result fields shared by the case scorer and PostgreSQL storage."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import JsonValue

Category = Literal["single_tool", "multi_hop", "exception", "self_healing"]
FailureType = Literal[
    "replay_miss",
    "invalid_args",
    "agent_error",
    "step_limit",
    "repair_budget_exceeded",
    "tool_error",
    "wrong_tool",
    "rule_failed",
    "judge_call_error",
    "judge_parse_error",
    "judge_below_threshold",
]


@dataclass(frozen=True, kw_only=True)
class CaseResult:
    case_id: str
    thread_id: UUID
    trace_id: str
    tools_correct: bool
    params_valid: bool | None
    input_tokens: int | None
    elapsed_ms: int
    judge_score: int | None = None
    failure_type: FailureType | None = None
    error_detail: dict[str, JsonValue] | None = None
    repair_attempts: int | None = None

    @property
    def success(self) -> bool:
        return self.failure_type is None
