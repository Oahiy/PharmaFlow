"""Typed channels and Send payloads for the conversation graphs."""

from operator import add
from typing import Annotated, Literal, NotRequired, Self, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_validator,
)

type ChatMessage = Annotated[
    HumanMessage | AIMessage | ToolMessage, Field(discriminator="type")
]
type TaskKind = Literal["lit", "mol"]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HistorySummary(BaseModel):
    text: str
    through: int
    ids: list[str]


class TurnInput(TypedDict):
    messages: Annotated[list[ChatMessage], add_messages]
    step_count: int
    max_steps: int


class AgentState(TurnInput):
    summary: NotRequired[HistorySummary | None]


class Subtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonBlank
    kind: TaskKind
    goal: NonBlank
    tools: list[NonBlank]


class Plan(BaseModel):
    """Independent subtasks to dispatch in parallel."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[Subtask]

    @model_validator(mode="after")
    def check_tasks(self, info: ValidationInfo) -> Self:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"子任务 ID 重复：{task_ids!r}")
        if sum(task.kind == "mol" for task in self.tasks) > 1:
            raise ValueError("最多允许一个 mol 子任务。")
        assert info.context is not None
        names: dict[TaskKind, set[str]] = info.context["tool_names"]
        for task in self.tasks:
            invalid = set(task.tools) - names[task.kind]
            if invalid:
                raise ValueError(
                    f"子任务 {task.id} 的 {task.kind} 工具无效：{sorted(invalid)!r}"
                )
        return self


class CriticIssue(BaseModel):
    subtask_id: str
    kind: Literal["missing_result", "unverified_ids"]
    ids: list[str]


class WorkerResult(BaseModel):
    subtask_id: str
    conclusion: str | None
    cited_ids: list[str]
    observed_ids: list[str]
    messages: list[ChatMessage]
    summary: HistorySummary | None


class DialogueHistory(TypedDict):
    summary: HistorySummary | None
    messages: list[ChatMessage]


class WorkerInput(TypedDict):
    subtask: Subtask
    history: DialogueHistory
    feedback: list[CriticIssue]
    attempt: int
    worker_messages: Annotated[list[ChatMessage], add_messages]
    worker_summary: HistorySummary | None
    worker_steps: int
    worker_max_steps: int


class WorkerState(WorkerInput):
    result: WorkerResult


def merge_by_task_id(
    previous: list[WorkerResult], current: list[WorkerResult]
) -> list[WorkerResult]:
    return list(
        {result.subtask_id: result for result in (*previous, *current)}.values()
    )


class MultiState(TypedDict):
    messages: Annotated[list[ChatMessage], add_messages]
    step_count: Annotated[int, add]
    max_steps: int
    tasks: list[Subtask]
    evidence_results: Annotated[list[WorkerResult], merge_by_task_id]
    molecule_results: Annotated[list[WorkerResult], merge_by_task_id]
    turn_id: str
    summary: NotRequired[HistorySummary | None]
    critique_passes: int
    issues: list[CriticIssue]
    reporter_unverified_ids: list[str]


def results_by_task(state: MultiState) -> dict[str, WorkerResult]:
    return {
        result.subtask_id: result
        for result in (*state["evidence_results"], *state["molecule_results"])
    }


def ordered_results(state: MultiState) -> list[WorkerResult]:
    results = results_by_task(state)
    return [results[task.id] for task in state["tasks"] if task.id in results]
