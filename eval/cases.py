"""Load case definitions and validate them against the MCP request contracts."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)

from mcp_servers.chem.models import DruglikeInput, ExportPropertiesInput, MoleculeInput
from mcp_servers.lit.models import (
    LabelSectionInput,
    NonBlank,
    PubMedSearchInput,
    TrialAdverseEventsInput,
    TrialSearchInput,
)
from sandbox.models import RUN_PYTHON, RunPythonRequest, SandboxDatasets

from .digests import content_digest
from .results import Category

CASE_DIRECTORY = Path(__file__).resolve().parent / "cases"
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
FieldPath = Annotated[str, StringConstraints(pattern=r"^\w+(\.\w+)*$|^$")]
ToolName = Literal[
    "compute_properties",
    "export_properties",
    "filter_druglike",
    "search_pubmed",
    "search_trials",
    "get_trial_adverse_events",
    "get_label_section",
    "run_python",
]
ExpectedError = Literal["invalid_nct_format"]
LIT_TOOLS = frozenset(
    {"search_pubmed", "search_trials", "get_trial_adverse_events", "get_label_section"}
)
REQUEST_MODELS: dict[ToolName, type[BaseModel]] = {
    "compute_properties": MoleculeInput,
    "export_properties": ExportPropertiesInput,
    "filter_druglike": DruglikeInput,
    "search_pubmed": PubMedSearchInput,
    "search_trials": TrialSearchInput,
    "get_trial_adverse_events": TrialAdverseEventsInput,
    "get_label_section": LabelSectionInput,
    RUN_PYTHON: RunPythonRequest,
}


class CaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolArguments(CaseModel):
    request: dict[str, JsonValue]


class InvalidNctRequest(TrialAdverseEventsInput):
    nct_id: NonBlank

    @field_validator("nct_id")
    @classmethod
    def require_invalid_format(cls, value: str) -> str:
        if re.fullmatch(r"NCT[0-9]{8}", value):
            raise ValueError("应声明一个格式非法的 NCT 号。")
        return value


def normalize_request(
    tool: ToolName, request: dict[str, JsonValue], expected_error: ExpectedError | None
) -> dict[str, JsonValue]:
    request_type = (
        InvalidNctRequest
        if expected_error == "invalid_nct_format"
        else REQUEST_MODELS[tool]
    )
    values = request_type.model_validate(request).model_dump(mode="json")
    if tool == "get_label_section":
        values["drug"] = values["drug"].casefold()
        values["fields"] = sorted(values["fields"])
    if tool == "get_trial_adverse_events" and values["term"] is not None:
        values["term"] = values["term"].casefold()
    return values


def normalize_expected_request(
    tool: ToolName, request: dict[str, JsonValue], expected_error: ExpectedError | None
) -> dict[str, JsonValue]:
    if tool == RUN_PYTHON:
        return SandboxDatasets.model_validate(request).model_dump(mode="json")
    return normalize_request(tool, request, expected_error)


class ResultReference(CaseModel):
    step: Name
    path: FieldPath


class ExpectedCall(CaseModel):
    id: Name
    tool: ToolName
    request: dict[str, JsonValue]
    after: list[Name] = Field(default_factory=list)
    from_results: dict[str, ResultReference] = Field(default_factory=dict)
    expected_error: ExpectedError | None = None
    allow_repairable_failure: bool = False
    _normalized_request: dict[str, JsonValue] = PrivateAttr()

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if (
            self.expected_error == "invalid_nct_format"
            and self.tool != "get_trial_adverse_events"
        ):
            raise ValueError("invalid_nct_format 只适用于 get_trial_adverse_events。")
        if self.allow_repairable_failure and self.tool != RUN_PYTHON:
            raise ValueError("allow_repairable_failure 只适用于 run_python。")
        for field in self.from_results:
            if field not in self.request:
                raise ValueError(f"引用字段 {field} 需要在 request 中声明预检值。")
        # Keep the original values for replay preflight. Normalize only when comparing.
        self._normalized_request = normalize_expected_request(
            self.tool, self.request, self.expected_error
        )
        return self

    @property
    def normalized_request(self) -> dict[str, JsonValue]:
        return self._normalized_request

    @property
    def dependencies(self) -> set[str]:
        return {*self.after, *(ref.step for ref in self.from_results.values())}


class ToolField(CaseModel):
    step: Name
    path: FieldPath = ""


class ToolMatches(ToolField):
    type: Literal["tool_matches"]
    value: JsonValue


class ToolCount(ToolField):
    type: Literal["tool_count"]
    count: int = Field(ge=0)
    where: dict[str, JsonValue] = Field(default_factory=dict)


class NumberRange(CaseModel):
    min: FiniteFloat
    max: FiniteFloat

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.min > self.max:
            raise ValueError("数值范围要求 min <= max。")
        return self


class ToolNumber(ToolField, NumberRange):
    type: Literal["tool_number"]


class RegexPattern(CaseModel):
    pattern: NonBlank

    @field_validator("pattern")
    @classmethod
    def validate_regex(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"正则无效：{exc}") from exc
        return pattern


class AnswerRegex(RegexPattern):
    type: Literal["answer_regex"]
    absent: bool = False


class AnswerNumber(RegexPattern, NumberRange):
    type: Literal["answer_number"]

    @model_validator(mode="after")
    def require_number_group(self) -> Self:
        if re.compile(self.pattern).groups != 1:
            raise ValueError("answer_number 的 pattern 需要恰好一个数值捕获组。")
        return self


class SequencePattern(RegexPattern):
    values: list[NonBlank] = Field(min_length=1)
    allow_trailing: bool = False


class AnswerSequence(SequencePattern):
    type: Literal["answer_sequence"]


class ToolSequence(ToolField, SequencePattern):
    type: Literal["tool_sequence"]


Check = Annotated[
    ToolMatches
    | ToolCount
    | ToolNumber
    | ToolSequence
    | AnswerRegex
    | AnswerNumber
    | AnswerSequence,
    Field(discriminator="type"),
]


class JudgeRubric(CaseModel):
    coverage: NonBlank
    grounded: NonBlank
    correct_scope: NonBlank


class JudgeSpec(CaseModel):
    rubric: JudgeRubric
    threshold: int = Field(ge=0, le=3)


class InitialFailure(CaseModel):
    kind: Literal["code", "timeout", "oom"]
    stderr_contains: NonBlank | None = None


class EvalCase(CaseModel):
    id: Name
    category: Category
    question: NonBlank
    accepted_paths: list[list[ExpectedCall]] = Field(min_length=1)
    checks: list[Check] = Field(min_length=1)
    judge: JudgeSpec | None = None
    initial_export_properties: ExportPropertiesInput | None = None
    initial_run_python: RunPythonRequest | None = None
    initial_failure: InitialFailure | None = None

    def bind_property_export(self, dataset_id: str) -> Self:
        """Resolve the fixture's exported_properties ID in the seed and scoring rules."""
        spec = self.model_dump()
        requests = [spec["initial_run_python"]]
        requests.extend(
            step["request"]
            for path in spec["accepted_paths"]
            for step in path
            if step["tool"] == RUN_PYTHON
        )
        for request in requests:
            request["dataset_ids"] = [
                dataset_id if name == "exported_properties" else name
                for name in request["dataset_ids"]
            ]
        # Recompute normalized expected arguments for this concrete case only.
        return type(self).model_validate(spec)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if (self.category == "self_healing") != (self.initial_run_python is not None):
            raise ValueError("仅 self_healing 用例须提供 initial_run_python。")
        if (self.initial_run_python is None) != (self.initial_failure is None):
            raise ValueError("预置脚本和预期的初始失败须同时声明。")
        if (
            self.initial_export_properties is not None
            and self.initial_run_python is None
        ):
            raise ValueError("预置性质导出需要后续的沙盒脚本。")
        for path in self.accepted_paths:
            seen: set[str] = set()
            for step in path:
                if step.id in seen:
                    raise ValueError(f"路径中步骤 ID 重复：{step.id}")
                if step.dependencies - seen:
                    raise ValueError(f"步骤 {step.id} 只能引用路径中已声明的前置步骤。")
                seen.add(step.id)
            for check in self.checks:
                if isinstance(check, ToolField) and check.step not in seen:
                    raise ValueError(f"规则引用了路径中不存在的步骤：{check.step}")
        return self


@dataclass(frozen=True)
class CaseSuite:
    cases: list[EvalCase]
    sha256: str


def load_cases(directory: Path = CASE_DIRECTORY) -> CaseSuite:
    cases: list[EvalCase] = []
    files: dict[str, bytes] = {}
    ids: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        files[path.name] = path.read_bytes()
        try:
            case = EvalCase.model_validate(
                yaml.safe_load(files[path.name].decode("utf-8"))
            )
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"用例 {path.name} 无效：{exc}") from exc
        if case.id in ids:
            raise ValueError(f"用例 ID 重复：{case.id}")
        if case.id != path.stem:
            raise ValueError(f"用例 ID {case.id} 与文件名 {path.name} 不一致。")
        cases.append(case)
        ids.add(case.id)
    if not cases:
        raise ValueError(f"用例目录中没有 YAML 文件：{directory}")
    return CaseSuite(cases, content_digest(files.items()))
