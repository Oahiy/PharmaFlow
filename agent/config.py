"""Load and validate the agent's runtime configuration."""

import os
import subprocess
from pathlib import Path
from shutil import which
from typing import Annotated, Literal, Self

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    StringConstraints,
    model_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Load the project environment before importing LangGraph."""
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def api_revision() -> str | None:
    """Use the deployed Langfuse release or the Git checkout revision."""
    release = os.getenv("LANGFUSE_RELEASE", "").strip()
    if release:
        return release
    if (PROJECT_ROOT / ".git").exists() and which("git") is not None:
        return git_commit()
    return None


class LLMSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    base_url: HttpUrl
    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1)
    disable_thinking: bool = False

    @property
    def thinking_disabled(self) -> bool:
        return (
            self.disable_thinking
            or "deepseek" in self.model.casefold()
            or self.base_url.host == "api.deepseek.com"
        )

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                "base_url": os.getenv("LLM_BASE_URL"),
                "api_key": os.getenv("LLM_API_KEY"),
                "model": os.getenv("LLM_MODEL"),
                "disable_thinking": os.getenv("LLM_DISABLE_THINKING", "false"),
            }
        )


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    database_url: SecretStr = Field(min_length=1)

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate({"database_url": os.getenv("DATABASE_URL")})


type GraphMode = Literal["single", "multi"]


class GraphSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: GraphMode = "single"
    trim_tool_fields: bool = False
    structured_worker_results: bool = True
    summarize_history: bool = False

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                "mode": os.getenv("AGENT_GRAPH_MODE", "single"),
                "trim_tool_fields": os.getenv("AGENT_TRIM_TOOL_FIELDS", "false"),
                "structured_worker_results": os.getenv(
                    "AGENT_STRUCTURED_WORKER_RESULTS", "true"
                ),
                "summarize_history": os.getenv("AGENT_SUMMARIZE_HISTORY", "false"),
            }
        )


class AgentSettings(DatabaseSettings):
    llm: LLMSettings
    graph: GraphSettings = Field(default_factory=GraphSettings)
    cheap_model: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None
    ) = None

    @model_validator(mode="after")
    def require_summary_model(self) -> Self:
        if self.graph.summarize_history and self.cheap_model is None:
            raise ValueError(
                "开启 AGENT_SUMMARIZE_HISTORY 时必须设置 LLM_CHEAP_MODEL。"
            )
        return self

    @property
    def summary_llm(self) -> LLMSettings | None:
        if not self.graph.summarize_history:
            return None
        assert self.cheap_model is not None
        return self.llm.model_copy(update={"model": self.cheap_model})

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {
                "database_url": os.getenv("DATABASE_URL"),
                "llm": LLMSettings.from_env(),
                "graph": GraphSettings.from_env(),
                "cheap_model": os.getenv("LLM_CHEAP_MODEL"),
            }
        )
