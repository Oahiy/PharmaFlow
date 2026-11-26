"""Load and validate the agent's runtime configuration."""

import os
import subprocess
from pathlib import Path
from shutil import which
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

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


class AgentSettings(DatabaseSettings):
    llm: LLMSettings

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {"database_url": os.getenv("DATABASE_URL"), "llm": LLMSettings.from_env()}
        )
