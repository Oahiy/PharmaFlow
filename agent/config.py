"""Load and validate the agent's runtime configuration."""

import os
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Load the project environment before importing LangGraph."""
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


class LLMSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    base_url: HttpUrl
    api_key: SecretStr = Field(min_length=1)
    model: str = Field(min_length=1)
    disable_thinking: bool = False

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


class AgentSettings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    database_url: SecretStr = Field(min_length=1)
    llm: LLMSettings

    @classmethod
    def from_env(cls) -> Self:
        return cls.model_validate(
            {"database_url": os.getenv("DATABASE_URL"), "llm": LLMSettings.from_env()}
        )
