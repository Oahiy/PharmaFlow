"""Build the public configuration recorded for each run."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import JsonValue

from agent.config import PROJECT_ROOT, LLMSettings
from agent.llm import TEMPERATURE
from mcp_servers.lit.config import LitSettings
from rag.config import CHUNKS_PATH

from .cases import CaseSuite
from .digests import content_digest
from .judge import JUDGE_METHOD, JUDGE_RUBRIC_VERSION


def run_config(
    *,
    model: LLMSettings,
    suite: CaseSuite,
    lit: LitSettings,
    max_steps: int,
    commit: str,
    system_prompt: str,
    tools: list[BaseTool],
) -> dict[str, JsonValue]:
    url = urlsplit(str(model.base_url))
    base_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    model_config: dict[str, JsonValue] = {
        "model": model.model,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "thinking_disabled": model.thinking_disabled,
    }
    tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
    tool_schema_json = json.dumps(
        tool_schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        **model_config,
        "max_steps": max_steps,
        "lit_mode": lit.mode,
        "recordings_dir": str(lit.recordings_dir),
        "judge": {
            **model_config,
            "method": JUDGE_METHOD,
            "rubric_version": JUDGE_RUBRIC_VERSION,
            "thresholds": {
                case.id: case.judge.threshold
                for case in suite.cases
                if case.judge is not None
            },
        },
        "git_commit": commit,
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
                cwd=PROJECT_ROOT,
            )
        ),
        "cases_sha256": suite.sha256,
        "recordings_sha256": recordings_digest(lit.recordings_dir),
        "corpus_sha256": sha256(CHUNKS_PATH.read_bytes()).hexdigest(),
        "system_prompt_sha256": sha256(system_prompt.encode("utf-8")).hexdigest(),
        "tool_schema_sha256": sha256(tool_schema_json.encode("utf-8")).hexdigest(),
    }


def recordings_digest(directory: Path) -> str:
    return content_digest(
        (path.name, path.read_bytes()) for path in directory.glob("*.json")
    )
