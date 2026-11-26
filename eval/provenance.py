"""Build the public configuration recorded for each run."""

import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue

from agent.config import PROJECT_ROOT, LLMSettings
from agent.llm import TEMPERATURE
from mcp_servers.lit.config import LitSettings

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
) -> dict[str, JsonValue]:
    url = urlsplit(str(model.base_url))
    base_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    model_config: dict[str, JsonValue] = {
        "model": model.model,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "thinking_disabled": model.thinking_disabled,
    }
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
    }


def recordings_digest(directory: Path) -> str:
    return content_digest(
        (path.name, path.read_bytes()) for path in directory.glob("*.json")
    )
