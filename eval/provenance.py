"""Build the public configuration recorded for each run."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import JsonValue

from agent.config import PROJECT_ROOT, AgentSettings, LLMSettings
from agent.context import KEEP_UNITS, SUMMARY_PROMPT, TOKEN_BUDGET
from agent.graph import single_system_prompt
from agent.llm import TEMPERATURE
from agent.multi import (
    CRITIC,
    PLANNER,
    REPORTER,
    REPORTER_PROMPT,
    WORKER_NODES,
    planner_system_prompt,
    tool_catalog,
    worker_system_prompt,
)
from mcp_servers.lit.config import LitSettings
from rag.config import CHUNKS_PATH

from .cases import CaseSuite
from .digests import content_digest
from .judge import JUDGE_METHOD, JUDGE_RUBRIC_VERSION


def public_model_config(model: LLMSettings) -> dict[str, JsonValue]:
    url = urlsplit(str(model.base_url))
    base_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    return {
        "model": model.model,
        "base_url": base_url,
        "temperature": TEMPERATURE,
        "thinking_disabled": model.thinking_disabled,
    }


def tool_schema_digest(tools: list[BaseTool]) -> str:
    tool_schema_json = json.dumps(
        [convert_to_openai_tool(tool) for tool in tools],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(tool_schema_json.encode("utf-8")).hexdigest()


def graph_config(
    settings: AgentSettings, tools: list[BaseTool]
) -> dict[str, JsonValue]:
    graph = settings.graph
    model = public_model_config(settings.llm)
    models: dict[str, JsonValue] = {}
    worker_tools: dict[str, JsonValue] = {}
    legacy: dict[str, JsonValue] = {}
    if graph.mode == "multi":
        catalog = tool_catalog(tools)
        prompts = {PLANNER: planner_system_prompt(catalog), REPORTER: REPORTER_PROMPT}
        for kind, name in WORKER_NODES.items():
            prompts[name] = worker_system_prompt(kind)
            worker_tools[name] = tool_schema_digest(list(catalog[kind].values()))
        models.update({node: model for node in prompts})
        models[CRITIC] = None
    else:
        prompts = {"agent": single_system_prompt(graph)}
        models["agent"] = model
        legacy = {
            "system_prompt_sha256": sha256(
                prompts["agent"].encode("utf-8")
            ).hexdigest(),
            "tool_schema_sha256": tool_schema_digest(tools),
        }

    summary = settings.summary_llm
    if summary is not None:
        prompts["summary"] = SUMMARY_PROMPT
        models["summary"] = public_model_config(summary)

    return {
        **legacy,
        "graph_mode": graph.mode,
        "node_models": models,
        "compression": {
            "trim_tool_fields": graph.trim_tool_fields,
            "structured_worker_results": graph.structured_worker_results,
            "summarize_history": graph.summarize_history,
            "token_budget": TOKEN_BUDGET,
            "keep_units": KEEP_UNITS,
        },
        "node_system_prompt_sha256": {
            node: sha256(prompt.encode("utf-8")).hexdigest()
            for node, prompt in prompts.items()
        },
        "worker_tool_schema_sha256": worker_tools,
    }


def run_config(
    *,
    settings: AgentSettings,
    suite: CaseSuite,
    lit: LitSettings,
    max_steps: int,
    commit: str,
    tools: list[BaseTool],
) -> dict[str, JsonValue]:
    model_config = public_model_config(settings.llm)
    return {
        **model_config,
        **graph_config(settings, tools),
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
    }


def recordings_digest(directory: Path) -> str:
    return content_digest(
        (path.name, path.read_bytes()) for path in directory.glob("*.json")
    )
