"""Plan independent work, run checkpointed ReAct branches, and report their results."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Overwrite, Send
from pydantic import ValidationError

from sandbox.models import RUN_PYTHON

from .config import GraphSettings
from .context import (
    DATA_INSTRUCTIONS,
    data_block,
    message_data,
    model_config,
    prepare_input,
    render_prompt,
    retained_messages,
)
from .graph import (
    MAX_CRITIC_RETURNS,
    MULTI_GRAPH_NAME,
    execute_tools,
    react_recursion_limit,
    react_reply,
    requests_tools,
)
from .mcp_client import MCP_SERVER
from .references import cited_ids, observed_ids
from .state import (
    ChatMessage,
    CriticIssue,
    DialogueHistory,
    HistorySummary,
    MultiState,
    Plan,
    Subtask,
    TaskKind,
    WorkerInput,
    WorkerResult,
    WorkerState,
    ordered_results,
    results_by_task,
)

PLANNER = "Planner"
REPORTER = "Reporter"
LIT_RESEARCHER = "LitResearcher"
MOL_ANALYST = "MolAnalyst"
CRITIC = "Critic"
WORKER_NODES = {"lit": LIT_RESEARCHER, "mol": MOL_ANALYST}

PLANNER_PROMPT = (
    "你是 PharmaFlow Planner，负责安排临床安全性证据核查和候选分子的类药性筛选。"
    "按本轮请求规划独立子任务，给出唯一 ID、类型、目标和必要的可用工具。"
    "如果后一步的参数来自前一步的结果，例如从书目标题或试验组名取得药名再查说明书，"
    "把整个调用链放在同一个子任务中，分配每一步的工具，在目标中写明顺序和参数来源。"
    "仅互不依赖的任务可以并行，整个计划最多有一个 mol 子任务。\n"
    "按请求选择专用工具，只分配完成目标所需的工具。\n"
    "- PubMed 检索用 search_pubmed。按条件发现试验用 search_trials，"
    "已知试验号查不良事件用 get_trial_adverse_events。查 openFDA 说明书章节用 get_label_section。"
    "跨来源检索证据片段或明确要求该工具时才分配 search_evidence。\n"
    "- 逐条计算分子性质（包括报告非法 SMILES 行）用 compute_properties。"
    "类药性筛选用 filter_druglike。自定义排序、统计或画图才用 run_python，"
    "如果脚本需要新的分子性质数据，同时分配 export_properties。\n"
    "将用户指定的查询参数和输出要求交给 worker。"
    "worker 会读取原始用户消息和对话历史，由它们取得原样参数。无需工具时返回空列表。"
    + DATA_INSTRUCTIONS
)
WORKER_PROMPTS = {
    "lit": (
        "你是 PharmaFlow LitResearcher。完成文献、试验或药品说明书子任务，用工具核实证据。"
        "按子任务使用指定来源的专用工具。跨来源检索证据片段时才用 search_evidence，"
        "将问题改写为英文检索词并引用 citation_id。"
        "下一个请求的参数依赖前一步时，等工具结果返回后取所需字段，再发起下一次调用。"
        "工具成功返回零条时，报告本次检索没有匹配结果，同一个请求只执行一次。"
        "保留用户指定的原样检索式、检索限制和数据口径。引用查到的 PMID、NCT 号和说明书 set_id。"
    ),
    "mol": (
        "你是 PharmaFlow MolAnalyst。完成候选分子类药性子任务。"
        "从用户的原始消息读取 SMILES 或数据集 ID，内置候选集 ID 为 candidates。"
        "性质计算、Lipinski 和 PAINS 筛选使用 chem-mcp。自定义排序、统计或画图才使用 run_python。"
        "用默认条件筛选时省略 thresholds，同一份数据和条件只调用一次 filter_druglike。"
        "按结果中 passes_filter 为 true 的项计数。"
        "需要新的分子性质数据时，先用 export_properties，再把派生数据集 ID 传给 run_python。"
        "历史回答给出本次所需的派生数据集 ID 时可直接复用。"
        "报告被排除的非法 SMILES 行，导出结果的派生数据集 ID 为空时结束该项分析。"
        "脚本读取 /work/input 下的数据，禁止手写 SMILES。图表保存为 /work/output 下的 PNG。"
        "每条回复最多调用一次 run_python。遇到代码错误根据 traceback 修正，超时或内存超限时缩小规模。"
    ),
}
WORKER_COMMON = (
    "用子任务数据块确定工作范围，按用户的原始请求确定精确参数和展示要求。"
    "工具调用结束后直接给出结论正文，保留 Reporter 需要的数字、错误和引用。"
    "对已经确定的事实使用明确的引用格式，例如 PMID 123、NCT 编号、citation_id 或说明书 set_id。"
    + DATA_INSTRUCTIONS
)
REPORTER_PROMPT = (
    "你是 PharmaFlow Reporter。根据本轮用户问题、对话历史和 worker 的结论写中文答复。"
    "遵守用户指定的语言、顺序、字段、数量和格式，保留结论里的数值和实际出处。"
    "只用已核实的出处支持结论。缺结果时说明缺什么，工具出错时说明原始问题。"
    "如果每条输入只需一行，在同一行给出结果或解析错误，每条只列一次。"
    + DATA_INSTRUCTIONS
)


def worker_system_prompt(kind: TaskKind) -> str:
    return WORKER_PROMPTS[kind] + WORKER_COMMON


type ToolCatalog = dict[TaskKind, dict[str, BaseTool]]


def tool_catalog(tools: list[BaseTool]) -> ToolCatalog:
    catalog: ToolCatalog = {"lit": {}, "mol": {}}
    for tool in tools:
        if tool.name == RUN_PYTHON:
            catalog["mol"][tool.name] = tool
            continue
        server = None if tool.metadata is None else tool.metadata.get(MCP_SERVER)
        if server == "lit":
            catalog["lit"][tool.name] = tool
        elif server == "chem":
            catalog["mol"][tool.name] = tool
        else:
            raise ValueError(f"工具 {tool.name!r} 缺少 lit-mcp 或 chem-mcp 来源。")
    return catalog


def planner_system_prompt(catalog: ToolCatalog) -> str:
    return f"{PLANNER_PROMPT}\n\n" + data_block(
        "available_tools",
        {kind: list(available) for kind, available in catalog.items()},
    )


def _question(messages: list[ChatMessage]) -> HumanMessage:
    question = messages[-1]
    assert isinstance(question, HumanMessage)
    return question


def _history_for_workers(state: MultiState) -> DialogueHistory:
    history = state["messages"][:-1]
    summary = state["summary"]
    return {"summary": summary, "messages": retained_messages(history, summary)}


def _feedback_message(feedback: list[CriticIssue]) -> HumanMessage | None:
    if not feedback:
        return None
    parts = ["根据 Critic 的反馈修订本轮子任务结论，继续遵守本轮原始请求。"]
    if any(issue.kind == "missing_result" for issue in feedback):
        parts.append("补充缺失的结论。")
    if any(issue.kind == "unverified_ids" for issue in feedback):
        parts.append("核实或删除下列未在工具结果中出现的 ID。")
    parts.append(
        data_block(
            "critic_feedback",
            [{"subtask_id": issue.subtask_id, "ids": issue.ids} for issue in feedback],
        )
    )
    return HumanMessage(content="\n\n".join(parts))


def _build_worker(
    kind: TaskKind,
    model: BaseChatModel,
    available: dict[str, BaseTool],
    settings: GraphSettings,
    summary_model: BaseChatModel | None,
) -> CompiledStateGraph:
    async def react(state: WorkerState, config: RunnableConfig) -> dict:
        task = state["subtask"]
        tools = [available[name] for name in task.tools]
        model_with_tools = model.bind_tools(tools) if tools else model
        system = "\n\n".join(
            (
                worker_system_prompt(task.kind),
                data_block("subtask", task.model_dump(mode="json")),
            )
        )
        feedback = _feedback_message(state["feedback"])

        def render(
            recent: list[ChatMessage], summary: HistorySummary | None
        ) -> list[BaseMessage]:
            question = recent[0]
            assert isinstance(question, HumanMessage)
            messages = render_prompt(
                system,
                state["history"],
                question,
                recent[1:],
                trim=settings.trim_tool_fields,
                branch_summary=summary,
            )
            if feedback is not None:
                messages.append(feedback)
            return messages

        prepared = await prepare_input(
            state["worker_messages"],
            state["worker_summary"],
            render,
            config=config,
            role="worker",
            tool_schemas=[convert_to_openai_tool(tool) for tool in tools],
            summary_model=summary_model,
            trim=settings.trim_tool_fields,
        )
        reply = await model_with_tools.ainvoke(prepared.messages, prepared.config)
        steps = state["worker_steps"] + 1
        return {
            "worker_messages": react_reply(reply, steps, state["worker_max_steps"]),
            "worker_steps": steps,
            "worker_summary": prepared.summary,
        }

    async def tools(state: WorkerState, config: RunnableConfig) -> dict:
        node = ToolNode([available[name] for name in state["subtask"].tools])
        updates = await execute_tools(
            node, {"messages": state["worker_messages"]}, config
        )
        assert isinstance(updates, dict)
        return {"worker_messages": updates["messages"]}

    def route(state: WorkerState) -> Literal["tools", "finalize"]:
        if state["worker_steps"] < state["worker_max_steps"] and requests_tools(
            state["worker_messages"][-1]
        ):
            return "tools"
        return "finalize"

    def finalize(state: WorkerState) -> dict[str, WorkerResult]:
        messages = state["worker_messages"]
        last = messages[-1]
        conclusion = (
            last.text if isinstance(last, AIMessage) and last.text.strip() else None
        )
        return {
            "result": WorkerResult(
                subtask_id=state["subtask"].id,
                conclusion=conclusion,
                cited_ids=[] if conclusion is None else sorted(cited_ids(conclusion)),
                observed_ids=sorted(observed_ids(messages)),
                messages=messages,
                summary=state["worker_summary"],
            )
        }

    graph = StateGraph(WorkerState, input_schema=WorkerInput)
    graph.add_node("react", react)
    graph.add_node("tools", tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "react")
    graph.add_conditional_edges("react", route)
    graph.add_edge("tools", "react")
    graph.add_edge("finalize", END)
    return graph.compile(name=f"{WORKER_NODES[kind]}ReAct")


def _check_results(state: MultiState) -> dict:
    results = results_by_task(state)
    verified = {
        identifier for result in results.values() for identifier in result.observed_ids
    }
    issues = []
    for task in state["tasks"]:
        result = results.get(task.id)
        if result is None or result.conclusion is None:
            issues.append(
                CriticIssue(subtask_id=task.id, kind="missing_result", ids=[])
            )
        else:
            invalid = sorted(set(result.cited_ids) - verified)
            if invalid:
                issues.append(
                    CriticIssue(subtask_id=task.id, kind="unverified_ids", ids=invalid)
                )
    return {"issues": issues, "critique_passes": state["critique_passes"] + 1}


def _dispatch(state: MultiState, selected: list[Subtask]) -> list[Send]:
    history = _history_for_workers(state)
    current = _question(state["messages"])
    prior_results = results_by_task(state)
    branches = []
    for task in selected:
        prior = prior_results.get(task.id)
        payload: WorkerInput = {
            "subtask": task,
            "history": history,
            "feedback": [
                issue for issue in state["issues"] if issue.subtask_id == task.id
            ],
            "attempt": state["critique_passes"],
            "worker_messages": [current] if prior is None else prior.messages,
            "worker_summary": None if prior is None else prior.summary,
            "worker_steps": 0,
            "worker_max_steps": state["max_steps"],
        }
        branches.append(Send(WORKER_NODES[task.kind], payload))
    return branches


def _after_plan(state: MultiState) -> list[Send] | Literal["Reporter"]:
    return _dispatch(state, state["tasks"]) if state["tasks"] else REPORTER


def _after_critic(state: MultiState) -> list[Send] | Literal["Reporter"]:
    if not state["issues"] or state["critique_passes"] > MAX_CRITIC_RETURNS:
        return REPORTER
    retry = {issue.subtask_id for issue in state["issues"]}
    return _dispatch(state, [task for task in state["tasks"] if task.id in retry])


def _issue_details(issues: list[CriticIssue]) -> list[str]:
    return [
        f"{issue.subtask_id} 缺少结论"
        if issue.kind == "missing_result"
        else f"{issue.subtask_id} 的引用未出现在工具结果中（{'、'.join(issue.ids)}）"
        for issue in issues
    ]


def build_multi_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    settings: GraphSettings,
    summary_model: BaseChatModel | None,
) -> CompiledStateGraph:
    catalog = tool_catalog(tools)
    tool_names = {kind: set(available) for kind, available in catalog.items()}
    schema = convert_to_openai_tool(Plan)
    structured = model.with_structured_output(
        schema, method="function_calling", include_raw=True
    )
    plan_system = planner_system_prompt(catalog)

    async def planner(state: MultiState, config: RunnableConfig) -> dict:
        question = _question(state["messages"])
        assert question.id is not None

        def render(
            history: list[ChatMessage], summary: HistorySummary | None
        ) -> list[BaseMessage]:
            return render_prompt(
                plan_system,
                {"summary": summary, "messages": history},
                question,
                trim=settings.trim_tool_fields,
            )

        prepared = await prepare_input(
            state["messages"][:-1],
            state.get("summary"),
            render,
            config=config,
            role="planner",
            tool_schemas=[schema],
            summary_model=summary_model,
            trim=settings.trim_tool_fields,
        )
        output = await structured.ainvoke(prepared.messages, prepared.config)
        if output["parsing_error"] is not None:
            raise ValueError(
                f"Planner 输出无效：{output['raw']!r}；{output['parsing_error']}"
            ) from output["parsing_error"]
        try:
            plan = Plan.model_validate(
                output["parsed"], context={"tool_names": tool_names}
            )
        except ValidationError as exc:
            raise ValueError(f"Planner 输出无效：{output['raw']!r}；{exc}") from exc
        return {
            "tasks": plan.tasks,
            "step_count": Overwrite(0),
            "evidence_results": Overwrite([]),
            "molecule_results": Overwrite([]),
            "turn_id": question.id,
            "summary": prepared.summary,
            "issues": [],
            "critique_passes": 0,
            "reporter_unverified_ids": [],
        }

    def worker_node(kind: TaskKind, result_field: str):
        subgraph = _build_worker(kind, model, catalog[kind], settings, summary_model)

        async def run_worker(state: WorkerInput, config: RunnableConfig) -> dict:
            output = await subgraph.ainvoke(
                state,
                {
                    **config,
                    "recursion_limit": react_recursion_limit(state["worker_max_steps"]),
                    "metadata": {
                        **config.get("metadata", {}),
                        "subtask_id": state["subtask"].id,
                        "subtask_attempt": state["attempt"],
                    },
                },
            )
            return {
                result_field: [output["result"]],
                "step_count": output["worker_steps"],
            }

        return run_worker

    async def reporter(state: MultiState, config: RunnableConfig) -> dict:
        results = ordered_results(state)
        reports = []
        for result in results:
            report = result.model_dump(
                mode="json",
                include={"subtask_id", "conclusion", "cited_ids", "observed_ids"},
            )
            if not settings.structured_worker_results:
                report["messages"] = [
                    message_data(message, trim=settings.trim_tool_fields)
                    for message in result.messages
                ]
            reports.append(report)
        system = "\n\n".join(
            (
                REPORTER_PROMPT,
                data_block("worker_results", reports),
                data_block(
                    "critic_issues",
                    [issue.model_dump(mode="json") for issue in state["issues"]],
                ),
            )
        )
        inputs = render_prompt(
            system,
            _history_for_workers(state),
            _question(state["messages"]),
            trim=settings.trim_tool_fields,
        )
        reply = await model.ainvoke(inputs, model_config(config, inputs, "reporter"))
        verified = {
            identifier for result in results for identifier in result.observed_ids
        }
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                verified.update(cited_ids(message.text))
        unverified = sorted(cited_ids(reply.text) - verified)
        sections = [reply.text]
        if state["issues"]:
            sections.append(
                "仍未通过核查的子任务："
                + "；".join(_issue_details(state["issues"]))
                + "。"
            )
        if unverified:
            sections.append("以下引用未核验：" + "、".join(unverified) + "。")
        answer = reply.model_copy(update={"content": "\n\n".join(sections)})
        return {"messages": [answer], "reporter_unverified_ids": unverified}

    graph = StateGraph(MultiState)
    graph.add_node(PLANNER, planner)
    graph.add_node(
        LIT_RESEARCHER, worker_node("lit", "evidence_results"), input_schema=WorkerInput
    )
    graph.add_node(
        MOL_ANALYST, worker_node("mol", "molecule_results"), input_schema=WorkerInput
    )
    graph.add_node(CRITIC, _check_results)
    graph.add_node(REPORTER, reporter)
    graph.add_edge(START, PLANNER)
    graph.add_conditional_edges(
        PLANNER, _after_plan, [LIT_RESEARCHER, MOL_ANALYST, REPORTER]
    )
    graph.add_edge(LIT_RESEARCHER, CRITIC)
    graph.add_edge(MOL_ANALYST, CRITIC)
    graph.add_conditional_edges(
        CRITIC, _after_critic, [LIT_RESEARCHER, MOL_ANALYST, REPORTER]
    )
    graph.add_edge(REPORTER, END)
    return graph.compile(checkpointer=checkpointer, name=MULTI_GRAPH_NAME)
