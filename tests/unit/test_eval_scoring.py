import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import STEP_LIMIT_MESSAGE
from eval.cases import load_cases
from eval.scoring import score_case


@pytest.fixture(scope="module")
def cases():
    return {case.id: case for case in load_cases().cases}


def call(name, request, id="call"):
    return {"name": name, "id": id, "args": {"request": request}}


def tool_result(payload, id="call"):
    return ToolMessage(
        content="结果", artifact={"structured_content": payload}, tool_call_id=id
    )


def tool_error(text, id="call"):
    return ToolMessage(
        content=[{"type": "text", "text": text}], status="error", tool_call_id=id
    )


def test_replay_miss_uses_error_message_text_and_request_constraints(cases):
    case = cases["S05"]
    error = tool_error(
        "Error calling tool 'get_trial_adverse_events': 回放未命中: GET https://example.invalid"
    )
    final = AIMessage(content="没有获得结果。")
    step = case.accepted_paths[0][0]
    for nct, failure, valid in (
        ("NCT04303780", "replay_miss", True),
        ("NCT12345678", "invalid_args", False),
    ):
        args = {**step.request, "nct_id": nct}
        result = score_case(
            case,
            [AIMessage(content="", tool_calls=[call(step.tool, args)]), error, final],
        )
        assert result.failure_type == failure
        assert result.params_valid is valid
        assert result.tools_correct
        assert "回放未命中:" in result.error_detail["calls"][0]["tool_message"]


def test_successful_zero_hits_are_evidence_for_judge(cases):
    case = cases["E03"]
    search, label = case.accepted_paths[0]
    empty = {"total_count": 0, "returned_count": 0, "results": []}
    result = score_case(
        case,
        [
            AIMessage(
                content="",
                tool_calls=[
                    call(label.tool, label.request, "label"),
                    call(search.tool, search.request),
                ],
            ),
            tool_result(empty, "label"),
            tool_result(empty),
            AIMessage(content="本次 PubMed 未找到文献，openFDA 未找到匹配说明书。"),
        ],
    )
    assert result.failure_type is None
    assert result.tools_correct and result.params_valid
    assert result.evidence == {"articles": empty, "label": empty}


def test_invalid_smiles_is_a_successful_row_and_answer_affects_scoring(cases):
    case = cases["E01"]
    step = case.accepted_paths[0][0]
    payload = {
        "results": [
            {"smiles": "CCO", "mw": 46.069},
            {"smiles": "not-a-smiles", "error": "invalid_smiles"},
            {"smiles": "CC", "mw": 30.07},
        ]
    }
    messages = [
        AIMessage(content="", tool_calls=[call(step.tool, step.request)]),
        tool_result(payload),
    ]
    result = score_case(
        case,
        [
            *messages,
            AIMessage(
                content="| CCO | 46.07 Da |\n| not-a-smiles | 无法解析 |\n| CC | 30.07 Da |"
            ),
        ],
    )
    assert result.failure_type is None
    assert result.params_valid is True

    result = score_case(
        case,
        [
            *messages,
            AIMessage(
                content="CCO 46.07 Da\nnot-a-smiles 无法解析，35 Da\nCC 30.07 Da"
            ),
        ],
    )
    assert result.failure_type == "rule_failed"
    assert result.error_detail["rules"]

    result = score_case(
        case,
        [
            *messages,
            AIMessage(
                content=(
                    "| CCO | 46.069 Da |\n| not-a-smiles | 无法解析 |\n| CC | 30.07 Da |"
                    "\n\n说明\n- **CCO** 为 46.069 Da。"
                    "\n- **not-a-smiles** 无法解析。"
                    "\n- **CC** 为 30.07 Da。"
                ),
            ),
        ],
    )
    assert result.failure_type == "rule_failed"
    (failed_rule,) = result.error_detail["rules"]
    assert failed_rule["rule"]["type"] == "answer_sequence"


@pytest.mark.parametrize("uses_tool", [False, True])
def test_nct_format_accepts_both_declared_paths(cases, uses_tool):
    case = cases["E02"]
    messages = []
    if uses_tool:
        step = case.accepted_paths[1][0]
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        call(step.tool, {"nct_id": "NCT123", "category": "serious"})
                    ],
                ),
                tool_error(
                    "request.nct_id\nString should match pattern [type=string_pattern_mismatch]"
                ),
            ]
        )
    messages.append(AIMessage(content="编号格式错误。请提供 NCT 加 8 位数字的编号。"))
    result = score_case(case, messages)
    assert result.failure_type is None
    assert result.tools_correct
    assert result.params_valid is (True if uses_tool else None)


@pytest.mark.parametrize(
    ("answer", "passes"),
    [
        (
            (
                "前两项 NCT04303780、NCT04685135\n"
                "完整列表 NCT04303780、NCT04685135、NCT06119581、NCT07190248、NCT06881784"
            ),
            True,
        ),
        ("前两项 NCT04685135、NCT04303780", False),
        ("第一项 NCT04303780", False),
    ],
)
def test_trial_rank_checks_the_first_two_ids(cases, answer, passes):
    case = cases["S04"]
    step = case.accepted_paths[0][0]
    result = score_case(
        case,
        [
            AIMessage(content="", tool_calls=[call(step.tool, step.request)]),
            tool_result(
                {
                    "total_count": 25,
                    "returned_count": 5,
                    "results": [
                        {"nct_id": "NCT04303780"},
                        {"nct_id": "NCT04685135"},
                    ],
                }
            ),
            AIMessage(content=f"总命中数 25，本次返回数 5。\n{answer}"),
        ],
    )
    assert result.failure_type == (None if passes else "rule_failed")


@pytest.mark.parametrize(
    ("warning_line", "passes"),
    [
        ("✅ **有此章节**（status: present）", True),
        ("✅ 存在（present） | 内容未完整返回", True),
        ("✅ 存在（present） | 内容完整返回", False),
        ("✅ 存在（present） | 内容没有截断", False),
        ("✅ present | Content was not returned in full", True),
        ("✅ present | Content was returned in full", False),
        ("不存在", False),
        ("没有这个章节", False),
    ],
)
def test_label_accepts_formatted_status_and_checks_truncation(
    cases, warning_line, passes
):
    case = cases["S06"]
    step = case.accepted_paths[0][0]
    set_id = "c80a362c-7ac3-4894-a076-0691e68ef8c1"
    sections = [
        {
            "field": "warnings_and_cautions",
            "status": "present",
            "content": {"results": [{"truncated": True}]},
        },
        {"field": "boxed_warning", "status": "missing", "content": None},
    ]
    result = score_case(
        case,
        [
            AIMessage(content="", tool_calls=[call(step.tool, step.request)]),
            tool_result(
                {
                    "total_count": 1,
                    "returned_count": 1,
                    "results": [
                        {
                            "set_id": set_id,
                            "effective_time": "20261103",
                            "sections": {"results": sections},
                        }
                    ],
                }
            ),
            AIMessage(
                content=(
                    f"set ID {set_id}，生效日期 2026-11-03。\n"
                    "| **boxed_warning（黑框警告）** | ❌ 缺失（missing） |\n"
                    f"| **warnings_and_cautions（警告与注意事项）** | {warning_line} |\n"
                    "返回的说明书章节内容已截断。"
                )
            ),
        ],
    )
    assert result.failure_type == (None if passes else "rule_failed")
    if not passes:
        assert len(result.error_detail["rules"]) == 1


def test_hop_must_use_a_prior_result_and_not_a_parallel_guess(cases):
    case = cases["M02"]
    search, label = case.accepted_paths[0]
    search_payload = {
        "results": [{}] * 4 + [{"pmid": "37956237", "title": "Adagrasib"}]
    }
    label_payload = {
        "results": [
            {
                "set_id": "0b8bf078-34c2-4f45-9012-38a8ac082b01",
                "sections": {
                    "results": [{"field": "warnings_and_cautions", "status": "present"}]
                },
            }
        ]
    }
    final = AIMessage(
        content="PMID 37956237，set ID 0b8bf078-34c2-4f45-9012-38a8ac082b01。"
    )
    search_call = call(search.tool, search.request)
    label_call = call(label.tool, label.request)
    serial = score_case(
        case,
        [
            AIMessage(content="", tool_calls=[search_call]),
            tool_result(search_payload),
            AIMessage(content="", tool_calls=[label_call]),
            tool_result(label_payload),
            final,
        ],
    )
    assert serial.failure_type is None
    assert serial.params_valid and serial.tools_correct

    label_call["id"] = "label"
    parallel = score_case(
        case,
        [
            AIMessage(content="", tool_calls=[search_call, label_call]),
            tool_result(search_payload),
            tool_result(label_payload, "label"),
            final,
        ],
    )
    assert parallel.failure_type == "wrong_tool"
    assert parallel.params_valid and not parallel.tools_correct


def test_failure_priority_reports_qualified_miss_then_model_faults(cases):
    case = cases["E03"]
    search, label = case.accepted_paths[0]
    model_error = {
        "type": "RuntimeError",
        "message": "model failed",
        "traceback": "trace",
    }
    wrong_label = {**label.request, "drug": "another-drug"}
    request = AIMessage(
        content="",
        tool_calls=[
            call(search.tool, search.request, "search"),
            call(label.tool, wrong_label, "label"),
        ],
    )
    result = score_case(
        case,
        [
            request,
            tool_error("前缀：回放未命中: GET …", "search"),
            tool_error("回放未命中: GET …", "label"),
        ],
        model_error,
    )
    assert result.failure_type == "replay_miss"
    assert not result.params_valid
    assert result.error_detail["exception"] == model_error

    result = score_case(
        case,
        [
            request,
            tool_result({"total_count": 0}, "search"),
            tool_error("回放未命中: GET …", "label"),
        ],
        model_error,
    )
    assert result.failure_type == "invalid_args"

    result = score_case(
        case,
        [
            AIMessage(content="", tool_calls=[request.tool_calls[0]]),
            tool_error(STEP_LIMIT_MESSAGE, "search"),
        ],
        model_error,
    )
    assert result.failure_type == "agent_error"
    assert result.error_detail["calls"][0]["tool_message"] == STEP_LIMIT_MESSAGE


def test_unknown_tool_miss_does_not_exclude_a_case(cases):
    result = score_case(
        cases["S01"],
        [
            AIMessage(
                content="", tool_calls=[call("search_pubmed", {"query": "unrecorded"})]
            ),
            tool_error("前缀：回放未命中: GET …"),
            AIMessage(content="无输出。"),
        ],
    )
    assert result.failure_type == "wrong_tool"
    assert result.params_valid is None


def test_declared_tool_with_unparseable_arguments_is_invalid(cases):
    result = score_case(
        cases["S01"],
        [
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "compute_properties",
                        "id": "call",
                        "args": '{"request":',
                        "error": "invalid JSON",
                    }
                ],
            ),
            tool_error("invalid JSON"),
            AIMessage(content="无输出。"),
        ],
    )
    assert result.failure_type == "invalid_args"
    assert result.params_valid is False
