from collections import Counter

import pytest
import yaml

from eval.cases import CASE_DIRECTORY, EvalCase, load_cases


def test_load_all_baseline_cases():
    cases = load_cases().cases
    assert {case.id for case in cases} == {
        *(f"S{i:02}" for i in range(1, 7)),
        *(f"M{i:02}" for i in range(1, 4)),
        *(f"E{i:02}" for i in range(1, 4)),
    }
    assert Counter(case.category for case in cases) == {
        "single_tool": 6,
        "multi_hop": 3,
        "exception": 3,
    }


@pytest.mark.parametrize(
    ("names", "error"),
    [
        (["other.yaml"], "不一致"),
        (["S01.yaml", "duplicate.yaml"], "重复"),
    ],
)
def test_case_ids_must_be_unique_and_match_file_names(tmp_path, names, error):
    content = (CASE_DIRECTORY / "S01.yaml").read_text(encoding="utf-8")
    for name in names:
        (tmp_path / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_cases(tmp_path)


def test_case_rejects_a_reference_to_a_missing_dependency():
    case = yaml.safe_load((CASE_DIRECTORY / "M01.yaml").read_text(encoding="utf-8"))
    case["accepted_paths"][0][1]["from_results"]["nct_id"]["step"] = "missing-step"
    with pytest.raises(ValueError, match="前置步骤"):
        EvalCase.model_validate(case)


def test_case_digest_includes_raw_yaml_content(tmp_path):
    content = (CASE_DIRECTORY / "S01.yaml").read_text(encoding="utf-8")
    path = tmp_path / "S01.yaml"
    path.write_text(content, encoding="utf-8")
    first = load_cases(tmp_path)
    path.write_text(content + "\n# updated comment\n", encoding="utf-8")
    edited = load_cases(tmp_path)
    assert edited.cases == first.cases
    assert edited.sha256 != first.sha256
