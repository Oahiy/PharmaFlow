"""Measure Docker sandbox stage timings over five empty scripts."""

import logging
import re
from decimal import Decimal
from io import StringIO
from statistics import median

from sandbox import executor
from sandbox.models import RunPythonRequest

SAMPLE_COUNT = 5
STAGES = {
    "prepare_input",
    "create",
    "transfer_input",
    "execute",
    "collect_output",
    "remove",
}
STAGE_LOG = re.compile(
    r"^sandbox stage=(\w+) elapsed_ms=(\d+\.\d+) tool_call_id=(\S+)$", re.MULTILINE
)


def main() -> None:
    request = RunPythonRequest(code="pass", dataset_ids=[])
    logs = StringIO()
    handler = logging.StreamHandler(logs)
    logger = executor.logger
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    lifecycle_samples: list[Decimal] = []
    stage_samples: list[Decimal] = []
    print(f"{executor.IMAGE}，空脚本 pass，{SAMPLE_COUNT} 次")
    print("| 样本 | 创建 (ms) | 删除 (ms) | 创建＋删除 (ms) | 六阶段合计 (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    try:
        for sample in range(1, SAMPLE_COUNT + 1):
            call_id = f"desktop-benchmark-{sample}"
            logs.seek(0)
            logs.truncate()
            result = executor.execute_python(request, call_id)
            if result.error is not None:
                raise SystemExit(
                    f"{call_id} 执行失败\n{result.error.message}\n{result.stderr}"
                )

            readings = [
                (stage, Decimal(elapsed))
                for stage, elapsed, logged_id in STAGE_LOG.findall(logs.getvalue())
                if logged_id == call_id
            ]
            stages = dict(readings)
            if len(readings) != len(STAGES) or stages.keys() != STAGES:
                raise SystemExit(f"{call_id} 的阶段日志不完整\n{logs.getvalue()}")
            lifecycle_ms = stages["create"] + stages["remove"]
            stages_ms = sum(stages.values(), start=Decimal(0))
            lifecycle_samples.append(lifecycle_ms)
            stage_samples.append(stages_ms)
            print(
                f"| {sample} | {stages['create']:.1f} | {stages['remove']:.1f} "
                f"| {lifecycle_ms:.1f} | {stages_ms:.1f} |"
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    print(f"\n创建＋删除中位数 {median(lifecycle_samples):.1f} ms")
    print(f"六阶段合计中位数 {median(stage_samples):.1f} ms")


if __name__ == "__main__":
    main()
