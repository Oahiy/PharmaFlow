"""Run a single Python script in a disposable Docker container."""

import logging
import os
import tarfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from tempfile import TemporaryFile
from time import perf_counter
from typing import BinaryIO

import docker
from docker.errors import APIError, DockerException
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError

from mcp_servers.chem.datasets import resolve_dataset

from .artifacts import OutputArchiveError, read_output_archive
from .models import RunPythonRequest, SandboxError

logger = logging.getLogger(__name__)
IMAGE = "pharmaflow-sandbox:1"
MEMORY_BYTES = 512 * 1024 * 1024
CPU_PERIOD = CPU_QUOTA = 100_000
PIDS_LIMIT = 64
TIMEOUT_SECONDS = 15
STDOUT_BYTES = 16 * 1024
STDERR_BYTES = 8 * 1024


@dataclass(frozen=True, slots=True)
class PythonExecution:
    stdout: str
    stderr: str
    exit_code: int | None
    error: SandboxError | None = None
    files: dict[str, bytes] = field(default_factory=dict)


type PythonExecutor = Callable[[RunPythonRequest, str], PythonExecution]


class DatasetInputError(ValueError):
    """A requested dataset cannot be resolved for this execution."""


def write_input_archive(request: RunPythonRequest, output: BinaryIO) -> None:
    with tarfile.open(fileobj=output, mode="w") as tar:
        code = request.code.encode("utf-8")
        script = tarfile.TarInfo("main.py")
        script.mode = 0o444
        script.size = len(code)
        tar.addfile(script, BytesIO(code))
        for dataset_id in dict.fromkeys(request.dataset_ids):
            try:
                path = resolve_dataset(dataset_id)
            except ValueError as exc:
                raise DatasetInputError(str(exc)) from exc
            try:
                with path.open("rb") as source:
                    entry = tarfile.TarInfo(f"input/{dataset_id}.csv")
                    entry.mode = 0o444
                    entry.size = os.fstat(source.fileno()).st_size
                    tar.addfile(entry, source)
            except FileNotFoundError as exc:
                raise DatasetInputError(
                    f"数据集不存在：{dataset_id}（{exc.strerror}）"
                ) from exc
    output.seek(0)


def trim_log(chunks: Iterable[bytes], limit: int, *, keep_tail: bool) -> str:
    kept = bytearray()
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if keep_tail:
            kept.extend(chunk[-limit:])
            del kept[:-limit]
        else:
            kept.extend(chunk[: limit - len(kept)])
    text = kept.decode("utf-8", errors="replace")
    omitted = total - len(kept)
    if omitted == 0:
        return text
    marker = f"[已截断 {omitted} 字节]"
    return f"{marker}\n{text}" if keep_tail else f"{text}\n{marker}"


@contextmanager
def timed(stage: str, tool_call_id: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        logger.info(
            "sandbox stage=%s elapsed_ms=%.1f tool_call_id=%s",
            stage,
            (perf_counter() - start) * 1000,
            tool_call_id,
        )


def execute_python(request: RunPythonRequest, tool_call_id: str) -> PythonExecution:
    with ExitStack() as resources:
        incoming = resources.enter_context(TemporaryFile())
        with timed("prepare_input", tool_call_id):
            write_input_archive(request, incoming)
        api = resources.enter_context(closing(docker.from_env())).api
        with timed("create", tool_call_id):
            # Retain the ID directly from create so cleanup does not depend on inspect.
            container_id = api.create_container(
                IMAGE,
                user="10001:10001",
                use_config_proxy=False,
                host_config=api.create_host_config(
                    network_mode="none",
                    mem_limit=MEMORY_BYTES,
                    memswap_limit=MEMORY_BYTES,
                    cpu_period=CPU_PERIOD,
                    cpu_quota=CPU_QUOTA,
                    pids_limit=PIDS_LIMIT,
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges"],
                ),
            )["Id"]
        try:
            with timed("transfer_input", tool_call_id):
                if not api.put_archive(container_id, "/work", incoming):
                    raise DockerException("Docker put_archive 未确认写入。")
            with timed("execute", tool_call_id):
                error = None
                exit_code = None
                api.start(container_id)
                try:
                    status = api.wait(container_id, timeout=TIMEOUT_SECONDS)
                except (ReadTimeout, RequestsConnectionError) as wait_error:
                    # Requests wraps read timeouts in ConnectionError while consuming the body.
                    if isinstance(wait_error, RequestsConnectionError) and not (
                        len(wait_error.args) == 1
                        and isinstance(wait_error.args[0], ReadTimeoutError)
                    ):
                        raise
                    try:
                        api.kill(container_id)
                    except APIError as exc:
                        # The script can exit between the wait deadline and kill.
                        if (
                            exc.status_code != 409
                            or api.inspect_container(container_id)["State"]["Status"]
                            != "exited"
                        ):
                            raise
                    error = SandboxError(
                        kind="timeout",
                        message=f"脚本执行超过 {TIMEOUT_SECONDS} 秒。请缩小数据或计算规模后再提交脚本。",
                    )
                else:
                    if status.get("Error") is not None:
                        raise DockerException(f"Docker wait 失败：{status['Error']}")
                    exit_code = status["StatusCode"]
                    if api.inspect_container(container_id)["State"]["OOMKilled"]:
                        error = SandboxError(
                            kind="oom",
                            message="容器超过 512 MiB 内存限制。请缩小数据规模或分批处理后再提交脚本。",
                        )
                    elif exit_code != 0:
                        error = SandboxError(
                            kind="code",
                            message=f"脚本退出码为 {exit_code}。请根据 stderr 的 traceback 修正脚本。",
                        )
            with timed("collect_output", tool_call_id):
                with closing(
                    api.logs(
                        container_id,
                        stdout=True,
                        stderr=False,
                        stream=True,
                        follow=False,
                    )
                ) as logs:
                    stdout = trim_log(logs, STDOUT_BYTES, keep_tail=False)
                with closing(
                    api.logs(
                        container_id,
                        stdout=False,
                        stderr=True,
                        stream=True,
                        follow=False,
                    )
                ) as logs:
                    stderr = trim_log(logs, STDERR_BYTES, keep_tail=True)
                files = {}
                if error is None:
                    chunks, _ = api.get_archive(
                        container_id, "/work/output", chunk_size=64 * 1024
                    )
                    try:
                        with closing(chunks):
                            files = read_output_archive(chunks)
                    except OutputArchiveError as exc:
                        error = SandboxError(kind="output", message=str(exc))
            return PythonExecution(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                error=error,
                files=files,
            )
        finally:
            with timed("remove", tool_call_id):
                api.remove_container(container_id, force=True)
