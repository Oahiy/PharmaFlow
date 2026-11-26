import ast
import logging
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from docker.errors import APIError, DockerException
from pydantic import ValidationError
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ReadTimeoutError

from sandbox import artifacts, executor
from sandbox.artifacts import OutputArchiveError, read_output_archive
from sandbox.executor import (
    DatasetInputError,
    execute_python,
    trim_log,
    write_input_archive,
)
from sandbox.models import RunPythonRequest


def test_docker_test_scripts_compile_without_starting_containers():
    path = Path(__file__).resolve().parents[1] / "docker" / "test_sandbox_container.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "run"
        ):
            compile(ast.literal_eval(call.args[0]), f"{path}:{call.lineno}", "exec")


def test_missing_dataset_precedes_docker_connection(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    from_env = Mock(side_effect=DockerException("daemon unavailable"))
    monkeypatch.setattr(executor.docker, "from_env", from_env)
    with pytest.raises(DatasetInputError, match="upload_missing"):
        execute_python(
            RunPythonRequest(code="pass", dataset_ids=["upload_missing"]), "missing"
        )
    from_env.assert_not_called()


def output_tar(*entries):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        root = tarfile.TarInfo("output")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, kind, data in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "/work/input/candidates.csv"
            archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def fake_docker(monkeypatch):
    api = Mock()
    api.create_container.return_value = {"Id": "abc"}
    api.put_archive.return_value = True
    api.wait.return_value = {"StatusCode": 0}
    api.inspect_container.return_value = {"State": {"OOMKilled": False}}
    api.logs.side_effect = lambda *args, **kwargs: (chunk for chunk in ())
    api.get_archive.side_effect = lambda *args, **kwargs: (
        (chunk for chunk in [output_tar()]),
        {},
    )
    client = Mock(api=api)
    monkeypatch.setattr(executor.docker, "from_env", lambda: client)
    return api


def test_streaming_logs_keep_stdout_head_and_stderr_tail():
    trace = b"Traceback (most recent call last):\nKeyError: 'QED'\n"
    stderr = trim_log([b"x" * 96, trace], 64, keep_tail=True)
    assert stderr.startswith(f"[已截断 {96 + len(trace) - 64} 字节]\n")
    assert stderr.endswith(trace.decode())
    assert trim_log([b"123", b"456"], 4, keep_tail=False) == "1234\n[已截断 2 字节]"
    assert trim_log([b"123"], 4, keep_tail=True) == "123"


def test_archive_accepts_only_named_csv_inputs_and_python_code(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    source = b"name,smiles\nethanol,CCO\n"
    (tmp_path / "upload_test.csv").write_bytes(source)
    request = RunPythonRequest(
        code="print(1)", dataset_ids=["upload_test", "upload_test"]
    )
    output = BytesIO()
    write_input_archive(request, output)
    with tarfile.open(fileobj=output) as archive:
        assert archive.getnames() == ["main.py", "input/upload_test.csv"]
        assert archive.extractfile("main.py").read() == b"print(1)"
        assert archive.extractfile("input/upload_test.csv").read() == source
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        RunPythonRequest(code="print(1)", dataset_ids=["../outside"])
    with pytest.raises(DatasetInputError, match="upload_missing"):
        write_input_archive(
            RunPythonRequest(code="print(1)", dataset_ids=["upload_missing"]), BytesIO()
        )
    outside = tmp_path / "nested"
    outside.mkdir()
    (outside / "real.csv").write_bytes(source)
    (tmp_path / "upload_link.csv").symlink_to(outside / "real.csv")
    with pytest.raises(DatasetInputError, match="must resolve within"):
        write_input_archive(
            RunPythonRequest(code="print(1)", dataset_ids=["upload_link"]), BytesIO()
        )


def test_reads_png_archive_and_empty_output(png_bytes):
    body = output_tar(("output/chart.png", tarfile.REGTYPE, png_bytes))
    assert read_output_archive([body[:513], body[513:]]) == {"chart.png": png_bytes}
    assert read_output_archive([output_tar()]) == {}


@pytest.mark.parametrize(
    ("name", "kind", "message"),
    [
        ("/output/chart.png", tarfile.REGTYPE, "路径无效"),
        ("output/../chart.png", tarfile.REGTYPE, "路径无效"),
        ("output/sub/chart.png", tarfile.REGTYPE, "路径无效"),
        ("output/..\\chart.png", tarfile.REGTYPE, "路径无效"),
        ("output/chart.png", tarfile.SYMTYPE, "普通文件"),
        ("output/chart.png", tarfile.LNKTYPE, "普通文件"),
        ("output/chart.png", tarfile.FIFOTYPE, "普通文件"),
        ("output/chart.csv", tarfile.REGTYPE, "只支持 PNG"),
    ],
)
def test_rejects_unsafe_paths_and_member_types(name, kind, message):
    with pytest.raises(OutputArchiveError, match=message):
        read_output_archive([output_tar((name, kind, b""))])


def test_rejects_invalid_png_and_duplicate_names(png_bytes):
    with pytest.raises(OutputArchiveError, match="无效的 PNG"):
        read_output_archive([output_tar(("output/chart.png", tarfile.REGTYPE, b"bad"))])
    chart = ("output/chart.png", tarfile.REGTYPE, png_bytes)
    with pytest.raises(OutputArchiveError, match="同名文件"):
        read_output_archive([output_tar(chart, chart)])


def test_checks_archive_limits_before_copying_files(png_bytes, monkeypatch):
    chart = ("output/chart.png", tarfile.REGTYPE, png_bytes)
    second = ("output/second.png", tarfile.REGTYPE, png_bytes)
    body = output_tar(chart, second)
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_FILE_BYTES", len(png_bytes) - 1)
        with pytest.raises(OutputArchiveError, match="文件大小"):
            read_output_archive([body])
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_TOTAL_BYTES", len(png_bytes))
        with pytest.raises(OutputArchiveError, match="文件合计"):
            read_output_archive([body])
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_FILES", 1)
        with pytest.raises(OutputArchiveError, match="文件数"):
            read_output_archive([body])
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_ARCHIVE_BYTES", len(body) - 1)
        with pytest.raises(OutputArchiveError, match="传输字节数"):
            read_output_archive([body[:512], body[512:]])


def test_docker_lifecycle_and_stage_logging(fake_docker, caplog):
    with caplog.at_level(logging.INFO, logger=executor.__name__):
        result = execute_python(
            RunPythonRequest(code="pass", dataset_ids=[]), "tool-id"
        )
    assert result.error is None
    assert [call[0] for call in fake_docker.mock_calls] == [
        "create_host_config",
        "create_container",
        "put_archive",
        "start",
        "wait",
        "inspect_container",
        "logs",
        "logs",
        "get_archive",
        "remove_container",
    ]
    fake_docker.create_host_config.assert_called_once_with(
        network_mode="none",
        mem_limit=512 * 1024 * 1024,
        memswap_limit=512 * 1024 * 1024,
        cpu_period=100_000,
        cpu_quota=100_000,
        pids_limit=64,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
    )
    assert fake_docker.create_container.call_args.args == ("pharmaflow-sandbox:1",)
    assert fake_docker.create_container.call_args.kwargs["user"] == "10001:10001"
    assert fake_docker.create_container.call_args.kwargs["use_config_proxy"] is False
    fake_docker.wait.assert_called_once_with("abc", timeout=15)
    for name in ("create", "transfer_input", "execute", "collect_output", "remove"):
        assert any(
            f"stage={name} elapsed_ms=" in rec.message
            and "tool_call_id=tool-id" in rec.message
            for rec in caplog.records
        )


@pytest.mark.parametrize("kind", ["timeout", "oom", "code"])
def test_docker_classifies_script_failure_and_cleans_up(fake_docker, kind):
    fake_docker.wait.return_value = {"StatusCode": 137}
    if kind == "timeout":
        fake_docker.wait.side_effect = ReadTimeout("fake wait timeout")
    if kind == "oom":
        fake_docker.inspect_container.return_value = {"State": {"OOMKilled": True}}
    result = execute_python(RunPythonRequest(code="pass", dataset_ids=[]), "tool-id")
    assert result.error.kind == kind
    assert fake_docker.kill.called is (kind == "timeout")
    fake_docker.get_archive.assert_not_called()
    fake_docker.remove_container.assert_called_once_with("abc", force=True)


def test_timeout_reading_wait_response_body_stops_container(fake_docker):
    response = Response()
    response.status_code = 200
    response.raw = Mock()
    response.raw.stream.side_effect = ReadTimeoutError(
        None, "/containers/abc/wait", "Read timed out."
    )
    fake_docker.wait.side_effect = lambda *args, **kwargs: response.json()

    result = execute_python(
        RunPythonRequest(code="pass", dataset_ids=[]), "body-timeout"
    )

    assert result.error.kind == "timeout"
    assert "缩小" in result.error.message
    fake_docker.kill.assert_called_once_with("abc")
    fake_docker.remove_container.assert_called_once_with("abc", force=True)


def test_wait_connection_failure_propagates_after_removal(fake_docker):
    failure = RequestsConnectionError(ConnectionResetError("connection reset by peer"))
    fake_docker.wait.side_effect = failure

    with pytest.raises(RequestsConnectionError) as exc:
        execute_python(RunPythonRequest(code="pass", dataset_ids=[]), "disconnected")

    assert exc.value is failure
    fake_docker.kill.assert_not_called()
    fake_docker.remove_container.assert_called_once_with("abc", force=True)


@pytest.mark.parametrize("stage", ["put_archive", "start", "wait", "get_archive"])
def test_docker_errors_propagate_after_removal(fake_docker, stage):
    failure = APIError("original Docker error")
    getattr(fake_docker, stage).side_effect = failure
    with pytest.raises(DockerException) as exc:
        execute_python(RunPythonRequest(code="pass", dataset_ids=[]), "tool-id")
    assert exc.value is failure
    fake_docker.remove_container.assert_called_once_with("abc", force=True)


@pytest.mark.parametrize(
    ("status_code", "state", "is_timeout"),
    [(409, "exited", True), (409, "running", False), (500, "exited", False)],
)
def test_exit_race_during_timeout_kill(fake_docker, status_code, state, is_timeout):
    response = Response()
    response.status_code = status_code
    failure = APIError("kill failed", response=response)
    fake_docker.wait.side_effect = ReadTimeout("deadline")
    fake_docker.kill.side_effect = failure
    fake_docker.inspect_container.return_value = {"State": {"Status": state}}
    fake_docker.logs.side_effect = lambda *args, **kwargs: (s for s in [b"last line\n"])
    if is_timeout:
        result = execute_python(RunPythonRequest(code="pass", dataset_ids=[]), "race")
        assert result.error.kind == "timeout"
        assert result.stdout == result.stderr == "last line\n"
        assert fake_docker.mock_calls[-1][0] == "remove_container"
    else:
        with pytest.raises(APIError) as exc:
            execute_python(RunPythonRequest(code="pass", dataset_ids=[]), "race")
        assert exc.value is failure
    fake_docker.kill.assert_called_once_with("abc")
    fake_docker.remove_container.assert_called_once_with("abc", force=True)
