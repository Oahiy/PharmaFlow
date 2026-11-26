from contextlib import closing
from time import perf_counter
from uuid import uuid4

import docker
import pytest
from docker.errors import NotFound

from sandbox.executor import MEMORY_BYTES, PIDS_LIMIT, TIMEOUT_SECONDS, execute_python
from sandbox.models import RunPythonRequest

pytestmark = pytest.mark.docker


@pytest.fixture
def sandbox_run(monkeypatch):
    created = []
    inspected = []
    create = docker.APIClient.create_container
    inspect = docker.APIClient.inspect_container

    def record_creation(client, *args, **kwargs):
        result = create(client, *args, **kwargs)
        created.append(result["Id"])
        return result

    def record_inspect(client, *args, **kwargs):
        result = inspect(client, *args, **kwargs)
        inspected.append(result)
        return result

    monkeypatch.setattr(docker.APIClient, "create_container", record_creation)
    monkeypatch.setattr(docker.APIClient, "inspect_container", record_inspect)

    def run(code, dataset_ids=()):
        assert not created
        try:
            return execute_python(
                RunPythonRequest(code=code, dataset_ids=list(dataset_ids)), uuid4().hex
            )
        finally:
            if created:
                (container_id,) = created
                with closing(docker.from_env()) as client, pytest.raises(NotFound):
                    client.api.inspect_container(container_id)

    return run, inspected


def test_external_network_is_unreachable(sandbox_run):
    run, inspected = sandbox_run
    result = run("""\
import errno
import socket
try:
    with socket.create_connection(("1.1.1.1", 443), timeout=2):
        raise AssertionError("network unexpectedly reachable")
except OSError as exc:
    print(errno.errorcode[exc.errno])
""")
    assert result.error is None
    assert result.stdout.strip() == "ENETUNREACH"
    assert inspected[0]["HostConfig"]["NetworkMode"] == "none"


def test_memory_limit_disables_swap_and_reports_oom_killed(sandbox_run):
    run, inspected = sandbox_run
    result = run("""\
from pathlib import Path

swap = Path("/sys/fs/cgroup/memory.swap.max")
if swap.exists():
    assert swap.read_text().strip() == "0"
data = bytearray(640 * 1024 * 1024)
for offset in range(0, len(data), 4096):
    data[offset] = 1
""")
    assert result.error.kind == "oom"
    assert inspected[0]["State"]["OOMKilled"] is True
    resources = inspected[0]["HostConfig"]
    assert resources["Memory"] == resources["MemorySwap"] == MEMORY_BYTES


def test_cpu_loop_is_stopped_after_fifteen_seconds(sandbox_run):
    run, _ = sandbox_run
    start = perf_counter()
    result = run("while True:\n    pass")
    elapsed = perf_counter() - start
    assert result.error.kind == "timeout"
    assert TIMEOUT_SECONDS - 1 <= elapsed <= TIMEOUT_SECONDS + 10


def test_pids_limit_prevents_further_forks(sandbox_run):
    run, inspected = sandbox_run
    result = run("""\
import errno
import os
import signal
import time

children = []
try:
    for _ in range(80):
        try:
            pid = os.fork()
        except OSError as exc:
            if exc.errno != errno.EAGAIN:
                raise
            print(len(children))
            break
        if pid == 0:
            time.sleep(10)
            os._exit(0)
        children.append(pid)
    else:
        raise AssertionError("pids limit did not apply")
finally:
    for pid in children:
        os.kill(pid, signal.SIGTERM)
    for pid in children:
        os.waitpid(pid, 0)
""")
    assert result.error is None
    assert int(result.stdout.strip()) == PIDS_LIMIT - 1
    assert inspected[0]["HostConfig"]["PidsLimit"] == PIDS_LIMIT


def test_nonroot_script_reads_csv_and_writes_png(sandbox_run):
    run, inspected = sandbox_run
    result = run(
        """\
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

assert os.geteuid() == 10001
assert matplotlib.get_backend().lower() == "agg"
Path(os.environ["MPLCONFIGDIR"], "writable").touch()
assert Chem.rdBase.rdkitVersion
data = pd.read_csv("/work/input/candidates.csv")
print(len(data))
plt.hist(data["ic50_nm"])
plt.savefig("/work/output/example.png")
""",
        ["candidates"],
    )
    assert result.error is None, result.stderr
    assert result.stdout.strip() == "50"
    (name,) = result.files
    assert name == "example.png"
    assert result.files[name].startswith(b"\x89PNG\r\n\x1a\n")
    limits = inspected[0]["HostConfig"]
    assert limits["CapDrop"] == ["ALL"]
    assert limits["SecurityOpt"] == ["no-new-privileges"]
    assert limits["CpuQuota"] == limits["CpuPeriod"] == 100_000
