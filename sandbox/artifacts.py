"""Validate the Docker output archive and publish complete chart batches."""

import json
import tarfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4, uuid5

from PIL import Image

from mcp_servers.chem.datasets import PROJECT_ROOT

from .models import ArtifactReference

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
MAX_FILES = 5
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 64 * 1024


class OutputArchiveError(ValueError):
    """Generated output exceeds the sandbox's archive or PNG contract."""


def read_output_archive(chunks: Iterable[bytes]) -> dict[str, bytes]:
    """Read direct PNG children of output/ within the file and archive limits."""
    archive = BytesIO()
    for chunk in chunks:
        if archive.tell() + len(chunk) > MAX_ARCHIVE_BYTES:
            raise OutputArchiveError("输出 tar 的传输字节数超过上限。")
        archive.write(chunk)

    archive.seek(0)
    files: dict[str, bytes] = {}
    total_bytes = 0
    found_root = False
    try:
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            for member in tar:
                if member.name == "output" and member.isdir() and not found_root:
                    found_root = True
                    continue
                directory, _, filename = member.name.partition("/")
                if (
                    not found_root
                    or directory != "output"
                    or not filename
                    or "/" in filename
                    or "\\" in filename
                    or any(ord(char) < 32 or ord(char) == 127 for char in filename)
                ):
                    raise OutputArchiveError(
                        f"输出 tar 的文件路径无效：{member.name!r}"
                    )
                if (
                    member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or member.sparse is not None
                ):
                    raise OutputArchiveError(f"输出必须是普通文件：{filename!r}")
                if not filename.lower().endswith(".png"):
                    raise OutputArchiveError(f"输出只支持 PNG 文件：{filename!r}")
                if filename in files:
                    raise OutputArchiveError(f"输出 tar 含同名文件：{filename!r}")
                if len(files) == MAX_FILES:
                    raise OutputArchiveError(f"输出文件数超过 {MAX_FILES} 个。")
                if not 0 < member.size <= MAX_FILE_BYTES:
                    raise OutputArchiveError(
                        f"输出文件大小超出 1 至 {MAX_FILE_BYTES} 字节：{filename!r}"
                    )
                total_bytes += member.size
                if total_bytes > MAX_TOTAL_BYTES:
                    raise OutputArchiveError(
                        f"输出文件合计超过 {MAX_TOTAL_BYTES} 字节。"
                    )
                source = tar.extractfile(member)
                assert source is not None
                with source:
                    payload = source.read()
                try:
                    with Image.open(BytesIO(payload), formats=("PNG",)) as img:
                        img.verify()
                except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
                    raise OutputArchiveError(
                        f"无效的 PNG 文件 {filename!r}：{exc}"
                    ) from exc
                files[filename] = payload
    except tarfile.TarError as exc:
        raise OutputArchiveError(f"无法读取输出 tar：{exc}") from exc
    if not found_root:
        raise OutputArchiveError("输出 tar 缺少 output 目录。")
    return files


def publish_artifacts(
    thread_id: UUID, message_id: str, tool_call_id: str, files: dict[str, bytes]
) -> list[ArtifactReference]:
    if not files:
        return []
    thread_dir = ARTIFACTS_DIR / str(thread_id)
    thread_dir.mkdir(parents=True, exist_ok=True)
    call_key = uuid5(thread_id, json.dumps([message_id, tool_call_id]))
    # A re-executed checkpoint can produce different files for the same call.
    batch = thread_dir / f"{call_key.hex}-{uuid4().hex}"
    references = []
    with TemporaryDirectory(dir=thread_dir, prefix=".pending-") as staging:
        for filename, payload in files.items():
            artifact_id = uuid4()
            (Path(staging) / f"{artifact_id}.png").write_bytes(payload)
            references.append(
                ArtifactReference(
                    artifact_id=artifact_id, filename=filename, size_bytes=len(payload)
                )
            )
        Path(staging).rename(batch)
    return references


def find_artifact(thread_id: UUID, artifact_id: UUID) -> Path | None:
    thread_dir = ARTIFACTS_DIR / str(thread_id)
    # Published batch directories begin with a hex digit, while staging is hidden.
    return next(thread_dir.glob(f"[0-9a-f]*/{artifact_id}.png"), None)
