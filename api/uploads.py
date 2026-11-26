"""Validate uploaded CSVs with the chem-mcp reader before publishing their IDs."""

import csv
import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import HTTPException, Request
from pydantic import BaseModel, PositiveInt
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from mcp_servers.chem.datasets import (
    publish_dataset,
    read_dataset_csv,
    upload_directory,
)

from .schemas import UploadedDataset


class UploadSettings(BaseModel):
    directory: Path
    max_bytes: PositiveInt
    max_rows: PositiveInt

    @classmethod
    def from_env(cls) -> "UploadSettings":
        return cls.model_validate(
            {
                "directory": upload_directory(),
                "max_bytes": os.getenv("MAX_UPLOAD_BYTES", "5242880"),
                "max_rows": os.getenv("MAX_UPLOAD_ROWS", "5000"),
            }
        )


class UploadParser(MultiPartParser):
    """Count file bytes as multipart chunks arrive, before Starlette spools them."""

    def __init__(self, request: Request, max_bytes: int):
        super().__init__(request.headers, request.stream(), max_files=1)
        self.max_bytes = max_bytes
        self.file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self.file_bytes += end - start
            if self.file_bytes > self.max_bytes:
                raise MultiPartException(f"文件大小超过 {self.max_bytes} 字节")
        super().on_part_data(data, start, end)


@asynccontextmanager
async def receive_upload(request: Request, max_bytes: int) -> AsyncIterator[UploadFile]:
    content_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if content_type != "multipart/form-data":
        raise HTTPException(422, "请使用 multipart/form-data 上传 file 文件")
    try:
        form = await UploadParser(request, max_bytes).parse()
    except MultiPartException as exc:
        raise HTTPException(422, exc.message) from exc
    try:
        file = form.get("file")
        if not isinstance(file, UploadFile):
            raise HTTPException(422, "缺少 file 上传文件")
        yield file
    finally:
        await form.close()


def save_upload(file: UploadFile, settings: UploadSettings) -> UploadedDataset:
    filename = file.filename
    if not filename or Path(filename).suffix.lower() != ".csv":
        raise HTTPException(422, "文件必须使用 .csv 扩展名")
    payload = file.file.read()
    try:
        with io.StringIO(payload.decode("utf-8-sig"), newline="") as source:
            columns, rows = read_dataset_csv(
                source, filename, max_rows=settings.max_rows
            )
    except UnicodeDecodeError as exc:
        raise HTTPException(422, f"文件不是有效的 UTF-8：{exc}") from exc
    except (csv.Error, ValueError) as exc:
        raise HTTPException(422, f"CSV 格式错误：{exc}") from exc
    if not rows:
        raise HTTPException(422, "CSV 至少需要 1 条数据行")

    return UploadedDataset(
        dataset_id=publish_dataset(settings.directory, payload),
        filename=filename,
        row_count=len(rows),
        columns=columns,
    )
