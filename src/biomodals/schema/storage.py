"""Storage schemas shared by app results and workflow artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StorageKind(StrEnum):
    """Supported storage forms for app outputs and workflow artifacts."""

    INLINE_BYTES = "inline_bytes"
    VOLUME_PATH = "volume_path"


class InlineBytes(BaseModel):
    """Bytes returned directly in an app result before workflow materialization."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    kind: Literal[StorageKind.INLINE_BYTES] = StorageKind.INLINE_BYTES
    data: bytes
    filename: str
    media_type: str | None = None
    archive_format: Literal["tar.zst", "tar.gz", "zip"] | None = None


class VolumePath(BaseModel):
    """Path to data stored in a Modal volume."""

    kind: Literal[StorageKind.VOLUME_PATH] = StorageKind.VOLUME_PATH
    volume_name: str
    path: str
    media_type: str | None = None
