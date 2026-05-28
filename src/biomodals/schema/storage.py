"""Storage schemas shared by app results and workflow artifacts."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum  # noqa: UP035


class StorageKind(StrEnum):
    """Supported storage forms for app outputs and workflow artifacts."""

    INLINE_BYTES = "inline_bytes"
    VOLUME_PATH = "volume_path"


class InlineBytes(BaseModel):
    """UTF-8 text returned directly before workflow materialization."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[StorageKind.INLINE_BYTES] = StorageKind.INLINE_BYTES
    data: bytes
    filename: str
    media_type: str | None = None

    @field_validator("data")
    @classmethod
    def ensure_utf8_text(cls, value: bytes) -> bytes:
        """Reject non-text inline payloads."""
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "InlineBytes.data must be UTF-8 text; use VolumePath for binary data."
            ) from exc
        return value


class VolumePath(BaseModel):
    """Path to data stored in a Modal volume."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[StorageKind.VOLUME_PATH] = StorageKind.VOLUME_PATH
    volume_name: str
    path: str
    media_type: str | None = None

    @field_validator("path")
    @classmethod
    def ensure_relative_volume_path(cls, value: str) -> str:
        """Reject paths that can escape the declared volume root."""
        path = PurePosixPath(value)
        if value == "" or value == ".":
            raise ValueError("VolumePath.path must be a non-empty relative path")
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("VolumePath.path must be relative and must not traverse")
        if "\\" in value:
            raise ValueError("VolumePath.path must use POSIX separators")
        return value
