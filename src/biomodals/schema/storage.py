"""Storage schemas shared by app results and workflow artifacts."""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035,I001


class StorageKind(StrEnum):
    """Supported storage forms for app outputs and workflow artifacts."""

    INLINE_BYTES = "inline_bytes"
    VOLUME_PATH = "volume_path"


ZSTD_MEDIA_TYPE = "application/zstd"


class InlineBytes(BaseModel):
    """Inline app output bytes before workflow materialization."""

    model_config = ConfigDict(
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    kind: Literal[StorageKind.INLINE_BYTES] = StorageKind.INLINE_BYTES
    data: bytes
    filename: str
    media_type: str | None = None

    @model_validator(mode="after")
    def ensure_supported_bytes(self) -> InlineBytes:
        """Reject binary inline payloads unless they are zstd archives."""
        if self.media_type == ZSTD_MEDIA_TYPE:
            return self
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "InlineBytes.data must be UTF-8 text unless media_type is "
                f"{ZSTD_MEDIA_TYPE!r}."
            ) from exc
        return self


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

    def at_mountpoint(self, mountpoint: str | Path) -> Path:
        """Return the path of this volume on the given mountpoint."""
        return Path(mountpoint) / self.path

    def __str__(self) -> str:
        """Return a human-readable string representation of this volume path."""
        return f"'{self.path}' in volume '{self.volume_name}'"
