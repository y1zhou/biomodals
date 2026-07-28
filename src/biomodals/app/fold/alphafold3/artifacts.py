"""Durable byte and JSON artifact primitives for AlphaFold 3 workflows."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import orjson

_JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_MSA_FIELD_BYTES = 512 * 1024 * 1024


class VolumeHandle(Protocol):
    """Persistence barriers shared by mounted Modal Volume adapters."""

    def reload(self) -> None:
        """Reload commits made by other containers."""
        ...

    def commit(self) -> None:
        """Commit this container's writes."""
        ...


class VolumeReader(Protocol):
    """Chunked read interface exposed by a local Modal Volume handle."""

    def read_file(self, path: str) -> Iterable[bytes]:
        """Yield a Volume file as byte chunks."""
        ...


def read_volume_bytes(
    reader: VolumeReader,
    path: str,
    *,
    max_bytes: int,
) -> bytes | None:
    """Read one bounded Volume file, returning ``None`` when it is absent."""
    if not isinstance(path, str) or not path:
        raise ValueError("Volume path must be a non-empty string")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    content = bytearray()
    try:
        for chunk in reader.read_file(path):
            if not isinstance(chunk, bytes):
                raise TypeError(f"Volume returned non-bytes for {path}")
            if len(content) + len(chunk) > max_bytes:
                raise ValueError(
                    f"Volume file exceeds the {max_bytes}-byte limit: {path}"
                )
            content.extend(chunk)
    except FileNotFoundError:
        return None
    return bytes(content)


def json_bytes(value: object) -> bytes:
    """Serialize canonical, human-readable JSON bytes."""
    return orjson.dumps(value, option=_JSON_OPTIONS)


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of one byte string."""
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(UTC).isoformat()


def append_log(path: Path, message: str) -> None:
    """Append one timestamped line to a durable operation log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def require_regular_file(path: Path) -> None:
    """Require a non-symlink regular file with at least one byte."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Expected regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Expected nonempty file: {path}")


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 16 * 1024 * 1024,
    forbidden_bytes: bytes | None = None,
) -> str:
    """Compute a digest and optionally reject a byte marker while streaming."""
    require_regular_file(path)
    if forbidden_bytes == b"":
        raise ValueError("forbidden_bytes must be nonempty")
    digest = hashlib.sha256()
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if forbidden_bytes is not None:
                searchable = overlap + chunk
                if forbidden_bytes in searchable:
                    raise ValueError(f"Forbidden byte marker remains in {path}")
                overlap_size = len(forbidden_bytes) - 1
                overlap = searchable[-overlap_size:] if overlap_size else b""
    return digest.hexdigest()


def write_bytes_atomic(path: Path, value: bytes) -> None:
    """Atomically publish one byte artifact on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically publish one canonical JSON artifact."""
    write_bytes_atomic(path, json_bytes(value))


def load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting all other top-level values."""
    require_regular_file(path)
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def artifact_record(
    path: Path,
    root: Path,
    *,
    forbidden_bytes: bytes | None = None,
) -> dict[str, object]:
    """Describe one verified, nonempty artifact below ``root``."""
    require_regular_file(path)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Artifact escapes root: {path}")
    return {
        "path": resolved_path.relative_to(resolved_root).as_posix(),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": sha256_file(resolved_path, forbidden_bytes=forbidden_bytes),
    }


def _record_location(
    root: Path,
    record: object,
    expected_path: str | None,
) -> tuple[Path, int, str] | None:
    if not isinstance(record, dict):
        return None
    path_value = record.get("path")
    size_value = record.get("size_bytes")
    digest_value = record.get("sha256")
    if (
        not isinstance(path_value, str)
        or path_value == ""
        or (expected_path is not None and path_value != expected_path)
        or isinstance(size_value, bool)
        or not isinstance(size_value, int)
        or size_value <= 0
        or not isinstance(digest_value, str)
        or _SHA256_PATTERN.fullmatch(digest_value) is None
    ):
        return None
    relative = PurePosixPath(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return None
    try:
        require_regular_file(path)
        if path.stat().st_size != size_value:
            return None
    except (OSError, ValueError):
        return None
    return path, size_value, digest_value


def load_artifact_bytes(
    root: Path,
    record: object,
    expected_path: str,
) -> bytes | None:
    """Load an expected artifact only when size and digest match its record."""
    location = _record_location(root, record, expected_path)
    if location is None:
        return None
    path, expected_size, expected_digest = location
    try:
        with path.open("rb") as handle:
            value = handle.read(expected_size + 1)
    except (OSError, OverflowError):
        return None
    if len(value) != expected_size or sha256_bytes(value) != expected_digest:
        return None
    return value


def validate_artifact_record(
    root: Path,
    record: object,
) -> dict[str, object] | None:
    """Return a record only when its safe relative path, size, and digest match."""
    location = _record_location(root, record, None)
    if location is None:
        return None
    path, expected_size, expected_digest = location
    if path.stat().st_size != expected_size or sha256_file(path) != expected_digest:
        return None
    return cast(dict[str, object], record)
