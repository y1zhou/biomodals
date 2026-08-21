"""Durable byte and JSON artifact primitives."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from typing import Any, Protocol, cast

import orjson

if sys.version_info >= (3, 11):  # noqa: UP036 - mounted in Python 3.10 apps
    from datetime import UTC
else:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

_JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def read_volume_file_exact(
    reader: VolumeReader,
    path: str,
    *,
    size_bytes: object,
    content_sha256: object,
) -> bytes:
    """Read one Volume file only when it matches its publication record."""
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 1
        or not isinstance(content_sha256, str)
        or not _SHA256_PATTERN.fullmatch(content_sha256)
    ):
        raise ValueError("Volume publication has invalid size or SHA-256 metadata")
    content = bytearray()
    digest = hashlib.sha256()
    for chunk in reader.read_file(path):
        if not isinstance(chunk, bytes):
            raise TypeError(f"Volume returned non-bytes for {path}")
        if len(content) + len(chunk) > size_bytes:
            raise RuntimeError(f"Volume file exceeds its published size: {path}")
        content.extend(chunk)
        digest.update(chunk)
    if len(content) != size_bytes or digest.hexdigest() != content_sha256:
        raise RuntimeError(f"Volume file does not match its publication: {path}")
    return bytes(content)


def read_volume_json(
    reader: VolumeReader,
    path: str,
    *,
    max_bytes: int = 64 * 1024,
) -> object | None:
    """Read one bounded JSON record, returning ``None`` when absent or invalid."""
    content = read_volume_bytes(reader, path, max_bytes=max_bytes)
    if content is None:
        return None
    try:
        return orjson.loads(content)
    except orjson.JSONDecodeError:
        return None


def read_bounded_file_bytes(
    path: Path,
    *,
    field_name: str,
    max_bytes: int,
) -> bytes:
    """Read one stable regular file without exceeding its byte limit."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Expected regular file: {path}")
    stat_before = path.stat()
    if stat_before.st_size > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
    with path.open("rb") as handle:
        value = handle.read(max_bytes + 1)
    if len(value) > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
    stat_after = path.stat()
    before_identity = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    after_identity = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if before_identity != after_identity or len(value) != stat_after.st_size:
        raise RuntimeError(f"{field_name} changed while it was being read: {path}")
    return value


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
    if forbidden_bytes == b"":
        raise ValueError("forbidden_bytes must be nonempty")
    if forbidden_bytes is None:
        return file_size_sha256(path, chunk_size=chunk_size)[1]
    require_regular_file(path)
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


def file_size_sha256(
    path: Path,
    *,
    chunk_size: int = 16 * 1024 * 1024,
) -> tuple[int, str]:
    """Return one regular file's size and SHA-256 digest in one pass."""
    require_regular_file(path)
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def publish_content_addressed_file(source: Path, root: Path) -> tuple[Path, int, str]:
    """Copy one file into ``root/<sha256>/<name>`` in a single read pass."""
    require_regular_file(source)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{source.name}.{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            while chunk := input_file.read(16 * 1024 * 1024):
                output_file.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
        content_sha256 = digest.hexdigest()
        destination = root / content_sha256 / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, size_bytes, content_sha256


def file_matches_sha256(
    path: Path,
    expected_size: object,
    expected_digest: object,
) -> bool:
    """Return whether a regular file matches one size and SHA-256 record."""
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
    ):
        return False
    try:
        if path.is_symlink():
            return False
        stat = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return (
        S_ISREG(stat.st_mode)
        and stat.st_size == expected_size
        and sha256_file(path) == expected_digest
    )


def replace_bytes_atomic(path: Path, value: bytes) -> None:
    """Atomically replace one byte artifact without forcing a disk sync."""
    _write_bytes_atomic(path, value, sync=False)


def write_bytes_atomic(path: Path, value: bytes) -> None:
    """Atomically publish one byte artifact on the destination filesystem."""
    _write_bytes_atomic(path, value, sync=True)


def _write_bytes_atomic(path: Path, value: bytes, *, sync: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            if sync:
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
