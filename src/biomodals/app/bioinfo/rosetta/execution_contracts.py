"""Rosetta Task commands and workload-owned completion publications."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import orjson

PUBLICATION_SCHEMA_VERSION = 1
_MAX_PUBLICATION_BYTES = 64 * 1024


class _VolumeReader(Protocol):
    def read_file(self, path: str) -> Iterable[bytes]:
        """Yield chunks for one Volume-relative file."""


@dataclass(frozen=True)
class RosettaTaskSpec:
    """One independently scheduled Rosetta command."""

    task_key: str
    index: int
    binary: str
    pdb: str
    rosetta_script: str | None
    flags_file: str | None
    output_dir: str
    worker_log: str
    expected_files: tuple[str, ...]
    input_sha256: str
    script_sha256: str | None = None
    flags_sha256: str | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous paths before a Task reaches a worker."""
        if not self.task_key or not self.binary:
            raise ValueError("Rosetta Task identity and binary cannot be empty")
        if self.index < 1:
            raise ValueError("Rosetta Task index must be positive")
        for value in (
            self.pdb,
            self.rosetta_script,
            self.flags_file,
            self.output_dir,
            self.worker_log,
            *self.expected_files,
        ):
            if value is not None:
                _relative_path(value)
        for digest in (
            self.input_sha256,
            self.script_sha256,
            self.flags_sha256,
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Rosetta input digests must be lowercase SHA-256")

    @property
    def scientific_payload(self) -> dict[str, object]:
        """Return result-affecting Task identity for kernel fingerprinting."""
        return {
            "task_key": self.task_key,
            "index": self.index,
            "binary": self.binary,
            "expected_files": list(self.expected_files),
            "input_sha256": self.input_sha256,
            "script_sha256": self.script_sha256,
            "flags_sha256": self.flags_sha256,
            "candidate_id": self.candidate_id,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize this trusted execution payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> RosettaTaskSpec:
        """Decode a staged or claimed execution payload."""
        if not isinstance(value, dict):
            raise TypeError("Rosetta Task payload must be a mapping")
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Rosetta Task payload keys must be text")
        record = cast(dict[str, Any], value.copy())
        expected = record.get("expected_files")
        if not isinstance(expected, list | tuple):
            raise TypeError("Rosetta expected_files must be a collection")
        record["expected_files"] = tuple(expected)
        return cls(**record)


def execute_rosetta_task(
    *,
    run_root: str | Path,
    task: RosettaTaskSpec,
    task_fingerprint: str,
    run_command: Callable[..., object],
) -> dict[str, object]:
    """Run or reuse one command and publish its fingerprint-bound completion."""
    root = Path(run_root)
    if validate_task_publication(root, task, task_fingerprint):
        return _execution_result(task)

    output_dir = root.joinpath(*_relative_path(task.output_dir).parts)
    worker_log = root.joinpath(*_relative_path(task.worker_log).parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    command = [task.binary]
    if task.rosetta_script is not None:
        command.extend([
            "-parser:protocol",
            str(root.joinpath(*_relative_path(task.rosetta_script).parts)),
        ])
    if task.flags_file is not None:
        command.append(f"@{root.joinpath(*_relative_path(task.flags_file).parts)}")
    command.extend([
        "-s",
        str(root.joinpath(*_relative_path(task.pdb).parts)),
        "-out:path:all",
        str(output_dir),
    ])
    run_command(
        command,
        output_mode="capture",
        log_file=worker_log,
    )
    missing = [
        path
        for path in task.expected_files
        if not root.joinpath(*_relative_path(path).parts).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Rosetta returned without required output: " + ", ".join(missing)
        )
    _write_task_publication(root, task, task_fingerprint)
    return _execution_result(task)


def validate_task_publication(
    run_root: str | Path,
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> bool:
    """Return whether this exact Task has a complete durable publication."""
    path = task_publication_path(run_root, task.task_key)
    if not path.is_file():
        return False
    try:
        value: Any = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return False
    return (
        _publication_payload_matches(value, task, task_fingerprint)
        and Path(run_root).joinpath(*_relative_path(task.output_dir).parts).is_dir()
        and Path(run_root).joinpath(*_relative_path(task.worker_log).parts).is_file()
        and all(
            Path(run_root).joinpath(*_relative_path(path).parts).is_file()
            for path in task.expected_files
        )
    )


def validate_task_publication_from_volume(
    volume: _VolumeReader,
    run_root: str | PurePosixPath,
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> bool:
    """Validate one Task through a remote Volume's bounded read interface."""
    relative_root = _relative_path(str(run_root))
    marker_path = (
        relative_root
        / ".biomodals"
        / "tasks"
        / f"{sha256(task.task_key.encode()).hexdigest()}.json"
    )
    try:
        marker = _read_volume_file(
            volume,
            marker_path.as_posix(),
            max_bytes=_MAX_PUBLICATION_BYTES,
        )
    except FileNotFoundError:
        return False
    try:
        value: Any = orjson.loads(marker)
    except orjson.JSONDecodeError:
        return False
    if not _publication_payload_matches(value, task, task_fingerprint):
        return False
    required = (task.worker_log, *task.expected_files)
    return all(
        _volume_file_exists(
            volume,
            (relative_root / _relative_path(path)).as_posix(),
        )
        for path in required
    )


def task_publication_path(run_root: str | Path, task_key: str) -> Path:
    """Return a collision-resistant marker path outside scientific outputs."""
    marker_name = sha256(task_key.encode()).hexdigest() + ".json"
    return Path(run_root) / ".biomodals" / "tasks" / marker_name


def _write_task_publication(
    run_root: Path,
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> None:
    path = task_publication_path(run_root, task.task_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = orjson.dumps(
        {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "task_key": task.task_key,
            "task_fingerprint": task_fingerprint,
            "output_dir": task.output_dir,
            "worker_log": task.worker_log,
            "expected_files": list(task.expected_files),
        },
        option=orjson.OPT_SORT_KEYS,
    )
    temporary = path.with_suffix(f".{time.time_ns()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _execution_result(task: RosettaTaskSpec) -> dict[str, object]:
    return {
        "status": "succeeded",
        "task_key": task.task_key,
        "candidate_id": task.candidate_id,
        "index": task.index,
        "output_dir": task.output_dir,
        "worker_log": task.worker_log,
        "expected_files": list(task.expected_files),
    }


def _publication_payload_matches(
    value: object,
    task: RosettaTaskSpec,
    task_fingerprint: str,
) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema_version") == PUBLICATION_SCHEMA_VERSION
        and value.get("status") == "complete"
        and value.get("task_key") == task.task_key
        and value.get("task_fingerprint") == task_fingerprint
        and value.get("output_dir") == task.output_dir
        and value.get("worker_log") == task.worker_log
        and value.get("expected_files") == list(task.expected_files)
    )


def _read_volume_file(
    volume: _VolumeReader,
    path: str,
    *,
    max_bytes: int,
) -> bytes:
    content = bytearray()
    for chunk in volume.read_file(path):
        if not isinstance(chunk, bytes):
            raise TypeError(f"Volume returned non-bytes for {path}")
        if len(content) + len(chunk) > max_bytes:
            raise ValueError(f"Volume file exceeds {max_bytes} bytes: {path}")
        content.extend(chunk)
    return bytes(content)


def _volume_file_exists(volume: _VolumeReader, path: str) -> bool:
    try:
        iterator = iter(volume.read_file(path))
        next(iterator, None)
    except FileNotFoundError:
        return False
    return True


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Rosetta Task paths must be relative and contained")
    return path
