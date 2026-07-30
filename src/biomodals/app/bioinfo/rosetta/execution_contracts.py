"""Rosetta Task commands and workload-owned completion publications."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast

import orjson

PUBLICATION_SCHEMA_VERSION = 1


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
        isinstance(value, dict)
        and value.get("schema_version") == PUBLICATION_SCHEMA_VERSION
        and value.get("status") == "complete"
        and value.get("task_key") == task.task_key
        and value.get("task_fingerprint") == task_fingerprint
        and Path(run_root).joinpath(*_relative_path(task.output_dir).parts).is_dir()
        and Path(run_root).joinpath(*_relative_path(task.worker_log).parts).is_file()
        and all(
            Path(run_root).joinpath(*_relative_path(path).parts).is_file()
            for path in task.expected_files
        )
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


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Rosetta Task paths must be relative and contained")
    return path
