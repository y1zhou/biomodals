"""Shared durable-artifact helper tests."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.helper.artifacts import (
    file_matches_sha256,
    file_size_sha256,
    publish_content_addressed_file,
    read_volume_file_exact,
    read_volume_json,
    replace_bytes_atomic,
    sha256_bytes,
)


class _Reader:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]


def test_file_size_sha256_reads_one_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")

    assert file_size_sha256(path) == (8, sha256_bytes(b"artifact"))


def test_file_matches_sha256_rejects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    size, digest = file_size_sha256(path)

    assert file_matches_sha256(path, size, digest)

    path.write_bytes(b"changed!")

    assert not file_matches_sha256(path, size, digest)


def test_replace_bytes_atomic_replaces_content(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.bin"
    replace_bytes_atomic(path, b"first")
    replace_bytes_atomic(path, b"second")

    assert path.read_bytes() == b"second"
    assert not tuple(path.parent.glob(".*.tmp"))


def test_publish_content_addressed_file_uses_digest_directory(tmp_path: Path) -> None:
    source = tmp_path / "source" / "artifact.bin"
    source.parent.mkdir()
    source.write_bytes(b"artifact")

    destination, size, digest = publish_content_addressed_file(
        source,
        tmp_path / "published",
    )

    assert destination == tmp_path / "published" / digest / source.name
    assert destination.read_bytes() == b"artifact"
    assert size == 8
    assert digest == sha256_bytes(b"artifact")


def test_volume_file_download_requires_exact_publication() -> None:
    reader = _Reader({"artifact.bin": b"artifact"})

    assert (
        read_volume_file_exact(
            reader,
            "artifact.bin",
            size_bytes=8,
            content_sha256=sha256_bytes(b"artifact"),
        )
        == b"artifact"
    )
    with pytest.raises(RuntimeError, match="does not match"):
        read_volume_file_exact(
            reader,
            "artifact.bin",
            size_bytes=8,
            content_sha256=sha256_bytes(b"changed!"),
        )


def test_volume_json_is_bounded_and_fail_closed() -> None:
    reader = _Reader({"record.json": b'{"value":1}', "invalid.json": b"{"})

    assert read_volume_json(reader, "record.json") == {"value": 1}
    assert read_volume_json(reader, "invalid.json") is None
    assert read_volume_json(reader, "missing.json") is None
