"""Shared durable-artifact helper tests."""

# ruff: noqa: D103

from pathlib import Path

from biomodals.helper.artifacts import (
    file_matches_sha256,
    file_size_sha256,
    publish_content_addressed_file,
    replace_bytes_atomic,
    sha256_bytes,
)


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
