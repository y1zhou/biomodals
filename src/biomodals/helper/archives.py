"""Bounded readers for flat scientific table/report archives."""

import tarfile
from io import BytesIO
from pathlib import PurePosixPath

import zstandard

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def flat_archive_members(content: bytes) -> dict[str, bytes]:
    """Read a flat tar.zst without filesystem extraction or unbounded expansion."""
    if not content or len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid compressed archive size")
    with zstandard.ZstdDecompressor().stream_reader(BytesIO(content)) as reader:
        expanded = reader.read(MAX_ARCHIVE_BYTES + 1)
    if len(expanded) > MAX_ARCHIVE_BYTES:
        raise ValueError("Expanded archive exceeds byte limit")
    files = {}
    with tarfile.open(fileobj=BytesIO(expanded), mode="r:") as archive:
        for index, member in enumerate(archive):
            path = PurePosixPath(member.name)
            if index > 1000 or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or len(path.parts) > 2 or path.name in files:
                raise ValueError("Unexpected archive member or duplicate basename")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive member")
            files[path.name] = stream.read()
    return files
