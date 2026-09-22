"""Content-verified scientific table bundles used by humanization services."""

import hashlib
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field


class BuiltResultArchive(BaseModel):
    """Metadata for publishing an immutable service-built ZIP."""

    size_bytes: int
    sha256: str


class ManifestFile(BaseModel):
    """One scientific publication member with its expected content identity."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def validate_manifest_paths(files: Sequence[ManifestFile]) -> None:
    """Reject unsafe, duplicate and recursive destinations before downloads."""
    names = [record.path for record in files]
    if len(names) != len(set(names)) or "manifest.json" in names:
        raise ValueError("Duplicate or recursive manifest entries")
    for name in names:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
        ):
            raise ValueError(f"Unsafe result path: {name}")


def build_table_archive(
    root: Path, destination: BinaryIO, manifest_files: Sequence[ManifestFile]
) -> BuiltResultArchive:
    """Verify declared bytes while streaming a deterministic stored ZIP."""
    records = {record.path: record for record in manifest_files}
    paths = sorted(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Result directory contains a symbolic link")
    files = {
        path.relative_to(root).as_posix(): path for path in paths if path.is_file()
    }
    if set(files) != {*records, "manifest.json"}:
        raise ValueError("Result directory does not match the workflow manifest")
    destination.seek(0)
    destination.truncate()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            digest = hashlib.sha256()
            size = 0
            with (
                path.open("rb") as source,
                archive.open(info, "w", force_zip64=True) as member,
            ):
                while chunk := source.read(1024 * 1024):
                    member.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if name in records:
                record = records[name]
                if (size, digest.hexdigest()) != (
                    record.size_bytes,
                    record.content_sha256,
                ):
                    raise ValueError(f"Result file does not match its manifest: {name}")
    size_bytes = destination.tell()
    destination.seek(0)
    digest = hashlib.sha256()
    while chunk := destination.read(1024 * 1024):
        digest.update(chunk)
    destination.seek(0)
    return BuiltResultArchive(size_bytes=size_bytes, sha256=digest.hexdigest())
