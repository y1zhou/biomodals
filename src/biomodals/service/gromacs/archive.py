"""Validation for the immutable GROMACS API result archive."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO, cast

import orjson

_CHUNK_SIZE = 1024 * 1024
_SHA256_LENGTH = 64
_SMALL_DOCUMENT_LIMIT = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidatedGromacsArchive:
    """Request identity recovered from a fully checked result ZIP."""

    request_sha256: str


def _expected_manifest_files(run_name: str) -> list[tuple[str, str]]:
    prefix = f"production_{run_name}"
    return [
        ("input.pdb", "input_structure"),
        ("parameters.json", "normalized_parameters"),
        ("provenance.json", "provenance"),
        ("run.log", "run_log"),
        ("outputs/production.mdp", "production_parameters"),
        (f"outputs/{prefix}.xtc", "trajectory"),
        (f"outputs/{prefix}.tpr", "production_topology"),
        (f"outputs/{prefix}_nopbc_centered.pdb", "centered_structure"),
        (f"outputs/rmsd_{prefix}.csv", "rmsd"),
        (f"outputs/rg_{prefix}.csv", "radius_of_gyration"),
        (f"outputs/rmsf_{prefix}.csv", "rmsf"),
    ]


def _expected_members(run_name: str) -> list[str]:
    return [
        *(name for name, _role in _expected_manifest_files(run_name)),
        "manifest.json",
        "checksums.sha256",
    ]


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        not info.is_dir()
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
    )


def _read_small(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > _SMALL_DOCUMENT_LIMIT:
        raise ValueError(f"GROMACS archive member is too large: {name}")
    return archive.read(info)


def _member_digest(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(info) as member:
        while chunk := member.read(_CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _manifest_records(document: object) -> list[dict[str, object]]:
    if (
        not isinstance(document, dict)
        or set(document) != {"archive_schema_version", "run_name", "files"}
        or document.get("archive_schema_version") != 1
    ):
        raise ValueError("GROMACS archive manifest is invalid")
    records = document.get("files")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("GROMACS archive manifest is invalid")
    return cast("list[dict[str, object]]", records)


def validate_gromacs_archive(
    handle: object,
    *,
    run_name: str,
) -> ValidatedGromacsArchive:
    """Validate exact members, CRCs, manifest records, and checksums."""
    binary_handle = cast("BinaryIO", handle)
    binary_handle.seek(0)
    try:
        with zipfile.ZipFile(binary_handle) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != _expected_members(run_name) or len(names) != len(set(names)):
                raise ValueError("GROMACS result archive has unexpected members")
            if not all(_safe_member(info) for info in infos):
                raise ValueError("GROMACS result archive has an unsafe member")

            try:
                manifest = orjson.loads(_read_small(archive, "manifest.json"))
            except orjson.JSONDecodeError as exc:
                raise ValueError("GROMACS archive manifest is invalid") from exc
            records = _manifest_records(manifest)
            if manifest.get("run_name") != run_name:
                raise ValueError("GROMACS archive manifest has the wrong run name")

            expected_records = _expected_manifest_files(run_name)
            record_names = [name for name, _role in expected_records]
            if len(records) != len(expected_records):
                raise ValueError("GROMACS archive manifest is incomplete")
            digests: dict[str, tuple[int, str]] = {}
            for (name, role), record in zip(expected_records, records, strict=True):
                size_bytes = record.get("size_bytes")
                sha256 = record.get("sha256")
                if (
                    set(record) != {"path", "role", "size_bytes", "sha256"}
                    or record.get("path") != name
                    or record.get("role") != role
                    or type(size_bytes) is not int
                    or size_bytes < 0
                    or not isinstance(sha256, str)
                    or len(sha256) != _SHA256_LENGTH
                    or any(character not in "0123456789abcdef" for character in sha256)
                ):
                    raise ValueError("GROMACS archive manifest record is invalid")
                actual = _member_digest(archive, archive.getinfo(name))
                if actual != (size_bytes, sha256):
                    raise ValueError(
                        f"GROMACS archive member does not match manifest: {name}"
                    )
                digests[name] = actual

            manifest_bytes = _read_small(archive, "manifest.json")
            manifest_record = (
                len(manifest_bytes),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            checksums_bytes = _read_small(archive, "checksums.sha256")
            expected_checksum_names = [*record_names, "manifest.json"]
            expected_checksums = "".join(
                f"{(manifest_record[1] if name == 'manifest.json' else digests[name][1])}  {name}\n"
                for name in expected_checksum_names
            ).encode("ascii")
            if checksums_bytes != expected_checksums:
                raise ValueError("GROMACS archive checksums do not match")

            input_bytes = _read_small(archive, "input.pdb")
            parameters_bytes = _read_small(archive, "parameters.json")
    except zipfile.BadZipFile as exc:
        raise ValueError("GROMACS result archive is invalid") from exc

    request_digest = hashlib.sha256()
    request_digest.update(len(input_bytes).to_bytes(8, byteorder="big"))
    request_digest.update(input_bytes)
    request_digest.update(parameters_bytes)
    return ValidatedGromacsArchive(request_sha256=request_digest.hexdigest())
