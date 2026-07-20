"""Validation for the immutable GROMACS API result archive."""

from __future__ import annotations

import csv
import hashlib
import io
import stat
import struct
import zipfile
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO, Protocol, TypeVar, cast

import orjson

_CHUNK_SIZE = 1024 * 1024
_SHA256_LENGTH = 64
_SMALL_DOCUMENT_LIMIT = 1024 * 1024
_MAX_PDB_BYTES = 10 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

ReadRemoteFile = Callable[[str], AsyncIterable[bytes]]
_T = TypeVar("_T")


class RunBounded(Protocol):
    """Execute a blocking whole-file operation outside the event loop."""

    def __call__(
        self,
        operation: Callable[..., _T],
        /,
        *args: object,
        **kwargs: object,
    ) -> Awaitable[_T]:
        """Run one blocking operation and return its eventual result."""
        ...


@dataclass(frozen=True, slots=True)
class ValidatedGromacsArchive:
    """Request identity recovered from a fully checked result ZIP."""

    request_sha256: str


@dataclass(frozen=True, slots=True)
class BuiltGromacsArchive:
    """Metadata for one deterministic service-built result ZIP."""

    request_sha256: str
    size_bytes: int
    sha256: str


_ARCHIVE_SCHEMA_VERSION = 2


def _required_output_files(run_name: str) -> list[tuple[str, str]]:
    prefix = f"production_{run_name}"
    return [
        ("outputs/production.mdp", "production_parameters"),
        (f"outputs/{prefix}_nopbc.xtc", "trajectory"),
        (f"outputs/{prefix}.tpr", "production_topology"),
        (f"outputs/{prefix}_nopbc_centered.pdb", "centered_structure"),
        (f"outputs/rmsd_{prefix}.csv", "rmsd"),
        (f"outputs/rmsd_{prefix}.png", "rmsd_plot"),
        (f"outputs/rg_{prefix}.csv", "radius_of_gyration"),
        (f"outputs/rg_{prefix}.png", "radius_of_gyration_plot"),
        (f"outputs/rmsf_{prefix}.csv", "rmsf"),
        (f"outputs/rmsf_{prefix}.png", "rmsf_plot"),
    ]


def _required_manifest_files(run_name: str) -> list[tuple[str, str]]:
    return [
        ("input.pdb", "input_structure"),
        *_required_output_files(run_name),
        ("metadata/parameters.json", "normalized_parameters"),
        ("metadata/provenance.json", "provenance"),
        ("metadata/stages.json", "stages"),
        ("metadata/run.log", "run_log"),
    ]


def _optional_remote_files(run_name: str) -> list[tuple[str, str, str]]:
    """Return fixed diagnostic remote name, archive path, and manifest role."""
    basenames = (
        f"em_{run_name}.log",
        f"em_{run_name}.full.log",
        f"cg_{run_name}.log",
        f"cg_{run_name}.full.log",
        f"nvt_{run_name}.log",
        f"nvt_{run_name}.full.log",
        f"npt_{run_name}.log",
        f"npt_{run_name}.full.log",
        f"production_{run_name}.log",
        "minim.mdp",
        "cg.mdp",
        "nvt.mdp",
        "npt.mdp",
    )
    return [
        (basename, f"metadata/gromacs/{basename}", "gromacs_diagnostic")
        for basename in basenames
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


def _read_small(
    archive: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int = _SMALL_DOCUMENT_LIMIT,
) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > max_bytes:
        raise ValueError(f"GROMACS archive member is too large: {name}")
    return archive.read(info)


def _read_prefix(archive: zipfile.ZipFile, name: str, size: int = 128) -> bytes:
    with archive.open(name) as member:
        return member.read(size)


def _validate_pdb(content: bytes, *, name: str) -> None:
    try:
        lines = content.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"GROMACS archive PDB is invalid: {name}") from exc
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
        raise ValueError(f"GROMACS archive PDB has no atoms: {name}")


def _validate_csv(content: bytes, *, name: str, header: tuple[str, str]) -> None:
    try:
        rows = list(csv.reader(io.StringIO(content.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"GROMACS archive CSV is invalid: {name}") from exc
    if len(rows) < 2 or tuple(rows[0]) != header:
        raise ValueError(f"GROMACS archive CSV has the wrong schema: {name}")
    try:
        for row in rows[1:]:
            if len(row) != 2:
                raise ValueError
            float(row[0])
            float(row[1])
    except ValueError as exc:
        raise ValueError(f"GROMACS archive CSV has invalid rows: {name}") from exc


def _validate_required_formats(archive: zipfile.ZipFile, run_name: str) -> None:
    """Reject inexpensive-to-detect truncation before publishing success."""
    prefix = f"production_{run_name}"
    input_name = "input.pdb"
    centered_name = f"outputs/{prefix}_nopbc_centered.pdb"
    _validate_pdb(
        _read_small(archive, input_name, max_bytes=_MAX_PDB_BYTES),
        name=input_name,
    )
    _validate_pdb(_read_small(archive, centered_name), name=centered_name)

    mdp_name = "outputs/production.mdp"
    try:
        mdp = _read_small(archive, mdp_name).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("GROMACS production MDP is invalid") from exc
    if not any(
        "=" in line
        for line in mdp.splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ):
        raise ValueError("GROMACS production MDP has no parameters")

    xtc_name = f"outputs/{prefix}_nopbc.xtc"
    xtc_prefix = _read_prefix(archive, xtc_name)
    if len(xtc_prefix) < 16 or struct.unpack(">i", xtc_prefix[:4])[0] != 1995:
        raise ValueError("GROMACS trajectory is invalid")

    tpr_name = f"outputs/{prefix}.tpr"
    tpr_prefix = _read_prefix(archive, tpr_name)
    if len(tpr_prefix) < 16 or b"VERSION" not in tpr_prefix:
        raise ValueError("GROMACS production topology is invalid")

    csv_contracts = (
        (f"outputs/rmsd_{prefix}.csv", ("time_ns", "rmsd")),
        (f"outputs/rg_{prefix}.csv", ("time_ns", "rg")),
        (f"outputs/rmsf_{prefix}.csv", ("residue_index", "rmsf")),
    )
    for name, header in csv_contracts:
        _validate_csv(_read_small(archive, name), name=name, header=header)

    for name in (
        f"outputs/rmsd_{prefix}.png",
        f"outputs/rg_{prefix}.png",
        f"outputs/rmsf_{prefix}.png",
    ):
        png_prefix = _read_prefix(archive, name, 24)
        if (
            len(png_prefix) < 24
            or png_prefix[:8] != b"\x89PNG\r\n\x1a\n"
            or png_prefix[12:16] != b"IHDR"
        ):
            raise ValueError(f"GROMACS archive PNG is invalid: {name}")

    for name, expected_type in (
        ("metadata/parameters.json", dict),
        ("metadata/provenance.json", dict),
        ("metadata/stages.json", list),
    ):
        try:
            document = orjson.loads(_read_small(archive, name))
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"GROMACS archive JSON is invalid: {name}") from exc
        if not isinstance(document, expected_type):
            raise ValueError(f"GROMACS archive JSON has the wrong shape: {name}")


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


def _zip_info(name: str, *, stored: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    return info


def _write_bytes(
    archive: zipfile.ZipFile,
    *,
    name: str,
    role: str,
    content: bytes,
) -> dict[str, str | int]:
    archive.writestr(_zip_info(name), content)
    return {
        "path": name,
        "role": role,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


async def _read_bounded(
    read_file: ReadRemoteFile,
    path: str,
    *,
    max_bytes: int,
) -> bytes:
    content = bytearray()
    async for chunk in read_file(path):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise ValueError(f"Remote GROMACS file is too large: {path}")
    return bytes(content)


async def _write_remote(
    archive: zipfile.ZipFile,
    *,
    read_file: ReadRemoteFile,
    remote_path: str,
    name: str,
    role: str,
) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with archive.open(
        _zip_info(name, stored=PurePosixPath(name).suffix in {".tpr", ".xtc"}),
        mode="w",
        force_zip64=True,
    ) as destination:
        async for chunk in read_file(remote_path):
            size_bytes += len(chunk)
            digest.update(chunk)
            destination.write(chunk)
    return {
        "path": name,
        "role": role,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


async def write_gromacs_archive(
    handle: object,
    *,
    run_name: str,
    parameters_json: str,
    modal_app_name: str,
    job_id: str,
    stages_json: str,
    started_at: int,
    completed_at: int,
    read_file: ReadRemoteFile,
    run_bounded: RunBounded | None = None,
) -> BuiltGromacsArchive:
    """Package the established GROMACS app's expected Volume files."""
    binary_handle = cast("BinaryIO", handle)
    binary_handle.seek(0)
    binary_handle.truncate(0)
    input_bytes = await _read_bounded(
        read_file,
        f"{run_name}/{run_name}.pdb",
        max_bytes=_MAX_PDB_BYTES,
    )
    parameters_bytes = parameters_json.encode()
    try:
        stages_document = orjson.loads(stages_json)
    except orjson.JSONDecodeError as exc:
        raise ValueError("GROMACS stage history is invalid") from exc
    if not isinstance(stages_document, list):
        raise ValueError("GROMACS stage history is invalid")
    stages_bytes = (
        orjson.dumps(
            stages_document,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    provenance_bytes = (
        orjson.dumps(
            {
                "archive_schema_version": _ARCHIVE_SCHEMA_VERSION,
                "job_id": job_id,
                "tool": "gromacs",
                "modal_app_name": modal_app_name,
                "software_version": "GROMACS 2026.1",
                "started_at": started_at,
                "completed_at": completed_at,
            },
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        + b"\n"
    )
    run_log = (
        f"Biomodals GROMACS job\nrun_name: {run_name}\nstatus: succeeded\n"
    ).encode()

    records: list[dict[str, str | int]] = []
    with zipfile.ZipFile(binary_handle, mode="w", allowZip64=True) as archive:
        records.append(
            _write_bytes(
                archive,
                name="input.pdb",
                role="input_structure",
                content=input_bytes,
            )
        )
        for name, role in _required_output_files(run_name):
            records.append(
                await _write_remote(
                    archive,
                    read_file=read_file,
                    remote_path=f"{run_name}/{PurePosixPath(name).name}",
                    name=name,
                    role=role,
                )
            )
        for name, role, content in (
            ("metadata/parameters.json", "normalized_parameters", parameters_bytes),
            ("metadata/provenance.json", "provenance", provenance_bytes),
            ("metadata/stages.json", "stages", stages_bytes),
            ("metadata/run.log", "run_log", run_log),
        ):
            records.append(_write_bytes(archive, name=name, role=role, content=content))
        for remote_name, name, role in _optional_remote_files(run_name):
            try:
                content = await _read_bounded(
                    read_file,
                    f"{run_name}/{remote_name}",
                    max_bytes=_SMALL_DOCUMENT_LIMIT,
                )
            except FileNotFoundError:
                continue
            records.append(_write_bytes(archive, name=name, role=role, content=content))

        manifest_bytes = (
            orjson.dumps(
                {
                    "archive_schema_version": _ARCHIVE_SCHEMA_VERSION,
                    "run_name": run_name,
                    "files": records,
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
            + b"\n"
        )
        manifest_record = _write_bytes(
            archive,
            name="metadata/manifest.json",
            role="manifest",
            content=manifest_bytes,
        )
        checksums = "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in (*records, manifest_record)
        ).encode()
        _write_bytes(
            archive,
            name="metadata/checksums.sha256",
            role="checksums",
            content=checksums,
        )

    if run_bounded is None:
        validated, size_bytes, sha256 = _validate_and_measure(
            binary_handle,
            run_name,
        )
    else:
        validated, size_bytes, sha256 = await run_bounded(
            _validate_and_measure,
            binary_handle,
            run_name,
        )
    binary_handle.seek(0)
    return BuiltGromacsArchive(
        request_sha256=validated.request_sha256,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _validate_and_measure(
    handle: BinaryIO,
    run_name: str,
) -> tuple[ValidatedGromacsArchive, int, str]:
    """Validate and hash a complete archive in one bounded worker task."""
    validated = validate_gromacs_archive(handle, run_name=run_name)
    handle.seek(0)
    digest = hashlib.sha256()
    size_bytes = 0
    while chunk := handle.read(_CHUNK_SIZE):
        size_bytes += len(chunk)
        digest.update(chunk)
    handle.seek(0)
    return validated, size_bytes, digest.hexdigest()


def _manifest_records(document: object) -> list[dict[str, object]]:
    if (
        not isinstance(document, dict)
        or set(document) != {"archive_schema_version", "run_name", "files"}
        or document.get("archive_schema_version") != _ARCHIVE_SCHEMA_VERSION
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
            if len(names) != len(set(names)):
                raise ValueError("GROMACS result archive has unexpected members")
            if not all(_safe_member(info) for info in infos):
                raise ValueError("GROMACS result archive has an unsafe member")

            try:
                manifest = orjson.loads(_read_small(archive, "metadata/manifest.json"))
            except orjson.JSONDecodeError as exc:
                raise ValueError("GROMACS archive manifest is invalid") from exc
            records = _manifest_records(manifest)
            if manifest.get("run_name") != run_name:
                raise ValueError("GROMACS archive manifest has the wrong run name")

            required_records = dict(_required_manifest_files(run_name))
            optional_records = {
                name: role for _remote, name, role in _optional_remote_files(run_name)
            }
            record_names = [record.get("path") for record in records]
            if not all(isinstance(name, str) for name in record_names):
                raise ValueError("GROMACS archive manifest record is invalid")
            record_names = cast("list[str]", record_names)
            if len(record_names) != len(set(record_names)):
                raise ValueError("GROMACS archive manifest contains duplicates")
            if names != [
                *record_names,
                "metadata/manifest.json",
                "metadata/checksums.sha256",
            ]:
                raise ValueError("GROMACS archive manifest membership is invalid")
            if not set(required_records).issubset(record_names):
                raise ValueError("GROMACS archive manifest is incomplete")
            if not set(record_names).issubset(required_records | optional_records):
                raise ValueError("GROMACS archive manifest has unexpected files")
            digests: dict[str, tuple[int, str]] = {}
            for record in records:
                name = record.get("path")
                if not isinstance(name, str):
                    raise ValueError("GROMACS archive manifest record is invalid")
                role = (required_records | optional_records)[name]
                size_bytes = record.get("size_bytes")
                sha256 = record.get("sha256")
                if (
                    set(record) != {"path", "role", "size_bytes", "sha256"}
                    or record.get("path") != name
                    or record.get("role") != role
                    or type(size_bytes) is not int
                    or size_bytes < 0
                    or (name in required_records and size_bytes == 0)
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

            manifest_bytes = _read_small(archive, "metadata/manifest.json")
            manifest_record = (
                len(manifest_bytes),
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            checksums_bytes = _read_small(archive, "metadata/checksums.sha256")
            expected_checksum_names = [*record_names, "metadata/manifest.json"]
            expected_checksums = "".join(
                f"{(manifest_record[1] if name == 'metadata/manifest.json' else digests[name][1])}  {name}\n"
                for name in expected_checksum_names
            ).encode("ascii")
            if checksums_bytes != expected_checksums:
                raise ValueError("GROMACS archive checksums do not match")

            _validate_required_formats(archive, run_name)
            input_bytes = _read_small(
                archive,
                "input.pdb",
                max_bytes=_MAX_PDB_BYTES,
            )
            parameters_bytes = _read_small(archive, "metadata/parameters.json")
    except zipfile.BadZipFile as exc:
        raise ValueError("GROMACS result archive is invalid") from exc

    request_digest = hashlib.sha256()
    request_digest.update(len(input_bytes).to_bytes(8, byteorder="big"))
    request_digest.update(input_bytes)
    request_digest.update(parameters_bytes)
    return ValidatedGromacsArchive(request_sha256=request_digest.hexdigest())
