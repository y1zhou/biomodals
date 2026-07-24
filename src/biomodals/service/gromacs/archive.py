"""Validation for the immutable GROMACS API result archive."""

from __future__ import annotations

import hashlib
import stat
import struct
import time
import zipfile
import zlib
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO, Protocol, TypeVar, cast

import orjson
import polars as pl

from biomodals.service.artifacts import ArtifactSourceMissingError
from biomodals.service.gromacs.contracts import artifact_request_sha256

_CHUNK_SIZE = 1024 * 1024
_SHA256_LENGTH = 64
_SMALL_DOCUMENT_LIMIT = 1024 * 1024
_MAX_PDB_BYTES = 10 * 1024 * 1024
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_EXTENDED_TIMESTAMP_HEADER_ID = 0x5455
_MIN_EXTENDED_TIMESTAMP = -0x80000000
_MAX_EXTENDED_TIMESTAMP = 0x7FFFFFFF
_LOCAL_HEADER = struct.Struct("<4s5H3I2H")
_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"

ReadRemoteFile = Callable[[str], AsyncIterable[bytes]]
_T = TypeVar("_T")


class RunBounded(Protocol):
    """Execute blocking artifact I/O outside the event loop."""

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


GROMACS_ARCHIVE_SCHEMA_VERSION = 4


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


def _tpr_software_version(prefix: bytes) -> str:
    """Recover the GROMACS version string embedded in a TPR XDR header."""
    version_length = struct.unpack(">I", prefix[4:8])[0] if len(prefix) >= 8 else 0
    if (
        version_length < len("VERSION")
        or version_length > 120
        or len(prefix) < 8 + version_length
        or not prefix[8 : 8 + version_length].startswith(b"VERSION")
    ):
        raise ValueError("GROMACS production topology is invalid")
    try:
        version = prefix[8 : 8 + version_length].decode("ascii").removeprefix("VERSION")
    except UnicodeDecodeError as exc:
        raise ValueError("GROMACS production topology is invalid") from exc
    return f"GROMACS {version.strip()}".rstrip()


def _validate_pdb_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int | None = None,
) -> None:
    info = archive.getinfo(name)
    if max_bytes is not None and info.file_size > max_bytes:
        raise ValueError(f"GROMACS archive member is too large: {name}")
    has_atom = False
    with archive.open(info) as member:
        for raw_line in member:
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(f"GROMACS archive PDB is invalid: {name}") from exc
            has_atom = has_atom or line.startswith(("ATOM  ", "HETATM"))
    if not has_atom:
        raise ValueError(f"GROMACS archive PDB has no atoms: {name}")


def _validate_csv_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    header: tuple[str, str],
) -> None:
    try:
        with archive.open(name) as member:
            frame = pl.read_csv(
                member,
                schema_overrides={column: pl.Float64 for column in header},
            )
        if (
            frame.columns != list(header)
            or frame.height == 0
            or any(frame.null_count().row(0))
            or any(not column.is_finite().all() for column in frame.iter_columns())
        ):
            raise ValueError
        axis = frame.get_column(header[0])
        values = frame.get_column(header[1])
        if (
            cast("float", axis.min()) < (1 if header[0] == "residue_index" else 0)
            or axis.diff().drop_nulls().le(0).any()
            or (header[0] == "residue_index" and axis.ne(axis.floor()).any())
            or values.lt(0).any()
        ):
            raise ValueError
    except (pl.exceptions.PolarsError, ValueError) as exc:
        raise ValueError(f"GROMACS archive CSV has the wrong schema: {name}") from exc


def _validate_png_member(archive: zipfile.ZipFile, name: str) -> None:
    """Validate one PNG envelope and every chunk CRC without decoding pixels."""
    try:
        info = archive.getinfo(name)
        with archive.open(info) as member:
            if member.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError
            first_chunk = True
            has_image_data = False
            while True:
                header = member.read(8)
                if len(header) != 8:
                    raise ValueError
                length, chunk_type = struct.unpack(">I4s", header)
                if length > info.file_size:
                    raise ValueError
                crc = zlib.crc32(chunk_type)
                remaining = length
                ihdr = bytearray()
                while remaining:
                    content = member.read(min(_CHUNK_SIZE, remaining))
                    if not content:
                        raise ValueError
                    if chunk_type == b"IHDR":
                        ihdr.extend(content)
                    crc = zlib.crc32(content, crc)
                    remaining -= len(content)
                expected_crc = member.read(4)
                if (
                    len(expected_crc) != 4
                    or struct.unpack(">I", expected_crc)[0] != crc
                ):
                    raise ValueError
                if first_chunk:
                    if chunk_type != b"IHDR" or length != 13:
                        raise ValueError
                    width, height = struct.unpack(">II", ihdr[:8])
                    if (
                        width == 0
                        or height == 0
                        or ihdr[10] != 0
                        or ihdr[11] != 0
                        or ihdr[12] not in {0, 1}
                    ):
                        raise ValueError
                    first_chunk = False
                elif chunk_type == b"IHDR":
                    raise ValueError
                if chunk_type == b"IDAT" and length > 0:
                    has_image_data = True
                if chunk_type == b"IEND":
                    if length != 0 or not has_image_data or member.read(1):
                        raise ValueError
                    return
    except (IndexError, struct.error, ValueError) as exc:
        raise ValueError(f"GROMACS archive PNG is invalid: {name}") from exc


def _validate_xtc_member(archive: zipfile.ZipFile, name: str) -> None:
    """Validate the fixed XDR header and complete first coordinate envelope."""
    info = archive.getinfo(name)
    prefix = _read_prefix(archive, name, 92)
    if len(prefix) < 56:
        raise ValueError("GROMACS trajectory is invalid")
    magic, atom_count = struct.unpack(">ii", prefix[:8])
    coordinate_count = struct.unpack(">i", prefix[52:56])[0]
    if magic != 1995 or atom_count < 1 or coordinate_count != atom_count:
        raise ValueError("GROMACS trajectory is invalid")
    if atom_count <= 9:
        minimum_size = 56 + 12 * atom_count
    else:
        if len(prefix) < 92:
            raise ValueError("GROMACS trajectory is invalid")
        payload_size = struct.unpack(">I", prefix[88:92])[0]
        if payload_size == 0:
            raise ValueError("GROMACS trajectory is invalid")
        minimum_size = 92 + ((payload_size + 3) & ~3)
    if info.file_size < minimum_size:
        raise ValueError("GROMACS trajectory is invalid")


def _validate_required_formats(archive: zipfile.ZipFile, run_name: str) -> None:
    """Reject inexpensive-to-detect truncation before publishing success."""
    prefix = f"production_{run_name}"
    input_name = "input.pdb"
    centered_name = f"outputs/{prefix}_nopbc_centered.pdb"
    _validate_pdb_member(archive, input_name, max_bytes=_MAX_PDB_BYTES)
    _validate_pdb_member(archive, centered_name)

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
    _validate_xtc_member(archive, xtc_name)

    tpr_name = f"outputs/{prefix}.tpr"
    tpr_prefix = _read_prefix(archive, tpr_name)
    if archive.getinfo(tpr_name).file_size < 1024:
        raise ValueError("GROMACS production topology is invalid")
    _tpr_software_version(tpr_prefix)

    csv_contracts = (
        (f"outputs/rmsd_{prefix}.csv", ("time_ns", "rmsd")),
        (f"outputs/rg_{prefix}.csv", ("time_ns", "rg")),
        (f"outputs/rmsf_{prefix}.csv", ("residue_index", "rmsf")),
    )
    for name, header in csv_contracts:
        _validate_csv_member(archive, name, header=header)

    for name in (
        f"outputs/rmsd_{prefix}.png",
        f"outputs/rg_{prefix}.png",
        f"outputs/rmsf_{prefix}.png",
    ):
        _validate_png_member(archive, name)

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


def _zip_info(name: str, *, mtime: int | None = None) -> zipfile.ZipInfo:
    if mtime is None:
        date_time = _ZIP_EPOCH
    else:
        if (
            type(mtime) is not int
            or not _MIN_EXTENDED_TIMESTAMP <= mtime <= _MAX_EXTENDED_TIMESTAMP
        ):
            raise ValueError(
                "Remote GROMACS file modification time must fit a signed 32-bit "
                "extended timestamp"
            )
        date_time = _dos_timestamp(mtime)
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    info.compress_type = zipfile.ZIP_STORED
    if mtime is not None:
        info.extra = struct.pack(
            "<HHBi",
            _EXTENDED_TIMESTAMP_HEADER_ID,
            5,
            1,
            mtime,
        )
    return info


def _source_member(name: str) -> bool:
    return (
        name == "input.pdb"
        or name.startswith("outputs/")
        or name.startswith("metadata/gromacs/")
    )


def _dos_timestamp(mtime: int) -> tuple[int, int, int, int, int, int]:
    source_time = time.gmtime(mtime)[:6]
    date_time = source_time if source_time[0] >= _ZIP_EPOCH[0] else _ZIP_EPOCH
    return (*date_time[:5], date_time[5] // 2 * 2)


def _encoded_dos_timestamp(
    date_time: tuple[int, int, int, int, int, int],
) -> tuple[int, int]:
    year, month, day, hour, minute, second = date_time
    return (
        hour << 11 | minute << 5 | second // 2,
        (year - _ZIP_EPOCH[0]) << 9 | month << 5 | day,
    )


def _extra_fields(extra: bytes) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            raise ValueError("GROMACS archive ZIP extra fields are invalid")
        header_id, size = struct.unpack_from("<HH", extra, offset)
        data_offset = offset + 4
        offset = data_offset + size
        if offset > len(extra):
            raise ValueError("GROMACS archive ZIP extra fields are invalid")
        if header_id in fields:
            raise ValueError("GROMACS archive ZIP extra fields are invalid")
        fields[header_id] = extra[data_offset:offset]
    return fields


def _local_zip_metadata(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[int, int, int, bytes]:
    source = archive.fp
    if source is None:
        raise ValueError("GROMACS archive local ZIP header is unavailable")
    previous_offset = source.tell()
    try:
        source.seek(info.header_offset)
        header = source.read(_LOCAL_HEADER.size)
        if len(header) != _LOCAL_HEADER.size:
            raise ValueError("GROMACS archive local ZIP header is invalid")
        (
            signature,
            _extract_version,
            flags,
            compression,
            dos_time,
            dos_date,
            _crc,
            _compressed_size,
            _file_size,
            name_length,
            extra_length,
        ) = _LOCAL_HEADER.unpack(header)
        local_name = source.read(name_length)
        local_extra = source.read(extra_length)
    finally:
        source.seek(previous_offset)
    expected_name = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    if (
        signature != _LOCAL_HEADER_SIGNATURE
        or flags != info.flag_bits
        or local_name != expected_name
        or len(local_extra) != extra_length
    ):
        raise ValueError("GROMACS archive local ZIP header is invalid")
    return compression, dos_time, dos_date, local_extra


def _source_mtime(info: zipfile.ZipInfo) -> int:
    fields = _extra_fields(info.extra)
    if set(fields) - {0x0001, _EXTENDED_TIMESTAMP_HEADER_ID}:
        raise ValueError("GROMACS archive source timestamp metadata is invalid")
    try:
        flags, mtime = struct.unpack(
            "<Bi",
            fields[_EXTENDED_TIMESTAMP_HEADER_ID],
        )
    except struct.error as exc:
        raise ValueError(
            "GROMACS archive source timestamp metadata is invalid"
        ) from exc
    except KeyError as exc:
        raise ValueError(
            "GROMACS archive source timestamp metadata is invalid"
        ) from exc
    if flags != 1 or info.date_time != _dos_timestamp(mtime):
        raise ValueError("GROMACS archive source timestamp metadata is invalid")
    return mtime


def _validate_zip_metadata(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
) -> None:
    for info in infos:
        local_compression, local_time, local_date, local_extra = _local_zip_metadata(
            archive,
            info,
        )
        expected_time, expected_date = _encoded_dos_timestamp(info.date_time)
        if (
            info.compress_type != zipfile.ZIP_STORED
            or local_compression != zipfile.ZIP_STORED
        ):
            raise ValueError("GROMACS archive members must use the stored ZIP method")
        if (local_time, local_date) != (expected_time, expected_date):
            raise ValueError("GROMACS archive local timestamp metadata is invalid")
        if _source_member(info.filename):
            mtime = _source_mtime(info)
            local_fields = _extra_fields(local_extra)
            if set(local_fields) - {
                0x0001,
                _EXTENDED_TIMESTAMP_HEADER_ID,
            } or local_fields.get(_EXTENDED_TIMESTAMP_HEADER_ID) != struct.pack(
                "<Bi", 1, mtime
            ):
                raise ValueError("GROMACS archive source timestamp metadata is invalid")
        elif info.extra or info.date_time != _ZIP_EPOCH or local_extra:
            raise ValueError("GROMACS archive generated timestamp metadata is invalid")


def _write_bytes(
    archive: zipfile.ZipFile,
    *,
    name: str,
    role: str,
    content: bytes,
    mtime: int | None = None,
) -> dict[str, str | int]:
    archive.writestr(_zip_info(name, mtime=mtime), content)
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
    mtime: int,
    remote_path: str,
    name: str,
    role: str,
    run_bounded: RunBounded | None,
) -> dict[str, str | int]:
    return await _write_remote_chunks(
        archive,
        chunks=read_file(remote_path),
        mtime=mtime,
        name=name,
        role=role,
        run_bounded=run_bounded,
    )


async def _write_remote_chunks(
    archive: zipfile.ZipFile,
    *,
    chunks: AsyncIterable[bytes],
    mtime: int,
    name: str,
    role: str,
    run_bounded: RunBounded | None,
) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with archive.open(
        _zip_info(name, mtime=mtime),
        mode="w",
        force_zip64=True,
    ) as destination:

        def write_chunk(chunk: bytes) -> None:
            digest.update(chunk)
            destination.write(chunk)

        async for chunk in chunks:
            size_bytes += len(chunk)
            if run_bounded is None:
                write_chunk(chunk)
            else:
                await run_bounded(write_chunk, chunk)
    return {
        "path": name,
        "role": role,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


async def _write_optional_remote(
    archive: zipfile.ZipFile,
    *,
    read_file: ReadRemoteFile,
    remote_mtimes: Mapping[str, int],
    remote_path: str,
    name: str,
    role: str,
    run_bounded: RunBounded | None,
) -> dict[str, str | int] | None:
    try:
        mtime = remote_mtimes[remote_path]
    except KeyError:
        return None
    chunks = read_file(remote_path).__aiter__()
    try:
        first = await anext(chunks)
    except FileNotFoundError:
        return None
    except StopAsyncIteration:
        first = None

    async def with_first() -> AsyncIterator[bytes]:
        if first is not None:
            yield first
        async for chunk in chunks:
            yield chunk

    return await _write_remote_chunks(
        archive,
        chunks=with_first(),
        mtime=mtime,
        name=name,
        role=role,
        run_bounded=run_bounded,
    )


async def write_gromacs_archive(
    handle: object,
    *,
    run_name: str,
    parameters_json: str,
    modal_app_name: str,
    modal_app_version: int,
    job_id: str,
    stages_json: str,
    started_at: int,
    completed_at: int,
    read_file: ReadRemoteFile,
    remote_mtimes: Mapping[str, int],
    run_bounded: RunBounded | None = None,
) -> BuiltGromacsArchive:
    """Package the established GROMACS app's expected Volume files."""

    async def read_required(path: str) -> AsyncIterator[bytes]:
        try:
            async for chunk in read_file(path):
                yield chunk
        except FileNotFoundError as exc:
            raise ArtifactSourceMissingError(
                "A required GROMACS output is missing"
            ) from exc

    def required_mtime(path: str) -> int:
        try:
            return remote_mtimes[path]
        except KeyError as exc:
            raise ArtifactSourceMissingError(
                "A required GROMACS output is missing"
            ) from exc

    binary_handle = cast("BinaryIO", handle)
    binary_handle.seek(0)
    binary_handle.truncate(0)
    input_path = f"{run_name}/{run_name}.pdb"
    input_bytes = await _read_bounded(
        read_required,
        input_path,
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
                mtime=required_mtime(input_path),
            )
        )
        for name, role in _required_output_files(run_name):
            remote_path = f"{run_name}/{PurePosixPath(name).name}"
            records.append(
                await _write_remote(
                    archive,
                    read_file=read_required,
                    mtime=required_mtime(remote_path),
                    remote_path=remote_path,
                    name=name,
                    role=role,
                    run_bounded=run_bounded,
                )
            )
        topology_name = f"outputs/production_{run_name}.tpr"
        software_version = _tpr_software_version(_read_prefix(archive, topology_name))
        provenance_bytes = (
            orjson.dumps(
                {
                    "archive_schema_version": GROMACS_ARCHIVE_SCHEMA_VERSION,
                    "job_id": job_id,
                    "tool": "gromacs",
                    "modal_app_name": modal_app_name,
                    "modal_app_version": modal_app_version,
                    "software_version": software_version,
                    "started_at": started_at,
                    "completed_at": completed_at,
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
            + b"\n"
        )
        for name, role, content in (
            ("metadata/parameters.json", "normalized_parameters", parameters_bytes),
            ("metadata/provenance.json", "provenance", provenance_bytes),
            ("metadata/stages.json", "stages", stages_bytes),
            ("metadata/run.log", "run_log", run_log),
        ):
            records.append(_write_bytes(archive, name=name, role=role, content=content))
        for remote_name, name, role in _optional_remote_files(run_name):
            record = await _write_optional_remote(
                archive,
                read_file=read_file,
                remote_mtimes=remote_mtimes,
                remote_path=f"{run_name}/{remote_name}",
                name=name,
                role=role,
                run_bounded=run_bounded,
            )
            if record is not None:
                records.append(record)

        manifest_bytes = (
            orjson.dumps(
                {
                    "archive_schema_version": GROMACS_ARCHIVE_SCHEMA_VERSION,
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
        or document.get("archive_schema_version") != GROMACS_ARCHIVE_SCHEMA_VERSION
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
            _validate_zip_metadata(archive, infos)

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

    return ValidatedGromacsArchive(
        request_sha256=artifact_request_sha256(
            input_bytes,
            parameters_bytes.decode(),
        )
    )
