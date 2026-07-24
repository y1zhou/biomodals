"""Service-owned packaging for established GROMACS app outputs."""

# ruff: noqa: D103

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import struct
import time
import zipfile
import zlib
from collections.abc import Buffer, Callable
from threading import Event, Thread

import orjson
import pytest

from biomodals.service.artifacts import ArtifactCache, ArtifactSourceMissingError
from biomodals.service.gromacs.archive import (
    GROMACS_ARCHIVE_SCHEMA_VERSION,
    BuiltGromacsArchive,
    validate_gromacs_archive,
    write_gromacs_archive,
)

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
PARAMETERS = '{"cpu_only":false,"simulation_time_ns":5}'
XTC = struct.pack(
    ">iiif9fi3f",
    1995,
    1,
    0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    1,
    0.0,
    0.0,
    0.0,
)
_TPR_VERSION = b"VERSION 2026.1"
TPR = (struct.pack(">II", 15, len(_TPR_VERSION)) + _TPR_VERSION).ljust(1024, b"\0")


def _png_chunk(chunk_type: bytes, content: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(content, checksum)
    return (
        struct.pack(">I", len(content))
        + chunk_type
        + content
        + struct.pack(">I", checksum)
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    + _png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0"))
    + _png_chunk(b"IEND", b"")
)
CENTERED_PDB = PDB + b"END\n"


def _remote_files() -> dict[str, bytes]:
    prefix = f"production_{RUN_NAME}"
    return {
        f"{RUN_NAME}/{RUN_NAME}.pdb": PDB,
        f"{RUN_NAME}/production.mdp": b"integrator = md\n",
        f"{RUN_NAME}/{prefix}.xtc": b"full trajectory",
        f"{RUN_NAME}/{prefix}_nopbc.xtc": XTC,
        f"{RUN_NAME}/{prefix}.tpr": TPR,
        f"{RUN_NAME}/{prefix}_nopbc_centered.pdb": CENTERED_PDB,
        f"{RUN_NAME}/rmsd_{prefix}.csv": b"time_ns,rmsd\n0.0,0.1\n",
        f"{RUN_NAME}/rmsd_{prefix}.png": PNG,
        f"{RUN_NAME}/rg_{prefix}.csv": b"time_ns,rg\n0.0,1.2\n",
        f"{RUN_NAME}/rg_{prefix}.png": PNG,
        f"{RUN_NAME}/rmsf_{prefix}.csv": b"residue_index,rmsf\n1,0.2\n",
        f"{RUN_NAME}/rmsf_{prefix}.png": PNG,
    }


def _mtimes_for_files(remote_files: dict[str, bytes]) -> dict[str, int]:
    return dict.fromkeys(remote_files, 1_700_000_000)


def _build_archive(
    *,
    remote_files: dict[str, bytes] | None = None,
    remote_mtimes: dict[str, int] | None = None,
) -> tuple[bytes, BuiltGromacsArchive]:
    files = _remote_files() if remote_files is None else remote_files

    async def read_file(path: str):
        try:
            content = files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        midpoint = len(content) // 2
        yield content[:midpoint]
        yield content[midpoint:]

    output = io.BytesIO()
    result = asyncio.run(
        write_gromacs_archive(
            output,
            run_name=RUN_NAME,
            parameters_json=PARAMETERS,
            modal_app_name="Gromacs",
            modal_app_version=17,
            job_id="11111111-1111-4111-8111-111111111111",
            stages_json="[]",
            started_at=1,
            completed_at=2,
            read_file=read_file,
            remote_mtimes=(
                _mtimes_for_files(files) if remote_mtimes is None else remote_mtimes
            ),
        )
    )
    return output.getvalue(), result


def _rewrite_archive_member(
    archive_bytes: bytes,
    name: str,
    update: Callable[[zipfile.ZipInfo], None],
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        members = [(copy.copy(info), source.read(info)) for info in source.infolist()]
    with zipfile.ZipFile(output, "w") as archive:
        for info, content in members:
            if info.filename == name:
                update(info)
            archive.writestr(info, content)
    return output.getvalue()


def _rewrite_local_header(
    archive_bytes: bytes,
    name: str,
    update: Callable[[bytearray, int], None],
) -> bytes:
    mutated = bytearray(archive_bytes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        offset = archive.getinfo(name).header_offset
    update(mutated, offset)
    return bytes(mutated)


def test_service_packages_established_remote_files_deterministically() -> None:
    prefix = f"production_{RUN_NAME}"
    first_bytes, first = _build_archive()
    second_bytes, second = _build_archive()

    assert first_bytes == second_bytes
    assert first == second
    assert first.size_bytes == len(first_bytes)
    assert first.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert validate_gromacs_archive(io.BytesIO(first_bytes), run_name=RUN_NAME)
    with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_STORED
        }
        assert archive.read("input.pdb") == PDB
        assert f"outputs/{prefix}.xtc" not in archive.namelist()
        assert archive.read(f"outputs/{prefix}_nopbc.xtc") == XTC
        assert archive.read(f"outputs/rmsd_{prefix}.png") == PNG
        assert archive.read(f"outputs/rg_{prefix}.png") == PNG
        assert archive.read(f"outputs/rmsf_{prefix}.png") == PNG
        assert archive.read("metadata/parameters.json") == PARAMETERS.encode()
        provenance = orjson.loads(archive.read("metadata/provenance.json"))
        assert provenance["archive_schema_version"] == 4
        assert GROMACS_ARCHIVE_SCHEMA_VERSION == 4
        assert provenance["modal_app_version"] == 17
        assert provenance["software_version"] == "GROMACS 2026.1"
        assert {name.split("/", 1)[0] for name in archive.namelist()} == {
            "input.pdb",
            "outputs",
            "metadata",
        }


@pytest.mark.parametrize(
    "blocking_content",
    [PDB, XTC],
    ids=["input-pdb", "remote-output"],
)
def test_archive_writes_do_not_block_the_event_loop(
    tmp_path,
    blocking_content: bytes,
) -> None:
    files = _remote_files()
    write_started = Event()
    release_write = Event()

    class BlockingBuffer(io.BytesIO):
        def write(self, content: Buffer, /) -> int:
            if bytes(content) == blocking_content and not write_started.is_set():
                write_started.set()
                if not release_write.wait(timeout=5):
                    raise RuntimeError("test archive write timed out")
            return super().write(content)

    async def read_file(path: str):
        yield files[path]

    async def scenario() -> None:
        cache = ArtifactCache(tmp_path / "cache")
        loop = asyncio.get_running_loop()
        heartbeat = asyncio.Event()
        heartbeat_observed = Event()
        responsive: list[bool] = []

        async def observe_heartbeat() -> None:
            await heartbeat.wait()
            heartbeat_observed.set()

        def probe() -> None:
            if not write_started.wait(timeout=5):
                responsive.append(False)
                release_write.set()
                return
            loop.call_soon_threadsafe(heartbeat.set)
            responsive.append(heartbeat_observed.wait(timeout=0.5))
            release_write.set()

        observer = asyncio.create_task(observe_heartbeat())
        probe_thread = Thread(target=probe)
        probe_thread.start()
        try:
            await write_gromacs_archive(
                BlockingBuffer(),
                run_name=RUN_NAME,
                parameters_json=PARAMETERS,
                modal_app_name="Gromacs",
                modal_app_version=17,
                job_id="11111111-1111-4111-8111-111111111111",
                stages_json="[]",
                started_at=1,
                completed_at=2,
                read_file=read_file,
                remote_mtimes=_mtimes_for_files(files),
                run_bounded=cache.run_bounded,
            )
            await observer
            probe_thread.join(timeout=5)
            assert responsive == [True]
        finally:
            release_write.set()
            await cache.shutdown()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("rmsd", b"time_ns,rmsd\n0.0,nan\n"),
        ("rg", b"time_ns,rg\n0.0,inf\n"),
        ("rmsd", b"time_ns,rmsd\n0.0,0.1\n0.0,0.2\n"),
        ("rg", b"time_ns,rg\n1.0,1.2\n0.5,1.3\n"),
        ("rmsf", b"residue_index,rmsf\n1.5,0.2\n"),
        ("rmsd", b"time_ns,rmsd\n0.0,-0.1\n"),
        ("rg", b"time_ns,rg\n0.0,-1.2\n"),
        ("rmsf", b"residue_index,rmsf\n1,-0.2\n"),
    ],
)
def test_archive_validator_rejects_invalid_analysis_axes_and_values(
    relative_path: str,
    content: bytes,
) -> None:
    remote_files = _remote_files()
    prefix = f"production_{RUN_NAME}"
    remote_files[f"{RUN_NAME}/{relative_path}_{prefix}.csv"] = content

    with pytest.raises(ValueError, match="wrong schema"):
        _build_archive(remote_files=remote_files)


def test_service_preserves_remote_file_modification_times() -> None:
    prefix = f"production_{RUN_NAME}"
    remote_files = _remote_files()
    diagnostic_path = f"{RUN_NAME}/{prefix}.log"
    remote_files[diagnostic_path] = b"production log\n"
    remote_mtimes = {
        path: 1_718_372_121 + index * 120 for index, path in enumerate(remote_files)
    }

    async def read_file(path: str):
        try:
            yield remote_files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    output = io.BytesIO()
    asyncio.run(
        write_gromacs_archive(
            output,
            run_name=RUN_NAME,
            parameters_json=PARAMETERS,
            modal_app_name="Gromacs",
            modal_app_version=17,
            job_id="11111111-1111-4111-8111-111111111111",
            stages_json="[]",
            started_at=1,
            completed_at=2,
            read_file=read_file,
            remote_mtimes=remote_mtimes,
        )
    )

    expected = {
        "input.pdb": remote_mtimes[f"{RUN_NAME}/{RUN_NAME}.pdb"],
        f"outputs/{prefix}_nopbc.xtc": remote_mtimes[f"{RUN_NAME}/{prefix}_nopbc.xtc"],
        f"metadata/gromacs/{prefix}.log": remote_mtimes[diagnostic_path],
    }
    with zipfile.ZipFile(output) as archive:
        for name, mtime in expected.items():
            info = archive.getinfo(name)
            expected_dos_time = time.gmtime(mtime)[:6]
            expected_dos_time = (*expected_dos_time[:5], expected_dos_time[5] // 2 * 2)
            assert info.date_time == expected_dos_time
            assert info.extra == struct.pack("<HHBI", 0x5455, 5, 1, mtime)


def test_archive_validator_enforces_schema_four_zip_metadata() -> None:
    archive_bytes, _result = _build_archive()

    def deflate(info: zipfile.ZipInfo) -> None:
        info.compress_type = zipfile.ZIP_DEFLATED

    def remove_timestamp(info: zipfile.ZipInfo) -> None:
        info.extra = b""

    def mismatch_timestamp(info: zipfile.ZipInfo) -> None:
        info.extra = struct.pack("<HHBI", 0x5455, 5, 3, 1_700_000_000)

    def mismatch_dos_time(info: zipfile.ZipInfo) -> None:
        info.date_time = (2023, 1, 1, 0, 0, 0)

    mutations = (
        ("metadata/parameters.json", deflate, "stored"),
        ("input.pdb", remove_timestamp, "timestamp"),
        ("input.pdb", mismatch_timestamp, "timestamp"),
        ("input.pdb", mismatch_dos_time, "timestamp"),
    )
    for name, update, message in mutations:
        mutated = _rewrite_archive_member(archive_bytes, name, update)
        with pytest.raises(ValueError, match=message):
            validate_gromacs_archive(io.BytesIO(mutated), run_name=RUN_NAME)


def test_archive_validator_enforces_local_header_zip_metadata() -> None:
    archive_bytes, _result = _build_archive()

    def deflate_locally(content: bytearray, offset: int) -> None:
        struct.pack_into("<H", content, offset + 8, zipfile.ZIP_DEFLATED)

    def remove_local_timestamp_flag(content: bytearray, offset: int) -> None:
        name_length, _extra_length = struct.unpack_from("<HH", content, offset + 26)
        extra_offset = offset + 30 + name_length
        header_id, size = struct.unpack_from("<HH", content, extra_offset)
        assert (header_id, size) == (0x5455, 5)
        content[extra_offset + 4] = 0

    def mismatch_local_dos_time(content: bytearray, offset: int) -> None:
        dos_time = struct.unpack_from("<H", content, offset + 10)[0]
        struct.pack_into("<H", content, offset + 10, dos_time ^ 1)

    mutations = (
        (deflate_locally, "stored"),
        (remove_local_timestamp_flag, "timestamp"),
        (mismatch_local_dos_time, "timestamp"),
    )
    for update, message in mutations:
        mutated = _rewrite_local_header(archive_bytes, "input.pdb", update)
        with pytest.raises(ValueError, match=message):
            validate_gromacs_archive(io.BytesIO(mutated), run_name=RUN_NAME)


def test_archive_enforces_signed_extended_timestamp_range() -> None:
    remote_files = _remote_files()
    remote_mtimes = _mtimes_for_files(remote_files)
    input_path = f"{RUN_NAME}/{RUN_NAME}.pdb"
    remote_mtimes[input_path] = 0x7FFFFFFF

    archive_bytes, _result = _build_archive(
        remote_files=remote_files,
        remote_mtimes=remote_mtimes,
    )
    assert validate_gromacs_archive(io.BytesIO(archive_bytes), run_name=RUN_NAME)

    remote_mtimes[input_path] = -0x80000000
    archive_bytes, _result = _build_archive(
        remote_files=remote_files,
        remote_mtimes=remote_mtimes,
    )
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert archive.getinfo("input.pdb").extra == struct.pack(
            "<HHBi",
            0x5455,
            5,
            1,
            -0x80000000,
        )

    remote_mtimes[input_path] = 0x80000000
    with pytest.raises(ValueError, match="signed 32-bit"):
        _build_archive(
            remote_files=remote_files,
            remote_mtimes=remote_mtimes,
        )

    remote_mtimes[input_path] = -0x80000001
    with pytest.raises(ValueError, match="signed 32-bit"):
        _build_archive(
            remote_files=remote_files,
            remote_mtimes=remote_mtimes,
        )


@pytest.mark.parametrize(
    ("remote_name", "content", "message"),
    [
        ("trajectory", b"", "manifest record"),
        ("trajectory", b"not-an-xtc", "trajectory is invalid"),
        (
            "trajectory",
            struct.pack(">i", 1995) + b"\0" * 28,
            "trajectory is invalid",
        ),
        (
            "topology",
            b"\0\0\0\x10VERSION 2026.1\0\0\0\0",
            "topology is invalid",
        ),
        ("topology", TPR[:128], "topology is invalid"),
        ("rmsd_plot", b"not-a-png", "PNG is invalid"),
        (
            "rmsd_plot",
            b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + b"\0" * 16,
            "PNG is invalid",
        ),
        (
            "rmsd_plot",
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
            + _png_chunk(b"IDAT", b"")
            + _png_chunk(b"IEND", b""),
            "PNG is invalid",
        ),
        ("rmsd", b"time_ns,rmsd\n", "wrong schema"),
        ("rmsd", b"time_ns,rmsd\n0.0,not-a-number\n", "wrong schema"),
    ],
)
def test_mandatory_scientific_outputs_must_be_nonempty_and_structurally_valid(
    remote_name: str,
    content: bytes,
    message: str,
) -> None:
    prefix = f"production_{RUN_NAME}"
    remote_paths = {
        "trajectory": f"{RUN_NAME}/{prefix}_nopbc.xtc",
        "topology": f"{RUN_NAME}/{prefix}.tpr",
        "rmsd": f"{RUN_NAME}/rmsd_{prefix}.csv",
        "rmsd_plot": f"{RUN_NAME}/rmsd_{prefix}.png",
    }
    remote_files = _remote_files()
    remote_files[remote_paths[remote_name]] = content

    async def read_file(path: str):
        try:
            content = remote_files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        yield content

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            write_gromacs_archive(
                io.BytesIO(),
                run_name=RUN_NAME,
                parameters_json=PARAMETERS,
                modal_app_name="Gromacs",
                modal_app_version=17,
                job_id="11111111-1111-4111-8111-111111111111",
                stages_json="[]",
                started_at=1,
                completed_at=2,
                read_file=read_file,
                remote_mtimes=_mtimes_for_files(remote_files),
            )
        )


def test_large_centered_structure_and_diagnostics_stream_without_a_size_cap() -> None:
    prefix = f"production_{RUN_NAME}"
    remote_files = _remote_files()
    centered = PDB * 20_000 + b"END\n"
    diagnostic = b"production log line\n" * 70_000
    assert len(centered) > 1024 * 1024
    assert len(diagnostic) > 1024 * 1024
    remote_files[f"{RUN_NAME}/{prefix}_nopbc_centered.pdb"] = centered
    remote_files[f"{RUN_NAME}/{prefix}.log"] = diagnostic

    async def read_file(path: str):
        try:
            content = remote_files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        for offset in range(0, len(content), 64 * 1024):
            yield content[offset : offset + 64 * 1024]

    output = io.BytesIO()
    asyncio.run(
        write_gromacs_archive(
            output,
            run_name=RUN_NAME,
            parameters_json=PARAMETERS,
            modal_app_name="Gromacs",
            modal_app_version=17,
            job_id="11111111-1111-4111-8111-111111111111",
            stages_json="[]",
            started_at=1,
            completed_at=2,
            read_file=read_file,
            remote_mtimes=_mtimes_for_files(remote_files),
        )
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.getinfo(f"outputs/{prefix}_nopbc_centered.pdb").file_size == len(
            centered
        )
        assert archive.getinfo(f"metadata/gromacs/{prefix}.log").file_size == len(
            diagnostic
        )


def test_missing_required_remote_output_has_a_distinct_failure() -> None:
    remote_files = _remote_files()
    del remote_files[f"{RUN_NAME}/production.mdp"]

    async def read_file(path: str):
        try:
            content = remote_files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        yield content

    with pytest.raises(ArtifactSourceMissingError):
        asyncio.run(
            write_gromacs_archive(
                io.BytesIO(),
                run_name=RUN_NAME,
                parameters_json=PARAMETERS,
                modal_app_name="Gromacs",
                modal_app_version=17,
                job_id="11111111-1111-4111-8111-111111111111",
                stages_json="[]",
                started_at=1,
                completed_at=2,
                read_file=read_file,
                remote_mtimes=_mtimes_for_files(remote_files),
            )
        )
