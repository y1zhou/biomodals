"""Run MD simulation with GROMACS: <https://www.gromacs.org/>.

**It is recommended to run this app in detached mode, as the runs can be very long.**

## Outputs

* All output files are saved to a Modal volume named `Gromacs-outputs`.
* The production trajectory should be under the name `production_{run_name}.xtc`.
"""
# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import hashlib
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import (
    AppRunLayout,
    volume_app_output,
)
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.pdb import validate_pdb_content
from biomodals.helper.shell import run_command, sanitize_filename, warmup_directory
from biomodals.helper.task_budget import bounded_map
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Gromacs",
    repo_url="https://github.com/gromacs/gromacs",
    version="2026.1",
    python_version="3.13",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", MAX_TIMEOUT)),
)


@dataclass
class AppInfo:
    """Container for Gromacs-specific configuration and constants."""

    # Build configs
    gmx_scripts: str = "/biomodals-gromacs-scripts"
    gmx_threads: int = int(os.environ.get("N_GMX_THREADS", "16"))
    # Dependency versions
    ucx_tag: str = "1.20.0"
    openmpi_tag: str = "5.0.9"
    fftw_tag: str = "3.3.10"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()

_API_ARCHIVE_SCHEMA_VERSION = 1
_API_PUBLICATION_SCHEMA_VERSION = 1
_API_RESULTS_DIR = "api-results"
_API_ARCHIVE_FILENAME = "result.zip"
_API_RESULT_MARKER_FILENAME = "result.json"
_ARCHIVE_CHUNK_SIZE = 1024 * 1024
_RUN_LOG_TAIL_BYTES = 1024 * 1024


def prepared_workflow_files(run_name: str) -> list[ArtifactFile]:
    """Return expected workflow files from a GROMACS preparation run."""
    return [
        ArtifactFile(path=f"{run_name}.pdb", role="input_structure"),
        ArtifactFile(path=f"production_{run_name}.tpr", role="production_topology"),
        ArtifactFile(path="production.mdp", role="production_parameters"),
    ]


def production_workflow_files(run_name: str) -> list[ArtifactFile]:
    """Return expected workflow files from a GROMACS production run."""
    prefix = f"production_{run_name}"
    return [
        ArtifactFile(path=f"{prefix}.xtc", role="trajectory"),
        ArtifactFile(path=f"{prefix}.tpr", role="production_topology"),
        ArtifactFile(path=f"{prefix}_nopbc_centered.pdb", role="centered_structure"),
        ArtifactFile(path=f"rmsd_{prefix}.csv", role="rmsd"),
        ArtifactFile(path=f"rg_{prefix}.csv", role="radius_of_gyration"),
        ArtifactFile(path=f"rmsf_{prefix}.csv", role="rmsf"),
    ]


runtime_image = (
    modal.Image
    .from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python=CONF.python_version
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "wget",
        "libboost-dev",
        "zlib1g",
        "zlib1g-dev",
        "libsqlite3-dev",
        "libopenblas-dev",
        "unzip",
        "libgomp1",
        "liblapack3",
    )
    .env(CONF.default_env | {"PATH": "/root/.local/bin:$PATH"})
    .run_commands("curl -L micro.mamba.pm/install.sh | bash")
    .micromamba_install(
        "ambertools=23", "pdbfixer", channels=["conda-forge", "bioconda"]
    )
    # Follow https://manual.gromacs.org/2024.5/install-guide/index.html#gpu-aware-mpi-support
    .workdir("/opt")
    # Build UCX
    .run_commands(
        " && ".join(
            (
                "cd /opt",
                f"wget https://github.com/openucx/ucx/releases/download/v{APP_INFO.ucx_tag}/ucx-{APP_INFO.ucx_tag}.tar.gz",
                f"tar -xzf ucx-{APP_INFO.ucx_tag}.tar.gz",
                f"rm ucx-{APP_INFO.ucx_tag}.tar.gz",
                f"cd ucx-{APP_INFO.ucx_tag}/",
                "./contrib/configure-release --with-cuda=/usr/local/cuda prefix=/usr/local",
                "make -j install",
            ),
        ),
    )
    # Build OpenMPI
    .run_commands(
        " && ".join(
            (
                "cd /opt",
                f"wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-{APP_INFO.openmpi_tag}.tar.bz2",
                f"tar -xf openmpi-{APP_INFO.openmpi_tag}.tar.bz2",
                f"rm openmpi-{APP_INFO.openmpi_tag}.tar.bz2",
                f"cd openmpi-{APP_INFO.openmpi_tag}/",
                "./configure --with-cuda=/usr/local/cuda --with-ucx=/usr/local/ prefix=/usr/local",
                "make -j install",
            ),
        ),
    )
    # Build FFTW
    .run_commands(
        " && ".join(
            (
                "cd /opt",
                f"wget http://www.fftw.org/fftw-{APP_INFO.fftw_tag}.tar.gz",
                f"tar -xzf fftw-{APP_INFO.fftw_tag}.tar.gz",
                f"rm fftw-{APP_INFO.fftw_tag}.tar.gz",
                f"cd fftw-{APP_INFO.fftw_tag}/",
                "./configure --disable-fortran --disable-shared --enable-static "
                "--with-pic --enable-avx512 --enable-avx2 --enable-avx --enable-sse2 "
                "--enable-float --prefix=/usr/local",
                "make -j install",
            ),
        )
    )
    # Build GROMACS
    .env({
        "PATH": "/usr/local/gromacs/bin:/root/micromamba/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib:${LD_LIBRARY_PATH}",
    })
    .run_commands(
        " && ".join(
            (
                # gmx binaries
                "cd /opt",
                f"wget https://ftp.gromacs.org/gromacs/gromacs-{CONF.version}.tar.gz",
                f"tar -xzf gromacs-{CONF.version}.tar.gz",
                f"rm gromacs-{CONF.version}.tar.gz",
                f"cd gromacs-{CONF.version}/",
                "mkdir build",
                "cd build",
                "cmake .. "
                "-DCMAKE_BUILD_TYPE=Release "
                "-DCMAKE_PREFIX_PATH='/usr/local' "
                "-DGMX_GPU=CUDA "
                "-DGMX_BUILD_OWN_FFTW=OFF -DGMX_FFT_LIBRARY=fftw3 "
                "-DGMX_SIMD=AVX2_256",  # AVX_512
                "make -j install",
                # Build GROMACS with OpenMPI
                f"cd /opt/gromacs-{CONF.version}/",
                "mkdir build_mpi",
                "cd build_mpi",
                "cmake .. "
                "-DCMAKE_BUILD_TYPE=Release "
                "-DCMAKE_PREFIX_PATH='/usr/local' "
                "-DGMX_GPU=CUDA "
                "-DGMX_MPI=ON "
                "-DCMAKE_C_COMPILER=mpicc "
                "-DCMAKE_CXX_COMPILER=mpicxx "
                "-DGMX_BUILD_OWN_FFTW=OFF -DGMX_FFT_LIBRARY=fftw3 "
                "-DGMX_SIMD=AVX2_256",
                "make -j install",
            ),
        ),
    )
    .run_commands(
        "echo 'micromamba activate base' >> /etc/profile",
        "echo 'source /usr/local/gromacs/bin/GMXRC' >> /etc/profile",
    )
    .add_local_dir(Path(__file__).parent / "gromacs", APP_INFO.gmx_scripts, copy=True)
    .pipe(patch_image_for_helper)
)

biotite_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential")
    .uv_pip_install("biotite", "numpy", "scipy", "seaborn", "matplotlib")
    .pipe(patch_image_for_helper)
)

coordinator_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
api_result_publications = modal.Dict.from_name(
    f"{CONF.name}-api-result-publications",
    create_if_missing=True,
)


##########################################
# Helper functions
##########################################
def file1_needs_update(file1: Path, file2: Path) -> bool:
    """Return True if file1 doesn't exist or is older than file2."""
    if not file1.exists():
        return True
    if not file2.exists():
        raise FileNotFoundError(f"File not found for timestamp comparison: {file2}")
    return file1.stat().st_mtime < file2.stat().st_mtime


def _validated_run_name(run_name: str) -> str:
    safe_run_name = sanitize_filename(run_name)
    if safe_run_name != run_name or len(run_name) > 128:
        raise ValueError("run_name must be a safe filename of at most 128 characters")
    return safe_run_name


def _validate_pdb_content(pdb_content: bytes) -> None:
    validate_pdb_content(pdb_content, max_bytes=10 * 1024 * 1024)


def _json_document(value: object) -> bytes:
    """Serialize one stable, human-readable JSON document."""
    return (
        orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"
    )


def _api_parameters(
    *,
    run_name: str,
    simulation_time_ns: int,
    run_pdbfixer: bool,
    cpu_only: bool,
    num_threads: int,
    use_openmp_threads: bool,
    ld_seed: int,
    gen_seed: int,
    genion_seed: int,
) -> dict[str, str | int | bool]:
    """Return the normalized parameters recorded in an API result archive."""
    return {
        "run_name": run_name,
        "simulation_time_ns": simulation_time_ns,
        "run_pdbfixer": run_pdbfixer,
        "cpu_only": cpu_only,
        "num_threads": num_threads,
        "use_openmp_threads": use_openmp_threads,
        "ld_seed": ld_seed,
        "gen_seed": gen_seed,
        "genion_seed": genion_seed,
    }


def _api_request_sha256(pdb_content: bytes, parameters_json: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(pdb_content).to_bytes(8, byteorder="big"))
    digest.update(pdb_content)
    digest.update(parameters_json)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_ARCHIVE_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_member(
    archive: zipfile.ZipFile,
    *,
    name: str,
    content: bytes,
    role: str,
) -> dict[str, str | int]:
    archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
    return {
        "path": name,
        "role": role,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _sanitized_run_log(
    source_root: Path,
    *,
    run_name: str,
    header: str,
) -> bytes:
    """Append a bounded, path-scrubbed tail of the real GROMACS log."""
    log_path = source_root / f"production_{run_name}.log"
    if not log_path.is_file() or log_path.is_symlink():
        return f"{header}\nGROMACS production log was not available.\n".encode()
    descriptor = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError("GROMACS production log is not a regular file")
        os.lseek(
            descriptor,
            max(file_stat.st_size - _RUN_LOG_TAIL_BYTES, 0),
            os.SEEK_SET,
        )
        content = os.read(descriptor, _RUN_LOG_TAIL_BYTES)
    finally:
        os.close(descriptor)
    text = content.decode("utf-8", errors="replace")
    text = text.replace(str(source_root), "<run-directory>")
    text = text.replace(CONF.output_volume_mountpoint, "<volume>")
    text = re.sub(r"[^\x09\x0a\x20-\x7e]", "?", text.replace("\r", ""))
    return (
        f"{header}\n--- GROMACS production log (last {_RUN_LOG_TAIL_BYTES} bytes) ---\n"
        f"{text.rstrip()}\n"
    ).encode("ascii")


def _file_record(
    *,
    source: Path,
    name: str,
    role: str,
) -> dict[str, str | int]:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Missing required GROMACS {role}: {source.name}")

    return {
        "path": name,
        "role": role,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _required_api_output_files(
    source_root: Path,
    run_name: str,
) -> list[tuple[Path, str, str]]:
    """Return the complete allowlist of scientific files for the API ZIP."""
    files = [
        (
            source_root / "production.mdp",
            "outputs/production.mdp",
            "production_parameters",
        )
    ]
    files.extend(
        (
            source_root / artifact.path,
            f"outputs/{artifact.path}",
            artifact.role or "output",
        )
        for artifact in production_workflow_files(run_name)
    )
    return files


def _expected_api_manifest_files(run_name: str) -> list[tuple[str, str]]:
    """Return the ordered member names and roles recorded by the manifest."""
    return [
        ("input.pdb", "input_structure"),
        ("parameters.json", "normalized_parameters"),
        ("provenance.json", "provenance"),
        ("run.log", "run_log"),
        ("outputs/production.mdp", "production_parameters"),
        *(
            (f"outputs/{artifact.path}", artifact.role or "output")
            for artifact in production_workflow_files(run_name)
        ),
    ]


def _expected_api_archive_members(run_name: str) -> list[str]:
    return [
        *(path for path, _role in _expected_api_manifest_files(run_name)),
        "manifest.json",
        "checksums.sha256",
    ]


def _zip_member_facts(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[int, str]:
    """Read one member fully, checking its CRC while computing size and SHA-256."""
    digest = hashlib.sha256()
    size_bytes = 0
    with archive.open(name) as member:
        while chunk := member.read(_ARCHIVE_CHUNK_SIZE):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _validate_api_archive(
    archive_path: Path,
    *,
    run_name: str,
    pdb_content: bytes,
    parameters_json: bytes,
) -> None:
    """Validate archive members, manifests, checksums, and request identity."""
    expected_members = _expected_api_archive_members(run_name)
    expected_manifest_files = _expected_api_manifest_files(run_name)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            if archive.namelist() != expected_members:
                raise RuntimeError("GROMACS result archive has unexpected members")

            try:
                manifest = orjson.loads(archive.read("manifest.json"))
            except orjson.JSONDecodeError as exc:
                raise RuntimeError("GROMACS result manifest is invalid") from exc
            if (
                not isinstance(manifest, dict)
                or set(manifest) != {"archive_schema_version", "run_name", "files"}
                or manifest.get("archive_schema_version") != _API_ARCHIVE_SCHEMA_VERSION
                or manifest.get("run_name") != run_name
                or not isinstance(manifest.get("files"), list)
                or len(manifest["files"]) != len(expected_manifest_files)
            ):
                raise RuntimeError("GROMACS result manifest is invalid")

            member_facts: dict[str, tuple[int, str]] = {}
            for record, (expected_path, expected_role) in zip(
                manifest["files"], expected_manifest_files, strict=True
            ):
                if (
                    not isinstance(record, dict)
                    or set(record) != {"path", "role", "size_bytes", "sha256"}
                    or record.get("path") != expected_path
                    or record.get("role") != expected_role
                    or type(record.get("size_bytes")) is not int
                    or record["size_bytes"] < 0
                    or not isinstance(record.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                ):
                    raise RuntimeError("GROMACS result manifest is invalid")
                facts = _zip_member_facts(archive, expected_path)
                if facts != (record["size_bytes"], record["sha256"]):
                    raise RuntimeError(
                        f"GROMACS result manifest does not match {expected_path}"
                    )
                member_facts[expected_path] = facts

            manifest_facts = _zip_member_facts(archive, "manifest.json")
            checksum_members = [
                *(path for path, _role in expected_manifest_files),
                "manifest.json",
            ]
            expected_checksums = "".join(
                f"{(manifest_facts if path == 'manifest.json' else member_facts[path])[1]}  {path}\n"
                for path in checksum_members
            ).encode("ascii")
            if archive.read("checksums.sha256") != expected_checksums:
                raise RuntimeError("GROMACS result checksums are inconsistent")

            if member_facts["input.pdb"] != (
                len(pdb_content),
                hashlib.sha256(pdb_content).hexdigest(),
            ):
                raise RuntimeError("Existing GROMACS archive belongs to another input")
            if member_facts["parameters.json"] != (
                len(parameters_json),
                hashlib.sha256(parameters_json).hexdigest(),
            ):
                raise RuntimeError("Existing GROMACS archive uses different parameters")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("GROMACS result archive is invalid") from exc


def _api_result_paths(run_name: str) -> tuple[Path, Path]:
    mount_root = Path(CONF.output_volume_mountpoint).resolve()
    result_dir = (mount_root / _API_RESULTS_DIR / run_name).resolve()
    try:
        result_dir.relative_to(mount_root)
    except ValueError as exc:
        raise ValueError("API result directory escapes the output volume") from exc
    return (
        result_dir / _API_ARCHIVE_FILENAME,
        result_dir / _API_RESULT_MARKER_FILENAME,
    )


def _select_api_archive_candidate(
    *,
    run_name: str,
    request_sha256: str,
    candidate_path: Path,
    archive_sha256: str,
    size_bytes: int,
) -> dict[str, str | int]:
    """Atomically elect one durable candidate for a stable API run name."""
    mount_root = Path(CONF.output_volume_mountpoint).resolve()
    candidate_relative = candidate_path.resolve().relative_to(mount_root).as_posix()
    record: dict[str, str | int] = {
        "publication_schema_version": _API_PUBLICATION_SCHEMA_VERSION,
        "run_name": run_name,
        "request_sha256": request_sha256,
        "candidate_path": candidate_relative,
        "archive_sha256": archive_sha256,
        "size_bytes": size_bytes,
    }
    created = api_result_publications.put(
        run_name,
        record,
        skip_if_exists=True,
    )
    selected = record if created else api_result_publications.get(run_name)
    selected_sha256 = (
        selected.get("archive_sha256") if isinstance(selected, dict) else None
    )
    selected_size = selected.get("size_bytes") if isinstance(selected, dict) else None
    if (
        not isinstance(selected, dict)
        or set(selected) != set(record)
        or selected.get("publication_schema_version") != _API_PUBLICATION_SCHEMA_VERSION
        or selected.get("run_name") != run_name
        or selected.get("request_sha256") != request_sha256
        or not isinstance(selected.get("candidate_path"), str)
        or not isinstance(selected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", selected_sha256) is None
        or type(selected_size) is not int
        or selected_size < 1
    ):
        raise RuntimeError("GROMACS result publication record is invalid")
    return cast("dict[str, str | int]", selected)


def _selected_candidate_path(
    publication: dict[str, str | int],
    *,
    run_name: str,
) -> Path:
    mount_root = Path(CONF.output_volume_mountpoint).resolve()
    run_root = (mount_root / run_name).resolve()
    candidate = (mount_root / str(publication["candidate_path"])).resolve()
    try:
        candidate.relative_to(run_root)
    except ValueError as exc:
        raise RuntimeError("GROMACS publication candidate escapes its run") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("GROMACS publication candidate is missing")
    return candidate


def _copy_verified_candidate(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Copy through no-follow descriptors and verify the copied bytes."""
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    destination_descriptor: int | None = None
    try:
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise RuntimeError("GROMACS publication candidate is not a file")
        destination_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := os.read(source_descriptor, _ARCHIVE_CHUNK_SIZE):
            size_bytes += len(chunk)
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                remaining = remaining[os.write(destination_descriptor, remaining) :]
        os.close(destination_descriptor)
        destination_descriptor = None
        if size_bytes != expected_size or digest.hexdigest() != expected_sha256:
            raise RuntimeError("GROMACS publication candidate changed while copying")
        os.replace(temporary, destination)
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        temporary.unlink(missing_ok=True)


def _publish_selected_api_archive(
    *,
    publication: dict[str, str | int],
    run_name: str,
    pdb_content: bytes,
    parameters_json: bytes,
) -> tuple[Path, str, int]:
    """Publish the elected bytes; concurrent publishers copy identical data."""
    archive_path, _marker_path = _api_result_paths(run_name)
    expected_sha256 = str(publication["archive_sha256"])
    expected_size = int(publication["size_bytes"])
    CONF.output_volume.reload()
    if archive_path.exists():
        _validate_api_archive(
            archive_path,
            run_name=run_name,
            pdb_content=pdb_content,
            parameters_json=parameters_json,
        )
        if (
            archive_path.stat().st_size != expected_size
            or _sha256_file(archive_path) != expected_sha256
        ):
            raise RuntimeError("Published GROMACS result differs from elected bytes")
        return archive_path, expected_sha256, expected_size

    candidate = _selected_candidate_path(publication, run_name=run_name)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    _copy_verified_candidate(
        candidate,
        archive_path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    _validate_api_archive(
        archive_path,
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
    )
    CONF.output_volume.commit()
    return archive_path, expected_sha256, expected_size


def _publish_api_result_marker(
    marker_path: Path,
    *,
    request_sha256: str,
    archive_sha256: str,
    size_bytes: int,
    completed_at: str,
) -> None:
    marker = _json_document({
        "archive_schema_version": _API_ARCHIVE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "archive_sha256": archive_sha256,
        "size_bytes": size_bytes,
        "completed_at": completed_at,
    })
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f".{marker_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(marker)
        os.replace(temporary, marker_path)
        CONF.output_volume.commit()
    finally:
        temporary.unlink(missing_ok=True)


def _archive_app_result(
    *,
    run_name: str,
    archive_path: Path,
    archive_sha256: str,
    size_bytes: int,
) -> AppRunResult:
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="gromacs_run",
                kind=ArtifactKind.ARCHIVE,
                remote_path=str(archive_path),
                mount_root=CONF.output_volume_mountpoint,
                volume_name=CONF.output_volume_name,
                media_type="application/zip",
                metadata={
                    "run_name": run_name,
                    "filename": f"{run_name}.zip",
                    "size_bytes": size_bytes,
                    "sha256": archive_sha256,
                    "archive_format": "zip",
                },
            )
        ],
    )


def _existing_api_archive(
    *,
    run_name: str,
    pdb_content: bytes,
    parameters_json: bytes,
    request_sha256: str,
) -> AppRunResult | None:
    archive_path, marker_path = _api_result_paths(run_name)
    if not archive_path.exists() and not marker_path.exists():
        return None
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RuntimeError("Published GROMACS result archive is missing")

    _validate_api_archive(
        archive_path,
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
    )
    archive_sha256 = _sha256_file(archive_path)
    size_bytes = archive_path.stat().st_size

    if marker_path.exists():
        try:
            marker = orjson.loads(marker_path.read_bytes())
        except orjson.JSONDecodeError as exc:
            raise RuntimeError("GROMACS result completion marker is invalid") from exc
        if not isinstance(marker, dict) or any((
            marker.get("archive_schema_version") != _API_ARCHIVE_SCHEMA_VERSION,
            marker.get("request_sha256") != request_sha256,
            marker.get("archive_sha256") != archive_sha256,
            marker.get("size_bytes") != size_bytes,
        )):
            raise RuntimeError(
                "GROMACS result completion marker does not match archive"
            )
    else:
        _publish_api_result_marker(
            marker_path,
            request_sha256=request_sha256,
            archive_sha256=archive_sha256,
            size_bytes=size_bytes,
            completed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    return _archive_app_result(
        run_name=run_name,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        size_bytes=size_bytes,
    )


def _package_gromacs_api_archive(
    *,
    source_root: Path,
    run_name: str,
    pdb_content: bytes,
    parameters_json: bytes,
    request_sha256: str,
    started_at: datetime,
    completed_at: datetime,
) -> AppRunResult:
    """Publish the API-only final ZIP and its completion marker."""
    required_files = _required_api_output_files(source_root, run_name)
    production_log_name = f"production_{run_name}.log"
    warmup_pattern = (
        "^(?:"
        + "|".join([
            *(re.escape(source.name) for source, _name, _role in required_files),
            re.escape(production_log_name),
        ])
        + ")$"
    )
    warmup_directory(source_root, file_pattern=warmup_pattern)

    started_at_text = started_at.isoformat().replace("+00:00", "Z")
    completed_at_text = completed_at.isoformat().replace("+00:00", "Z")
    provenance_json = _json_document({
        "archive_schema_version": _API_ARCHIVE_SCHEMA_VERSION,
        "app": CONF.name,
        "gromacs_version": CONF.version,
        "repository": CONF.repo_url,
        "started_at": started_at_text,
        "completed_at": completed_at_text,
    })
    run_log_header = (
        "Biomodals GROMACS job\n"
        f"run_name: {run_name}\n"
        "status: succeeded\n"
        f"started_at: {started_at_text}\n"
        f"completed_at: {completed_at_text}\n"
        f"gromacs_version: {CONF.version}\n"
    )
    run_log = _sanitized_run_log(
        source_root,
        run_name=run_name,
        header=run_log_header,
    )

    _archive_path, marker_path = _api_result_paths(run_name)
    candidate = source_root / f".api-result-candidate-{uuid4().hex}.zip"
    records: list[dict[str, str | int]] = []
    with zipfile.ZipFile(
        candidate,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        records.append(
            _write_bytes_member(
                archive,
                name="input.pdb",
                content=pdb_content,
                role="input_structure",
            )
        )
        records.append(
            _write_bytes_member(
                archive,
                name="parameters.json",
                content=parameters_json,
                role="normalized_parameters",
            )
        )
        records.append(
            _write_bytes_member(
                archive,
                name="provenance.json",
                content=provenance_json,
                role="provenance",
            )
        )
        records.append(
            _write_bytes_member(
                archive,
                name="run.log",
                content=run_log,
                role="run_log",
            )
        )
        for source, name, role in required_files:
            records.append(_file_record(source=source, name=name, role=role))
            archive.write(
                source,
                arcname=name,
                compress_type=(
                    zipfile.ZIP_STORED
                    if source.suffix.lower() in {".tpr", ".xtc"}
                    else zipfile.ZIP_DEFLATED
                ),
            )

        manifest_json = _json_document({
            "archive_schema_version": _API_ARCHIVE_SCHEMA_VERSION,
            "run_name": run_name,
            "files": records,
        })
        manifest_record = _write_bytes_member(
            archive,
            name="manifest.json",
            content=manifest_json,
            role="manifest",
        )
        checksum_records = [*records, manifest_record]
        checksums = "".join(
            f"{record['sha256']}  {record['path']}\n" for record in checksum_records
        ).encode("ascii")
        _write_bytes_member(
            archive,
            name="checksums.sha256",
            content=checksums,
            role="checksums",
        )

    _validate_api_archive(
        candidate,
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
    )
    candidate_sha256 = _sha256_file(candidate)
    candidate_size = candidate.stat().st_size
    CONF.output_volume.commit()
    publication = _select_api_archive_candidate(
        run_name=run_name,
        request_sha256=request_sha256,
        candidate_path=candidate,
        archive_sha256=candidate_sha256,
        size_bytes=candidate_size,
    )
    archive_path, archive_sha256, size_bytes = _publish_selected_api_archive(
        publication=publication,
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
    )
    _publish_api_result_marker(
        marker_path,
        request_sha256=request_sha256,
        archive_sha256=archive_sha256,
        size_bytes=size_bytes,
        completed_at=completed_at_text,
    )

    return _archive_app_result(
        run_name=run_name,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        size_bytes=size_bytes,
    )


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def prepare_tpr_gpu(
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
) -> str:
    """Prepare inputs for production Gromacs run.

    Steps: clean input PDB, build topology with Amber FF19SB and TIP3P water,
    solvate, add ions, minimize (em and cg), equilibrate (NVT and NPT), and
    generate production TPR file.
    """
    run_name = _validated_run_name(run_name)
    _validate_pdb_content(pdb_content)
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    work_path = layout.run_root
    work_path.mkdir(parents=True, exist_ok=True)

    # Skip prep if production tpr already exists
    if all(
        f.exists()
        for f in (
            work_path / f"production_{run_name}.tpr",
            work_path / "production.mdp",
        )
    ):
        print("✅ Preparation already completed, skipping.")
        return str(work_path)

    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    staged_input_pdb_path = layout.inputs_dir / f"{run_name}.pdb"
    input_pdb_path = work_path / f"{run_name}.pdb"
    staged_input_pdb_path.write_bytes(pdb_content)
    input_pdb_path.write_bytes(pdb_content)
    CONF.output_volume.commit()

    script_path = Path(APP_INFO.gmx_scripts) / "prepare-tpr.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Gromacs script not found: {script_path}")

    cmd = [
        str(script_path),
        "-i",
        str(input_pdb_path),
        "-t",
        str(simulation_time_ns),
        "-j",
        str(num_threads),
        "--ld-seed",
        str(ld_seed),
        "--gen-seed",
        str(gen_seed),
        "--genion-seed",
        str(genion_seed),
    ]
    if run_pdbfixer:
        cmd.append("--fix-pdb")

    if use_openmp_threads:
        cmd.append("--use-openmp-threads")
    # Modal adds this automatically but we want Gromacs to handle threading
    _ = run_command(cmd, cwd=str(work_path), env={"OMP_NUM_THREADS": None})
    CONF.output_volume.commit()
    return str(work_path)


@app.function(
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def prepare_tpr_cpu(
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
) -> str:
    """Prepare inputs for production Gromacs run.

    Steps: clean input PDB, build topology with Amber FF19SB and TIP3P water,
    solvate, add ions, minimize (em and cg), equilibrate (NVT and NPT), and
    generate production TPR file.
    """
    run_name = _validated_run_name(run_name)
    _validate_pdb_content(pdb_content)
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    work_path = layout.run_root
    work_path.mkdir(parents=True, exist_ok=True)

    # Skip prep if production tpr already exists
    if all(
        f.exists()
        for f in (
            work_path / f"production_{run_name}.tpr",
            work_path / "production.mdp",
        )
    ):
        print("✅ Preparation already completed, skipping.")
        return str(work_path)

    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    staged_input_pdb_path = layout.inputs_dir / f"{run_name}.pdb"
    input_pdb_path = work_path / f"{run_name}.pdb"
    staged_input_pdb_path.write_bytes(pdb_content)
    input_pdb_path.write_bytes(pdb_content)
    CONF.output_volume.commit()

    script_path = Path(APP_INFO.gmx_scripts) / "prepare-tpr.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Gromacs script not found: {script_path}")

    cmd = [
        str(script_path),
        "-i",
        str(input_pdb_path),
        "-t",
        str(simulation_time_ns),
        "--cpu-only",
        "-j",
        str(num_threads),
        "--ld-seed",
        str(ld_seed),
        "--gen-seed",
        str(gen_seed),
        "--genion-seed",
        str(genion_seed),
    ]
    if run_pdbfixer:
        cmd.append("--fix-pdb")
    if use_openmp_threads:
        cmd.append("--use-openmp-threads")
    # Modal adds this automatically but we want Gromacs to handle threading
    _ = run_command(cmd, cwd=str(work_path), env={"OMP_NUM_THREADS": None})

    CONF.output_volume.commit()
    return str(work_path)


@app.function(
    image=runtime_image,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def find_traj_last_time_ns(traj_file: str) -> float:
    """Calculate the last-readable simulation time (ns) in a trajectory.

    In our setup, dt=2fs=0.002ps; `gmx check` normally reports the simulation
    time in ps, so we can convert it to #steps by dividing by `dt=0.002`.

    Because we setup the simulation by inputting the expected nanoseconds,
    #steps = ns * 500000.

    When the simulation was interrupted, `gmx check` may only report the #frames
    and timestep size, so we need to manually calculate the closest last step
    that is within the trajectory bounds.
    """
    import shutil

    CONF.output_volume.reload()
    traj_path = Path(traj_file)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    gmx = shutil.which("gmx") or shutil.which("gmx_mpi")
    if gmx is None:
        raise RuntimeError("Gromacs executable not found")

    cmd = [gmx, "check", "-f", str(traj_path)]
    result = run_command(cmd, cwd=traj_path.parent, output_mode="capture")

    for line in result:
        # Last frame      20000 time 200000.000
        if line.startswith("Last frame"):
            last_time_ps = float(line.strip().split(" ")[-1])
            return last_time_ps * 0.001

    # Be robust in case the run was interrupted
    # Item        #frames Timestep (ps)
    # Step         20001    10
    header_line_idx = -1
    header_cols = ["Item", "#frames", "Timestep", "(ps)"]
    for i, line in enumerate(result):
        if line.startswith("Item") and line.strip().split() == header_cols:
            header_line_idx = i
            break
    if header_line_idx != -1:
        readable_line = result[header_line_idx + 1].strip()
        _, frames, timestep_ps = readable_line.split()
        return float((int(frames) - 1) * float(timestep_ps)) * 0.001

    raise ValueError("Last frame time not found in trajectory")


@app.function(
    gpu=CONF.gpu,
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def production_run_gpu(
    run_name: str,
    simulation_time_ns: int,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
) -> str:
    """Production Gromacs run."""
    import shutil

    run_name = _validated_run_name(run_name)
    CONF.output_volume.reload()
    work_path = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_name
    ).run_root
    deffnm = f"production_{run_name}"
    tpr_file_path = work_path / f"{deffnm}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # Pick up exisiting trajectory and continue simulation when checkpoint exists
    traj_file_path = work_path / f"{deffnm}.xtc"
    checkpoint_file_path = work_path / f"{deffnm}.cpt"
    nsteps = -2  # default: use nsteps from the prepared TPR
    if traj_file_path.exists() and checkpoint_file_path.exists():
        simulated_ns = find_traj_last_time_ns.remote(str(traj_file_path))
        nsteps = int((simulation_time_ns - simulated_ns) * 500000)  # 2 fs timestep
        if nsteps <= 0:
            print("✅ Production run already completed, skipping.")
            return str(work_path)

    gmx = shutil.which("gmx_mpi") if use_openmp_threads else shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError("Gromacs binary not found in PATH.")

    cmd = [
        gmx,
        "mdrun",
        "-deffnm",
        deffnm,
        "-cpi",
        checkpoint_file_path.name,
        "-nsteps",
        str(nsteps),
        "-gpu_id",
        "0",
        "-nb",
        "gpu",
        "-pmefft",
        "gpu",
        "-pme",
        "gpu",
        "-bonded",
        "gpu",
        "-update",
        "gpu",
    ]
    if use_openmp_threads:
        cmd.extend(["-ntmpi", "1", "-ntomp", str(num_threads)])
    else:
        cmd.extend(["-nt", str(num_threads)])

    # Modal adds this automatically but we want Gromacs to handle threading
    _ = run_command(cmd, cwd=str(work_path), env={"OMP_NUM_THREADS": None})
    CONF.output_volume.commit()
    return str(work_path)


@app.function(
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def production_run_cpu(
    run_name: str,
    simulation_time_ns: int,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
) -> str:
    """Production Gromacs run."""
    import shutil

    run_name = _validated_run_name(run_name)
    CONF.output_volume.reload()
    work_path = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_name
    ).run_root
    deffnm = f"production_{run_name}"
    tpr_file_path = work_path / f"{deffnm}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # Pick up exisiting trajectory and continue simulation when checkpoint exists
    traj_file_path = work_path / f"{deffnm}.xtc"
    checkpoint_file_path = work_path / f"{deffnm}.cpt"
    nsteps = -2  # default: use nsteps from the prepared TPR
    if traj_file_path.exists() and checkpoint_file_path.exists():
        simulated_ns = find_traj_last_time_ns.remote(str(traj_file_path))
        nsteps = int((simulation_time_ns - simulated_ns) * 500000)  # 2 fs timestep
        if nsteps <= 0:
            print("✅ Production run already completed, skipping.")
            return str(work_path)

        print(f"Continuing production run for additional {nsteps} steps...")

    gmx = shutil.which("gmx_mpi") if use_openmp_threads else shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError("Gromacs binary not found in PATH.")

    cmd = [
        gmx,
        "mdrun",
        "-deffnm",
        deffnm,
        "-cpi",
        checkpoint_file_path.name,
        "-nsteps",
        str(nsteps),
        "-nb",
        "cpu",
        "-pmefft",
        "cpu",
        "-pme",
        "cpu",
        "-bonded",
        "cpu",
        "-update",
        "cpu",
    ]
    if use_openmp_threads:
        cmd.extend(["-ntmpi", "1", "-ntomp", str(num_threads)])
    else:
        cmd.extend(["-nt", str(num_threads)])

    # Modal adds this automatically but we want Gromacs to handle threading
    _ = run_command(cmd, cwd=str(work_path), env={"OMP_NUM_THREADS": None})
    CONF.output_volume.commit()
    return str(work_path)


@app.function(
    image=runtime_image,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def postprocess_traj(
    traj_file: str,
    tpr_file: str,
    processed_traj_file: str,
    ref_struct_file: str | None = None,
) -> None:
    """Process Gromacs trajectory.

    Remove PBC for the protein chains (best-effort), and dump centered structures.
    """
    CONF.output_volume.reload()
    script_path = Path(APP_INFO.gmx_scripts) / "postprocess-traj.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Gromacs script not found: {script_path}")

    cmd = [
        str(script_path),
        "--tpr-file",
        tpr_file,
        "--xtc-file",
        traj_file,
        "--output-file",
        processed_traj_file,
    ]
    if ref_struct_file is not None:
        cmd.extend(["--ref-structure", ref_struct_file])
    _ = run_command(
        cmd,
        cwd=str(Path(processed_traj_file).parent),
        env={"OMP_NUM_THREADS": None},
        output_mode="capture",
    )
    CONF.output_volume.commit()


@app.function(
    image=biotite_image,
    cpu=1,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def collect_traj_stats(
    traj_prefix: str,
    run_name: str,
    save_processed_traj: bool = False,
    make_figures: bool = True,
) -> str:
    """Process Gromacs trajectory and generate analysis plots.

    Ref: https://www.biotite-python.org/latest/examples/gallery/structure/modeling/md_analysis.html
    """
    import biotite  # type: ignore[ty:unresolved-import]
    import biotite.structure as struc  # type: ignore[ty:unresolved-import]
    import biotite.structure.io as strucio  # type: ignore[ty:unresolved-import]
    import biotite.structure.io.xtc as xtc  # type: ignore[ty:unresolved-import]
    import matplotlib.pyplot as plt  # type: ignore[ty:unresolved-import]
    import numpy as np

    run_name = _validated_run_name(run_name)
    out_vol = CONF.output_volume
    out_vol.reload()
    work_path = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_name
    ).run_root
    traj_path = work_path / f"{traj_prefix}{run_name}.xtc"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Remove PBC and align to reference structure
    processed_traj_path = work_path / f"{traj_prefix}{run_name}_nopbc.xtc"
    if file1_needs_update(processed_traj_path, traj_path):
        # remove outdated processed trajectory
        processed_traj_path.unlink(missing_ok=True)
        out_vol.commit()
    if not processed_traj_path.exists():
        postprocess_traj.remote(
            str(traj_path),
            str(work_path / f"{traj_prefix}{run_name}.tpr"),
            str(processed_traj_path),
            ref_struct_file=str(work_path / f"{run_name}.pdb"),
        )

    out_vol.reload()
    traj_1st_frame_pdb_path = work_path / f"{traj_prefix}{run_name}_nopbc_centered.pdb"
    if not traj_1st_frame_pdb_path.exists():
        raise RuntimeError(
            f"Postprocessing trajectory did not generate expected PDB: {traj_1st_frame_pdb_path}"
        )

    # Gromacs does not set the element symbol in its PDB files,
    # but Biotite guesses the element names from the atom names,
    # emitting a warning
    template = strucio.load_structure(traj_1st_frame_pdb_path)
    # The structure still has water and ions, that are not needed for our
    # calculations, we are only interested in the protein itself
    # These are removed for the sake of computational speed using a boolean
    # mask
    protein_mask = struc.filter_amino_acids(template)
    template = template[protein_mask]

    # We could have loaded the trajectory also with
    # 'strucio.load_structure()', but in this case we only want to load
    # those coordinates that belong to the already selected atoms of the
    # template structure.
    # Hence, we use the 'XTCFile' class directly to load the trajectory
    # This gives us the additional option that allows us to select the
    # coordinates belonging to the amino acids.
    xtc_file = xtc.XTCFile.read(processed_traj_path, atom_i=np.where(protein_mask)[0])
    trajectory = xtc_file.get_structure(template)
    if not save_processed_traj:
        processed_traj_path.unlink()
        out_vol.commit()

    # Get simulation time (ns) for plotting purposes
    time = xtc_file.get_time() / 1000.0
    print(f"Simulated {time[-1]:.1f} ns in {traj_path}")

    # Remove PBC (gmx trjconv)
    # trajectory = struc.remove_pbc(trajectory)
    trajectory, _ = struc.superimpose(trajectory[0], trajectory)

    # Dump the last frame of the processed trajectory as PDB
    last_frame_path = work_path / f"{traj_prefix}{run_name}_last_frame.pdb"
    if file1_needs_update(last_frame_path, traj_path):
        last_frame_path.unlink(missing_ok=True)  # remove outdated last frame
    if not last_frame_path.exists():
        strucio.save_structure(last_frame_path, trajectory[-1])
        out_vol.commit()

    # RMSD vs. the initial frame
    rmsd_fig_path = work_path / f"rmsd_{traj_prefix}{run_name}.png"
    rmsd_csv_path = rmsd_fig_path.with_suffix(".csv")
    if file1_needs_update(rmsd_csv_path, traj_path):
        rmsd_csv_path.unlink(missing_ok=True)
        rmsd_fig_path.unlink(missing_ok=True)
    if not rmsd_csv_path.exists():
        rmsd = struc.rmsd(trajectory[0], trajectory)
        np.savetxt(
            rmsd_csv_path,
            np.column_stack((time, rmsd)),
            fmt="%.5f",
            delimiter=",",
            header="time_ns,rmsd",
            comments="",
        )
        out_vol.commit()

        if not rmsd_fig_path.exists() and make_figures:
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            ax.plot(time, rmsd, color=biotite.colors["dimorange"])
            ax.set_xlim(time[0], time[-1])
            ax.set_title(run_name)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("RMSD (Å)")
            figure.savefig(rmsd_fig_path)
            plt.close(figure)

            out_vol.commit()

    # Radius of gyration
    rg_fig_path = work_path / f"rg_{traj_prefix}{run_name}.png"
    rg_csv_path = rg_fig_path.with_suffix(".csv")
    if file1_needs_update(rg_csv_path, traj_path):
        rg_csv_path.unlink(missing_ok=True)
        rg_fig_path.unlink(missing_ok=True)
    if not rg_csv_path.exists():
        rg = struc.gyration_radius(trajectory)
        np.savetxt(
            rg_csv_path,
            np.column_stack((time, rg)),
            fmt="%.5f",
            delimiter=",",
            header="time_ns,rg",
            comments="",
        )
        out_vol.commit()
        if not rg_fig_path.exists() and make_figures:
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            ax.plot(time, rg, color=biotite.colors["dimgreen"])
            ax.set_xlim(time[0], time[-1])
            ax.set_title(run_name)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Radius of Gyration (Å)")
            figure.savefig(rg_fig_path)
            plt.close(figure)

            out_vol.commit()

    # RMSF of each residue
    rmsf_fig_path = work_path / f"rmsf_{traj_prefix}{run_name}.png"
    rmsf_csv_path = rmsf_fig_path.with_suffix(".csv")
    if file1_needs_update(rmsf_csv_path, traj_path):
        rmsf_csv_path.unlink(missing_ok=True)
        rmsf_fig_path.unlink(missing_ok=True)
    if not rmsf_csv_path.exists():
        # Sidechain atoms fluctuate too much, so we only consider CA atoms
        ca_trajectory = trajectory[:, trajectory.atom_name == "CA"]
        rmsf = struc.rmsf(struc.average(ca_trajectory), ca_trajectory)
        res_count = struc.get_residue_count(trajectory)
        res_idx = np.arange(1, res_count + 1)
        np.savetxt(
            rmsf_csv_path,
            np.column_stack((res_idx, rmsf)),
            fmt="%.5f",
            delimiter=",",
            header="residue_index,rmsf",
            comments="",
        )
        out_vol.commit()
        if not rmsf_fig_path.exists() and make_figures:
            # Sidechain atoms fluctuate too much, so we only consider CA atoms
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            ax.plot(res_idx, rmsf, color=biotite.colors["dimorange"])
            ax.set_xlim(1, res_count)
            ax.set_title(run_name)
            ax.set_xlabel("Residue Index")
            ax.set_ylabel("RMSF (Å)")
            figure.savefig(rmsf_fig_path)
            plt.close(figure)

            out_vol.commit()

    return str(work_path)


def _run_gromacs_job(
    *,
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    cpu_only: bool = False,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
    max_parallel_analysis: int | None = None,
) -> AppRunResult:
    """Run one complete GROMACS job through the shared app functions."""
    run_name = _validated_run_name(run_name)
    _validate_pdb_content(pdb_content)
    if simulation_time_ns < 1:
        raise ValueError("simulation_time_ns must be at least 1")
    if not 1 <= num_threads <= APP_INFO.gmx_threads:
        raise ValueError(f"num_threads must be between 1 and {APP_INFO.gmx_threads}")

    print("🧬 Preparing Gromacs production run...")
    prepare_tpr_conf = {
        "pdb_content": pdb_content,
        "run_name": run_name,
        "simulation_time_ns": simulation_time_ns,
        "run_pdbfixer": run_pdbfixer,
        "num_threads": num_threads,
        "use_openmp_threads": use_openmp_threads,
        "ld_seed": ld_seed,
        "gen_seed": gen_seed,
        "genion_seed": genion_seed,
    }
    prepare_function = prepare_tpr_cpu if cpu_only else prepare_tpr_gpu
    remote_workdir = prepare_function.remote(**prepare_tpr_conf)

    bounded_map(
        ["nvt_", "npt_"],
        lambda prefix: collect_traj_stats.remote(prefix, run_name=run_name),
        max_parallel=max_parallel_analysis,
    )

    print("🧬 Starting Gromacs production MD simulation...")
    production_function = production_run_cpu if cpu_only else production_run_gpu
    production_function.remote(
        run_name=run_name,
        simulation_time_ns=simulation_time_ns,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
    )

    print("🧬 Postprocessing Gromacs trajectory and generating analysis plots...")
    collect_traj_stats.remote(
        "production_",
        run_name=run_name,
        save_processed_traj=True,
    )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="gromacs_run",
                kind=ArtifactKind.DIRECTORY,
                remote_path=remote_workdir,
                mount_root=CONF.output_volume_mountpoint,
                volume_name=CONF.output_volume_name,
                metadata={"run_name": run_name},
                files=cast(
                    list[ArtifactFile | str],
                    production_workflow_files(run_name),
                ),
            )
        ],
    )


@app.function(
    image=coordinator_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    max_containers=20,
    volumes=CONF.mounts(output_volume=True),
)
def run_gromacs_job(
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    cpu_only: bool = False,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
) -> AppRunResult:
    """Run one detached API job and publish one immutable result ZIP."""
    run_name = _validated_run_name(run_name)
    _validate_pdb_content(pdb_content)
    if simulation_time_ns < 1:
        raise ValueError("simulation_time_ns must be at least 1")
    if not 1 <= num_threads <= APP_INFO.gmx_threads:
        raise ValueError(f"num_threads must be between 1 and {APP_INFO.gmx_threads}")

    parameters = _api_parameters(
        run_name=run_name,
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
    )
    parameters_json = _json_document(parameters)
    request_sha256 = _api_request_sha256(pdb_content, parameters_json)

    CONF.output_volume.reload()
    if existing := _existing_api_archive(
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
        request_sha256=request_sha256,
    ):
        return existing

    started_at = datetime.now(UTC)
    directory_result = _run_gromacs_job(
        pdb_content=pdb_content,
        run_name=run_name,
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
    )
    if directory_result.status != AppRunStatus.SUCCEEDED:
        raise RuntimeError("GROMACS run did not complete successfully")
    directory_storage = directory_result.outputs[0].storage
    if not isinstance(directory_storage, VolumePath):
        raise RuntimeError("GROMACS run did not return a volume-backed directory")

    CONF.output_volume.reload()
    mount_root = Path(CONF.output_volume_mountpoint).resolve()
    source_root = directory_storage.at_mountpoint(mount_root).resolve()
    try:
        source_root.relative_to(mount_root)
    except ValueError as exc:
        raise RuntimeError("GROMACS run directory escapes the output volume") from exc
    if not source_root.is_dir() or source_root.is_symlink():
        raise FileNotFoundError("GROMACS run directory is missing")

    return _package_gromacs_api_archive(
        source_root=source_root,
        run_name=run_name,
        pdb_content=pdb_content,
        parameters_json=parameters_json,
        request_sha256=request_sha256,
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_gromacs_task(
    input_pdb: str,
    run_name: str | None = None,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    cpu_only: bool = False,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
    max_parallel_analysis: int | None = None,
) -> None:
    """Run GROMACS MD simulations on Modal and save results to a volume.

    Args:
        input_pdb: Path to the input PDB file.
        run_name: Name for this simulation run. Defaults to input PDB filename
            stem. Note that if the name exists in the remote volume, files in
            the remote will be preferred over the local one. Make sure to use
            unique names if you want to start a new run!
        simulation_time_ns: Length of the production MD simulation in nanoseconds.
        run_pdbfixer: Whether to run PDBFixer to clean the input PDB file
            before preparation.
        cpu_only: Whether to run GROMACS on CPU only. If False, GROMACS will
            use GPU acceleration.
        num_threads: Number of CPU threads to use for GROMACS.
        use_openmp_threads: Whether to use OpenMP threading in GROMACS.
        ld_seed: Random seed for the Langevin dynamics thermostat during
            equilibration. If -1, a random seed will be chosen.
        gen_seed: Random seed for initial velocity generation during
            equilibration. If -1, a random seed will be chosen.
        genion_seed: Random seed for ion placement during system neutralization.
        max_parallel_analysis: Maximum number of trajectory-analysis containers
            to run at once.
    """
    # Load input PDB
    pdb_path = Path(input_pdb).expanduser().resolve()
    pdb_str = pdb_path.read_bytes()
    if run_name is None:
        run_name = pdb_path.stem

    result = _run_gromacs_job(
        pdb_content=pdb_str,
        run_name=run_name,
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
        max_parallel_analysis=max_parallel_analysis,
    )
    remote_vol = result.outputs[0].storage
    print(f"🧬 Gromacs preparation complete! Check data in {remote_vol}")
