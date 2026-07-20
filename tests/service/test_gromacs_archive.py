"""Service-owned packaging for established GROMACS app outputs."""

# ruff: noqa: D103

from __future__ import annotations

import asyncio
import hashlib
import io
import struct
import zipfile

import pytest

from biomodals.service.gromacs.archive import (
    BuiltGromacsArchive,
    validate_gromacs_archive,
    write_gromacs_archive,
)

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
PARAMETERS = '{"cpu_only":false,"simulation_time_ns":5}'
XTC = struct.pack(">i", 1995) + b"\0" * 28
TPR = b"\0\0\0\x10VERSION 2026.1\0\0\0\0"
PNG = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR" + b"\0" * 16
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


def test_service_packages_established_remote_files_deterministically() -> None:
    prefix = f"production_{RUN_NAME}"
    remote_files = _remote_files()

    def build() -> tuple[bytes, BuiltGromacsArchive]:
        async def read_file(path: str):
            try:
                content = remote_files[path]
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
                job_id="11111111-1111-4111-8111-111111111111",
                stages_json="[]",
                started_at=1,
                completed_at=2,
                read_file=read_file,
            )
        )
        return output.getvalue(), result

    first_bytes, first = build()
    second_bytes, second = build()

    assert first_bytes == second_bytes
    assert first == second
    assert first.size_bytes == len(first_bytes)
    assert first.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert validate_gromacs_archive(io.BytesIO(first_bytes), run_name=RUN_NAME)
    with zipfile.ZipFile(io.BytesIO(first_bytes)) as archive:
        assert archive.read("input.pdb") == PDB
        assert f"outputs/{prefix}.xtc" not in archive.namelist()
        assert archive.read(f"outputs/{prefix}_nopbc.xtc") == XTC
        assert archive.read(f"outputs/rmsd_{prefix}.png") == PNG
        assert archive.read(f"outputs/rg_{prefix}.png") == PNG
        assert archive.read(f"outputs/rmsf_{prefix}.png") == PNG
        assert archive.read("metadata/parameters.json") == PARAMETERS.encode()
        assert {name.split("/", 1)[0] for name in archive.namelist()} == {
            "input.pdb",
            "outputs",
            "metadata",
        }


@pytest.mark.parametrize(
    ("remote_name", "content", "message"),
    [
        ("trajectory", b"", "manifest record"),
        ("trajectory", b"not-an-xtc", "trajectory is invalid"),
        ("rmsd_plot", b"not-a-png", "PNG is invalid"),
        ("rmsd", b"time_ns,rmsd\n", "wrong schema"),
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
                job_id="11111111-1111-4111-8111-111111111111",
                stages_json="[]",
                started_at=1,
                completed_at=2,
                read_file=read_file,
            )
        )
