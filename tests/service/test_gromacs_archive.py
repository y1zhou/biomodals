"""Service-owned packaging for established GROMACS app outputs."""

# ruff: noqa: D103

from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile

from biomodals.service.gromacs.archive import (
    BuiltGromacsArchive,
    validate_gromacs_archive,
    write_gromacs_archive,
)

RUN_NAME = "first-simulation-0123456789abcdef0123456789abcdef"
PDB = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
PARAMETERS = '{"cpu_only":false,"simulation_time_ns":5}'


def test_service_packages_established_remote_files_deterministically() -> None:
    prefix = f"production_{RUN_NAME}"
    remote_files = {
        f"{RUN_NAME}/{RUN_NAME}.pdb": PDB,
        f"{RUN_NAME}/production.mdp": b"integrator = md\n",
        f"{RUN_NAME}/{prefix}.xtc": b"full trajectory",
        f"{RUN_NAME}/{prefix}_nopbc.xtc": b"processed trajectory",
        f"{RUN_NAME}/{prefix}.tpr": b"topology",
        f"{RUN_NAME}/{prefix}_nopbc_centered.pdb": b"MODEL\nEND\n",
        f"{RUN_NAME}/rmsd_{prefix}.csv": b"time,rmsd\n",
        f"{RUN_NAME}/rmsd_{prefix}.png": b"rmsd plot",
        f"{RUN_NAME}/rg_{prefix}.csv": b"time,rg\n",
        f"{RUN_NAME}/rg_{prefix}.png": b"rg plot",
        f"{RUN_NAME}/rmsf_{prefix}.csv": b"residue,rmsf\n",
        f"{RUN_NAME}/rmsf_{prefix}.png": b"rmsf plot",
    }

    def build() -> tuple[bytes, BuiltGromacsArchive]:
        async def read_file(path: str):
            content = remote_files[path]
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
        assert archive.read(f"outputs/{prefix}_nopbc.xtc") == b"processed trajectory"
        assert archive.read(f"outputs/rmsd_{prefix}.png") == b"rmsd plot"
        assert archive.read(f"outputs/rg_{prefix}.png") == b"rg plot"
        assert archive.read(f"outputs/rmsf_{prefix}.png") == b"rmsf plot"
        assert archive.read("parameters.json") == PARAMETERS.encode()
