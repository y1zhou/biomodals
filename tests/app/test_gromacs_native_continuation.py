"""Optional local GROMACS 2026.1 contract test; no Modal or user inputs."""

# Trusted local executable and fixed synthetic inputs; no shell or user data.
# ruff: noqa: S603

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from test_gromacs_continuation import _publish
from test_gromacs_execution_runtime import RUN_ID, _request

from biomodals.app.bioinfo.gromacs import app
from biomodals.app.bioinfo.gromacs.continuation import inspect_continuation_source
from biomodals.app.bioinfo.gromacs.continue_run import (
    native_endpoint,
    prepare_continuation_files,
)
from biomodals.app.bioinfo.gromacs.execution import PREPARE_RESULT
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    gromacs_node_paths,
    gromacs_publication_path,
    persist_execution_request,
)


@pytest.mark.parametrize("unlisted_energy", [False, True])
def test_real_native_copy_extend_append_and_completed_redelivery(
    tmp_path, monkeypatch, unlisted_energy
):
    """A noninteracting atom exercises native state and time, not MD quality."""
    gmx = shutil.which("gmx")
    if gmx is None:
        pytest.skip("Local GROMACS is not installed")
    version = subprocess.run(
        [gmx, "--version"], capture_output=True, text=True, check=True
    )
    if "GROMACS version:     2026.1\n" not in version.stdout:
        pytest.skip("Native contract is pinned to GROMACS 2026.1")
    parent = replace(
        _request(),
        run_name="parent",
        simulation_time_ns=1,
        cpu_only=True,
        num_threads=1,
        execution_plan_version="2",
    )
    root = parent.run_root(tmp_path)
    root.mkdir()
    (root / "input.gro").write_text(
        "Synthetic inert atom\n1\n"
        "    1ARG     AR    1   1.000   1.000   1.000  0.0010  0.0000  0.0000\n"
        "   5.00000   5.00000   5.00000\n"
    )
    (root / "topol.top").write_text(
        "[ defaults ]\n1 2 no 1 1\n[ atomtypes ]\nAR 39.948 0 A 0 0\n"
        "[ moleculetype ]\nARG 1\n[ atoms ]\n1 AR 1 ARG AR 1 0 39.948\n"
        "[ system ]\nSynthetic checkpoint contract\n[ molecules ]\nARG 1\n"
    )
    # Large dt is deliberate for ten cheap steps of a force-free atom. This is
    # a file/restart contract fixture, not a physically representative protein.
    (root / "production.mdp").write_text(
        "integrator = md\ndt = 100\nnsteps = 10\n"
        "nstxout-compressed = 1\nnstenergy = 1\nnstcalcenergy = 1\nnstlog = 1\n"
        "nstlist = 1\ncutoff-scheme = Verlet\nverlet-buffer-tolerance = -1\n"
        "rlist = 1.1\ncoulombtype = Cut-off\nrcoulomb = 1\nvdwtype = Cut-off\n"
        "rvdw = 1\npbc = xyz\ncomm-mode = None\ngen-vel = no\ntcoupl = no\npcoupl = no\n"
    )
    for args in (
        [
            "grompp",
            "-f",
            "production.mdp",
            "-c",
            "input.gro",
            "-p",
            "topol.top",
            "-o",
            "production_parent.tpr",
        ],
        [
            "mdrun",
            "-deffnm",
            "production_parent",
            "-nt",
            "1",
            "-nb",
            "cpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-pin",
            "off",
        ],
    ):
        subprocess.run(
            [gmx, *args], cwd=root, capture_output=True, check=True, timeout=30
        )
    # Analysis placeholders only construct the retained historical publication;
    # the native checkpoint, TPR, log, energy and trajectory above are real.
    for node in parent.execution_plan.nodes:
        for name in gromacs_node_paths(parent, node.node_key):
            path = root / name
            if not path.exists():
                path.write_bytes(
                    parent.pdb_content if name == "parent.pdb" else b"analysis"
                )
        _publish(tmp_path, parent, node.node_key)
    persist_execution_request(tmp_path, RUN_ID, parent)
    if unlisted_energy:
        for node in (PREPARE_RESULT, "production_run_cpu"):
            path = tmp_path / gromacs_publication_path(parent, node)
            marker = orjson.loads(path.read_bytes())
            marker["files"] = [
                file for file in marker["files"] if not file["path"].endswith(".edr")
            ]
            path.write_bytes(orjson.dumps(marker))
    child = inspect_continuation_source(tmp_path, RUN_ID).request(
        run_name="child",
        additional_time_ns=1,
        cpu_only=True,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=0,
    )
    original = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    child_root = prepare_continuation_files(child, tmp_path)
    # Simulate an interrupted provider input after half of the added interval.
    subprocess.run(
        [
            gmx,
            "mdrun",
            "-deffnm",
            "production_parent",
            "-cpi",
            "production_parent.cpt",
            "-append",
            "-nsteps",
            "5",
            "-nt",
            "1",
            "-nb",
            "cpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-pin",
            "off",
        ],
        cwd=child_root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    partial_checkpoint = (child_root / "production_parent.cpt").read_bytes()
    prepare_continuation_files(child, tmp_path)
    assert (child_root / "production_parent.cpt").read_bytes() == partial_checkpoint
    monkeypatch.setattr(
        app,
        "CONF",
        SimpleNamespace(
            output_volume_mountpoint=str(tmp_path),
            output_volume=SimpleNamespace(commit=lambda: None, reload=lambda: None),
        ),
    )
    for _ in range(2):
        app.production_run_cpu.get_raw_f()(
            run_name="child",
            simulation_time_ns=2,
            file_stem="parent",
            num_threads=1,
            require_checkpoint=True,
            fixed_target=True,
        )
        step, time_ps, _ = native_endpoint(
            gmx,
            child_root / "production_parent.tpr",
            child_root / "production_parent.cpt",
            child_root,
        )
        assert (step, time_ps) == (20, 2000)
    assert (
        (child_root / "production_parent.xtc")
        .read_bytes()
        .startswith(original[Path("production_parent.xtc")])
    )
    assert original == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
