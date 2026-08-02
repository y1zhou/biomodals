"""Run MD simulation with GROMACS: <https://www.gromacs.org/>.

**It is recommended to run this app in detached mode, as the runs can be very long.**

## Outputs

* All output files are saved to a Modal volume named `Gromacs-outputs`.
* The production trajectory should be under the name `production_{run_name}.xtc`.
"""
# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import modal

from biomodals.app.bioinfo.gromacs_execution_runtime import (
    GromacsExecutionCoordinator,
    GromacsExecutionRequest,
    stage_execution_request,
)
from biomodals.app.config import AppConfig
from biomodals.execution import DeploymentIdentity, ExecutionSnapshot, RunStatus
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_identity,
)
from biomodals.execution.modal import (
    execution_coordinator_handle as _execution_coordinator_handle,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout, volume_path_from_mount_path
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import run_command
from biomodals.schema import ArtifactFile

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

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_gromacs_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8


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


def write_analysis_csv(path: Path, columns: dict[str, object]) -> None:
    """Write one stable five-decimal analysis checkpoint with Polars."""
    import polars as pl

    pl.DataFrame(columns).write_csv(path, float_precision=5)


def remove_stale_analysis_outputs(
    csv_path: Path,
    figure_path: Path,
    trajectory_path: Path,
    *,
    make_figures: bool,
) -> None:
    """Invalidate each stale analysis member independently."""
    if file1_needs_update(csv_path, trajectory_path):
        csv_path.unlink(missing_ok=True)
    if make_figures and file1_needs_update(figure_path, trajectory_path):
        figure_path.unlink(missing_ok=True)


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
    if not processed_traj_path.exists():
        postprocess_traj.remote(
            str(traj_path),
            str(work_path / f"{traj_prefix}{run_name}.tpr"),
            str(processed_traj_path),
            ref_struct_file=str(work_path / f"{run_name}.pdb"),
        )

    out_vol = CONF.output_volume
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
    remove_stale_analysis_outputs(
        rmsd_csv_path,
        rmsd_fig_path,
        traj_path,
        make_figures=make_figures,
    )
    if not rmsd_csv_path.exists() or (make_figures and not rmsd_fig_path.exists()):
        rmsd = struc.rmsd(trajectory[0], trajectory)
        if not rmsd_csv_path.exists():
            write_analysis_csv(
                rmsd_csv_path,
                {"time_ns": time, "rmsd": rmsd},
            )

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
    remove_stale_analysis_outputs(
        rg_csv_path,
        rg_fig_path,
        traj_path,
        make_figures=make_figures,
    )
    if not rg_csv_path.exists() or (make_figures and not rg_fig_path.exists()):
        rg = struc.gyration_radius(trajectory)
        if not rg_csv_path.exists():
            write_analysis_csv(
                rg_csv_path,
                {"time_ns": time, "rg": rg},
            )
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
    remove_stale_analysis_outputs(
        rmsf_csv_path,
        rmsf_fig_path,
        traj_path,
        make_figures=make_figures,
    )
    if not rmsf_csv_path.exists() or (make_figures and not rmsf_fig_path.exists()):
        # Sidechain atoms fluctuate too much, so we only consider CA atoms
        ca_trajectory = trajectory[:, trajectory.atom_name == "CA"]
        rmsf = struc.rmsf(struc.average(ca_trajectory), ca_trajectory)
        res_count = struc.get_residue_count(trajectory)
        res_idx = np.arange(1, res_count + 1)
        if not rmsf_csv_path.exists():
            write_analysis_csv(
                rmsf_csv_path,
                {
                    "residue_index": res_idx.astype(float),
                    "rmsf": rmsf,
                },
            )
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


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with GROMACS functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh output state before accepting lifecycle methods."""
        self._coordinator_adapter = None
        self._development = None
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionSnapshot:
        """Drive one staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionSnapshot:
        """Read this Run's durable kernel snapshot."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionSnapshot:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying failed Tasks."""
        return self._adapter().resume()

    @modal.method()
    def restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive one compatible Successor Run."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
        workload_plan_fingerprint: str,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
    ) -> ExecutionSnapshot:
        """Create a compatible Successor while inferring predecessor identity."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            expected_workload_plan_fingerprint=workload_plan_fingerprint,
        )

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return execution_coordinator_identity(self)

    def _adapter(
        self,
        *,
        development: bool | None = None,
    ) -> GromacsExecutionCoordinator:
        adapter = getattr(self, "_coordinator_adapter", None)
        selected_mode = getattr(self, "_development", None)
        if adapter is not None:
            if development is not None and selected_mode != development:
                raise ValueError("Coordinator execution mode cannot change in place")
            return adapter
        execution_run_id, deployment = self._identity()
        selected_mode = False if development is None else development
        adapter = GromacsExecutionCoordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=Path(CONF.output_volume_mountpoint),
            output_volume=CONF.output_volume,
            modal_driver=_coordinator_modal_driver(development=selected_mode),
        )
        self._coordinator_adapter = adapter
        self._development = selected_mode
        return adapter


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "prepare_tpr_cpu": prepare_tpr_cpu,
            "prepare_tpr_gpu": prepare_tpr_gpu,
            "collect_traj_stats": collect_traj_stats,
            "production_run_cpu": production_run_cpu,
            "production_run_gpu": production_run_gpu,
        },
        workload_name="GROMACS",
    )


##########################################
# Local entrypoint client
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
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        use_deployed_coordinator: Target the exact deployed coordinator. The
            Biomodals CLI supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    # Load input PDB
    pdb_path = Path(input_pdb).expanduser().resolve()
    pdb_str = pdb_path.read_bytes()
    if run_name is None:
        run_name = pdb_path.stem

    analysis_limit = 2 if max_parallel_analysis is None else max_parallel_analysis
    if analysis_limit < 1:
        raise ValueError("max_parallel_analysis must be positive")
    request = GromacsExecutionRequest(
        run_name=run_name,
        pdb_content=pdb_str,
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
        max_active_provider_calls=min(analysis_limit, 2) + 1,
        max_active_gpu_provider_calls=0 if cpu_only else 1,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    if predecessor_execution_run_id is None:
        stage_execution_request(CONF.output_volume, execution_run_id, request)
    coordinator = _execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(development=not use_deployed_coordinator)
    else:
        call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
            workload_plan_fingerprint=(
                request.execution_plan.workload_plan_fingerprint
            ),
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=request.max_active_gpu_provider_calls,
        )
    print(f"Execution Run ID: {execution_run_id}")
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    snapshot = call.get()
    if snapshot.run.status != RunStatus.SUCCEEDED:
        diagnostic = snapshot.run.status_message or (
            snapshot.run.status_reason.value
            if snapshot.run.status_reason is not None
            else snapshot.run.status.value
        )
        raise RuntimeError(
            f"{CONF.name} Execution Run ended as "
            f"{snapshot.run.status.value}: {diagnostic}"
        )
    remote_workdir = str(Path(CONF.output_volume_mountpoint) / run_name)
    remote_vol = volume_path_from_mount_path(
        remote_workdir, CONF.output_volume_mountpoint, CONF.output_volume_name
    )
    print(f"🧬 Gromacs preparation complete! Check data in {remote_vol}")
