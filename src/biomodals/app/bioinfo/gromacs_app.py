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

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT
from biomodals.helper import patch_image_for_helper
from biomodals.helper.shell import run_command

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
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
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
OUTPUTS_VOLUME = CONF.get_out_volume()
OUTPUTS_VOLUME_NAME = OUTPUTS_VOLUME.name or f"{CONF.name}-outputs"

runtime_image = patch_image_for_helper(
    modal.Image.from_registry(
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
    .env(
        {
            "PATH": "/usr/local/gromacs/bin:/root/micromamba/bin:$PATH",
            "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib:${LD_LIBRARY_PATH}",
        }
    )
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
)

biotite_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential")
    .uv_pip_install("biotite", "numpy", "scipy", "seaborn", "matplotlib")
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


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


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
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
) -> Path:
    """Prepare inputs for production Gromacs run.

    Steps: clean input PDB, build topology with Amber FF19SB and TIP3P water,
    solvate, add ions, minimize (em and cg), equilibrate (NVT and NPT), and
    generate production TPR file.
    """
    from pathlib import Path

    work_path = Path(CONF.output_volume_mountpoint) / run_name
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
        return work_path

    input_pdb_path = work_path / f"{run_name}.pdb"
    input_pdb_path.write_bytes(pdb_content)
    OUTPUTS_VOLUME.commit()

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
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
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
) -> Path:
    """Prepare inputs for production Gromacs run.

    Steps: clean input PDB, build topology with Amber FF19SB and TIP3P water,
    solvate, add ions, minimize (em and cg), equilibrate (NVT and NPT), and
    generate production TPR file.
    """
    from pathlib import Path

    work_path = Path(CONF.output_volume_mountpoint) / run_name
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
        return work_path

    input_pdb_path = work_path / f"{run_name}.pdb"
    input_pdb_path.write_bytes(pdb_content)
    OUTPUTS_VOLUME.commit()

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

    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    image=biotite_image,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def find_traj_last_step(traj_file: str, topology_file: str) -> int:
    """Calculated simulated steps from the simulation time (in ps) in a trajectory."""
    from pathlib import Path

    import biotite.structure as struc
    import biotite.structure.io as strucio
    import biotite.structure.io.xtc as xtc
    import numpy as np

    input_pdb_path = Path(topology_file)
    if not input_pdb_path.exists():
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb_path}")
    traj_path = Path(traj_file)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    template = strucio.load_structure(input_pdb_path)
    protein_mask = struc.filter_amino_acids(template)
    template = template[protein_mask]

    xtc_file = xtc.XTCFile.read(traj_path, atom_i=np.where(protein_mask)[0])
    trajectory = xtc_file.get_structure(template)

    # Get simulation time in ps
    time = xtc_file.get_time()
    return int(time[-1] / 0.002)  # dt=2 fs, which is 0.002 ps


@app.function(
    gpu=CONF.gpu,
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def production_run_gpu(
    run_name: str,
    simulation_time_ns: int,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
) -> Path:
    """Production Gromacs run."""
    import shutil
    from pathlib import Path

    work_path = Path(CONF.output_volume_mountpoint) / run_name
    deffnm = f"production_{run_name}"
    tpr_file_path = work_path / f"{deffnm}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # Pick up exisiting trajectory and continue simulation when checkpoint exists
    traj_file_path = work_path / f"{deffnm}.xtc"
    checkpoint_file_path = work_path / f"{deffnm}.cpt"
    nsteps = -2  # default: find nsteps from the mdp file
    if traj_file_path.exists() and checkpoint_file_path.exists():
        simulated_steps = find_traj_last_step.remote(
            str(traj_file_path), str(work_path / f"{run_name}.pdb")
        )
        total_steps = simulation_time_ns * 500000  # 2 fs timestep
        nsteps = total_steps - simulated_steps
        if nsteps <= 0:
            print("✅ Production run already completed, skipping.")
            return work_path

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
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    cpu=APP_INFO.gmx_threads + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def production_run_cpu(
    run_name: str,
    simulation_time_ns: int,
    num_threads: int = APP_INFO.gmx_threads,
    use_openmp_threads: bool = False,
) -> Path:
    """Production Gromacs run."""
    import shutil
    from pathlib import Path

    work_path = Path(CONF.output_volume_mountpoint) / run_name
    deffnm = f"production_{run_name}"
    tpr_file_path = work_path / f"{deffnm}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # Pick up exisiting trajectory and continue simulation when checkpoint exists
    traj_file_path = work_path / f"{deffnm}.xtc"
    checkpoint_file_path = work_path / f"{deffnm}.cpt"
    nsteps = -2  # default: find nsteps from the mdp file
    if traj_file_path.exists() and checkpoint_file_path.exists():
        simulated_steps = find_traj_last_step.remote(
            str(traj_file_path), str(work_path / f"{run_name}.pdb")
        )
        total_steps = simulation_time_ns * 500000  # 2 fs timestep
        nsteps = total_steps - simulated_steps
        if nsteps <= 0:
            print("✅ Production run already completed, skipping.")
            return work_path

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
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    image=biotite_image,
    cpu=1,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def postprocess_traj(
    traj_prefix: str,
    run_name: str,
    save_processed_traj: bool = False,
    make_figures: bool = True,
) -> Path:
    """Process Gromacs trajectory and generate analysis plots.

    Ref: https://www.biotite-python.org/latest/examples/gallery/structure/modeling/md_analysis.html
    """
    from pathlib import Path

    import biotite
    import biotite.structure as struc
    import biotite.structure.io as strucio
    import biotite.structure.io.xtc as xtc
    import matplotlib.pyplot as plt
    import numpy as np

    work_path = Path(CONF.output_volume_mountpoint) / run_name
    input_pdb_path = work_path / f"{run_name}.pdb"
    if not input_pdb_path.exists():
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb_path}")
    traj_path = work_path / f"{traj_prefix}{run_name}.xtc"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # Gromacs does not set the element symbol in its PDB files,
    # but Biotite guesses the element names from the atom names,
    # emitting a warning
    template = strucio.load_structure(input_pdb_path)
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
    xtc_file = xtc.XTCFile.read(traj_path, atom_i=np.where(protein_mask)[0])
    trajectory = xtc_file.get_structure(template)

    # Get simulation time (ns) for plotting purposes
    time = xtc_file.get_time() / 1000.0
    print(f"Simulated {time[-1]:.1f} ns")

    # Remove PBC (gmx trjconv)
    trajectory = struc.remove_pbc(trajectory)
    trajectory, _ = struc.superimpose(trajectory[0], trajectory)

    # Save the processed trajectory
    processed_traj_path = work_path / f"{traj_prefix}{run_name}_nopbc.xtc"
    if file1_needs_update(processed_traj_path, traj_path):
        processed_traj_path.unlink(
            missing_ok=True
        )  # remove outdated processed trajectory
    if not processed_traj_path.exists() and save_processed_traj:
        new_traj_file = xtc.XTCFile()
        new_traj_file.set_structure(trajectory, time=time * 1000.0)  # time in ps
        new_traj_file.write(processed_traj_path)
        OUTPUTS_VOLUME.commit()

    # Dump the last frame of the processed trajectory as PDB
    last_frame_path = work_path / f"{traj_prefix}{run_name}.pdb"
    if file1_needs_update(last_frame_path, traj_path):
        last_frame_path.unlink(missing_ok=True)  # remove outdated last frame
    if not last_frame_path.exists():
        strucio.save_structure(last_frame_path, trajectory[-1])
        OUTPUTS_VOLUME.commit()

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
        OUTPUTS_VOLUME.commit()

        if not rmsd_fig_path.exists() and make_figures:
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            ax.plot(time, rmsd, color=biotite.colors["dimorange"])
            ax.set_xlim(time[0], time[-1])
            ax.set_title(run_name)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("RMSD (Å)")
            figure.savefig(rmsd_fig_path)
            plt.close(figure)

            OUTPUTS_VOLUME.commit()

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
        OUTPUTS_VOLUME.commit()
        if not rg_fig_path.exists() and make_figures:
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            ax.plot(time, rg, color=biotite.colors["dimgreen"])
            ax.set_xlim(time[0], time[-1])
            ax.set_title(run_name)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Radius of Gyration (Å)")
            figure.savefig(rg_fig_path)
            plt.close(figure)

            OUTPUTS_VOLUME.commit()

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
        OUTPUTS_VOLUME.commit()
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

            OUTPUTS_VOLUME.commit()

    return work_path


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
    """
    # Load input PDB
    pdb_path = Path(input_pdb).expanduser().resolve()
    pdb_str = pdb_path.read_bytes()
    if run_name is None:
        run_name = pdb_path.stem

    print("🧬 Preparing Gromacs production run...")
    prepare_tpr_conf = {
        "pdb_content": pdb_str,
        "run_name": run_name,
        "simulation_time_ns": simulation_time_ns,
        "run_pdbfixer": run_pdbfixer,
        "num_threads": num_threads,
        "use_openmp_threads": use_openmp_threads,
        "ld_seed": ld_seed,
        "gen_seed": gen_seed,
        "genion_seed": genion_seed,
    }
    if cpu_only:
        remote_workdir = prepare_tpr_cpu.remote(**prepare_tpr_conf)
    else:
        remote_workdir = prepare_tpr_gpu.remote(**prepare_tpr_conf)

    process_traj_tasks = [
        postprocess_traj.spawn(prefix, run_name=run_name) for prefix in ["nvt_", "npt_"]
    ]

    print("🧬 Starting Gromacs production MD simulation...")
    if cpu_only:
        _ = production_run_cpu.remote(
            run_name=run_name,
            simulation_time_ns=simulation_time_ns,
            num_threads=num_threads,
            use_openmp_threads=use_openmp_threads,
        )
    else:
        _ = production_run_gpu.remote(
            run_name=run_name,
            simulation_time_ns=simulation_time_ns,
            num_threads=num_threads,
            use_openmp_threads=use_openmp_threads,
        )

    print("🧬 Postprocessing Gromacs trajectory and generating analysis plots...")
    prod_traj_task = postprocess_traj.spawn(
        run_name=run_name, traj_prefix="production_", save_processed_traj=True
    )

    _ = modal.FunctionCall.gather(*process_traj_tasks, prod_traj_task)

    remote_volume_dir = remote_workdir.relative_to(CONF.output_volume_mountpoint)
    print("🧬 Gromacs preparation complete! Check data with: \n")
    print(f"  modal volume ls {OUTPUTS_VOLUME_NAME} {remote_volume_dir}")
