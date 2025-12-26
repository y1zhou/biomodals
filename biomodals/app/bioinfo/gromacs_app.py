"""Run MD simulation with GROMACS: <https://www.gromacs.org/>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-pdb` | **Required** | Path to the input PDB file. |
| `--run-name` | PDB filename stem | Name for this simulation run. Note that if the name exists in the remote volume, files in the remote will be preferred over the local one. Make sure to use unique names if you want to start a new run! |
| `--simulation-time-ns` | 5 | Length of the production MD simulation in nanoseconds. |
| `--run-pdbfixer` | False | Whether to run PDBFixer to clean the input PDB file before preparation. |
| `--cpu-only` | False | Whether to run GROMACS on CPU only. If False, GROMACS will use GPU acceleration. |
| `--num-threads` | 32 | Number of CPU threads to use for GROMACS. |
| `--use-openmp-threads` | False | Whether to use OpenMP threading in GROMACS. If False, GROMACS will use its own internal threading. |
| `--ld-seed` | -1 | Random seed for the Langevin dynamics thermostat during equilibration. If -1, a random seed will be chosen. |
| `--gen-seed` | -1 | Random seed for initial velocity generation during equilibration. If -1, a random seed will be chosen. |
| `--genion-seed` | 0 | Random seed for ion placement during system neutralization. |

## Outputs

* All output files are saved to a Modal volume named `gromacs-outputs`.
* The production trajectory should be under the name `production_{run_name}.xtc`.
"""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from pathlib import Path

import modal
from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "86400"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "Gromacs")
N_GMX_THREADS = int(os.environ.get("N_GMX_THREADS", "16"))

# Volume for outputs
OUTPUTS_VOLUME_NAME = "gromacs-outputs"
OUTPUTS_VOLUME = Volume.from_name(
    OUTPUTS_VOLUME_NAME, create_if_missing=True, version=2
)
OUTPUTS_DIR = f"/{OUTPUTS_VOLUME_NAME}"
GMX_SCRIPTS = "/gromacs-scripts"

# Dependency versions
UCX_TAG = "1.18.0"
OPENMPI_TAG = "5.0.6"
FFTW_TAG = "3.3.10"
GROMACS_TAG = "2024.5"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.13")
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
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu128",  # find best torch and CUDA versions
            "PATH": "/root/.local/bin:$PATH",
        }
    )
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
                f"wget https://github.com/openucx/ucx/releases/download/v{UCX_TAG}/ucx-{UCX_TAG}.tar.gz",
                f"tar -xzf ucx-{UCX_TAG}.tar.gz",
                f"rm ucx-{UCX_TAG}.tar.gz",
                f"cd ucx-{UCX_TAG}/",
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
                f"wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-{OPENMPI_TAG}.tar.bz2",
                f"tar -xf openmpi-{OPENMPI_TAG}.tar.bz2",
                f"rm openmpi-{OPENMPI_TAG}.tar.bz2",
                f"cd openmpi-{OPENMPI_TAG}/",
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
                f"wget http://www.fftw.org/fftw-{FFTW_TAG}.tar.gz",
                f"tar -xzf fftw-{FFTW_TAG}.tar.gz",
                f"rm fftw-{FFTW_TAG}.tar.gz",
                f"cd fftw-{FFTW_TAG}/",
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
                "wget https://ftp.gromacs.org/gromacs/gromacs-2024.5.tar.gz",
                f"tar -xzf gromacs-{GROMACS_TAG}.tar.gz",
                f"rm gromacs-{GROMACS_TAG}.tar.gz",
                f"cd gromacs-{GROMACS_TAG}/",
                "mkdir build",
                "cd build",
                "cmake .. "
                "-DCMAKE_BUILD_TYPE=Release "
                "-DCMAKE_PREFIX_PATH='/usr/local' "
                "-DGMX_GPU=CUDA "
                "-DGMX_BUILD_OWN_FFTW=OFF -DGMX_FFT_LIBRARY=fftw3 "
                "-DGMX_SIMD=AVX_512",
                "make -j install",
                # Build GROMACS with OpenMPI
                f"cd /opt/gromacs-{GROMACS_TAG}/",
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
                "-DGMX_SIMD=AVX_512",
                "make -j install",
            ),
        ),
    )
    .run_commands(
        "echo 'micromamba activate base' >> /etc/profile",
        "echo 'source /usr/local/gromacs/bin/GMXRC' >> /etc/profile",
    )
    .add_local_dir(Path(__file__).parent / "gromacs", GMX_SCRIPTS, copy=True)
)

biotite_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential")
    .uv_pip_install("biotite", "numpy", "scipy", "seaborn", "matplotlib")
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=N_GMX_THREADS + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def prepare_tpr_gpu(
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    num_threads: int = N_GMX_THREADS,
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

    work_path = Path(OUTPUTS_DIR) / run_name
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

    script_path = Path(GMX_SCRIPTS) / "prepare-tpr.sh"
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
    env = os.environ.copy()
    if "OMP_NUM_THREADS" in env:
        del env["OMP_NUM_THREADS"]

    _ = run_command(cmd, cwd=str(work_path), env=env)
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    cpu=N_GMX_THREADS + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def prepare_tpr_cpu(
    pdb_content: bytes,
    run_name: str,
    simulation_time_ns: int = 5,
    run_pdbfixer: bool = False,
    num_threads: int = N_GMX_THREADS,
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

    work_path = Path(OUTPUTS_DIR) / run_name
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

    script_path = Path(GMX_SCRIPTS) / "prepare-tpr.sh"
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

    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    gpu=GPU,
    cpu=N_GMX_THREADS + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def production_run_gpu(
    run_name: str, num_threads: int = N_GMX_THREADS, use_openmp_threads: bool = False
) -> Path:
    """Production Gromacs run."""
    import shutil
    from pathlib import Path

    work_path = Path(OUTPUTS_DIR) / run_name
    tpr_file_path = work_path / f"production_{run_name}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # TODO: pick up exisiting trajectory and continue simulation
    traj_file_path = work_path / f"production_{run_name}.xtc"
    if traj_file_path.exists():
        return work_path

    gmx = shutil.which("gmx_mpi") if use_openmp_threads else shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError("Gromacs binary not found in PATH.")

    cmd = [
        gmx,
        "mdrun",
        "-deffnm",
        tpr_file_path.stem,
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
    env = os.environ.copy()
    if "OMP_NUM_THREADS" in env:
        del env["OMP_NUM_THREADS"]

    _ = run_command(cmd, cwd=str(work_path), env=env)
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    cpu=N_GMX_THREADS + 0.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def production_run_cpu(
    run_name: str, num_threads: int = N_GMX_THREADS, use_openmp_threads: bool = False
) -> Path:
    """Production Gromacs run."""
    import shutil
    from pathlib import Path

    work_path = Path(OUTPUTS_DIR) / run_name
    tpr_file_path = work_path / f"production_{run_name}.tpr"
    if not tpr_file_path.exists():
        raise FileNotFoundError(f"Production topology file not found: {tpr_file_path}")

    # TODO: pick up exisiting trajectory and continue simulation
    traj_file_path = work_path / f"production_{run_name}.xtc"
    if traj_file_path.exists():
        return work_path

    gmx = shutil.which("gmx_mpi") if use_openmp_threads else shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError("Gromacs binary not found in PATH.")

    cmd = [
        gmx,
        "mdrun",
        "-deffnm",
        tpr_file_path.stem,
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
    env = os.environ.copy()
    if "OMP_NUM_THREADS" in env:
        del env["OMP_NUM_THREADS"]

    _ = run_command(cmd, cwd=str(work_path), env=env)
    OUTPUTS_VOLUME.commit()
    return work_path


@app.function(
    image=biotite_image,
    cpu=1,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
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

    work_path = Path(OUTPUTS_DIR) / run_name
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
    if not processed_traj_path.exists() and save_processed_traj:
        new_traj_file = xtc.XTCFile()
        new_traj_file.set_structure(trajectory, time=time * 1000.0)  # time in ps
        new_traj_file.write(processed_traj_path)
        OUTPUTS_VOLUME.commit()

    # Dump the last frame of the processed trajectory as PDB
    last_frame_path = work_path / f"{traj_prefix}{run_name}.pdb"
    if not last_frame_path.exists():
        strucio.save_structure(last_frame_path, trajectory[-1])
        OUTPUTS_VOLUME.commit()

    # RMSD vs. the initial frame
    rmsd_fig_path = work_path / f"rmsd_{traj_prefix}{run_name}.png"
    rmsd_csv_path = rmsd_fig_path.with_suffix(".csv")
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
    num_threads: int = N_GMX_THREADS,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
) -> None:
    """Run GROMACS MD simulations on Modal and save results to a volume."""
    from pathlib import Path

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
            num_threads=num_threads,
            use_openmp_threads=use_openmp_threads,
        )
    else:
        _ = production_run_gpu.remote(
            run_name=run_name,
            num_threads=num_threads,
            use_openmp_threads=use_openmp_threads,
        )

    print("🧬 Postprocessing Gromacs trajectory and generating analysis plots...")
    prod_traj_task = postprocess_traj.spawn(
        run_name=run_name, traj_prefix="production_", save_processed_traj=True
    )

    _ = modal.FunctionCall.gather(*process_traj_tasks, prod_traj_task)

    remote_volume_dir = remote_workdir.relative_to(OUTPUTS_DIR)
    print("🧬 Gromacs preparation complete! Check data with: \n")
    print(f"  modal volume ls {OUTPUTS_VOLUME_NAME} {remote_volume_dir}")
