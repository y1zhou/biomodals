"""AF3Score source repo: <https://github.com/Mingchenchen/AF3Score>.

## Additional notes

- AF3Score scores existing protein structures rather than predicting new folds.
- Inputs can be a single `.pdb` file or a directory of `.pdb` files.
- The wrapper preserves AF3Score's internal length-based batching and schedules
  those internal batches in GPU waves when needed.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import (
    copy_files,
    run_command,
    run_command_with_log,
    sanitize_filename,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AF3Score",
    repo_url="https://github.com/Mingchenchen/AF3Score",
    repo_commit_hash="b0764aaa4101f8a22a5f404faef7acc13ee52d06",
    python_version="3.11",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)


@dataclass(frozen=True)
class AppInfo:
    """Container for AF3Score-specific configuration and constants."""

    af3_weights: str = "AlphaFold3/af3.bin"
    out_volume: modal.Volume = CONF.get_out_volume()


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install(
        "build-essential", "cmake", "git", "ninja-build", "pkg-config", "zlib1g-dev"
    )
    .env(
        CONF.default_env
        | {
            "CC": "gcc",
            "CXX": "g++",
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
                f"cd {CONF.git_clone_dir}",
                f"git checkout {CONF.repo_commit_hash}",
            )
        )
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(str(CONF.git_clone_dir), "biopython", "h5py", "pandas")
    .run_commands("build_data")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
def _get_af3_sanitized_name(name: str) -> str:
    """Sanitize a name to be compatible with AF3Score's internal naming."""
    import string

    lower_spaceless_name = name.lower().replace(" ", "_")
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(c for c in lower_spaceless_name if c in allowed_chars)


def _run_paths(run_name: str) -> dict[str, Path]:
    """Return the standard run-level paths for one AF3Score output directory."""
    mount_path = Path(CONF.output_volume_mountpoint)
    run_root = mount_path / run_name
    output_dir = run_root / "outputs"
    return {
        "mount_root": mount_path,
        "run_root": run_root,
        "inputs_dir": run_root / "inputs",
        "prep_dir": run_root / "prepare",
        "output_dir": output_dir,
        "failed_dir": output_dir / "failed_records",
        "metrics_csv": run_root / "af3score_metrics.csv",
    }


def _has_completed_outputs(output_dir: Path, input_id: str) -> bool:
    """Check whether AF3Score wrote the required output JSON files."""
    sample_dir = output_dir / input_id / "seed-10_sample-0"
    return (sample_dir / "summary_confidences.json").exists() and (
        sample_dir / "confidences.json"
    ).exists()


def _collect_input_files(input_root: Path, stage_dir: Path) -> list[Path]:
    """Collect supported AF3Score input files from a file or directory."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    if input_root.is_file():
        all_files = [input_root] if input_root.suffix == ".pdb" else []
    else:
        all_files = list(input_root.glob("*.pdb"))

    if not all_files:
        raise ValueError(f"No .pdb files were found in '{input_root}'.")

    symlinks: list[Path] = []
    for f in all_files:
        symlink_path = stage_dir / _get_af3_sanitized_name(f.name)
        if symlink_path.exists():
            raise ValueError(f"Duplicated sanitized file name: {symlink_path.name}")
        symlink_path.symlink_to(f)
        symlinks.append(symlink_path)
    return symlinks


def _adjust_num_cpu_gpu(
    total_num_files: int, max_num_batches: int, max_num_workers: int
) -> tuple[int, int]:
    """Adjust the number of CPU workers and GPU batches based on the total number of files."""
    max_num_batches = min(max(1, max_num_batches), total_num_files)
    num_jobs_per_batch = max(
        1, (total_num_files + max_num_batches - 1) // max_num_batches
    )
    adjusted_max_num_workers = min(max(1, max_num_workers), num_jobs_per_batch)
    return adjusted_max_num_workers, max_num_batches


@dataclass
class ChunkSpec:
    """Container for AF3Score batch chunk specifications."""

    batch_name: str  # Unique name for the batch, e.g. "batch_0"
    batch_json_dir: str  # Path to the batch's input JSON directory
    batch_pdb_dir: str  # Path to the batch's input PDB directory


@dataclass
class TaskSpec:
    """Container for AF3Score batch task specifications."""

    total: int  # Total number of input files
    pending: int  # Number of input files pending AF3Score processing
    skipped: int  # Number of input files skipped due to existing outputs
    input_files: list[str]  # List of all input file names (including suffix)
    chunk_specs: list[ChunkSpec]  # List of batch chunk specifications
    output_dir: str  # Path to the remote AF3Score output directory
    failed_dir: str  # Path to the remote AF3Score failed records directory


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 1.125),
    memory=(512, 2048),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: AppInfo.out_volume},
)
def af3score_manage_lock(run_name: str, acquire: bool = True) -> None:
    """Internal-only remote helper for acquiring or releasing one run-level lock."""
    # TODO: replace with a task queue; mkdir in Volumes may not be atomic
    AppInfo.out_volume.reload()
    paths = _run_paths(run_name)
    root_dir = paths["run_root"]
    lock_dir = root_dir / ".run.lock"
    if acquire:
        root_dir.mkdir(parents=True, exist_ok=True)
        try:
            lock_dir.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"`{run_name=}` is already in use by another active AF3Score run."
            ) from exc
        AppInfo.out_volume.commit()
        return

    if lock_dir.exists():
        lock_dir.rmdir()
        AppInfo.out_volume.commit()


@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: AppInfo.out_volume},
)
def af3score_prepare(
    paths: dict[str, Path], input_files: list[str], num_jobs: int, prepare_workers: int
) -> TaskSpec:
    """Prepare AF3Score batches from staged inputs."""
    AppInfo.out_volume.reload()
    staged_dir = paths["inputs_dir"].resolve()
    if not staged_dir.exists():
        raise FileNotFoundError(f"Staged input directory not found: {staged_dir}")

    for path in (paths["output_dir"], paths["failed_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    all_files = [staged_dir / input_name for input_name in input_files]
    input_names = [path.name for path in all_files]
    total_files = len(all_files)
    print(f"💊 [INFO] Total files: {total_files}")
    print(f"💊 [INFO] Output root: {paths['run_root']}")

    pending_files: list[Path] = []
    skipped = 0
    out_dir = paths["output_dir"]
    for pdb_file in all_files:
        if _has_completed_outputs(out_dir, pdb_file.stem):
            # print(f"💊 [SKIP] {pdb_file.name}", flush=True)
            skipped += 1
            continue
        # print(f"💊 [BATCH] Pending: {pdb_file.name}", flush=True)
        pending_files.append(pdb_file)

    if not pending_files:
        return TaskSpec(
            total=total_files,
            pending=0,
            skipped=skipped,
            input_files=input_names,
            chunk_specs=[],
            output_dir=str(out_dir),
            failed_dir=str(paths["failed_dir"]),
        )

    prepare_root = paths["prep_dir"]
    pending_input_dir = prepare_root / "pending_inputs"
    batch_dir = prepare_root / "input_batch"
    if prepare_root.exists():
        shutil.rmtree(prepare_root)
    pending_input_dir.mkdir(parents=True, exist_ok=True)

    copy_files(
        {
            source_path: pending_input_dir / source_path.name
            for source_path in pending_files
        }
    )
    # Adjust CPU and GPU resources
    n_cpu, n_batches = _adjust_num_cpu_gpu(
        len(pending_files), num_jobs, prepare_workers
    )
    run_command(
        [
            sys.executable,
            str(CONF.git_clone_dir / "01_prepare_get_json.py"),
            f"--input_dir={pending_input_dir}",
            f"--output_dir_cif={prepare_root / 'single_chain_cif'}",
            f"--save_csv={prepare_root / 'single_seq.csv'}",
            f"--output_dir_json={prepare_root / 'json'}",
            f"--batch_dir={batch_dir}",
            f"--num_jobs={n_batches}",
            f"--num_workers={n_cpu}",
        ]
    )

    chunk_specs: list[ChunkSpec] = []
    batch_json_root = batch_dir / "json"
    if batch_json_root.exists():
        for batch_json_dir in batch_json_root.iterdir():
            if not batch_json_dir.is_dir():
                continue
            chunk_specs.append(
                ChunkSpec(
                    batch_name=batch_json_dir.name,
                    batch_json_dir=str(batch_json_dir),
                    batch_pdb_dir=str(batch_dir / "pdb" / batch_json_dir.name),
                )
            )

    print(f"💊 [INFO] Prepared {len(chunk_specs)} internal batches")
    return TaskSpec(
        total=total_files,
        pending=len(pending_files),
        skipped=skipped,
        input_files=input_names,
        chunk_specs=chunk_specs,
        output_dir=str(paths["output_dir"]),
        failed_dir=str(paths["failed_dir"]),
    )


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes={
        CONF.output_volume_mountpoint: AppInfo.out_volume,
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
    },
)
def af3score_run(
    paths: dict[str, Path], batch_name: str, batch_json_dir: str, batch_pdb_dir: str
):
    """Run one AF3Score batch."""
    AppInfo.out_volume.reload()
    af3_weights = Path(CONF.model_volume_mountpoint) / APP_INFO.af3_weights
    if not af3_weights.exists():
        raise FileNotFoundError(f"AlphaFold3 model weights not found: {af3_weights}")

    with TemporaryDirectory(prefix=f"af3score_gpu_{batch_name}_") as temp_dir:
        batch_gpu_root = Path(temp_dir)
        batch_h5_dir = batch_gpu_root / "jax"
        batch_h5_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Benchmark whether AF3Score's JAX preprocessing can safely scale past one worker.
        jax_workers = 1
        run_command(
            [
                sys.executable,
                str(CONF.git_clone_dir / "02_prepare_pdb2jax.py"),
                f"--pdb_folder={batch_pdb_dir}",
                f"--output_folder={batch_h5_dir}",
                f"--num_workers={jax_workers}",
            ]
        )

        # TODO: this or reuse AlphaFold3 buckets?
        bucket = batch_name.rsplit("_", 1)[-1]
        out_dir = paths["output_dir"]
        run_command_with_log(
            [
                sys.executable,
                str(CONF.git_clone_dir / "run_af3score.py"),
                f"--model_dir={af3_weights.parent}",
                f"--batch_json_dir={batch_json_dir}",
                f"--batch_h5_dir={batch_h5_dir}",
                f"--output_dir={out_dir}",
                "--run_data_pipeline=False",
                "--run_inference=true",
                "--init_guess=true",
                "--num_samples=1",
                f"--buckets={bucket}",
                "--write_cif_model=False",
                "--write_summary_confidences=true",
                "--write_full_confidences=true",
                "--write_best_model_root=false",
                "--write_ranking_scores_csv=false",
                "--write_terms_of_use_file=false",
                "--write_fold_input_json_file=false",
            ],
            log_file=out_dir / f"{batch_name}.log",
        )
        AppInfo.out_volume.commit()


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: AppInfo.out_volume},
)
def af3score_postprocess(
    input_files: list[str], paths: dict[str, Path]
) -> dict[str, int | str]:
    """Validate records and collect metrics for all inputs."""
    AppInfo.out_volume.reload()
    for path in (paths["output_dir"], paths["failed_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    completed_output_dirs: list[Path] = []
    out_dir = paths["output_dir"]
    for input_name in input_files:
        input_id = Path(input_name).stem
        failed_record = paths["failed_dir"] / f"{input_id}.err"
        if _has_completed_outputs(out_dir, input_id):
            if failed_record.exists():
                failed_record.unlink()
            processed += 1
            completed_output_dirs.append(out_dir / input_id)
        else:
            failed_record.write_text(
                f"Missing AF3 output files for sample '{input_id}'"
            )
            failed += 1

    out_csv_path = paths["metrics_csv"]
    if not completed_output_dirs:
        if out_csv_path.exists():
            out_csv_path.unlink()
        raise RuntimeError(
            "No completed AF3Score outputs were found; cannot generate metrics CSV."
        )

    with TemporaryDirectory(prefix="af3score_metrics_") as temp_dir:
        temp_root = Path(temp_dir)
        metrics_view_dir = temp_root / "metrics_view"
        metrics_view_dir.mkdir()
        for candidate in completed_output_dirs:
            (metrics_view_dir / candidate.name).symlink_to(
                candidate,
                target_is_directory=True,
            )
        run_command(
            [
                sys.executable,
                str(CONF.git_clone_dir / "04_get_metrics.py"),
                f"--input_pdb_dir={paths['inputs_dir']}",
                f"--af3score_output_dir={metrics_view_dir}",
                f"--save_metric_csv={out_csv_path}",
                f"--num_workers={max(1, min(16, len(completed_output_dirs)))}",
            ]
        )

    with out_csv_path.open(encoding="utf-8") as f:
        metrics_rows = max(0, sum(1 for _ in f) - 1)

    if paths["prep_dir"].exists():
        shutil.rmtree(paths["prep_dir"])
    AppInfo.out_volume.commit()
    return {
        "output_dir": str(out_dir),
        "failed_dir": str(paths["failed_dir"]),
        "total": len(input_files),
        "processed": processed,
        "failed": failed,
        "metrics_csv_exists": int(out_csv_path.exists()),
        "metrics_csv": str(out_csv_path),
        "metrics_rows": metrics_rows,
    }


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_af3score_task(
    input_dir: str,
    run_name: str,
    output_dir: str | None = None,
    prepare_workers: int = 8,
    max_batches: int = 10,
    force: bool = True,
) -> None:
    """Stage local PDB inputs, run AF3Score on Modal, and download the final metrics CSV.

    Args:
        input_dir: Path to a single PDB file or a directory of PDB files. Note
            that only `.pdb` files are supported as structural inputs.
        run_name: Remote run directory name under the Modal volume root.
        output_dir: Local directory to save the final AF3Score metrics CSV. If
            not specified, the current working directory will be used.
        prepare_workers: Number of CPUs to use for processing input PDBs into
            AlphaFold3-style input files (JSON and each chain as CIF template).
        max_batches: Maximum number of batches (GPU tasks) to run at the same
            time. AF3Score internally batches inputs of similar lengths
            together in the `01_prepare_get_json.py` script, so we don't need
            to batch manually when uploading inputs.
        force: If True, ignore existing PDB files when uploading `input_dir`.
    """
    input_root = Path(input_dir).expanduser().resolve()
    stage_tmp_dir = Path(mkdtemp())
    all_files = _collect_input_files(input_root, stage_tmp_dir)
    num_files = len(all_files)
    print(f"🧬 Total files: {num_files}")

    # TODO: Deal with duplicate inputs. Locks? Distributed dicts?
    run_name = sanitize_filename(run_name)
    run_paths = _run_paths(run_name)
    if not force:
        for x in AppInfo.out_volume.iterdir("/"):
            if x.path == run_name:
                raise ValueError(
                    f"Run name '{run_name}' already exists in Modal volume."
                )
    af3score_manage_lock.remote(run_name=run_name, acquire=True)
    try:
        print(f"🧬 Uploading files in {input_root} to Modal")
        stage_root = run_paths["inputs_dir"].relative_to(run_paths["mount_root"])
        with AppInfo.out_volume.batch_upload(force=force) as batch:
            if num_files == 1:
                f = all_files[0]
                batch.put_file(f, f"/{stage_root}/{f.name}")
            else:
                batch.put_directory(all_files[0].parent, f"/{stage_root}/")

        prepare_result = af3score_prepare.remote(
            paths=run_paths,
            input_files=[path.name for path in all_files],
            num_jobs=max_batches,
            prepare_workers=prepare_workers,
        )
        for k in ("total", "pending", "skipped"):
            print(f"🧬 [PREPARE] {k}: {getattr(prepare_result, k)}")

        chunk_specs = prepare_result.chunk_specs
        total_chunks = len(chunk_specs)

        def _af3score_run(spec: ChunkSpec) -> None:
            """Submit one AF3Score batch run as a remote function call."""
            af3score_run.remote(
                paths=run_paths,
                batch_name=spec.batch_name,
                batch_json_dir=spec.batch_json_dir,
                batch_pdb_dir=spec.batch_pdb_dir,
            )

        if total_chunks:
            max_batches = min(max_batches, total_chunks)
            print(f"🧬 Running {total_chunks} batches with a max of {max_batches} GPUs")

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_batches) as executor:
                futures = [executor.submit(_af3score_run, spec) for spec in chunk_specs]
                for future in futures:
                    future.result()  # wait for all workers to finish

        postprocess_result = af3score_postprocess.remote(
            input_files=prepare_result.input_files,
            paths=run_paths,
        )
        for key, value in postprocess_result.items():
            prefix = "[METRICS]" if str(key).startswith("metrics_") else "[POSTPROCESS]"
            print(f"🧬 {prefix} {key}: {value}")

        total_processed = postprocess_result.get("metrics_rows")
        if isinstance(total_processed, int):
            print(f"🧬 [INFO] {total_processed}/{len(all_files)} done")

        if postprocess_result["metrics_csv_exists"]:
            if output_dir is None:
                local_out_dir = Path.cwd()
            else:
                local_out_dir = Path(output_dir).expanduser().resolve()
            local_out_dir.mkdir(parents=True, exist_ok=True)

            local_metrics_csv = local_out_dir / f"{run_name}_af3score_metrics.csv"
            print("🧬 Downloading metrics CSV...")
            with open(local_metrics_csv, "wb") as f:
                for chunk in AppInfo.out_volume.read_file(
                    str(run_paths["metrics_csv"].relative_to(run_paths["mount_root"]))
                ):
                    f.write(chunk)
            print(f"🧬 Local metrics CSV: {local_metrics_csv}")
        else:
            print("🧬 Local metrics CSV: not generated")
    finally:
        af3score_manage_lock.remote(run_name=run_name, acquire=False)
        shutil.rmtree(stage_tmp_dir)
