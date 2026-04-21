"""AF3Score source repo: <https://github.com/Mingchenchen/AF3Score>.

## Additional notes

- AF3Score scores existing protein structures rather than predicting new folds.
- Inputs can be a single `.pdb` file or a directory of `.pdb` files.
- The wrapper preserves AF3Score's internal length-based batching and schedules
  those internal batches in GPU waves when needed.

## Outputs

- Outputs are persisted in the Modal volume `af3score-outputs`, mounted at `/af3score-outputs`.
- Each run writes to `/af3score-outputs/<output_dir_name>`.
- Official AF3Score per-structure directories are written under
  `/af3score-outputs/<output_dir_name>/outputs`.
- Aggregate metrics are written to
  `/af3score-outputs/<output_dir_name>/af3score_metrics.csv`.
- A copy of the aggregate metrics CSV is also written to
  `/af3score-outputs/<output_dir_name>/outputs/af3score_metrics.csv`.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.app.helper import hash_string, patch_image_for_helper
from biomodals.app.helper.shell import copy_files, run_command, sanitize_filename

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

    stage_prefix: str = "af3score_inputs"
    af3_weights: str = "AlphaFold3/af3.bin"
    supported_input_suffixes: frozenset[str] = frozenset({".pdb"})


APP_INFO = AppInfo()
OUTPUTS_VOLUME = CONF.get_out_volume()

##########################################
# Image and app definitions
##########################################
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
def _run_paths(output_dir_name: str) -> dict[str, Path]:
    """Return the standard run-level paths for one AF3Score output directory."""
    run_root = Path(CONF.output_volume_mountpoint) / output_dir_name
    output_dir = run_root / "outputs"
    return {
        "run_root": run_root,
        "lock_dir": run_root / ".run.lock",
        "work_root": run_root / "work",
        "output_dir": output_dir,
        "failed_dir": output_dir / "failed_records",
        "metrics_input_dir": run_root / "metric_inputs",
        "metrics_csv": run_root / "af3score_metrics.csv",
    }


def _sample_output_dir(output_dir: Path, input_name: str | Path) -> Path:
    """Return the canonical AF3Score output directory for one input structure."""
    # TODO: why is this hardcoded?
    return output_dir / Path(input_name).stem.casefold() / "seed-10_sample-0"


def _has_completed_outputs(output_dir: Path, input_name: str | Path) -> bool:
    """Check whether AF3Score wrote the required output JSON files."""
    sample_dir = _sample_output_dir(output_dir, input_name)
    return (sample_dir / "summary_confidences.json").exists() and (
        sample_dir / "confidences.json"
    ).exists()


def _collect_input_files(input_root: Path) -> list[Path]:
    """Collect supported AF3Score input files from a file or directory."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    if input_root.is_file():
        all_files = (
            [input_root]
            if input_root.suffix.lower() in APP_INFO.supported_input_suffixes
            else []
        )
    else:
        all_files = [
            path
            for path in input_root.iterdir()
            if path.is_file()
            and path.suffix.lower() in APP_INFO.supported_input_suffixes
        ]

    if not all_files:
        raise ValueError(
            "No supported structure files were found in the provided input path. "
            f"Supported suffixes: {', '.join(sorted(APP_INFO.supported_input_suffixes))}"
        )

    unique: dict[str, Path] = {}
    for structure_path in all_files:
        stem_key = structure_path.stem.casefold()
        if stem_key in unique and unique[stem_key] != structure_path:
            raise ValueError(
                "Duplicate input structure stems are not supported because output names "
                "must stay stable across resume runs: "
                f"{unique[stem_key]} and {structure_path}"
            )
        unique[stem_key] = structure_path
    return sorted(unique.values(), key=lambda path: path.name.casefold())


def _local_metrics_filename(output_dir_name: str) -> str:
    """Return the local metrics filename for one AF3Score run."""
    return f"{output_dir_name.replace('/', '_')}_af3score_metrics.csv"


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 1.125),
    memory=(512, 2048),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def af3score_manage_lock(output_dir_name: str, acquire: bool = True) -> None:
    """Internal-only remote helper for acquiring or releasing one run-level lock."""
    # TODO: replace with a task queue; mkdir in Volumes may not be atomic
    OUTPUTS_VOLUME.reload()
    paths = _run_paths(output_dir_name)
    if acquire:
        paths["run_root"].mkdir(parents=True, exist_ok=True)
        try:
            paths["lock_dir"].mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"`output_dir_name={output_dir_name}` is already in use by another active AF3Score run."
            ) from exc
        OUTPUTS_VOLUME.commit()
        return

    if paths["lock_dir"].exists():
        shutil.rmtree(paths["lock_dir"])
        OUTPUTS_VOLUME.commit()


@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def af3score_prepare(
    staged_input_dir: str,
    input_files: list[str],
    output_dir_name: str,
    num_jobs: int = 10,
    prepare_workers: int = 8,
) -> dict[str, object]:
    """Internal-only remote step for preparing AF3Score batches from staged inputs."""
    OUTPUTS_VOLUME.reload()
    staged_dir = Path(staged_input_dir).resolve()
    if not staged_dir.exists():
        raise FileNotFoundError(f"Staged input directory not found: {staged_dir}")

    paths = _run_paths(output_dir_name)
    for path in (paths["work_root"], paths["output_dir"], paths["metrics_input_dir"]):
        path.mkdir(parents=True, exist_ok=True)
    paths["failed_dir"].mkdir(exist_ok=True)

    all_files = [
        staged_dir / input_name for input_name in sorted(input_files, key=str.casefold)
    ]
    input_names = [path.name for path in all_files]
    total_files = len(all_files)
    print(f"💊 [INFO] Total files: {total_files}", flush=True)
    print(f"💊 [INFO] Output root: {paths['run_root']}", flush=True)

    copy_files(
        {
            pdb: paths["metrics_input_dir"] / f"{pdb.stem.casefold()}.pdb"
            for pdb in all_files
        }
    )

    pending_files: list[Path] = []
    skipped = 0
    for pdb in all_files:
        if _has_completed_outputs(paths["output_dir"], pdb.name):
            print(f"💊 [SKIP] {pdb.name}", flush=True)
            skipped += 1
            continue
        print(f"💊 [BATCH] Pending: {pdb.name}", flush=True)
        pending_files.append(pdb)

    if not pending_files:
        return {
            "total": total_files,
            "pending": 0,
            "skipped": skipped,
            "input_files": input_names,
            "chunk_specs": [],
            "output_dir": str(paths["output_dir"]),
            "failed_dir": str(paths["failed_dir"]),
        }

    prepare_root = paths["work_root"] / "prepare"
    pending_input_dir = prepare_root / "pending_inputs"
    batch_dir = prepare_root / "input_batch"
    if prepare_root.exists():
        shutil.rmtree(prepare_root)
    prepare_root.mkdir(parents=True, exist_ok=True)
    pending_input_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Let AF3Score consume the canonicalized metric input directory directly.
    copy_files(
        {
            paths["metrics_input_dir"] / f"{source_path.stem.casefold()}.pdb": (
                pending_input_dir / f"{source_path.stem}.pdb"
            )
            for source_path in pending_files
        }
    )

    run_command(
        [
            sys.executable,
            str(CONF.git_clone_dir / "01_prepare_get_json.py"),
            "--input_dir",
            str(pending_input_dir),
            "--output_dir_cif",
            str(prepare_root / "single_chain_cif"),
            "--save_csv",
            str(prepare_root / "single_seq.csv"),
            "--output_dir_json",
            str(prepare_root / "json"),
            "--batch_dir",
            str(batch_dir),
            "--num_jobs",
            str(max(1, num_jobs)),
            "--num_workers",
            str(max(1, prepare_workers)),
        ]
    )

    chunk_specs: list[dict[str, object]] = []
    batch_json_root = batch_dir / "json"
    if batch_json_root.exists():
        for batch_json_dir in sorted(
            path for path in batch_json_root.iterdir() if path.is_dir()
        ):
            chunk_specs.append(
                {
                    "batch_name": batch_json_dir.name,
                    "batch_json_dir": str(batch_json_dir),
                    "batch_pdb_dir": str((batch_dir / "pdb") / batch_json_dir.name),
                }
            )

    print(f"💊 [INFO] Prepared {len(chunk_specs)} internal batches", flush=True)
    return {
        "total": total_files,
        "pending": len(pending_files),
        "skipped": skipped,
        "input_files": input_names,
        "chunk_specs": chunk_specs,
        "output_dir": str(paths["output_dir"]),
        "failed_dir": str(paths["failed_dir"]),
    }


@app.function(
    gpu=CONF.gpu,
    cpu=(2.125, 16.125),
    memory=(4096, 65536),
    timeout=CONF.timeout,
    volumes={
        CONF.output_volume_mountpoint: OUTPUTS_VOLUME,
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
    },
)
def af3score_run(
    output_dir_name: str, batch_name: str, batch_json_dir: str, batch_pdb_dir: str
) -> dict[str, str]:
    """Internal-only remote step for running one AF3Score batch on one GPU."""
    OUTPUTS_VOLUME.reload()
    af3_weights = Path(CONF.model_volume_mountpoint) / APP_INFO.af3_weights
    if not af3_weights.exists():
        raise FileNotFoundError(f"AlphaFold3 model weights not found: {af3_weights}")

    paths = _run_paths(output_dir_name)
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
                "--pdb_folder",
                str(Path(batch_pdb_dir)),
                "--output_folder",
                str(batch_h5_dir),
                "--num_workers",
                str(jax_workers),
            ]
        )

        bucket = batch_name.rsplit("_", 1)[-1]
        run_command(
            [
                sys.executable,
                str(CONF.git_clone_dir / "run_af3score.py"),
                "--model_dir",
                str(af3_weights.parent),
                "--batch_json_dir",
                str(Path(batch_json_dir)),
                "--batch_h5_dir",
                str(batch_h5_dir),
                "--output_dir",
                str(paths["output_dir"]),
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
            ]
        )

    return {
        "batch_name": batch_name,
        "output_dir": str(paths["output_dir"]),
    }


@app.function(
    cpu=(1.125, 8.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def af3score_postprocess(
    input_files: list[str],
    output_dir_name: str = "",
) -> dict[str, int | str]:
    """Internal-only remote step for validation, failure records, and metrics generation."""
    OUTPUTS_VOLUME.reload()
    all_files = sorted(input_files, key=str.casefold)
    paths = _run_paths(output_dir_name)
    for path in (paths["output_dir"], paths["failed_dir"], paths["metrics_input_dir"]):
        path.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    completed_output_dirs: list[Path] = []
    for input_name in all_files:
        stem = Path(input_name).stem
        sample_dir = _sample_output_dir(paths["output_dir"], input_name)
        if _has_completed_outputs(paths["output_dir"], input_name):
            failed_record = paths["failed_dir"] / f"{stem}.err"
            if failed_record.exists():
                failed_record.unlink()
            processed += 1
            completed_output_dirs.append(sample_dir.parent)
            continue

        (paths["failed_dir"] / f"{stem}.err").write_text(
            f"Missing AF3 output files: {sample_dir}",
            encoding="utf-8",
        )
        failed += 1

    metrics_copy_path = paths["output_dir"] / "af3score_metrics.csv"
    metrics_rows = 0
    if completed_output_dirs:
        with TemporaryDirectory(prefix="af3score_metrics_") as temp_dir:
            temp_root = Path(temp_dir)
            metrics_view_dir = temp_root / "metrics_view"
            metrics_view_dir.mkdir(parents=True, exist_ok=True)
            for candidate in completed_output_dirs:
                (metrics_view_dir / candidate.name).symlink_to(
                    candidate,
                    target_is_directory=True,
                )

            temp_metrics_csv = temp_root / "af3score_metrics.csv"
            run_command(
                [
                    sys.executable,
                    str(CONF.git_clone_dir / "04_get_metrics.py"),
                    "--input_pdb_dir",
                    str(paths["metrics_input_dir"]),
                    "--af3score_output_dir",
                    str(metrics_view_dir),
                    "--save_metric_csv",
                    str(temp_metrics_csv),
                    "--num_workers",
                    str(max(1, min(16, os.cpu_count() or 4))),
                ]
            )

            shutil.copy2(temp_metrics_csv, paths["metrics_csv"])
            shutil.copy2(paths["metrics_csv"], metrics_copy_path)

        with paths["metrics_csv"].open(encoding="utf-8") as handle:
            metrics_rows = max(0, sum(1 for _ in handle) - 1)
    else:
        for stale_path in (paths["metrics_csv"], metrics_copy_path):
            if stale_path.exists():
                stale_path.unlink()

    if paths["work_root"].exists():
        shutil.rmtree(paths["work_root"])
    OUTPUTS_VOLUME.commit()
    return {
        "output_dir": str(paths["output_dir"]),
        "failed_dir": str(paths["failed_dir"]),
        "total": len(all_files),
        "processed": processed,
        "failed": failed,
        "metrics_csv_exists": int(paths["metrics_csv"].exists()),
        "metrics_csv": str(paths["metrics_csv"]),
        "metrics_csv_in_output_dir": str(metrics_copy_path),
        "metrics_rows": metrics_rows,
    }


@app.function(
    cpu=(0.125, 1.125),
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUTPUTS_VOLUME},
)
def af3score_fetch_metrics_csv(output_dir_name: str = "") -> bytes | None:
    """Internal-only remote helper for reading the final AF3Score metrics CSV."""
    OUTPUTS_VOLUME.reload()
    metrics_csv = _run_paths(output_dir_name)["metrics_csv"]
    if not metrics_csv.exists():
        return None
    return metrics_csv.read_bytes()


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_af3score_task(
    input_dir: str,
    output_dir_name: str,
    output_dir: str,
    num_jobs: int = 10,
    prepare_workers: int = 8,
    max_concurrent_gpus: int = 10,
) -> None:
    """Stage local PDB inputs, run AF3Score on Modal, and download the final metrics CSV.

    Args:
        input_dir: Path to a single PDB file or a directory of PDB files. Note
            that only `.pdb` files are supported as structural inputs.
        output_dir_name: Remote run directory name under the `af3score-outputs`
            Modal volume root.
        output_dir: Local directory where the final AF3Score metrics CSV will
            be written.
        num_jobs: Target number of internal batches for
            `01_prepare_get_json.py`.
        prepare_workers: Worker count for `01_prepare_get_json.py`.
        max_concurrent_gpus: Maximum number of internal AF3Score batches to run
            at the same time.
    """
    output_dir_name = sanitize_filename(output_dir_name)
    if max_concurrent_gpus < 1:
        raise ValueError("`--max-concurrent-gpus` must be >= 1.")

    input_root = Path(input_dir).expanduser().resolve()
    all_files = _collect_input_files(input_root)

    print(f"🧬 Total files: {len(all_files)}")
    af3score_manage_lock.remote(output_dir_name=output_dir_name, acquire=True)
    try:
        # TODO: Use a content-derived staging key if cross-machine reuse becomes important.
        dataset_hash = hash_string(str(input_root))[:12]
        stage_root = Path(APP_INFO.stage_prefix) / dataset_hash / "inputs"
        remote_stage_root = CONF.output_volume_mountpoint / stage_root

        with OUTPUTS_VOLUME.batch_upload(force=True) as batch:
            print(f"🧬 Processing {len(all_files)} files", flush=True)
            for pdb_path in all_files:
                # TODO: use .put_dir
                # TODO: split into batches based on max_concurrent_gpus
                batch.put_file(pdb_path, f"/{stage_root / pdb_path.name}")

        prepare_result = af3score_prepare.remote(
            staged_input_dir=str(remote_stage_root),
            input_files=[path.name for path in all_files],
            output_dir_name=output_dir_name,
            num_jobs=max(1, num_jobs),
            prepare_workers=max(1, prepare_workers),
        )
        for key, value in prepare_result.items():
            if key not in {"chunk_specs", "input_files"}:
                print(f"🧬 [PREPARE] {key}: {value}", flush=True)

        chunk_specs = prepare_result.get("chunk_specs", [])
        if not isinstance(chunk_specs, list):
            raise RuntimeError(
                "`af3score_prepare` returned an invalid chunk spec list."
            )
        if any(not isinstance(spec, dict) for spec in chunk_specs):
            raise RuntimeError(
                f"`af3score_prepare` returned an invalid chunk spec list: {chunk_specs}"
            )
        required_chunk_keys = {"batch_name", "batch_json_dir", "batch_pdb_dir"}
        if any(not required_chunk_keys.issubset(spec) for spec in chunk_specs):
            raise RuntimeError(
                f"`af3score_prepare` returned incomplete chunk specs: {chunk_specs}"
            )

        total_chunks = len(chunk_specs)
        if total_chunks:
            print(
                "🧬 Running "
                f"{total_chunks} internal batches with "
                f"max_concurrent_gpus={max_concurrent_gpus}",
                flush=True,
            )

        for wave_start in range(0, total_chunks, max_concurrent_gpus):
            wave_specs = chunk_specs[wave_start : wave_start + max_concurrent_gpus]
            wave_index = (wave_start // max_concurrent_gpus) + 1
            total_waves = (
                total_chunks + max_concurrent_gpus - 1
            ) // max_concurrent_gpus
            print(
                "🧬 Launching wave "
                f"{wave_index}/{total_waves} with {len(wave_specs)} internal batches",
                flush=True,
            )
            batch_names: list[str] = []
            function_calls: list[modal.FunctionCall] = []
            for spec in wave_specs:
                batch_name = str(spec["batch_name"])
                function_call = af3score_run.spawn(
                    output_dir_name=output_dir_name,
                    batch_name=batch_name,
                    batch_json_dir=str(spec["batch_json_dir"]),
                    batch_pdb_dir=str(spec["batch_pdb_dir"]),
                )
                batch_names.append(batch_name)
                function_calls.append(function_call)

            wave_results = modal.FunctionCall.gather(*function_calls)
            for batch_name, result in zip(batch_names, wave_results, strict=True):
                print(f"🧬 [RESULT] internal_batch={batch_name} {result}", flush=True)
                print(f"🧬 [INFO] finished internal_batch={batch_name}", flush=True)

        postprocess_result = af3score_postprocess.remote(
            input_files=list(prepare_result.get("input_files", [])),
            output_dir_name=output_dir_name,
        )
        for key, value in postprocess_result.items():
            prefix = "[METRICS]" if str(key).startswith("metrics_") else "[POSTPROCESS]"
            print(f"🧬 {prefix} {key}: {value}", flush=True)

        total_processed = postprocess_result.get("metrics_rows")
        if isinstance(total_processed, int):
            print(f"🧬 [INFO] {total_processed}/{len(all_files)} done", flush=True)

        if bool(postprocess_result.get("metrics_csv_exists")):
            local_base = Path(output_dir).expanduser().resolve()
            local_base.mkdir(parents=True, exist_ok=True)
            if not os.access(local_base, os.W_OK):
                raise PermissionError(
                    f"Local output path is not writable: {local_base}"
                )

            metrics_csv_bytes = af3score_fetch_metrics_csv.remote(
                output_dir_name=output_dir_name
            )
            if metrics_csv_bytes is None:
                raise FileNotFoundError(
                    "AF3Score reported a metrics CSV, but the file could not be read."
                )

            local_metrics_csv = local_base / _local_metrics_filename(output_dir_name)
            local_metrics_csv.write_bytes(metrics_csv_bytes)
            print(f"🧬 Local metrics CSV: {local_metrics_csv}", flush=True)
        else:
            print("🧬 Local metrics CSV: not generated", flush=True)
    finally:
        af3score_manage_lock.remote(output_dir_name=output_dir_name, acquire=False)
