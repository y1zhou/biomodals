"""AF3Score source repo: <https://github.com/Mingchenchen/AF3Score>.

## Overview

- Modal wrapper around AF3Score for scoring existing protein structures.
- Input can be a single `.pdb` file or a directory of `.pdb` files.
- Uses AF3Score's internal length-based grouping and a GPU concurrency cap for large jobs.
- Preserves AF3Score-style output directories and generates the aggregate metrics CSV.
- Designed to be portable across servers: the image bootstraps AF3Score from a pinned remote repository instead of depending on a local checkout layout.
- The local entrypoint expects the `modal` CLI to be installed and available on `PATH` for the final metrics CSV download.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | **Required** | Path to a single `.pdb` file or a directory of `.pdb` files. |
| `--output-dir-name` | **Required** | Remote run directory name under the `af3score-outputs` Modal volume root. |
| `--output-dir` | **Required** | Local output directory for the final downloaded metrics CSV. |
| `--max-concurrent-gpus` | `10` | Maximum number of internal AF3Score batches to run at the same time. |
| `--num-jobs` | `10` | Target number of internal batches for `01_prepare_get_json.py`. |
| `--prepare-workers` | `8` | Worker count for `01_prepare_get_json.py`. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `AF3Score` | Name of the Modal app. |
| `GPU` | `L40S` | GPU type to request from Modal. |
| `TIMEOUT` | `86400` | Timeout for Modal functions in seconds. |

## Input Support

- Supported suffixes: `.pdb`.
- Directory input is collected with a stable sorted order before batching.
- Resume behavior is based on the existence of official AF3Score output files for each structure.

## Outputs

- Outputs are persisted in the Modal volume `af3score-outputs`, mounted at `/af3score-outputs`.
- Each run writes to `/af3score-outputs/<output_dir_name>`.
- Official AF3Score per-structure directories are written under `/af3score-outputs/<output_dir_name>/outputs`.
- Aggregate metrics are written to `/af3score-outputs/<output_dir_name>/af3score_metrics.csv`.
- A copy of the aggregate metrics CSV is also written to `/af3score-outputs/<output_dir_name>/outputs/af3score_metrics.csv`.
- The final `af3score_metrics.csv` file is downloaded locally to `--output-dir`.

## Batching Model

- `01_prepare_get_json.py` creates the only user-visible batching layer now.
- It groups samples heuristically by total complex length and writes subdirectories such as `batch_0_259`.
- Those internal batch directories are used as the GPU scheduling units for AF3Score inference.
- `--num-jobs` controls how many internal groups `01_prepare_get_json.py` attempts to create.
- `--max-concurrent-gpus` controls how many of those internal groups may run at once.
- If more internal batches are created than the GPU limit allows, the wrapper runs them in waves instead of launching all of them together.

## Internal API Note

- The remote Modal functions in this file are internal workflow steps for the local entrypoint only.
- They are not intended to be called manually as standalone public APIs.

## CLI

Single structure:

```bash
modal run af3score_app.py \
  --input-dir test.pdb \
  --output-dir-name af3score_run_single \
  --output-dir .
```

Batch directory:

```bash
modal run af3score_app.py \
  --input-dir test_pdbs \
  --output-dir-name af3score_run_batch \
  --output-dir . \
  --num-jobs 4 \
  --max-concurrent-gpus 4
```

"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from modal import App, FunctionCall, Image, Volume

##########################################
# Modal configs
##########################################
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", str(24 * 60 * 60)))

# AF3 model weights
AF3_MODEL_VOLUME = Volume.from_name("ppiflow-models")
AF3_MODEL_DIR = "/models"

OUTPUTS_VOLUME = Volume.from_name(
    "af3score-outputs",
    create_if_missing=True,
    version=2,
)
# Volume for outputs/inputs
OUTPUTS_DIR = Path("/af3score-outputs")
OUTPUT_ROOT = OUTPUTS_DIR

# Repositories and commit hashes
REPO_DIR = "/root/AF3Score"
REPO_ROOT = Path(REPO_DIR)
SUPPORTED_INPUT_SUFFIXES = {".pdb"}

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "ninja-build",
        "pkg-config",
        "zlib1g-dev",
    )
    .env(
        {
            "CC": "gcc",  # C compiler
            "CXX": "g++",  # C++ compiler
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=true",  # XLA GPU flags
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",  # Preallocate GPU memory
            "XLA_CLIENT_MEM_FRACTION": "0.95",  # GPU memory fraction
        }
    )
    .run_commands(
        f"git clone https://github.com/Mingchenchen/AF3Score.git {REPO_DIR} && cd {REPO_DIR} && git checkout b0764aaa4101f8a22a5f404faef7acc13ee52d06",
    )
    .uv_pip_install(REPO_DIR, "biopython", "h5py", "pandas")
    .run_commands(f"cd {REPO_DIR} && build_data")
    .workdir(REPO_DIR)
)

app = App(os.environ.get("MODAL_APP", "AF3Score"), image=runtime_image)

##########################################
# Helper functions
##########################################

def _run_paths(output_dir_name: str) -> dict[str, Path]:
    """Return the standard run-level paths for one AF3Score output directory."""
    run_root = OUTPUT_ROOT / output_dir_name
    output_dir = run_root / "outputs"
    return {
        "run_root": run_root,
        "lock_dir": run_root / ".run.lock",
        "work_root": run_root / "work",
        "output_dir": output_dir,
        "failed_dir": output_dir / "failed_records",
        "metrics_input_dir": run_root / "metric_inputs",
        "metrics_view_dir": run_root / "metrics_view",
        "metrics_csv": run_root / "af3score_metrics.csv",
    }

def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"[CMD] {' '.join(cmd)}", flush=True)
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("cwd", REPO_DIR)

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise RuntimeError(f"Command failed with return code {p.returncode}")


##########################################
# Inference functions
##########################################
#------------------------------------------
# Manage one exclusive run-level lock.
#------------------------------------------
@app.function(
    timeout=TIMEOUT,
    volumes={
        str(OUTPUTS_DIR): OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def af3score_manage_lock(output_dir_name: str = "", acquire: bool = True) -> None:
    """Internal-only remote helper for acquiring or releasing one run-level lock."""
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


#-------------------------------------------------------
# Prepare internal AF3Score batches from staged inputs.
#-------------------------------------------------------
@app.function(
    timeout=TIMEOUT,
    volumes={
        str(OUTPUTS_DIR): OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def af3score_prepare(
    staged_input_dir: str,
    input_files: list[str],
    output_dir_name: str = "",
    num_jobs: int = 10,
    prepare_workers: int = 8,
) -> dict[str, object]:
    """Internal-only remote step for preparing AF3Score batches from staged inputs."""
    OUTPUTS_VOLUME.reload()

    # 1. Resolve inputs and create run-level working directories.
    staged_dir = Path(staged_input_dir).resolve()

    paths = _run_paths(output_dir_name)
    run_root = paths["run_root"]

    # Output directories
    output_dir = paths["output_dir"]
    failed_dir = paths["failed_dir"]

    # Metrics directories
    metrics_input_dir = paths["metrics_input_dir"]

    # Prepare/work directories
    work_root = paths["work_root"]

    work_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(exist_ok=True)
    metrics_input_dir.mkdir(parents=True, exist_ok=True)
    all_files = [staged_dir / input_name for input_name in sorted(input_files, key=str.casefold)]
    input_names = [path.name for path in all_files]
    total_files = len(all_files)
    print(f"Preparing: Total files: {total_files}", flush=True)
    print(f"Preparing: Processing {total_files} files", flush=True)
    print(f"[INFO] Output root: {run_root}", flush=True)

    # 2. Prepare shared input PDBs for `01_prepare_get_json.py` and `04_get_metrics.py`, then determine pending inputs.
    pending_files: list[Path] = []
    skipped = 0
    for pdb in all_files:
        stem = pdb.stem.casefold()
        if not (metrics_input_dir / f"{stem}.pdb").exists():
            shutil.copy2(pdb, metrics_input_dir / f"{stem}.pdb")

        if (
            (output_dir / stem / "seed-10_sample-0" / "summary_confidences.json").exists()
            and (output_dir / stem / "seed-10_sample-0" / "confidences.json").exists()
        ):
            print(f"[SKIP] {pdb.name}", flush=True)
            skipped += 1
            continue
        print(f"[BATCH] Pending: Processing {pdb.name}", flush=True)
        pending_files.append(pdb)

    if not pending_files:
        return {
            "total": total_files,
            "pending": 0,
            "skipped": skipped,
            "input_files": input_names,
            "chunk_specs": [],
            "output_dir": str(output_dir),
            "failed_dir": str(failed_dir),
        }
    # 3. Stage only still-pending inputs for AF3Score preprocessing.
    prepare_root = work_root / "prepare"
    pending_input_dir = prepare_root / "pending_inputs"
    batch_dir = prepare_root / "input_batch"  # `01_prepare_get_json.py` batch root.
    if prepare_root.exists():
        shutil.rmtree(prepare_root)
    prepare_root.mkdir(parents=True, exist_ok=True)
    pending_input_dir.mkdir(parents=True, exist_ok=True)
    for source_path in pending_files:
        shutil.copy2(metrics_input_dir / f"{source_path.stem.casefold()}.pdb", pending_input_dir / f"{source_path.stem}.pdb")

    # 4. Run `01_prepare_get_json.py` to generate internal AF3Score batches.
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "01_prepare_get_json.py"),
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
        ],
    )

    # 5. Collect internal batch specs for downstream GPU workers.
    chunk_specs: list[dict[str, object]] = []
    for batch_json_dir in sorted(path for path in (batch_dir / "json").iterdir() if path.is_dir()):
        chunk_specs.append(
            {
                "batch_name": batch_json_dir.name,  # Generated by `01_prepare_get_json.py` as `batch_{index}_{max_length}`.
                "batch_json_dir": str(batch_json_dir),  # For `run_af3score.py`.
                "batch_pdb_dir": str((batch_dir / "pdb") / batch_json_dir.name),  # For `02_prepare_pdb2jax.py`.
            }
        )

    print(f"[INFO] Prepared {len(chunk_specs)} internal batches", flush=True)
    return {
        "total": total_files,
        "pending": len(pending_files),
        "skipped": skipped,
        "input_files": input_names,
        "chunk_specs": chunk_specs,
        "output_dir": str(output_dir),
        "failed_dir": str(failed_dir),
    }

#--------------------------------------------
# Run one internal AF3Score batch on one GPU.
#--------------------------------------------
@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={
        str(OUTPUTS_DIR): OUTPUTS_VOLUME,
        AF3_MODEL_DIR: AF3_MODEL_VOLUME.read_only(),
    },
    image=runtime_image,
)
def af3score_run(
    output_dir_name: str = "",
    batch_name: str = "",
    batch_json_dir: str = "",
    batch_pdb_dir: str = "",
) -> dict[str, int | str]:
    """Internal-only remote step for running one AF3Score batch on one GPU."""
    OUTPUTS_VOLUME.reload()
    model_weights_path = Path(AF3_MODEL_DIR) / "af3.bin"

    # 1. Prepare a temporary GPU work directory for this batch.
    paths = _run_paths(output_dir_name)
    batch_gpu_root = paths["run_root"] / "gpu_batch" / batch_name
    if batch_gpu_root.exists():
        shutil.rmtree(batch_gpu_root)
    batch_gpu_root.mkdir(parents=True, exist_ok=True)

    try:
        jax_workers = 1  # Fixed at 1 for JAX prep stability.
        batch_pdb_path = Path(batch_pdb_dir)

        # 2. Run `02_prepare_pdb2jax.py` to convert this batch's PDB inputs into JAX/H5 features.
        batch_h5_dir = batch_gpu_root / "jax"
        batch_h5_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "02_prepare_pdb2jax.py"),
                "--pdb_folder",
                str(batch_pdb_path),
                "--output_folder",
                str(batch_h5_dir),
                "--num_workers",
                str(jax_workers),
            ],
        )

        # 3. Get the bucket suffix from `01_prepare_get_json.py` batch name for `run_af3score.py`.
        bucket = batch_name.rsplit("_", 1)[-1]
        batch_json_path = Path(batch_json_dir)

        # 4. Run `run_af3score.py` for this batch.
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "run_af3score.py"),
                "--model_dir",
                str(model_weights_path.parent),
                "--batch_json_dir",
                str(batch_json_path),
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
            ],
        )

        return {
            "batch_name": batch_name,
            "output_dir": str(paths["output_dir"]),
        }
    finally:
        # 5. Clean up the temporary GPU work directory.
        if batch_gpu_root.exists():
            shutil.rmtree(batch_gpu_root)

#---------------------------------------------------------------
# Validate outputs and write the final aggregate AF3Score metrics.
#---------------------------------------------------------------
@app.function(
    timeout=TIMEOUT,
    volumes={
        str(OUTPUTS_DIR): OUTPUTS_VOLUME,
    },
    image=runtime_image,
)
def af3score_postprocess(
    input_files: list[str],
    output_dir_name: str = "",
) -> dict[str, int | str]:
    """Internal-only remote step for validation, failure records, and metrics generation."""
    OUTPUTS_VOLUME.reload()  # Refresh the outputs volume view.
    all_files = sorted(input_files, key=str.casefold)  # Reuse the prepared input file list.
    paths = _run_paths(output_dir_name)  # Get standard run-level paths.
    for path in (paths["output_dir"], paths["failed_dir"], paths["metrics_input_dir"]):
        path.mkdir(parents=True, exist_ok=True)  # Ensure required postprocess dirs exist.

    # 1. Validate per-structure AF3Score outputs and write failure records.
    processed = 0
    failed = 0
    for input_name in all_files:
        stem = Path(input_name).stem
        sample_dir = paths["output_dir"] / stem.casefold() / "seed-10_sample-0"
        summary_confidences = sample_dir / "summary_confidences.json"
        confidences = sample_dir / "confidences.json"
        if summary_confidences.exists() and confidences.exists():
            if (paths["failed_dir"] / f"{stem}.err").exists():
                (paths["failed_dir"] / f"{stem}.err").unlink()
            processed += 1
            continue

        (paths["failed_dir"] / f"{stem}.err").write_text(
            f"Missing AF3 output files: {sample_dir}",
            encoding="utf-8",
        )
        failed += 1

    # 2. Build the metrics view from completed output directories.
    if paths["metrics_view_dir"].exists():
        shutil.rmtree(paths["metrics_view_dir"])
    paths["metrics_view_dir"].mkdir(parents=True, exist_ok=True)

    has_completed_outputs = False
    for candidate in sorted(paths["output_dir"].iterdir(), key=lambda path: path.name.casefold()):
        if not candidate.is_dir():
            continue
        sample_dir = candidate / "seed-10_sample-0"
        summary_confidences = sample_dir / "summary_confidences.json"
        confidences = sample_dir / "confidences.json"
        if not (summary_confidences.exists() and confidences.exists()):
            continue
        (paths["metrics_view_dir"] / candidate.name).symlink_to(candidate, target_is_directory=True)
        has_completed_outputs = True

    metrics_rows = 0
    if has_completed_outputs:
        # 3. Run `04_get_metrics.py` to generate the aggregate metrics CSV.
        temp_metrics_csv = paths["metrics_csv"].with_suffix(".tmp")
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "04_get_metrics.py"),
                "--input_pdb_dir",
                str(paths["metrics_input_dir"]),
                "--af3score_output_dir",
                str(paths["metrics_view_dir"]),
                "--save_metric_csv",
                str(temp_metrics_csv),
                "--num_workers",
                str(max(1, min(16, os.cpu_count() or 4))),
            ],
        )

        # 4. Save the final metrics CSV under the run root and outputs dir.
        temp_metrics_csv.replace(paths["metrics_csv"])
        shutil.copy2(paths["metrics_csv"], paths["output_dir"] / "af3score_metrics.csv")

        with paths["metrics_csv"].open(encoding="utf-8") as handle:
            metrics_rows = max(0, sum(1 for _ in handle) - 1)

    # 5. Clean up the work directory and return the postprocess summary.
    if (paths["run_root"] / "work").exists():
        shutil.rmtree(paths["run_root"] / "work")
    OUTPUTS_VOLUME.commit()
    return {
        "output_dir": str(paths["output_dir"]),
        "failed_dir": str(paths["failed_dir"]),
        "total": len(all_files),
        "processed": processed,
        "failed": failed,
        "metrics_csv_exists": int(paths["metrics_csv"].exists()),
        "metrics_csv": str(paths["metrics_csv"]),
        "metrics_csv_in_output_dir": str(paths["output_dir"] / "af3score_metrics.csv"),
        "metrics_rows": metrics_rows,
    }

##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_af3score_task(
    input_dir: str = "",  # Required.
    output_dir_name: str = "",  # Required remote folder name.
    output_dir: str = "",  # Local metrics CSV dir.
    num_jobs: int = 10,  # Target internal batches.
    prepare_workers: int = 8,  # `01_prepare_get_json.py` workers.
    max_concurrent_gpus: int = 10,  # Max GPU batches at once.
) -> None:
    """Stage local `.pdb` inputs, run AF3Score on Modal, and download the final metrics CSV."""
    # 1. Check CLI arguments.
    output_dir_name = output_dir_name.strip().strip("/")
    if not output_dir_name:
        raise ValueError("`--output-dir-name` is required.")
    if not output_dir:
        raise ValueError("`--output-dir` is required.")
    if max_concurrent_gpus < 1:
        raise ValueError("`--max-concurrent-gpus` must be >= 1.")
    if not input_dir:
        raise ValueError("`--input-dir` is required.")
    if shutil.which("modal") is None:
        raise RuntimeError(
            "The local `modal` CLI must be installed and available on PATH to download "
            "the final AF3Score metrics CSV."
        )

    # 2. Check local input files.
    input_root = Path(input_dir).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    if input_root.is_file():
        all_files = [input_root] if input_root.suffix.lower() in SUPPORTED_INPUT_SUFFIXES else []
    else:
        all_files = [
            path
            for path in input_root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        ]

    if not all_files:
        raise ValueError(
            "No supported structure files were found in the provided input path. "
            f"Supported suffixes: {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}"
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
    all_files = sorted(unique.values(), key=lambda path: path.name.casefold())

    print(f"[INFO] Total files: {len(all_files)}", flush=True)
    print(f"[INFO] Output root: {OUTPUT_ROOT / output_dir_name}", flush=True)

    af3score_manage_lock.remote(output_dir_name=output_dir_name, acquire=True)
    try:
        # 3. Upload inputs to the Modal volume.
        dataset_hash = hashlib.sha256(
            str(input_root.resolve()).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]
        stage_root = Path("af3score_inputs") / dataset_hash / "inputs"
        remote_stage_root = OUTPUT_ROOT / stage_root

        with OUTPUTS_VOLUME.batch_upload(force=True) as batch:
            print(f"[INFO] Processing {len(all_files)} files", flush=True)
            for pdb_path in all_files:
                batch.put_file(str(pdb_path), str(stage_root / pdb_path.name))

        # 4. Run remote `af3score_prepare(...)`.
        prepare_result = af3score_prepare.remote(
            staged_input_dir=str(remote_stage_root),
            input_files=[path.name for path in all_files],
            output_dir_name=output_dir_name,
            num_jobs=max(1, num_jobs),
            prepare_workers=max(1, prepare_workers),
        )
        for key, value in prepare_result.items():
            if key not in {"chunk_specs", "input_files"}:
                print(f"[PREPARE] {key}: {value}", flush=True)

        chunk_specs = prepare_result.get("chunk_specs", [])
        if not isinstance(chunk_specs, list):
            raise RuntimeError("`af3score_prepare` returned an invalid chunk spec list.")
        if any(not isinstance(spec, dict) for spec in chunk_specs):
            raise RuntimeError(f"`af3score_prepare` returned an invalid chunk spec list: {chunk_specs}")

        total_chunks = len(chunk_specs)
        if total_chunks:
            print(
                f"[INFO] Running {total_chunks} internal batches with max_concurrent_gpus={max_concurrent_gpus}",
                flush=True,
            )

        # 5. Run remote `af3score_run(...)` in waves.
        for wave_start in range(0, total_chunks, max_concurrent_gpus):
            wave_specs = chunk_specs[wave_start : wave_start + max_concurrent_gpus]
            wave_index = (wave_start // max_concurrent_gpus) + 1
            total_waves = (total_chunks + max_concurrent_gpus - 1) // max_concurrent_gpus
            print(
                f"[INFO] Launching wave {wave_index}/{total_waves} with {len(wave_specs)} internal batches",
                flush=True,
            )
            batch_names: list[str] = []
            function_calls: list[FunctionCall] = []
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

            wave_results = FunctionCall.gather(*function_calls)
            for batch_name, result in zip(batch_names, wave_results, strict=True):
                print(f"[RESULT] internal_batch={batch_name} {result}", flush=True)
                print(f"[INFO] finished internal_batch={batch_name}", flush=True)

        # 6. Run remote `af3score_postprocess(...)`.
        postprocess_result = af3score_postprocess.remote(
            input_files=list(prepare_result.get("input_files", [])),
            output_dir_name=output_dir_name,
        )
        for key, value in postprocess_result.items():
            prefix = "[METRICS]" if str(key).startswith("metrics_") else "[POSTPROCESS]"
            print(f"{prefix} {key}: {value}", flush=True)

        total_processed = postprocess_result.get("metrics_rows")
        if isinstance(total_processed, int):
            print(f"[INFO] {total_processed}/{len(all_files)} done", flush=True)

        # 7. Download the final metrics CSV to local output when it exists.
        if bool(postprocess_result.get("metrics_csv_exists")):
            local_base = Path(output_dir).expanduser().resolve()
            local_base.mkdir(parents=True, exist_ok=True)
            if not os.access(local_base, os.W_OK):
                raise PermissionError(f"Local output path is not writable: {local_base}")

            local_metrics_csv = local_base / f"{output_dir_name}_af3score_metrics.csv"
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    "af3score-outputs",
                    f"{output_dir_name}/af3score_metrics.csv",
                    str(local_metrics_csv),
                ],
                check=True,
            )

            print(f"[LOCAL] metrics_csv: {local_metrics_csv}", flush=True)
        else:
            print("[LOCAL] metrics_csv: not generated", flush=True)
    finally:
        af3score_manage_lock.remote(output_dir_name=output_dir_name, acquire=False)
