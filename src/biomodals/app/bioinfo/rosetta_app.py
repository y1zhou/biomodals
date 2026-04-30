"""Rosetta source repo: <https://github.com/RosettaCommons/rosetta>.

Use for commercial purposes requires purchase of a separate license.
Please see <https://els2.comotion.uw.edu/product/rosetta> or email
license@uw.edu for more information.

See <https://docs.rosettacommons.org/docs/latest/Home> for documentation.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import modal
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.shell import package_outputs, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Rosetta",
    repo_url="https://github.com/RosettaCommons/rosetta",
    package_name="rosetta",
    version="2025.51+release.612b6ef9e9",  # 2025-12-19 release
    python_version="3.12",
    timeout=int(os.environ.get("TIMEOUT", "14400")),
)
OUT_VOLUME = CONF.get_out_volume()
ROSETTA_DIR = Path(__file__).parent / "rosetta"


@dataclass(frozen=True)
class _RosettaCommandJob:
    index: str
    binary: str
    pdb_path: Path
    out_dir: Path
    rosetta_script_path: Path | None = None
    flags_path: Path | None = None


def _build_rosetta_command(
    *,
    binary: str,
    pdb_path: Path,
    out_dir: Path,
    rosetta_script_path: Path | None = None,
    flags_path: Path | None = None,
) -> list[str]:
    cmd = [binary]
    if rosetta_script_path is not None:
        cmd.extend(["-parser:protocol", str(rosetta_script_path)])
    if flags_path is not None:
        cmd.append(f"@{flags_path}")
    cmd.extend(["-s", str(pdb_path), "-out:path:all", str(out_dir)])
    return cmd


def _command_for_rosetta_job(job: _RosettaCommandJob) -> list[str]:
    return _build_rosetta_command(
        binary=job.binary,
        pdb_path=job.pdb_path,
        out_dir=job.out_dir,
        rosetta_script_path=job.rosetta_script_path,
        flags_path=job.flags_path,
    )


def _required_rosetta_job_value(job_spec: dict[str, object], key: str) -> object:
    value = job_spec[key]
    if value is None:
        raise ValueError(f"Rosetta job is missing {key!r}")
    return value


def _optional_mounted_path(mount_dir: Path, path: object) -> Path | None:
    if path is None:
        return None
    return mount_dir / str(path)


def _normalize_volume_rosetta_job(
    job_spec: dict[str, object], *, mount_dir: Path, workdir: Path
) -> _RosettaCommandJob:
    task_idx = str(_required_rosetta_job_value(job_spec, "index"))
    return _RosettaCommandJob(
        index=task_idx,
        binary=str(_required_rosetta_job_value(job_spec, "binary")),
        pdb_path=mount_dir / str(_required_rosetta_job_value(job_spec, "pdb")),
        out_dir=workdir / task_idx,
        rosetta_script_path=_optional_mounted_path(
            mount_dir, job_spec.get("rosetta_script")
        ),
        flags_path=_optional_mounted_path(mount_dir, job_spec.get("flags_file")),
    )


def _write_optional_rosetta_text_file(
    job_dir: Path, job_spec: dict[str, object], key: str
) -> Path | None:
    text = job_spec.get(f"{key}_text")
    if text is None:
        return None
    name = sanitize_filename(str(job_spec.get(f"{key}_name") or f"{key}.txt"))
    path = job_dir / name
    path.write_text(str(text))
    return path


def _normalize_bytes_rosetta_job(
    job_spec: dict[str, object], *, inputs_dir: Path, outputs_dir: Path
) -> _RosettaCommandJob:
    idx = sanitize_filename(str(job_spec.get("index") or len(str(job_spec))))
    job_dir = inputs_dir / idx
    out_dir = outputs_dir / idx
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_bytes = job_spec.get("pdb_bytes")
    if not isinstance(pdb_bytes, bytes):
        raise TypeError(f"Rosetta job {idx!r} is missing pdb_bytes")
    pdb_name = sanitize_filename(str(job_spec.get("pdb_name") or f"{idx}.pdb"))
    pdb_path = job_dir / pdb_name
    pdb_path.write_bytes(pdb_bytes)

    return _RosettaCommandJob(
        index=idx,
        binary=str(job_spec.get("binary") or "relax"),
        pdb_path=pdb_path,
        out_dir=out_dir,
        rosetta_script_path=_write_optional_rosetta_text_file(
            job_dir, job_spec, "rosetta_script"
        ),
        flags_path=_write_optional_rosetta_text_file(job_dir, job_spec, "flags"),
    )


##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    modal.Image.from_registry(
        "rosettacommons/rosetta:serial-420", add_python=CONF.python_version
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 30.125),  # Each pod can run 1-30 jobs
    memory=(1024, 43008),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def run_rosetta(run_name: str, run_id: str, num_cpu_per_pod: int):
    """Run Rosetta scripts."""
    from biomodals.helper.shell import run_command_with_log

    mount_dir = Path(CONF.output_volume_mountpoint)
    workdir = mount_dir / f"{run_name}-{run_id}"
    queue = modal.Queue.from_name(f"{CONF.name}-queue-{run_id}")

    def _worker(worker_idx: int) -> None:
        while True:
            job_spec = queue.get(block=False)
            if job_spec is None:
                print(f"💊 No more jobs in queue for worker {worker_idx}")
                return

            job = _normalize_volume_rosetta_job(
                job_spec,
                mount_dir=mount_dir,
                workdir=workdir,
            )
            run_command_with_log(
                _command_for_rosetta_job(job), log_file=job.out_dir / "rosetta.log"
            )
            OUT_VOLUME.commit()

    # Run workers in parallel within the pod
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=num_cpu_per_pod) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_cpu_per_pod)]
        for future in futures:
            future.result()  # wait for all workers to finish


@app.function(
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def package_outputs_helper(
    root: str | Path,
    paths_to_bundle: Iterable[str | Path] | None = None,
    tar_args: list[str] | None = None,
    num_threads: int = 16,
) -> bytes:
    """Modal runner to package directories into a tar.zst archive and return as bytes."""
    return package_outputs(
        root,
        paths_to_bundle=paths_to_bundle,
        tar_args=tar_args,
        num_threads=num_threads,
    )


@app.function(
    cpu=(0.125, 30.125),
    memory=(1024, 43008),
    timeout=CONF.timeout,
)
def run_rosetta_batch_bytes(
    jobs: list[dict[str, object]],
    run_name: str,
    num_cpu_per_pod: int = 1,
) -> bytes:
    """Run Rosetta jobs from in-memory inputs and return packaged outputs."""
    from concurrent.futures import ThreadPoolExecutor

    from biomodals.helper.shell import run_command_with_log

    if not jobs:
        raise ValueError("At least one Rosetta job is required")

    safe_run_name = sanitize_filename(run_name)
    with TemporaryDirectory(prefix=f"rosetta_{safe_run_name}_") as tmpdir:
        workdir = Path(tmpdir) / safe_run_name
        inputs_dir = workdir / "inputs"
        outputs_dir = workdir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        def _worker(job_spec: dict[str, object]) -> None:
            job = _normalize_bytes_rosetta_job(
                job_spec,
                inputs_dir=inputs_dir,
                outputs_dir=outputs_dir,
            )

            try:
                run_command_with_log(
                    _command_for_rosetta_job(job), log_file=job.out_dir / "rosetta.log"
                )
            except subprocess.CalledProcessError:
                (job.out_dir / "FAILED").write_text("Rosetta command failed\n")
                raise

        max_workers = max(1, min(num_cpu_per_pod, len(jobs), 30))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, job) for job in jobs]
            for future in futures:
                future.result()

        return package_outputs(workdir)


##########################################
# Entrypoint for ephemeral usage
##########################################
def _prepare_input_csv(
    rosetta_binary: str = "rosetta_scripts",
    input_pdb: str | None = None,
    input_rosetta_script: str | None = None,
    input_flags_file: str | None = None,
    input_csv: str | None = None,
    rosetta_search_path: Path = ROSETTA_DIR,
) -> pl.DataFrame:
    """Make a standardized input CSV for Rosetta runs.

    The CSV will have columns:

    * index: a unique one-based index for each row
    * binary: the Rosetta binary to use for this run
    * pdb: local file path to the input PDB file for this run
    * rosetta_script: local file path to the input Rosetta script for this run
        if the `binary` column is `rosetta_scripts`, otherwise can be None
    * flags_file: local file path to the input Rosetta flags file for this run
        if additional flags are needed, otherwise can be None
    * script_hash: a hash of the Rosetta script file (content-based)
    * flags_hash: a hash of the flags file (content-based)
    """
    cols = ["binary", "pdb", "rosetta_script", "flags_file"]
    if input_csv is not None:
        input_csv_path = Path(input_csv).expanduser().resolve()
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

        rel_root_dir = input_csv_path.parent

        df = pl.read_csv(input_csv_path)
        all_cols = df.columns
        if "pdb" not in all_cols:
            raise ValueError(
                "Input CSV file must have a 'pdb' column with paths to input PDB files"
            )
        if "binary" not in all_cols:
            df = df.with_columns(pl.lit(rosetta_binary).alias("binary"))

        for optional_col in ("rosetta_script", "flags_file"):
            if optional_col not in all_cols:
                df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(optional_col))

    else:
        if input_pdb is None:
            raise ValueError(
                "'input_pdb' needs to be provided if 'input_csv' is not provided"
            )
        rel_root_dir = Path.cwd()
        df = pl.DataFrame({
            "binary": [rosetta_binary],
            "pdb": [input_pdb],
            "rosetta_script": [input_rosetta_script],
            "flags_file": [input_flags_file],
        })

    # Check for missing values in required columns
    df = df.select(pl.col(c).cast(pl.Utf8) for c in cols)
    df_missing_script = df.filter(
        (pl.col("binary") == pl.lit("rosetta_scripts"))
        & pl.col("rosetta_script").is_null()
    )
    if df_missing_script.height > 0:
        raise ValueError(f"Missing 'rosetta_script':\n{df_missing_script}")

    def _localize_input_path(
        path_str: str, *, col_name: str, allow_search_path: bool = False
    ) -> Path:
        local_path = Path(path_str).expanduser()
        # Absolute path, or relative to $PWD
        if local_path.exists():
            return local_path

        if (abs_path := (rel_root_dir / local_path)).exists():
            return abs_path
        if allow_search_path:
            search_path = rosetta_search_path / local_path
            if search_path.exists():
                return search_path
        raise FileNotFoundError(f"'{col_name}' file not found locally: {local_path}")

    df_pdbs = (
        df
        .select("pdb")
        .unique()
        .with_columns(
            pl
            .col("pdb")
            .map_elements(
                lambda p: str(_localize_input_path(p, col_name="pdb")),
                return_dtype=pl.Utf8,
            )
            .alias("pdb_path")
        )
    )

    # Get hashes for script and flags files to identify unique files for upload
    def _get_file_hashes(
        col_name: str, hash_col_name: str, real_path_col_name: str
    ) -> pl.DataFrame:
        df_files = df.filter(pl.col(col_name).is_not_null()).select(col_name).unique()
        file_abs_paths: list[str] = []
        file_hashes: list[str] = []
        for f in df_files.get_column(col_name):
            local_path = _localize_input_path(
                f,
                col_name=col_name,
                allow_search_path=True,
            )
            file_abs_paths.append(str(local_path))
            file_hashes.append(hash_string(local_path.read_text()))
        return df_files.with_columns(
            pl.Series(hash_col_name, file_hashes),
            pl.Series(real_path_col_name, file_abs_paths),
        )

    df_scripts = _get_file_hashes("rosetta_script", "script_hash", "script_path")
    df_flags = _get_file_hashes("flags_file", "flags_hash", "flags_path")

    return (
        df
        .join(df_pdbs, on="pdb", how="left", maintain_order="left")
        .join(df_scripts, on="rosetta_script", how="left", maintain_order="left")
        .join(df_flags, on="flags_file", how="left", maintain_order="left")
        .with_columns(
            pl.col("pdb_path").alias("pdb"),
            pl.col("script_path").alias("rosetta_script"),
            pl.col("flags_path").alias("flags_file"),
        )
        .drop("pdb_path", "script_path", "flags_path")
        .with_row_index(name="index", offset=1)
    )


@app.local_entrypoint()
def submit_rosetta_task(
    rosetta_binary: str = "rosetta_scripts",  # .default.linuxgccrelease
    input_pdb: str | None = None,
    input_rosetta_script: str | None = None,
    input_flags_file: str | None = None,
    input_csv: str | None = None,
    out_dir: str | None = None,
    max_num_pods: int = 1,
    rosetta_search_path: str = str(ROSETTA_DIR),
) -> None:
    """Run Rosetta scripts on Modal and fetch results to `out_dir`.

    Args:
        rosetta_binary: Path to the Rosetta binary to use.
        input_pdb: Path to input PDB file. Needs to be provided unless
            `input_csv` is specified.
        input_rosetta_script: Path to input Rosetta script file. Needs to be
            provided together with `input_pdb` if `rosetta_binary` is `rosetta_scripts`.
            Can be omitted if `rosetta_binary` is some other application such as `relax`.
            Can be a filename in `biomodals/app/bioinfo/rosetta/`.
        input_flags_file: Path to input Rosetta flag file, if additional flags
            are needed. Can be a filename in `biomodals/app/bioinfo/rosetta/`.
            Please do not include file paths in the flags, as it is difficult
            to identify them and upload files to Modal. For file specs, see
            <https://docs.rosettacommons.org/docs/latest/development_documentation/code_structure/namespaces/namespace-utility-options#flagsfile>.
        input_csv: Path to an input CSV file, if `input_pdb`, `input_rosetta_script`,
            and `input_flags_file` are not provided. The CSV file should have columns
            "pdb", "rosetta_script", and optionally "flags_file" that specify the
            *local* file paths to the respective files for each run. If there is
            a `rosetta_binary` column, the binary will be used; otherwise the
            binary specified by `rosetta_binary` will be used for all rows.
            This argument takes precedence over the individual `input_*` arguments.
            This allows batch processing of multiple Rosetta runs with one input.
            If the `pdb` column contains relative paths, they will be resolved
            relative to `$PWD` as well as the parent directory of the CSV file.
            For the `rosetta_script` and `flags_file` columns, they will be resolved
            relative to `$PWD`, CSV parent, and `rosetta_search_path`.
        out_dir: Optional output directory. If not provided, results will only
            be saved to the Modal output volume and not downloaded locally. If
            provided, results will be saved to `out_dir` with the same filename as
            the input PDB file but with a `.tar.zst` extension.
        max_num_pods: Maximum number of parallel pods to run. Only applicable when
            `input_csv` is provided, because otherwise there's no point to spawn
            multiple pods. Default is 1. Note that a maximum of 30 CPUs can be
            allocated per pod. Also note that the parallelism is achieved by running
            multiple Rosetta jobs, not by parallelizing a single Rosetta job, so
            more threads for a single job will not speed up the runtime.
        rosetta_search_path: The additional search path for Rosetta to find
            Rosetta scripts and flags files.
    """
    # Validate and read input
    run_id = uuid4().hex
    if input_csv is not None:
        run_name = Path(input_csv).stem
    elif input_pdb is not None:
        run_name = Path(input_pdb).stem
    else:
        raise ValueError("Either 'input_csv' or 'input_pdb' must be provided")

    tasks_df = _prepare_input_csv(
        rosetta_binary,
        input_pdb,
        input_rosetta_script,
        input_flags_file,
        input_csv,
        Path(rosetta_search_path),
    )
    if tasks_df.height == 0:
        raise ValueError("No valid tasks found in the input CSV")

    print(f"🧬 Preparing queue for {run_name} tasks...")
    queue = modal.Queue.from_name(f"{CONF.name}-queue-{run_id}", create_if_missing=True)
    uploaded_files = set()
    with OUT_VOLUME.batch_upload() as batch:
        for r in tasks_df.iter_rows(named=True):
            # Structure file should always be present
            local_pdb = Path(r["pdb"]).expanduser().resolve()
            remote_pdb = f"{run_name}-{run_id}/{r['index']}/{local_pdb.name}"
            batch.put_file(local_pdb, f"/{remote_pdb}")

            # Other files may or may not be present, depending on the input CSV
            remote_script, remote_flags = None, None
            if r["rosetta_script"] is not None:
                local_script = Path(r["rosetta_script"]).expanduser().resolve()
                r_script_hash = r["script_hash"]
                remote_script = f"{run_name}-{run_id}/_script/{r_script_hash}.xml"
                if remote_script not in uploaded_files:
                    batch.put_file(local_script, f"/{remote_script}")
                    uploaded_files.add(remote_script)
            if r["flags_file"] is not None:
                local_flags = Path(r["flags_file"]).expanduser().resolve()
                r_flags_hash = r["flags_hash"]
                remote_flags = f"{run_name}-{run_id}/_flags/{r_flags_hash}.flags"
                if remote_flags not in uploaded_files:
                    batch.put_file(local_flags, f"/{remote_flags}")
                    uploaded_files.add(remote_flags)

            queue.put({
                "index": r["index"],
                "binary": r["binary"],
                "pdb": remote_pdb,
                "rosetta_script": remote_script,
                "flags_file": remote_flags,
            })
        buffer = BytesIO()
        tasks_df.write_parquet(buffer)
        batch.put_file(buffer, f"/{run_name}-{run_id}/tasks.parquet")

    # Tune numbers based on total number of tasks
    num_cpu_per_pod = min(30, max(1, tasks_df.height))
    max_num_pods = min(
        max_num_pods, (tasks_df.height + num_cpu_per_pod - 1) // num_cpu_per_pod
    )
    max_num_pods = max(1, max_num_pods)  # ensure at least 1 pod

    print(
        f"🧬 Running task {run_name}-{run_id} in {max_num_pods} {num_cpu_per_pod}-CPU pods..."
    )
    rosetta_tasks = [
        run_rosetta.spawn(run_name, run_id, num_cpu_per_pod)
        for _ in range(max_num_pods)
    ]
    _ = modal.FunctionCall.gather(*rosetta_tasks)

    modal.Queue.objects.delete(f"{CONF.name}-queue-{run_id}")

    # Save results locally
    out_vol_name = OUT_VOLUME.name or f"{CONF.name}-outputs"
    remote_data_dir = f"/{run_name}-{run_id}"

    if out_dir is None:
        print(
            f"🧬 {CONF.name} run complete!\n"
            f"Results saved to Modal volume '{out_vol_name}' at '{remote_data_dir}'"
        )
        return

    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = local_out_dir / f"{run_name}-{run_id}.tar.zst"
    tarball_bytes = package_outputs_helper.remote(
        root=f"{CONF.output_volume_mountpoint}/{run_name}-{run_id}",
    )
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
