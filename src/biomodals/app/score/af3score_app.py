"""AF3Score source repo: <https://github.com/Mingchenchen/AF3Score>.

## Additional notes

- AF3Score scores existing protein structures rather than predicting new folds.
- Inputs can be a single `.pdb` file or a directory of `.pdb` files.
- The wrapper preserves AF3Score's internal length-based batching and schedules
  those internal batches in GPU waves when needed.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shutil
import string
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.app.score.af3score_execution import (
    COMPLETION_REQUIRED_FILES,
    COMPLETION_SAMPLE_SUBDIR,
    METRICS_FILENAME,
    AF3ScoreExecutionCoordinator,
    AF3ScoreExecutionRequest,
    ChunkSpec,
    TaskSpec,
    load_execution_request,
    stage_execution_request,
)
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
from biomodals.helper.app_run import (
    AppRunLayout,
    has_completed_output_files,
    volume_path_from_mount_path,
)
from biomodals.helper.shell import (
    copy_files,
    run_command,
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
    metrics_filename: str = METRICS_FILENAME
    completion_sample_subdir: str = COMPLETION_SAMPLE_SUBDIR
    completion_required_files: tuple[str, ...] = COMPLETION_REQUIRED_FILES


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
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
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(str(CONF.git_clone_dir), "biopython", "h5py", "pandas")
    .run_commands("build_data")
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
AF3SCORE_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_af3score_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_METRICS_PUBLICATION_SCHEMA_VERSION = 1


def _metrics_publication_path(run_root: str | Path) -> Path:
    return Path(run_root) / ".biomodals" / "af3score-metrics.json"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metrics_publication(
    run_root: str | Path,
    publication_key: str,
    metrics_path: Path,
) -> None:
    """Atomically bind the metrics artifact to one scientific request."""
    size = metrics_path.stat().st_size
    if size < 1:
        raise RuntimeError("AF3Score metrics publication is empty")
    marker = _metrics_publication_path(run_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f".{marker.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(
            orjson.dumps(
                {
                    "schema_version": _METRICS_PUBLICATION_SCHEMA_VERSION,
                    "publication_key": publication_key,
                    "metrics_filename": metrics_path.name,
                    "size": size,
                    "sha256": _file_sha256(metrics_path),
                },
                option=orjson.OPT_SORT_KEYS,
            )
        )
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics_publication_ready(
    run_root: str | Path,
    publication_key: str,
) -> bool:
    """Validate fingerprint-bound metrics without hiding unreadable state."""
    marker_path = _metrics_publication_path(run_root)
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    if not (
        isinstance(marker, dict)
        and marker.get("schema_version") == _METRICS_PUBLICATION_SCHEMA_VERSION
        and marker.get("publication_key") == publication_key
        and marker.get("metrics_filename") == APP_INFO.metrics_filename
        and isinstance(marker.get("size"), int)
        and not isinstance(marker.get("size"), bool)
        and marker["size"] > 0
        and isinstance(marker.get("sha256"), str)
    ):
        return False
    metrics = Path(run_root) / APP_INFO.metrics_filename
    try:
        return (
            not metrics.is_symlink()
            and metrics.stat().st_size == marker["size"]
            and _file_sha256(metrics) == marker["sha256"]
        )
    except (FileNotFoundError, NotADirectoryError):
        return False


##########################################
# Local input collection
##########################################
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
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    for f in all_files:
        safe_name = "".join(
            c for c in f.name.lower().replace(" ", "_") if c in allowed_chars
        )
        if not safe_name:
            raise ValueError(f"Input file name has no AF3Score-safe characters: {f}")
        symlink_path = stage_dir / safe_name
        if symlink_path.exists():
            raise ValueError(f"Duplicated sanitized file name: {symlink_path.name}")
        symlink_path.symlink_to(f)
        symlinks.append(symlink_path)
    return symlinks


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def af3score_prepare(
    run_name: str, input_files: list[str], num_jobs: int, prepare_workers: int
) -> TaskSpec:
    """Prepare AF3Score batches from staged inputs."""
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    staged_dir = layout.inputs_dir.resolve()
    if not staged_dir.exists():
        raise FileNotFoundError(f"Staged input directory not found: {staged_dir}")

    for path in (layout.outputs_dir, layout.failures_dir):
        path.mkdir(parents=True, exist_ok=True)

    all_files = [staged_dir / input_name for input_name in input_files]
    input_names = [path.name for path in all_files]
    total_files = len(all_files)
    print(f"💊 [PREP] Processing {total_files} files in '{layout.run_root}'")

    pending_files: list[Path] = []
    skipped = 0
    out_dir = layout.outputs_dir
    for pdb_file in all_files:
        if has_completed_output_files(
            out_dir,
            pdb_file.stem,
            sample_subdir=APP_INFO.completion_sample_subdir,
            required_files=APP_INFO.completion_required_files,
        ):
            skipped += 1
            continue
        pending_files.append(pdb_file)

    if not pending_files:
        return TaskSpec(
            total=total_files,
            pending=0,
            skipped=skipped,
            input_files=input_names,
            chunk_specs=[],
            output_dir=str(out_dir),
            failed_dir=str(layout.failures_dir),
        )

    prepare_root = layout.prep_dir
    pending_input_dir = prepare_root / "pending_inputs"
    batch_dir = prepare_root / "input_batch"
    if prepare_root.exists():
        shutil.rmtree(prepare_root)
    pending_input_dir.mkdir(parents=True, exist_ok=True)

    copy_files({
        source_path: pending_input_dir / source_path.name
        for source_path in pending_files
    })
    # Adjust CPU and GPU resources
    n_batches = min(max(1, num_jobs), len(pending_files))
    num_jobs_per_batch = max(1, (len(pending_files) + n_batches - 1) // n_batches)
    n_cpu = min(max(1, prepare_workers), num_jobs_per_batch)
    run_command([
        sys.executable,
        str(CONF.git_clone_dir / "01_prepare_get_json.py"),
        f"--input_dir={pending_input_dir}",
        f"--output_dir_cif={prepare_root / 'single_chain_cif'}",
        f"--save_csv={prepare_root / 'single_seq.csv'}",
        f"--output_dir_json={prepare_root / 'json'}",
        f"--batch_dir={batch_dir}",
        f"--num_jobs={n_batches}",
        f"--num_workers={n_cpu}",
    ])

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

    print(f"💊 [PREP] Inputs split into {len(chunk_specs)} batches")
    return TaskSpec(
        total=total_files,
        pending=len(pending_files),
        skipped=skipped,
        input_files=input_names,
        chunk_specs=chunk_specs,
        output_dir=str(layout.outputs_dir),
        failed_dir=str(layout.failures_dir),
    )


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(
        output_volume=True, model_volume=True, model_mount_subdir=False
    ),
)
def af3score_run(
    run_name: str, batch_name: str, batch_json_dir: str, batch_pdb_dir: str
) -> None:
    """Run one AF3Score batch."""
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    af3_weights = Path(CONF.model_volume_mountpoint) / APP_INFO.af3_weights
    if not af3_weights.exists():
        raise FileNotFoundError(f"AlphaFold3 model weights not found: {af3_weights}")

    with TemporaryDirectory(prefix=f"af3score_gpu_{batch_name}_") as temp_dir:
        batch_gpu_root = Path(temp_dir)
        batch_h5_dir = batch_gpu_root / "jax"
        batch_h5_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Benchmark whether AF3Score's JAX preprocessing can safely scale past one worker.
        jax_workers = 1
        print(f"💊 [RUN] Converting PDB to JAX arrays for batch '{batch_name}'")
        run_command([
            sys.executable,
            str(CONF.git_clone_dir / "02_prepare_pdb2jax.py"),
            f"--pdb_folder={batch_pdb_dir}",
            f"--output_folder={batch_h5_dir}",
            f"--num_workers={jax_workers}",
        ])

        # TODO: this or reuse AlphaFold3 buckets?
        bucket = batch_name.rsplit("_", 1)[-1]
        out_dir = layout.outputs_dir
        print(f"💊 [RUN] Starting AF3Score batch '{batch_name}'")
        run_command(
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
            output_mode="capture",
            log_file=out_dir / f"{batch_name}.log",
        )
        CONF.output_volume.commit()


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def af3score_postprocess(
    run_name: str,
    input_files: list[str],
    publication_key: str,
) -> dict[str, int | str]:
    """Validate records and collect metrics for all inputs."""
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    for path in (layout.outputs_dir, layout.failures_dir):
        path.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    completed_output_dirs: list[Path] = []
    out_dir = layout.outputs_dir
    for input_name in input_files:
        input_id = Path(input_name).stem
        failed_record = layout.failures_dir / f"{input_id}.err"
        if has_completed_output_files(
            out_dir,
            input_id,
            sample_subdir=APP_INFO.completion_sample_subdir,
            required_files=APP_INFO.completion_required_files,
        ):
            if failed_record.exists():
                failed_record.unlink()
            processed += 1
            completed_output_dirs.append(out_dir / input_id)
        else:
            failed_record.write_text(
                f"Missing AF3 output files for sample '{input_id}'"
            )
            failed += 1

    out_csv_path = layout.run_root / APP_INFO.metrics_filename
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
        run_command([
            sys.executable,
            str(CONF.git_clone_dir / "04_get_metrics.py"),
            f"--input_pdb_dir={layout.inputs_dir}",
            f"--af3score_output_dir={metrics_view_dir}",
            f"--save_metric_csv={out_csv_path}",
            f"--num_workers={max(1, min(16, len(completed_output_dirs)))}",
        ])

    with out_csv_path.open(encoding="utf-8") as f:
        metrics_rows = max(0, sum(1 for _ in f) - 1)

    _write_metrics_publication(layout.run_root, publication_key, out_csv_path)

    if layout.prep_dir.exists():
        shutil.rmtree(layout.prep_dir)
    CONF.output_volume.commit()
    return {
        "output_dir": str(out_dir),
        "failed_dir": str(layout.failures_dir),
        "total": len(input_files),
        "processed": processed,
        "failed": failed,
        "metrics_csv_exists": int(out_csv_path.exists()),
        "metrics_csv": str(out_csv_path),
        "metrics_rows": metrics_rows,
    }


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with AF3Score functions."""

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
    ) -> ExecutionSnapshot:
        """Create a compatible Successor while inferring predecessor identity."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                CONF.output_volume_mountpoint,
                UUID(self.execution_run_id),
            ),
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
    ) -> AF3ScoreExecutionCoordinator:
        adapter = getattr(self, "_coordinator_adapter", None)
        selected_mode = getattr(self, "_development", None)
        if adapter is not None:
            if development is not None and selected_mode != development:
                raise ValueError("Coordinator execution mode cannot change in place")
            return adapter
        execution_run_id, deployment = self._identity()
        selected_mode = False if development is None else development
        adapter = AF3ScoreExecutionCoordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=Path(CONF.output_volume_mountpoint),
            output_volume=CONF.output_volume,
            output_claims=AF3SCORE_OUTPUT_CLAIMS,
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
            "af3score_prepare": af3score_prepare,
            "af3score_run": af3score_run,
            "af3score_postprocess": af3score_postprocess,
        },
        workload_name="AF3Score",
    )


##########################################
# Local entrypoint client
##########################################
@app.local_entrypoint()
def submit_af3score_task(
    input_dir: str,
    run_name: str,
    output_dir: str | None = None,
    prepare_workers: int = 8,
    max_batches: int = 10,
    force: bool = True,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        use_deployed_coordinator: Target the exact deployed coordinator. The
            Biomodals CLI supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    input_root = Path(input_dir).expanduser().resolve()
    with TemporaryDirectory(prefix="af3score_stage_") as stage_tmp:
        all_files = _collect_input_files(input_root, Path(stage_tmp))
        num_files = len(all_files)
        print(f"🧬 Total files: {num_files} found in '{input_root}'")

        run_name = sanitize_filename(run_name)
        mount_root = Path(CONF.output_volume_mountpoint)
        layout = AppRunLayout.from_run_root(mount_root / run_name)
        metrics_csv = layout.run_root / APP_INFO.metrics_filename
        predecessor_execution_run_id = (
            None if restart_from is None else UUID(restart_from)
        )
        if predecessor_execution_run_id is None and not force:
            for item in CONF.output_volume.iterdir("/"):
                if item.path == run_name:
                    raise ValueError(
                        f"Run name '{run_name}' already exists in Modal volume."
                    )
        remote_run_dir = volume_path_from_mount_path(
            str(layout.run_root),
            CONF.output_volume_mountpoint,
            CONF.output_volume_name,
        )
        print(f"🧬 Uploading '{input_root}' to {remote_run_dir}")
        stage_root = layout.inputs_dir.relative_to(mount_root)
        if predecessor_execution_run_id is None:
            with CONF.output_volume.batch_upload(force=force) as batch:
                if num_files == 1:
                    path = all_files[0]
                    batch.put_file(path, f"/{stage_root}/{path.name}")
                else:
                    batch.put_directory(all_files[0].parent, f"/{stage_root}/")

        request = AF3ScoreExecutionRequest(
            run_name=run_name,
            inputs=tuple(
                (path.name, sha256(path.read_bytes()).hexdigest()) for path in all_files
            ),
            prepare_workers=prepare_workers,
            max_batches=max_batches,
            app_version=CONF.repo_commit_hash or CONF.version or "unknown",
        )
        execution_run_id = uuid4()
        deployment = DeploymentIdentity(
            deployment_environment,
            deployment_name,
            deployment_version,
        )
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
        postprocess_call = next(
            (
                provider_call
                for provider_call in snapshot.provider_calls
                if provider_call.node_key == "postprocess"
            ),
            None,
        )
        if postprocess_call is not None:
            result = postprocess_call.result_envelope.get("result", {})
            for key, value in result.items():
                prefix = (
                    "[METRICS]" if str(key).startswith("metrics_") else "[POSTPROCESS]"
                )
                print(f"🧬 {prefix} {key}: {value}")

        local_out_dir = (
            Path.cwd()
            if output_dir is None
            else Path(output_dir).expanduser().resolve()
        )
        local_out_dir.mkdir(parents=True, exist_ok=True)
        local_metrics_csv = local_out_dir / f"{run_name}_af3score_metrics.csv"
        print("🧬 Downloading metrics CSV...")
        with local_metrics_csv.open("wb") as stream:
            for chunk in CONF.output_volume.read_file(
                str(metrics_csv.relative_to(mount_root))
            ):
                stream.write(chunk)
        print(f"🧬 Local metrics CSV: {local_metrics_csv}")
