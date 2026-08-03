"""Rosetta source repo: <https://github.com/RosettaCommons/rosetta>.

Use for commercial purposes requires purchase of a separate license.
Please see <https://els2.comotion.uw.edu/product/rosetta> or email
license@uw.edu for more information.

See <https://docs.rosettacommons.org/docs/latest/Home> for documentation.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import modal
import polars as pl

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
)
from biomodals.app.bioinfo.rosetta.execution_coordinator import (
    RosettaExecutionCoordinator,
)
from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
    load_execution_request_from_volume,
    stage_execution_request,
)
from biomodals.app.config import AppConfig
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    DeploymentIdentity,
    ExecutionOverview,
    RunStatus,
    WorkerAssignmentRecord,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
)
from biomodals.execution.modal import (
    execution_coordinator_handle as _execution_coordinator_handle,
)
from biomodals.execution.pull_worker import drive_pull_worker
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.app_execution import stage_execution_launch
from biomodals.helper.app_run import AppRunLayout, volume_path_from_mount_path
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import package_outputs, sanitize_filename, warmup_directory

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
ROSETTA_DIR = Path(__file__).parent / "rosetta"


##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .from_registry("rosettacommons/rosetta:serial-420", add_python=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.bioinfo.rosetta")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_rosetta_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 32


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 30.125),  # Each pod can run 1-30 jobs
    memory=(1024, 43008),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def run_rosetta_worker(
    coordinator,
    provider_call_id: str,
    run_name: str,
    run_id: str,
    claim_capacity: int,
    max_parallel: int,
) -> dict[str, int]:
    """Claim, execute, and report Rosetta Tasks until the durable pool is empty."""
    from biomodals.helper.shell import run_command

    if sanitize_filename(run_name) != run_name:
        raise ValueError("run_name must be a safe filename component")
    if sanitize_filename(run_id) != run_id:
        raise ValueError("run_id must be a safe filename component")
    layout = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / f"{run_name}-{run_id}"
    )
    call_id = UUID(provider_call_id)

    def claim(request_id: str, capacity: int):
        return coordinator.claim_tasks.remote(
            provider_call_id,
            request_id,
            capacity,
        )

    def execute(assignment: WorkerAssignmentRecord) -> dict[str, object]:
        task = RosettaTaskSpec.from_dict(assignment.execution_payload)
        try:
            return execute_rosetta_task(
                run_root=layout.run_root,
                task=task,
                task_fingerprint=assignment.task_fingerprint,
                run_command=run_command,
            )
        except Exception as error:  # noqa: BLE001
            return {
                "status": "failed",
                "task_key": task.task_key,
                "error": str(error) or type(error).__name__,
            }

    def complete_batch(
        completions: tuple[
            tuple[WorkerAssignmentRecord, str, dict[str, object]],
            ...,
        ],
    ) -> None:
        coordinator.complete_tasks.remote(
            provider_call_id,
            tuple(
                (assignment.task_key, request_id, result)
                for assignment, request_id, result in completions
            ),
        )

    summary = drive_pull_worker(
        provider_call_id=call_id,
        claim_capacity=claim_capacity,
        claim=claim,
        execute=execute,
        complete_batch=complete_batch,
        checkpoint_batch=CONF.output_volume.commit,
        max_parallel=max_parallel,
    )
    return asdict(summary)


@app.function(
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def package_outputs_helper(
    root: str | Path,
    paths_to_bundle: Iterable[str | Path] | None = None,
    tar_args: list[str] | None = None,
    num_threads: int = 16,
) -> bytes:
    """Modal runner to package directories into a tar.zst archive and return as bytes."""
    warmup_directory(root)
    return package_outputs(
        root,
        paths_to_bundle=paths_to_bundle,
        tar_args=tar_args,
        num_threads=num_threads,
    )


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with Rosetta's pull workers."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh the output Volume before accepting lifecycle calls."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive one staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read this Run's durable kernel overview."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume this Run without retrying failed Tasks."""
        return self._adapter().resume()

    @modal.method()
    def prepare_restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> None:
        """Persist a validated Successor request without driving it."""
        if max_active_gpu_provider_calls not in {None, 0}:
            raise ValueError("Rosetta does not admit GPU Provider Calls")
        self._adapter().prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
        )

    @modal.method()
    def drive_prepared(self) -> ExecutionOverview:
        """Drive one previously prepared root or Successor Run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
        workload_plan_fingerprint: str,
        max_active_provider_calls: int,
        claim_capacity: int,
        max_parallel_per_worker: int,
    ) -> ExecutionOverview:
        """Create a launch-time compatible Successor Run."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            max_active_provider_calls=max_active_provider_calls,
            claim_capacity=claim_capacity,
            max_parallel_per_worker=max_parallel_per_worker,
            expected_workload_plan_fingerprint=workload_plan_fingerprint,
        )
        return adapter.drive_prepared()

    @modal.method()
    def claim_tasks(
        self,
        provider_call_id: str,
        request_id: str,
        capacity: int,
    ):
        """Return one checkpointed pull Task microbatch."""
        return self._adapter().claim_tasks(
            UUID(provider_call_id),
            request_id=request_id,
            capacity=capacity,
        )

    @modal.method()
    def complete_tasks(
        self,
        provider_call_id: str,
        completions: tuple[
            tuple[str, str, dict[str, object]],
            ...,
        ],
    ):
        """Validate and checkpoint one pull Task result microbatch."""
        return self._adapter().complete_tasks(
            UUID(provider_call_id),
            completions,
        )

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached workers."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return execution_coordinator_identity(self)

    def _adapter(
        self,
        *,
        development: bool | None = None,
    ) -> RosettaExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: RosettaExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                modal_driver=_coordinator_modal_driver(development=selected_mode),
                pull_worker_coordinator=self._worker_coordinator_handle(),
                app_version=CONF.version or "",
            ),
        )

    def _worker_coordinator_handle(self):
        execution_run_id, deployment = self._identity()
        return ExecutionCoordinator(
            execution_run_id=str(execution_run_id),
            deployment_environment=deployment.environment,
            deployment_name=deployment.deployment_name,
            deployment_version=deployment.deployment_version,
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source development handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {"run_rosetta_worker": run_rosetta_worker},
        workload_name="Rosetta",
    )


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
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        use_deployed_coordinator: Target the exact deployed coordinator. The
            `biomodals app run` client supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric Modal deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    predecessor_request = (
        None
        if predecessor_execution_run_id is None
        else load_execution_request_from_volume(
            CONF.output_volume,
            predecessor_execution_run_id,
        )
    )
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

    if predecessor_request is None:
        run_id = uuid4().hex
    else:
        run_name = predecessor_request.run_name
        run_id = predecessor_request.run_id
    mount_root = Path(CONF.output_volume_mountpoint)
    layout = AppRunLayout.from_run_root(mount_root / f"{run_name}-{run_id}")
    remote_run_root = layout.run_root.relative_to(mount_root)
    remote_input_root = layout.inputs_dir.relative_to(mount_root)
    uploaded_files: set[str] = set()
    task_uploads: list[tuple[Path, str]] = []
    task_specs = []
    for row in tasks_df.iter_rows(named=True):
        local_pdb = Path(row["pdb"]).expanduser().resolve()
        remote_pdb = f"inputs/{row['index']}/{local_pdb.name}"
        task_uploads.append((local_pdb, remote_pdb))

        remote_script, remote_flags = None, None
        script_hash = row["script_hash"]
        if row["rosetta_script"] is not None:
            local_script = Path(row["rosetta_script"]).expanduser().resolve()
            remote_script = f"inputs/_script/{script_hash}.xml"
            if remote_script not in uploaded_files:
                task_uploads.append((local_script, remote_script))
                uploaded_files.add(remote_script)
        flags_hash = row["flags_hash"]
        if row["flags_file"] is not None:
            local_flags = Path(row["flags_file"]).expanduser().resolve()
            remote_flags = f"inputs/_flags/{flags_hash}.flags"
            if remote_flags not in uploaded_files:
                task_uploads.append((local_flags, remote_flags))
                uploaded_files.add(remote_flags)

        index = int(row["index"])
        task_specs.append(
            RosettaTaskSpec(
                task_key=str(index),
                index=index,
                binary=str(row["binary"]),
                pdb=remote_pdb,
                rosetta_script=remote_script,
                flags_file=remote_flags,
                output_dir=f"outputs/{index}",
                worker_log=f"logs/{index}.log",
                expected_files=(),
                input_sha256=sha256(local_pdb.read_bytes()).hexdigest(),
                script_sha256=None if script_hash is None else str(script_hash),
                flags_sha256=None if flags_hash is None else str(flags_hash),
            )
        )

    if predecessor_request is None:
        print(f"🧬 Staging {len(task_specs)} Rosetta Tasks for {run_name}...")
        with CONF.output_volume.batch_upload() as batch:
            for local_path, remote_path in task_uploads:
                batch.put_file(
                    local_path,
                    f"/{remote_run_root}/{remote_path}",
                )
            buffer = BytesIO()
            tasks_df.write_parquet(buffer)
            batch.put_file(buffer, f"/{remote_input_root}/tasks.parquet")

    num_cpu_per_pod = min(30, tasks_df.height)
    max_num_pods = min(
        max(1, max_num_pods),
        (tasks_df.height + num_cpu_per_pod - 1) // num_cpu_per_pod,
    )
    claim_capacity = (tasks_df.height + max_num_pods - 1) // max_num_pods
    max_parallel_per_worker = min(30, claim_capacity)
    app_version = CONF.version
    if app_version is None:
        raise RuntimeError("Rosetta scientific version metadata is incomplete")
    request = RosettaExecutionRequest(
        run_name=run_name,
        run_id=run_id,
        tasks=tuple(task_specs),
        app_version=app_version,
        max_active_provider_calls=max_num_pods,
        claim_capacity=claim_capacity,
        max_parallel_per_worker=max_parallel_per_worker,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    stage_execution_launch(
        CONF.output_volume,
        execution_run_id,
        predecessor_execution_run_id,
    )
    coordinator = _execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(
            development=not use_deployed_coordinator,
        )
    else:
        call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
            workload_plan_fingerprint=(
                request.execution_plan.workload_plan_fingerprint
            ),
            max_active_provider_calls=request.max_active_provider_calls,
            claim_capacity=request.claim_capacity,
            max_parallel_per_worker=request.max_parallel_per_worker,
        )
    print(f"Execution Run ID: {execution_run_id}")
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    overview = call.get()
    if overview.run.status != RunStatus.SUCCEEDED:
        diagnostic = overview.run.status_message or (
            overview.run.status_reason.value
            if overview.run.status_reason is not None
            else overview.run.status.value
        )
        raise RuntimeError(
            f"{CONF.name} Execution Run ended as "
            f"{overview.run.status.value}: {diagnostic}"
        )
    completed_request = load_execution_request_from_volume(
        CONF.output_volume,
        execution_run_id,
    )
    layout = AppRunLayout.from_run_root(mount_root / completed_request.workload_run_key)

    # Save results locally
    out_vol = volume_path_from_mount_path(
        str(layout.run_root),
        CONF.output_volume_mountpoint,
        CONF.output_volume_name,
    )
    if out_dir is None:
        print(f"🧬 {CONF.name} run complete!\nResults saved to {out_vol}")
        return

    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = local_out_dir / f"{completed_request.workload_run_key}.tar.zst"
    tarball_bytes = package_outputs_helper.remote(
        root=str(layout.run_root),
    )
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
