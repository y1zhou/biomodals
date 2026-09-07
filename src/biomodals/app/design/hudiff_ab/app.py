"""Workflow-compatible HuDiff-Ab stochastic paired humanization app."""

# ruff: noqa: PLC0415

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.app.design.hudiff_ab.execution import (
    MAX_RESULT_BYTES,
    HuDiffAbExecutionCoordinator,
    HuDiffAbExecutionRequest,
    load_execution_request,
    result_from_overview,
    stage_execution_request,
)
from biomodals.app.design.hudiff_ab.models import (
    ANTIBODY_CHECKPOINT_SHA256,
    CUBLAS_WORKSPACE_CONFIG,
    MODEL_REVISION,
    RUNTIME_IDENTITY,
    SOURCE_COMMIT,
    assert_runtime_environment,
    stage_hudiff_assets,
)
from biomodals.app.design.hudiff_ab.patches import (
    apply_hudiff_inference_patches,
    patch_identity,
)
from biomodals.app.design.hudiff_ab.validation import (
    MAX_INPUT_BYTES,
    parse_hudiff_ab_csv,
)
from biomodals.app.design.hudiff_ab.worker import (
    hudiff_ab_humanize_batch as _hudiff_ab_humanize_batch,
)
from biomodals.app.design.hudiff_ab.worker import (
    hudiff_ab_humanize_pair as _hudiff_ab_humanize_pair,
)
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    DeploymentIdentity,
    ExecutionOverview,
    RunStatus,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
    resolve_provider_call_limits,
    submit_staged_execution_run,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import build_local_output_path, fasta_identifier
from biomodals.helper.shell import package_outputs, sanitize_filename
from biomodals.schema import InlineBytes

MAX_TOTAL_ATTEMPTS = 10_000
MODEL_ROOT = Path("/biomodals-store/hudiff")
_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_DEFAULT_MAX_GPU_CONTAINERS = 1

CONF = AppConfig(
    tags={"group": "design"},
    name="HuDiff-Ab",
    repo_url="https://github.com/TencentAI4S/HuDiff",
    repo_commit_hash=SOURCE_COMMIT,
    package_name="hudiff_ab",
    version="1.0.0",
    python_version="3.10",
    cuda_version="cu116",
    timeout=_TIMEOUT_SECONDS,
)

runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .env({"CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG})
    .apt_install("git")
    .micromamba_install(
        [
            "abnumber==0.3.2",
            "anarci==2020.04.23",
            "hmmer==3.3.2",
            "biopython==1.79",
            "numpy==1.23.5",
            "pandas==1.5.3",
            "scipy==1.9.3",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .uv_pip_install(
        "torch==1.13.0+cu116",
        index_url="https://download.pytorch.org/whl/cu116",
    )
    .uv_pip_install(
        "easydict==1.13",
        "einops==0.6.1",
        "orjson==3.12.0",
        "pyyaml==6.0.3",
        "sequence-models==1.8.0",
        "tqdm==4.70.0",
    )
    .run_commands(
        f"git clone {CONF.repo_url} /opt/HuDiff",
        f"git -C /opt/HuDiff checkout --detach {CONF.repo_commit_hash}",
    )
    .add_local_python_source(
        "biomodals.app.design.hudiff_ab.patches",
        "biomodals.app.design.hudiff_ab.upstream_runtime",
        copy=True,
    )
    .run_function(apply_hudiff_inference_patches)
    # UniAF3 requires Python 3.11 and is outside this worker's import closure.
    .pipe(
        patch_image_for_helper,
        copy_patch_files=True,
        skip_deps={"uniaf3"},
    )
    .add_local_python_source(
        "biomodals.app.design.hudiff_ab.models",
        copy=True,
    )
    .run_function(assert_runtime_environment)
    .add_local_python_source(
        "biomodals.app.design.hudiff_ab.worker",
    )
)

coordinator_image = (
    modal.Image
    .micromamba(python_version="3.12")
    .micromamba_install(
        [
            "anarci==2020.04.23",
            "hmmer==3.3.2",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.hudiff_ab")
)
app = modal.App(CONF.name, image=coordinator_image, tags=CONF.tags)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_hudiff_ab_task"})


def _validate_controls(
    *,
    pair_count: int,
    candidate_count: int,
    seed: int,
    sampling_order: str,
    upstream_inference_dropout: bool,
) -> None:
    if type(candidate_count) is not int or not 1 <= candidate_count <= 25:
        raise ValueError("candidate_count must be between 1 and 25")
    if pair_count * candidate_count > MAX_TOTAL_ATTEMPTS:
        raise ValueError("HuDiff-Ab request exceeds 10,000 total attempts")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if sampling_order not in {"shuffle", "left_to_right"}:
        raise ValueError("sampling_order must be shuffle or left_to_right")
    if type(upstream_inference_dropout) is not bool:
        raise ValueError("upstream_inference_dropout must be boolean")


def _write_bundle(
    *,
    run_name: str,
    input_frame: pl.DataFrame,
    pair_results: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> bytes:
    with TemporaryDirectory(prefix="hudiff_ab_bundle_") as temporary:
        root = Path(temporary) / run_name
        root.mkdir()
        input_frame.write_csv(root / "input.csv")
        attempts = [row for result in pair_results for row in result["attempts"]]
        candidates = [row for result in pair_results for row in result["candidates"]]
        mutations = [row for result in pair_results for row in result["mutations"]]
        pl.DataFrame(attempts, infer_schema_length=None).write_csv(
            root / "attempts.csv"
        )
        pl.DataFrame(
            candidates,
            schema={
                "id": pl.String,
                "candidate_id": pl.String,
                "attempt_index": pl.Int64,
                "vh": pl.String,
                "vl": pl.String,
            },
        ).write_csv(root / "candidates.csv")
        (root / "candidates.fasta").write_text(
            "".join(
                f">{fasta_identifier(row['candidate_id'])}_VH\n{row['vh']}\n"
                f">{fasta_identifier(row['candidate_id'])}_VL\n{row['vl']}\n"
                for row in candidates
            ),
            encoding="utf-8",
        )
        pl.DataFrame(
            mutations,
            schema={
                "id": pl.String,
                "candidate_id": pl.String,
                "chain": pl.String,
                "imgt_position": pl.String,
                "region": pl.String,
                "input_aa": pl.String,
                "final_aa": pl.String,
            },
        ).write_parquet(root / "mutations.parquet", compression="zstd")
        asset_manifests = {
            orjson.dumps(result["asset_manifest"], option=orjson.OPT_SORT_KEYS)
            for result in pair_results
        }
        if len(asset_manifests) != 1:
            raise ValueError("HuDiff-Ab pair results used different model assets")
        patch_identities = {str(result["patch_identity"]) for result in pair_results}
        if len(patch_identities) != 1:
            raise ValueError("HuDiff-Ab pair results used different source patches")
        runtime_versions = {
            orjson.dumps(result["runtime_versions"], option=orjson.OPT_SORT_KEYS)
            for result in pair_results
        }
        if len(runtime_versions) != 1:
            raise ValueError("HuDiff-Ab pair results used different runtime versions")
        files = sorted(path for path in root.iterdir() if path.is_file())
        manifest = {
            "schema_version": 2,
            "run_name": run_name,
            "pair_count": input_frame.height,
            "attempt_count": len(attempts),
            "candidate_count": len(candidates),
            "invalid_attempt_count": sum(
                row["status"] == "invalid" for row in attempts
            ),
            "duplicate_attempt_count": sum(
                row["status"] == "duplicate" for row in attempts
            ),
            "candidate_generation_status": {
                result["id"]: result["candidate_generation_status"]
                for result in pair_results
            },
            "input_sha256": hashlib.sha256(
                input_frame.write_csv().encode()
            ).hexdigest(),
            "parameters": {
                **parameters,
                "pair_seeds": {
                    result["id"]: result["pair_seed"] for result in pair_results
                },
                "cdr_definition": "kabat_no_vernier_on_imgt_grid",
                "invalid_sample_policy": "retain_attempt_without_resampling",
            },
            "scientific_identity": {
                "source_commit": SOURCE_COMMIT,
                "model_revision": MODEL_REVISION,
                "checkpoint_sha256": ANTIBODY_CHECKPOINT_SHA256,
                "patch_identity": patch_identities.pop(),
                "runtime": RUNTIME_IDENTITY,
                "runtime_versions": orjson.loads(runtime_versions.pop()),
                "staged_assets": orjson.loads(asset_manifests.pop()),
            },
            "telemetry": {
                "accelerators": sorted({
                    str(result["device"]) for result in pair_results
                })
            },
            "warnings": [
                "Generated sequences are research-use design candidates, not evidence of preserved binding, developability, or clinical immunogenicity."
            ],
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            },
        }
        (root / "manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        archive = package_outputs(root, num_threads=2)
        if len(archive) > MAX_RESULT_BYTES:
            raise ValueError("HuDiff-Ab result archive exceeds the byte limit")
        return archive


def _aggregate_hudiff_ab_results(
    *,
    run_name: str,
    csv_bytes: bytes,
    pair_results: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> tuple[bytes, dict[str, int]]:
    frame = parse_hudiff_ab_csv(csv_bytes)
    _validate_controls(pair_count=frame.height, **parameters)
    if [result["id"] for result in pair_results] != frame["id"].to_list():
        raise ValueError("HuDiff-Ab pair results do not match input order")
    archive = _write_bundle(
        run_name=run_name,
        input_frame=frame,
        pair_results=pair_results,
        parameters=parameters,
    )
    return archive, {
        "pair_count": len(pair_results),
        "attempt_count": sum(len(result["attempts"]) for result in pair_results),
        "candidate_count": sum(len(result["candidates"]) for result in pair_results),
        "mutation_count": sum(len(result["mutations"]) for result in pair_results),
    }


hudiff_ab_humanize_pair = app.function(
    image=runtime_image,
    cpu=(0.125, 8.125),
    memory=(512, 16384),
    gpu="A10G",
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False),
)(_hudiff_ab_humanize_pair)

hudiff_ab_humanize_batch = app.function(
    image=runtime_image,
    cpu=(0.125, 8.125),
    memory=(512, 65536),
    gpu="A10G",
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False),
)(_hudiff_ab_humanize_batch)


@app.function(
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(
        model_volume=True,
        model_ro=False,
        model_mount_subdir=False,
    ),
)
def stage_hudiff_models() -> dict[str, object]:
    """Explicitly download, verify, and commit the checkpoint-only publication."""
    manifest = stage_hudiff_assets(MODEL_ROOT)
    MODEL_VOLUME.commit()
    return manifest


@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with HuDiff-Ab workers."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh durable state before accepting lifecycle methods."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive the staged root run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read this run's durable kernel overview."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Request idempotent cancellation for this run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume this run without retrying failed work."""
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
        """Persist a validated successor request without driving it."""
        self._adapter().prepare_restart(
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
    def drive_prepared(self) -> ExecutionOverview:
        """Drive one previously prepared root or successor run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(self, predecessor_execution_run_id: str) -> ExecutionOverview:
        """Create a compatible successor while inferring predecessor identity."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                CONF.output_volume_mountpoint,
                UUID(self.execution_run_id),
            ),
        )
        return adapter.drive_prepared()

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return execution_coordinator_identity(self)

    def _adapter(
        self, *, development: bool | None = None
    ) -> HuDiffAbExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: HuDiffAbExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_volume_name=CONF.output_volume_name,
                provider_driver=_coordinator_modal_driver(development=selected_mode),
                source_commit=SOURCE_COMMIT,
                checkpoint_sha256=ANTIBODY_CHECKPOINT_SHA256,
                patch_identity=patch_identity(),
                runtime_identity=RUNTIME_IDENTITY,
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "hudiff_ab_humanize_batch": hudiff_ab_humanize_batch,
            "hudiff_ab_humanize_pair": hudiff_ab_humanize_pair,
        },
        workload_name=CONF.name,
    )


@app.local_entrypoint()
def submit_hudiff_ab_task(
    input_csv: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    candidate_count: int = 10,
    seed: int = 42,
    sampling_order: str = "shuffle",
    upstream_inference_dropout: bool = True,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Humanize paired VH-VL sequences and save one result archive.

    Args:
        input_csv: UTF-8 CSV with exactly ``id,vh,vl`` columns.
        output_dir: Local directory for the downloaded archive.
        run_name: Optional safe display name for this invocation.
        candidate_count: Sampling attempts per pair, from one through 25.
        seed: Unsigned 32-bit root seed; each pair receives a derived seed.
        sampling_order: ``shuffle`` or deterministic ``left_to_right``.
        upstream_inference_dropout: Preserve released always-active dropout.
        max_containers: Run-wide provider-call ceiling.
        max_gpu_containers: Run-wide GPU provider-call ceiling.
        use_deployed_coordinator: Use the pinned deployed coordinator.
        deployment_environment: Modal environment containing the deployment.
        deployment_name: Exact deployed app name.
        deployment_version: Exact deployed app version.
        restart_from: Optional predecessor Execution Run UUID.
    """
    path = Path(input_csv).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV is missing or exceeds {MAX_INPUT_BYTES} bytes")
    csv_bytes = path.read_bytes()
    frame = parse_hudiff_ab_csv(csv_bytes)
    _validate_controls(
        pair_count=frame.height,
        candidate_count=candidate_count,
        seed=seed,
        sampling_order=sampling_order,
        upstream_inference_dropout=upstream_inference_dropout,
    )
    selected_run_name = sanitize_filename(run_name or path.stem)
    total_limit, gpu_limit = resolve_provider_call_limits(
        default_max_containers=min(frame.height, _DEFAULT_MAX_GPU_CONTAINERS),
        default_max_gpu_containers=min(frame.height, _DEFAULT_MAX_GPU_CONTAINERS),
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
    )
    request = HuDiffAbExecutionRequest(
        run_name=selected_run_name,
        csv_bytes=csv_bytes,
        candidate_count=candidate_count,
        seed=seed,
        sampling_order=sampling_order,
        upstream_inference_dropout=upstream_inference_dropout,
        source_commit=SOURCE_COMMIT,
        checkpoint_sha256=ANTIBODY_CHECKPOINT_SHA256,
        patch_identity=patch_identity(),
        runtime_identity=RUNTIME_IDENTITY,
        max_active_provider_calls=total_limit,
        max_active_gpu_provider_calls=gpu_limit,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    overview = submit_staged_execution_run(
        CONF.output_volume,
        execution_run_id=execution_run_id,
        deployment=deployment,
        predecessor_execution_run_id=None
        if restart_from is None
        else UUID(restart_from),
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
        workload_name=CONF.name,
        accepted_statuses=(RunStatus.SUCCEEDED,),
    )
    result = result_from_overview(overview, CONF.output_volume)
    output = next(
        item for item in result.outputs if item.name == "hudiff_ab_humanization"
    )
    if not isinstance(output.storage, InlineBytes):
        raise TypeError("HuDiff-Ab output must be inline .tar.zst bytes")
    output_path = build_local_output_path(
        Path.cwd() if output_dir is None else Path(output_dir),
        run_name=selected_run_name,
        suffix="hudiff-ab",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output.storage.data)
    print(f"HuDiff-Ab results saved to: {output_path.resolve()}")
