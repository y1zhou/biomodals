"""Direct ABCFold2 adaptation of the shared execution kernel."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodeDependency,
    NodePlan,
    ProviderCallStatus,
    inline_json_result,
    republish_execution_artifact,
)
from biomodals.execution.modal import (
    ExecutionRequestFile,
    OutputClaimExecutionDefinitionCoordinatorLifecycle,
    load_execution_provider_result,
)
from biomodals.execution.nodes import (
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.helper.output_claim import acquire_output_claim
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
)

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 16 * 1024 * 1024
PREPARE_NODE = "prepare"
BOLTZ_DOWNLOAD_NODE = "download-boltz-models"
CHAI_DOWNLOAD_NODE = "download-chai-models"
BOLTZ_SEEDS_NODE = "run-boltz-seeds"
BOLTZ_ARCHIVE_NODE = "collect-boltz"
CHAI_SEEDS_NODE = "run-chai-seeds"
CHAI_ARCHIVE_NODE = "collect-chai"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "ABCFold2 execution request",
)


@dataclass(frozen=True, slots=True)
class ABCFold2RunConfig:
    """Validated cross-function ABCFold2 run configuration."""

    run_id: str
    workdir: str
    seeds: tuple[int, ...]
    num_trunk_recycles: int
    num_diffn_timesteps: int
    num_diffn_samples: int
    num_trunk_samples: int
    boltz_additional_cli_args: tuple[str, ...] | None

    def as_kwargs(self) -> dict[str, object]:
        """Return the primitive mapping expected by established workers."""
        return {
            "run_id": self.run_id,
            "workdir": self.workdir,
            "seeds": list(self.seeds),
            "num_trunk_recycles": self.num_trunk_recycles,
            "num_diffn_timesteps": self.num_diffn_timesteps,
            "num_diffn_samples": self.num_diffn_samples,
            "num_trunk_samples": self.num_trunk_samples,
            "boltz_additional_cli_args": (
                None
                if self.boltz_additional_cli_args is None
                else list(self.boltz_additional_cli_args)
            ),
        }


@dataclass(frozen=True)
class ABCFold2ExecutionRequest:
    """Immutable scientific input plus operational execution settings."""

    run_name: str
    yaml_content: bytes
    msa_chains: str | None
    search_templates: bool
    download_models: bool
    force_redownload: bool
    run_boltz: bool
    run_chai: bool
    max_active_provider_calls: int
    app_version: str
    boltz_version: str
    chai_version: str
    max_active_gpu_provider_calls: int | None = None
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject empty inputs and unusable coordinator capacity."""
        if not self.run_name or not self.yaml_content:
            raise ValueError("ABCFold2 run name and YAML cannot be empty")
        gpu_limit = self.max_active_gpu_provider_calls
        if gpu_limit is None:
            gpu_limit = self.max_active_provider_calls
            object.__setattr__(
                self,
                "max_active_gpu_provider_calls",
                gpu_limit,
            )
        if (
            self.max_active_provider_calls < 1
            or gpu_limit < 0
            or gpu_limit > self.max_active_provider_calls
        ):
            raise ValueError("ABCFold2 provider-call limits are invalid")
        if not self.app_version or not self.boltz_version or not self.chai_version:
            raise ValueError("ABCFold2 deployment versions cannot be empty")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build the selected parallel seed and archive branches."""
        nodes = [NodePlan(PREPARE_NODE)]
        if self.download_models:
            nodes.extend((
                NodePlan(BOLTZ_DOWNLOAD_NODE),
                NodePlan(CHAI_DOWNLOAD_NODE),
            ))
        if self.run_boltz:
            dependencies = [NodeDependency(PREPARE_NODE)]
            if self.download_models:
                dependencies.append(NodeDependency(BOLTZ_DOWNLOAD_NODE))
            nodes.extend((
                NodePlan(BOLTZ_SEEDS_NODE, dependencies=tuple(dependencies)),
                NodePlan(
                    BOLTZ_ARCHIVE_NODE,
                    dependencies=(NodeDependency(BOLTZ_SEEDS_NODE),),
                ),
            ))
        if self.run_chai:
            dependencies = [NodeDependency(PREPARE_NODE)]
            if self.download_models:
                dependencies.append(NodeDependency(CHAI_DOWNLOAD_NODE))
            nodes.extend((
                NodePlan(CHAI_SEEDS_NODE, dependencies=tuple(dependencies)),
                NodePlan(
                    CHAI_ARCHIVE_NODE,
                    dependencies=(NodeDependency(CHAI_SEEDS_NODE),),
                ),
            ))
        scientific_versions = {
            "abcfold2": self.app_version,
            "biomodals.abcfold2.execution_request": str(REQUEST_SCHEMA_VERSION),
        }
        if self.download_models or self.run_boltz:
            scientific_versions["boltz"] = self.boltz_version
        if self.download_models or self.run_chai:
            scientific_versions["chai"] = self.chai_version
        return ExecutionPlan(
            workload_name="abcfold2",
            workload_run_key=self.run_name,
            nodes=tuple(nodes),
            scientific_payload={
                "yaml_sha256": sha256(self.yaml_content).hexdigest(),
                "msa_chains": self.msa_chains,
                "search_templates": self.search_templates,
                "run_boltz": self.run_boltz,
                "run_chai": self.run_chai,
            },
            scientific_versions=scientific_versions,
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "yaml_content": b64encode(self.yaml_content).decode("ascii"),
                "msa_chains": self.msa_chains,
                "search_templates": self.search_templates,
                "download_models": self.download_models,
                "force_redownload": self.force_redownload,
                "run_boltz": self.run_boltz,
                "run_chai": self.run_chai,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
                "app_version": self.app_version,
                "boltz_version": self.boltz_version,
                "chai_version": self.chai_version,
                "replace_claim_owner": self.replace_claim_owner,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("ABCFold2 execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> ABCFold2ExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("ABCFold2 execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("ABCFold2 execution request schema is unsupported")
        encoded_yaml = value.pop("yaml_content", None)
        if not isinstance(encoded_yaml, str):
            raise TypeError("ABCFold2 YAML content must be base64 text")
        value["yaml_content"] = b64decode(encoded_yaml, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: ABCFold2ExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: ABCFold2ExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> ABCFold2ExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return ABCFold2ExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


class ABCFold2Publications:
    """Validate ABCFold2's established seed and archive publications."""

    def __init__(
        self,
        *,
        request: ABCFold2ExecutionRequest,
        execution_run_id: UUID,
        output_root: str | Path,
        output_volume_name: str,
        output_claims: Any,
    ) -> None:
        """Bind one Run to its ABCFold2 publication namespace."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_root = Path(output_root)
        self.output_volume_name = output_volume_name
        self.output_claims = output_claims

    def claim(self, model_name: str, run_config: ABCFold2RunConfig) -> str:
        """Claim one model publication before admitting its provider call."""
        publication_key = self.publication_key(model_name, run_config)
        acquire_output_claim(
            self.output_claims,
            claim_key=f"abcfold2-{model_name}:{publication_key}",
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        return publication_key

    @staticmethod
    def publication_key(model_name: str, run_config: ABCFold2RunConfig) -> str:
        """Return the established model publication identity."""
        return _workload_module()._model_publication_key(
            model_name,
            run_config.as_kwargs(),
        )

    def seed_result(
        self,
        model_name: str,
        seed: int,
        run_config: ABCFold2RunConfig,
    ) -> AppRunResult | None:
        """Reconstruct one exact seed result from its workload marker."""
        publication_key = self.publication_key(model_name, run_config)
        app = _workload_module()
        if not app._seed_ready(
            run_config.workdir,
            model_name,
            seed,
            publication_key,
        ):
            return None
        result_dir = (
            Path(run_config.workdir)
            / f"{model_name}_models"
            / (
                f"boltz_results_seed-{seed}"
                if model_name == "boltz"
                else f"chai_seed-{seed}"
            )
        )
        marker = orjson.loads(
            (
                Path(run_config.workdir)
                / ".biomodals"
                / f"{model_name}-seed-{seed}.json"
            ).read_bytes()
        )
        raw_files = marker.get("artifacts") if isinstance(marker, dict) else None
        if not isinstance(raw_files, list):
            return None
        files = [
            ArtifactFile(
                path=str(file["path"]),
                size_bytes=int(file["size"]),
                content_sha256=str(file["sha256"]),
            )
            for file in raw_files
            if isinstance(file, dict)
        ]
        if len(files) != len(raw_files):
            return None
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=f"{model_name}-seed-{seed}",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=result_dir.relative_to(self.output_root).as_posix(),
                    ),
                    metadata={
                        "files": [
                            file.model_dump(mode="json", exclude_none=True)
                            for file in files
                        ],
                        "model": model_name,
                        "seed": str(seed),
                    },
                )
            ],
        )

    def archive_result(
        self,
        model_name: str,
        run_config: ABCFold2RunConfig,
    ) -> AppRunResult | None:
        """Reconstruct one exact archive result from its workload marker."""
        app = _workload_module()
        publication_key = self.publication_key(model_name, run_config)
        if not app._archive_ready(
            run_config.workdir,
            model_name,
            publication_key,
        ):
            return None
        path = Path(run_config.workdir) / f"{model_name}_models.tar.zst"
        marker = orjson.loads(
            (
                Path(run_config.workdir) / ".biomodals" / f"{model_name}-archive.json"
            ).read_bytes()
        )
        if not isinstance(marker, dict):
            return None
        file = ArtifactFile(
            path=path.name,
            size_bytes=marker.get("size"),
            content_sha256=marker.get("sha256"),
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=f"{model_name}-archive",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=path.relative_to(self.output_root).as_posix(),
                    ),
                    metadata={
                        "files": [file.model_dump(mode="json", exclude_none=True)],
                        "model": model_name,
                    },
                )
            ],
        )


@dataclass
class _ABCFold2PrepareNode(ProviderNode):
    request: ABCFold2ExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name="prepare_abcfold2",
            uses_gpu=False,
            runtime_image_key="abcfold2-prepare",
            kwargs={
                "yaml_str": self.request.yaml_content,
                "search_templates": self.request.search_templates,
                "msa_chains": self.request.msa_chains,
            },
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        run_config = _run_config_from_value(result)
        return inline_json_result(
            name="run-config",
            value=run_config.as_kwargs(),
            filename="run-config.json",
        )


@dataclass
class _ABCFold2DownloadNode(ProviderNode):
    function_name: str
    force: bool

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name=self.function_name,
            uses_gpu=False,
            runtime_image_key=f"abcfold2-{self.function_name}",
            kwargs={"force": self.force},
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        if result is not None:
            raise ValueError("ABCFold2 model download returned an unexpected value")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


@dataclass
class _ABCFold2SeedNode(TaskProviderNode):
    model_name: str
    publications: ABCFold2Publications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        run_config = _run_config_from_context(context)
        return tuple(
            TaskDefinition(
                task_key=str(seed),
                scientific_payload={"run_id": run_config.run_id, "seed": seed},
            )
            for seed in run_config.seeds
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        run_config = _run_config_from_context(context)
        publication_key = self.publications.claim(self.model_name, run_config)
        return ProviderCallSpec(
            function_name=f"run_abcfold2_{self.model_name}",
            uses_gpu=True,
            runtime_image_key="abcfold2-gpu",
            kwargs={
                "seed": int(task.task_key),
                **run_config.as_kwargs(),
                "publication_key": publication_key,
            },
            metadata={"run_config": run_config.as_kwargs()},
        )

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        if not isinstance(result, str):
            raise ValueError("ABCFold2 seed worker returned no output path")
        run_config = _run_config_from_value(metadata.get("run_config"))
        publication = self.publications.seed_result(
            self.model_name,
            int(task_key),
            run_config,
        )
        if publication is None:
            raise FileNotFoundError("ABCFold2 seed publication is unavailable")
        return publication

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del expected_fingerprint
        return self.publications.seed_result(
            self.model_name,
            int(task.task_key),
            _run_config_from_context(context),
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del expected_fingerprint, result, artifacts
        try:
            available = self.recover_remote_task_result(context, task, "") is not None
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        if errors:
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[
                    "; ".join(f"{key}: {value}" for key, value in errors.items())
                ],
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                republish_execution_artifact(context.single_input("run_config")),
                *(output for result in results.values() for output in result.outputs),
            ],
        )


@dataclass
class _ABCFold2ArchiveNode(ProviderNode):
    model_name: str
    publications: ABCFold2Publications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        run_config = _run_config_from_context(context)
        return ProviderCallSpec(
            function_name=f"collect_abcfold2_{self.model_name}_data",
            uses_gpu=False,
            runtime_image_key=f"abcfold2-collect-{self.model_name}",
            kwargs={
                "run_conf": run_config.as_kwargs(),
                "publication_key": self.publications.claim(
                    self.model_name,
                    run_config,
                ),
            },
            metadata={"run_config": run_config.as_kwargs()},
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        if not isinstance(result, dict) or "archive_path" not in result:
            raise ValueError("ABCFold2 archive worker returned invalid metadata")
        run_config = _run_config_from_value(metadata.get("run_config"))
        publication = self.publications.archive_result(self.model_name, run_config)
        if publication is None:
            raise FileNotFoundError("ABCFold2 archive publication is unavailable")
        return publication

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        try:
            run_config = _run_config_from_context(context)
        except ValueError:
            return None
        return self.publications.archive_result(
            self.model_name,
            run_config,
        )

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        try:
            available = self.recover_result_publication(context) is not None
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING


def _run_config_from_value(value: object) -> ABCFold2RunConfig:
    if not isinstance(value, dict):
        raise TypeError("ABCFold2 run config is invalid")
    value = cast(dict[str, object], value)
    run_id = value.get("run_id")
    workdir = value.get("workdir")
    seeds = value.get("seeds")
    num_trunk_recycles = value.get("num_trunk_recycles")
    num_diffn_timesteps = value.get("num_diffn_timesteps")
    num_diffn_samples = value.get("num_diffn_samples")
    num_trunk_samples = value.get("num_trunk_samples")
    additional_args = value.get("boltz_additional_cli_args")
    if (
        not isinstance(run_id, str)
        or not isinstance(workdir, str)
        or not isinstance(seeds, list)
        or not all(type(seed) is int for seed in seeds)
        or type(num_trunk_recycles) is not int
        or type(num_diffn_timesteps) is not int
        or type(num_diffn_samples) is not int
        or type(num_trunk_samples) is not int
        or (
            additional_args is not None
            and (
                not isinstance(additional_args, list)
                or not all(isinstance(item, str) for item in additional_args)
            )
        )
    ):
        raise TypeError("ABCFold2 run config has invalid fields")
    return ABCFold2RunConfig(
        run_id=run_id,
        workdir=workdir,
        seeds=tuple(cast(list[int], seeds)),
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        num_trunk_samples=num_trunk_samples,
        boltz_additional_cli_args=(
            None if additional_args is None else tuple(cast(list[str], additional_args))
        ),
    )


def _run_config_from_context(context: NodeRunContext) -> ABCFold2RunConfig:
    return _run_config_from_value(orjson.loads(context.read_input_bytes("run_config")))


def run_config_from_overview(
    overview: ExecutionOverview,
    output_volume: Any,
) -> ABCFold2RunConfig:
    """Return the validated preparation result from a completed overview."""
    for call in overview.representative_provider_calls:
        if (
            call.node_key == PREPARE_NODE
            and call.status == ProviderCallStatus.SUCCEEDED
        ):
            return _run_config_from_value(
                load_execution_provider_result(
                    output_volume,
                    execution_run_id=overview.run.execution_run_id,
                    envelope=call.result_envelope,
                )
            )
    raise LookupError("ABCFold2 preparation result is unavailable")


def _workload_module():
    """Import workload-owned publication probes after Modal app loading."""
    from biomodals.app.fold import abcfold2_app

    return abcfold2_app


def abcfold2_execution_graph(
    request: ABCFold2ExecutionRequest,
    publications: ABCFold2Publications,
) -> ExecutionGraph:
    """Build ABCFold2's established parallel model branches."""
    graph = ExecutionGraph(
        "abcfold2",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="abcfold2",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    prepare = graph.add_node(
        _ABCFold2PrepareNode(request),
        id=PREPARE_NODE,
    )
    boltz_download = None
    chai_download = None
    if request.download_models:
        boltz_download = graph.add_node(
            _ABCFold2DownloadNode("download_boltz_models", request.force_redownload),
            id=BOLTZ_DOWNLOAD_NODE,
        )
        chai_download = graph.add_node(
            _ABCFold2DownloadNode("download_chai_models", request.force_redownload),
            id=CHAI_DOWNLOAD_NODE,
        )

    def add_model_branch(model_name: str, download: Any) -> None:
        seeds = graph.add_node(
            _ABCFold2SeedNode(model_name, publications),
            id=f"run-{model_name}-seeds",
            inputs={
                "run_config": prepare.outputs(kind=ArtifactKind.TABLE),
            },
            depends_on=[] if download is None else [download],
        )
        graph.add_node(
            _ABCFold2ArchiveNode(model_name, publications),
            id=f"collect-{model_name}",
            inputs={
                "run_config": seeds.outputs(kind=ArtifactKind.TABLE),
            },
        )

    if request.run_boltz:
        add_model_branch("boltz", boltz_download)
    if request.run_chai:
        add_model_branch("chai", chai_download)
    return graph


class ABCFold2ExecutionCoordinator(OutputClaimExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to ABCFold2 publications."""

    _request_loader = staticmethod(load_execution_request)
    _request_persister = staticmethod(persist_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        output_volume_name: str,
        output_claims: Any,
        provider_driver: Any,
        app_version: str,
        boltz_version: str,
        chai_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only resources used by this workload adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            output_claims=output_claims,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={
                "abcfold2": app_version,
                "boltz": boltz_version,
                "chai": chai_version,
            },
            poll_interval_seconds=poll_interval_seconds,
        )
        self.output_volume_name = output_volume_name
        self.output_claims = output_claims

    def _graph(
        self,
        request: ABCFold2ExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return abcfold2_execution_graph(
            request=request,
            publications=ABCFold2Publications(
                request=request,
                execution_run_id=self.execution_run_id,
                output_root=self.volume_root,
                output_volume_name=self.output_volume_name,
                output_claims=self.output_claims,
            ),
        )
