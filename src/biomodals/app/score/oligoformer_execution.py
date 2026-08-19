"""Direct OligoFormer adaptation of the shared execution kernel."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodeDependency,
    NodePlan,
    inline_json_result,
    republish_execution_artifact,
)
from biomodals.execution.modal import (
    ExecutionRequestFile,
    OutputClaimExecutionDefinitionCoordinatorLifecycle,
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
    InlineBytes,
    VolumePath,
)

REQUEST_SCHEMA_VERSION = 2
MAX_REQUEST_BYTES = 64 * 1024 * 1024
DOWNLOAD_NODE = "download-models"
PREPARE_NODE = "prepare-run"
REFERENCE_PLAN_NODE = "plan-reference-cache"
REFERENCE_SHARDS_NODE = "build-reference-cache"
REFERENCE_FINALIZE_NODE = "publish-reference-cache"
EFFICACY_NODE = "predict-efficacy"
EVIDENCE_PLAN_NODE = "plan-off-target-evidence"
PITA_REFERENCE_NODE = "prepare-pita-references"
PITA_CANDIDATES_NODE = "score-pita-candidates"
TARGETSCAN_TILES_NODE = "score-targetscan-tiles"
EVIDENCE_MERGE_NODE = "publish-off-target-evidence"
FINAL_NODE = "build-final-tables"
PUBLISH_NODE = "publish-result"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "OligoFormer execution request",
)


@dataclass(frozen=True)
class OligoformerExecutionRequest:
    """Immutable inputs, scientific flags, and operational concurrency."""

    run_name: str
    mrna_fasta_bytes: bytes
    sirna_fasta_bytes: bytes | None
    off_target: bool
    toxicity: bool
    all_human: bool
    utr_bytes: bytes | None
    orf_bytes: bytes | None
    top_n: int
    functionality_filter: bool
    pita_threshold: float
    targetscan_threshold: float
    toxicity_threshold: float
    off_target_nodes: int
    off_target_workers: int
    off_target_process_slots: int
    off_target_prep_workers: int
    pita_prepare_nodes: int
    pita_prepare_workers: int
    pita_prepare_utr_shard_size: int
    pita_row_shard_size: int
    pita_row_attempts: int
    targetscan_rnaplfold_nodes: int
    targetscan_rnaplfold_workers: int
    targetscan_rnaplfold_shard_size: int
    targetscan_prepare_nodes: int
    targetscan_ref_shard_size: int | None
    targetscan_candidate_shard_size: int
    targetscan_context_nodes: int
    targetscan_context_workers: int
    targetscan_context_shard_size: int
    targetscan_context_attempts: int
    targetscan_merge_nodes: int
    force: bool
    force_generation: str | None
    app_version: str
    model_version: str
    reference_version: str | None
    max_active_provider_calls: int | None = None
    max_active_gpu_provider_calls: int = 1
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Validate only request invariants needed before remote staging."""
        if (
            not self.run_name
            or not self.mrna_fasta_bytes
            or not self.app_version
            or not self.model_version
        ):
            raise ValueError(
                "OligoFormer run name, mRNA input, app version, and model "
                "version are required"
            )
        if self.off_target and self.all_human and not self.reference_version:
            raise ValueError("Full-human OligoFormer runs require a reference version")
        if self.top_n != -1 and self.top_n < 1:
            raise ValueError("top_n must be -1 or a positive integer")
        total_limit = self.max_active_provider_calls
        if total_limit is None:
            total_limit = max(
                2,
                self.off_target_process_slots,
                self.targetscan_rnaplfold_nodes,
            )
            object.__setattr__(
                self,
                "max_active_provider_calls",
                total_limit,
            )
        if (
            total_limit < 1
            or self.max_active_gpu_provider_calls < 0
            or self.max_active_gpu_provider_calls > total_limit
        ):
            raise ValueError("OligoFormer provider-call limits are invalid")
        if (
            self.off_target
            and not self.all_human
            and (self.utr_bytes is None or self.orf_bytes is None)
        ):
            raise ValueError(
                "Set --utr-file and --orf-file for off-target prediction, or pass "
                "--all-human."
            )
        if (
            self.targetscan_ref_shard_size is not None
            and self.targetscan_ref_shard_size < 1
        ):
            raise ValueError("targetscan_ref_shard_size must be positive")
        _ = self.execution_config

    @property
    def execution_config(self):
        """Build the workload-owned operational configuration lazily."""
        app = _workload_module()
        return app.OligoformerExecutionConfig(
            off_target_nodes=self.off_target_nodes,
            off_target_workers=self.off_target_workers,
            off_target_process_slots=self.off_target_process_slots,
            off_target_prep_workers=self.off_target_prep_workers,
            pita_prepare_nodes=self.pita_prepare_nodes,
            pita_prepare_workers=self.pita_prepare_workers,
            pita_prepare_utr_shard_size=self.pita_prepare_utr_shard_size,
            pita_row_shard_size=self.pita_row_shard_size,
            pita_row_attempts=self.pita_row_attempts,
            targetscan_rnaplfold_nodes=self.targetscan_rnaplfold_nodes,
            targetscan_rnaplfold_workers=self.targetscan_rnaplfold_workers,
            targetscan_rnaplfold_shard_size=self.targetscan_rnaplfold_shard_size,
            targetscan_prepare_nodes=self.targetscan_prepare_nodes,
            targetscan_candidate_shard_size=self.targetscan_candidate_shard_size,
            targetscan_context_nodes=self.targetscan_context_nodes,
            targetscan_context_workers=self.targetscan_context_workers,
            targetscan_context_shard_size=self.targetscan_context_shard_size,
            targetscan_context_attempts=self.targetscan_context_attempts,
            targetscan_merge_nodes=self.targetscan_merge_nodes,
        )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build the conditional scientific graph for this direct App Run."""
        nodes = [
            NodePlan(DOWNLOAD_NODE),
            NodePlan(PREPARE_NODE, dependencies=(NodeDependency(DOWNLOAD_NODE),)),
            NodePlan(EFFICACY_NODE, dependencies=(NodeDependency(PREPARE_NODE),)),
        ]
        evidence_dependencies = [NodeDependency(EFFICACY_NODE)]
        if self.off_target and self.all_human:
            nodes.extend((
                NodePlan(
                    REFERENCE_PLAN_NODE,
                    dependencies=(NodeDependency(PREPARE_NODE),),
                ),
                NodePlan(
                    REFERENCE_SHARDS_NODE,
                    dependencies=(NodeDependency(REFERENCE_PLAN_NODE),),
                ),
                NodePlan(
                    REFERENCE_FINALIZE_NODE,
                    dependencies=(NodeDependency(REFERENCE_SHARDS_NODE),),
                ),
            ))
            evidence_dependencies.append(NodeDependency(REFERENCE_FINALIZE_NODE))
        if self.off_target:
            nodes.extend((
                NodePlan(EVIDENCE_PLAN_NODE, dependencies=tuple(evidence_dependencies)),
                NodePlan(
                    PITA_REFERENCE_NODE,
                    dependencies=(NodeDependency(EVIDENCE_PLAN_NODE),),
                ),
                NodePlan(
                    PITA_CANDIDATES_NODE,
                    dependencies=(NodeDependency(PITA_REFERENCE_NODE),),
                ),
                NodePlan(
                    TARGETSCAN_TILES_NODE,
                    dependencies=(NodeDependency(EVIDENCE_PLAN_NODE),),
                ),
                NodePlan(
                    EVIDENCE_MERGE_NODE,
                    dependencies=(
                        NodeDependency(PITA_CANDIDATES_NODE),
                        NodeDependency(TARGETSCAN_TILES_NODE),
                    ),
                ),
            ))
            final_dependencies = (NodeDependency(EVIDENCE_MERGE_NODE),)
        else:
            final_dependencies = (NodeDependency(EFFICACY_NODE),)
        nodes.extend((
            NodePlan(FINAL_NODE, dependencies=final_dependencies),
            NodePlan(PUBLISH_NODE, dependencies=(NodeDependency(FINAL_NODE),)),
        ))
        scientific_versions = {
            "oligoformer": self.app_version,
            "oligoformer.model": self.model_version,
            "biomodals.oligoformer.execution_request": str(REQUEST_SCHEMA_VERSION),
        }
        if self.off_target and self.all_human and self.reference_version is not None:
            scientific_versions["oligoformer.reference"] = self.reference_version
        return ExecutionPlan(
            workload_name="oligoformer",
            workload_run_key=self.run_name,
            nodes=tuple(nodes),
            scientific_payload={
                "mrna_sha256": sha256(self.mrna_fasta_bytes).hexdigest(),
                "sirna_sha256": _optional_digest(self.sirna_fasta_bytes),
                "utr_sha256": _optional_digest(self.utr_bytes),
                "orf_sha256": _optional_digest(self.orf_bytes),
                "off_target": self.off_target,
                "toxicity": self.toxicity,
                "all_human": self.all_human,
                "top_n": self.top_n,
                "functionality_filter": self.functionality_filter,
                "pita_threshold": self.pita_threshold,
                "targetscan_threshold": self.targetscan_threshold,
                "toxicity_threshold": self.toxicity_threshold,
                "force_generation": self.force_generation,
            },
            scientific_versions=scientific_versions,
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        value = asdict(self)
        for name in ("mrna_fasta_bytes", "sirna_fasta_bytes", "utr_bytes", "orf_bytes"):
            content = value[name]
            value[name] = (
                None if content is None else b64encode(content).decode("ascii")
            )
        value["schema_version"] = REQUEST_SCHEMA_VERSION
        content = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("OligoFormer execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> OligoformerExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("OligoFormer execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("OligoFormer execution request schema is unsupported")
        for name in ("mrna_fasta_bytes", "sirna_fasta_bytes", "utr_bytes", "orf_bytes"):
            encoded = value.get(name)
            if encoded is not None and not isinstance(encoded, str):
                raise TypeError(f"{name} must be base64 text")
            value[name] = None if encoded is None else b64decode(encoded, validate=True)
        return cls(**value)


def _optional_digest(content: bytes | None) -> str:
    return "" if content is None else sha256(content).hexdigest()


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: OligoformerExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: OligoformerExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> OligoformerExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return OligoformerExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


class OligoformerPublications:
    """Own OligoFormer cache claims and publication reconstruction."""

    def __init__(
        self,
        *,
        request: OligoformerExecutionRequest,
        execution_run_id: UUID,
        output_root: str | Path,
        output_volume_name: str,
        model_volume: Any,
        output_claims: Any,
    ) -> None:
        """Bind one execution Run to app-owned caches and output claims."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_root = Path(output_root)
        self.output_volume_name = output_volume_name
        self.model_volume = model_volume
        self.output_claims = output_claims
        self._claimed: set[str] = set()

    def claim_reference(self, plan: Any) -> None:
        """Claim the shared full-human reference cache generation."""
        identity = plan.reference_identity
        if identity is None:
            raise ValueError("OligoFormer reference identity is unavailable")
        self._claim(f"oligoformer-reference-cache:{identity}")

    def claim_evidence(self, plan: Any, stem: str) -> None:
        """Claim one run/stem off-target evidence publication."""
        identity = _workload_module()._off_target_evidence_identity(
            plan.run_root,
            stem,
        )
        self._claim(f"oligoformer-evidence:{identity}")

    def _claim(self, claim_key: str) -> None:
        if claim_key in self._claimed:
            return
        acquire_output_claim(
            self.output_claims,
            claim_key=claim_key,
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._claimed.add(claim_key)

    def result(self, plan: Any | None = None) -> AppRunResult | None:
        """Reconstruct the exact published standalone result archive."""
        app = _workload_module()
        try:
            expected_identities = (
                (
                    plan.model_identity
                    if plan is not None and plan.model_identity is not None
                    else app._oligoformer_model_volume_identity_digest()
                ),
                (
                    plan.reference_identity
                    if plan is not None
                    else (
                        app._oligoformer_reference_volume_identity_digest()
                        if self.request.off_target and self.request.all_human
                        else None
                    )
                ),
            )
        except FileNotFoundError:
            return None
        publication = app._oligoformer_result_publication(
            self.output_root,
            self.request.execution_plan.workload_plan_fingerprint,
            expected_identities=expected_identities,
        )
        if publication is None:
            return None
        relative = cast(str, publication["result_path"])
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="oligoformer-result",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=relative,
                    ),
                    metadata={
                        "files": [
                            ArtifactFile(
                                path=Path(relative).name,
                                size_bytes=cast(int, publication["size_bytes"]),
                                content_sha256=cast(str, publication["sha256"]),
                            ).model_dump(mode="json")
                        ]
                    },
                )
            ],
        )


class _OligoformerProviderNode(ProviderNode):
    def refresh_artifact_storage_before_result(self) -> bool:
        return True


class _OligoformerTaskNode(TaskProviderNode):
    def refresh_artifact_storage_before_result(self) -> bool:
        return True


@dataclass
class _DownloadNode(_OligoformerProviderNode):
    publications: OligoformerPublications

    def refresh_result_storage(self) -> None:
        self.publications.model_volume.reload()

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return _call("download_oligoformer_models", kwargs={"force": False})

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        if result is not None or not _workload_module()._oligoformer_models_ready():
            raise FileNotFoundError("OligoFormer models are unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        del context
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _workload_module()._oligoformer_models_ready()
            else None
        )

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        return _status(_workload_module()._oligoformer_models_ready())


@dataclass
class _PrepareNode(_OligoformerProviderNode):
    request: OligoformerExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return _call(
            "prepare_oligoformer_run",
            kwargs={
                "mrna_fasta_bytes": self.request.mrna_fasta_bytes,
                "sirna_fasta_bytes": self.request.sirna_fasta_bytes,
                "off_target": self.request.off_target,
                "toxicity": self.request.toxicity,
                "all_human": self.request.all_human,
                "utr_bytes": self.request.utr_bytes,
                "orf_bytes": self.request.orf_bytes,
                "top_n": self.request.top_n,
                "functionality_filter": self.request.functionality_filter,
                "pita_threshold": self.request.pita_threshold,
                "targetscan_threshold": self.request.targetscan_threshold,
                "toxicity_threshold": self.request.toxicity_threshold,
                "force": self.request.force,
                "force_generation": self.request.force_generation,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        return _plan_result("run-plan", _run_plan_from_value(result))


@dataclass
class _ReferencePlanNode(_OligoformerProviderNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        plan = _run_plan_from_context(context)
        self.publications.claim_reference(plan)
        return _call(
            "plan_oligoformer_targetscan_rnaplfold_cache",
            kwargs={"force": False, "execution": self.request.execution_config},
            metadata={"run_plan": asdict(plan)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        return _plan_result(
            "reference-plan",
            _reference_plan_from_value(result),
            upstream=(_inline_plan_output("run-plan", metadata.get("run_plan")),),
        )


@dataclass
class _ReferenceShardsNode(_OligoformerTaskNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        self.publications.claim_reference(_run_plan_from_context(context))
        return tuple(
            TaskDefinition(
                task_key=f"{spec.shard_index:05d}",
                scientific_payload={"shard_index": spec.shard_index},
                execution_payload={"spec": asdict(spec)},
            )
            for spec in _reference_plan_from_context(context).shard_specs
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        self.publications.claim_reference(_run_plan_from_context(context))
        spec = _reference_shard_from_task(task)
        return _call(
            "run_oligoformer_targetscan_rnaplfold_shard",
            kwargs={
                "spec": spec,
                "local_workers": (
                    self.request.execution_config.targetscan_rnaplfold_workers
                ),
            },
            metadata={"spec": asdict(spec)},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del task_key
        spec = _reference_shard_from_metadata(metadata)
        if type(result) is not int or not _reference_shard_ready(spec):
            raise FileNotFoundError("OligoFormer reference shard is unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _reference_shard_ready(_reference_shard_from_task(task))
            else None
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, expected_fingerprint, result, artifacts
        return _status(_reference_shard_ready(_reference_shard_from_task(task)))

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        return _republish_inputs(
            context,
            ("run-plan", "reference-plan"),
            results,
            errors,
        )

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        return _status(
            all(
                _reference_shard_ready(spec)
                for spec in _reference_plan_from_context(context).shard_specs
            )
        )


@dataclass
class _ReferenceFinalizeNode(_OligoformerProviderNode):
    publications: OligoformerPublications

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        self.publications.claim_reference(_run_plan_from_context(context))
        return _call(
            "finalize_oligoformer_targetscan_rnaplfold_cache",
            kwargs={"plan": _reference_plan_from_context(context)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        if (
            result is not None
            or not _workload_module()._targetscan_rnaplfold_cache_ready()
        ):
            raise FileNotFoundError("OligoFormer reference cache is unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        del context
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _workload_module()._targetscan_rnaplfold_cache_ready()
            else None
        )

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        return _status(_workload_module()._targetscan_rnaplfold_cache_ready())


@dataclass
class _EfficacyNode(_OligoformerProviderNode):
    request: OligoformerExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        return _call(
            "run_oligoformer_efficacy",
            uses_gpu=True,
            kwargs={
                "plan": _run_plan_from_context(context),
                "functionality_filter": self.request.functionality_filter,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        plan = _run_plan_from_value(result)
        if not plan.efficacy_ready:
            raise FileNotFoundError("OligoFormer efficacy output is unavailable")
        return _plan_result("run-plan", plan)

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        if not context.inputs.get("run-plan"):
            return None
        plan = _refresh_run_plan(_run_plan_from_context(context))
        return _plan_result("run-plan", plan) if plan.efficacy_ready else None

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        return _status(
            _refresh_run_plan(_run_plan_from_context(context)).efficacy_ready
        )


@dataclass
class _EvidencePlanNode(_OligoformerProviderNode):
    request: OligoformerExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        plan = _run_plan_from_context(context)
        return _call(
            "plan_oligoformer_off_target_evidence",
            kwargs={
                "plan": plan,
                "targetscan_ref_shard_size": self.request.targetscan_ref_shard_size,
                "execution": self.request.execution_config,
            },
            metadata={"run_plan": asdict(plan)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        return _plan_result(
            "evidence-plan",
            _evidence_plan_from_value(result),
            upstream=(_inline_plan_output("run-plan", metadata.get("run_plan")),),
        )


@dataclass
class _PitaReferenceNode(_OligoformerTaskNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        return tuple(
            TaskDefinition(
                task_key=stem.stem,
                scientific_payload={"stem": stem.stem},
                execution_payload={"spec": asdict(stem.pita_specs[0])},
            )
            for stem in _evidence_plan_from_context(context).stems
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        spec = _pita_spec_from_task(task)
        self.publications.claim_evidence(_run_plan_from_context(context), spec.stem)
        return _call(
            "prepare_oligoformer_pita_reference",
            kwargs={"spec": spec, "execution": self.request.execution_config},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        return _pita_reference_result(task_key, _pita_reference_from_value(result))

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        return _aggregate_results(
            results,
            errors,
            upstream=tuple(
                republish_execution_artifact(context.single_input(name))
                for name in ("run-plan", "evidence-plan")
            ),
        )


@dataclass
class _PitaCandidatesNode(_OligoformerTaskNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        return tuple(
            TaskDefinition(
                task_key=_pita_task_key(spec),
                scientific_payload={
                    "stem": spec.stem,
                    "record_name": spec.record_name,
                    "record_sequence_sha256": sha256(
                        spec.record_sequence.encode()
                    ).hexdigest(),
                },
                execution_payload={"spec": asdict(spec)},
            )
            for stem in _evidence_plan_from_context(context).stems
            for spec in stem.pita_specs
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        spec = _pita_spec_from_task(task)
        self.publications.claim_evidence(_run_plan_from_context(context), spec.stem)
        return _call(
            "run_oligoformer_pita_candidate",
            kwargs={
                "spec": spec,
                "reference": _pita_reference_for_stem(context, spec.stem),
                "execution": self.request.execution_config,
            },
            metadata={"spec": asdict(spec)},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del task_key
        spec = _pita_spec_from_metadata(metadata)
        _off_target_result_from_value(result)
        if not _workload_module()._pita_candidate_ready(spec):
            raise FileNotFoundError("OligoFormer PITA candidate is unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _workload_module()._pita_candidate_ready(_pita_spec_from_task(task))
            else None
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, expected_fingerprint, result, artifacts
        return _status(
            _workload_module()._pita_candidate_ready(_pita_spec_from_task(task))
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        return _aggregate_results(
            results,
            errors,
            upstream=tuple(
                republish_execution_artifact(context.single_input(name))
                for name in ("run-plan", "evidence-plan")
            ),
        )

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        return _status(
            all(
                _workload_module()._pita_candidate_ready(spec)
                for stem in _evidence_plan_from_context(context).stems
                for spec in stem.pita_specs
            )
        )


@dataclass
class _TargetscanTilesNode(_OligoformerTaskNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        return tuple(
            TaskDefinition(
                task_key=_targetscan_task_key(spec),
                scientific_payload={
                    "stem": spec.stem,
                    "candidate_shard_index": spec.candidate_shard_index,
                    "reference_shard_index": spec.shard_index,
                },
                execution_payload={"spec": asdict(spec)},
            )
            for stem in _evidence_plan_from_context(context).stems
            for spec in stem.targetscan_specs
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        spec = _targetscan_spec_from_task(task)
        self.publications.claim_evidence(_run_plan_from_context(context), spec.stem)
        return _call(
            "run_oligoformer_targetscan_tile",
            kwargs={"spec": spec, "execution": self.request.execution_config},
            metadata={"spec": asdict(spec)},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del task_key
        if not isinstance(result, str) or not _workload_module()._targetscan_tile_ready(
            _targetscan_spec_from_metadata(metadata)
        ):
            raise FileNotFoundError("OligoFormer TargetScan tile is unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _workload_module()._targetscan_tile_ready(
                _targetscan_spec_from_task(task)
            )
            else None
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, expected_fingerprint, result, artifacts
        return _status(
            _workload_module()._targetscan_tile_ready(_targetscan_spec_from_task(task))
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context
        return _aggregate_results(results, errors)

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        return _status(
            all(
                _workload_module()._targetscan_tile_ready(spec)
                for stem in _evidence_plan_from_context(context).stems
                for spec in stem.targetscan_specs
            )
        )


@dataclass
class _EvidenceMergeNode(_OligoformerTaskNode):
    publications: OligoformerPublications

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        return tuple(
            TaskDefinition(
                task_key=stem.stem,
                scientific_payload={"stem": stem.stem},
                execution_payload={"stem_plan": asdict(stem)},
            )
            for stem in _evidence_plan_from_context(context).stems
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        plan = _run_plan_from_context(context)
        stem = cast(str, cast(Mapping[str, Any], task.scientific_payload)["stem"])
        self.publications.claim_evidence(plan, stem)
        return _call(
            "publish_oligoformer_off_target_evidence",
            kwargs={
                "run_root": plan.run_root,
                "stem_plan": _evidence_stem_from_value(
                    cast(Mapping[str, Any], task.execution_payload)["stem_plan"]
                ),
            },
            metadata={"run_root": plan.run_root, "stem": stem},
        )

    def process_remote_task_result(
        self, task_key: str, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        if result is not None:
            raise ValueError("OligoFormer evidence merge returned a value")
        return cast(
            AppRunResult,
            _evidence_result(
                cast(str, metadata.get("run_root")),
                cast(str, metadata.get("stem")),
            ),
        )

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del expected_fingerprint
        return _evidence_result(
            _run_plan_from_context(context).run_root,
            task.task_key,
            required=False,
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
        return _status(
            _evidence_result(
                _run_plan_from_context(context).run_root,
                task.task_key,
                required=False,
            )
            is not None
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        return _republish_inputs(context, ("run-plan",), results, errors)

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        plan = _run_plan_from_context(context)
        return _status(
            all(
                _evidence_result(plan.run_root, stem, required=False) is not None
                for stem in plan.output_stems
            )
        )


@dataclass
class _FinalNode(_OligoformerProviderNode):
    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        return _call(
            "build_oligoformer_final_tables",
            kwargs={"plan": _run_plan_from_context(context)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        plan = _run_plan_from_value(result)
        if not plan.final_ready:
            raise FileNotFoundError("OligoFormer final tables are unavailable")
        return _plan_result("run-plan", plan)

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        if not context.inputs.get("run-plan"):
            return None
        plan = _refresh_run_plan(_run_plan_from_context(context))
        return _plan_result("run-plan", plan) if plan.final_ready else None

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        return _status(_refresh_run_plan(_run_plan_from_context(context)).final_ready)


@dataclass
class _PublishNode(_OligoformerProviderNode):
    request: OligoformerExecutionRequest
    publications: OligoformerPublications

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        plan = _run_plan_from_context(context)
        return _call(
            "publish_oligoformer_outputs",
            kwargs={
                "plan": plan,
                "publication_key": (
                    self.request.execution_plan.workload_plan_fingerprint
                ),
            },
            metadata={"run_plan": asdict(plan)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        if not isinstance(result, Mapping) or "result_path" not in result:
            raise ValueError("OligoFormer result publication is invalid")
        publication = self.publications.result(
            _run_plan_from_value(metadata.get("run_plan"))
        )
        if publication is None:
            raise FileNotFoundError("OligoFormer result publication is unavailable")
        return publication

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        plan = (
            _run_plan_from_context(context) if context.inputs.get("run-plan") else None
        )
        return self.publications.result(plan)

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        plan = (
            _run_plan_from_context(context) if context.inputs.get("run-plan") else None
        )
        return _status(self.publications.result(plan) is not None)


def _pita_task_key(spec: Any) -> str:
    return f"{spec.stem}:{spec.index:05d}"


def _targetscan_task_key(spec: Any) -> str:
    return (
        f"{spec.stem}:candidate-{spec.candidate_shard_index:05d}:"
        f"reference-{spec.shard_index:05d}"
    )


def _run_plan_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer run plan is invalid")
    value = cast(Mapping[str, Any], value)
    config = value.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("OligoFormer run configuration is invalid")
    parsed = dict(value)
    parsed["config"] = app.OligoformerRunConfig(**dict(config))
    parsed["output_stems"] = tuple(parsed["output_stems"])
    return app.OligoformerRunPlan(**parsed)


def _reference_plan_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer reference plan is invalid")
    value = cast(Mapping[str, Any], value)
    if not isinstance(value.get("shard_specs"), (list, tuple)):
        raise TypeError("OligoFormer reference plan is invalid")
    shard_specs = cast(list[dict[str, Any]], value["shard_specs"])
    return app.OligoformerReferencePlan(
        record_count=value["record_count"],
        shard_specs=tuple(
            app.TargetscanRnaPlfoldShardSpec(**spec) for spec in shard_specs
        ),
    )


def _evidence_stem_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, dict):
        raise TypeError("OligoFormer evidence stem is invalid")
    value = cast(dict[str, Any], value)
    pita_specs = value.get("pita_specs")
    targetscan_specs = value.get("targetscan_specs")
    stem = value.get("stem")
    if (
        not isinstance(stem, str)
        or not isinstance(pita_specs, (list, tuple))
        or not isinstance(targetscan_specs, (list, tuple))
    ):
        raise TypeError("OligoFormer evidence stem fields are invalid")
    return app.OligoformerEvidenceStemPlan(
        stem=stem,
        pita_specs=tuple(
            app.OffTargetShardSpec(**cast(dict[str, Any], spec)) for spec in pita_specs
        ),
        targetscan_specs=tuple(
            app.TargetscanBatchSpec(**cast(dict[str, Any], spec))
            for spec in targetscan_specs
        ),
    )


def _evidence_plan_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer evidence plan is invalid")
    value = cast(Mapping[str, Any], value)
    if not isinstance(value.get("stems"), (list, tuple)):
        raise TypeError("OligoFormer evidence plan is invalid")
    stems = cast(list[object], value["stems"])
    return app.OligoformerEvidencePlan(
        stems=tuple(_evidence_stem_from_value(stem) for stem in stems)
    )


def _pita_reference_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer PITA reference is invalid")
    value = cast(Mapping[str, Any], value)
    utr_shard_paths = value.get("utr_shard_paths")
    ext_utr_path = value.get("ext_utr_path")
    if not isinstance(utr_shard_paths, (list, tuple)) or not isinstance(
        ext_utr_path, str
    ):
        raise TypeError("OligoFormer PITA reference fields are invalid")
    return app.PitaReferencePlan(
        utr_shard_paths=tuple(cast(list[str], utr_shard_paths)),
        ext_utr_path=ext_utr_path,
    )


def _off_target_result_from_value(value: object):
    app = _workload_module()
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer PITA result is invalid")
    value = cast(Mapping[str, Any], value)
    index = value.get("index")
    pita_path = value.get("pita_path")
    if type(index) is not int or not isinstance(pita_path, str):
        raise TypeError("OligoFormer PITA result fields are invalid")
    return app.OffTargetShardResult(index=index, pita_path=pita_path)


def _run_plan_from_context(context: NodeRunContext):
    return _run_plan_from_value(orjson.loads(context.read_input_bytes("run-plan")))


def _reference_plan_from_context(context: NodeRunContext):
    return _reference_plan_from_value(
        orjson.loads(context.read_input_bytes("reference-plan"))
    )


def _evidence_plan_from_context(context: NodeRunContext):
    return _evidence_plan_from_value(
        orjson.loads(context.read_input_bytes("evidence-plan"))
    )


def _inline_plan_output(name: str, value: object) -> AppOutput:
    if not isinstance(value, Mapping):
        raise TypeError(f"OligoFormer {name} metadata is invalid")
    return AppOutput(
        name=name,
        kind=ArtifactKind.TABLE,
        storage=InlineBytes(
            data=orjson.dumps(value, option=orjson.OPT_SORT_KEYS),
            filename=f"{name}.json",
            media_type="application/json",
        ),
    )


def _plan_result(
    name: str,
    value: Any,
    *,
    upstream: tuple[AppOutput, ...] = (),
) -> AppRunResult:
    result = inline_json_result(
        name=name,
        value=asdict(value),
        filename=f"{name}.json",
    )
    return result.model_copy(update={"outputs": [*upstream, *result.outputs]})


def _call(
    function_name: str,
    *,
    uses_gpu: bool = False,
    kwargs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> ProviderCallSpec:
    return ProviderCallSpec(
        function_name=function_name,
        uses_gpu=uses_gpu,
        runtime_image_key="oligoformer-gpu" if uses_gpu else "oligoformer-cpu",
        kwargs=kwargs,
        metadata=metadata or {},
    )


def _task_payload(task: TaskDefinition) -> Mapping[str, Any]:
    if not isinstance(task.execution_payload, Mapping):
        raise TypeError("OligoFormer Task payload is invalid")
    return cast(Mapping[str, Any], task.execution_payload)


def _reference_shard_from_task(task: TaskDefinition):
    return _workload_module().TargetscanRnaPlfoldShardSpec(
        **cast(dict[str, Any], _task_payload(task)["spec"])
    )


def _reference_shard_from_metadata(metadata: Mapping[str, Any]):
    value = metadata.get("spec")
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer reference shard metadata is invalid")
    return _workload_module().TargetscanRnaPlfoldShardSpec(**dict(value))


def _reference_shard_ready(spec: Any) -> bool:
    return bool(
        _workload_module()._targetscan_rnaplfold_shard_state(
            spec,
            verify_output_hashes=False,
        )[0]
    )


def _pita_spec_from_task(task: TaskDefinition):
    return _workload_module().OffTargetShardSpec(
        **cast(dict[str, Any], _task_payload(task)["spec"])
    )


def _pita_spec_from_metadata(metadata: Mapping[str, Any]):
    value = metadata.get("spec")
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer PITA Task metadata is invalid")
    return _workload_module().OffTargetShardSpec(**dict(value))


def _targetscan_spec_from_task(task: TaskDefinition):
    return _workload_module().TargetscanBatchSpec(
        **cast(dict[str, Any], _task_payload(task)["spec"])
    )


def _targetscan_spec_from_metadata(metadata: Mapping[str, Any]):
    value = metadata.get("spec")
    if not isinstance(value, Mapping):
        raise TypeError("OligoFormer TargetScan Task metadata is invalid")
    return _workload_module().TargetscanBatchSpec(**dict(value))


def _pita_reference_result(stem: str, reference: Any) -> AppRunResult:
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="pita-reference",
                kind=ArtifactKind.TABLE,
                storage=InlineBytes(
                    data=orjson.dumps(asdict(reference), option=orjson.OPT_SORT_KEYS),
                    filename="pita-reference.json",
                    media_type="application/json",
                ),
                metadata={"stem": stem},
            )
        ],
    )


def _pita_reference_for_stem(context: NodeRunContext, stem: str):
    matches = {
        artifact.storage.path: artifact
        for artifact in context.inputs.get("pita-references", [])
        if artifact.metadata.get("stem") == stem
    }
    if len(matches) != 1:
        raise ValueError(f"OligoFormer requires one PITA reference for {stem!r}")
    artifact = next(iter(matches.values()))
    return _pita_reference_from_value(
        orjson.loads(context.resolve_artifact(artifact).read_bytes())
    )


def _aggregate_results(
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
    *,
    upstream: tuple[AppOutput, ...] = (),
) -> AppRunResult:
    del results
    if errors:
        return AppRunResult(
            status=AppRunStatus.FAILED,
            warnings=list(errors.values()),
        )
    return AppRunResult(status=AppRunStatus.SUCCEEDED, outputs=list(upstream))


def _republish_inputs(
    context: NodeRunContext,
    names: tuple[str, ...],
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
) -> AppRunResult:
    return _aggregate_results(
        results,
        errors,
        upstream=tuple(
            republish_execution_artifact(context.single_input(name)) for name in names
        ),
    )


def _refresh_run_plan(plan: Any):
    return _workload_module()._build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )


def _evidence_result(
    run_root: str,
    stem: str,
    *,
    required: bool = True,
) -> AppRunResult | None:
    app = _workload_module()
    evidence_dir = (
        app.AppRunLayout.from_run_root(run_root).prep_dir / "off_target" / stem
    )
    available = app._raw_off_target_ready(
        evidence_dir,
        expected_identity=app._off_target_evidence_identity(run_root, stem),
    )
    if not available:
        if required:
            raise FileNotFoundError(
                f"OligoFormer off-target evidence is unavailable: {stem}"
            )
        return None
    return AppRunResult(status=AppRunStatus.SUCCEEDED)


def _status(available: bool) -> AvailabilityStatus:
    return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING


def _workload_module():
    """Import workload-owned publications after Modal app loading."""
    from biomodals.app.score import oligoformer_app

    return oligoformer_app


def oligoformer_execution_graph(
    request: OligoformerExecutionRequest,
    publications: OligoformerPublications,
) -> ExecutionGraph:
    """Build OligoFormer's cache, GPU, evidence, and publication graph."""
    graph = ExecutionGraph(
        "oligoformer",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="oligoformer",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    download = graph.add_node(_DownloadNode(publications), id=DOWNLOAD_NODE)
    prepare = graph.add_node(
        _PrepareNode(request),
        id=PREPARE_NODE,
        depends_on=[download],
    )
    efficacy = graph.add_node(
        _EfficacyNode(request),
        id=EFFICACY_NODE,
        inputs={"run-plan": prepare.outputs(kind=ArtifactKind.TABLE)},
    )
    reference_final = None
    if request.off_target and request.all_human:
        reference_plan = graph.add_node(
            _ReferencePlanNode(request, publications),
            id=REFERENCE_PLAN_NODE,
            inputs={"run-plan": prepare.outputs(kind=ArtifactKind.TABLE)},
        )
        reference_shards = graph.add_node(
            _ReferenceShardsNode(request, publications),
            id=REFERENCE_SHARDS_NODE,
            inputs={
                "run-plan": reference_plan.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="run-plan.json",
                ),
                "reference-plan": reference_plan.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="reference-plan.json",
                ),
            },
        )
        reference_final = graph.add_node(
            _ReferenceFinalizeNode(publications),
            id=REFERENCE_FINALIZE_NODE,
            inputs={
                "run-plan": reference_shards.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="run-plan.json",
                ),
                "reference-plan": reference_shards.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="reference-plan.json",
                ),
            },
        )
    if request.off_target:
        evidence_plan = graph.add_node(
            _EvidencePlanNode(request),
            id=EVIDENCE_PLAN_NODE,
            inputs={"run-plan": efficacy.outputs(kind=ArtifactKind.TABLE)},
            depends_on=[] if reference_final is None else [reference_final],
        )
        plan_inputs = {
            "run-plan": evidence_plan.outputs(
                kind=ArtifactKind.TABLE,
                pattern="run-plan.json",
            ),
            "evidence-plan": evidence_plan.outputs(
                kind=ArtifactKind.TABLE,
                pattern="evidence-plan.json",
            ),
        }
        pita_reference = graph.add_node(
            _PitaReferenceNode(request, publications),
            id=PITA_REFERENCE_NODE,
            inputs=plan_inputs,
        )
        pita_candidates = graph.add_node(
            _PitaCandidatesNode(request, publications),
            id=PITA_CANDIDATES_NODE,
            inputs={
                "run-plan": pita_reference.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="run-plan.json",
                ),
                "evidence-plan": pita_reference.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="evidence-plan.json",
                ),
                "pita-references": pita_reference.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="pita-reference.json",
                ),
            },
        )
        targetscan = graph.add_node(
            _TargetscanTilesNode(request, publications),
            id=TARGETSCAN_TILES_NODE,
            inputs=plan_inputs,
        )
        evidence_merge = graph.add_node(
            _EvidenceMergeNode(publications),
            id=EVIDENCE_MERGE_NODE,
            inputs={
                "run-plan": pita_candidates.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="run-plan.json",
                ),
                "evidence-plan": pita_candidates.outputs(
                    kind=ArtifactKind.TABLE,
                    pattern="evidence-plan.json",
                ),
            },
            depends_on=[targetscan],
        )
        final_inputs = {
            "run-plan": evidence_merge.outputs(
                kind=ArtifactKind.TABLE,
                pattern="run-plan.json",
            )
        }
    else:
        final_inputs = {"run-plan": efficacy.outputs(kind=ArtifactKind.TABLE)}
    final = graph.add_node(_FinalNode(), id=FINAL_NODE, inputs=final_inputs)
    graph.add_node(
        _PublishNode(request, publications),
        id=PUBLISH_NODE,
        inputs={"run-plan": final.outputs(kind=ArtifactKind.TABLE)},
    )
    return graph


class OligoformerExecutionCoordinator(
    OutputClaimExecutionDefinitionCoordinatorLifecycle
):
    """Bind one run-scoped writer to OligoFormer publications."""

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
        model_volume: Any,
        output_claims: Any,
        provider_driver: Any,
        app_version: str,
        model_version: str,
        reference_version: str,
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
                "oligoformer": app_version,
                "oligoformer.model": model_version,
                "oligoformer.reference": reference_version,
            },
            poll_interval_seconds=poll_interval_seconds,
        )
        self.output_volume_name = output_volume_name
        self.model_volume = model_volume
        self.output_claims = output_claims

    def _graph(
        self,
        request: OligoformerExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return oligoformer_execution_graph(
            request=request,
            publications=OligoformerPublications(
                request=request,
                execution_run_id=self.execution_run_id,
                output_root=self.volume_root,
                output_volume_name=self.output_volume_name,
                model_volume=self.model_volume,
                output_claims=self.output_claims,
            ),
        )
