"""Direct OligoFormer adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from base64 import b64decode, b64encode
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    ExecutionRunNotFoundError,
    ExecutionRuntime,
    ExecutionSnapshot,
    NodeDependency,
    NodePlan,
    ProviderBinding,
    ProviderCallStatus,
    RunStatus,
    TaskPlan,
    drive_execution_run,
    resume_execution_run,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
    ExecutionVolumeSync,
)
from biomodals.helper.output_claim import (
    acquire_output_claim,
    register_output_claim_successor,
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

    @property
    def max_active_provider_calls(self) -> int:
        """Return the largest configured container pool needed by any stage."""
        return max(
            2,
            self.off_target_process_slots,
            self.targetscan_rnaplfold_nodes,
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


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> OligoformerExecutionRequest:
    """Load one request through Modal's Volume API."""
    return OligoformerExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class OligoformerExecutionRuntime:
    """Drive deterministic OligoFormer scientific tiles through the kernel."""

    def __init__(
        self,
        *,
        request: OligoformerExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        model_volume: Any,
        output_claims: Any,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to OligoFormer's publication volumes."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.model_volume = model_volume
        self.output_claims = output_claims
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._claimed_publications: set[str] = set()
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            commit_local=store.commit,
            transaction=store.transaction,
        )

    def run(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Create or recover the Run and drive it until it stops."""
        with synchronize():
            repository = self._initialize()
        return drive_execution_run(
            repository,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def resume(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        with synchronize():
            repository = self._initialize()
            resume_execution_run(
                repository,
                self.execution_run_id,
                reconcile_once=self.advance_once,
                checkpoint=self._checkpoint,
                now=self._now(),
            )
        return drive_execution_run(
            self.store.execution,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation while retaining uncertain call ownership."""
        self._provider.repository = self.store.execution
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        """Close SQLite without cancelling attached Provider Calls."""
        self.store.close()

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._recover_publications()
        self._reconcile_nodes_and_run()
        run = self.store.execution.get_run(self.execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            self._reconcile_provider_calls(set(run.plan.node_keys))
            self._decode_completed_calls()
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status == RunStatus.STATE_UNKNOWN:
            required = self._required_nodes()
            required_nodes = set(run.plan.node_keys if required is None else required)
            if required is not None:
                self._prune_unrequired(required)
            self._reconcile_provider_calls(required_nodes)
            self._decode_completed_calls()
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self._required_nodes()
        if required is None:
            return
        self._prune_unrequired(required)
        self._reconcile_provider_calls(set(required))
        self._decode_completed_calls()
        self._recover_publications()
        self._reconcile_nodes_and_run()
        if self.store.execution.get_run(self.execution_run_id).status not in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
        }:
            return
        self._start_ready_nodes(set(required))
        self._recover_publications()
        required = self._required_nodes()
        if required is not None:
            self._admit_remote_tasks(set(required))
        self._reconcile_nodes_and_run()

    def _initialize(self):
        self._reload_volumes()
        repository = self.store.execution
        plan = self.request.execution_plan
        try:
            existing = repository.get_run(self.execution_run_id)
        except LookupError:
            with self.store.transaction():
                repository.create_run(
                    execution_run_id=self.execution_run_id,
                    predecessor_execution_run_id=self.predecessor_execution_run_id,
                    plan=plan,
                    deployment=self.deployment,
                    max_active_provider_calls=self.request.max_active_provider_calls,
                    max_active_gpu_provider_calls=1,
                    now=self._now(),
                )
            return repository
        if (
            existing.plan != plan
            or existing.predecessor_execution_run_id
            != self.predecessor_execution_run_id
            or existing.deployment != self.deployment
            or existing.max_active_provider_calls
            != self.request.max_active_provider_calls
            or existing.max_active_gpu_provider_calls != 1
        ):
            raise ValueError("OligoFormer request does not match Execution Run")
        return repository

    def _required_nodes(self) -> tuple[str, ...] | None:
        self._provider.repository = self.store.execution
        return self._provider.required_node_keys(self.execution_run_id)

    def _prune_unrequired(self, required: tuple[str, ...]) -> tuple[UUID, ...]:
        self._provider.repository = self.store.execution
        return self._provider.prune_unrequired_nodes(
            self.execution_run_id,
            required_node_keys=required,
            now=self._now(),
        )

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        reconciled = self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=_result_envelope,
            now=self._now(),
        )
        if any(
            not original.status.is_terminal and updated.status.is_terminal
            for original, updated in reconciled
        ):
            self._reload_volumes()

    def _recover_publications(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.recover_publications(
            self.execution_run_id,
            observe_node=self._node_observation,
            observe_task=lambda node_key, task: (
                None
                if node_key in self._plan_nodes
                else self._task_observation(node_key, task)
            ),
            now=self._now(),
        )

    @property
    def _plan_nodes(self) -> frozenset[str]:
        return frozenset({PREPARE_NODE, REFERENCE_PLAN_NODE, EVIDENCE_PLAN_NODE})

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        app = _workload_module()
        try:
            if node_key in self._plan_nodes:
                available = False
            elif node_key == DOWNLOAD_NODE:
                available = app._oligoformer_models_ready()
            elif node_key in {REFERENCE_SHARDS_NODE, REFERENCE_FINALIZE_NODE}:
                available = app._targetscan_rnaplfold_cache_ready()
            elif node_key == EFFICACY_NODE:
                plan = self._try_run_plan()
                available = (
                    plan is not None
                    and app._build_plan(
                        plan.cache_key,
                        plan.efficacy_key,
                        plan.output_stems,
                        plan.run_root,
                        config=plan.config,
                        postprocess_key=plan.postprocess_key,
                        reference_identity=plan.reference_identity,
                        model_identity=plan.model_identity,
                    ).efficacy_ready
                )
            elif node_key in {PITA_REFERENCE_NODE}:
                available = False
            elif node_key in {PITA_CANDIDATES_NODE, TARGETSCAN_TILES_NODE}:
                tasks = self._planned_task_records(node_key)
                available = tasks is not None and all(
                    self._task_observation(node_key, task)
                    == AvailabilityStatus.AVAILABLE
                    for task in tasks
                )
            elif node_key == EVIDENCE_MERGE_NODE:
                plan = self._try_run_plan()
                available = (
                    plan is not None
                    and plan.config.off_target
                    and all(
                        app._raw_off_target_ready(
                            app.AppRunLayout.from_run_root(plan.run_root).prep_dir
                            / "off_target"
                            / stem,
                            expected_identity=app._off_target_evidence_identity(
                                plan.run_root,
                                stem,
                            ),
                        )
                        for stem in plan.output_stems
                    )
                )
            elif node_key == FINAL_NODE:
                plan = self._try_run_plan()
                available = (
                    plan is not None
                    and app._build_plan(
                        plan.cache_key,
                        plan.efficacy_key,
                        plan.output_stems,
                        plan.run_root,
                        config=plan.config,
                        postprocess_key=plan.postprocess_key,
                        reference_identity=plan.reference_identity,
                        model_identity=plan.model_identity,
                    ).final_ready
                )
            elif node_key == PUBLISH_NODE:
                plan = self._try_run_plan()
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
                    available = False
                else:
                    available = (
                        app._oligoformer_result_publication(
                            self.store.volume_root,
                            self.request.execution_plan.workload_plan_fingerprint,
                            expected_identities=expected_identities,
                        )
                        is not None
                    )
            else:
                raise ValueError(f"Unknown OligoFormer Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _planned_task_records(self, node_key: str) -> tuple[Any, ...] | None:
        node = self.store.execution.get_node(self.execution_run_id, node_key)
        if not node.discovery_complete:
            return None
        return self.store.execution.list_tasks(self.execution_run_id, node_key)

    def _task_observation(self, node_key: str, task: Any) -> AvailabilityStatus:
        app = _workload_module()
        try:
            if node_key == REFERENCE_SHARDS_NODE:
                spec = app.TargetscanRnaPlfoldShardSpec(
                    **task.execution_payload["spec"]
                )
                available = app._targetscan_rnaplfold_shard_state(
                    spec,
                    verify_output_hashes=False,
                )[0]
            elif node_key == PITA_CANDIDATES_NODE:
                spec = app.OffTargetShardSpec(**task.execution_payload["spec"])
                available = app._pita_candidate_ready(spec)
            elif node_key == TARGETSCAN_TILES_NODE:
                spec = app.TargetscanBatchSpec(**task.execution_payload["spec"])
                available = app._targetscan_tile_ready(spec)
            else:
                return self._node_observation(node_key)
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _decode_completed_calls(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.decode_completed_calls(
            self.execution_run_id,
            observe_task=self._completed_task_observation,
            missing_message="OligoFormer returned without a valid publication",
            now=self._now(),
        )

    def _completed_task_observation(
        self,
        node_key: str,
        task: Any,
        envelope: object,
    ) -> AvailabilityStatus:
        if not isinstance(envelope, dict):
            return AvailabilityStatus.MISSING
        try:
            if node_key in {PREPARE_NODE, EFFICACY_NODE, FINAL_NODE}:
                _run_plan_from_envelope(envelope)
            elif node_key == REFERENCE_PLAN_NODE:
                _reference_plan_from_envelope(envelope)
            elif node_key == EVIDENCE_PLAN_NODE:
                _evidence_plan_from_envelope(envelope)
            elif node_key == PITA_REFERENCE_NODE:
                _pita_reference_from_envelope(envelope)
            elif node_key == PUBLISH_NODE:
                if envelope.get("kind") != "result":
                    return AvailabilityStatus.MISSING
            elif envelope.get("kind") not in {
                "none",
                "count",
                "pita-result",
                "path",
            }:
                return AvailabilityStatus.MISSING
        except (TypeError, ValueError):
            return AvailabilityStatus.MISSING
        if node_key in self._plan_nodes or node_key == PITA_REFERENCE_NODE:
            return AvailabilityStatus.AVAILABLE
        return self._task_observation(node_key, task)

    def _start_ready_nodes(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        self._provider.start_ready_nodes(
            self.execution_run_id,
            required_node_keys=required,
            task_plans=self._task_plans,
            observe_task=lambda node_key, task: (
                AvailabilityStatus.MISSING
                if node_key in self._plan_nodes or node_key == PITA_REFERENCE_NODE
                else self._task_observation(node_key, task)
            ),
            now=self._now(),
        )

    def _task_plans(self, node_key: str) -> tuple[TaskPlan, ...]:
        if node_key in {
            DOWNLOAD_NODE,
            PREPARE_NODE,
            REFERENCE_PLAN_NODE,
            REFERENCE_FINALIZE_NODE,
            EFFICACY_NODE,
            EVIDENCE_PLAN_NODE,
            FINAL_NODE,
            PUBLISH_NODE,
        }:
            return (TaskPlan(node_key, {"stage": node_key}),)
        if node_key == REFERENCE_SHARDS_NODE:
            return tuple(
                TaskPlan(
                    f"{spec.shard_index:05d}",
                    {"shard_index": spec.shard_index},
                    {"spec": asdict(spec)},
                )
                for spec in self._reference_plan().shard_specs
            )
        evidence = self._evidence_plan()
        if node_key == PITA_REFERENCE_NODE:
            return tuple(
                TaskPlan(
                    stem.stem,
                    {"stem": stem.stem},
                    {"spec": asdict(stem.pita_specs[0])},
                )
                for stem in evidence.stems
            )
        if node_key == PITA_CANDIDATES_NODE:
            return tuple(
                TaskPlan(
                    _pita_task_key(spec),
                    {
                        "stem": spec.stem,
                        "record_name": spec.record_name,
                        "record_sequence_sha256": sha256(
                            spec.record_sequence.encode()
                        ).hexdigest(),
                    },
                    {"spec": asdict(spec)},
                )
                for stem in evidence.stems
                for spec in stem.pita_specs
            )
        if node_key == TARGETSCAN_TILES_NODE:
            return tuple(
                TaskPlan(
                    _targetscan_task_key(spec),
                    {
                        "stem": spec.stem,
                        "candidate_shard_index": spec.candidate_shard_index,
                        "reference_shard_index": spec.shard_index,
                    },
                    {"spec": asdict(spec)},
                )
                for stem in evidence.stems
                for spec in stem.targetscan_specs
            )
        if node_key == EVIDENCE_MERGE_NODE:
            return tuple(
                TaskPlan(
                    stem.stem,
                    {"stem": stem.stem},
                    {"stem_plan": asdict(stem)},
                )
                for stem in evidence.stems
            )
        raise ValueError(f"Unknown OligoFormer Node {node_key!r}")

    def _try_run_plan(self):
        try:
            return self._run_plan()
        except (LookupError, TypeError, ValueError):
            return None

    def _run_plan(self):
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == PREPARE_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return _run_plan_from_envelope(call.result_envelope)
        raise LookupError("OligoFormer run plan is unavailable")

    def _reference_plan(self):
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == REFERENCE_PLAN_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return _reference_plan_from_envelope(call.result_envelope)
        raise LookupError("OligoFormer reference plan is unavailable")

    def _evidence_plan(self):
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == EVIDENCE_PLAN_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return _evidence_plan_from_envelope(call.result_envelope)
        raise LookupError("OligoFormer evidence plan is unavailable")

    def _pita_reference(self, stem: str):
        repository = self.store.execution
        for call in repository.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == PITA_REFERENCE_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
                and stem in call.task_keys
            ):
                return _pita_reference_from_envelope(call.result_envelope)
        raise LookupError(f"OligoFormer PITA reference plan is unavailable: {stem}")

    def _reconcile_nodes_and_run(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.reconcile_nodes_and_run(
            self.execution_run_id,
            now=self._now(),
        )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        self._provider.repository = repository
        counts = repository.active_provider_call_counts(self.execution_run_id)
        ordered = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=lambda node, task, rank: TaskDispatchDescriptor(
                node_key=node.node_key,
                node_ordinal=node.ordinal,
                task_key=task.task_key,
                task_ordinal=task.ordinal,
                binding=self._binding(node.node_key),
                compatibility_key=self._binding(node.node_key).function_name,
                max_tasks_per_call=1,
                depth=rank.depth,
                unblocking_span=rank.unblocking_span,
            ),
            available_total_slots=None,
            available_gpu_slots=max(
                0,
                run.max_active_gpu_provider_calls - counts.gpu,
            ),
            now=self._now(),
        )
        available_total = max(0, run.max_active_provider_calls - counts.total)
        active_by_node = Counter(
            call.node_key
            for call in repository.list_provider_calls(self.execution_run_id)
            if not call.status.is_terminal
        )
        selected = []
        for candidate in ordered:
            if len(selected) >= available_total:
                break
            if active_by_node[candidate.node_key] >= self._node_call_limit(
                candidate.node_key
            ):
                continue
            selected.append(candidate)
            active_by_node[candidate.node_key] += 1
        for candidate in selected:
            self._ensure_publication_claim(
                candidate.node_key,
                candidate.task_keys[0],
            )
            self._provider.repository = self.store.execution
            submitted = self._provider.submit_fixed_batch(
                self.execution_run_id,
                candidate,
                submission_token=candidate.candidate_key,
                kwargs=self._invocation_kwargs(
                    candidate.node_key,
                    candidate.task_keys[0],
                ),
                now=self._now(),
            )
            if submitted is None:
                return

    def _node_call_limit(self, node_key: str) -> int:
        execution = self.request.execution_config
        targetscan_slots, pita_slots = _workload_module()._off_target_branch_slots(
            execution
        )
        if node_key == REFERENCE_SHARDS_NODE:
            return execution.targetscan_rnaplfold_nodes
        if node_key == PITA_CANDIDATES_NODE:
            return min(
                execution.off_target_nodes,
                execution.pita_prepare_nodes,
                pita_slots,
            )
        if node_key == TARGETSCAN_TILES_NODE:
            return min(
                execution.targetscan_prepare_nodes,
                execution.targetscan_context_nodes,
                targetscan_slots,
            )
        if node_key == EVIDENCE_MERGE_NODE:
            return execution.targetscan_merge_nodes
        return 1

    def _binding(self, node_key: str) -> ProviderBinding:
        function_name = {
            DOWNLOAD_NODE: "download_oligoformer_models",
            PREPARE_NODE: "prepare_oligoformer_run",
            REFERENCE_PLAN_NODE: "plan_oligoformer_targetscan_rnaplfold_cache",
            REFERENCE_SHARDS_NODE: "run_oligoformer_targetscan_rnaplfold_shard",
            REFERENCE_FINALIZE_NODE: "finalize_oligoformer_targetscan_rnaplfold_cache",
            EFFICACY_NODE: "run_oligoformer_efficacy",
            EVIDENCE_PLAN_NODE: "plan_oligoformer_off_target_evidence",
            PITA_REFERENCE_NODE: "prepare_oligoformer_pita_reference",
            PITA_CANDIDATES_NODE: "run_oligoformer_pita_candidate",
            TARGETSCAN_TILES_NODE: "run_oligoformer_targetscan_tile",
            EVIDENCE_MERGE_NODE: "publish_oligoformer_off_target_evidence",
            FINAL_NODE: "build_oligoformer_final_tables",
            PUBLISH_NODE: "publish_oligoformer_outputs",
        }[node_key]
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=function_name,
            uses_gpu=node_key == EFFICACY_NODE,
            runtime_image_key=(
                "oligoformer-gpu" if node_key == EFFICACY_NODE else "oligoformer-cpu"
            ),
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        app = _workload_module()
        execution = self.request.execution_config
        if node_key == DOWNLOAD_NODE:
            return {"force": False}
        if node_key == PREPARE_NODE:
            return {
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
            }
        if node_key == REFERENCE_PLAN_NODE:
            return {"force": False, "execution": execution}
        if node_key == REFERENCE_SHARDS_NODE:
            task = self._task(node_key, task_key)
            return {
                "spec": app.TargetscanRnaPlfoldShardSpec(
                    **task.execution_payload["spec"]
                ),
                "local_workers": execution.targetscan_rnaplfold_workers,
            }
        if node_key == REFERENCE_FINALIZE_NODE:
            return {"plan": self._reference_plan()}
        if node_key == EFFICACY_NODE:
            return {
                "plan": self._run_plan(),
                "functionality_filter": self.request.functionality_filter,
            }
        if node_key == EVIDENCE_PLAN_NODE:
            return {
                "plan": self._run_plan(),
                "targetscan_ref_shard_size": self.request.targetscan_ref_shard_size,
                "execution": execution,
            }
        if node_key == PITA_REFERENCE_NODE:
            task = self._task(node_key, task_key)
            return {
                "spec": app.OffTargetShardSpec(**task.execution_payload["spec"]),
                "execution": execution,
            }
        if node_key == PITA_CANDIDATES_NODE:
            task = self._task(node_key, task_key)
            spec = app.OffTargetShardSpec(**task.execution_payload["spec"])
            return {
                "spec": spec,
                "reference": self._pita_reference(spec.stem),
                "execution": execution,
            }
        if node_key == TARGETSCAN_TILES_NODE:
            task = self._task(node_key, task_key)
            return {
                "spec": app.TargetscanBatchSpec(**task.execution_payload["spec"]),
                "execution": execution,
            }
        if node_key == EVIDENCE_MERGE_NODE:
            task = self._task(node_key, task_key)
            return {
                "run_root": self._run_plan().run_root,
                "stem_plan": _evidence_stem_from_value(
                    task.execution_payload["stem_plan"]
                ),
            }
        if node_key == FINAL_NODE:
            return {"plan": self._run_plan()}
        if node_key == PUBLISH_NODE:
            return {
                "plan": self._run_plan(),
                "publication_key": (
                    self.request.execution_plan.workload_plan_fingerprint
                ),
            }
        raise ValueError(f"Unknown OligoFormer Node {node_key!r}")

    def _task(self, node_key: str, task_key: str):
        return self.store.execution.get_task(
            self.execution_run_id,
            node_key,
            task_key,
        )

    def _ensure_publication_claim(self, node_key: str, task_key: str) -> None:
        claim_key = None
        if node_key in {
            REFERENCE_PLAN_NODE,
            REFERENCE_SHARDS_NODE,
            REFERENCE_FINALIZE_NODE,
        }:
            identity = self._run_plan().reference_identity
            if identity is None:
                raise ValueError("OligoFormer reference identity is unavailable")
            claim_key = f"oligoformer-reference-cache:{identity}"
        elif node_key in {
            PITA_REFERENCE_NODE,
            PITA_CANDIDATES_NODE,
            TARGETSCAN_TILES_NODE,
            EVIDENCE_MERGE_NODE,
        }:
            stem = str(self._task(node_key, task_key).scientific_payload["stem"])
            claim_key = (
                "oligoformer-evidence:"
                + _workload_module()._off_target_evidence_identity(
                    self._run_plan().run_root,
                    stem,
                )
            )
        if claim_key is None or claim_key in self._claimed_publications:
            return
        acquire_output_claim(
            self.output_claims,
            claim_key=claim_key,
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._claimed_publications.add(claim_key)

    def _checkpoint(self):
        self._volume_sync.commit()
        repository = self.store.execution
        self._provider.repository = repository
        return repository

    def _reload_volumes(self) -> None:
        self._volume_sync.reload()
        self.model_volume.reload()
        self._provider.repository = self.store.execution


def _pita_task_key(spec: Any) -> str:
    return f"{spec.stem}:{spec.index:05d}"


def _targetscan_task_key(spec: Any) -> str:
    return (
        f"{spec.stem}:candidate-{spec.candidate_shard_index:05d}:"
        f"reference-{spec.shard_index:05d}"
    )


def _run_plan_from_envelope(envelope: object):
    app = _workload_module()
    if not isinstance(envelope, dict) or envelope.get("kind") != "run-plan":
        raise TypeError("OligoFormer run-plan envelope is invalid")
    value = envelope.get("plan")
    if not isinstance(value, dict):
        raise TypeError("OligoFormer run plan is invalid")
    value = cast(dict[str, Any], value)
    config = value.get("config")
    if not isinstance(config, dict):
        raise TypeError("OligoFormer run configuration is invalid")
    config = cast(dict[str, Any], config)
    parsed = dict(value)
    parsed["config"] = app.OligoformerRunConfig(**config)
    parsed["output_stems"] = tuple(parsed["output_stems"])
    return app.OligoformerRunPlan(**parsed)


def _reference_plan_from_envelope(envelope: object):
    app = _workload_module()
    if not isinstance(envelope, dict) or envelope.get("kind") != "reference-plan":
        raise TypeError("OligoFormer reference-plan envelope is invalid")
    value = envelope.get("plan")
    if not isinstance(value, dict) or not isinstance(value.get("shard_specs"), list):
        raise TypeError("OligoFormer reference plan is invalid")
    value = cast(dict[str, Any], value)
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


def _evidence_plan_from_envelope(envelope: object):
    app = _workload_module()
    if not isinstance(envelope, dict) or envelope.get("kind") != "evidence-plan":
        raise TypeError("OligoFormer evidence-plan envelope is invalid")
    value = envelope.get("plan")
    if not isinstance(value, dict) or not isinstance(value.get("stems"), list):
        raise TypeError("OligoFormer evidence plan is invalid")
    value = cast(dict[str, Any], value)
    stems = cast(list[object], value["stems"])
    return app.OligoformerEvidencePlan(
        stems=tuple(_evidence_stem_from_value(stem) for stem in stems)
    )


def _pita_reference_from_envelope(envelope: object):
    app = _workload_module()
    if not isinstance(envelope, dict) or envelope.get("kind") != "pita-reference":
        raise TypeError("OligoFormer PITA-reference envelope is invalid")
    value = envelope.get("plan")
    if not isinstance(value, dict):
        raise TypeError("OligoFormer PITA reference is invalid")
    value = cast(dict[str, Any], value)
    utr_shard_paths = value.get("utr_shard_paths")
    ext_utr_path = value.get("ext_utr_path")
    if not isinstance(utr_shard_paths, list) or not isinstance(ext_utr_path, str):
        raise TypeError("OligoFormer PITA reference fields are invalid")
    return app.PitaReferencePlan(
        utr_shard_paths=tuple(cast(list[str], utr_shard_paths)),
        ext_utr_path=ext_utr_path,
    )


def _result_envelope(result: object) -> dict[str, object]:
    """Encode only bounded plans and publication metadata."""
    app = _workload_module()
    if isinstance(result, app.OligoformerRunPlan):
        return {"kind": "run-plan", "plan": asdict(result)}
    if isinstance(result, app.OligoformerReferencePlan):
        return {"kind": "reference-plan", "plan": asdict(result)}
    if isinstance(result, app.OligoformerEvidencePlan):
        return {"kind": "evidence-plan", "plan": asdict(result)}
    if isinstance(result, app.PitaReferencePlan):
        return {"kind": "pita-reference", "plan": asdict(result)}
    if isinstance(result, app.OffTargetShardResult):
        return {"kind": "pita-result", "result": asdict(result)}
    if result is None:
        return {"kind": "none"}
    if type(result) is int:
        return {"kind": "count", "count": result}
    if isinstance(result, str):
        return {"kind": "path", "path": result}
    if isinstance(result, dict) and "result_path" in result:
        return {"kind": "result", "publication": orjson.loads(orjson.dumps(result))}
    return {"kind": "invalid"}


def _workload_module():
    """Import workload-owned publications after Modal app loading."""
    from biomodals.app.score import oligoformer_app

    return oligoformer_app


class OligoformerExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to OligoFormer publications."""

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        model_volume: Any,
        output_claims: Any,
        modal_driver: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only resources used by this workload adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
        )
        self.output_volume = output_volume
        self.model_volume = model_volume
        self.output_claims = output_claims
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds

    def run(self) -> ExecutionSnapshot:
        """Load the staged request and drive one root Run."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_runtime(
                    load_execution_request(self.volume_root, self.execution_run_id)
                )
            return self._drive(runtime, resume=False)

    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation and reconcile it to a terminal result."""
        with self._writer_lock:
            runtime = self._open_runtime(
                load_execution_request(self.volume_root, self.execution_run_id)
            )
            snapshot = runtime.cancel()
            self._verify_snapshot(snapshot)
        if snapshot.run.status.is_terminal:
            return snapshot
        return self._drive(runtime, resume=False)

    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_runtime(
                    load_execution_request(self.volume_root, self.execution_run_id)
                )
            return self._drive(runtime, resume=True)

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: OligoformerExecutionRequest | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor Run."""
        if candidate_request is not None and (
            max_active_provider_calls is not None
            or max_active_gpu_provider_calls is not None
        ):
            raise ValueError(
                "Candidate request and generic restart overrides are mutually exclusive"
            )
        del max_active_gpu_provider_calls
        if predecessor_execution_run_id == self.execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                predecessor_store = ExecutionRunStore(
                    self.volume_root,
                    predecessor_execution_run_id,
                )
                if not predecessor_store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(predecessor_execution_run_id))
                try:
                    predecessor = predecessor_store.execution.validate_successor_source(
                        predecessor_execution_run_id
                    )
                    if (
                        expected_workload_plan_fingerprint is not None
                        and predecessor.plan.workload_plan_fingerprint
                        != expected_workload_plan_fingerprint
                    ):
                        raise ValueError(
                            "Predecessor workload plan fingerprint changed"
                        )
                    if (
                        predecessor_deployment is not None
                        and predecessor.deployment != predecessor_deployment
                    ):
                        raise ValueError("Predecessor Deployment Identity changed")
                    predecessor_request = load_execution_request(
                        self.volume_root,
                        predecessor_execution_run_id,
                    )
                finally:
                    predecessor_store.close()
                request = candidate_request or predecessor_request
                if (
                    request.execution_plan.workload_plan_fingerprint
                    != predecessor.plan.workload_plan_fingerprint
                ):
                    raise ValueError(
                        "Restart arguments changed the Workload Plan Fingerprint"
                    )
                capacity = (
                    request.max_active_provider_calls
                    if max_active_provider_calls is None
                    else max_active_provider_calls
                )
                if capacity != request.max_active_provider_calls:
                    raise ValueError(
                        "OligoFormer successor capacity is derived from its request"
                    )
                register_output_claim_successor(
                    self.output_claims,
                    owner=str(self.execution_run_id),
                    predecessor=str(predecessor_execution_run_id),
                )
                request = replace(
                    request,
                    replace_claim_owner=str(predecessor_execution_run_id),
                )
                persist_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                    request,
                )
                self.output_volume.commit()
                runtime = self._open_runtime(
                    request,
                    predecessor_execution_run_id=predecessor_execution_run_id,
                )
            return self._drive(runtime, resume=False)

    def _open_runtime(
        self,
        request: OligoformerExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> OligoformerExecutionRuntime:
        self._close_runtime()
        store = self._run_store()
        runtime = OligoformerExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            deployment=self.deployment,
            store=store,
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            model_volume=self.model_volume,
            output_claims=self.output_claims,
            predecessor_execution_run_id=predecessor_execution_run_id,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
