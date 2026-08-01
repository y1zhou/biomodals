"""Direct ABCFold2 adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from base64 import b64decode, b64encode
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
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
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject empty inputs and unusable coordinator capacity."""
        if not self.run_name or not self.yaml_content:
            raise ValueError("ABCFold2 run name and YAML cannot be empty")
        if self.max_active_provider_calls < 1:
            raise ValueError("ABCFold2 provider-call limit must be positive")
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
            scientific_versions={
                "abcfold2": self.app_version,
                "boltz": self.boltz_version,
                "chai": self.chai_version,
                "biomodals.abcfold2.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
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


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> ABCFold2ExecutionRequest:
    """Load one request through Modal's Volume API."""
    return ABCFold2ExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class ABCFold2ExecutionRuntime:
    """Drive one ABCFold2 request through parallel per-seed Tasks."""

    def __init__(
        self,
        *,
        request: ABCFold2ExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        output_claims: Any,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to ABCFold2's established run layout."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.output_claims = output_claims
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._claimed_models: set[str] = set()
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
        self._reload_output()
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
                    max_active_gpu_provider_calls=(
                        self.request.max_active_provider_calls
                    ),
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
            or existing.max_active_gpu_provider_calls
            != self.request.max_active_provider_calls
        ):
            raise ValueError("ABCFold2 request does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.recover_publications(
            self.execution_run_id,
            observe_node=self._node_observation,
            observe_task=lambda node_key, task: (
                None
                if node_key in {PREPARE_NODE, BOLTZ_DOWNLOAD_NODE, CHAI_DOWNLOAD_NODE}
                else self._task_observation(node_key, task.task_key)
            ),
            now=self._now(),
        )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        if node_key in {PREPARE_NODE, BOLTZ_DOWNLOAD_NODE, CHAI_DOWNLOAD_NODE}:
            return AvailabilityStatus.MISSING
        run_conf = self._try_run_config()
        if run_conf is None:
            return AvailabilityStatus.MISSING
        app = _workload_module()
        try:
            if node_key == BOLTZ_SEEDS_NODE:
                publication_key = self._model_publication_key("boltz", run_conf)
                available = all(
                    app._seed_ready(
                        run_conf.workdir,
                        "boltz",
                        seed,
                        publication_key,
                    )
                    for seed in run_conf.seeds
                )
            elif node_key == CHAI_SEEDS_NODE:
                publication_key = self._model_publication_key("chai", run_conf)
                available = all(
                    app._seed_ready(
                        run_conf.workdir,
                        "chai",
                        seed,
                        publication_key,
                    )
                    for seed in run_conf.seeds
                )
            elif node_key == BOLTZ_ARCHIVE_NODE:
                available = app._archive_ready(
                    run_conf.workdir,
                    "boltz",
                    self._model_publication_key("boltz", run_conf),
                )
            elif node_key == CHAI_ARCHIVE_NODE:
                available = app._archive_ready(
                    run_conf.workdir,
                    "chai",
                    self._model_publication_key("chai", run_conf),
                )
            else:
                raise ValueError(f"Unknown ABCFold2 Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _task_observation(
        self,
        node_key: str,
        task_key: str,
    ) -> AvailabilityStatus:
        if node_key in {BOLTZ_SEEDS_NODE, CHAI_SEEDS_NODE}:
            run_conf = self._try_run_config()
            if run_conf is None:
                return AvailabilityStatus.MISSING
            model_name = "boltz" if node_key == BOLTZ_SEEDS_NODE else "chai"
            try:
                available = _workload_module()._seed_ready(
                    run_conf.workdir,
                    model_name,
                    int(task_key),
                    self._model_publication_key(model_name, run_conf),
                )
            except OSError:
                return AvailabilityStatus.UNKNOWN
            return (
                AvailabilityStatus.AVAILABLE
                if available
                else AvailabilityStatus.MISSING
            )
        return self._node_observation(node_key)

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
            self._reload_output()

    def _decode_completed_calls(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.decode_completed_calls(
            self.execution_run_id,
            observe_task=lambda node_key, task, envelope: (
                self._completed_task_observation(
                    node_key,
                    task.task_key,
                    envelope,
                )
            ),
            missing_message="ABCFold2 returned without a valid publication",
            now=self._now(),
        )

    def _completed_task_observation(
        self,
        node_key: str,
        task_key: str,
        envelope: object,
    ) -> AvailabilityStatus:
        if not isinstance(envelope, dict):
            return AvailabilityStatus.MISSING
        if node_key in {BOLTZ_DOWNLOAD_NODE, CHAI_DOWNLOAD_NODE}:
            return (
                AvailabilityStatus.AVAILABLE
                if envelope.get("kind") == "none"
                else AvailabilityStatus.MISSING
            )
        if node_key == PREPARE_NODE:
            try:
                _run_config_from_envelope(envelope)
            except (TypeError, ValueError):
                return AvailabilityStatus.MISSING
            return AvailabilityStatus.AVAILABLE
        expected_kind = (
            "path" if node_key in {BOLTZ_SEEDS_NODE, CHAI_SEEDS_NODE} else "archive"
        )
        if envelope.get("kind") != expected_kind:
            return AvailabilityStatus.MISSING
        return self._task_observation(node_key, task_key)

    def _start_ready_nodes(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        self._provider.start_ready_nodes(
            self.execution_run_id,
            required_node_keys=required,
            task_plans=self._task_plans,
            observe_task=lambda node_key, task: (
                AvailabilityStatus.MISSING
                if node_key in {PREPARE_NODE, BOLTZ_DOWNLOAD_NODE, CHAI_DOWNLOAD_NODE}
                else self._task_observation(node_key, task.task_key)
            ),
            now=self._now(),
        )

    def _task_plans(self, node_key: str) -> tuple[TaskPlan, ...]:
        if node_key == PREPARE_NODE:
            return (
                TaskPlan(
                    "prepare",
                    {"yaml_sha256": sha256(self.request.yaml_content).hexdigest()},
                ),
            )
        if node_key == BOLTZ_DOWNLOAD_NODE:
            return (TaskPlan("boltz-models", {"version": self.request.boltz_version}),)
        if node_key == CHAI_DOWNLOAD_NODE:
            return (TaskPlan("chai-models", {"version": self.request.chai_version}),)
        run_conf = self._run_config()
        if node_key in {BOLTZ_SEEDS_NODE, CHAI_SEEDS_NODE}:
            return tuple(
                TaskPlan(
                    str(seed),
                    {"run_id": run_conf.run_id, "seed": seed},
                )
                for seed in run_conf.seeds
            )
        if node_key == BOLTZ_ARCHIVE_NODE:
            return (TaskPlan("boltz-archive", {"run_id": run_conf.run_id}),)
        if node_key == CHAI_ARCHIVE_NODE:
            return (TaskPlan("chai-archive", {"run_id": run_conf.run_id}),)
        raise ValueError(f"Unknown ABCFold2 Node {node_key!r}")

    def _try_run_config(self) -> ABCFold2RunConfig | None:
        try:
            return self._run_config()
        except (LookupError, TypeError, ValueError):
            return None

    def _run_config(self) -> ABCFold2RunConfig:
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == PREPARE_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return _run_config_from_envelope(call.result_envelope)
        raise LookupError("ABCFold2 preparation result is unavailable")

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
        selected = self._provider.fixed_call_candidates(
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
            available_total_slots=max(0, run.max_active_provider_calls - counts.total),
            available_gpu_slots=max(0, run.max_active_gpu_provider_calls - counts.gpu),
            now=self._now(),
        )
        for candidate in selected:
            self._ensure_publication_claim(candidate.node_key)
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

    def _binding(self, node_key: str) -> ProviderBinding:
        function_name = {
            PREPARE_NODE: "prepare_abcfold2",
            BOLTZ_DOWNLOAD_NODE: "download_boltz_models",
            CHAI_DOWNLOAD_NODE: "download_chai_models",
            BOLTZ_SEEDS_NODE: "run_abcfold2_boltz",
            BOLTZ_ARCHIVE_NODE: "collect_abcfold2_boltz_data",
            CHAI_SEEDS_NODE: "run_abcfold2_chai",
            CHAI_ARCHIVE_NODE: "collect_abcfold2_chai_data",
        }[node_key]
        uses_gpu = node_key in {BOLTZ_SEEDS_NODE, CHAI_SEEDS_NODE}
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=function_name,
            uses_gpu=uses_gpu,
            runtime_image_key=(
                f"abcfold2-{node_key}" if not uses_gpu else "abcfold2-gpu"
            ),
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        if node_key == PREPARE_NODE:
            return {
                "yaml_str": self.request.yaml_content,
                "search_templates": self.request.search_templates,
                "msa_chains": self.request.msa_chains,
            }
        if node_key in {BOLTZ_DOWNLOAD_NODE, CHAI_DOWNLOAD_NODE}:
            return {"force": self.request.force_redownload}
        run_conf = self._run_config()
        if node_key in {BOLTZ_SEEDS_NODE, CHAI_SEEDS_NODE}:
            model_name = "boltz" if node_key == BOLTZ_SEEDS_NODE else "chai"
            return {
                "seed": int(task_key),
                **run_conf.as_kwargs(),
                "publication_key": self._model_publication_key(
                    model_name,
                    run_conf,
                ),
            }
        if node_key in {BOLTZ_ARCHIVE_NODE, CHAI_ARCHIVE_NODE}:
            model_name = "boltz" if node_key == BOLTZ_ARCHIVE_NODE else "chai"
            return {
                "run_conf": run_conf.as_kwargs(),
                "publication_key": self._model_publication_key(
                    model_name,
                    run_conf,
                ),
            }
        raise ValueError(f"Unknown ABCFold2 Node {node_key!r}")

    def _ensure_publication_claim(self, node_key: str) -> None:
        model_name = {
            BOLTZ_SEEDS_NODE: "boltz",
            BOLTZ_ARCHIVE_NODE: "boltz",
            CHAI_SEEDS_NODE: "chai",
            CHAI_ARCHIVE_NODE: "chai",
        }.get(node_key)
        if model_name is None or model_name in self._claimed_models:
            return
        run_conf = self._run_config()
        publication_key = self._model_publication_key(model_name, run_conf)
        acquire_output_claim(
            self.output_claims,
            claim_key=f"abcfold2-{model_name}:{publication_key}",
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._claimed_models.add(model_name)

    @staticmethod
    def _model_publication_key(
        model_name: str,
        run_conf: ABCFold2RunConfig,
    ) -> str:
        return _workload_module()._model_publication_key(
            model_name,
            run_conf.as_kwargs(),
        )

    def _checkpoint(self):
        self._volume_sync.commit()
        repository = self.store.execution
        self._provider.repository = repository
        return repository

    def _reload_output(self) -> None:
        self._volume_sync.reload()
        self._provider.repository = self.store.execution


def _run_config_from_envelope(envelope: object) -> ABCFold2RunConfig:
    if not isinstance(envelope, dict) or envelope.get("kind") != "run-config":
        raise TypeError("ABCFold2 run-config envelope is invalid")
    value = envelope.get("run_config")
    if not isinstance(value, dict):
        raise TypeError("ABCFold2 run config is invalid")
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


def _result_envelope(result: object) -> dict[str, object]:
    """Encode only bounded run configuration or publication metadata."""
    if result is None:
        return {"kind": "none"}
    if isinstance(result, str):
        return {"kind": "path", "path": result}
    if isinstance(result, dict):
        if "archive_path" in result:
            return {
                "kind": "archive",
                "archive": orjson.loads(orjson.dumps(result)),
            }
        return {
            "kind": "run-config",
            "run_config": orjson.loads(orjson.dumps(result)),
        }
    return {"kind": "invalid"}


def run_config_from_snapshot(snapshot: ExecutionSnapshot) -> ABCFold2RunConfig:
    """Return the validated preparation result from a completed snapshot."""
    for call in snapshot.provider_calls:
        if (
            call.node_key == PREPARE_NODE
            and call.status == ProviderCallStatus.SUCCEEDED
        ):
            return _run_config_from_envelope(call.result_envelope)
    raise LookupError("ABCFold2 preparation result is unavailable")


def _workload_module():
    """Import workload-owned publication probes after Modal app loading."""
    from biomodals.app.fold import abcfold2_app

    return abcfold2_app


class ABCFold2ExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to ABCFold2 publications."""

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
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
        candidate_request: ABCFold2ExecutionRequest | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor from conclusive state."""
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
                            "Restart arguments changed the Workload Plan Fingerprint"
                        )
                    if (
                        predecessor_deployment is not None
                        and predecessor.deployment != predecessor_deployment
                    ):
                        raise ValueError(
                            "Predecessor Deployment Identity does not match Execution Run"
                        )
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
                if candidate_request is None:
                    request = replace(
                        request,
                        max_active_provider_calls=(
                            predecessor.max_active_provider_calls
                            if max_active_provider_calls is None
                            else max_active_provider_calls
                        ),
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
        request: ABCFold2ExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> ABCFold2ExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active ABCFold2 runtime does not match request")
            return runtime
        runtime = ABCFold2ExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_claims=self.output_claims,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
