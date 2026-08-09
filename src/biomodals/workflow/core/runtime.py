"""Workflow-owned adaptation of the shared execution kernel."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionNodeRecord,
    ExecutionRuntime,
    ExecutionTaskRecord,
    NodeAggregationPolicy,
    NodeStatus,
    ProviderBinding,
    ProviderCallRecord,
    ProviderCallStatus,
    ProviderCallSubmission,
    PullTaskClaim,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
    TaskPlan,
    TaskStatus,
    drive_execution_run,
    ready_node_keys,
    required_node_keys,
    result_probe_frontier,
    resume_execution_run,
)
from biomodals.execution.modal import ModalCallDriver
from biomodals.execution.runtime import ModalDriver
from biomodals.execution.scheduler import (
    NodeAdmissionRank,
    PullWorkerDispatchDescriptor,
    TaskDispatchDescriptor,
    form_pull_worker_candidates,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import ExecutionVolume, ExecutionVolumeSync
from biomodals.schema import AppRunResult, AppRunStatus, VolumePath, WorkflowArtifact
from biomodals.workflow.core.artifact_availability import (
    ExternalArtifactChecker,
    check_artifact_availability,
    mounted_volume_checker,
)
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import Workflow, WorkflowDefinition
from biomodals.workflow.core.execution import execution_plan, node_task_plan
from biomodals.workflow.core.nodes import (
    NodeRunContext,
    RemoteNodeCall,
    RemotePullTaskWorkflowNode,
    RemotePullWorkerCall,
    RemoteTaskWorkflowNode,
    RemoteWorkflowNode,
    RemoteWorkflowTask,
)
from biomodals.workflow.core.run_store import WorkflowRunStore

_TASK_KEY = "node"


def _task_storage_scope(task_key: str) -> str:
    """Map an arbitrary Task key to one collision-resistant path component."""
    return hashlib.sha256(task_key.encode()).hexdigest()


@dataclass(frozen=True)
class _PreparedTask:
    plan: TaskPlan
    observation: AvailabilityStatus


@dataclass(frozen=True)
class _PreparedNode:
    node_id: str
    context: NodeRunContext
    tasks: tuple[_PreparedTask, ...]
    error: Exception | None = None


@dataclass(frozen=True)
class _PreparedProviderResult:
    """A serialized provider return awaiting coordinator-owned publication."""

    temporary_file: BinaryIO
    sha256: str
    size_bytes: int


class WorkflowRuntime:
    """Advance one workflow through a per-Run kernel repository."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        workflow_volume_name: str,
        workflow_volume: ExecutionVolume | None = None,
        modal_driver: ModalDriver | None = None,
        max_parallel_nodes: int = 32,
        max_active_provider_calls: int = 32,
        max_active_gpu_provider_calls: int | None = None,
        strict_external_artifact_checks: bool = False,
        external_artifact_checker: ExternalArtifactChecker | None = None,
        external_volume_roots: Mapping[str, str | Path] | None = None,
        pull_worker_coordinator: Any | None = None,
        store: WorkflowRunStore | None = None,
        now: Callable[[], int] | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Bind workflow code to one opaque Execution Run identity."""
        if max_parallel_nodes < 1:
            raise ValueError("max_parallel_nodes must be positive")
        if strict_external_artifact_checks:
            if external_artifact_checker is None and external_volume_roots is None:
                raise ValueError(
                    "strict_external_artifact_checks requires "
                    "external_artifact_checker or external_volume_roots"
                )
            if external_artifact_checker is None:
                external_artifact_checker = mounted_volume_checker(
                    workflow_volume_name=workflow_volume_name,
                    volume_roots=external_volume_roots or {},
                )
        self.workflow = workflow
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.workflow_volume_name = workflow_volume_name
        self.max_parallel_nodes = max_parallel_nodes
        self.max_active_provider_calls = max_active_provider_calls
        self.max_active_gpu_provider_calls = (
            max_active_provider_calls
            if max_active_gpu_provider_calls is None
            else max_active_gpu_provider_calls
        )
        self.external_artifact_checker = external_artifact_checker
        self.pull_worker_coordinator = pull_worker_coordinator
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self.store = store or WorkflowRunStore(self.volume_root, execution_run_id)
        self._volume_sync = ExecutionVolumeSync(
            volume=workflow_volume,
            store=self.store,
        )
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver or ModalCallDriver(),
            checkpoint=self._checkpoint,
            transaction=self.store.transaction,
            synchronize=self.store.synchronize,
        )
        self._definition: WorkflowDefinition | None = None
        self._workload_run_key: str | None = None

    def run(
        self,
        *,
        workload_run_key: str,
    ) -> AppRunResult:
        """Create or recover this Run and drive it until it cannot advance."""
        with self.store.synchronize():
            repository = self._initialize(workload_run_key)
        snapshot = drive_execution_run(
            repository,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=self.store.synchronize,
        )
        return _app_result_for_run(snapshot.run.status, snapshot.run.status_message)

    def resume(
        self,
        *,
        workload_run_key: str,
    ) -> AppRunResult:
        """Explicitly resume this persisted Run, then drive it."""
        with self.store.synchronize():
            repository = self._initialize(workload_run_key)
        resume_execution_run(
            repository,
            self.execution_run_id,
            reconcile_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            synchronize=self.store.synchronize,
            now=self._now(),
        )
        snapshot = drive_execution_run(
            self.store.execution,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=self.store.synchronize,
        )
        return _app_result_for_run(snapshot.run.status, snapshot.run.status_message)

    def advance_once(self) -> None:
        """Apply one caller-driven workflow scheduling cycle."""
        definition = self._require_definition()
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=lambda: None,
            start_ready_nodes=lambda _required: self._start_ready_nodes(definition),
            after_start_ready_nodes=lambda: self._run_local_tasks(definition),
            admit_remote_tasks=lambda required: self._admit_remote_tasks(
                definition,
                required,
            ),
            reconcile_results=self._reconcile_nodes_and_run,
            now=self._now,
        )

    def cancel(self) -> None:
        """Request cancellation through the shared provider lifecycle."""
        self._provider.cancel_run(self.execution_run_id, now=self._now())

    def attach(self, *, workload_run_key: str) -> None:
        """Open and verify a Run without refreshing worker publications."""
        self._initialize(workload_run_key, reload_volume=False)

    def prepare(self, *, workload_run_key: str) -> None:
        """Create and checkpoint a pending Run before asynchronous driving."""
        self._initialize(workload_run_key, reload_volume=False)
        self._checkpoint()

    def claim_pull_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Checkpoint one idempotent worker claim before returning its payloads."""
        return self._provider.claim_pull_tasks(
            provider_call_id,
            request_id=request_id,
            capacity=capacity,
            now=self._now(),
        )

    def complete_pull_tasks_and_claim(
        self,
        provider_call_id: UUID,
        completions: tuple[tuple[str, str, AppRunResult], ...],
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Publish one microbatch and checkpoint its successor claim together."""
        if not completions:
            raise ValueError("fused pull completion requires a nonempty batch")
        validated = tuple(
            (task_key, request_id, AppRunResult.model_validate(result))
            for task_key, request_id, result in completions
        )
        if len({task_key for task_key, _request_id, _result in validated}) != len(
            validated
        ):
            raise ValueError("fused pull completion contains duplicate Task keys")
        if len({request_id for _task_key, request_id, _result in validated}) != len(
            validated
        ):
            raise ValueError("fused pull completion contains duplicate request IDs")
        with self.store.synchronize():
            self.store.execution.preflight_pull_task_claim(
                provider_call_id,
                request_id=request_id,
                capacity=capacity,
            )
            call = self.store.execution.get_provider_call(
                provider_call_id,
                include_task_keys=False,
            )
            cancellation_is_durable = self.store.execution.get_run(
                self.execution_run_id
            ).cancellation_is_durable
            preflight = []
            for task_key, completion_request_id, _result in validated:
                task = self.store.execution.get_task(
                    self.execution_run_id,
                    call.node_key,
                    task_key,
                )
                if task.worker_provider_call_id != provider_call_id:
                    raise ValueError("Task is not assigned to this Provider Call")
                receipt = self.store.execution.get_pull_task_completion_receipt(
                    provider_call_id,
                    task_key,
                    request_id=completion_request_id,
                )
                published_result = self.store.artifacts.load_task_result(
                    call.node_key,
                    task_key,
                )
                published_artifacts = (
                    self.store.artifacts.load_task_output_artifacts(
                        call.node_key,
                        task_key,
                    )
                    if published_result is not None
                    else ()
                )
                if (
                    receipt is None
                    and not cancellation_is_durable
                    and published_result is None
                    and task.status.is_terminal
                ):
                    raise ValueError(
                        f"cannot complete terminal Task {task.status.value}"
                    )
                preflight.append((
                    task,
                    receipt,
                    published_result,
                    published_artifacts,
                ))
        node = self._require_definition().nodes[call.node_key].node
        if not isinstance(node, RemotePullTaskWorkflowNode):
            raise ValueError("Provider Call does not belong to a pull-worker Node")
        if any(
            receipt is None
            and not cancellation_is_durable
            and published_result is None
            and self._uses_workflow_volume(result)
            for (
                (_task_key, _request_id, result),
                (_task, receipt, published_result, _published_artifacts),
            ) in zip(validated, preflight, strict=True)
        ):
            self._reload_volume()
        prepared = []
        staged_publications: dict[str, tuple[Path, str]] = {}
        for (
            (task_key, completion_request_id, result),
            (task, receipt, published_result, published_artifacts),
        ) in zip(validated, preflight, strict=True):
            if receipt is not None:
                observation, message = receipt
                prepared.append((
                    task,
                    completion_request_id,
                    observation,
                    message,
                    None,
                    None,
                ))
                continue
            if cancellation_is_durable:
                prepared.append((
                    task,
                    completion_request_id,
                    AvailabilityStatus.MISSING,
                    "Completion ignored after workflow cancellation",
                    None,
                    None,
                ))
                continue
            if published_result is not None:
                if task.status == TaskStatus.SUCCEEDED:
                    observation = AvailabilityStatus.AVAILABLE
                else:
                    context = self._node_context(
                        self._require_definition(),
                        call.node_key,
                        task_key=task_key,
                    )
                    try:
                        observation = self._observe_remote_task_publication(
                            node,
                            context,
                            RemoteWorkflowTask(
                                task_key=task.task_key,
                                scientific_payload=task.scientific_payload,
                                execution_payload=task.execution_payload,
                            ),
                            task.fingerprint,
                            published_result,
                            published_artifacts,
                        )
                    except Exception:
                        self._discard_pull_completion_staging(
                            *staged_publications.values()
                        )
                        raise
                prepared.append((
                    task,
                    completion_request_id,
                    observation,
                    (
                        "Published workflow Task result is unavailable"
                        if observation == AvailabilityStatus.MISSING
                        else None
                    ),
                    None,
                    (published_result, published_artifacts),
                ))
                continue
            if result.status != AppRunStatus.SUCCEEDED:
                prepared.append((
                    task,
                    completion_request_id,
                    AvailabilityStatus.MISSING,
                    _node_error_message(result),
                    None,
                    None,
                ))
                continue
            context = self._node_context(
                self._require_definition(),
                call.node_key,
                task_key=task_key,
            )
            publication_token = uuid4().hex
            publication_scope = f"{_task_storage_scope(task_key)}-{publication_token}"
            publication_dir = context.work_dir / "completions" / publication_scope
            staged_publications[task.task_key] = (
                publication_dir,
                publication_token,
            )
            try:
                materialized = materialize_app_run_result(
                    result=result,
                    workflow_volume_name=self.workflow_volume_name,
                    result_dir=publication_dir,
                    artifact_dir=self.store.output_root / "artifacts",
                    producing_node_id=call.node_key,
                    artifact_id_scope=publication_scope,
                    volume_root=self.volume_root,
                )
                artifacts = tuple(materialized.artifacts)
                observation = self._observe_remote_task_publication(
                    node,
                    context,
                    RemoteWorkflowTask(
                        task_key=task.task_key,
                        scientific_payload=task.scientific_payload,
                        execution_payload=task.execution_payload,
                    ),
                    task.fingerprint,
                    materialized.result,
                    artifacts,
                    workflow_artifacts_validated=True,
                )
            except (FileNotFoundError, ValueError) as error:
                self._discard_pull_completion_staging(
                    staged_publications.pop(task.task_key)
                )
                prepared.append((
                    task,
                    completion_request_id,
                    AvailabilityStatus.MISSING,
                    f"Could not publish workflow Task result: {error}",
                    None,
                    None,
                ))
                continue
            except Exception:
                self._discard_pull_completion_staging(*staged_publications.values())
                raise
            prepared.append((
                task,
                completion_request_id,
                observation,
                (
                    "Published workflow Task result is unavailable"
                    if observation == AvailabilityStatus.MISSING
                    else None
                ),
                (materialized.result, artifacts),
                None,
            ))
        now = self._now()
        published_staging: set[str] = set()
        with self.store.synchronize():
            try:
                with self.store.transaction():
                    cancellation_is_durable = self.store.execution.get_run(
                        self.execution_run_id
                    ).cancellation_is_durable
                    kernel_completions = []
                    for (
                        task,
                        completion_request_id,
                        observation,
                        message,
                        publication,
                        revalidated_publication,
                    ) in prepared:
                        receipt = self.store.execution.get_pull_task_completion_receipt(
                            provider_call_id,
                            task.task_key,
                            request_id=completion_request_id,
                        )
                        existing_publication = self.store.artifacts.load_task_result(
                            call.node_key,
                            task.task_key,
                        )
                        current_task = self.store.execution.get_task(
                            self.execution_run_id,
                            call.node_key,
                            task.task_key,
                        )
                        if receipt is not None:
                            observation, message = receipt
                        elif cancellation_is_durable:
                            observation = AvailabilityStatus.MISSING
                            message = "Completion ignored after workflow cancellation"
                        elif (
                            existing_publication is not None
                            and current_task.status == TaskStatus.SUCCEEDED
                        ):
                            observation = AvailabilityStatus.AVAILABLE
                            message = None
                        elif revalidated_publication is not None:
                            revalidated_result, revalidated_artifacts = (
                                revalidated_publication
                            )
                            current_artifacts = (
                                self.store.artifacts.load_task_output_artifacts(
                                    call.node_key,
                                    task.task_key,
                                )
                                if existing_publication is not None
                                else ()
                            )
                            if (
                                existing_publication != revalidated_result
                                or current_artifacts != revalidated_artifacts
                            ):
                                observation = (
                                    current_task.result_observation
                                    or AvailabilityStatus.MISSING
                                )
                                message = None
                        elif existing_publication is not None:
                            observation = (
                                current_task.result_observation
                                or AvailabilityStatus.MISSING
                            )
                            message = None
                        elif current_task.status.is_terminal:
                            raise ValueError(
                                f"cannot complete terminal Task {current_task.status.value}"
                            )
                        elif publication is not None:
                            result, artifacts = publication
                            self.store.artifacts.record_task_publication(
                                call.node_key,
                                task.task_key,
                                task_fingerprint=task.fingerprint,
                                result=result,
                                artifacts=artifacts,
                                now=now,
                            )
                            published_staging.add(task.task_key)
                        kernel_completions.append((
                            task.task_key,
                            completion_request_id,
                            observation,
                            message,
                        ))
                    claim = self.store.execution.record_pull_task_completions_and_claim(
                        provider_call_id,
                        kernel_completions,
                        request_id=request_id,
                        capacity=capacity,
                        now=now,
                    )
            except Exception:
                self._discard_pull_completion_staging(*staged_publications.values())
                raise
            self._discard_pull_completion_staging(
                *(
                    staged
                    for task_key, staged in staged_publications.items()
                    if task_key not in published_staging
                )
            )
            self._checkpoint()
        return claim

    def _discard_pull_completion_staging(
        self,
        *staged: tuple[Path, str],
    ) -> None:
        """Remove callback-owned output scopes that were not published."""
        artifact_dir = self.store.output_root / "artifacts"
        for result_dir, publication_token in staged:
            shutil.rmtree(result_dir, ignore_errors=True)
            try:
                result_dir.parent.rmdir()
            except OSError:
                pass
            for sidecar in artifact_dir.glob(f"*{publication_token}*.json"):
                sidecar.unlink(missing_ok=True)

    def close(self) -> None:
        """Close local resources without cancelling attached child calls."""
        self.store.close()

    def _initialize(
        self,
        workload_run_key: str,
        *,
        reload_volume: bool = False,
    ) -> SqliteExecutionRepository:
        """Load the workflow definition and create or verify its Run."""
        definition = self.workflow.validate()
        self._definition = definition
        self._workload_run_key = workload_run_key
        if reload_volume:
            self._reload_volume()
        return self._ensure_run(definition, workload_run_key)

    def _ensure_run(
        self,
        definition: WorkflowDefinition,
        workload_run_key: str,
    ) -> SqliteExecutionRepository:
        plan = execution_plan(definition, workload_run_key=workload_run_key)
        with self.store.synchronize():
            repository = self.store.execution
            try:
                existing = repository.get_run(self.execution_run_id)
            except LookupError:
                with self.store.transaction():
                    self.store.execution.create_run(
                        execution_run_id=self.execution_run_id,
                        plan=plan,
                        deployment=self.deployment,
                        max_active_provider_calls=self.max_active_provider_calls,
                        max_active_gpu_provider_calls=self.max_active_gpu_provider_calls,
                        now=self._now(),
                    )
                return self.store.execution

            if (
                existing.plan.workload_plan_fingerprint
                != plan.workload_plan_fingerprint
            ):
                raise ValueError(
                    "Workflow Plan Fingerprint does not match Execution Run"
                )
            if existing.plan.workload_run_key != workload_run_key:
                raise ValueError("Workload Run Key does not match Execution Run")
            if existing.deployment != self.deployment:
                raise ValueError("Deployment Identity does not match Execution Run")
            return repository

    def _recover_publications(self) -> None:
        definition = self._require_definition()
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            nodes = repository.list_nodes(self.execution_run_id)
        observations: dict[str, AvailabilityStatus | None] = {}
        for node in nodes:
            if node.status == NodeStatus.SUCCEEDED:
                observations[node.node_key] = AvailabilityStatus.AVAILABLE
            elif node.status.is_terminal:
                observations[node.node_key] = AvailabilityStatus.MISSING
            else:
                observations[node.node_key] = None

        while frontier := result_probe_frontier(run.plan, observations):
            observed = [
                (node_id, self._publication_observation(node_id))
                for node_id in frontier
            ]
            with self.store.transaction():
                if self.store.execution.get_run(
                    self.execution_run_id
                ).cancellation_is_durable:
                    return
                for node_id, observation in observed:
                    if self.store.execution.get_node(
                        self.execution_run_id,
                        node_id,
                    ).status.is_terminal:
                        continue
                    if observation == AvailabilityStatus.MISSING:
                        self.store.artifacts.discard_node_publication(node_id)
                    self.store.execution.record_node_result_observation(
                        self.execution_run_id,
                        node_id,
                        observation,
                        now=self._now(),
                    )
                    observations[node_id] = observation
            if any(
                observation == AvailabilityStatus.UNKNOWN for _, observation in observed
            ):
                return

        required = required_node_keys(
            run.plan,
            {
                node_key: (
                    AvailabilityStatus.MISSING if observation is None else observation
                )
                for node_key, observation in observations.items()
            },
        )
        if required is None:
            return

        task_observations: list[tuple[str, str, AvailabilityStatus]] = []
        with self.store.synchronize():
            repository = self.store.execution
            nodes = repository.list_nodes(self.execution_run_id)
            tasks_by_node = {
                node.node_key: repository.list_tasks_requiring_publication_recovery(
                    self.execution_run_id,
                    node.node_key,
                )
                for node in nodes
                if (
                    node.node_key in required
                    and node.status == NodeStatus.RUNNING
                    and node.discovery_complete
                )
            }
        for node in nodes:
            if (
                node.node_key in required
                and node.status == NodeStatus.RUNNING
                and node.discovery_complete
            ):
                implementation = definition.nodes[node.node_key].node
                if isinstance(implementation, RemoteTaskWorkflowNode):
                    for task in tasks_by_node[node.node_key]:
                        context = self._node_context(
                            definition,
                            node.node_key,
                            task_key=task.task_key,
                        )
                        task_definition = RemoteWorkflowTask(
                            task_key=task.task_key,
                            scientific_payload=task.scientific_payload,
                            execution_payload=task.execution_payload,
                        )
                        task_observations.append((
                            node.node_key,
                            task.task_key,
                            self._remote_task_publication_observation(
                                node.node_key,
                                implementation,
                                context,
                                task_definition,
                                task.fingerprint,
                            ),
                        ))
                else:
                    observation = observations[node.node_key]
                    if observation is None:
                        raise RuntimeError(
                            f"Workflow Node {node.node_key!r} was not probed"
                        )
                    for task in tasks_by_node[node.node_key]:
                        task_observations.append((
                            node.node_key,
                            task.task_key,
                            observation,
                        ))

        if not task_observations:
            return
        with self.store.transaction():
            if self.store.execution.get_run(
                self.execution_run_id
            ).cancellation_is_durable:
                return
            for node_id, task_key, observation in task_observations:
                if self.store.execution.get_task(
                    self.execution_run_id,
                    node_id,
                    task_key,
                ).status.is_terminal:
                    continue
                if observation == AvailabilityStatus.MISSING:
                    self.store.artifacts.discard_task_publication(
                        node_id,
                        task_key,
                    )
                self.store.execution.record_task_result_observation(
                    self.execution_run_id,
                    node_id,
                    task_key,
                    observation,
                    now=self._now(),
                )

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        reconciled = self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=self._prepare_result_envelope,
            finalize_result=self._finalize_result_envelope,
            discard_result=self._discard_prepared_result,
            recover_terminal_publications=(self._recover_terminal_task_publications),
            now=self._now(),
        )
        for _, call in reconciled:
            if (
                call.status != ProviderCallStatus.SUCCEEDED
                or call.node_key not in required
            ):
                continue
            self._publish_provider_result(call)

    def _recover_terminal_task_publications(
        self,
        terminal_calls: tuple[ProviderCallRecord, ...],
    ) -> frozenset[UUID]:
        """Recover callback-lost pull publications before owner projection."""
        definition = self._require_definition()
        pull_calls = {
            call.provider_call_id: call
            for call in terminal_calls
            if isinstance(
                definition.nodes[call.node_key].node,
                RemotePullTaskWorkflowNode,
            )
        }
        if not pull_calls:
            return frozenset()

        call_ids_by_node: dict[str, set[UUID]] = {}
        for call in pull_calls.values():
            call_ids_by_node.setdefault(call.node_key, set()).add(call.provider_call_id)
        tasks = [
            task
            for node_key, call_ids in call_ids_by_node.items()
            for task in self.store.execution.list_tasks(
                self.execution_run_id,
                node_key,
            )
            if (
                not task.status.is_terminal and task.worker_provider_call_id in call_ids
            )
        ]
        prepared: list[
            tuple[
                ExecutionTaskRecord,
                AvailabilityStatus,
                AppRunResult | None,
                tuple[WorkflowArtifact, ...],
            ]
        ] = []
        for task in tasks:
            implementation = definition.nodes[task.node_key].node
            if not isinstance(implementation, RemotePullTaskWorkflowNode):
                continue
            context = self._node_context(
                definition,
                task.node_key,
                task_key=task.task_key,
            )
            task_definition = RemoteWorkflowTask(
                task_key=task.task_key,
                scientific_payload=task.scientific_payload,
                execution_payload=task.execution_payload,
            )
            try:
                result = implementation.recover_remote_task_result(
                    context,
                    task_definition,
                    task.fingerprint,
                )
            except Exception:  # noqa: BLE001 - inconclusive workload validation
                prepared.append((
                    task,
                    AvailabilityStatus.UNKNOWN,
                    None,
                    (),
                ))
                continue
            if result is None:
                continue
            result = AppRunResult.model_validate(result)
            if result.status != AppRunStatus.SUCCEEDED:
                raise ValueError("Recovered workflow Task result must be succeeded")
            materialized = materialize_app_run_result(
                result=result,
                workflow_volume_name=self.workflow_volume_name,
                result_dir=context.work_dir,
                artifact_dir=self.store.output_root / "artifacts",
                producing_node_id=task.node_key,
                artifact_id_scope=_task_storage_scope(task.task_key),
                volume_root=self.volume_root,
            )
            artifacts = tuple(materialized.artifacts)
            prepared.append((
                task,
                self._artifact_observation(
                    artifacts,
                    workflow_artifacts_validated=True,
                ),
                materialized.result,
                artifacts,
            ))

        deferred: set[UUID] = set()
        if prepared:
            with self.store.transaction():
                for task, observation, result, artifacts in prepared:
                    current = self.store.execution.get_task(
                        self.execution_run_id,
                        task.node_key,
                        task.task_key,
                    )
                    if current.status.is_terminal:
                        continue
                    if observation == AvailabilityStatus.UNKNOWN:
                        if current.worker_provider_call_id is not None:
                            deferred.add(current.worker_provider_call_id)
                    elif observation == AvailabilityStatus.MISSING:
                        continue
                    elif result is not None:
                        self.store.artifacts.record_task_publication(
                            task.node_key,
                            task.task_key,
                            task_fingerprint=task.fingerprint,
                            result=result,
                            artifacts=artifacts,
                            now=self._now(),
                        )
                    self.store.execution.record_task_result_observation(
                        self.execution_run_id,
                        task.node_key,
                        task.task_key,
                        observation,
                        now=self._now(),
                    )
        return frozenset(deferred)

    def _publish_provider_result(
        self,
        call: ProviderCallRecord,
    ) -> None:
        node_id = call.node_key
        envelope = call.result_envelope
        node = self._require_definition().nodes[node_id].node
        if isinstance(node, RemotePullTaskWorkflowNode):
            return
        if isinstance(node, RemoteTaskWorkflowNode):
            self._publish_provider_task_results(
                node_id,
                call.task_keys,
                envelope,
                node,
            )
            return
        with self.store.synchronize():
            task = self.store.execution.get_task(
                self.execution_run_id,
                node_id,
                _TASK_KEY,
            )
        if task.status.is_terminal:
            return
        if not isinstance(node, RemoteWorkflowNode):
            self._fail_task(node_id, "Provider result belongs to a local Node")
            return
        try:
            raw_result = self._raw_result(envelope)
            metadata = _remote_metadata(task.execution_payload)
            result = AppRunResult.model_validate(
                node.process_remote_result(raw_result, metadata)
            )
        except Exception as error:
            self._fail_task(node_id, f"Could not decode provider result: {error}")
            return
        if self._uses_workflow_volume(result):
            self._reload_volume()
        self._publish_result(node_id, result)

    def _publish_provider_task_results(
        self,
        node_id: str,
        task_keys: tuple[str, ...],
        envelope: object,
        node: RemoteTaskWorkflowNode,
    ) -> None:
        with self.store.synchronize():
            tasks = tuple(
                self.store.execution.get_task(
                    self.execution_run_id,
                    node_id,
                    task_key,
                )
                for task_key in task_keys
            )
        unfinished = tuple(task for task in tasks if not task.status.is_terminal)
        if not unfinished:
            return
        try:
            task_definitions = tuple(
                RemoteWorkflowTask(
                    task_key=task.task_key,
                    scientific_payload=task.scientific_payload,
                    execution_payload=task.execution_payload,
                )
                for task in tasks
            )
            invocation = node.prepare_remote_task_batch(
                self._node_context(
                    self._require_definition(),
                    node_id,
                    task_key=task_keys[0],
                ),
                task_definitions,
            )
            decoded = node.process_remote_task_batch_result(
                task_keys,
                self._raw_result(envelope),
                invocation.metadata,
            )
            if set(decoded) != set(task_keys):
                raise ValueError(
                    "Batched provider result Task keys do not match call ownership"
                )
            results = {
                task_key: AppRunResult.model_validate(result)
                for task_key, result in decoded.items()
            }
        except Exception as error:
            for task in unfinished:
                self._fail_discovered_task(
                    node_id,
                    task.task_key,
                    f"Could not decode provider result: {error}",
                )
            return
        if any(self._uses_workflow_volume(result) for result in results.values()):
            self._reload_volume()
        for task in unfinished:
            self._publish_task_result(
                node_id,
                task.task_key,
                results[task.task_key],
            )

    def _start_ready_nodes(self, definition: WorkflowDefinition) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            node_records = repository.list_nodes(self.execution_run_id)
            statuses = {node.node_key: node.status for node in node_records}
            plan = repository.get_run(self.execution_run_id).plan
        ready = ready_node_keys(plan, statuses)
        available_slots = self.max_parallel_nodes - sum(
            node.status == NodeStatus.RUNNING for node in node_records
        )
        if not ready or available_slots <= 0:
            return

        prepared = [
            self._prepare_node(definition, node_id)
            for node_id in ready[:available_slots]
        ]
        with self.store.transaction():
            for item in prepared:
                if self.store.execution.get_node(
                    self.execution_run_id,
                    item.node_id,
                ).status.is_terminal:
                    continue
                self.store.execution.start_node(
                    self.execution_run_id,
                    item.node_id,
                    now=self._now(),
                )
                self.store.artifacts.record_node_inputs(
                    item.node_id,
                    item.context.inputs,
                )
                if item.error is not None:
                    self.store.execution.fail_node(
                        self.execution_run_id,
                        item.node_id,
                        message=f"Could not prepare workflow Node: {item.error}",
                        now=self._now(),
                    )
                    continue
                self.store.execution.discover_tasks(
                    self.execution_run_id,
                    item.node_id,
                    tuple(task.plan for task in item.tasks),
                    now=self._now(),
                )
                for task in item.tasks:
                    if task.observation == AvailabilityStatus.MISSING:
                        self.store.artifacts.discard_task_publication(
                            item.node_id,
                            task.plan.task_key,
                        )
                    self.store.execution.record_task_result_observation(
                        self.execution_run_id,
                        item.node_id,
                        task.plan.task_key,
                        task.observation,
                        now=self._now(),
                    )

    def _prepare_node(
        self,
        definition: WorkflowDefinition,
        node_id: str,
    ) -> _PreparedNode:
        context = self._node_context(definition, node_id)
        node = definition.nodes[node_id].node
        try:
            if isinstance(node, RemoteTaskWorkflowNode):
                discovered = node.discover_remote_tasks(context)
                with self.store.synchronize():
                    workload_plan_fingerprint = self.store.execution.get_run(
                        self.execution_run_id
                    ).plan.workload_plan_fingerprint
                tasks: list[_PreparedTask] = []
                for task in discovered:
                    plan = TaskPlan(
                        task_key=task.task_key,
                        scientific_payload=_json_value(task.scientific_payload),
                        execution_payload=_json_value(task.execution_payload),
                    )
                    tasks.append(
                        _PreparedTask(
                            plan=plan,
                            observation=self._remote_task_publication_observation(
                                node_id,
                                node,
                                context,
                                task,
                                plan.fingerprint(
                                    workload_plan_fingerprint=(
                                        workload_plan_fingerprint
                                    ),
                                    node_key=node_id,
                                ),
                            ),
                        )
                    )
                return _PreparedNode(node_id, context, tuple(tasks))
            invocation = (
                node.prepare_remote(context)
                if isinstance(node, RemoteWorkflowNode)
                else None
            )
            task = _PreparedTask(
                plan=TaskPlan(
                    task_key=_TASK_KEY,
                    scientific_payload=node_task_plan(node_id).scientific_payload,
                    execution_payload=_execution_payload(invocation),
                ),
                observation=AvailabilityStatus.MISSING,
            )
            return _PreparedNode(node_id, context, (task,))
        except Exception as error:
            return _PreparedNode(node_id, context, (), error)

    def _run_local_tasks(self, definition: WorkflowDefinition) -> bool:
        progressed = False
        with self.store.synchronize():
            node_records = self.store.execution.list_nodes(self.execution_run_id)
        for node_record in node_records:
            if (
                node_record.status != NodeStatus.RUNNING
                or not node_record.discovery_complete
            ):
                continue
            node = definition.nodes[node_record.node_key].node
            if isinstance(node, RemoteWorkflowNode | RemoteTaskWorkflowNode):
                continue
            with self.store.synchronize():
                task = self.store.execution.get_task(
                    self.execution_run_id,
                    node_record.node_key,
                    _TASK_KEY,
                )
            if task.status.is_terminal:
                continue
            with self.store.synchronize():
                with self.store.transaction():
                    acquired = self.store.execution.acquire_local_task(
                        self.execution_run_id,
                        node_record.node_key,
                        _TASK_KEY,
                        now=self._now(),
                    )
                if acquired:
                    self._checkpoint()
            if not acquired:
                continue
            progressed = True
            context = self._node_context(definition, node_record.node_key)
            try:
                result = AppRunResult.model_validate(node.run(context))
            except Exception as error:
                self._fail_task(
                    node_record.node_key,
                    f"Coordinator-local Node failed: {error}",
                )
                continue
            self._publish_result(node_record.node_key, result)
        return progressed

    def _admit_remote_tasks(
        self,
        definition: WorkflowDefinition,
        required: set[str],
    ) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            nodes = {
                node.node_key: node
                for node in repository.list_nodes(run.execution_run_id)
            }
            counts = repository.active_provider_call_counts(self.execution_run_id)
        available_total_slots = max(
            0,
            run.max_active_provider_calls - counts.total,
        )
        available_gpu_slots = max(
            0,
            run.max_active_gpu_provider_calls - counts.gpu,
        )
        if available_total_slots == 0:
            return
        unfinished = {
            node.node_key for node in nodes.values() if not node.status.is_terminal
        }
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys=unfinished,
        )
        fixed_node_keys: set[str] = set()
        pull_invocations: dict[str, RemotePullWorkerCall] = {}
        pull_descriptors: list[PullWorkerDispatchDescriptor] = []
        pull_node_keys = tuple(
            node_id
            for node_id, node_record in nodes.items()
            if (
                node_id in required
                and node_record.status == NodeStatus.RUNNING
                and node_record.discovery_complete
                and isinstance(
                    definition.nodes[node_id].node,
                    RemotePullTaskWorkflowNode,
                )
            )
        )
        with self.store.synchronize():
            provider_counts_by_node = repository.provider_call_counts_by_node(
                self.execution_run_id,
                pull_node_keys,
            )
        for node_id, node_record in nodes.items():
            if (
                node_id not in required
                or node_record.status != NodeStatus.RUNNING
                or not node_record.discovery_complete
            ):
                continue
            node = definition.nodes[node_id].node
            if not isinstance(node, RemoteWorkflowNode | RemoteTaskWorkflowNode):
                continue
            if isinstance(node, RemotePullTaskWorkflowNode):
                try:
                    invocation = node.prepare_pull_worker(
                        self._node_context(definition, node_id)
                    )
                except Exception as error:
                    self._fail_node_publication(
                        node_id,
                        f"Could not prepare pull worker: {error}",
                    )
                    continue
                rank = ranks[node_id]
                binding = ProviderBinding(
                    environment=run.deployment.environment,
                    app_name=run.deployment.deployment_name,
                    app_version=run.deployment.deployment_version,
                    function_name=invocation.function_name,
                    uses_gpu=invocation.uses_gpu,
                    runtime_image_key=invocation.runtime_image_key,
                )
                with self.store.synchronize():
                    unfinished_task_count = (
                        self.store.execution.unfinished_pull_task_count(
                            self.execution_run_id,
                            node_id,
                        )
                    )
                    total_workers, nonterminal_workers = provider_counts_by_node.get(
                        node_id,
                        (0, 0),
                    )
                pull_invocations[node_id] = invocation
                pull_descriptors.append(
                    PullWorkerDispatchDescriptor(
                        node_key=node_id,
                        node_ordinal=node_record.ordinal,
                        binding=binding,
                        compatibility_key=(
                            invocation.compatibility_key or invocation.function_name
                        ),
                        claim_capacity=invocation.claim_capacity,
                        max_worker_calls=invocation.max_worker_calls,
                        unfinished_task_count=unfinished_task_count,
                        nonterminal_worker_count=nonterminal_workers,
                        next_worker_ordinal=total_workers,
                        depth=rank.depth,
                        unblocking_span=rank.unblocking_span,
                    )
                )
                continue
            fixed_node_keys.add(node_id)

        def describe_task(
            node_record: ExecutionNodeRecord,
            task: ExecutionTaskRecord,
            rank: NodeAdmissionRank,
        ) -> TaskDispatchDescriptor | None:
            node = definition.nodes[node_record.node_key].node
            try:
                if isinstance(node, RemoteTaskWorkflowNode):
                    invocation = node.prepare_remote_task(
                        self._node_context(
                            definition,
                            node_record.node_key,
                            task_key=task.task_key,
                        ),
                        RemoteWorkflowTask(
                            task_key=task.task_key,
                            scientific_payload=task.scientific_payload,
                            execution_payload=task.execution_payload,
                        ),
                    )
                    _json_value(_execution_payload(invocation))
                elif isinstance(node, RemoteWorkflowNode):
                    invocation = node.prepare_remote(
                        self._node_context(definition, node_record.node_key)
                    )
                    payload = _json_value(_execution_payload(invocation))
                    if payload != task.execution_payload:
                        raise ValueError(
                            "Remote Node preparation changed after Task discovery"
                        )
                else:  # pragma: no cover - filtered by fixed_node_keys
                    raise TypeError("Fixed dispatch requires a remote workflow Node")
            except Exception as error:
                self._fail_discovered_task(
                    node_record.node_key,
                    task.task_key,
                    f"Could not prepare provider call: {error}",
                )
                return None
            binding = ProviderBinding(
                environment=run.deployment.environment,
                app_name=run.deployment.deployment_name,
                app_version=run.deployment.deployment_version,
                function_name=invocation.function_name,
                uses_gpu=invocation.uses_gpu,
                runtime_image_key=invocation.runtime_image_key,
            )
            return TaskDispatchDescriptor(
                node_key=node_record.node_key,
                node_ordinal=node_record.ordinal,
                task_key=task.task_key,
                task_ordinal=task.ordinal,
                binding=binding,
                compatibility_key=(
                    invocation.compatibility_key or invocation.function_name
                ),
                max_tasks_per_call=invocation.max_tasks_per_call,
                depth=rank.depth,
                unblocking_span=rank.unblocking_span,
            )

        fixed_candidates = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            candidate_node_keys=fixed_node_keys,
            describe_task=describe_task,
            available_total_slots=available_total_slots,
            available_gpu_slots=available_gpu_slots,
            now=self._now(),
        )

        pull_descriptors = [
            self._provider.persist_pull_worker_dispatch_policy(
                self.execution_run_id,
                descriptor,
                now=self._now(),
            )
            for descriptor in pull_descriptors
        ]
        selected = select_admissible_candidates(
            (
                *fixed_candidates,
                *form_pull_worker_candidates(tuple(pull_descriptors)),
            ),
            available_total_slots=available_total_slots,
            available_gpu_slots=available_gpu_slots,
        )
        submissions = []
        for candidate in selected:
            node = definition.nodes[candidate.node_key].node
            if isinstance(node, RemotePullTaskWorkflowNode):
                if self.pull_worker_coordinator is None:
                    self._fail_node_publication(
                        candidate.node_key,
                        "Pull-worker coordinator handle is unavailable",
                    )
                    continue
                invocation = pull_invocations[candidate.node_key]
                kwargs = dict(invocation.kwargs)
                if "coordinator" in kwargs:
                    raise ValueError(
                        "Pull-worker coordinator argument is runtime-owned"
                    )
                kwargs["coordinator"] = self.pull_worker_coordinator
                submissions.append(
                    ProviderCallSubmission(
                        candidate=candidate,
                        claim_capacity=invocation.claim_capacity,
                        provider_call_id_kwarg="provider_call_id",
                        submission_token=candidate.candidate_key,
                        args=invocation.args,
                        kwargs=kwargs,
                    )
                )
                continue

            with self.store.synchronize():
                tasks = tuple(
                    self.store.execution.get_task(
                        self.execution_run_id,
                        candidate.node_key,
                        task_key,
                    )
                    for task_key in candidate.task_keys
                )
            try:
                if isinstance(node, RemoteTaskWorkflowNode):
                    task_definitions = tuple(
                        RemoteWorkflowTask(
                            task_key=task.task_key,
                            scientific_payload=task.scientific_payload,
                            execution_payload=task.execution_payload,
                        )
                        for task in tasks
                    )
                    if len(task_definitions) == 1:
                        invocation = node.prepare_remote_task(
                            self._node_context(
                                definition,
                                candidate.node_key,
                                task_key=task_definitions[0].task_key,
                            ),
                            task_definitions[0],
                        )
                    else:
                        invocation = node.prepare_remote_task_batch(
                            self._node_context(
                                definition,
                                candidate.node_key,
                                task_key=task_definitions[0].task_key,
                            ),
                            task_definitions,
                        )
                elif isinstance(node, RemoteWorkflowNode):
                    if len(tasks) != 1:  # pragma: no cover - scheduler contract
                        raise RuntimeError(
                            "Only remote Task Nodes may own batched calls"
                        )
                    invocation = node.prepare_remote(
                        self._node_context(definition, candidate.node_key)
                    )
                    if (
                        _json_value(_execution_payload(invocation))
                        != tasks[0].execution_payload
                    ):
                        raise ValueError(
                            "Remote Node preparation changed after Task discovery"
                        )
                else:  # pragma: no cover - scheduler contract
                    raise TypeError("Fixed dispatch requires a remote workflow Node")

                invocation_binding = ProviderBinding(
                    environment=run.deployment.environment,
                    app_name=run.deployment.deployment_name,
                    app_version=run.deployment.deployment_version,
                    function_name=invocation.function_name,
                    uses_gpu=invocation.uses_gpu,
                    runtime_image_key=invocation.runtime_image_key,
                )
                if (
                    invocation_binding != candidate.binding
                    or (invocation.compatibility_key or invocation.function_name)
                    != candidate.compatibility_key
                    or invocation.max_tasks_per_call < len(candidate.task_keys)
                ):
                    raise ValueError(
                        "Provider preparation changed its dispatch contract"
                    )
            except Exception as error:
                for task_key in candidate.task_keys:
                    self._fail_discovered_task(
                        candidate.node_key,
                        task_key,
                        f"Could not prepare provider call: {error}",
                    )
                continue

            submissions.append(
                ProviderCallSubmission(
                    candidate=candidate,
                    submission_token=candidate.candidate_key,
                    args=invocation.args,
                    kwargs=invocation.kwargs,
                )
            )
        self._provider.submit_provider_calls(
            self.execution_run_id,
            tuple(submissions),
            now=self._now(),
        )

    def _publish_result(self, node_id: str, result: AppRunResult) -> None:
        if result.status != AppRunStatus.SUCCEEDED:
            self._fail_task(node_id, _node_error_message(result))
            return
        context = self._node_context(self._require_definition(), node_id)
        materialized = materialize_app_run_result(
            result=result,
            workflow_volume_name=self.workflow_volume_name,
            result_dir=context.work_dir,
            artifact_dir=self.store.output_root / "artifacts",
            producing_node_id=node_id,
            volume_root=self.volume_root,
        )
        observation = self._artifact_observation(
            tuple(materialized.artifacts),
            workflow_artifacts_validated=True,
        )
        with self.store.transaction():
            if self.store.execution.get_task(
                self.execution_run_id,
                node_id,
                _TASK_KEY,
            ).status.is_terminal:
                return
            self.store.artifacts.record_node_publication(
                node_id,
                result=materialized.result,
                artifacts=tuple(materialized.artifacts),
                now=self._now(),
            )
            if observation == AvailabilityStatus.MISSING:
                self.store.execution.fail_task(
                    self.execution_run_id,
                    node_id,
                    _TASK_KEY,
                    message="Published workflow result is unavailable",
                    now=self._now(),
                )
            else:
                self.store.execution.record_task_result_observation(
                    self.execution_run_id,
                    node_id,
                    _TASK_KEY,
                    observation,
                    now=self._now(),
                )
            if observation != AvailabilityStatus.UNKNOWN:
                self.store.execution.reconcile_node_tasks(
                    self.execution_run_id,
                    node_id,
                    now=self._now(),
                )

    def _publish_task_result(
        self,
        node_id: str,
        task_key: str,
        result: AppRunResult,
    ) -> None:
        with self.store.synchronize():
            task = self.store.execution.get_task(
                self.execution_run_id,
                node_id,
                task_key,
            )
        if result.status != AppRunStatus.SUCCEEDED:
            self._fail_discovered_task(
                node_id,
                task_key,
                _node_error_message(result),
            )
            return
        context = self._node_context(
            self._require_definition(),
            node_id,
            task_key=task_key,
        )
        materialized = materialize_app_run_result(
            result=result,
            workflow_volume_name=self.workflow_volume_name,
            result_dir=context.work_dir,
            artifact_dir=self.store.output_root / "artifacts",
            producing_node_id=node_id,
            artifact_id_scope=_task_storage_scope(task_key),
            volume_root=self.volume_root,
        )
        observation = self._artifact_observation(
            tuple(materialized.artifacts),
            workflow_artifacts_validated=True,
        )
        with self.store.transaction():
            if self.store.execution.get_task(
                self.execution_run_id,
                node_id,
                task_key,
            ).status.is_terminal:
                return
            self.store.artifacts.record_task_publication(
                node_id,
                task_key,
                task_fingerprint=task.fingerprint,
                result=materialized.result,
                artifacts=tuple(materialized.artifacts),
                now=self._now(),
            )
            if observation == AvailabilityStatus.MISSING:
                self.store.execution.fail_task(
                    self.execution_run_id,
                    node_id,
                    task_key,
                    message="Published workflow Task result is unavailable",
                    now=self._now(),
                )
            else:
                self.store.execution.record_task_result_observation(
                    self.execution_run_id,
                    node_id,
                    task_key,
                    observation,
                    now=self._now(),
                )

    def _fail_task(self, node_id: str, message: str) -> None:
        with self.store.transaction():
            self.store.execution.fail_task(
                self.execution_run_id,
                node_id,
                _TASK_KEY,
                message=message,
                now=self._now(),
            )
            self.store.execution.reconcile_node_tasks(
                self.execution_run_id,
                node_id,
                now=self._now(),
            )
            self.store.execution.skip_unreachable_nodes(
                self.execution_run_id,
                now=self._now(),
            )
            self.store.execution.finalize_run_from_results(
                self.execution_run_id,
                now=self._now(),
            )

    def _fail_discovered_task(
        self,
        node_id: str,
        task_key: str,
        message: str,
    ) -> None:
        """Fail one prepared Task without collapsing its independent siblings."""
        with self.store.transaction():
            self.store.execution.fail_task(
                self.execution_run_id,
                node_id,
                task_key,
                message=message,
                now=self._now(),
            )

    def _reconcile_nodes_and_run(self) -> None:
        definition = self._require_definition()
        with self.store.synchronize():
            nodes = self.store.execution.list_nodes(self.execution_run_id)
        for node in nodes:
            if node.status != NodeStatus.RUNNING or not node.discovery_complete:
                continue
            implementation = definition.nodes[node.node_key].node
            if isinstance(implementation, RemoteTaskWorkflowNode):
                self._finalize_remote_task_node(
                    node.node_key,
                    node.aggregation_policy,
                    node.allow_empty_result,
                    implementation,
                )
                continue
            with self.store.transaction():
                self.store.execution.reconcile_node_tasks(
                    self.execution_run_id,
                    node.node_key,
                    now=self._now(),
                )
        with self.store.transaction():
            self.store.execution.skip_unreachable_nodes(
                self.execution_run_id,
                now=self._now(),
            )
            self.store.execution.finalize_run_from_results(
                self.execution_run_id,
                now=self._now(),
            )

    def _node_publication_is_fenced(self, node_id: str) -> bool:
        """Return whether cancellation or terminal state forbids publication."""
        return (
            self.store.execution.get_run(self.execution_run_id).cancellation_is_durable
            or self.store.execution.get_node(
                self.execution_run_id,
                node_id,
            ).status.is_terminal
        )

    def _finalize_remote_task_node(
        self,
        node_id: str,
        aggregation_policy: NodeAggregationPolicy,
        allow_empty_result: bool,
        implementation: RemoteTaskWorkflowNode,
    ) -> None:
        with self.store.transaction():
            if self._node_publication_is_fenced(node_id):
                return
            self.store.execution.apply_task_failure_policy(
                self.execution_run_id,
                node_id,
                now=self._now(),
            )
        with self.store.synchronize():
            task_count, outcome = self.store.execution.summarize_node_tasks(
                self.execution_run_id,
                node_id,
                aggregation_policy,
            )
        empty_result = task_count == 0
        outcome = (
            NodeStatus.SUCCEEDED if empty_result and allow_empty_result else outcome
        )
        if outcome is None:
            return
        if outcome == NodeStatus.CANCELLED:
            with self.store.transaction():
                self.store.execution.reconcile_node_tasks(
                    self.execution_run_id,
                    node_id,
                    now=self._now(),
                )
            return

        with self.store.synchronize():
            existing = self.store.artifacts.load_node_result(node_id)
            existing_artifacts = self.store.artifacts.load_node_output_artifacts(
                node_id
            )
        if existing is not None:
            observation = self._artifact_observation(existing_artifacts)
            if observation == AvailabilityStatus.MISSING:
                with self.store.transaction():
                    if self._node_publication_is_fenced(node_id):
                        return
                    self.store.artifacts.discard_node_publication(node_id)
            elif observation == AvailabilityStatus.UNKNOWN:
                with self.store.transaction():
                    if self._node_publication_is_fenced(node_id):
                        return
                    self.store.execution.transition_run(
                        self.execution_run_id,
                        RunStatus.SUSPENDED,
                        reason=RunStatusReason.RESULT_VALIDATION_UNKNOWN,
                        message=f"Could not validate workflow Node {node_id!r}",
                        now=self._now(),
                    )
                return
            else:
                if existing.status != _app_status_for_node(outcome):
                    self._fail_node_publication(
                        node_id,
                        "Persisted Node publication status does not match "
                        "terminal Task outcomes",
                    )
                    return
                with self.store.transaction():
                    if self._node_publication_is_fenced(node_id):
                        return
                    if empty_result:
                        self.store.execution.record_node_result_observation(
                            self.execution_run_id,
                            node_id,
                            AvailabilityStatus.AVAILABLE,
                            now=self._now(),
                        )
                    else:
                        self.store.execution.reconcile_node_tasks(
                            self.execution_run_id,
                            node_id,
                            now=self._now(),
                        )
                return

        with self.store.synchronize():
            tasks = self.store.execution.list_tasks(self.execution_run_id, node_id)
        results: dict[str, AppRunResult] = {}
        errors: dict[str, str] = {}
        task_artifacts: list[WorkflowArtifact] = []
        with self.store.synchronize():
            publications = {
                task.task_key: (
                    self.store.artifacts.load_task_result(node_id, task.task_key),
                    self.store.artifacts.load_task_output_artifacts(
                        node_id,
                        task.task_key,
                    ),
                )
                for task in tasks
                if task.status == TaskStatus.SUCCEEDED
            }
        for task in tasks:
            if task.status == TaskStatus.SUCCEEDED:
                result, artifacts = publications[task.task_key]
                if result is None:
                    self._fail_node_publication(
                        node_id,
                        f"Successful Task {task.task_key!r} has no publication",
                    )
                    return
                results[task.task_key] = result
                task_artifacts.extend(artifacts)
            else:
                errors[task.task_key] = (
                    task.error_message or f"Task ended as {task.status.value}"
                )
        try:
            finalization = AppRunResult.model_validate(
                implementation.finalize_remote_tasks(
                    self._node_context(self._require_definition(), node_id),
                    results,
                    errors,
                )
            )
            expected_status = _app_status_for_node(outcome)
            if finalization.status != expected_status:
                raise ValueError(
                    f"finalizer returned {finalization.status.value}; "
                    f"expected {expected_status.value}"
                )
            with self.store.synchronize():
                if self._node_publication_is_fenced(node_id):
                    return
            materialized = materialize_app_run_result(
                result=finalization,
                workflow_volume_name=self.workflow_volume_name,
                result_dir=(
                    self.store.output_root / "nodes" / node_id / "result" / "aggregate"
                ),
                artifact_dir=self.store.output_root / "artifacts",
                producing_node_id=node_id,
                artifact_id_scope="aggregate",
                volume_root=self.volume_root,
            )
            combined_result = materialized.result.model_copy(
                update={
                    "outputs": [
                        output
                        for result in results.values()
                        for output in result.outputs
                    ]
                    + materialized.result.outputs,
                    "logs": [
                        output for result in results.values() for output in result.logs
                    ]
                    + materialized.result.logs,
                    "warnings": [
                        warning
                        for result in results.values()
                        for warning in result.warnings
                    ]
                    + materialized.result.warnings,
                }
            )
            artifacts = (*task_artifacts, *materialized.artifacts)
            observation = self._artifact_observation((
                *task_artifacts,
                *(
                    artifact
                    for artifact in materialized.artifacts
                    if artifact.storage.volume_name != self.workflow_volume_name
                ),
            ))
        except Exception as error:
            self._fail_node_publication(
                node_id,
                f"Could not finalize workflow Node: {error}",
            )
            return

        with self.store.transaction():
            if self._node_publication_is_fenced(node_id):
                return
            self.store.artifacts.record_node_publication(
                node_id,
                result=combined_result,
                artifacts=artifacts,
                now=self._now(),
            )
            if observation == AvailabilityStatus.AVAILABLE:
                if empty_result:
                    self.store.execution.record_node_result_observation(
                        self.execution_run_id,
                        node_id,
                        AvailabilityStatus.AVAILABLE,
                        now=self._now(),
                    )
                else:
                    self.store.execution.reconcile_node_tasks(
                        self.execution_run_id,
                        node_id,
                        now=self._now(),
                    )
            elif observation == AvailabilityStatus.MISSING:
                self.store.execution.fail_node(
                    self.execution_run_id,
                    node_id,
                    message="Published workflow Node result is unavailable",
                    now=self._now(),
                )
            else:
                self.store.execution.transition_run(
                    self.execution_run_id,
                    RunStatus.SUSPENDED,
                    reason=RunStatusReason.RESULT_VALIDATION_UNKNOWN,
                    message=f"Could not validate workflow Node {node_id!r}",
                    now=self._now(),
                )

    def _fail_node_publication(self, node_id: str, message: str) -> None:
        with self.store.transaction():
            self.store.execution.fail_node(
                self.execution_run_id,
                node_id,
                message=message,
                now=self._now(),
            )

    def _publication_observation(self, node_id: str) -> AvailabilityStatus:
        with self.store.synchronize():
            result = self.store.artifacts.load_node_result(node_id)
            artifacts = self.store.artifacts.load_node_output_artifacts(node_id)
        if result is None or result.status != AppRunStatus.SUCCEEDED:
            return AvailabilityStatus.MISSING
        return self._artifact_observation(artifacts)

    def _remote_task_publication_observation(
        self,
        node_id: str,
        implementation: RemoteTaskWorkflowNode,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
        expected_fingerprint: str,
    ) -> AvailabilityStatus:
        """Validate a stored Task result and its workload-owned publication."""
        with self.store.synchronize():
            result = self.store.artifacts.load_task_result(node_id, task.task_key)
            fingerprint = self.store.artifacts.load_task_fingerprint(
                node_id,
                task.task_key,
            )
            artifacts = self.store.artifacts.load_task_output_artifacts(
                node_id,
                task.task_key,
            )
        if (
            result is None
            or result.status != AppRunStatus.SUCCEEDED
            or fingerprint != expected_fingerprint
        ):
            return AvailabilityStatus.MISSING
        return self._observe_remote_task_publication(
            implementation,
            context,
            task,
            expected_fingerprint,
            result,
            artifacts,
        )

    def _observe_remote_task_publication(
        self,
        implementation: RemoteTaskWorkflowNode,
        context: NodeRunContext,
        task: RemoteWorkflowTask,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[WorkflowArtifact, ...],
        *,
        workflow_artifacts_validated: bool = False,
    ) -> AvailabilityStatus:
        """Combine durable workflow artifacts with a workload-specific probe."""
        artifact_observation = self._artifact_observation(
            artifacts,
            workflow_artifacts_validated=workflow_artifacts_validated,
        )
        if artifact_observation != AvailabilityStatus.AVAILABLE:
            return artifact_observation
        workload_observation = implementation.observe_remote_task_publication(
            context,
            task,
            expected_fingerprint,
            result,
            artifacts,
        )
        return workload_observation or AvailabilityStatus.AVAILABLE

    def _artifact_observation(
        self,
        artifacts: tuple[WorkflowArtifact, ...],
        *,
        workflow_artifacts_validated: bool = False,
    ) -> AvailabilityStatus:
        statuses = [
            check_artifact_availability(
                artifact,
                workflow_volume_name=self.workflow_volume_name,
                volume_root=self.volume_root,
                external_artifact_checker=self.external_artifact_checker,
            ).status
            for artifact in artifacts
            if not (
                workflow_artifacts_validated
                and artifact.storage.volume_name == self.workflow_volume_name
            )
        ]
        if AvailabilityStatus.UNKNOWN in statuses:
            return AvailabilityStatus.UNKNOWN
        if AvailabilityStatus.MISSING in statuses:
            return AvailabilityStatus.MISSING
        return AvailabilityStatus.AVAILABLE

    def _uses_workflow_volume(self, result: AppRunResult) -> bool:
        return any(
            isinstance(output.storage, VolumePath)
            and output.storage.volume_name == self.workflow_volume_name
            for output in (*result.outputs, *result.logs)
        )

    def _prepare_result_envelope(self, result: object) -> _PreparedProviderResult:
        """Serialize a provider return without touching the shared Volume."""
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        content = orjson.dumps(result)
        digest = hashlib.sha256(content).hexdigest()
        temporary_file = tempfile.TemporaryFile()
        try:
            temporary_file.write(content)
            temporary_file.seek(0)
        except BaseException:
            temporary_file.close()
            raise
        return _PreparedProviderResult(
            temporary_file=temporary_file,
            sha256=digest,
            size_bytes=len(content),
        )

    def _finalize_result_envelope(
        self,
        prepared: _PreparedProviderResult,
    ) -> dict[str, object]:
        """Publish a prepared return within the coordinator writer boundary."""
        digest = prepared.sha256
        relative_path = Path("provider-results") / f"{digest}.json"
        result_path = self.store.output_root / relative_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        volume_temporary = tempfile.NamedTemporaryFile(
            dir=result_path.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(volume_temporary.name)
        try:
            with volume_temporary:
                shutil.copyfileobj(prepared.temporary_file, volume_temporary)
            temporary_path.replace(result_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return {
            "result_file": {
                "path": relative_path.as_posix(),
                "sha256": digest,
                "size_bytes": prepared.size_bytes,
            }
        }

    @staticmethod
    def _discard_prepared_result(prepared: _PreparedProviderResult) -> None:
        """Release one coordinator-local serialization temporary file."""
        prepared.temporary_file.close()

    def _raw_result(self, envelope: object) -> object:
        """Load and verify one workflow-owned provider return."""
        if not isinstance(envelope, dict):
            raise ValueError("Workflow Result Envelope must be an object")
        reference = envelope.get("result_file")
        if not isinstance(reference, dict):
            raise ValueError("Workflow Result Envelope has no result file")
        relative_value = reference.get("path")
        expected_digest = reference.get("sha256")
        expected_size = reference.get("size_bytes")
        if (
            not isinstance(relative_value, str)
            or not isinstance(expected_digest, str)
            or not isinstance(expected_size, int)
        ):
            raise ValueError("Workflow Result Envelope reference is invalid")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or relative_path.parts[:1] != ("provider-results",)
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("Workflow Result Envelope path is invalid")
        result_path = self.store.output_root.joinpath(*relative_path.parts)
        content = result_path.read_bytes()
        if len(content) != expected_size:
            raise ValueError("Workflow provider result size does not match")
        if hashlib.sha256(content).hexdigest() != expected_digest:
            raise ValueError("Workflow provider result checksum does not match")
        return orjson.loads(content)

    def _node_context(
        self,
        definition: WorkflowDefinition,
        node_id: str,
        *,
        task_key: str | None = None,
    ) -> NodeRunContext:
        spec = definition.nodes[node_id]
        with self.store.synchronize():
            inputs = {
                input_name: list(self.store.artifacts.select_artifacts(selector))
                for input_name, selector in spec.inputs.items()
            }
        node_root = self.store.output_root / "nodes" / node_id
        context_task_key = _TASK_KEY if task_key is None else task_key
        if task_key is None:
            work_dir = node_root / "result"
            cache_dir = node_root / "cache"
        else:
            task_root = node_root / "tasks" / _task_storage_scope(task_key)
            work_dir = task_root / "result"
            cache_dir = task_root / "cache"
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return NodeRunContext(
            execution_run_id=self.execution_run_id,
            workload_run_key=self._require_workload_run_key(),
            node_id=node_id,
            task_key=context_task_key,
            work_dir=work_dir,
            cache_dir=cache_dir,
            inputs=inputs,
            volume_root=self.volume_root,
            workflow_volume_name=self.workflow_volume_name,
        )

    def _checkpoint(self) -> SqliteExecutionRepository:
        with self.store.synchronize():
            try:
                self._volume_sync.commit()
            finally:
                repository = self.store.execution
                self._provider.repository = repository
        return repository

    def _reload_volume(self) -> None:
        """Refresh cross-container publications and reopen the shared ledger."""
        with self.store.synchronize():
            try:
                self._volume_sync.reload()
            finally:
                self._provider.repository = self.store.execution

    def _require_definition(self) -> WorkflowDefinition:
        if self._definition is None:
            raise RuntimeError("Workflow Run has not been initialized")
        return self._definition

    def _require_workload_run_key(self) -> str:
        if self._workload_run_key is None:
            raise RuntimeError("Workflow Run has not been initialized")
        return self._workload_run_key


def _execution_payload(invocation: RemoteNodeCall | None) -> dict[str, object]:
    if invocation is None:
        return {"mode": "local"}
    return {
        "compatibility_key": (invocation.compatibility_key or invocation.function_name),
        "function_name": invocation.function_name,
        "metadata": invocation.metadata,
        "mode": "remote",
        "runtime_image_key": invocation.runtime_image_key,
        "uses_gpu": invocation.uses_gpu,
    }


def _remote_metadata(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("mode") != "remote":
        raise ValueError("Task does not contain remote execution metadata")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Remote Task metadata must be an object")
    if not all(isinstance(key, str) for key in metadata):
        raise ValueError("Remote Task metadata keys must be strings")
    return cast(dict[str, Any], metadata)


def _json_value(value: object) -> Any:
    return orjson.loads(orjson.dumps(value))


def _node_error_message(result: AppRunResult) -> str:
    if result.warnings:
        return result.warnings[0]
    if result.status == AppRunStatus.PARTIAL:
        return "Node returned partial status"
    return "Node returned failed status"


def _app_status_for_node(status: NodeStatus) -> AppRunStatus:
    if status == NodeStatus.SUCCEEDED:
        return AppRunStatus.SUCCEEDED
    if status == NodeStatus.PARTIAL:
        return AppRunStatus.PARTIAL
    if status == NodeStatus.FAILED:
        return AppRunStatus.FAILED
    raise ValueError(f"Node status {status.value} has no App result status")


def _app_result_for_run(
    status: RunStatus,
    message: str | None,
) -> AppRunResult:
    if status == RunStatus.SUCCEEDED:
        app_status = AppRunStatus.SUCCEEDED
    elif status == RunStatus.PARTIAL:
        app_status = AppRunStatus.PARTIAL
    elif status in {RunStatus.SUSPENDED, RunStatus.STATE_UNKNOWN}:
        app_status = AppRunStatus.PARTIAL
    else:
        app_status = AppRunStatus.FAILED
    warnings = [message] if message else []
    return AppRunResult(status=app_status, warnings=warnings)
