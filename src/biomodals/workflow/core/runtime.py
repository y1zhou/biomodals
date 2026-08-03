"""Workflow-owned adaptation of the shared execution kernel."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import UUID

import orjson
from pydantic import BaseModel

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    DispatchMode,
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
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import AppRunResult, AppRunStatus, WorkflowArtifact
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
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> AppRunResult:
        """Create or recover this Run and drive it until it cannot advance."""
        with synchronize():
            repository = self._initialize(workload_run_key)
        snapshot = drive_execution_run(
            repository,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )
        return _app_result_for_run(snapshot.run.status, snapshot.run.status_message)

    def resume(
        self,
        *,
        workload_run_key: str,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> AppRunResult:
        """Explicitly resume this persisted Run, then drive it."""
        with synchronize():
            repository = self._initialize(workload_run_key)
        resume_execution_run(
            repository,
            self.execution_run_id,
            reconcile_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            synchronize=synchronize,
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
            synchronize=synchronize,
        )
        return _app_result_for_run(snapshot.run.status, snapshot.run.status_message)

    def advance_once(self) -> None:
        """Apply one caller-driven workflow scheduling cycle."""
        definition = self._require_definition()
        self._recover_publications()
        self._reconcile_nodes_and_run()

        with self.store.synchronize():
            run = self.store.execution.get_run(self.execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            self._reconcile_provider_calls(set(run.plan.node_keys))
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status == RunStatus.STATE_UNKNOWN:
            required = self._required_nodes()
            if required is None:
                required_nodes = set(run.plan.node_keys)
            else:
                required_nodes = required
                for provider_call_id in self._prune_unrequired(required):
                    self._provider.request_provider_call_cancellation(
                        provider_call_id,
                        now=self._now(),
                    )
            self._reconcile_provider_calls(required_nodes)
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self._required_nodes()
        if required is None:
            return

        calls_to_cancel = self._prune_unrequired(required)
        for provider_call_id in calls_to_cancel:
            self._provider.request_provider_call_cancellation(
                provider_call_id,
                now=self._now(),
            )

        self._reconcile_provider_calls(required)
        self._reconcile_nodes_and_run()
        with self.store.synchronize():
            can_continue = self.store.execution.get_run(
                self.execution_run_id
            ).status in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
            }
        if not can_continue:
            return

        self._start_ready_nodes(definition)
        self._run_local_tasks(definition)
        required = self._required_nodes()
        if required is not None:
            self._admit_remote_tasks(definition, required)
        self._reconcile_nodes_and_run()

    def cancel(self) -> None:
        """Request cancellation through the shared provider lifecycle."""
        self._provider.cancel_run(self.execution_run_id, now=self._now())

    def attach(self, *, workload_run_key: str) -> None:
        """Open and verify a Run without refreshing worker publications."""
        self._initialize(workload_run_key, reload_volume=False)

    def refresh_publications(self, *, workload_run_key: str) -> None:
        """Refresh worker publications and verify an existing Run."""
        self._initialize(workload_run_key, reload_volume=True)

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

    def complete_pull_task(
        self,
        provider_call_id: UUID,
        task_key: str,
        *,
        request_id: str,
        result: AppRunResult,
    ) -> ExecutionTaskRecord:
        """Publish one worker result and checkpoint its idempotent completion."""
        result = AppRunResult.model_validate(result)
        with self.store.synchronize():
            call = self.store.execution.get_provider_call(provider_call_id)
            task = self.store.execution.get_task(
                self.execution_run_id,
                call.node_key,
                task_key,
            )
        node = self._require_definition().nodes[call.node_key].node
        if not isinstance(node, RemotePullTaskWorkflowNode):
            raise ValueError("Provider Call does not belong to a pull-worker Node")
        if result.status != AppRunStatus.SUCCEEDED:
            with self.store.synchronize():
                with self.store.transaction():
                    completed = self.store.execution.record_pull_task_completion(
                        provider_call_id,
                        task_key,
                        request_id=request_id,
                        observation=AvailabilityStatus.MISSING,
                        message=_node_error_message(result),
                        now=self._now(),
                    )
                self._checkpoint()
            return completed

        context = self._node_context(
            self._require_definition(),
            call.node_key,
            task_key=task_key,
        )
        materialized = materialize_app_run_result(
            result=result,
            workflow_volume_name=self.workflow_volume_name,
            result_dir=context.work_dir,
            artifact_dir=self.store.output_root / "artifacts",
            producing_node_id=call.node_key,
            artifact_id_scope=task_key,
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
        )
        with self.store.synchronize():
            with self.store.transaction():
                self.store.artifacts.record_task_publication(
                    call.node_key,
                    task_key,
                    task_fingerprint=task.fingerprint,
                    result=materialized.result,
                    artifacts=artifacts,
                    now=self._now(),
                )
                completed = self.store.execution.record_pull_task_completion(
                    provider_call_id,
                    task_key,
                    request_id=request_id,
                    observation=observation,
                    message=(
                        "Published workflow Task result is unavailable"
                        if observation == AvailabilityStatus.MISSING
                        else None
                    ),
                    now=self._now(),
                )
            self._checkpoint()
        return completed

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
                node.node_key: repository.list_tasks(
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
                        if task.status.is_terminal:
                            continue
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
                    task = next(
                        task
                        for task in tasks_by_node[node.node_key]
                        if task.task_key == _TASK_KEY
                    )
                    if not task.status.is_terminal:
                        observation = observations[node.node_key]
                        if observation is None:
                            raise RuntimeError(
                                f"Workflow Node {node.node_key!r} was not probed"
                            )
                        task_observations.append((
                            node.node_key,
                            _TASK_KEY,
                            observation,
                        ))

        if not task_observations:
            return
        with self.store.transaction():
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

    def _required_nodes(self) -> set[str] | None:
        required = self._provider.required_node_keys(self.execution_run_id)
        return None if required is None else set(required)

    def _prune_unrequired(self, required: set[str]) -> tuple[UUID, ...]:
        with self.store.synchronize():
            with self.store.transaction():
                calls = self.store.execution.prune_unrequired_nodes(
                    self.execution_run_id,
                    required_node_keys=required,
                    now=self._now(),
                )
            if calls:
                self._checkpoint()
        return calls

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        reconciled = self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=self._prepare_result_envelope,
            finalize_result=self._finalize_result_envelope,
            discard_result=self._discard_prepared_result,
            now=self._now(),
        )
        if any(
            not original.status.is_terminal
            and updated.status == ProviderCallStatus.SUCCEEDED
            and updated.dispatch_mode != DispatchMode.PULL_WORKER
            for original, updated in reconciled
        ):
            self._reload_volume()
        for _, call in reconciled:
            if (
                call.status != ProviderCallStatus.SUCCEEDED
                or call.node_key not in required
            ):
                continue
            self._publish_provider_result(call)

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

    def _run_local_tasks(self, definition: WorkflowDefinition) -> None:
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
                    unfinished_task_count = self.store.execution.unfinished_task_count(
                        self.execution_run_id,
                        node_id,
                    )
                    total_workers, nonterminal_workers = (
                        self.store.execution.provider_call_counts_for_node(
                            self.execution_run_id,
                            node_id,
                        )
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
        submitted = self._provider.submit_provider_calls(
            self.execution_run_id,
            tuple(submissions),
            now=self._now(),
        )
        if any(call is None for call in submitted):
            return

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
        observation = self._artifact_observation(tuple(materialized.artifacts))
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
            artifact_id_scope=task_key,
            volume_root=self.volume_root,
        )
        observation = self._artifact_observation(tuple(materialized.artifacts))
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

    def _finalize_remote_task_node(
        self,
        node_id: str,
        aggregation_policy: NodeAggregationPolicy,
        allow_empty_result: bool,
        implementation: RemoteTaskWorkflowNode,
    ) -> None:
        with self.store.transaction():
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
                    self.store.artifacts.discard_node_publication(node_id)
            elif observation == AvailabilityStatus.UNKNOWN:
                with self.store.transaction():
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
            observation = self._artifact_observation(artifacts)
        except Exception as error:
            self._fail_node_publication(
                node_id,
                f"Could not finalize workflow Node: {error}",
            )
            return

        with self.store.transaction():
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
    ) -> AvailabilityStatus:
        """Combine durable workflow artifacts with a workload-specific probe."""
        artifact_observation = self._artifact_observation(artifacts)
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
    ) -> AvailabilityStatus:
        statuses = [
            check_artifact_availability(
                artifact,
                workflow_volume_name=self.workflow_volume_name,
                volume_root=self.volume_root,
                external_artifact_checker=self.external_artifact_checker,
            ).status
            for artifact in artifacts
        ]
        if AvailabilityStatus.UNKNOWN in statuses:
            return AvailabilityStatus.UNKNOWN
        if AvailabilityStatus.MISSING in statuses:
            return AvailabilityStatus.MISSING
        return AvailabilityStatus.AVAILABLE

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
        task_key: str = _TASK_KEY,
    ) -> NodeRunContext:
        spec = definition.nodes[node_id]
        with self.store.synchronize():
            inputs = {
                input_name: list(self.store.artifacts.select_artifacts(selector))
                for input_name, selector in spec.inputs.items()
            }
        node_root = self.store.output_root / "nodes" / node_id
        if task_key == _TASK_KEY:
            work_dir = node_root / "result"
            cache_dir = node_root / "cache"
        else:
            task_root = node_root / "tasks" / sanitize_filename(task_key)
            work_dir = task_root / "result"
            cache_dir = task_root / "cache"
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return NodeRunContext(
            execution_run_id=self.execution_run_id,
            workload_run_key=self._require_workload_run_key(),
            node_id=node_id,
            task_key=task_key,
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
