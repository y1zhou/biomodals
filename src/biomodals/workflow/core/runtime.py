"""Workflow-owned adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

import orjson
from pydantic import BaseModel

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionRuntime,
    NodeAggregationPolicy,
    NodeStatus,
    ProviderBinding,
    ProviderCallRecord,
    ProviderCallStatus,
    RunStatus,
    RunStatusReason,
    SqliteExecutionRepository,
    TaskPlan,
    TaskStatus,
    aggregate_task_outcome,
    drive_execution_run,
    ready_node_keys,
    required_node_keys,
    resume_execution_run,
)
from biomodals.execution.modal import ModalCallDriver
from biomodals.execution.scheduler import (
    TaskDispatchDescriptor,
    form_fixed_batches,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import AppRunResult, AppRunStatus, WorkflowArtifact
from biomodals.workflow.core._runtime.volume_sync import (
    WorkflowVolume,
    WorkflowVolumeSync,
)
from biomodals.workflow.core.artifact_availability import (
    ArtifactAvailabilityStatus,
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
    RemoteTaskWorkflowNode,
    RemoteWorkflowNode,
    RemoteWorkflowTask,
)
from biomodals.workflow.core.run_store import WorkflowRunStore

_TASK_KEY = "node"


class WorkflowModalDriver(Protocol):
    """Modal call operations used by the workflow adapter."""

    def resolve(self, binding: ProviderBinding) -> Any:
        """Resolve an exact deployed function."""
        ...

    def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Submit one function invocation."""
        ...

    def observe(self, provider_call_handle_id: str) -> Any:
        """Observe one retained provider call."""
        ...

    def cancel(self, provider_call_handle_id: str) -> None:
        """Request provider-call cancellation."""
        ...


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
        workflow_volume: WorkflowVolume | None = None,
        modal_driver: WorkflowModalDriver | None = None,
        max_active_provider_calls: int = 32,
        max_active_gpu_provider_calls: int | None = None,
        strict_external_artifact_checks: bool = False,
        external_artifact_checker: ExternalArtifactChecker | None = None,
        external_volume_roots: Mapping[str, str | Path] | None = None,
        now: Callable[[], int] | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Bind workflow code to one opaque Execution Run identity."""
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
        self.max_active_provider_calls = max_active_provider_calls
        self.max_active_gpu_provider_calls = (
            max_active_provider_calls
            if max_active_gpu_provider_calls is None
            else max_active_gpu_provider_calls
        )
        self.external_artifact_checker = external_artifact_checker
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self.store = WorkflowRunStore(self.volume_root, execution_run_id)
        self._volume_sync = WorkflowVolumeSync(
            workflow_volume=workflow_volume,
            ledger=self.store,
        )
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver or ModalCallDriver(),
            checkpoint=self._checkpoint,
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
                checkpoint=self._checkpoint,
                now=self._now(),
            )
        snapshot = drive_execution_run(
            self.store.execution,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )
        return _app_result_for_run(snapshot.run.status, snapshot.run.status_message)

    def advance_once(self) -> None:
        """Apply one caller-driven workflow scheduling cycle."""
        definition = self._require_definition()
        self._volume_sync.reload()
        self._provider.repository = self.store.execution
        self._recover_publications()
        self._reconcile_nodes_and_run()

        run = self.store.execution.get_run(self.execution_run_id)
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self._required_nodes()
        if required is None:
            return

        calls_to_cancel = self._prune_unrequired(required)
        for provider_call_id in calls_to_cancel:
            self._provider.repository = self.store.execution
            self._provider.request_provider_call_cancellation(
                provider_call_id,
                now=self._now(),
            )

        self._reconcile_provider_calls(required)
        self._reconcile_nodes_and_run()
        if self.store.execution.get_run(self.execution_run_id).status not in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
        }:
            return

        self._start_ready_nodes(definition)
        self._run_local_tasks(definition)
        required = self._required_nodes()
        if required is not None:
            self._admit_remote_tasks(definition, required)
        self._reconcile_nodes_and_run()

    def cancel(self) -> None:
        """Request cancellation through the shared provider lifecycle."""
        self._provider.repository = self.store.execution
        self._provider.cancel_run(self.execution_run_id, now=self._now())

    def close(self) -> None:
        """Close local resources without cancelling attached child calls."""
        self.store.close()

    def _initialize(self, workload_run_key: str) -> SqliteExecutionRepository:
        """Load the workflow definition and create or verify its Run."""
        definition = self.workflow.validate()
        self._definition = definition
        self._workload_run_key = workload_run_key
        self._volume_sync.reload()
        return self._ensure_run(definition, workload_run_key)

    def _ensure_run(
        self,
        definition: WorkflowDefinition,
        workload_run_key: str,
    ) -> SqliteExecutionRepository:
        plan = execution_plan(
            definition,
            workload_run_key=workload_run_key,
        )
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
            return self._checkpoint()

        if existing.plan.workload_plan_fingerprint != plan.workload_plan_fingerprint:
            raise ValueError("Workflow Plan Fingerprint does not match Execution Run")
        if existing.plan.workload_run_key != workload_run_key:
            raise ValueError("Workload Run Key does not match Execution Run")
        if existing.deployment != self.deployment:
            raise ValueError("Deployment Identity does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        repository = self.store.execution
        definition = self._require_definition()
        node_observations: list[tuple[str, AvailabilityStatus]] = []
        task_observations: list[tuple[str, str, AvailabilityStatus]] = []
        for node in repository.list_nodes(self.execution_run_id):
            observation = self._publication_observation(node.node_key)
            if node.status == NodeStatus.PENDING:
                node_observations.append((node.node_key, observation))
            elif node.status == NodeStatus.RUNNING and node.discovery_complete:
                implementation = definition.nodes[node.node_key].node
                if isinstance(implementation, RemoteTaskWorkflowNode):
                    for task in repository.list_tasks(
                        self.execution_run_id,
                        node.node_key,
                    ):
                        if task.status.is_terminal:
                            continue
                        task_observations.append((
                            node.node_key,
                            task.task_key,
                            self._task_publication_observation(
                                node.node_key,
                                task.task_key,
                                task.fingerprint,
                            ),
                        ))
                else:
                    task = repository.get_task(
                        self.execution_run_id,
                        node.node_key,
                        _TASK_KEY,
                    )
                    if not task.status.is_terminal:
                        task_observations.append((
                            node.node_key,
                            _TASK_KEY,
                            observation,
                        ))

        if not node_observations and not task_observations:
            return
        with self.store.transaction():
            for node_id, observation in node_observations:
                if observation == AvailabilityStatus.MISSING:
                    self.store.artifacts.discard_node_publication(node_id)
                self.store.execution.record_node_result_observation(
                    self.execution_run_id,
                    node_id,
                    observation,
                    now=self._now(),
                )
            for node_id, task_key, observation in task_observations:
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
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        observations: dict[str, AvailabilityStatus] = {}
        for node in repository.list_nodes(self.execution_run_id):
            if node.status == NodeStatus.SUCCEEDED:
                observations[node.node_key] = AvailabilityStatus.AVAILABLE
            elif node.result_observation is not None:
                observations[node.node_key] = node.result_observation
            else:
                observations[node.node_key] = AvailabilityStatus.MISSING
        required = required_node_keys(run.plan, observations)
        return None if required is None else set(required)

    def _prune_unrequired(self, required: set[str]) -> tuple[UUID, ...]:
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
        calls = self.store.execution.list_provider_calls(self.execution_run_id)
        for original in calls:
            call = original
            if not call.status.is_terminal:
                self._provider.repository = self.store.execution
                call = self._provider.reconcile_provider_call(
                    call.provider_call_id,
                    encode_result=_result_envelope,
                    result_already_satisfied=call.node_key not in required,
                    now=self._now(),
                )
            if (
                call.status == ProviderCallStatus.SUCCEEDED
                and call.node_key in required
            ):
                self._publish_provider_result(call)

    def _publish_provider_result(
        self,
        call: ProviderCallRecord,
    ) -> None:
        node_id = call.node_key
        envelope = call.result_envelope
        node = self._require_definition().nodes[node_id].node
        if isinstance(node, RemoteTaskWorkflowNode):
            self._publish_provider_task_results(
                node_id,
                call.task_keys,
                envelope,
                node,
            )
            return
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
            raw_result = _raw_result(envelope)
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
                _raw_result(envelope),
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
        repository = self.store.execution
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(self.execution_run_id)
        }
        ready = ready_node_keys(
            repository.get_run(self.execution_run_id).plan,
            statuses,
        )
        if not ready:
            return

        prepared = [self._prepare_node(definition, node_id) for node_id in ready]
        with self.store.transaction():
            for item in prepared:
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
        self._checkpoint()

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
                            observation=self._task_publication_observation(
                                node_id,
                                task.task_key,
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
        for node_record in self.store.execution.list_nodes(self.execution_run_id):
            if (
                node_record.status != NodeStatus.RUNNING
                or not node_record.discovery_complete
            ):
                continue
            node = definition.nodes[node_record.node_key].node
            if isinstance(node, RemoteWorkflowNode | RemoteTaskWorkflowNode):
                continue
            task = self.store.execution.get_task(
                self.execution_run_id,
                node_record.node_key,
                _TASK_KEY,
            )
            if task.status.is_terminal:
                continue
            with self.store.transaction():
                acquired = self.store.execution.acquire_local_task(
                    self.execution_run_id,
                    node_record.node_key,
                    _TASK_KEY,
                    now=self._now(),
                )
            if not acquired:
                continue
            self._checkpoint()
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
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        nodes = {
            node.node_key: node for node in repository.list_nodes(run.execution_run_id)
        }
        unfinished = {
            node.node_key for node in nodes.values() if not node.status.is_terminal
        }
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys=unfinished,
        )
        invocations: dict[tuple[str, str], RemoteNodeCall] = {}
        descriptors: list[TaskDispatchDescriptor] = []
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
            for task in repository.list_tasks(self.execution_run_id, node_id):
                if (
                    task.status != TaskStatus.PENDING
                    or task.result_observation != AvailabilityStatus.MISSING
                ):
                    continue
                try:
                    if isinstance(node, RemoteTaskWorkflowNode):
                        invocation = node.prepare_remote_task(
                            self._node_context(
                                definition,
                                node_id,
                                task_key=task.task_key,
                            ),
                            RemoteWorkflowTask(
                                task_key=task.task_key,
                                scientific_payload=task.scientific_payload,
                                execution_payload=task.execution_payload,
                            ),
                        )
                        _json_value(_execution_payload(invocation))
                    else:
                        invocation = node.prepare_remote(
                            self._node_context(definition, node_id)
                        )
                        payload = _json_value(_execution_payload(invocation))
                        if payload != task.execution_payload:
                            raise ValueError(
                                "Remote Node preparation changed after Task discovery"
                            )
                except Exception as error:
                    self._fail_discovered_task(
                        node_id,
                        task.task_key,
                        f"Could not prepare provider call: {error}",
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
                invocations[(node_id, task.task_key)] = invocation
                descriptors.append(
                    TaskDispatchDescriptor(
                        node_key=node_id,
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
                )

        counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = select_admissible_candidates(
            form_fixed_batches(tuple(descriptors)),
            available_total_slots=max(
                0,
                run.max_active_provider_calls - counts.total,
            ),
            available_gpu_slots=max(
                0,
                run.max_active_gpu_provider_calls - counts.gpu,
            ),
        )
        for candidate in selected:
            node = definition.nodes[candidate.node_key].node
            if len(candidate.task_keys) == 1:
                invocation = invocations[(candidate.node_key, candidate.task_keys[0])]
            elif isinstance(node, RemoteTaskWorkflowNode):
                task_definitions = tuple(
                    RemoteWorkflowTask(
                        task_key=task.task_key,
                        scientific_payload=task.scientific_payload,
                        execution_payload=task.execution_payload,
                    )
                    for task_key in candidate.task_keys
                    for task in (
                        repository.get_task(
                            self.execution_run_id,
                            candidate.node_key,
                            task_key,
                        ),
                    )
                )
                invocation = node.prepare_remote_task_batch(
                    self._node_context(
                        definition,
                        candidate.node_key,
                        task_key=candidate.task_keys[0],
                    ),
                    task_definitions,
                )
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
                    for task_key in candidate.task_keys:
                        self._fail_discovered_task(
                            candidate.node_key,
                            task_key,
                            "Batched provider preparation changed its dispatch contract",
                        )
                    continue
            else:  # pragma: no cover - scheduler never batches ordinary Nodes
                raise RuntimeError("Only remote Task Nodes may own batched calls")
            self._provider.repository = self.store.execution
            self._provider.submit_fixed_batch(
                self.execution_run_id,
                candidate,
                submission_token=candidate.candidate_key,
                args=invocation.args,
                kwargs=invocation.kwargs,
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
        observation = self._artifact_observation(tuple(materialized.artifacts))
        with self.store.transaction():
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
        self._checkpoint()

    def _publish_task_result(
        self,
        node_id: str,
        task_key: str,
        result: AppRunResult,
    ) -> None:
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
        self._checkpoint()

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
        self._checkpoint()

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
        self._checkpoint()

    def _reconcile_nodes_and_run(self) -> None:
        definition = self._require_definition()
        for node in self.store.execution.list_nodes(self.execution_run_id):
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
            skipped = self.store.execution.apply_task_failure_policy(
                self.execution_run_id,
                node_id,
                now=self._now(),
            )
        if skipped:
            self._checkpoint()
        tasks = self.store.execution.list_tasks(self.execution_run_id, node_id)
        empty_result = not tasks
        outcome = (
            NodeStatus.SUCCEEDED
            if empty_result and allow_empty_result
            else aggregate_task_outcome(
                aggregation_policy,
                tuple(task.status for task in tasks),
            )
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

        existing = self.store.artifacts.load_node_result(node_id)
        if existing is not None:
            observation = self._artifact_observation(
                self.store.artifacts.load_node_output_artifacts(node_id)
            )
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

        results: dict[str, AppRunResult] = {}
        errors: dict[str, str] = {}
        task_artifacts: list[WorkflowArtifact] = []
        for task in tasks:
            if task.status == TaskStatus.SUCCEEDED:
                result = self.store.artifacts.load_task_result(
                    node_id,
                    task.task_key,
                )
                if result is None:
                    self._fail_node_publication(
                        node_id,
                        f"Successful Task {task.task_key!r} has no publication",
                    )
                    return
                results[task.task_key] = result
                task_artifacts.extend(
                    self.store.artifacts.load_task_output_artifacts(
                        node_id,
                        task.task_key,
                    )
                )
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
        self._checkpoint()

    def _fail_node_publication(self, node_id: str, message: str) -> None:
        with self.store.transaction():
            self.store.execution.fail_node(
                self.execution_run_id,
                node_id,
                message=message,
                now=self._now(),
            )
        self._checkpoint()

    def _publication_observation(self, node_id: str) -> AvailabilityStatus:
        result = self.store.artifacts.load_node_result(node_id)
        if result is None or result.status != AppRunStatus.SUCCEEDED:
            return AvailabilityStatus.MISSING
        artifacts = self.store.artifacts.load_node_output_artifacts(node_id)
        return self._artifact_observation(artifacts)

    def _task_publication_observation(
        self,
        node_id: str,
        task_key: str,
        expected_fingerprint: str,
    ) -> AvailabilityStatus:
        result = self.store.artifacts.load_task_result(node_id, task_key)
        if (
            result is None
            or result.status != AppRunStatus.SUCCEEDED
            or self.store.artifacts.load_task_fingerprint(node_id, task_key)
            != expected_fingerprint
        ):
            return AvailabilityStatus.MISSING
        artifacts = self.store.artifacts.load_task_output_artifacts(
            node_id,
            task_key,
        )
        return self._artifact_observation(artifacts)

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
        if ArtifactAvailabilityStatus.UNKNOWN in statuses:
            return AvailabilityStatus.UNKNOWN
        if ArtifactAvailabilityStatus.MISSING in statuses:
            return AvailabilityStatus.MISSING
        return AvailabilityStatus.AVAILABLE

    def _node_context(
        self,
        definition: WorkflowDefinition,
        node_id: str,
        *,
        task_key: str = _TASK_KEY,
    ) -> NodeRunContext:
        spec = definition.nodes[node_id]
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
        self._volume_sync.commit()
        return self.store.execution

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


def _result_envelope(result: object) -> dict[str, object]:
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    return {"result": _json_value(result)}


def _raw_result(envelope: object) -> object:
    if not isinstance(envelope, dict) or "result" not in envelope:
        raise ValueError("Workflow Result Envelope has no result")
    return cast(dict[str, object], envelope)["result"]


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
