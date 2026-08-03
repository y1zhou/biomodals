"""Direct Rosetta App Run adapter for SQLite-backed pull workers."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    validate_task_publication,
)
from biomodals.app.bioinfo.rosetta.execution_request import (
    ROSETTA_TASKS_NODE,
    RosettaExecutionRequest,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionRuntime,
    NodeStatus,
    ProviderBinding,
    ProviderCallSubmission,
    PullTaskClaim,
    TaskPlan,
    TaskStatus,
    form_pull_worker_candidates,
    ready_node_keys,
)
from biomodals.execution.scheduler import (
    PullWorkerDispatchDescriptor,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import (
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
)


class RosettaExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one direct Rosetta request through a durable pull-worker pool."""

    def __init__(
        self,
        *,
        request: RosettaExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        output_root: str | Path,
        pull_worker_coordinator: Any,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to Rosetta's Task publications."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self.output_root = Path(output_root)
        self.pull_worker_coordinator = pull_worker_coordinator
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            transaction=self.store.transaction,
            synchronize=self.store.synchronize,
        )

    @property
    def run_root(self) -> Path:
        """Return the existing app-owned run directory."""
        return self.output_root / self.request.workload_run_key

    def attach(self) -> None:
        """Open and verify this Run without refreshing worker publications."""
        self._initialize(reload_output=False)

    def refresh_publications(self) -> None:
        """Refresh worker publications and verify this Run."""
        self._initialize(reload_output=True)

    def claim_pull_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Checkpoint one idempotent claim before returning Task payloads."""
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
        result: Mapping[str, object],
    ):
        """Validate one worker publication and checkpoint its completion."""
        with self.store.synchronize():
            call = self.store.execution.get_provider_call(provider_call_id)
            task = self.store.execution.get_task(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
                task_key,
            )
        if call.node_key != ROSETTA_TASKS_NODE:
            raise ValueError("Provider Call does not belong to Rosetta Tasks")
        status = result.get("status")
        message = result.get("error")
        if not isinstance(status, str):
            raise TypeError("Rosetta worker result has no status")
        if message is not None and not isinstance(message, str):
            raise TypeError("Rosetta worker error must be text")
        observation = AvailabilityStatus.MISSING
        if status == "succeeded":
            spec = self._task_specs()[task_key]
            try:
                observation = (
                    AvailabilityStatus.AVAILABLE
                    if validate_task_publication(
                        self.run_root,
                        spec,
                        task.fingerprint,
                    )
                    else AvailabilityStatus.MISSING
                )
            except OSError:
                observation = AvailabilityStatus.UNKNOWN
            if observation == AvailabilityStatus.MISSING:
                message = "Rosetta worker returned without a valid publication"
            elif observation == AvailabilityStatus.UNKNOWN:
                message = "Rosetta worker publication could not be validated"
        elif status != "failed":
            raise ValueError(f"Unknown Rosetta worker status {status!r}")
        return self._provider.record_pull_task_completion(
            provider_call_id,
            task_key,
            request_id=request_id,
            observation=observation,
            message=message,
            now=self._now(),
        )

    def advance_once(self) -> None:
        """Apply one result-driven recovery and greedy admission cycle."""
        # Pull workers complete Tasks through coordinator callbacks.
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=lambda: None,
            start_ready_nodes=lambda _required: self._start_ready_node(),
            admit_remote_tasks=self._admit_pull_workers,
            now=self._now,
        )

    def _initialize(self, *, reload_output: bool = False):
        if reload_output:
            self._reload_output()
        self._provider.create_or_verify_run(
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=self.predecessor_execution_run_id,
            plan=self.request.execution_plan,
            deployment=self.deployment,
            max_active_provider_calls=self.request.max_active_provider_calls,
            max_active_gpu_provider_calls=0,
            now=self._now(),
        )
        return self.store.execution

    def _recover_publications(self) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            node = repository.get_node(self.execution_run_id, ROSETTA_TASKS_NODE)
        if node.status == NodeStatus.PENDING:
            observation = self._node_observation()
            with self.store.transaction():
                repository = self.store.execution
                if repository.get_node(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                ).status.is_terminal:
                    return
                repository.record_node_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    observation,
                    now=self._now(),
                )
            return
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        specs = self._task_specs()
        with self.store.synchronize():
            tasks = self.store.execution.list_tasks(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
            )
        observations = []
        for task in tasks:
            if task.status.is_terminal:
                continue
            try:
                observation = (
                    AvailabilityStatus.AVAILABLE
                    if validate_task_publication(
                        self.run_root,
                        specs[task.task_key],
                        task.fingerprint,
                    )
                    else AvailabilityStatus.MISSING
                )
            except OSError:
                observation = AvailabilityStatus.UNKNOWN
            observations.append((task.task_key, observation))
        if not observations:
            return
        with self.store.transaction():
            repository = self.store.execution
            for task_key, observation in observations:
                if repository.get_task(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    task_key,
                ).status.is_terminal:
                    continue
                repository.record_task_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    task_key,
                    observation,
                    now=self._now(),
                )

    def _node_observation(self) -> AvailabilityStatus:
        try:
            available = all(
                validate_task_publication(
                    self.run_root,
                    spec,
                    self._task_plan(spec).fingerprint(
                        workload_plan_fingerprint=(
                            self.request.execution_plan.workload_plan_fingerprint
                        ),
                        node_key=ROSETTA_TASKS_NODE,
                    ),
                )
                for spec in self.request.tasks
            )
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=_result_envelope,
            now=self._now(),
        )

    def _start_ready_node(self) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            statuses = {
                node.node_key: node.status
                for node in repository.list_nodes(self.execution_run_id)
            }
            plan = repository.get_run(self.execution_run_id).plan
        if ROSETTA_TASKS_NODE not in ready_node_keys(
            plan,
            statuses,
        ):
            return
        plans = tuple(self._task_plan(task) for task in self.request.tasks)
        with self.store.transaction():
            repository = self.store.execution
            if repository.get_node(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
            ).status.is_terminal:
                return
            repository.start_node(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
                now=self._now(),
            )
            records = repository.discover_tasks(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
                plans,
                now=self._now(),
            )
        observations = []
        for record, spec in zip(records, self.request.tasks, strict=True):
            try:
                observation = (
                    AvailabilityStatus.AVAILABLE
                    if validate_task_publication(
                        self.run_root,
                        spec,
                        record.fingerprint,
                    )
                    else AvailabilityStatus.MISSING
                )
            except OSError:
                observation = AvailabilityStatus.UNKNOWN
            observations.append((record.task_key, observation))
        with self.store.transaction():
            repository = self.store.execution
            for task_key, observation in observations:
                if repository.get_task(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    task_key,
                ).status.is_terminal:
                    continue
                repository.record_task_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    task_key,
                    observation,
                    now=self._now(),
                )

    def _admit_pull_workers(self, required: set[str]) -> None:
        if ROSETTA_TASKS_NODE not in required:
            return
        with self.store.synchronize():
            repository = self.store.execution
            node = repository.get_node(self.execution_run_id, ROSETTA_TASKS_NODE)
            run = repository.get_run(self.execution_run_id)
            calls = tuple(
                call
                for call in repository.list_provider_calls(self.execution_run_id)
                if call.node_key == ROSETTA_TASKS_NODE
            )
            tasks = repository.list_tasks(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
            )
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        rank = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys={ROSETTA_TASKS_NODE},
        )[ROSETTA_TASKS_NODE]
        binding = ProviderBinding(
            environment=run.deployment.environment,
            app_name=run.deployment.deployment_name,
            app_version=run.deployment.deployment_version,
            function_name="run_rosetta_worker",
            uses_gpu=False,
            runtime_image_key="rosetta-cpu",
        )
        unfinished = sum(
            task.status in {TaskStatus.PENDING, TaskStatus.RUNNING} for task in tasks
        )
        descriptor = PullWorkerDispatchDescriptor(
            node_key=ROSETTA_TASKS_NODE,
            node_ordinal=node.ordinal,
            binding=binding,
            compatibility_key="rosetta-worker",
            claim_capacity=self.request.claim_capacity,
            unfinished_task_count=unfinished,
            nonterminal_worker_count=sum(not call.status.is_terminal for call in calls),
            next_worker_ordinal=len(calls),
            depth=rank.depth,
            unblocking_span=rank.unblocking_span,
        )
        descriptor = self._provider.persist_pull_worker_dispatch_policy(
            self.execution_run_id,
            descriptor,
            now=self._now(),
        )
        with self.store.synchronize():
            counts = self.store.execution.active_provider_call_counts(
                self.execution_run_id
            )
        selected = select_admissible_candidates(
            form_pull_worker_candidates((descriptor,)),
            available_total_slots=max(
                0,
                run.max_active_provider_calls - counts.total,
            ),
            available_gpu_slots=0,
        )
        submitted = self._provider.submit_provider_calls(
            self.execution_run_id,
            tuple(
                ProviderCallSubmission(
                    candidate=candidate,
                    submission_token=candidate.candidate_key,
                    claim_capacity=self.request.claim_capacity,
                    provider_call_id_kwarg="provider_call_id",
                    kwargs={
                        "coordinator": self.pull_worker_coordinator,
                        "run_name": self.request.run_name,
                        "run_id": self.request.run_id,
                        "claim_capacity": self.request.claim_capacity,
                        "max_parallel": self.request.max_parallel_per_worker,
                    },
                )
                for candidate in selected
            ),
            now=self._now(),
        )
        if any(call is None for call in submitted):
            return

    def _task_specs(self) -> dict[str, RosettaTaskSpec]:
        return {task.task_key: task for task in self.request.tasks}

    @staticmethod
    def _task_plan(task: RosettaTaskSpec) -> TaskPlan:
        return TaskPlan(
            task_key=task.task_key,
            scientific_payload=task.scientific_payload,
            execution_payload=task.to_dict(),
        )


def _result_envelope(result: object) -> dict[str, object]:
    """Persist only a bounded JSON-compatible pull-worker summary."""
    try:
        value = orjson.loads(orjson.dumps(result))
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError("Rosetta worker returned a non-JSON result") from error
    return {"result": value}
