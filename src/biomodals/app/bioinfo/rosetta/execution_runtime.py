"""Direct Rosetta App Run adapter for SQLite-backed pull workers."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
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
    ExecutionSnapshot,
    NodeStatus,
    ProviderBinding,
    PullTaskClaim,
    RunStatus,
    TaskPlan,
    TaskStatus,
    drive_execution_run,
    form_pull_worker_candidates,
    ready_node_keys,
    required_node_keys,
    resume_execution_run,
)
from biomodals.execution.scheduler import (
    PullWorkerDispatchDescriptor,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import AppExecutionRunStore


class RosettaExecutionRuntime:
    """Drive one direct Rosetta request through a durable pull-worker pool."""

    def __init__(
        self,
        *,
        request: RosettaExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: AppExecutionRunStore,
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
        self.output_root = Path(output_root)
        self.pull_worker_coordinator = pull_worker_coordinator
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
        )

    @property
    def run_root(self) -> Path:
        """Return the existing app-owned run directory."""
        return self.output_root / self.request.workload_run_key

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
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def resume(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive Task failures."""
        with synchronize():
            repository = self._initialize()
            resume_execution_run(
                repository,
                self.execution_run_id,
                checkpoint=self._checkpoint,
                now=self._now(),
            )
        return drive_execution_run(
            self.store.execution,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def cancel(self) -> ExecutionSnapshot:
        """Request explicit cancellation without releasing uncertain owners."""
        self._provider.repository = self.store.execution
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        """Close local SQLite state without cancelling Provider Calls."""
        self.store.close()

    def attach(self) -> None:
        """Open and verify this Run for concurrent worker callbacks."""
        self._initialize()

    def claim_pull_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
    ) -> PullTaskClaim:
        """Checkpoint one idempotent claim before returning Task payloads."""
        self._provider.repository = self.store.execution
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
        call = self.store.execution.get_provider_call(provider_call_id)
        if call.node_key != ROSETTA_TASKS_NODE:
            raise ValueError("Provider Call does not belong to Rosetta Tasks")
        task = self.store.execution.get_task(
            self.execution_run_id,
            ROSETTA_TASKS_NODE,
            task_key,
        )
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
        self._provider.repository = self.store.execution
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
        self._reload_output()
        self._recover_publications()
        self._reconcile_nodes_and_run()
        run = self.store.execution.get_run(self.execution_run_id)
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self._required_nodes()
        if required is None:
            return
        for provider_call_id in self._prune_unrequired(required):
            self._provider.repository = self.store.execution
            self._provider.request_provider_call_cancellation(
                provider_call_id,
                now=self._now(),
            )
        self._reconcile_provider_calls(set(required))
        self._recover_publications()
        self._reconcile_nodes_and_run()
        if self.store.execution.get_run(self.execution_run_id).status not in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
        }:
            return
        self._start_ready_node()
        self._recover_publications()
        required = self._required_nodes()
        if required is not None:
            self._admit_pull_workers(set(required))
        self._reconcile_nodes_and_run()

    def _initialize(self):
        self._reload_output()
        repository = self.store.execution
        try:
            existing = repository.get_run(self.execution_run_id)
        except LookupError:
            with self.store.transaction():
                repository.create_run(
                    execution_run_id=self.execution_run_id,
                    predecessor_execution_run_id=self.predecessor_execution_run_id,
                    plan=self.request.execution_plan,
                    deployment=self.deployment,
                    max_active_provider_calls=(self.request.max_active_provider_calls),
                    max_active_gpu_provider_calls=0,
                    now=self._now(),
                )
            return self._checkpoint()
        if (
            existing.plan != self.request.execution_plan
            or existing.predecessor_execution_run_id
            != self.predecessor_execution_run_id
            or existing.deployment != self.deployment
            or existing.max_active_provider_calls
            != self.request.max_active_provider_calls
            or existing.max_active_gpu_provider_calls != 0
        ):
            raise ValueError("Rosetta request does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        repository = self.store.execution
        node = repository.get_node(self.execution_run_id, ROSETTA_TASKS_NODE)
        if node.status == NodeStatus.PENDING:
            observation = self._node_observation()
            with self.store.transaction():
                repository.record_node_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    observation,
                    now=self._now(),
                )
            self._checkpoint()
            return
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        specs = self._task_specs()
        observations = []
        for task in repository.list_tasks(
            self.execution_run_id,
            ROSETTA_TASKS_NODE,
        ):
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
            for task_key, observation in observations:
                repository.record_task_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    task_key,
                    observation,
                    now=self._now(),
                )
        self._checkpoint()

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

    def _required_nodes(self) -> tuple[str, ...] | None:
        node = self.store.execution.get_node(
            self.execution_run_id,
            ROSETTA_TASKS_NODE,
        )
        observation = (
            AvailabilityStatus.AVAILABLE
            if node.status == NodeStatus.SUCCEEDED
            else node.result_observation or AvailabilityStatus.MISSING
        )
        return required_node_keys(
            self.store.execution.get_run(self.execution_run_id).plan,
            {ROSETTA_TASKS_NODE: observation},
        )

    def _prune_unrequired(self, required: tuple[str, ...]) -> tuple[UUID, ...]:
        with self.store.transaction():
            calls = self.store.execution.prune_unrequired_nodes(
                self.execution_run_id,
                required_node_keys=set(required),
                now=self._now(),
            )
        if calls:
            self._checkpoint()
        return calls

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if call.status.is_terminal:
                continue
            self._provider.repository = self.store.execution
            self._provider.reconcile_provider_call(
                call.provider_call_id,
                encode_result=_result_envelope,
                result_already_satisfied=call.node_key not in required,
                now=self._now(),
            )

    def _start_ready_node(self) -> None:
        repository = self.store.execution
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(self.execution_run_id)
        }
        if ROSETTA_TASKS_NODE not in ready_node_keys(
            repository.get_run(self.execution_run_id).plan,
            statuses,
        ):
            return
        plans = tuple(self._task_plan(task) for task in self.request.tasks)
        with self.store.transaction():
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
                repository.record_task_result_observation(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    record.task_key,
                    observation,
                    now=self._now(),
                )
        self._checkpoint()

    def _admit_pull_workers(self, required: set[str]) -> None:
        if ROSETTA_TASKS_NODE not in required:
            return
        repository = self.store.execution
        node = repository.get_node(self.execution_run_id, ROSETTA_TASKS_NODE)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        run = repository.get_run(self.execution_run_id)
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
        calls = [
            call
            for call in repository.list_provider_calls(self.execution_run_id)
            if call.node_key == ROSETTA_TASKS_NODE
        ]
        unfinished = sum(
            task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
            for task in repository.list_tasks(
                self.execution_run_id,
                ROSETTA_TASKS_NODE,
            )
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
        counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = select_admissible_candidates(
            form_pull_worker_candidates((descriptor,)),
            available_total_slots=max(
                0,
                run.max_active_provider_calls - counts.total,
            ),
            available_gpu_slots=0,
        )
        for candidate in selected:
            self._provider.repository = self.store.execution
            submitted = self._provider.submit_pull_worker(
                self.execution_run_id,
                node_key=ROSETTA_TASKS_NODE,
                submission_token=candidate.candidate_key,
                binding=binding,
                compatibility_key="rosetta-worker",
                claim_capacity=self.request.claim_capacity,
                kwargs={
                    "coordinator": self.pull_worker_coordinator,
                    "run_name": self.request.run_name,
                    "run_id": self.request.run_id,
                    "claim_capacity": self.request.claim_capacity,
                    "max_parallel": self.request.max_parallel_per_worker,
                },
                now=self._now(),
            )
            if submitted is None:
                return

    def _reconcile_nodes_and_run(self) -> None:
        repository = self.store.execution
        node = repository.get_node(self.execution_run_id, ROSETTA_TASKS_NODE)
        if node.status == NodeStatus.RUNNING and node.discovery_complete:
            with self.store.transaction():
                repository.reconcile_node_tasks(
                    self.execution_run_id,
                    ROSETTA_TASKS_NODE,
                    now=self._now(),
                )
        with self.store.transaction():
            repository.finalize_run_from_results(
                self.execution_run_id,
                now=self._now(),
            )

    def _task_specs(self) -> dict[str, RosettaTaskSpec]:
        return {task.task_key: task for task in self.request.tasks}

    @staticmethod
    def _task_plan(task: RosettaTaskSpec) -> TaskPlan:
        return TaskPlan(
            task_key=task.task_key,
            scientific_payload=task.scientific_payload,
            execution_payload=task.to_dict(),
        )

    def _checkpoint(self):
        with self.store.closed_for_volume_sync():
            self.output_volume.commit()
        repository = self.store.execution
        self._provider.repository = repository
        return repository

    def _reload_output(self) -> None:
        with self.store.closed_for_volume_sync():
            self.output_volume.reload()
        self._provider.repository = self.store.execution


def _result_envelope(result: object) -> dict[str, object]:
    """Persist only a bounded JSON-compatible pull-worker summary."""
    try:
        value = orjson.loads(orjson.dumps(result))
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError("Rosetta worker returned a non-JSON result") from error
    return {"result": value}
