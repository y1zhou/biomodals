"""Caller-driven BoltzGen adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.design.boltzgen.execution_contracts import (
    boltzgen_run_root,
    is_boltzgen_run_complete,
    load_collection_publication,
)
from biomodals.app.design.boltzgen.execution_request import (
    COLLECT_RESULTS_NODE,
    DESIGN_RUNS_NODE,
    BoltzGenExecutionRequest,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionRuntime,
    NodeStatus,
    ProviderBinding,
    ProviderCallStatus,
    ProviderCallSubmission,
    TaskPlan,
    ready_node_keys,
    required_node_keys,
    result_probe_frontier,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import (
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
)


@dataclass(frozen=True)
class _PlannedTask:
    plan: TaskPlan
    run_id: str | None = None


class BoltzGenExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one BoltzGen request through direct one-Task GPU calls."""

    def __init__(
        self,
        *,
        request: BoltzGenExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        output_root: str | Path,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the immutable request to its app-owned Volume publications."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self.output_root = Path(output_root)
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

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        # BoltzGen completes Tasks from validated Volume publications.
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=lambda: None,
            start_ready_nodes=lambda _required: self._start_ready_nodes(),
            admit_remote_tasks=self._admit_remote_tasks,
            now=self._now,
        )

    def _initialize(self):
        self._provider.create_or_verify_run(
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=self.predecessor_execution_run_id,
            plan=self.request.execution_plan,
            deployment=self.deployment,
            max_active_provider_calls=self.request.max_active_provider_calls,
            max_active_gpu_provider_calls=(self.request.max_active_gpu_provider_calls),
            now=self._now(),
        )
        return self.store.execution

    def _recover_publications(self) -> None:
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
                (node_key, self._node_observation(node_key)) for node_key in frontier
            ]
            with self.store.transaction():
                repository = self.store.execution
                for node_key, observation in observed:
                    if repository.get_node(
                        self.execution_run_id,
                        node_key,
                    ).status.is_terminal:
                        continue
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node_key,
                        observation,
                        now=self._now(),
                    )
                    observations[node_key] = observation
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

        with self.store.synchronize():
            repository = self.store.execution
            nodes = repository.list_nodes(self.execution_run_id)
            pending_tasks = tuple(
                (node, task)
                for node in nodes
                if (
                    node.node_key in required
                    and node.status == NodeStatus.RUNNING
                    and node.discovery_complete
                )
                for task in repository.list_tasks(
                    self.execution_run_id,
                    node.node_key,
                )
                if not task.status.is_terminal
            )
        task_observations = []
        for node, task in pending_tasks:
            planned = {
                item.plan.task_key: item for item in self._planned_tasks(node.node_key)
            }
            task_observations.append((
                node.node_key,
                task.task_key,
                self._task_observation(
                    node.node_key,
                    planned[task.task_key],
                    task.fingerprint,
                ),
            ))
        if not task_observations:
            return
        with self.store.transaction():
            repository = self.store.execution
            for node_key, task_key, observation in task_observations:
                current_task = repository.get_task(
                    self.execution_run_id,
                    node_key,
                    task_key,
                )
                if current_task.status.is_terminal:
                    continue
                call = (
                    None
                    if current_task.provider_call_id is None
                    else repository.get_provider_call(current_task.provider_call_id)
                )
                if (
                    observation == AvailabilityStatus.MISSING
                    and call is not None
                    and call.status == ProviderCallStatus.SUCCEEDED
                ):
                    repository.fail_task(
                        self.execution_run_id,
                        node_key,
                        task_key,
                        message="Provider Call returned without a valid publication",
                        now=self._now(),
                    )
                else:
                    repository.record_task_result_observation(
                        self.execution_run_id,
                        node_key,
                        task_key,
                        observation,
                        now=self._now(),
                    )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        try:
            if node_key == DESIGN_RUNS_NODE:
                # This Node has no aggregate publication. Individual Task
                # publications are validated after atomic Task discovery.
                return AvailabilityStatus.MISSING
            elif node_key == COLLECT_RESULTS_NODE:
                available = (
                    load_collection_publication(
                        self.output_root,
                        self.request.collection_publication_path,
                    )
                    is not None
                )
            else:
                raise ValueError(f"Unknown BoltzGen Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _task_observation(
        self,
        node_key: str,
        item: _PlannedTask,
        task_fingerprint: str,
    ) -> AvailabilityStatus:
        if node_key == DESIGN_RUNS_NODE:
            if item.run_id is None:
                raise ValueError("BoltzGen design Task has no run ID")
            try:
                available = is_boltzgen_run_complete(
                    boltzgen_run_root(
                        self.output_root,
                        self.request.run_name,
                        item.run_id,
                    ),
                    task_fingerprint=task_fingerprint,
                )
            except OSError:
                return AvailabilityStatus.UNKNOWN
            return (
                AvailabilityStatus.AVAILABLE
                if available
                else AvailabilityStatus.MISSING
            )
        return self._node_observation(node_key)

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        reconciled = self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=_result_envelope,
            now=self._now(),
        )
        if any(
            not original.status.is_terminal
            and updated.status == ProviderCallStatus.SUCCEEDED
            for original, updated in reconciled
        ):
            self._reload_output()

    def _start_ready_nodes(self) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            statuses = {
                node.node_key: node.status
                for node in repository.list_nodes(self.execution_run_id)
            }
            plan = repository.get_run(self.execution_run_id).plan
        ready = ready_node_keys(plan, statuses)
        for node_key in ready:
            planned = self._planned_tasks(node_key)
            with self.store.transaction():
                repository = self.store.execution
                if repository.get_node(
                    self.execution_run_id,
                    node_key,
                ).status.is_terminal:
                    continue
                repository.start_node(
                    self.execution_run_id,
                    node_key,
                    now=self._now(),
                )
                records = repository.discover_tasks(
                    self.execution_run_id,
                    node_key,
                    tuple(item.plan for item in planned),
                    now=self._now(),
                )
            observations = tuple(
                self._task_observation(node_key, item, record.fingerprint)
                for item, record in zip(planned, records, strict=True)
            )
            with self.store.transaction():
                repository = self.store.execution
                for record, observation in zip(
                    records,
                    observations,
                    strict=True,
                ):
                    if repository.get_task(
                        self.execution_run_id,
                        node_key,
                        record.task_key,
                    ).status.is_terminal:
                        continue
                    repository.record_task_result_observation(
                        self.execution_run_id,
                        node_key,
                        record.task_key,
                        observation,
                        now=self._now(),
                    )

    def _planned_tasks(self, node_key: str) -> tuple[_PlannedTask, ...]:
        if node_key == DESIGN_RUNS_NODE:
            return tuple(
                _PlannedTask(
                    TaskPlan(
                        task_key=run_id,
                        scientific_payload={"run_id": run_id},
                        execution_payload={"run_id": run_id},
                    ),
                    run_id,
                )
                for run_id in self.request.run_ids
            )
        if node_key == COLLECT_RESULTS_NODE:
            return (
                _PlannedTask(
                    TaskPlan(
                        task_key="collection",
                        scientific_payload={
                            "publication": (
                                self.request.collection_publication_path.as_posix()
                            )
                        },
                    )
                ),
            )
        raise ValueError(f"Unknown BoltzGen Node {node_key!r}")

    def _admit_remote_tasks(self, required: set[str]) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            counts = repository.active_provider_call_counts(self.execution_run_id)
        bindings: dict[str, ProviderBinding] = {}

        def describe_task(node, task, rank):
            binding = bindings.get(node.node_key)
            if binding is None:
                binding = self._binding(node.node_key)
                bindings[node.node_key] = binding
            return TaskDispatchDescriptor(
                node_key=node.node_key,
                node_ordinal=node.ordinal,
                task_key=task.task_key,
                task_ordinal=task.ordinal,
                binding=binding,
                compatibility_key=binding.function_name,
                max_tasks_per_call=1,
                depth=rank.depth,
                unblocking_span=rank.unblocking_span,
            )

        selected = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=describe_task,
            available_total_slots=max(
                0,
                run.max_active_provider_calls - counts.total,
            ),
            available_gpu_slots=max(
                0,
                run.max_active_gpu_provider_calls - counts.gpu,
            ),
            now=self._now(),
        )
        submitted = self._provider.submit_provider_calls(
            self.execution_run_id,
            tuple(
                ProviderCallSubmission(
                    candidate=candidate,
                    submission_token=candidate.candidate_key,
                    kwargs=self._invocation_kwargs(
                        candidate.node_key,
                        candidate.task_keys[0],
                    ),
                    provider_call_id_kwarg=(
                        "claim_owner"
                        if candidate.node_key == DESIGN_RUNS_NODE
                        else None
                    ),
                )
                for candidate in selected
            ),
            now=self._now(),
        )
        if any(call is None for call in submitted):
            return

    def _binding(self, node_key: str) -> ProviderBinding:
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=(
                "run_boltzgen_task"
                if node_key == DESIGN_RUNS_NODE
                else "collect_boltzgen_data"
            ),
            uses_gpu=node_key == DESIGN_RUNS_NODE,
            runtime_image_key=(
                "boltzgen-gpu" if node_key == DESIGN_RUNS_NODE else "boltzgen-cpu"
            ),
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        if node_key == DESIGN_RUNS_NODE:
            replace_claim_owner = dict(self.request.replace_claim_owners).get(task_key)
            with self.store.synchronize():
                task = self.store.execution.get_task(
                    self.execution_run_id,
                    node_key,
                    task_key,
                )
            return {
                "out_dir": str(
                    boltzgen_run_root(
                        self.output_root,
                        self.request.run_name,
                        task_key,
                    )
                ),
                "input_yaml_path": str(
                    self.output_root.joinpath(*self.request.config_path.parts)
                ),
                "protocol": self.request.protocol,
                "num_designs": self.request.num_designs,
                "steps": self.request.steps,
                "extra_args": self.request.extra_args,
                "replace_claim_owner": replace_claim_owner,
                "task_fingerprint": task.fingerprint,
            }
        if node_key == COLLECT_RESULTS_NODE:
            with self.store.synchronize():
                design_tasks = self.store.execution.list_tasks(
                    self.execution_run_id,
                    DESIGN_RUNS_NODE,
                )
            task_fingerprints = {
                task.task_key: task.fingerprint for task in design_tasks
            }
            if set(task_fingerprints) != set(self.request.run_ids):
                raise RuntimeError("BoltzGen design Task fingerprints are incomplete")
            return {
                "run_name": self.request.run_name,
                "run_ids": list(self.request.run_ids),
                "task_fingerprints": task_fingerprints,
                "protocol": self.request.protocol,
                "num_designs": self.request.num_designs,
                "budget": self.request.budget,
                "steps": self.request.steps,
                "extra_args": self.request.extra_args,
                "filter_results": self.request.filter_results,
                "filter_rmsd_threshold": self.request.filter_rmsd_threshold,
                "publication_path": (
                    self.request.collection_publication_path.as_posix()
                ),
            }
        raise ValueError(f"Unknown BoltzGen Node {node_key!r}")


def _result_envelope(result: object) -> dict[str, object]:
    """Store only bounded JSON-compatible worker returns."""
    try:
        value = orjson.loads(orjson.dumps(result))
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError("BoltzGen worker returned a non-JSON result") from error
    return {"result": value}
