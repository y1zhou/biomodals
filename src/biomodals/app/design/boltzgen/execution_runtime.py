"""Caller-driven BoltzGen adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
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
    ExecutionSnapshot,
    NodeStatus,
    ProviderBinding,
    ProviderCallStatus,
    RunStatus,
    TaskPlan,
    TaskStatus,
    drive_execution_run,
    ready_node_keys,
    required_node_keys,
    result_probe_frontier,
    resume_execution_run,
)
from biomodals.execution.scheduler import (
    TaskDispatchDescriptor,
    form_fixed_batches,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import AppExecutionRunStore


@dataclass(frozen=True)
class _PlannedTask:
    plan: TaskPlan
    run_id: str | None = None


class BoltzGenExecutionRuntime:
    """Drive one BoltzGen request through direct one-Task GPU calls."""

    def __init__(
        self,
        *,
        request: BoltzGenExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: AppExecutionRunStore,
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
        self.output_root = Path(output_root)
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
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
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def resume(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Resume this same Run without retrying conclusive Task failures."""
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
        """Request explicit cancellation while retaining uncertain ownership."""
        self._provider.repository = self.store.execution
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        """Close SQLite without cancelling attached Provider Calls."""
        self.store.close()

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._reload_output()
        self._recover_publications()
        self._reconcile_nodes_and_run()
        run = self.store.execution.get_run(self.execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            self._reconcile_provider_calls(set(run.plan.node_keys))
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
        self._start_ready_nodes()
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
                    max_active_provider_calls=(self.request.max_active_provider_calls),
                    max_active_gpu_provider_calls=(
                        self.request.max_active_gpu_provider_calls
                    ),
                    now=self._now(),
                )
            return self._checkpoint()
        if (
            existing.plan != plan
            or existing.predecessor_execution_run_id
            != self.predecessor_execution_run_id
            or existing.deployment != self.deployment
            or existing.max_active_provider_calls
            != self.request.max_active_provider_calls
            or existing.max_active_gpu_provider_calls
            != self.request.max_active_gpu_provider_calls
        ):
            raise ValueError("BoltzGen request does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        observations: dict[str, AvailabilityStatus | None] = {}
        for node in repository.list_nodes(self.execution_run_id):
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
                for node_key, observation in observed:
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node_key,
                        observation,
                        now=self._now(),
                    )
                    observations[node_key] = observation
            repository = self._checkpoint()
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

        task_observations = []
        for node in repository.list_nodes(self.execution_run_id):
            if (
                node.node_key in required
                and node.status == NodeStatus.RUNNING
                and node.discovery_complete
            ):
                planned = {
                    item.plan.task_key: item
                    for item in self._planned_tasks(node.node_key)
                }
                for task in repository.list_tasks(
                    self.execution_run_id,
                    node.node_key,
                ):
                    if task.status.is_terminal:
                        continue
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
            for node_key, task_key, observation in task_observations:
                call = self._task_call(node_key, task_key)
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
        self._checkpoint()

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

    def _required_nodes(self) -> tuple[str, ...] | None:
        repository = self.store.execution
        observations = {}
        for node in repository.list_nodes(self.execution_run_id):
            observations[node.node_key] = (
                AvailabilityStatus.AVAILABLE
                if node.status == NodeStatus.SUCCEEDED
                else node.result_observation or AvailabilityStatus.MISSING
            )
        return required_node_keys(
            repository.get_run(self.execution_run_id).plan,
            observations,
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

    def _start_ready_nodes(self) -> None:
        repository = self.store.execution
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(self.execution_run_id)
        }
        ready = ready_node_keys(
            repository.get_run(self.execution_run_id).plan,
            statuses,
        )
        for node_key in ready:
            planned = self._planned_tasks(node_key)
            with self.store.transaction():
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
            repository = self._checkpoint()
            observations = tuple(
                self._task_observation(node_key, item, record.fingerprint)
                for item, record in zip(planned, records, strict=True)
            )
            with self.store.transaction():
                for record, observation in zip(
                    records,
                    observations,
                    strict=True,
                ):
                    repository.record_task_result_observation(
                        self.execution_run_id,
                        node_key,
                        record.task_key,
                        observation,
                        now=self._now(),
                    )
            self._checkpoint()

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

    def _reconcile_nodes_and_run(self) -> None:
        repository = self.store.execution
        for node in repository.list_nodes(self.execution_run_id):
            if node.status != NodeStatus.RUNNING or not node.discovery_complete:
                continue
            with self.store.transaction():
                repository.reconcile_node_tasks(
                    self.execution_run_id,
                    node.node_key,
                    now=self._now(),
                )
        with self.store.transaction():
            repository.skip_unreachable_nodes(
                self.execution_run_id,
                now=self._now(),
            )
            repository.finalize_run_from_results(
                self.execution_run_id,
                now=self._now(),
            )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        unfinished = {
            node.node_key
            for node in repository.list_nodes(self.execution_run_id)
            if not node.status.is_terminal
        }
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys=unfinished,
        )
        descriptors = []
        for node in repository.list_nodes(self.execution_run_id):
            if (
                node.node_key not in required
                or node.status != NodeStatus.RUNNING
                or not node.discovery_complete
            ):
                continue
            binding = self._binding(node.node_key)
            for task in repository.list_tasks(
                self.execution_run_id,
                node.node_key,
            ):
                if (
                    task.status != TaskStatus.PENDING
                    or task.result_observation != AvailabilityStatus.MISSING
                ):
                    continue
                rank = ranks[node.node_key]
                descriptors.append(
                    TaskDispatchDescriptor(
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
            kwargs = self._invocation_kwargs(
                candidate.node_key,
                candidate.task_keys[0],
            )
            self._provider.repository = self.store.execution
            submitted = self._provider.submit_fixed_batch(
                self.execution_run_id,
                candidate,
                submission_token=candidate.candidate_key,
                kwargs=kwargs,
                provider_call_id_kwarg=(
                    "claim_owner" if candidate.node_key == DESIGN_RUNS_NODE else None
                ),
                now=self._now(),
            )
            if submitted is None:
                return

    def _binding(self, node_key: str) -> ProviderBinding:
        run = self.store.execution.get_run(self.execution_run_id)
        return ProviderBinding(
            environment=run.deployment.environment,
            app_name=run.deployment.deployment_name,
            app_version=run.deployment.deployment_version,
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

    def _task_call(self, node_key: str, task_key: str):
        task = self.store.execution.get_task(
            self.execution_run_id,
            node_key,
            task_key,
        )
        if task.provider_call_id is None:
            return None
        return self.store.execution.get_provider_call(task.provider_call_id)

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
    """Store only bounded JSON-compatible worker returns."""
    try:
        value = orjson.loads(orjson.dumps(result))
    except (TypeError, orjson.JSONEncodeError) as error:
        raise ValueError("BoltzGen worker returned a non-JSON result") from error
    return {"result": value}
