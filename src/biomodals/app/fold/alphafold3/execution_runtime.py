"""Caller-driven AlphaFold3 adaptation of the shared execution kernel."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    load_json_object,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.execution_plan import (
    ALPHAFOLD3_EXECUTION_NODE_KEYS,
)
from biomodals.app.fold.alphafold3.execution_publications import (
    execution_result_path,
    load_execution_result,
    load_execution_result_path,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)
from biomodals.app.fold.alphafold3.execution_tasks import (
    combined_msa_task_plan,
    inference_summary_task_plan,
    raw_search_task_plan,
    request_publication_task_plan,
    seed_prediction_task_plan,
    stage_request_task_plan,
    staged_inference_task_plan,
    template_search_task_plan,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    PreparedInferenceRun,
    load_staged_inference_input,
    prepare_inference_run,
    validate_upstream_af3_input,
)
from biomodals.app.fold.alphafold3.input_enrichment import (
    MsaAssemblyResolution,
    apply_msa_resolution,
    apply_template_results,
    chain_msa_states,
    fill_missing_msa_for_inference,
    plan_template_searches,
    reduce_msa_assembly_results,
    reduce_template_cache_results,
    validate_template_result,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    build_invocation_receipt,
    load_invocation_manifest,
)
from biomodals.app.fold.alphafold3.modal_adapters import (
    publish_invocation_receipt,
    stage_inference_run,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
    SearchRuntime,
    inspect_msa_cache,
    plan_msa_resolution,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    load_request_manifest,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    InferenceRuntime,
    claim_seed_predictions,
    inference_run_root,
    inspect_seed_predictions,
    load_summary_entry,
)
from biomodals.app.fold.alphafold3.template_search import (
    TemplateRuntime,
    TemplateTask,
    inspect_template_entries,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionRuntime,
    NodeStatus,
    ProviderBinding,
    ProviderCallRecord,
    ProviderCallStatus,
    RunStatus,
    TaskPlan,
    TaskStatus,
    ready_node_keys,
    result_probe_frontier,
)
from biomodals.execution.modal import (
    ModalDefiniteSubmissionError,
    ModalSubmissionOutcomeUnknownError,
)
from biomodals.execution.scheduler import (
    ProviderCallCandidate,
    TaskDispatchDescriptor,
    form_fixed_batches,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import (
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
)

(
    _STAGE_REQUEST,
    _RAW_SEARCHES,
    _MSA_ASSEMBLIES,
    _TEMPLATE_SEARCHES,
    _STAGE_INFERENCE,
    _SEED_PREDICTIONS,
    _INFERENCE_SUMMARY,
    _REQUEST_PUBLICATION,
) = ALPHAFOLD3_EXECUTION_NODE_KEYS
_REMOTE_NODE_FUNCTIONS = {
    _RAW_SEARCHES: "search_database_msa",
    _MSA_ASSEMBLIES: "assemble_sequence_msas",
    _TEMPLATE_SEARCHES: "search_protein_templates",
    _SEED_PREDICTIONS: "run_inference_pipeline",
    _INFERENCE_SUMMARY: "finalize_inference_summary",
    _REQUEST_PUBLICATION: "finalize_inference_request",
}
_LOCAL_NODE_KEYS = {_STAGE_REQUEST, _STAGE_INFERENCE}
_EMPTY_PUBLICATION_SCHEMA = 1


class _ConclusivePublicationError(RuntimeError):
    """A completed call did not leave its required scientific publication."""


class _IncompletePrerequisiteError(RuntimeError):
    """A downstream publication cannot exist before its prerequisite."""


@dataclass(frozen=True, slots=True)
class _PlannedTask:
    plan: TaskPlan
    value: object


class AlphaFold3ExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one AlphaFold3 App Run through ordinary kernel operations."""

    def __init__(
        self,
        *,
        request: AlphaFold3ExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        search_runtime: SearchRuntime,
        template_runtime: TemplateRuntime,
        inference_runtime: InferenceRuntime,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind existing AlphaFold3 plans, publications, and Modal functions."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self.search_runtime = search_runtime
        self.template_runtime = template_runtime
        self.inference_runtime = inference_runtime
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._provider = ExecutionRuntime(
            self.store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            commit_local=self.store.commit,
            transaction=self.store.transaction,
        )
        self._msa_inventory_cache: (
            tuple[
                tuple[RawSearchTask, ...],
                tuple[dict[str, object], ...],
                tuple[MsaAssemblyTask, ...],
            ]
            | None
        ) = None
        self._combined_msa_cache: dict[
            tuple[MsaAssemblyTask, ...],
            tuple[dict[str, object], ...],
        ] = {}
        self._prepared_inference_cache: PreparedInferenceRun | None = None
        self._prepared_inference_error: _IncompletePrerequisiteError | None = None
        self._seed_prediction_cache: dict[int, dict[str, object]] | None = None

    def advance_once(self) -> None:
        """Apply one AlphaFold3-specific scheduling and publication cycle."""
        repository = self.store.execution
        self._provider.repository = repository

        self._recover_publications()
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
        self._publish_request_receipt()
        self._recover_task_publications()
        self._reconcile_nodes_and_run()
        run = self.store.execution.get_run(self.execution_run_id)
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return

        self._start_ready_nodes()
        self._run_local_tasks()
        self._publish_request_receipt()
        self._recover_task_publications()
        required = self._required_nodes()
        if required is not None:
            self._admit_remote_tasks(set(required))
        self._reconcile_nodes_and_run()

    def _initialize(self):
        self._reload_output()
        self._provider.create_or_verify_run(
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=self.predecessor_execution_run_id,
            plan=self.request.execution_plan,
            deployment=self.deployment,
            max_active_provider_calls=self.request.max_active_provider_calls,
            max_active_gpu_provider_calls=self.request.max_num_gpus,
            now=self._now(),
        )
        return self.store.execution

    def _recover_publications(self) -> None:
        """Probe result Nodes backward until reusable work closes each branch."""
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
            if any(
                observation == AvailabilityStatus.UNKNOWN for _, observation in observed
            ):
                return

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        """Validate one complete Node result without admitting its Tasks."""
        try:
            planned = self._planned_tasks(node_key)
        except _IncompletePrerequisiteError:
            return AvailabilityStatus.MISSING
        except Exception:
            return AvailabilityStatus.UNKNOWN
        if not planned:
            return self._empty_node_observation(node_key)

        observations: list[AvailabilityStatus] = []
        for item in planned:
            try:
                observations.append(self._task_observation(node_key, item))
            except (
                _ConclusivePublicationError,
                _IncompletePrerequisiteError,
            ):
                return AvailabilityStatus.MISSING
            except Exception:
                return AvailabilityStatus.UNKNOWN
        if AvailabilityStatus.UNKNOWN in observations:
            return AvailabilityStatus.UNKNOWN
        if all(
            observation == AvailabilityStatus.AVAILABLE for observation in observations
        ):
            return AvailabilityStatus.AVAILABLE
        return AvailabilityStatus.MISSING

    def _invocation_observation(self) -> AvailabilityStatus:
        try:
            manifest = load_invocation_manifest(
                self.output_volume,
                self.request.invocation,
            )
        except Exception:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if manifest is not None
            else AvailabilityStatus.MISSING
        )

    def _required_nodes(self) -> tuple[str, ...] | None:
        self._provider.repository = self.store.execution
        return self._provider.required_node_keys(self.execution_run_id)

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

    def _reconcile_provider_calls(self, required: tuple[str, ...]) -> None:
        self._provider.repository = self.store.execution
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
        repository = self.store.execution
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(self.execution_run_id)
        }
        for node_key in ready_node_keys(
            repository.get_run(self.execution_run_id).plan,
            statuses,
        ):
            try:
                planned = self._planned_tasks(node_key)
            except Exception:
                with self.store.transaction():
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node_key,
                        AvailabilityStatus.UNKNOWN,
                        now=self._now(),
                    )
                return
            observations: list[AvailabilityStatus] = []
            for item in planned:
                try:
                    observations.append(self._task_observation(node_key, item))
                except Exception:
                    observations.append(AvailabilityStatus.UNKNOWN)
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
            if not planned:
                self._publish_empty_node(node_key)

    def _recover_task_publications(self) -> None:
        repository = self.store.execution
        for node in repository.list_nodes(self.execution_run_id):
            if node.status != NodeStatus.RUNNING or not node.discovery_complete:
                continue
            try:
                planned = self._planned_tasks(node.node_key)
            except Exception:
                with self.store.transaction():
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node.node_key,
                        AvailabilityStatus.UNKNOWN,
                        now=self._now(),
                    )
                return
            records = repository.list_tasks(self.execution_run_id, node.node_key)
            if len(records) != len(planned):
                raise RuntimeError("Persisted AlphaFold3 Task count changed")
            with self.store.transaction():
                for record, item in zip(records, planned, strict=True):
                    expected = item.plan.fingerprint(
                        workload_plan_fingerprint=(
                            repository.get_run(
                                self.execution_run_id
                            ).plan.workload_plan_fingerprint
                        ),
                        node_key=node.node_key,
                    )
                    if (
                        record.task_key != item.plan.task_key
                        or record.fingerprint != expected
                    ):
                        raise RuntimeError("Persisted AlphaFold3 Task identity changed")
                    if record.status.is_terminal:
                        continue
                    try:
                        observation = self._task_observation(node.node_key, item)
                    except _ConclusivePublicationError as error:
                        repository.fail_task(
                            self.execution_run_id,
                            node.node_key,
                            record.task_key,
                            message=str(error),
                            now=self._now(),
                        )
                        continue
                    except Exception:
                        observation = AvailabilityStatus.UNKNOWN
                    if observation == AvailabilityStatus.MISSING:
                        continue
                    repository.record_task_result_observation(
                        self.execution_run_id,
                        node.node_key,
                        record.task_key,
                        observation,
                        now=self._now(),
                    )

    def _run_local_tasks(self) -> None:
        repository = self.store.execution
        node = repository.get_node(self.execution_run_id, _STAGE_INFERENCE)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        task = repository.get_task(
            self.execution_run_id,
            _STAGE_INFERENCE,
            "staged-input",
        )
        if (
            task.status.is_terminal
            or task.result_observation != AvailabilityStatus.MISSING
        ):
            return
        with self.store.transaction():
            acquired = repository.acquire_local_task(
                self.execution_run_id,
                _STAGE_INFERENCE,
                task.task_key,
                now=self._now(),
            )
        if not acquired:
            return
        self._checkpoint()
        prepared = self._prepared_inference()
        try:
            with self.store.closed_for_volume_sync():
                stage_inference_run(self.output_volume, prepared)
                self.output_volume.reload()
            self._provider.repository = self.store.execution
            loaded = load_staged_inference_input(
                self.inference_runtime.output_root,
                run_id=prepared.run_id,
                request_id=prepared.request_id,
                staged_input_record=prepared.staged_input.to_record(),
            )
            if loaded.recycle != prepared.recycle:
                raise RuntimeError("Staged AlphaFold3 input changed")
        except Exception as error:
            with self.store.transaction():
                self.store.execution.fail_task(
                    self.execution_run_id,
                    _STAGE_INFERENCE,
                    task.task_key,
                    message=f"Could not stage inference input: {error}",
                    now=self._now(),
                )
            return
        with self.store.transaction():
            self.store.execution.record_task_result_observation(
                self.execution_run_id,
                _STAGE_INFERENCE,
                task.task_key,
                AvailabilityStatus.AVAILABLE,
                now=self._now(),
            )

    def _publish_request_receipt(self) -> None:
        repository = self.store.execution
        node = repository.get_node(self.execution_run_id, _REQUEST_PUBLICATION)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            return
        task = repository.get_task(
            self.execution_run_id,
            _REQUEST_PUBLICATION,
            "request-view",
        )
        if task.status.is_terminal or task.provider_call_id is None:
            return
        call = repository.get_provider_call(task.provider_call_id)
        if call.status != ProviderCallStatus.SUCCEEDED:
            return
        prepared = self._prepared_inference()
        publication = RequestPublication.from_prepared(prepared)
        try:
            manifest = load_request_manifest(self.output_volume, publication)
            returned = self._load_call_result(call)
            if manifest is None or returned != manifest:
                raise _ConclusivePublicationError(
                    "Request finalizer returned without its exact manifest"
                )
            with self.store.closed_for_volume_sync():
                publish_invocation_receipt(
                    self.output_volume,
                    build_invocation_receipt(
                        self.request.invocation,
                        prepared,
                        manifest,
                    ),
                )
                self.output_volume.reload()
            self._provider.repository = self.store.execution
        except _ConclusivePublicationError as error:
            with self.store.transaction():
                self.store.execution.fail_task(
                    self.execution_run_id,
                    _REQUEST_PUBLICATION,
                    task.task_key,
                    message=str(error),
                    now=self._now(),
                )

    def _reconcile_nodes_and_run(self) -> None:
        repository = self.store.execution
        for node in repository.list_nodes(self.execution_run_id):
            if node.status != NodeStatus.RUNNING or not node.discovery_complete:
                continue
            tasks = repository.list_tasks(self.execution_run_id, node.node_key)
            if not tasks:
                observation = self._empty_node_observation(node.node_key)
                if observation == AvailabilityStatus.MISSING:
                    self._publish_empty_node(node.node_key)
                    return
                with self.store.transaction():
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node.node_key,
                        observation,
                        now=self._now(),
                    )
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
        descriptors: list[TaskDispatchDescriptor] = []
        planned_by_node: dict[str, dict[str, _PlannedTask]] = {}
        for node in repository.list_nodes(self.execution_run_id):
            if (
                node.node_key not in _REMOTE_NODE_FUNCTIONS
                or node.node_key not in required
                or node.status != NodeStatus.RUNNING
                or not node.discovery_complete
            ):
                continue
            planned = {
                item.plan.task_key: item for item in self._planned_tasks(node.node_key)
            }
            planned_by_node[node.node_key] = planned
            for task in repository.list_tasks(self.execution_run_id, node.node_key):
                if (
                    task.status != TaskStatus.PENDING
                    or task.result_observation != AvailabilityStatus.MISSING
                ):
                    continue
                binding, maximum = self._dispatch_binding(node.node_key)
                rank = ranks[node.node_key]
                descriptors.append(
                    TaskDispatchDescriptor(
                        node_key=node.node_key,
                        node_ordinal=node.ordinal,
                        task_key=task.task_key,
                        task_ordinal=task.ordinal,
                        binding=binding,
                        compatibility_key=binding.function_name,
                        max_tasks_per_call=maximum,
                        depth=rank.depth,
                        unblocking_span=rank.unblocking_span,
                    )
                )
        self._provider.repository = repository
        descriptors = list(
            self._provider.persist_fixed_dispatch_policy(
                self.execution_run_id,
                tuple(descriptors),
                now=self._now(),
            )
        )
        repository = self.store.execution
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
        if any(candidate.node_key == _SEED_PREDICTIONS for candidate in selected):
            self._reload_output()
        for candidate in selected:
            planned = planned_by_node[candidate.node_key]
            self._provider.repository = self.store.execution
            binding_function = self._provider.resolve_provider_binding(
                self.execution_run_id,
                candidate.binding,
                now=self._now(),
            )
            if binding_function is None:
                return
            selected_candidate, claimed = self._claim_seed_candidate(
                candidate,
                planned,
            )
            if selected_candidate is None:
                continue
            args, kwargs = self._provider_arguments(
                selected_candidate,
                planned,
                claimed,
            )
            self._provider.repository = self.store.execution
            try:
                submitted = self._provider.submit_resolved_fixed_batch(
                    self.execution_run_id,
                    selected_candidate,
                    function=binding_function,
                    submission_token=selected_candidate.candidate_key,
                    args=args,
                    kwargs=kwargs,
                    now=self._now(),
                )
                if submitted is None:
                    return
            except (
                ModalDefiniteSubmissionError,
                ModalSubmissionOutcomeUnknownError,
            ):
                return

    def _claim_seed_candidate(
        self,
        candidate: ProviderCallCandidate,
        planned: dict[str, _PlannedTask],
    ) -> tuple[ProviderCallCandidate | None, dict[str, ClaimedSeed]]:
        if candidate.node_key != _SEED_PREDICTIONS:
            return candidate, {}
        prepared = self._prepared_inference()
        seeds = tuple(cast(int, planned[key].value) for key in candidate.task_keys)
        task_records = {
            task.task_key: task
            for task in self.store.execution.list_tasks(
                self.execution_run_id,
                _SEED_PREDICTIONS,
            )
        }
        generations = {
            seed: self._generation_id(task_records[f"seed:{seed}"]) for seed in seeds
        }
        plan = claim_seed_predictions(
            self.inference_runtime,
            prepared.run_id,
            seeds,
            sample_count=prepared.sample_count,
            generation_ids=generations,
            reload_volume=False,
        )
        if plan.reused_seeds:
            with self.store.transaction():
                for seed in plan.reused_seeds:
                    self.store.execution.record_task_result_observation(
                        self.execution_run_id,
                        _SEED_PREDICTIONS,
                        f"seed:{seed}",
                        AvailabilityStatus.AVAILABLE,
                        now=self._now(),
                    )
        owned = {f"seed:{item.seed}": item for item in plan.owned}
        task_keys = tuple(key for key in candidate.task_keys if key in owned)
        if not task_keys:
            return None, {}
        first = self.store.execution.get_task(
            self.execution_run_id,
            _SEED_PREDICTIONS,
            task_keys[0],
        )
        return (
            replace(
                candidate,
                candidate_key=(
                    f"{candidate.node_key}:{candidate.binding.function_name}:"
                    f"{candidate.compatibility_key}:{first.ordinal}"
                ),
                task_keys=task_keys,
                task_ordinal=first.ordinal,
            ),
            owned,
        )

    def _provider_arguments(
        self,
        candidate: ProviderCallCandidate,
        planned: dict[str, _PlannedTask],
        claimed: dict[str, ClaimedSeed],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        path = self._candidate_result_path(candidate)
        values = [planned[key].value for key in candidate.task_keys]
        if candidate.node_key == _RAW_SEARCHES:
            task = cast(RawSearchTask, values[0])
            record = self.store.execution.get_task(
                self.execution_run_id,
                candidate.node_key,
                candidate.task_keys[0],
            )
            return (), {
                "database_id": task.database_id,
                "sequence": task.sequence,
                "generation_id": self._generation_id(record),
                "execution_result_path": path.as_posix(),
            }
        if candidate.node_key == _MSA_ASSEMBLIES:
            task = cast(MsaAssemblyTask, values[0])
            record = self.store.execution.get_task(
                self.execution_run_id,
                candidate.node_key,
                candidate.task_keys[0],
            )
            return (), {
                "polymer": task.polymer,
                "sequence": task.sequence,
                "include_unpaired": task.include_unpaired,
                "include_paired": task.include_paired,
                "generation_id": self._generation_id(record),
                "execution_result_path": path.as_posix(),
            }
        if candidate.node_key == _TEMPLATE_SEARCHES:
            task = cast(TemplateTask, values[0])
            record = self.store.execution.get_task(
                self.execution_run_id,
                candidate.node_key,
                candidate.task_keys[0],
            )
            return (), {
                "sequence": task.sequence,
                "unpaired_msa": task.unpaired_msa,
                "unpaired_msa_reference": (
                    task.unpaired_msa_reference.to_record()
                    if task.unpaired_msa_reference is not None
                    else None
                ),
                "publish_canonical": task.publish_canonical,
                "max_template_date": task.max_template_date,
                "generation_id": self._generation_id(record),
                "execution_result_path": path.as_posix(),
            }
        prepared = self._prepared_inference()
        if candidate.node_key == _SEED_PREDICTIONS:
            return (), {
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "staged_input_record": prepared.staged_input.to_record(),
                "claimed_seed_records": [
                    claimed[key].to_dict() for key in candidate.task_keys
                ],
                "execution_result_path": path.as_posix(),
            }
        if candidate.node_key == _INFERENCE_SUMMARY:
            return (), {
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "staged_input_record": prepared.staged_input.to_record(),
                "execution_result_path": path.as_posix(),
            }
        if candidate.node_key == _REQUEST_PUBLICATION:
            return (), {
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "submitted_seeds": list(prepared.submitted_seeds),
                "normalized_seeds": list(prepared.normalized_seeds),
                "sample_count": prepared.sample_count,
                "display_name": prepared.display_name,
                "execution_result_path": path.as_posix(),
            }
        raise RuntimeError(f"Unsupported AlphaFold3 Provider Node {candidate.node_key}")

    def _dispatch_binding(self, node_key: str) -> tuple[ProviderBinding, int]:
        uses_gpu = node_key == _SEED_PREDICTIONS
        maximum = 1
        if uses_gpu:
            seed_count = len(self._prepared_inference().normalized_seeds)
            maximum = max(1, math.ceil(seed_count / self.request.max_num_gpus))
        return (
            ProviderBinding(
                environment=self.deployment.environment,
                app_name=self.deployment.deployment_name,
                app_version=self.deployment.deployment_version,
                function_name=_REMOTE_NODE_FUNCTIONS[node_key],
                uses_gpu=uses_gpu,
                runtime_image_key="alphafold3-runtime",
            ),
            maximum,
        )

    def _planned_tasks(self, node_key: str) -> tuple[_PlannedTask, ...]:
        if node_key == _STAGE_REQUEST:
            return (
                _PlannedTask(
                    stage_request_task_plan(self.request.invocation),
                    self.request.invocation,
                ),
            )
        if node_key == _RAW_SEARCHES:
            tasks, statuses, _ = self._msa_inventory()
            return tuple(
                _PlannedTask(
                    raw_search_task_plan(
                        task,
                        search_identity=cast(str, status["search_identity"]),
                    ),
                    task,
                )
                for task, status in zip(tasks, statuses, strict=True)
            )
        if node_key == _MSA_ASSEMBLIES:
            raw_tasks, raw_statuses, assembly_tasks = self._msa_inventory()
            identities = {
                (task.database_id, task.sequence): cast(
                    str,
                    status["search_identity"],
                )
                for task, status in zip(raw_tasks, raw_statuses, strict=True)
            }
            return tuple(
                _PlannedTask(
                    combined_msa_task_plan(
                        task,
                        raw_search_identities={
                            database_id: identity
                            for (database_id, sequence), identity in identities.items()
                            if sequence == task.sequence
                        },
                    ),
                    task,
                )
                for task in assembly_tasks
            )
        if node_key == _TEMPLATE_SEARCHES:
            template_plan = self._template_plan()
            return tuple(
                _PlannedTask(template_search_task_plan(task), task)
                for task in template_plan.tasks
            )
        prepared = self._prepared_inference()
        if node_key == _STAGE_INFERENCE:
            return (_PlannedTask(staged_inference_task_plan(prepared), prepared),)
        if node_key == _SEED_PREDICTIONS:
            return tuple(
                _PlannedTask(seed_prediction_task_plan(prepared, seed), seed)
                for seed in prepared.normalized_seeds
            )
        if node_key == _INFERENCE_SUMMARY:
            return (_PlannedTask(inference_summary_task_plan(prepared), prepared),)
        if node_key == _REQUEST_PUBLICATION:
            return (_PlannedTask(request_publication_task_plan(prepared), prepared),)
        raise ValueError(f"Unknown AlphaFold3 Execution Node {node_key!r}")

    def _task_observation(
        self,
        node_key: str,
        item: _PlannedTask,
    ) -> AvailabilityStatus:
        if node_key == _STAGE_REQUEST:
            return AvailabilityStatus.AVAILABLE
        if node_key == _RAW_SEARCHES:
            task = cast(RawSearchTask, item.value)
            raw_tasks, statuses, _ = self._msa_inventory()
            status = statuses[raw_tasks.index(task)]
            return self._remote_publication_observation(
                node_key,
                item.plan.task_key,
                status.get("status") == "reused",
            )
        if node_key == _MSA_ASSEMBLIES:
            task = cast(MsaAssemblyTask, item.value)
            _, _, tasks = self._msa_inventory()
            canonical = tuple(value for value in tasks if value.publishes_canonical)
            statuses = self._inspect_combined(canonical)
            status_by_key = {
                (value.polymer, value.sequence): status
                for value, status in zip(canonical, statuses, strict=True)
            }
            status = status_by_key.get((task.polymer, task.sequence))
            if status is not None and status.get("status") == "reused":
                return AvailabilityStatus.AVAILABLE
            if task.publishes_canonical:
                return self._remote_publication_observation(
                    node_key,
                    item.plan.task_key,
                    False,
                )
            result = self._task_result_if_returned(
                node_key,
                item.plan.task_key,
                task_plan=item.plan,
            )
            if result is None:
                return self._remote_publication_observation(
                    node_key,
                    item.plan.task_key,
                    False,
                )
            reduce_msa_assembly_results((task,), (result,))
            return AvailabilityStatus.AVAILABLE
        if node_key == _TEMPLATE_SEARCHES:
            task = cast(TemplateTask, item.value)
            if task.publish_canonical:
                status = inspect_template_entries(
                    self.template_runtime.cache_root,
                    (
                        (
                            task.sequence,
                            task.unpaired_msa_sha256,
                            task.max_template_date,
                        ),
                    ),
                )[0]
                if status.get("status") == "reused":
                    return AvailabilityStatus.AVAILABLE
                return self._remote_publication_observation(
                    node_key,
                    item.plan.task_key,
                    False,
                )
            result = self._task_result_if_returned(
                node_key,
                item.plan.task_key,
                task_plan=item.plan,
            )
            if result is None:
                return self._remote_publication_observation(
                    node_key,
                    item.plan.task_key,
                    False,
                )
            validate_template_result(
                task,
                result,
                allowed_statuses=(
                    frozenset({"published", "reused"})
                    if task.publish_canonical
                    else frozenset({"request-local"})
                ),
            )
            return AvailabilityStatus.AVAILABLE
        if node_key == _STAGE_INFERENCE:
            prepared = cast(PreparedInferenceRun, item.value)
            output_root = self.inference_runtime.output_root
            if not output_root.is_absolute() or not output_root.is_dir():
                return AvailabilityStatus.UNKNOWN
            try:
                load_staged_inference_input(
                    output_root,
                    run_id=prepared.run_id,
                    request_id=prepared.request_id,
                    staged_input_record=prepared.staged_input.to_record(),
                )
            except OSError:
                return AvailabilityStatus.UNKNOWN
            except (RuntimeError, ValueError):
                return AvailabilityStatus.MISSING
            return AvailabilityStatus.AVAILABLE
        if node_key == _SEED_PREDICTIONS:
            prepared = self._prepared_inference()
            seed = cast(int, item.value)
            if self._seed_prediction_cache is None:
                statuses = inspect_seed_predictions(
                    self.inference_runtime,
                    prepared.run_id,
                    prepared.normalized_seeds,
                    sample_count=prepared.sample_count,
                    reload_volume=False,
                )
                self._seed_prediction_cache = dict(
                    zip(prepared.normalized_seeds, statuses, strict=True)
                )
            status = self._seed_prediction_cache[seed]
            return self._remote_publication_observation(
                node_key,
                item.plan.task_key,
                status.get("status") == "reused",
            )
        if node_key == _INFERENCE_SUMMARY:
            prepared = cast(PreparedInferenceRun, item.value)
            entry = load_summary_entry(
                inference_run_root(
                    self.inference_runtime.output_root,
                    prepared.run_id,
                ),
                prepared.run_id,
            )
            available = entry is not None and set(prepared.normalized_seeds).issubset(
                entry.included_seeds
            )
            return self._remote_publication_observation(
                node_key,
                item.plan.task_key,
                available,
            )
        if node_key == _REQUEST_PUBLICATION:
            observation = self._invocation_observation()
            if observation == AvailabilityStatus.UNKNOWN:
                return observation
            return self._remote_publication_observation(
                node_key,
                item.plan.task_key,
                observation == AvailabilityStatus.AVAILABLE,
            )
        raise ValueError(f"Unknown AlphaFold3 Execution Node {node_key!r}")

    def _remote_publication_observation(
        self,
        node_key: str,
        task_key: str,
        available: bool,
    ) -> AvailabilityStatus:
        if available:
            return AvailabilityStatus.AVAILABLE
        call = self._task_call(node_key, task_key)
        if call is not None and call.status == ProviderCallStatus.SUCCEEDED:
            raise _ConclusivePublicationError(
                f"{node_key}/{task_key} returned without a valid publication"
            )
        return AvailabilityStatus.MISSING

    def _task_result_if_returned(
        self,
        node_key: str,
        task_key: str,
        *,
        task_plan: TaskPlan | None = None,
    ) -> dict[str, object] | None:
        call = self._task_call(node_key, task_key)
        if call is not None and call.status == ProviderCallStatus.SUCCEEDED:
            return self._load_call_result(call)
        return load_execution_result_path(
            self.inference_runtime.output_root,
            self._task_result_path(
                node_key,
                task_key,
                task_plan=task_plan,
            ),
        )

    def _task_call(
        self,
        node_key: str,
        task_key: str,
    ) -> ProviderCallRecord | None:
        try:
            task = self.store.execution.get_task(
                self.execution_run_id,
                node_key,
                task_key,
            )
        except LookupError:
            return None
        if task.provider_call_id is None:
            return None
        return self.store.execution.get_provider_call(task.provider_call_id)

    def _load_call_result(self, call: ProviderCallRecord) -> dict[str, object]:
        envelope = call.result_envelope
        if not isinstance(envelope, dict):
            raise _ConclusivePublicationError("Provider result envelope is invalid")
        reference = envelope.get("execution_result")
        result = load_execution_result(
            self.inference_runtime.output_root,
            reference,
            expected_path=self._call_result_path(call),
        )
        if result is None:
            diagnostic = envelope.get("invalid_result")
            raise _ConclusivePublicationError(
                "Provider result publication is unavailable"
                + (f": {diagnostic}" if isinstance(diagnostic, str) else "")
            )
        return result

    def _msa_inventory(
        self,
    ) -> tuple[
        tuple[RawSearchTask, ...],
        tuple[dict[str, object], ...],
        tuple[MsaAssemblyTask, ...],
    ]:
        if self._msa_inventory_cache is not None:
            return self._msa_inventory_cache
        if not self.request.search_msa:
            self._msa_inventory_cache = (), (), ()
            return self._msa_inventory_cache
        config = self.request.config.model_copy(deep=True)
        states = chain_msa_states(config)
        plan = plan_msa_resolution(states)
        canonical = tuple(task for task in plan.assemblies if task.publishes_canonical)
        self.search_runtime.sharded_volume.reload()
        self.search_runtime.cache_volume.reload()
        raw_statuses, _ = inspect_msa_cache(
            self.search_runtime.sharded_root,
            self.search_runtime.cache_root,
            plan.raw_searches,
            canonical,
        )
        self._msa_inventory_cache = (
            plan.raw_searches,
            tuple(raw_statuses),
            plan.assemblies,
        )
        return self._msa_inventory_cache

    def _inspect_combined(
        self,
        tasks: tuple[MsaAssemblyTask, ...],
    ) -> tuple[dict[str, object], ...]:
        if not tasks:
            return ()
        cached = self._combined_msa_cache.get(tasks)
        if cached is not None:
            return cached
        raw_tasks, _, _ = self._msa_inventory()
        _, statuses = inspect_msa_cache(
            self.search_runtime.sharded_root,
            self.search_runtime.cache_root,
            raw_tasks,
            tasks,
        )
        cached = tuple(statuses)
        self._combined_msa_cache[tasks] = cached
        return cached

    def _msa_resolution(self) -> tuple[Any, Any, MsaAssemblyResolution]:
        config = self.request.config.model_copy(deep=True)
        if not self.request.search_msa:
            return (
                validate_upstream_af3_input(fill_missing_msa_for_inference(config)),
                (),
                MsaAssemblyResolution({}, {}),
            )
        states = chain_msa_states(config)
        plan = plan_msa_resolution(states)
        canonical = tuple(task for task in plan.assemblies if task.publishes_canonical)
        cached = {
            (task.polymer, task.sequence): status
            for task, status in zip(
                canonical,
                self._inspect_combined(canonical),
                strict=True,
            )
            if status.get("status") == "reused"
        }
        raw_tasks, raw_statuses, _ = self._msa_inventory()
        identities = {
            (task.database_id, task.sequence): cast(
                str,
                status["search_identity"],
            )
            for task, status in zip(raw_tasks, raw_statuses, strict=True)
        }
        outcomes: list[dict[str, object]] = []
        for task in plan.assemblies:
            outcome = cached.get((task.polymer, task.sequence))
            if outcome is None:
                task_plan = combined_msa_task_plan(
                    task,
                    raw_search_identities={
                        database_id: identity
                        for (database_id, sequence), identity in identities.items()
                        if sequence == task.sequence
                    },
                )
                outcome = self._task_result_if_returned(
                    _MSA_ASSEMBLIES,
                    task_plan.task_key,
                    task_plan=task_plan,
                )
            if outcome is None:
                raise _IncompletePrerequisiteError(
                    "MSA assembly publication is unavailable"
                )
            outcomes.append(outcome)
        resolution = reduce_msa_assembly_results(plan.assemblies, outcomes)
        apply_msa_resolution(
            config,
            states,
            resolution,
            search_protein_templates=self.request.search_protein_templates,
        )
        return config, states, resolution

    def _template_plan(self):
        config, states, resolution = self._msa_resolution()
        if not self.request.search_msa or not self.request.search_protein_templates:
            return plan_template_searches(
                fill_missing_msa_for_inference(config),
                (),
                resolution,
            )
        return plan_template_searches(config, states, resolution)

    def _enriched_config(self):
        config, states, resolution = self._msa_resolution()
        if not self.request.search_msa:
            return config
        if not self.request.search_protein_templates:
            return validate_upstream_af3_input(config)
        plan = plan_template_searches(config, states, resolution)
        canonical = plan.canonical_tasks
        self.template_runtime.cache_volume.reload()
        statuses = inspect_template_entries(
            self.template_runtime.cache_root,
            tuple(
                (
                    task.sequence,
                    task.unpaired_msa_sha256,
                    task.max_template_date,
                )
                for task in canonical
            ),
        )
        cached = reduce_template_cache_results(canonical, statuses)
        templates = dict(cached.templates_by_identity)
        for task in plan.tasks:
            if task.template_identity in templates:
                continue
            task_plan = template_search_task_plan(task)
            result = self._task_result_if_returned(
                _TEMPLATE_SEARCHES,
                task_plan.task_key,
                task_plan=task_plan,
            )
            if result is None:
                raise _IncompletePrerequisiteError(
                    "Template publication is unavailable"
                )
            templates[task.template_identity] = validate_template_result(
                task,
                result,
                allowed_statuses=(
                    frozenset({"published", "reused"})
                    if task.publish_canonical
                    else frozenset({"request-local"})
                ),
            )
        apply_template_results(config, plan, templates)
        return validate_upstream_af3_input(config)

    def _prepared_inference(self) -> PreparedInferenceRun:
        if self._prepared_inference_cache is not None:
            return self._prepared_inference_cache
        if self._prepared_inference_error is not None:
            raise self._prepared_inference_error
        try:
            prepared = prepare_inference_run(
                self._enriched_config(),
                recycle=self.request.recycle,
                sample=self.request.sample,
            )
        except _IncompletePrerequisiteError as error:
            self._prepared_inference_error = error
            raise
        self._prepared_inference_cache = prepared
        return prepared

    def _publish_empty_node(self, node_key: str) -> None:
        path = self._empty_node_path(node_key)
        write_json_atomic(
            self.inference_runtime.output_root.joinpath(*path.parts),
            {
                "schema_version": _EMPTY_PUBLICATION_SCHEMA,
                "status": "complete",
                "execution_run_id": str(self.execution_run_id),
                "node_key": node_key,
                "workload_plan_fingerprint": (
                    self.request.execution_plan.workload_plan_fingerprint
                ),
            },
        )
        with self.store.transaction():
            self.store.execution.record_node_result_observation(
                self.execution_run_id,
                node_key,
                AvailabilityStatus.AVAILABLE,
                now=self._now(),
            )

    def _empty_node_observation(self, node_key: str) -> AvailabilityStatus:
        path = self.inference_runtime.output_root.joinpath(
            *self._empty_node_path(node_key).parts
        )
        try:
            value = load_json_object(path)
        except FileNotFoundError:
            return AvailabilityStatus.MISSING
        except Exception:
            return AvailabilityStatus.UNKNOWN
        expected = {
            "schema_version": _EMPTY_PUBLICATION_SCHEMA,
            "status": "complete",
            "execution_run_id": str(self.execution_run_id),
            "node_key": node_key,
            "workload_plan_fingerprint": (
                self.request.execution_plan.workload_plan_fingerprint
            ),
        }
        return (
            AvailabilityStatus.AVAILABLE
            if value == expected
            else AvailabilityStatus.UNKNOWN
        )

    def _empty_node_path(self, node_key: str) -> PurePosixPath:
        return (
            PurePosixPath("execution-publications")
            / str(self.execution_run_id)
            / node_key
            / "empty.json"
        )

    def _generation_id(self, task: Any) -> str:
        return sha256(
            (f"{self.execution_run_id}:{task.node_key}:{task.fingerprint}").encode()
        ).hexdigest()

    def _candidate_result_path(
        self,
        candidate: ProviderCallCandidate,
    ) -> PurePosixPath:
        return self._result_path(candidate.node_key, candidate.task_keys)

    def _result_path(
        self,
        node_key: str,
        task_keys: tuple[str, ...],
    ) -> PurePosixPath:
        tasks = [
            self.store.execution.get_task(
                self.execution_run_id,
                node_key,
                key,
            )
            for key in task_keys
        ]
        digest = sha256(
            orjson.dumps(
                [task.fingerprint for task in tasks],
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()
        return execution_result_path(
            self.request.execution_plan.workload_plan_fingerprint,
            node_key,
            digest,
        )

    def _task_result_path(
        self,
        node_key: str,
        task_key: str,
        *,
        task_plan: TaskPlan | None = None,
    ) -> PurePosixPath:
        if task_plan is not None:
            if task_plan.task_key != task_key:
                raise ValueError("AlphaFold3 Task Plan key does not match result key")
            fingerprint = task_plan.fingerprint(
                workload_plan_fingerprint=(
                    self.request.execution_plan.workload_plan_fingerprint
                ),
                node_key=node_key,
            )
            digest = sha256(
                orjson.dumps([fingerprint], option=orjson.OPT_SORT_KEYS)
            ).hexdigest()
            return execution_result_path(
                self.request.execution_plan.workload_plan_fingerprint,
                node_key,
                digest,
            )
        return self._result_path(node_key, (task_key,))

    def _call_result_path(self, call: ProviderCallRecord) -> PurePosixPath:
        return self._result_path(call.node_key, call.task_keys)

    def _reload_output(self) -> None:
        super()._reload_output()
        self._invalidate_planning_cache()

    def _invalidate_planning_cache(self) -> None:
        """Discard cache observations after cross-container publications change."""
        self._msa_inventory_cache = None
        self._combined_msa_cache.clear()
        self._prepared_inference_cache = None
        self._prepared_inference_error = None
        self._seed_prediction_cache = None


def _result_envelope(result: object) -> dict[str, object]:
    """Normalize one worker return without making malformed output uncertain."""
    if isinstance(result, dict) and isinstance(result.get("execution_result"), dict):
        return cast(dict[str, object], result)
    return {
        "invalid_result": repr(result)[:4096],
    }
