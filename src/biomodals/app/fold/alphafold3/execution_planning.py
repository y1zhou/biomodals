"""Scientific planning and publication checks for AlphaFold3 execution."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

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
    template_search_task_plan,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    PreparedInferenceRun,
    load_staged_inference_input,
    prepare_inference_run,
    stage_inference_run_mounted,
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
from biomodals.app.fold.alphafold3.modal_adapters import publish_invocation_receipt
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
    SearchRuntime,
    inspect_msa_cache,
    plan_msa_resolution,
)
from biomodals.app.fold.alphafold3.profiles import (
    profile_root,
    resolve_database_profile,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    load_request_manifest,
    request_manifest_artifacts_available,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
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
    PreparedTaskBatch,
    ProviderCallSpec,
    ResultPublicationPendingError,
    TaskPlan,
)
from biomodals.schema import AppRunResult, AppRunStatus

(
    STAGE_REQUEST,
    PREPARE_ENVIRONMENT,
    RAW_SEARCHES,
    MSA_ASSEMBLIES,
    TEMPLATE_SEARCHES,
    STAGE_INFERENCE,
    SEED_PREDICTIONS,
    INFERENCE_SUMMARY,
    REQUEST_PUBLICATION,
) = ALPHAFOLD3_EXECUTION_NODE_KEYS


class IncompletePrerequisiteError(RuntimeError):
    """A downstream AlphaFold3 stage cannot yet be planned."""


@dataclass(frozen=True, slots=True)
class PlannedTask:
    """One deterministic Task Plan and its typed workload value."""

    plan: TaskPlan
    value: object


class AlphaFold3ExecutionPlanning:
    """Share deterministic planning state across AlphaFold3 graph Nodes."""

    def __init__(
        self,
        *,
        request: AlphaFold3ExecutionRequest,
        execution_run_id: UUID,
        output_volume: Any,
        search_runtime: SearchRuntime,
        template_runtime: TemplateRuntime,
        inference_runtime: InferenceRuntime,
    ) -> None:
        """Bind one request to its mounted scientific stores."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_volume = output_volume
        self.search_runtime = search_runtime
        self.template_runtime = template_runtime
        self.inference_runtime = inference_runtime
        self._msa_inventory_cache: (
            tuple[
                tuple[RawSearchTask, ...],
                tuple[dict[str, object], ...],
                tuple[MsaAssemblyTask, ...],
            ]
            | None
        ) = None
        self._combined_msa_cache: dict[
            tuple[MsaAssemblyTask, ...], tuple[dict[str, object], ...]
        ] = {}
        self._prepared_inference_cache: PreparedInferenceRun | None = None
        self._prepared_inference_error: IncompletePrerequisiteError | None = None
        self._seed_prediction_cache: dict[int, dict[str, object]] | None = None

    def invalidate(self, changed_nodes: Collection[str] | None = None) -> None:
        """Discard observations affected by newly published workload state."""
        changed = (
            {RAW_SEARCHES, MSA_ASSEMBLIES, TEMPLATE_SEARCHES, SEED_PREDICTIONS}
            if changed_nodes is None
            else set(changed_nodes)
        )
        if RAW_SEARCHES in changed:
            self._msa_inventory_cache = None
        if changed & {RAW_SEARCHES, MSA_ASSEMBLIES}:
            self._combined_msa_cache.clear()
        if changed & {RAW_SEARCHES, MSA_ASSEMBLIES, TEMPLATE_SEARCHES}:
            self._prepared_inference_cache = None
            self._prepared_inference_error = None
        if changed & {
            RAW_SEARCHES,
            MSA_ASSEMBLIES,
            TEMPLATE_SEARCHES,
            SEED_PREDICTIONS,
        }:
            self._seed_prediction_cache = None

    def refresh_result_storage(self, node_key: str) -> None:
        """Refresh the workload store that publishes one Task stage."""
        self.invalidate({node_key})
        if node_key == TEMPLATE_SEARCHES:
            self.template_runtime.cache_volume.reload()

    def planned_tasks(self, node_key: str) -> tuple[PlannedTask, ...]:
        """Return the complete deterministic Task collection for one stage."""
        if node_key == RAW_SEARCHES:
            tasks, statuses, _ = self._msa_inventory()
            return tuple(
                PlannedTask(
                    raw_search_task_plan(
                        task,
                        search_identity=cast(str, status["search_identity"]),
                    ),
                    task,
                )
                for task, status in zip(tasks, statuses, strict=True)
            )
        if node_key == MSA_ASSEMBLIES:
            raw_tasks, raw_statuses, assembly_tasks = self._msa_inventory()
            identities = {
                (task.database_id, task.sequence): cast(str, status["search_identity"])
                for task, status in zip(raw_tasks, raw_statuses, strict=True)
            }
            return tuple(
                PlannedTask(
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
        if node_key == TEMPLATE_SEARCHES:
            return tuple(
                PlannedTask(template_search_task_plan(task), task)
                for task in self._template_plan().tasks
            )
        prepared = self.prepared_inference()
        if node_key == SEED_PREDICTIONS:
            return tuple(
                PlannedTask(seed_prediction_task_plan(prepared, seed), seed)
                for seed in prepared.normalized_seeds
            )
        raise ValueError(f"Unknown AlphaFold3 Task stage {node_key!r}")

    def planned_task(self, node_key: str, task_key: str) -> PlannedTask:
        """Return one Task by its persisted identity."""
        try:
            return next(
                item
                for item in self.planned_tasks(node_key)
                if item.plan.task_key == task_key
            )
        except StopIteration as error:
            raise RuntimeError("Persisted AlphaFold3 Task identity changed") from error

    def task_fingerprint(self, node_key: str, item: PlannedTask) -> str:
        """Derive the kernel Task fingerprint without consulting SQLite."""
        return item.plan.fingerprint(
            workload_plan_fingerprint=(
                self.request.execution_plan.workload_plan_fingerprint
            ),
            node_key=node_key,
        )

    def result_path(
        self,
        node_key: str,
        items: tuple[PlannedTask, ...],
    ) -> PurePosixPath:
        """Return the content-addressed provider-result path for one call."""
        digest = sha256(
            orjson.dumps(
                [self.task_fingerprint(node_key, item) for item in items],
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()
        return execution_result_path(
            self.request.execution_plan.workload_plan_fingerprint,
            node_key,
            digest,
        )

    def generation_id(self, node_key: str, item: PlannedTask) -> str:
        """Bind one workload writer generation to its Execution Task."""
        return self._generation_id(self.execution_run_id, node_key, item)

    def superseded_generation_ids(
        self,
        node_key: str,
        item: PlannedTask,
    ) -> tuple[str, ...]:
        """Identify only terminal service Runs authorized for claim repair."""
        return tuple(
            self._generation_id(execution_run_id, node_key, item)
            for execution_run_id in self.request.repair_execution_run_ids
        )

    def _generation_id(
        self,
        execution_run_id: UUID,
        node_key: str,
        item: PlannedTask,
    ) -> str:
        return sha256(
            f"{execution_run_id}:{node_key}:{item.plan.task_key}".encode()
        ).hexdigest()

    def task_call(self, node_key: str, item: PlannedTask) -> ProviderCallSpec:
        """Prepare one non-batched AlphaFold3 provider call."""
        path = self.result_path(node_key, (item,))
        if node_key == RAW_SEARCHES:
            task = cast(RawSearchTask, item.value)
            kwargs = {
                "database_id": task.database_id,
                "sequence": task.sequence,
                "generation_id": self.generation_id(node_key, item),
                "superseded_generation_ids": self.superseded_generation_ids(
                    node_key, item
                ),
                "execution_result_path": path.as_posix(),
            }
            function_name = "search_database_msa"
        elif node_key == MSA_ASSEMBLIES:
            task = cast(MsaAssemblyTask, item.value)
            kwargs = {
                "polymer": task.polymer,
                "sequence": task.sequence,
                "include_unpaired": task.include_unpaired,
                "include_paired": task.include_paired,
                "generation_id": self.generation_id(node_key, item),
                "superseded_generation_ids": self.superseded_generation_ids(
                    node_key, item
                ),
                "execution_result_path": path.as_posix(),
            }
            function_name = "assemble_sequence_msas"
        elif node_key == TEMPLATE_SEARCHES:
            task = cast(TemplateTask, item.value)
            kwargs = {
                "sequence": task.sequence,
                "unpaired_msa": task.unpaired_msa,
                "unpaired_msa_reference": (
                    task.unpaired_msa_reference.to_record()
                    if task.unpaired_msa_reference is not None
                    else None
                ),
                "publish_canonical": task.publish_canonical,
                "max_template_date": task.max_template_date,
                "generation_id": self.generation_id(node_key, item),
                "superseded_generation_ids": self.superseded_generation_ids(
                    node_key, item
                ),
                "execution_result_path": path.as_posix(),
            }
            function_name = "search_protein_templates"
        else:
            raise ValueError(f"Unknown AlphaFold3 fixed Task stage {node_key!r}")
        return ProviderCallSpec(
            function_name=function_name,
            uses_gpu=False,
            kwargs=kwargs,
            runtime_image_key="alphafold3-runtime",
        )

    def seed_call_preview(self, item: PlannedTask) -> ProviderCallSpec:
        """Describe seed batching before acquiring output generations."""
        prepared = self.prepared_inference()
        return ProviderCallSpec(
            function_name="run_inference_pipeline",
            uses_gpu=True,
            kwargs={
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "staged_input_record": prepared.staged_input.to_record(),
                "claimed_seed_records": [],
                "allow_large_inference": self.request.allow_large_inference,
                "execution_result_path": self.result_path(
                    SEED_PREDICTIONS, (item,)
                ).as_posix(),
            },
            runtime_image_key="alphafold3-runtime",
            max_tasks_per_call=self.seed_batch_size,
        )

    @property
    def seed_batch_size(self) -> int:
        """Balance seeds over the configured GPU-call ceiling."""
        seed_count = len(self.prepared_inference().normalized_seeds)
        gpu_calls = max(1, self.request.max_active_gpu_provider_calls)
        return max(1, math.ceil(seed_count / gpu_calls))

    def prepare_seed_batch(
        self,
        items: tuple[PlannedTask, ...],
    ) -> PreparedTaskBatch:
        """Claim incomplete seed generations after provider preflight."""
        prepared = self.prepared_inference()
        seeds = tuple(cast(int, item.value) for item in items)
        claims = claim_seed_predictions(
            self.inference_runtime,
            prepared.run_id,
            seeds,
            sample_count=prepared.sample_count,
            generation_ids={
                cast(int, item.value): self.generation_id(SEED_PREDICTIONS, item)
                for item in items
            },
            superseded_generation_ids={
                cast(int, item.value): self.superseded_generation_ids(
                    SEED_PREDICTIONS,
                    item,
                )
                for item in items
            },
            reload_volume=False,
            allow_large_inference=self.request.allow_large_inference,
        )
        completed = {
            f"seed:{seed}": AppRunResult(status=AppRunStatus.SUCCEEDED)
            for seed in claims.reused_seeds
        }
        owned = {f"seed:{item.seed}": item for item in claims.owned}
        selected = tuple(item for item in items if item.plan.task_key in owned)
        call = None
        if selected:
            call = ProviderCallSpec(
                function_name="run_inference_pipeline",
                uses_gpu=True,
                kwargs={
                    "run_id": prepared.run_id,
                    "request_id": prepared.request_id,
                    "staged_input_record": prepared.staged_input.to_record(),
                    "claimed_seed_records": [
                        owned[item.plan.task_key].to_dict() for item in selected
                    ],
                    "allow_large_inference": self.request.allow_large_inference,
                    "execution_result_path": self.result_path(
                        SEED_PREDICTIONS, selected
                    ).as_posix(),
                },
                runtime_image_key="alphafold3-runtime",
                max_tasks_per_call=self.seed_batch_size,
            )
        return PreparedTaskBatch(
            call=call,
            task_keys=tuple(item.plan.task_key for item in selected),
            completed=completed,
        )

    def recover_task_result(
        self,
        node_key: str,
        item: PlannedTask,
    ) -> AppRunResult | None:
        """Validate one existing workload publication."""
        if node_key == RAW_SEARCHES:
            task = cast(RawSearchTask, item.value)
            tasks, statuses, _ = self._msa_inventory()
            status = statuses[tasks.index(task)]
            available = status.get("status") == "reused"
        elif node_key == MSA_ASSEMBLIES:
            task = cast(MsaAssemblyTask, item.value)
            _, _, tasks = self._msa_inventory()
            canonical = tuple(value for value in tasks if value.publishes_canonical)
            status_by_key = {
                (value.polymer, value.sequence): status
                for value, status in zip(
                    canonical, self._inspect_combined(canonical), strict=True
                )
            }
            status = status_by_key.get((task.polymer, task.sequence))
            if status is not None and status.get("status") == "reused":
                available = True
            elif task.publishes_canonical:
                available = False
            else:
                result = self._load_task_result(node_key, item)
                if result is None:
                    available = False
                else:
                    reduce_msa_assembly_results((task,), (result,))
                    available = True
        elif node_key == TEMPLATE_SEARCHES:
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
                available = status.get("status") == "reused"
            else:
                result = self._load_task_result(node_key, item)
                if result is None:
                    available = False
                else:
                    validate_template_result(
                        task,
                        result,
                        allowed_statuses=frozenset({"request-local"}),
                    )
                    available = True
        elif node_key == SEED_PREDICTIONS:
            prepared = self.prepared_inference()
            seed = cast(int, item.value)
            if self._seed_prediction_cache is None:
                statuses = inspect_seed_predictions(
                    self.inference_runtime,
                    prepared.run_id,
                    prepared.normalized_seeds,
                    sample_count=prepared.sample_count,
                    reload_volume=False,
                    allow_large_inference=self.request.allow_large_inference,
                )
                self._seed_prediction_cache = dict(
                    zip(prepared.normalized_seeds, statuses, strict=True)
                )
            available = self._seed_prediction_cache[seed].get("status") == "reused"
        else:
            raise ValueError(f"Unknown AlphaFold3 Task stage {node_key!r}")
        return AppRunResult(status=AppRunStatus.SUCCEEDED) if available else None

    def process_task_result(
        self,
        node_key: str,
        item: PlannedTask,
        raw_result: object,
    ) -> AppRunResult:
        """Validate one completed provider Task against scientific storage."""
        if (
            node_key == MSA_ASSEMBLIES
            and not cast(MsaAssemblyTask, item.value).publishes_canonical
        ):
            task = cast(MsaAssemblyTask, item.value)
            reduce_msa_assembly_results(
                (task,),
                (self._decode_result(raw_result, node_key, (item,)),),
            )
            return AppRunResult(status=AppRunStatus.SUCCEEDED)
        if (
            node_key == TEMPLATE_SEARCHES
            and not cast(TemplateTask, item.value).publish_canonical
        ):
            task = cast(TemplateTask, item.value)
            validate_template_result(
                task,
                self._decode_result(raw_result, node_key, (item,)),
                allowed_statuses=frozenset({"request-local"}),
            )
            return AppRunResult(status=AppRunStatus.SUCCEEDED)
        recovered = self.recover_task_result(node_key, item)
        if recovered is None:
            raise ResultPublicationPendingError(
                f"{node_key}/{item.plan.task_key} returned without a publication"
            )
        return recovered

    def stage_inference(self) -> AppRunResult:
        """Publish and revalidate the immutable inference input."""
        prepared = self.prepared_inference()
        stage_inference_run_mounted(self.inference_runtime.output_root, prepared)
        if self.staged_inference_observation() != AvailabilityStatus.AVAILABLE:
            raise RuntimeError("Staged AlphaFold3 input changed")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def staged_inference_observation(self) -> AvailabilityStatus:
        """Observe the exact staged inference input without flattening errors."""
        try:
            prepared = self.prepared_inference()
        except IncompletePrerequisiteError:
            return AvailabilityStatus.MISSING
        root = self.inference_runtime.output_root
        if not root.is_absolute() or not root.is_dir():
            return AvailabilityStatus.UNKNOWN
        try:
            loaded = load_staged_inference_input(
                root,
                run_id=prepared.run_id,
                request_id=prepared.request_id,
                staged_input_record=prepared.staged_input.to_record(),
            )
        except OSError:
            return AvailabilityStatus.UNKNOWN
        except (RuntimeError, ValueError):
            return AvailabilityStatus.MISSING
        return (
            AvailabilityStatus.AVAILABLE
            if loaded.recycle == prepared.recycle
            else AvailabilityStatus.MISSING
        )

    def summary_call(self) -> ProviderCallSpec:
        """Prepare the accumulated inference-summary call."""
        prepared = self.prepared_inference()
        item = PlannedTask(inference_summary_task_plan(prepared), prepared)
        return ProviderCallSpec(
            function_name="finalize_inference_summary",
            uses_gpu=False,
            kwargs={
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "staged_input_record": prepared.staged_input.to_record(),
                "generation_id": self.generation_id(INFERENCE_SUMMARY, item),
                "superseded_generation_ids": self.superseded_generation_ids(
                    INFERENCE_SUMMARY,
                    item,
                ),
                "execution_result_path": self.result_path(
                    INFERENCE_SUMMARY, (item,)
                ).as_posix(),
            },
            runtime_image_key="alphafold3-runtime",
        )

    def summary_result(self, raw_result: object | None = None) -> AppRunResult | None:
        """Validate the accumulated summary for every requested seed."""
        try:
            prepared = self.prepared_inference()
        except IncompletePrerequisiteError:
            return None
        if raw_result is not None:
            item = PlannedTask(inference_summary_task_plan(prepared), prepared)
            self._decode_result(raw_result, INFERENCE_SUMMARY, (item,))
        entry = load_summary_entry(
            inference_run_root(self.inference_runtime.output_root, prepared.run_id),
            prepared.run_id,
        )
        available = entry is not None and set(prepared.normalized_seeds).issubset(
            entry.included_seeds
        )
        return AppRunResult(status=AppRunStatus.SUCCEEDED) if available else None

    def request_call(self) -> ProviderCallSpec:
        """Prepare the immutable request-view publication call."""
        prepared = self.prepared_inference()
        item = PlannedTask(request_publication_task_plan(prepared), prepared)
        return ProviderCallSpec(
            function_name="finalize_inference_request",
            uses_gpu=False,
            kwargs={
                "run_id": prepared.run_id,
                "request_id": prepared.request_id,
                "submitted_seeds": list(prepared.submitted_seeds),
                "normalized_seeds": list(prepared.normalized_seeds),
                "sample_count": prepared.sample_count,
                "display_name": prepared.display_name,
                "execution_result_path": self.result_path(
                    REQUEST_PUBLICATION, (item,)
                ).as_posix(),
            },
            runtime_image_key="alphafold3-runtime",
        )

    def request_result(self, raw_result: object | None = None) -> AppRunResult | None:
        """Validate the request manifest and publish its invocation receipt."""
        if raw_result is None:
            invocation = load_invocation_manifest(
                self.output_volume,
                self.request.invocation,
            )
            available = invocation is not None and request_manifest_artifacts_available(
                self.output_volume,
                invocation,
            )
            return AppRunResult(status=AppRunStatus.SUCCEEDED) if available else None
        prepared = self.prepared_inference()
        publication = RequestPublication.from_prepared(prepared)
        manifest = load_request_manifest(self.output_volume, publication)
        if manifest is None:
            return None
        if raw_result is not None:
            item = PlannedTask(request_publication_task_plan(prepared), prepared)
            if (
                self._decode_result(raw_result, REQUEST_PUBLICATION, (item,))
                != manifest
            ):
                raise ValueError("Request finalizer returned a different manifest")
            publish_invocation_receipt(
                self.output_volume,
                build_invocation_receipt(
                    self.request.invocation,
                    prepared,
                    manifest,
                ),
            )
        invocation = load_invocation_manifest(
            self.output_volume,
            self.request.invocation,
        )
        available = invocation is not None and request_manifest_artifacts_available(
            self.output_volume, invocation
        )
        return AppRunResult(status=AppRunStatus.SUCCEEDED) if available else None

    def prepared_inference(self) -> PreparedInferenceRun:
        """Build the enriched inference request once per publication epoch."""
        if self._prepared_inference_cache is not None:
            return self._prepared_inference_cache
        if self._prepared_inference_error is not None:
            raise self._prepared_inference_error
        try:
            prepared = prepare_inference_run(
                self._enriched_config(),
                recycle=self.request.recycle,
                sample=self.request.sample,
                allow_large_inference=self.request.allow_large_inference,
            )
        except IncompletePrerequisiteError as error:
            self._prepared_inference_error = error
            raise
        self._prepared_inference_cache = prepared
        return prepared

    def _load_task_result(
        self,
        node_key: str,
        item: PlannedTask,
    ) -> dict[str, object] | None:
        return load_execution_result_path(
            self.inference_runtime.output_root,
            self.result_path(node_key, (item,)),
        )

    def _decode_result(
        self,
        raw_result: object,
        node_key: str,
        items: tuple[PlannedTask, ...],
    ) -> dict[str, object]:
        if not isinstance(raw_result, dict):
            raise ValueError("AlphaFold3 provider result is invalid")
        envelope = cast(dict[str, object], raw_result)
        result = load_execution_result(
            self.inference_runtime.output_root,
            envelope.get("execution_result"),
            expected_path=self.result_path(node_key, items),
        )
        if result is None:
            diagnostic = envelope.get("invalid_result")
            raise FileNotFoundError(
                "AlphaFold3 provider result publication is unavailable"
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
        plan = plan_msa_resolution(chain_msa_states(config))
        canonical = tuple(task for task in plan.assemblies if task.publishes_canonical)
        self.search_runtime.sharded_volume.reload()
        self.search_runtime.cache_volume.reload()
        missing_profiles = {
            task.database_id
            for task in plan.raw_searches
            if not (
                profile_root(
                    self.search_runtime.sharded_root,
                    resolve_database_profile(task.database_id),
                )
                / "manifest.json"
            ).is_file()
        }
        if missing_profiles:
            raise IncompletePrerequisiteError(
                "Database profiles are not prepared: "
                + ", ".join(sorted(missing_profiles))
            )
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
        if cached is None:
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
                canonical, self._inspect_combined(canonical), strict=True
            )
            if status.get("status") == "reused"
        }
        raw_tasks, raw_statuses, _ = self._msa_inventory()
        identities = {
            (task.database_id, task.sequence): cast(str, status["search_identity"])
            for task, status in zip(raw_tasks, raw_statuses, strict=True)
        }
        outcomes = []
        for task in plan.assemblies:
            outcome = cached.get((task.polymer, task.sequence))
            if outcome is None:
                item = PlannedTask(
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
                outcome = self._load_task_result(MSA_ASSEMBLIES, item)
            if outcome is None:
                raise IncompletePrerequisiteError(
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
                fill_missing_msa_for_inference(config), (), resolution
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
                (task.sequence, task.unpaired_msa_sha256, task.max_template_date)
                for task in canonical
            ),
        )
        templates = dict(
            reduce_template_cache_results(canonical, statuses).templates_by_identity
        )
        for task in plan.tasks:
            if task.template_identity in templates:
                continue
            item = PlannedTask(template_search_task_plan(task), task)
            result = self._load_task_result(TEMPLATE_SEARCHES, item)
            if result is None:
                raise IncompletePrerequisiteError("Template publication is unavailable")
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
