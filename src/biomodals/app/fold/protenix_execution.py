"""Direct Protenix adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from base64 import b64decode, b64encode
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
    TaskPlan,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
)
from biomodals.helper.output_claim import (
    acquire_output_claim,
    register_output_claim_successor,
)

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 16 * 1024 * 1024
DOWNLOAD_NODE = "download-model-data"
PLAN_NODE = "plan-preprocessing"
MSA_NODE = "search-msa"
FINALIZE_NODE = "finalize-preprocessing"
INFERENCE_NODE = "run-protenix"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "Protenix execution request",
)


@dataclass(frozen=True, slots=True)
class ProtenixMsaTaskSpec:
    """One content-addressed MSA/template search."""

    task_key: str
    input_name: str
    query_command: str
    input_json_path: str
    output_dir: str
    msa_server_mode: str
    expected_json_path: str
    publication_key: str


@dataclass(frozen=True, slots=True)
class ProtenixPreparationPlan:
    """Prepared-input fan-out plus its stable final publication."""

    preparation_key: str
    prepared_json_path: str
    tasks: tuple[ProtenixMsaTaskSpec, ...]


@dataclass(frozen=True)
class ProtenixExecutionRequest:
    """Immutable input and scientific flags plus coordinator capacity."""

    run_name: str
    input_content: bytes
    model_name: str
    seeds: str
    cycle: int
    step: int
    sample: int
    dtype: str
    use_msa: bool
    msa_server_mode: str
    use_template: bool
    use_rna_msa: bool
    use_tfg_guidance: bool
    use_fast_layernorm: bool
    force_redownload: bool
    extra_args: str | None
    score_only: bool
    max_active_provider_calls: int
    app_version: str
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject empty inputs and unusable coordinator capacity."""
        if not self.run_name or not self.input_content or not self.model_name:
            raise ValueError("Protenix run name, input, and model cannot be empty")
        if self.max_active_provider_calls < 1:
            raise ValueError("Protenix provider-call limit must be positive")
        if not self.app_version:
            raise ValueError("Protenix app version cannot be empty")

    @property
    def requires_preprocessing(self) -> bool:
        """Return whether prediction needs MSA/template preparation."""
        return not self.score_only and (
            self.use_msa or self.use_template or self.use_rna_msa
        )

    @property
    def result_key(self) -> str:
        """Return the scientific identity used by the result publication."""
        return self.execution_plan.workload_plan_fingerprint

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build download, optional CPU fan-out, and GPU inference Nodes."""
        nodes = [NodePlan(DOWNLOAD_NODE)]
        inference_dependencies = [NodeDependency(DOWNLOAD_NODE)]
        if self.requires_preprocessing:
            nodes.extend((
                NodePlan(PLAN_NODE),
                NodePlan(MSA_NODE, dependencies=(NodeDependency(PLAN_NODE),)),
                NodePlan(
                    FINALIZE_NODE,
                    dependencies=(NodeDependency(MSA_NODE),),
                ),
            ))
            inference_dependencies.append(NodeDependency(FINALIZE_NODE))
        nodes.append(
            NodePlan(INFERENCE_NODE, dependencies=tuple(inference_dependencies))
        )
        return ExecutionPlan(
            workload_name="protenix",
            workload_run_key=self.run_name,
            nodes=tuple(nodes),
            scientific_payload={
                "input_sha256": sha256(self.input_content).hexdigest(),
                "model_name": self.model_name,
                "seeds": self.seeds,
                "cycle": self.cycle,
                "step": self.step,
                "sample": self.sample,
                "dtype": self.dtype,
                "use_msa": self.use_msa,
                "msa_server_mode": self.msa_server_mode,
                "use_template": self.use_template,
                "use_rna_msa": self.use_rna_msa,
                "use_tfg_guidance": self.use_tfg_guidance,
                "use_fast_layernorm": self.use_fast_layernorm,
                "extra_args": self.extra_args,
                "score_only": self.score_only,
            },
            scientific_versions={
                "protenix": self.app_version,
                "biomodals.protenix.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "input_content": b64encode(self.input_content).decode("ascii"),
                "model_name": self.model_name,
                "seeds": self.seeds,
                "cycle": self.cycle,
                "step": self.step,
                "sample": self.sample,
                "dtype": self.dtype,
                "use_msa": self.use_msa,
                "msa_server_mode": self.msa_server_mode,
                "use_template": self.use_template,
                "use_rna_msa": self.use_rna_msa,
                "use_tfg_guidance": self.use_tfg_guidance,
                "use_fast_layernorm": self.use_fast_layernorm,
                "force_redownload": self.force_redownload,
                "extra_args": self.extra_args,
                "score_only": self.score_only,
                "max_active_provider_calls": self.max_active_provider_calls,
                "app_version": self.app_version,
                "replace_claim_owner": self.replace_claim_owner,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Protenix execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> ProtenixExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Protenix execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Protenix execution request schema is unsupported")
        encoded_input = value.pop("input_content", None)
        if not isinstance(encoded_input, str):
            raise TypeError("Protenix input content must be base64 text")
        value["input_content"] = b64decode(encoded_input, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: ProtenixExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: ProtenixExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> ProtenixExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return ProtenixExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> ProtenixExecutionRequest:
    """Load one request through Modal's Volume API."""
    return ProtenixExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class ProtenixExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one Protenix request through optional MSA fan-out."""

    def __init__(
        self,
        *,
        request: ProtenixExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        msa_cache_volume: Any,
        output_claims: Any,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to Protenix's cache and result volumes."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.msa_cache_volume = msa_cache_volume
        self.output_claims = output_claims
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._claimed_publications: set[str] = set()
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            commit_local=store.commit,
            transaction=store.transaction,
        )

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._provider.repository = self.store.execution
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=self._decode_completed_calls,
            start_ready_nodes=self._start_ready_nodes,
            admit_remote_tasks=self._admit_remote_tasks,
            now=self._now,
        )

    def _initialize(self):
        self._reload_volumes()
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
                    max_active_gpu_provider_calls=1,
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
            or existing.max_active_gpu_provider_calls != 1
        ):
            raise ValueError("Protenix request does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.recover_publications(
            self.execution_run_id,
            observe_node=self._node_observation,
            observe_task=lambda node_key, task: (
                None
                if node_key in {DOWNLOAD_NODE, PLAN_NODE}
                else self._task_observation(node_key, task)
            ),
            now=self._now(),
        )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        app = _workload_module()
        try:
            if node_key in {DOWNLOAD_NODE, PLAN_NODE}:
                available = False
            elif node_key == MSA_NODE:
                plan = self._try_preparation_plan()
                available = plan is not None and all(
                    app._msa_task_ready(task) for task in plan.tasks
                )
            elif node_key == FINALIZE_NODE:
                plan = self._try_preparation_plan()
                available = plan is not None and app._prepared_ready(plan)
            elif node_key == INFERENCE_NODE:
                available = app._result_ready(
                    self.request.result_key,
                    self.request.run_name,
                )
            else:
                raise ValueError(f"Unknown Protenix Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _task_observation(self, node_key: str, task: Any) -> AvailabilityStatus:
        if node_key == MSA_NODE:
            try:
                spec = ProtenixMsaTaskSpec(**task.execution_payload["task"])
                available = _workload_module()._msa_task_ready(spec)
            except OSError:
                return AvailabilityStatus.UNKNOWN
            return (
                AvailabilityStatus.AVAILABLE
                if available
                else AvailabilityStatus.MISSING
            )
        return self._node_observation(node_key)

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
            self._reload_volumes()

    def _decode_completed_calls(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.decode_completed_calls(
            self.execution_run_id,
            observe_task=self._completed_task_observation,
            missing_message="Protenix returned without a valid publication",
            now=self._now(),
        )

    def _completed_task_observation(
        self,
        node_key: str,
        task: Any,
        envelope: object,
    ) -> AvailabilityStatus:
        if not isinstance(envelope, dict):
            return AvailabilityStatus.MISSING
        if node_key in {DOWNLOAD_NODE, MSA_NODE}:
            if envelope.get("kind") != "none":
                return AvailabilityStatus.MISSING
            return (
                AvailabilityStatus.AVAILABLE
                if node_key == DOWNLOAD_NODE
                else self._task_observation(node_key, task)
            )
        if node_key == PLAN_NODE:
            try:
                _preparation_plan_from_envelope(envelope)
            except (TypeError, ValueError):
                return AvailabilityStatus.MISSING
            return AvailabilityStatus.AVAILABLE
        expected_kind = "prepared" if node_key == FINALIZE_NODE else "result"
        if envelope.get("kind") != expected_kind:
            return AvailabilityStatus.MISSING
        return self._task_observation(node_key, task)

    def _start_ready_nodes(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        self._provider.start_ready_nodes(
            self.execution_run_id,
            required_node_keys=required,
            task_plans=self._task_plans,
            observe_task=lambda node_key, task: (
                AvailabilityStatus.MISSING
                if node_key in {DOWNLOAD_NODE, PLAN_NODE}
                else self._task_observation(node_key, task)
            ),
            now=self._now(),
        )

    def _task_plans(self, node_key: str) -> tuple[TaskPlan, ...]:
        if node_key == DOWNLOAD_NODE:
            return (
                TaskPlan(
                    "model-data",
                    {"model_name": self.request.model_name},
                ),
            )
        if node_key == PLAN_NODE:
            return (
                TaskPlan(
                    "plan",
                    {"input_sha256": sha256(self.request.input_content).hexdigest()},
                ),
            )
        if node_key == MSA_NODE:
            plan = self._preparation_plan()
            return tuple(
                TaskPlan(
                    task.task_key,
                    {"publication_key": task.publication_key},
                    {"task": asdict(task)},
                )
                for task in plan.tasks
            )
        if node_key == FINALIZE_NODE:
            plan = self._preparation_plan()
            return (
                TaskPlan(
                    "finalize",
                    {"preparation_key": plan.preparation_key},
                ),
            )
        if node_key == INFERENCE_NODE:
            return (
                TaskPlan(
                    "inference",
                    {"result_key": self.request.result_key},
                ),
            )
        raise ValueError(f"Unknown Protenix Node {node_key!r}")

    def _try_preparation_plan(self) -> ProtenixPreparationPlan | None:
        try:
            return self._preparation_plan()
        except (LookupError, TypeError, ValueError):
            return None

    def _preparation_plan(self) -> ProtenixPreparationPlan:
        for call in self.store.execution.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == PLAN_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return _preparation_plan_from_envelope(call.result_envelope)
        raise LookupError("Protenix preparation plan is unavailable")

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
            self._ensure_publication_claim(
                candidate.node_key,
                candidate.task_keys[0],
            )
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
            DOWNLOAD_NODE: "download_protenix_data",
            PLAN_NODE: "plan_protenix_inputs",
            MSA_NODE: "query_protenix_msa_server",
            FINALIZE_NODE: "finalize_protenix_inputs",
            INFERENCE_NODE: "run_protenix",
        }[node_key]
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=function_name,
            uses_gpu=node_key == INFERENCE_NODE,
            runtime_image_key=(
                "protenix-gpu" if node_key == INFERENCE_NODE else "protenix-cpu"
            ),
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        if node_key == DOWNLOAD_NODE:
            return {
                "model_name": self.request.model_name,
                "force": self.request.force_redownload,
                "include_templates": self.request.use_template,
            }
        if node_key == PLAN_NODE:
            return {
                "input_bytes": self.request.input_content,
                "msa_server_mode": self.request.msa_server_mode,
                "use_template": self.request.use_template,
                "use_rna_msa": self.request.use_rna_msa,
            }
        if node_key == MSA_NODE:
            task = self.store.execution.get_task(
                self.execution_run_id,
                MSA_NODE,
                task_key,
            )
            return {"task": ProtenixMsaTaskSpec(**task.execution_payload["task"])}
        if node_key == FINALIZE_NODE:
            return {
                "input_bytes": self.request.input_content,
                "plan": self._preparation_plan(),
            }
        if node_key == INFERENCE_NODE:
            prepared_path = (
                self._preparation_plan().prepared_json_path
                if self.request.requires_preprocessing
                else None
            )
            return {
                "input_bytes": (
                    None
                    if self.request.requires_preprocessing
                    else self.request.input_content
                ),
                "prepared_input_path": prepared_path,
                "run_name": self.request.run_name,
                "result_key": self.request.result_key,
                "model_name": self.request.model_name,
                "seeds": self.request.seeds,
                "cycle": self.request.cycle,
                "step": self.request.step,
                "sample": self.request.sample,
                "dtype": self.request.dtype,
                "use_msa": self.request.use_msa,
                "msa_server_mode": (
                    "colabfold"
                    if self.request.score_only
                    else self.request.msa_server_mode
                ),
                "use_template": self.request.use_template,
                "use_rna_msa": self.request.use_rna_msa,
                "use_tfg_guidance": self.request.use_tfg_guidance,
                "use_fast_layernorm": self.request.use_fast_layernorm,
                "extra_args": self.request.extra_args,
                "score_only": self.request.score_only,
            }
        raise ValueError(f"Unknown Protenix Node {node_key!r}")

    def _ensure_publication_claim(self, node_key: str, task_key: str) -> None:
        if node_key == MSA_NODE:
            task = self.store.execution.get_task(
                self.execution_run_id,
                MSA_NODE,
                task_key,
            )
            publication_key = str(task.scientific_payload["publication_key"])
            claim_key = f"protenix-msa:{publication_key}"
        elif node_key == INFERENCE_NODE:
            claim_key = f"protenix-result:{self.request.result_key}"
        else:
            return
        if claim_key in self._claimed_publications:
            return
        acquire_output_claim(
            self.output_claims,
            claim_key=claim_key,
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._claimed_publications.add(claim_key)

    def _reload_volumes(self) -> None:
        self._volume_sync.reload()
        self.msa_cache_volume.reload()
        self._provider.repository = self.store.execution


def _preparation_plan_from_envelope(envelope: object) -> ProtenixPreparationPlan:
    if not isinstance(envelope, dict) or envelope.get("kind") != "plan":
        raise TypeError("Protenix preparation-plan envelope is invalid")
    value = envelope.get("plan")
    if not isinstance(value, dict):
        raise TypeError("Protenix preparation plan is invalid")
    preparation_key = value.get("preparation_key")
    prepared_json_path = value.get("prepared_json_path")
    tasks = value.get("tasks")
    if (
        not isinstance(preparation_key, str)
        or not isinstance(prepared_json_path, str)
        or not isinstance(tasks, list)
    ):
        raise TypeError("Protenix preparation plan has invalid fields")
    parsed_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError("Protenix MSA Task is invalid")
        fields = (
            task.get("task_key"),
            task.get("input_name"),
            task.get("query_command"),
            task.get("input_json_path"),
            task.get("output_dir"),
            task.get("msa_server_mode"),
            task.get("expected_json_path"),
            task.get("publication_key"),
        )
        if not all(isinstance(field, str) for field in fields):
            raise TypeError("Protenix MSA Task has invalid fields")
        parsed_tasks.append(
            ProtenixMsaTaskSpec(
                task_key=cast(str, fields[0]),
                input_name=cast(str, fields[1]),
                query_command=cast(str, fields[2]),
                input_json_path=cast(str, fields[3]),
                output_dir=cast(str, fields[4]),
                msa_server_mode=cast(str, fields[5]),
                expected_json_path=cast(str, fields[6]),
                publication_key=cast(str, fields[7]),
            )
        )
    return ProtenixPreparationPlan(
        preparation_key=preparation_key,
        prepared_json_path=prepared_json_path,
        tasks=tuple(parsed_tasks),
    )


def _result_envelope(result: object) -> dict[str, object]:
    """Encode only bounded plans or publication metadata."""
    if isinstance(result, ProtenixPreparationPlan):
        return {"kind": "plan", "plan": asdict(result)}
    if result is None:
        return {"kind": "none"}
    if isinstance(result, dict):
        kind = "result" if "result_path" in result else "prepared"
        return {"kind": kind, "publication": orjson.loads(orjson.dumps(result))}
    return {"kind": "invalid"}


def _workload_module():
    """Import workload-owned publication probes after Modal app loading."""
    from biomodals.app.fold import protenix_app

    return protenix_app


class ProtenixExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to Protenix publications."""

    _request_loader = staticmethod(load_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        msa_cache_volume: Any,
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
        self.msa_cache_volume = msa_cache_volume
        self.output_claims = output_claims
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: ProtenixExecutionRequest | None = None,
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
        request: ProtenixExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> ProtenixExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active Protenix runtime does not match request")
            return runtime
        runtime = ProtenixExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            msa_cache_volume=self.msa_cache_volume,
            output_claims=self.output_claims,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
