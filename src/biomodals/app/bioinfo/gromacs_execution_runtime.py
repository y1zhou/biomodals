"""Direct GROMACS adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from base64 import b64decode, b64encode
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.bioinfo.gromacs_execution import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    NPT_ANALYSIS,
    NVT_ANALYSIS,
    PREPARE_RESULT,
    PRODUCTION_ANALYSIS,
    execution_plan,
    modal_invocation,
    operation_provider_binding,
    operation_task_plan,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionRuntime,
    ExecutionSnapshot,
    NodeStatus,
    ProviderCallStatus,
    ProviderCallSubmission,
    TaskStatus,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
    persist_execution_launch,
)
from biomodals.helper.shell import sanitize_filename

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 32 * 1024 * 1024
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "GROMACS execution request",
)
_PUBLICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GromacsExecutionRequest:
    """Immutable simulation input plus operational call limits."""

    run_name: str
    pdb_content: bytes
    simulation_time_ns: int
    run_pdbfixer: bool
    cpu_only: bool
    num_threads: int
    use_openmp_threads: bool
    ld_seed: int
    gen_seed: int
    genion_seed: int
    max_active_provider_calls: int
    max_active_gpu_provider_calls: int

    def __post_init__(self) -> None:
        """Reject invalid identities and unusable operational limits."""
        if not self.run_name or sanitize_filename(self.run_name) != self.run_name:
            raise ValueError("run_name must be a safe filename component")
        if not self.pdb_content:
            raise ValueError("pdb_content cannot be empty")
        if self.simulation_time_ns < 1 or self.num_threads < 1:
            raise ValueError("simulation time and thread count must be positive")
        if self.max_active_provider_calls < 1:
            raise ValueError("max_active_provider_calls must be positive")
        if (
            not 0
            <= self.max_active_gpu_provider_calls
            <= self.max_active_provider_calls
        ):
            raise ValueError("GPU call limit must fit within the total call limit")

    @property
    def execution_plan(self):
        """Build the shared service/direct GROMACS graph."""
        return execution_plan(
            cpu_only=self.cpu_only,
            workload_run_key=self.run_name,
            pdb_sha256=sha256(self.pdb_content).hexdigest(),
            simulation_time_ns=self.simulation_time_ns,
            run_pdbfixer=self.run_pdbfixer,
            ld_seed=self.ld_seed,
            gen_seed=self.gen_seed,
            genion_seed=self.genion_seed,
        )

    def run_root(self, volume_root: str | Path) -> Path:
        """Return the established app-owned output directory."""
        return Path(volume_root) / self.run_name

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "pdb_content": b64encode(self.pdb_content).decode("ascii"),
                "simulation_time_ns": self.simulation_time_ns,
                "run_pdbfixer": self.run_pdbfixer,
                "cpu_only": self.cpu_only,
                "num_threads": self.num_threads,
                "use_openmp_threads": self.use_openmp_threads,
                "ld_seed": self.ld_seed,
                "gen_seed": self.gen_seed,
                "genion_seed": self.genion_seed,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("GROMACS execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> GromacsExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("GROMACS execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("GROMACS execution request schema is unsupported")
        encoded_pdb = value.pop("pdb_content", None)
        if not isinstance(encoded_pdb, str):
            raise TypeError("GROMACS PDB content must be base64 text")
        value["pdb_content"] = b64decode(encoded_pdb, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: GromacsExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: GromacsExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> GromacsExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return GromacsExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> GromacsExecutionRequest:
    """Load one request through Modal's Volume API."""
    return GromacsExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class GromacsExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one direct GROMACS request through fixed one-Task calls."""

    def __init__(
        self,
        *,
        request: GromacsExecutionRequest,
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
        """Bind the kernel writer to the established output directory."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.output_root = Path(output_root)
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            transaction=store.transaction,
            synchronize=store.synchronize,
        )

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=self._decode_completed_calls,
            start_ready_nodes=self._start_ready_nodes,
            after_start_ready_nodes=self._complete_local_result,
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
        self._provider.recover_publications(
            self.execution_run_id,
            observe_node=self._node_observation,
            observe_task=lambda node_key, _task: self._node_observation(node_key),
            now=self._now(),
        )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        try:
            available = self._node_publication_ready(node_key)
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _node_publication_path(self, node_key: str) -> Path:
        marker = sha256(node_key.encode()).hexdigest() + ".json"
        return (
            self.request.run_root(self.output_root) / ".biomodals" / "gromacs" / marker
        )

    def _node_publication_ready(self, node_key: str) -> bool:
        marker_path = self._node_publication_path(node_key)
        try:
            marker = orjson.loads(marker_path.read_bytes())
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            orjson.JSONDecodeError,
        ):
            return False
        if not (
            isinstance(marker, dict)
            and marker.get("schema_version") == _PUBLICATION_SCHEMA_VERSION
            and marker.get("node_key") == node_key
            and marker.get("workload_plan_fingerprint")
            == self.request.execution_plan.workload_plan_fingerprint
        ):
            return False
        raw_artifacts = marker.get("artifacts")
        if not isinstance(raw_artifacts, list):
            return False
        root = self.request.run_root(self.output_root)
        expected = {
            path.relative_to(root).as_posix() for path in self._node_paths(node_key)
        }
        if {
            artifact.get("path")
            for artifact in raw_artifacts
            if isinstance(artifact, dict)
        } != expected:
            return False
        for artifact in raw_artifacts:
            if not isinstance(artifact, dict):
                return False
            relative_text = artifact.get("path")
            if not isinstance(relative_text, str):
                return False
            relative = PurePosixPath(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            if not self._artifact_matches(
                root.joinpath(*relative.parts),
                artifact.get("size"),
                artifact.get("sha256"),
            ):
                return False
        return True

    def _write_node_publication(self, node_key: str) -> bool:
        root = self.request.run_root(self.output_root)
        artifacts = []
        try:
            for path in self._node_paths(node_key):
                if path.is_symlink():
                    return False
                stat = path.stat()
                if not S_ISREG(stat.st_mode) or stat.st_size < 1:
                    return False
                artifacts.append({
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "sha256": self._file_sha256(path),
                })
        except (FileNotFoundError, NotADirectoryError):
            return False
        marker = self._node_publication_path(node_key)
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(f".{time.time_ns()}.tmp")
        try:
            temporary.write_bytes(
                orjson.dumps(
                    {
                        "schema_version": _PUBLICATION_SCHEMA_VERSION,
                        "node_key": node_key,
                        "workload_plan_fingerprint": (
                            self.request.execution_plan.workload_plan_fingerprint
                        ),
                        "artifacts": artifacts,
                    },
                    option=orjson.OPT_SORT_KEYS,
                )
            )
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    @staticmethod
    def _artifact_matches(
        path: Path,
        expected_size: object,
        expected_digest: object,
    ) -> bool:
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 1
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            return False
        try:
            if path.is_symlink():
                return False
            stat = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return False
        return (
            S_ISREG(stat.st_mode)
            and stat.st_size == expected_size
            and GromacsExecutionRuntime._file_sha256(path) == expected_digest
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _node_paths(self, node_key: str) -> tuple[Path, ...]:
        root = self.request.run_root(self.output_root)
        name = self.request.run_name
        prepare = (
            root / f"production_{name}.tpr",
            root / "production.mdp",
        )

        def analysis(prefix: str) -> tuple[Path, ...]:
            return tuple(
                root / f"{metric}_{prefix}{name}.{suffix}"
                for metric in ("rmsd", "rg", "rmsf")
                for suffix in ("csv", "png")
            )

        if node_key.startswith("prepare_tpr_"):
            return prepare
        if node_key == NVT_ANALYSIS:
            return analysis("nvt_")
        if node_key == NPT_ANALYSIS:
            return analysis("npt_")
        if node_key.startswith("production_run_"):
            return (root / f"production_{name}.xtc",)
        if node_key == PRODUCTION_ANALYSIS:
            return analysis("production_") + (root / f"production_{name}_nopbc.xtc",)
        if node_key == PREPARE_RESULT:
            return (
                analysis("nvt_")
                + analysis("npt_")
                + analysis("production_")
                + (root / f"production_{name}_nopbc.xtc",)
            )
        raise ValueError(f"Unknown GROMACS Node {node_key!r}")

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

    def _decode_completed_calls(self) -> None:
        self._provider.decode_completed_calls(
            self.execution_run_id,
            observe_task=self._completed_task_observation,
            missing_message="GROMACS returned without a valid publication",
            now=self._now(),
        )

    def _completed_task_observation(
        self,
        node_key: str,
        _task: Any,
        envelope: object,
    ) -> AvailabilityStatus:
        remote_workdir = (
            envelope.get("remote_workdir") if isinstance(envelope, dict) else None
        )
        valid = (
            isinstance(remote_workdir, str)
            and bool(remote_workdir)
            and self._write_node_publication(node_key)
        )
        return self._node_observation(node_key) if valid else AvailabilityStatus.MISSING

    def _start_ready_nodes(self, required: set[str]) -> None:
        self._provider.start_ready_nodes(
            self.execution_run_id,
            required_node_keys=required,
            task_plans=lambda node_key: (operation_task_plan(node_key),),
            observe_task=lambda node_key, _task: self._node_observation(node_key),
            now=self._now(),
        )

    def _complete_local_result(self) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            node = repository.get_node(self.execution_run_id, PREPARE_RESULT)
            if node.status != NodeStatus.RUNNING:
                return
            task = repository.get_task(
                self.execution_run_id,
                PREPARE_RESULT,
                "operation",
            )
        if task.status != TaskStatus.PENDING:
            return
        with self.store.synchronize():
            with self.store.transaction():
                acquired = self.store.execution.acquire_local_task(
                    self.execution_run_id,
                    PREPARE_RESULT,
                    "operation",
                    now=self._now(),
                )
            if acquired:
                self._checkpoint()
        if not acquired:
            return
        self._write_node_publication(PREPARE_RESULT)
        observation = self._node_observation(PREPARE_RESULT)
        with self.store.transaction():
            repository = self.store.execution
            if repository.get_task(
                self.execution_run_id,
                PREPARE_RESULT,
                "operation",
            ).status.is_terminal:
                return
            if observation == AvailabilityStatus.MISSING:
                repository.fail_task(
                    self.execution_run_id,
                    PREPARE_RESULT,
                    "operation",
                    message="GROMACS final outputs are incomplete",
                    now=self._now(),
                )
            else:
                repository.record_task_result_observation(
                    self.execution_run_id,
                    PREPARE_RESULT,
                    "operation",
                    observation,
                    now=self._now(),
                )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=lambda node, task, rank: (
                None
                if node.node_key == PREPARE_RESULT
                else TaskDispatchDescriptor(
                    node_key=node.node_key,
                    node_ordinal=node.ordinal,
                    task_key=task.task_key,
                    task_ordinal=task.ordinal,
                    binding=operation_provider_binding(
                        node.node_key,
                        environment=self.deployment.environment,
                        app_name=self.deployment.deployment_name,
                        app_version=self.deployment.deployment_version,
                    ),
                    compatibility_key=node.node_key,
                    max_tasks_per_call=1,
                    depth=rank.depth,
                    unblocking_span=rank.unblocking_span,
                )
            ),
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
                    kwargs=self._invocation_kwargs(candidate.node_key),
                )
                for candidate in selected
            ),
            now=self._now(),
        )
        if any(call is None for call in submitted):
            return

    def _invocation_kwargs(self, node_key: str) -> dict[str, object]:
        request = self.request
        if node_key.startswith("prepare_tpr_"):
            return {
                "pdb_content": request.pdb_content,
                "run_name": request.run_name,
                "simulation_time_ns": request.simulation_time_ns,
                "run_pdbfixer": request.run_pdbfixer,
                "num_threads": request.num_threads,
                "use_openmp_threads": request.use_openmp_threads,
                "ld_seed": request.ld_seed,
                "gen_seed": request.gen_seed,
                "genion_seed": request.genion_seed,
            }
        invocation = modal_invocation(
            node_key,
            cpu_only=request.cpu_only,
            run_name=request.run_name,
            simulation_time_ns=request.simulation_time_ns,
        )
        if invocation.function_name.startswith("production_run_"):
            invocation.kwargs.update({
                "num_threads": request.num_threads,
                "use_openmp_threads": request.use_openmp_threads,
            })
        return invocation.kwargs


def _result_envelope(result: object) -> dict[str, object]:
    """Retain only the bounded output-directory reference."""
    return {"remote_workdir": result if isinstance(result, str) else None}


class GromacsExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to GROMACS publications."""

    _request_loader = staticmethod(load_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        modal_driver: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            target_scientific_versions={
                "biomodals.gromacs.execution_plan": EXECUTION_PLAN_SCHEMA_VERSION,
            },
        )
        self.output_volume = output_volume
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
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor from conclusive state."""
        self.prepare_restart(
            predecessor_execution_run_id=predecessor_execution_run_id,
            predecessor_deployment=predecessor_deployment,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            expected_workload_plan_fingerprint=expected_workload_plan_fingerprint,
        )
        return self.drive_prepared()

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
    ) -> None:
        """Validate and persist a Successor request without driving it."""
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                with self._open_successor_source(
                    predecessor_execution_run_id,
                    predecessor_deployment=predecessor_deployment,
                    expected_workload_plan_fingerprint=(
                        expected_workload_plan_fingerprint
                    ),
                ) as (predecessor, predecessor_request, _):
                    request = replace(
                        predecessor_request,
                        max_active_provider_calls=(
                            predecessor.max_active_provider_calls
                            if max_active_provider_calls is None
                            else max_active_provider_calls
                        ),
                        max_active_gpu_provider_calls=(
                            predecessor.max_active_gpu_provider_calls
                            if max_active_gpu_provider_calls is None
                            else max_active_gpu_provider_calls
                        ),
                    )
                self._require_successor_plan_match(predecessor, request)
                persist_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                    request,
                )
                persist_execution_launch(
                    self.volume_root,
                    self.execution_run_id,
                    predecessor_execution_run_id,
                )
                self.output_volume.commit()

    def _open_runtime(
        self,
        request: GromacsExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> GromacsExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active GROMACS runtime does not match request")
            return runtime
        runtime = GromacsExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_root=self.volume_root,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
