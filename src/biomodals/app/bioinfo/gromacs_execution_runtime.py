"""Direct GROMACS adaptation of the shared execution kernel."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.bioinfo.gromacs_execution import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    GROMACS_SCIENTIFIC_VERSION,
    NPT_ANALYSIS,
    NVT_ANALYSIS,
    PREPARE_RESULT,
    PRODUCTION_ANALYSIS,
    execution_plan,
    modal_invocation,
    operation_target,
    preparation_execution_paths,
)
from biomodals.execution import (
    AvailabilityStatus,
    ContentBoundFileSet,
    DeploymentIdentity,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlanMetadata,
)
from biomodals.execution.modal import (
    ExecutionDefinitionCoordinatorLifecycle,
    ExecutionRequestFile,
)
from biomodals.execution.nodes import (
    CoordinatorNode,
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
)
from biomodals.helper.artifacts import replace_bytes_atomic
from biomodals.helper.io import require_safe_filename_component
from biomodals.helper.output_claim import (
    acquire_output_claim,
    register_output_claim_successor,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
)

REQUEST_SCHEMA_VERSION = 2
MAX_REQUEST_BYTES = 32 * 1024 * 1024
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "GROMACS execution request",
)
_RUN_IDENTITY_SCHEMA_VERSION = 2
_RUN_IDENTITY_FILE = "run.json"
_OUTPUT_CLAIMS_NAME = "Gromacs-output-claims"
_OUTPUT_VOLUME_NAME = "Gromacs-outputs"


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
    gromacs_version: str = GROMACS_SCIENTIFIC_VERSION
    execution_plan_version: str = EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject invalid identities and unusable operational limits."""
        require_safe_filename_component(self.run_name, field_name="run_name")
        if not self.pdb_content:
            raise ValueError("pdb_content cannot be empty")
        if self.simulation_time_ns < 1 or self.num_threads < 1:
            raise ValueError("simulation time and thread count must be positive")
        if self.ld_seed == -1 or self.gen_seed == -1 or self.genion_seed == 0:
            raise ValueError("GROMACS random sentinels must be materialized")
        if self.max_active_provider_calls < 1:
            raise ValueError("max_active_provider_calls must be positive")
        if not self.gromacs_version or not self.execution_plan_version:
            raise ValueError("GROMACS scientific versions cannot be empty")
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
            gromacs_version=self.gromacs_version,
            execution_plan_version=self.execution_plan_version,
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
                "gromacs_version": self.gromacs_version,
                "execution_plan_version": self.execution_plan_version,
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


class GromacsPublications:
    """Own GROMACS run-name claims and scientific publication markers."""

    def __init__(
        self,
        *,
        request: GromacsExecutionRequest,
        execution_run_id: UUID,
        predecessor_execution_run_id: UUID | None,
        output_root: str | Path,
        output_claims: Any,
        output_volume_name: str,
    ) -> None:
        """Bind one request to its established output directory and owner."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.output_root = Path(output_root)
        self.output_claims = output_claims
        self.output_volume_name = output_volume_name

    def ensure_run_identity(self) -> bool:
        """Bind the app-owned directory to exactly one scientific plan."""
        root = self.request.run_root(self.output_root)
        marker = self.run_identity_path()
        scientific_identity = {
            "schema_version": _RUN_IDENTITY_SCHEMA_VERSION,
            "workload_plan_fingerprint": (
                self.request.execution_plan.workload_plan_fingerprint
            ),
        }
        recorded: object = None
        publication_complete = False
        if root.is_symlink():
            raise ValueError(f"GROMACS run directory cannot be a symlink: {root}")
        if marker.exists():
            if marker.is_symlink() or not marker.is_file():
                raise ValueError(
                    f"GROMACS run identity is not a regular file: {marker}"
                )
            try:
                recorded = orjson.loads(marker.read_bytes())
            except orjson.JSONDecodeError as error:
                raise ValueError(
                    f"GROMACS run identity is invalid: {marker}"
                ) from error
            if not isinstance(recorded, dict) or any(
                recorded.get(key) != value for key, value in scientific_identity.items()
            ):
                raise ValueError(
                    f"GROMACS run name {self.request.run_name!r} is already bound "
                    "to different scientific inputs; choose a new run name"
                )
            marker_owner = recorded.get("owner_execution_run_id")
            try:
                if str(UUID(str(marker_owner))) != marker_owner:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"GROMACS run identity has an invalid owner: {marker}"
                ) from error
            publication_complete = all(
                self._read_node_publication(node_key) is not None
                for node_key in self.request.execution_plan.terminal_node_keys
            )
        if root.exists():
            if not root.is_dir():
                raise ValueError(f"GROMACS run path is not a directory: {root}")
            if recorded is None and any(root.iterdir()):
                raise ValueError(
                    f"GROMACS run name {self.request.run_name!r} has unclaimed "
                    "existing outputs; choose a new run name"
                )
        owner = str(self.execution_run_id)
        replace_owner = (
            None
            if self.predecessor_execution_run_id is None
            else str(self.predecessor_execution_run_id)
        )
        if replace_owner is not None:
            register_output_claim_successor(
                self.output_claims,
                owner=owner,
                predecessor=replace_owner,
            )
        if publication_complete:
            return False
        acquire_output_claim(
            self.output_claims,
            claim_key=f"gromacs-run:{self.request.run_name}",
            owner=owner,
            replace_owner=replace_owner,
        )
        expected = scientific_identity | {"owner_execution_run_id": owner}
        if recorded == expected:
            return False
        replace_bytes_atomic(
            marker,
            orjson.dumps(expected, option=orjson.OPT_SORT_KEYS),
        )
        return True

    def run_identity_path(self) -> Path:
        """Return the immutable run-name identity marker path."""
        return (
            self.request.run_root(self.output_root)
            / ".biomodals"
            / "gromacs"
            / _RUN_IDENTITY_FILE
        )

    def publication_path(self, node_key: str) -> Path:
        """Return one operation's content-bound publication marker path."""
        marker = sha256(node_key.encode()).hexdigest() + ".json"
        return (
            self.request.run_root(self.output_root) / ".biomodals" / "gromacs" / marker
        )

    def recover_result(self, node_key: str) -> AppRunResult | None:
        """Return a content-bound result only for a valid existing marker."""
        try:
            files = self._read_node_publication(node_key)
        except OSError:
            raise
        if files is None:
            if node_key != PREPARE_RESULT and self.publication_path(node_key).is_file():
                self.invalidate(node_key)
            return None
        return self.result(node_key, files=files)

    def commit(
        self,
        node_key: str,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus:
        """Write the workload marker from kernel-validated exact manifests."""
        files = tuple(file for artifact in artifacts for file in artifact.files)
        try:
            self._file_set(node_key).write(files)
        except ValueError:
            return AvailabilityStatus.MISSING
        return self.observe(node_key)

    def observe(self, node_key: str) -> AvailabilityStatus:
        """Validate both the workload marker and its exact file contents."""
        try:
            available = self._read_node_publication(node_key) is not None
        except OSError:
            return AvailabilityStatus.UNKNOWN
        if available:
            return AvailabilityStatus.AVAILABLE
        if node_key != PREPARE_RESULT and self.publication_path(node_key).is_file():
            self.invalidate(node_key)
        return AvailabilityStatus.MISSING

    def result(
        self,
        node_key: str,
        *,
        files: tuple[ArtifactFile, ...] | None = None,
    ) -> AppRunResult:
        """Describe one established output directory without copying it."""
        declared = files or tuple(
            ArtifactFile(
                path=path.relative_to(
                    self.request.run_root(self.output_root)
                ).as_posix()
            )
            for path in self.node_paths(node_key)
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=self.request.run_name,
                    ),
                    metadata={
                        "files": [
                            file.model_dump(mode="json", exclude_none=True)
                            for file in declared
                        ]
                    },
                )
            ],
        )

    def invalidate(self, node_key: str) -> None:
        """Remove digest-invalid outputs before authorizing repair."""
        for path in self.node_paths(node_key):
            path.unlink(missing_ok=True)
        self.publication_path(node_key).unlink(missing_ok=True)

    def node_paths(self, node_key: str) -> tuple[Path, ...]:
        """Return the exact scientific files published by one operation."""
        root = self.request.run_root(self.output_root)
        name = self.request.run_name
        prepare = tuple(root / path for path in preparation_execution_paths(name))

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
            return analysis("production_") + (
                root / f"production_{name}_nopbc.xtc",
                root / f"production_{name}_nopbc_centered.pdb",
            )
        if node_key == PREPARE_RESULT:
            return (
                analysis("nvt_")
                + analysis("npt_")
                + analysis("production_")
                + (
                    root / "production.mdp",
                    root / f"production_{name}.tpr",
                    root / f"production_{name}_nopbc.xtc",
                    root / f"production_{name}_nopbc_centered.pdb",
                )
            )
        raise ValueError(f"Unknown GROMACS Node {node_key!r}")

    def _read_node_publication(
        self,
        node_key: str,
    ) -> tuple[ArtifactFile, ...] | None:
        return self._file_set(node_key).load()

    def _file_set(self, node_key: str) -> ContentBoundFileSet:
        root = self.request.run_root(self.output_root)
        return ContentBoundFileSet(
            root=root,
            marker_path=self.publication_path(node_key),
            expected_paths=tuple(
                path.relative_to(root).as_posix() for path in self.node_paths(node_key)
            ),
            identity={
                "node_key": node_key,
                "workload_plan_fingerprint": (
                    self.request.execution_plan.workload_plan_fingerprint
                ),
            },
        )


class _GromacsPublicationHooks:
    operation: str
    publications: GromacsPublications

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        return self.publications.recover_result(self.operation)

    def commit_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus:
        del context, result
        return self.publications.commit(self.operation, artifacts)

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus:
        del context, result, artifacts
        return self.publications.observe(self.operation)


@dataclass(frozen=True)
class GromacsProviderNode(_GromacsPublicationHooks, ProviderNode):
    """Describe one established remote GROMACS operation."""

    operation: str
    request: GromacsExecutionRequest
    publications: GromacsPublications

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Describe the established deployed GROMACS function call."""
        del context
        target = operation_target(self.operation)
        return ProviderCallSpec(
            function_name=target.function_name,
            uses_gpu=target.uses_gpu,
            kwargs=_operation_kwargs(self.request, self.operation),
            runtime_image_key=target.runtime_image_key,
            compatibility_key=self.operation,
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Any,
    ) -> AppRunResult:
        """Convert the remote directory reference into an exact result."""
        del metadata
        if not isinstance(result, str) or not result:
            raise ValueError("GROMACS returned no output directory")
        return self.publications.result(self.operation)


@dataclass(frozen=True)
class GromacsResultNode(_GromacsPublicationHooks, CoordinatorNode):
    """Publish the complete user-facing GROMACS result boundary."""

    operation: str
    publications: GromacsPublications

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Describe the complete user-facing files after dependencies finish."""
        del context
        return self.publications.result(self.operation)


def gromacs_execution_graph(
    request: GromacsExecutionRequest,
    publications: GromacsPublications,
) -> ExecutionGraph:
    """Build the direct-app graph without reproducing kernel orchestration."""
    plan = request.execution_plan
    graph = ExecutionGraph(
        "gromacs",
        plan_metadata=ExecutionPlanMetadata(
            workload_name=plan.workload_name,
            scientific_payload=plan.scientific_payload,
            scientific_versions=dict(plan.scientific_versions),
        ),
    )
    handles = {}
    for node in plan.nodes:
        implementation = (
            GromacsResultNode(node.node_key, publications)
            if node.node_key == PREPARE_RESULT
            else GromacsProviderNode(node.node_key, request, publications)
        )
        handles[node.node_key] = graph.add_node(
            implementation,
            id=node.node_key,
            depends_on=[handles[item.node_key] for item in node.dependencies],
        )
    return graph


def _operation_kwargs(
    request: GromacsExecutionRequest,
    operation: str,
) -> dict[str, object]:
    if operation.startswith("prepare_tpr_"):
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
        operation,
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


class GromacsExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind GROMACS publications to the shared definition host."""

    _request_loader = staticmethod(load_execution_request)
    _request_persister = staticmethod(persist_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        provider_driver: Any,
        output_claims: Any | None = None,
        output_volume_name: str = _OUTPUT_VOLUME_NAME,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the deployment resources used by this adapter."""
        if output_claims is None:
            import modal

            output_claims = modal.Dict.from_name(
                _OUTPUT_CLAIMS_NAME,
                create_if_missing=True,
            )
        self.output_claims = output_claims
        self.output_volume_name = output_volume_name
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._build_graph,
            target_scientific_versions={
                "gromacs": GROMACS_SCIENTIFIC_VERSION,
                "biomodals.gromacs.execution_plan": EXECUTION_PLAN_SCHEMA_VERSION,
            },
            poll_interval_seconds=poll_interval_seconds,
        )

    def _build_graph(
        self,
        request: GromacsExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        publications = GromacsPublications(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            output_root=self.volume_root,
            output_claims=self.output_claims,
            output_volume_name=self.output_volume_name,
        )
        if publications.ensure_run_identity():
            self.output_volume.commit()
        return gromacs_execution_graph(request, publications)
