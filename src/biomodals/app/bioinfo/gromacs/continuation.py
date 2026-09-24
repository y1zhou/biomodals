"""Immutable source evidence for checkpoint-based production extensions."""

from __future__ import annotations

import errno
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.app.bioinfo.gromacs.execution import (
    GROMACS_SCIENTIFIC_VERSION,
    PREPARE_RESULT,
)
from biomodals.execution import DeploymentIdentity, parse_content_bound_file_set
from biomodals.helper.artifacts import read_bounded_file_bytes
from biomodals.helper.io import require_safe_filename_component
from biomodals.schema import ArtifactFile

if TYPE_CHECKING:
    from biomodals.app.bioinfo.gromacs.execution_runtime import GromacsExecutionRequest

MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


class ContinuationSource(BaseModel):
    """Bind a new plan to one completed source and its native checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_run_id: UUID
    run_name: str
    file_stem: str
    simulation_time_ns: int = Field(ge=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("run_name", "file_stem")
    @classmethod
    def safe_component(cls, value: str) -> str:
        """Keep saved source paths within a single app-owned directory."""
        require_safe_filename_component(value, field_name="continuation source")
        return value


class ContinuationInspection(BaseModel):
    """Small remote metadata response; never carries source file contents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: ContinuationSource
    pdb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cpu_only: bool
    run_pdbfixer: bool
    num_threads: int = Field(ge=1)
    use_openmp_threads: bool
    ld_seed: int
    gen_seed: int
    genion_seed: int
    parent_job_id: UUID | None

    @classmethod
    def from_request(
        cls, source: ContinuationSource, request: GromacsExecutionRequest
    ) -> ContinuationInspection:
        """Strip the retained PDB while preserving inherited settings."""
        return cls(
            source=source,
            pdb_sha256=request.pdb_sha256,
            cpu_only=request.cpu_only,
            run_pdbfixer=request.run_pdbfixer,
            num_threads=request.num_threads,
            use_openmp_threads=request.use_openmp_threads,
            ld_seed=request.ld_seed,
            gen_seed=request.gen_seed,
            genion_seed=request.genion_seed,
            parent_job_id=(
                request.continuation.execution_run_id if request.continuation else None
            ),
        )

    def request(
        self,
        *,
        run_name: str,
        additional_time_ns: int,
        cpu_only: bool,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
    ) -> GromacsExecutionRequest:
        """Build a child request referring to the retained input by digest."""
        from biomodals.app.bioinfo.gromacs.execution_runtime import (
            GromacsExecutionRequest,
        )

        return GromacsExecutionRequest(
            run_name=run_name,
            pdb_content=b"",
            retained_pdb_sha256=self.pdb_sha256,
            simulation_time_ns=self.source.simulation_time_ns + additional_time_ns,
            run_pdbfixer=self.run_pdbfixer,
            cpu_only=cpu_only,
            num_threads=self.num_threads,
            use_openmp_threads=self.use_openmp_threads,
            ld_seed=self.ld_seed,
            gen_seed=self.gen_seed,
            genion_seed=self.genion_seed,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            continuation=self.source,
        )


class ContinuationEvidence(BaseModel):
    """Native checkpoint facts validated on the copied preparation snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    workload_plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: ContinuationSource
    source_checkpoint_step: int = Field(gt=0)
    source_checkpoint_time_ps: float = Field(gt=0)
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    additional_time_ns: int = Field(ge=1, le=250)
    target_time_ns: int = Field(ge=1)
    trajectory_scope: Literal["cumulative"] = "cumulative"
    equilibration_analysis: Literal["inherited"] = "inherited"
    production_mdp: Literal["original input; extended TPR is authoritative"] = (
        "original input; extended TPR is authoritative"
    )

    def validate_request(self, request: GromacsExecutionRequest) -> None:
        """Reject evidence for a different source, child plan or endpoint."""
        if (
            self.workload_plan_fingerprint
            != request.execution_plan.workload_plan_fingerprint
            or self.source != request.continuation
            or self.target_time_ns != request.simulation_time_ns
            or self.additional_time_ns
            != self.target_time_ns - self.source.simulation_time_ns
            or self.source_checkpoint_time_ps != self.source.simulation_time_ns * 1000
            or (
                self.source.checkpoint_sha256 is not None
                and self.source_checkpoint_sha256 != self.source.checkpoint_sha256
            )
        ):
            raise ValueError(
                "Continuation preparation evidence differs from its request"
            )


def continuation_file_records(
    request: GromacsExecutionRequest, final_marker: bytes, production_marker: bytes
) -> dict[str, ArtifactFile]:
    """Read source evidence without imposing today's exact output inventory.

    Records remain bound to the saved scientific identity. Unrecorded restart
    files are resolved only inside that same run directory, then validated by
    the checkpoint's native append checks before use.
    """
    records: dict[str, ArtifactFile] = {}
    production = "production_run_cpu" if request.cpu_only else "production_run_gpu"
    for node, content in (
        (PREPARE_RESULT, final_marker),
        (production, production_marker),
    ):
        try:
            marker = orjson.loads(content)
            paths = tuple(file["path"] for file in marker["files"])
        except (orjson.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("Source publication is invalid") from error
        files = parse_content_bound_file_set(
            content,
            expected_paths=paths,
            identity={
                "node_key": node,
                "workload_plan_fingerprint": request.execution_plan.workload_plan_fingerprint,
            },
        )
        if not files or len(set(file.path for file in files)) != len(files):
            raise ValueError("Source publication is invalid")
        for file in files:
            require_safe_filename_component(
                file.path, field_name="source publication file"
            )
            previous = records.get(file.path)
            if previous and (previous.size_bytes, previous.content_sha256) != (
                file.size_bytes,
                file.content_sha256,
            ):
                raise ValueError(f"Source publications disagree for file: {file.path}")
            records[file.path] = file
    return records


def inspect_continuation_source(
    volume_root: Path, execution_run_id: UUID
) -> ContinuationInspection:
    """Read records and stat native files on the mounted Volume without copying.

    No checkpoint/trajectory content is read here. Full native validation is a
    submitted child's preparation Task, not a side effect of opening its form.
    """
    from biomodals.app.bioinfo.gromacs.execution_runtime import (
        GromacsExecutionRequest,
        gromacs_publication_path,
        load_execution_request,
    )

    request = load_execution_request(volume_root, execution_run_id)
    if not isinstance(request, GromacsExecutionRequest):
        raise ValueError("Only a simulation can be extended")
    if request.gromacs_version != GROMACS_SCIENTIFIC_VERSION:
        raise ValueError("Source uses an incompatible GROMACS version")
    root = request.run_root(volume_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source directory is missing or unsafe")
    marker = read_bounded_file_bytes(
        volume_root / gromacs_publication_path(request, PREPARE_RESULT),
        field_name="Source final publication",
        max_bytes=1024 * 1024,
    )
    production = "production_run_cpu" if request.cpu_only else "production_run_gpu"
    native_marker = read_bounded_file_bytes(
        volume_root / gromacs_publication_path(request, production),
        field_name="Source production publication",
        max_bytes=1024 * 1024,
    )
    records = continuation_file_records(request, marker, native_marker)
    for suffix in (".tpr", ".xtc", ".edr", ".log", ".cpt"):
        name = f"production_{request.file_stem}{suffix}"
        record = records.get(name)
        # Unlisted historical files use the directory bound by the saved request
        # and publications; never search other runs or accept client paths.
        path = root / (record.path if record else name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                errno.ENOENT, "Source restart file is missing or empty", name
            )
        if record and path.stat().st_size != record.size_bytes:
            raise ValueError(f"Source file no longer matches its publication: {name}")
        if suffix == ".cpt" and path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("Source checkpoint exceeds its byte limit")
    published_checkpoint = records.get(f"production_{request.file_stem}.cpt")
    source = ContinuationSource(
        execution_run_id=execution_run_id,
        run_name=request.run_name,
        file_stem=request.file_stem,
        simulation_time_ns=request.simulation_time_ns,
        request_sha256=sha256(request.to_bytes()).hexdigest(),
        publication_sha256=sha256(marker).hexdigest(),
        checkpoint_sha256=(
            published_checkpoint.content_sha256 if published_checkpoint else None
        ),
    )
    return ContinuationInspection.from_request(source, request)


async def read_continuation_source(
    deployment: DeploymentIdentity, execution_run_id: UUID
) -> ContinuationInspection:
    """Return only metadata from the selected deployment's read-only inspector."""
    import modal

    function = modal.Function.from_name(
        deployment.deployment_name,
        "inspect_continuation_source",
        environment_name=deployment.environment,
        version=deployment.deployment_version,
    )
    inspection = ContinuationInspection.model_validate_json(
        await function.remote.aio(str(execution_run_id))
    )
    if inspection.source.execution_run_id != execution_run_id:
        raise ValueError("Source inspection belongs to a different Execution Run")
    return inspection
