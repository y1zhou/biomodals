"""Exact all-frame GROMOS clustering of retained processed protein trajectories."""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
import zipfile
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from uuid import UUID

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.app.bioinfo.gromacs.execution import (
    GROMACS_SCIENTIFIC_VERSION,
    PREPARE_RESULT,
)
from biomodals.execution import ExecutionPlan, NodeDependency, NodePlan
from biomodals.helper.artifacts import file_size_sha256, read_bounded_file_bytes
from biomodals.helper.io import require_safe_filename_component
from biomodals.schema import ArtifactFile

if TYPE_CHECKING:
    import numpy as np

CLUSTER_TIMEOUT_SECONDS = 12 * 60 * 60
CLUSTER_POLICY_VERSION = "gromos-ca-v1"
CLUSTER_ARCHIVE = "clusters.zip"
CLUSTER_NODE = "cluster_trajectory"


class ClusteringSource(BaseModel):
    """Immutable source publication identity and small resource-estimate metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_run_id: UUID
    run_name: str
    publication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory: ArtifactFile
    template: ArtifactFile
    frame_count: int = Field(ge=1)
    protein_atoms: int = Field(ge=1)
    ca_atoms: int = Field(ge=3)

    @field_validator("run_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Keep source directory references within the output volume."""
        require_safe_filename_component(value, field_name="clustering source")
        return value

    @field_validator("trajectory", "template")
    @classmethod
    def bound_file(cls, value: ArtifactFile) -> ArtifactFile:
        """Require a single native file with a complete content identity."""
        require_safe_filename_component(value.path, field_name="clustering file")
        if not value.size_bytes or not value.content_sha256:
            raise ValueError("Clustering requires content-bound source files")
        return value

    @property
    def estimated_memory_bytes(self) -> int:
        """Conservative dense matrix/neighbour/coordinate estimate, not a limit."""
        return 8 * self.frame_count**2 + 12 * self.frame_count * self.ca_atoms

    @property
    def warnings(self) -> list[str]:
        """Warn without silently sampling or prohibiting explicit submission."""
        pair_work = self.frame_count * (self.frame_count - 1) // 2 * self.ca_atoms
        warnings = []
        if self.estimated_memory_bytes >= 8 * 1024**3 or pair_work >= 10**11:
            warnings.append(
                "Exact all-frame clustering can require substantial memory and time. "
                "This estimate is not a completion guarantee; the job has a 12-hour "
                "deadline and finite memory. All frames will be used without sampling."
            )
        return warnings


@dataclass(frozen=True)
class ClusteringExecutionRequest:
    """Analysis-only request, without artificial MD duration or velocity settings."""

    run_name: str
    source: ClusteringSource
    cutoff_angstrom: float = 2.0
    max_active_provider_calls: int = 1
    max_active_gpu_provider_calls: int = 1

    def __post_init__(self) -> None:
        """Validate the scientific cutoff and normal scheduler limits."""
        require_safe_filename_component(self.run_name, field_name="clustering run")
        if self.run_name == self.source.run_name:
            raise ValueError("Clustering cannot write into the source simulation")
        if not math.isfinite(self.cutoff_angstrom) or self.cutoff_angstrom <= 0:
            raise ValueError("RMSD cutoff must be finite and positive")
        if self.max_active_provider_calls < 1 or not (
            0 <= self.max_active_gpu_provider_calls <= self.max_active_provider_calls
        ):
            raise ValueError("Provider call limits are invalid")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Use the shared kernel for one native analysis and result publication."""
        return ExecutionPlan(
            workload_name="gromacs",
            workload_run_key=self.run_name,
            nodes=(
                NodePlan(node_key=CLUSTER_NODE),
                NodePlan(
                    node_key=PREPARE_RESULT,
                    dependencies=(NodeDependency(node_key=CLUSTER_NODE),),
                ),
            ),
            scientific_payload={
                "operation": "trajectory_clustering",
                "source": self.source.model_dump(mode="json"),
                "cutoff_angstrom": self.cutoff_angstrom,
            },
            scientific_versions={
                "gromacs": GROMACS_SCIENTIFIC_VERSION,
                "biomodals.gromacs.clustering": CLUSTER_POLICY_VERSION,
            },
        )

    def run_root(self, volume_root: str | Path) -> Path:
        """Return the analysis-owned output directory."""
        return Path(volume_root) / self.run_name

    def to_bytes(self) -> bytes:
        """Serialize a discriminated request without borrowing MD schema fields."""
        return orjson.dumps(
            {
                "schema_version": 1,
                "operation": "trajectory_clustering",
                "run_name": self.run_name,
                "source": self.source.model_dump(mode="json"),
                "cutoff_angstrom": self.cutoff_angstrom,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
            },
            option=orjson.OPT_SORT_KEYS,
        )

    @classmethod
    def from_bytes(cls, content: bytes) -> ClusteringExecutionRequest:
        """Read one current analysis request; reject incompatible versions."""
        data = orjson.loads(content)
        if (
            data.pop("schema_version") != 1
            or data.pop("operation") != "trajectory_clustering"
        ):
            raise ValueError("Unsupported GROMACS clustering request")
        data["source"] = ClusteringSource.model_validate(data["source"])
        return cls(**data)


def inspect_clustering_source(
    volume_root: Path, execution_run_id: UUID
) -> ClusteringSource:
    """Inspect published MD metadata remotely, without downloading the trajectory."""
    import biotite.structure as struc
    import biotite.structure.io as strucio
    import numpy as np
    import polars as pl

    from biomodals.app.bioinfo.gromacs.execution_runtime import (
        GromacsExecutionRequest,
        gromacs_publication_path,
        load_execution_request,
        parse_gromacs_publication,
    )

    request = load_execution_request(volume_root, execution_run_id)
    if not isinstance(request, GromacsExecutionRequest):
        raise ValueError(
            "Clustering requires a completed simulation, not another analysis"
        )
    root = request.run_root(volume_root)
    marker = read_bounded_file_bytes(
        volume_root / gromacs_publication_path(request, PREPARE_RESULT),
        max_bytes=1024 * 1024,
        field_name="GROMACS source publication",
    )
    files = parse_gromacs_publication(request, PREPARE_RESULT, marker)
    if files is None:
        raise ValueError("Source simulation publication is invalid")
    records = {file.path: file for file in files}
    prefix = f"production_{request.file_stem}"
    trajectory = records[f"{prefix}_nopbc.xtc"]
    template = records[f"{prefix}_nopbc_centered.pdb"]
    statistics = records[f"rmsd_{prefix}.csv"]
    for record in (trajectory, template, statistics):
        path = root / record.path
        if root.is_symlink() or path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Retained clustering input is unavailable")
        if path.stat().st_size != record.size_bytes:
            raise ValueError("Retained clustering input size changed")
        if record != trajectory and file_size_sha256(path)[1] != record.content_sha256:
            raise ValueError("Retained clustering metadata changed")
    atoms = strucio.load_structure(root / template.path)
    if not np.all(struc.filter_amino_acids(atoms)):
        raise ValueError("Processed trajectory template must contain protein only")
    frames = (
        pl
        .scan_csv(root / statistics.path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    return ClusteringSource(
        execution_run_id=execution_run_id,
        run_name=request.run_name,
        publication_sha256=sha256(marker).hexdigest(),
        trajectory=trajectory,
        template=template,
        frame_count=frames,
        protein_atoms=len(atoms),
        ca_atoms=int(np.count_nonzero(atoms.atom_name == "CA")),
    )


def read_memberships(path: Path, frame_count: int) -> np.ndarray:
    """Decode native one-based member ordinals; require exactly one assignment."""
    import numpy as np

    assignments = np.zeros(frame_count, dtype=np.int64)
    cluster = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("["):
                match = re.fullmatch(r"\s*\[Cluster_(\d+)\]\s*", line)
                if match is None or int(match[1]) != cluster + 1:
                    raise ValueError("Native cluster IDs are invalid")
                cluster += 1
            else:
                for value in line.split():
                    ordinal = int(value) - 1
                    if (
                        not cluster
                        or not 0 <= ordinal < frame_count
                        or assignments[ordinal]
                    ):
                        raise ValueError("Native cluster membership is invalid")
                    assignments[ordinal] = cluster
    if not assignments.all():
        raise ValueError("Native clustering did not assign every source frame")
    return assignments


def cluster_trajectory(
    trajectory: Path,
    template_path: Path,
    destination: Path,
    *,
    run_name: str,
    cutoff_angstrom: float,
    provenance: dict,
    expected_frames: int,
    gmx: str = "gmx",
    timeout: float = CLUSTER_TIMEOUT_SECONDS - 30,
) -> None:
    """Publish an exact CSV/medoid inventory, leaving native scratch off the volume."""
    import biotite.structure as struc
    import biotite.structure.io as strucio
    import numpy as np
    import polars as pl
    from biotite.structure.io.trr import TRRFile
    from biotite.structure.io.xtc import XTCFile

    require_safe_filename_component(run_name, field_name="clustering run name")
    if not np.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0:
        raise ValueError("RMSD cutoff must be finite and positive")
    deadline = time.monotonic() + timeout
    template = strucio.load_structure(template_path)
    ca = np.flatnonzero(template.atom_name == "CA")
    if len(ca) < 3 or not np.all(struc.filter_amino_acids(template)):
        raise ValueError(
            "Clustering requires a protein template with at least three C-alpha atoms"
        )
    # Only O(frames) scalar metadata is retained outside the native all-pairs tool.
    times = []
    with closing(XTCFile.read_iter(trajectory, stack_size=128, atom_i=ca)) as chunks:
        for _, _, chunk_times in chunks:
            if time.monotonic() >= deadline:
                raise TimeoutError("Clustering exceeded its execution deadline")
            times.append(chunk_times)
    times = np.concatenate(times) if times else np.array([])
    if (
        len(times) != expected_frames
        or not np.isfinite(times).all()
        or len(np.unique(times)) != len(times)
    ):
        raise ValueError(
            "Source trajectory must match the published frame count and have finite, unique timestamps"
        )
    partial = destination.with_suffix(".zip.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="gromacs-clustering-") as temporary:
            scratch = Path(temporary)
            command = [
                gmx,
                "cluster",
                "-f",
                str(trajectory.resolve()),
                "-s",
                str(template_path.resolve()),
                "-method",
                "gromos",
                "-cutoff",
                str(cutoff_angstrom / 10),
                "-fit",
                "-noav",
                "-nopbc",
                "-tu",
                "ps",
                "-cl",
                "medoids.trr",
                "-clndx",
                "members.ndx",
                "-g",
                "cluster.log",
                "-o",
                "clusters.xpm",
                "-om",
                "matrix.xpm",
                "-xvg",
                "none",
            ]
            with (scratch / "native.log").open("wb") as log:
                subprocess.run(  # noqa: S603 -- argv, no user shell expansion
                    command,
                    input=b"C-alpha\nC-alpha\n",
                    cwd=scratch,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(1, deadline - time.monotonic()),
                    env=os.environ | {"OMP_NUM_THREADS": "1", "GMX_MAXBACKUP": "-1"},
                )
            assignments = read_memberships(scratch / "members.ndx", len(times))
            # Binary timestamps retain distinctions lost by native text-log rounding.
            lookup = {float(value): index for index, value in enumerate(times)}
            medoids = []
            with closing(
                TRRFile.read_iter(scratch / "medoids.trr", stack_size=128)
            ) as chunks:
                for _, _, chunk_times in chunks:
                    medoids.extend(lookup[float(value)] for value in chunk_times)
            if len(medoids) != int(assignments.max()) or any(
                assignments[frame] != cluster
                for cluster, frame in enumerate(medoids, 1)
            ):
                raise ValueError("Native medoids do not match cluster memberships")
            medoid_mask = np.zeros(len(times), dtype=bool)
            medoid_mask[medoids] = True
            table = pl.DataFrame({
                "frame": np.arange(len(times), dtype=np.int64),
                "time_ns": times.astype(np.float64) / 1000,
                "cluster_id": assignments,
                "is_medoid": medoid_mask,
            }).with_columns(
                pl
                .when(pl.col("is_medoid"))
                .then(pl.lit("yes"))
                .otherwise(None)
                .alias("is_medoid")
            )
            csv = scratch / "clusters.csv"
            table.write_csv(csv)
            records = []
            with zipfile.ZipFile(
                partial, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:

                def add(path: Path, name: str) -> None:
                    size, digest = file_size_sha256(path)
                    records.append({"path": name, "size_bytes": size, "sha256": digest})
                    archive.write(path, name)

                add(csv, "clusters.csv")
                selected = set(medoids)
                offset = 0
                with closing(XTCFile.read_iter(trajectory, stack_size=128)) as chunks:
                    for coordinates, boxes, _ in chunks:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Clustering exceeded its execution deadline"
                            )
                        if coordinates.shape[1] != len(template):
                            raise ValueError(
                                "Trajectory and protein template atom counts differ"
                            )
                        for local in range(len(coordinates)):
                            frame = offset + local
                            if frame in selected:
                                atom_array = template.copy()
                                atom_array.coord = coordinates[local]
                                if boxes is not None:
                                    atom_array.box = boxes[local]
                                name = f"{run_name}-frame_{frame}.pdb"
                                path = scratch / name
                                strucio.save_structure(path, atom_array)
                                add(path, f"medoids/{name}")
                                path.unlink()
                                selected.remove(frame)
                        offset += len(coordinates)
                if selected or offset != len(times):
                    raise ValueError("Source trajectory changed during medoid export")
                archive.writestr(
                    "provenance.json",
                    orjson.dumps(
                        {
                            "schema_version": 1,
                            "source": provenance,
                            "policy": CLUSTER_POLICY_VERSION,
                            "cutoff_angstrom": cutoff_angstrom,
                            "frame_count": len(times),
                            "cluster_count": len(medoids),
                            "frame_index_base": 0,
                            "cluster_index_base": 1,
                            "coordinate_frame": "processed source trajectory",
                            "files": records,
                        },
                        option=orjson.OPT_SORT_KEYS,
                    ),
                )
            os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
