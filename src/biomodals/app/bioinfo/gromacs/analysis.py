"""Bounded-memory trajectory statistics using the original first-frame reference."""

from __future__ import annotations

from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from biomodals.app.bioinfo.gromacs.execution import (
    ANALYSIS_PACKAGES,
    ANALYSIS_POLICY_VERSION,
)
from biomodals.execution import ContentBoundFileSet
from biomodals.helper.artifacts import sha256_file
from biomodals.schema import ArtifactFile

_CHUNK_FRAMES = 128
_PLOT_POINTS = 5000


class TrajectoryStatistics:
    """Align chunks to one reference and merge Cα coordinate central moments."""

    def __init__(self, template: Any) -> None:
        """Retain protein annotations and O(Cα atoms) float64 moment arrays."""
        from biotite.structure.info import mass

        self.template = template
        self.ca = template.atom_name == "CA"
        self.masses = np.array([mass(element) for element in template.element])
        self.count = 0
        self.reference = None
        self.rmsd_reference = None
        self.last_frame = None
        self.mean = np.zeros((int(self.ca.sum()), 3), dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)

    def update(
        self, coordinates: np.ndarray, box: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return full-resolution RMSD/Rg and retain only O(atoms) running state."""
        import biotite.structure as struc

        if self.reference is None:
            self.reference = coordinates[0].copy()
        trajectory = struc.from_template(self.template, coordinates, box)
        aligned, _ = struc.superimpose(self.reference, trajectory)
        if self.rmsd_reference is None:
            self.rmsd_reference = aligned.coord[0].copy()
        self.last_frame = aligned[-1].copy()
        ca = aligned.coord[:, self.ca, :].astype(np.float64)
        chunk_mean = ca.mean(axis=0)
        chunk_m2 = np.square(ca - chunk_mean).sum(axis=0)
        n = len(coordinates)
        delta = chunk_mean - self.mean
        total = self.count + n
        self.m2 += chunk_m2 + delta * delta * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total
        return struc.rmsd(self.rmsd_reference, aligned), struc.gyration_radius(
            aligned, self.masses
        )

    def rmsf(self) -> np.ndarray:
        """Cα population RMSF after full-protein alignment, in angstroms."""
        if not self.count:
            raise ValueError("Trajectory contains no frames")
        return np.sqrt(self.m2.sum(axis=1) / self.count)


def plot_samples(path: Path, *, count: int) -> pl.DataFrame:
    """Read at most 5000 evenly spaced plotting rows, always including endpoints."""
    stride = max(1, (count - 1 + _PLOT_POINTS - 2) // (_PLOT_POINTS - 1))
    return (
        pl
        .scan_csv(path, row_index_name="_row")
        .filter((pl.col("_row") % stride == 0) | (pl.col("_row") == count - 1))
        .drop("_row")
        .collect(engine="streaming")
    )


def analyze_trajectory(
    trajectory_path: Path,
    template_path: Path,
    *,
    prefix: str,
    run_name: str,
    make_figures: bool,
    chunk_frames: int = _CHUNK_FRAMES,
) -> None:
    """Publish complete CSVs/plots without retaining cumulative coordinates.

    Stable per-output staging names are overwritten on recovery. A manifest is
    published last, so hard interruption never turns a truncated CSV into a hit.
    Chunk size is an internal test seam, not a scientific sampling parameter.
    """
    import biotite
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from biotite.structure.io.xtc import XTCFile

    root = trajectory_path.parent
    metrics = ("rmsd", "rg", "rmsf")
    paths = tuple(
        f"{metric}_{prefix}.{suffix}"
        for metric in metrics
        for suffix in (("csv", "png") if make_figures else ("csv",))
    ) + (f"{prefix}_last_frame.pdb",)
    publication = ContentBoundFileSet(
        root=root,
        marker_path=root / ".biomodals" / "gromacs" / f"analysis-{prefix}.json",
        expected_paths=paths,
        identity={
            "analysis_policy": ANALYSIS_POLICY_VERSION,
            "packages": list(ANALYSIS_PACKAGES),
            "trajectory_sha256": sha256_file(trajectory_path),
            "template_sha256": sha256_file(template_path),
            "run_name": run_name,
        },
    )
    if publication.load() is not None:
        return
    # Invalidate before overwriting any members; no mixed-generation cache hit.
    publication.marker_path.unlink(missing_ok=True)
    template = strucio.load_structure(template_path)
    protein_mask = struc.filter_amino_acids(template)
    template = template[protein_mask]
    if not template.array_length():
        raise ValueError("Trajectory has no protein atoms to analyze")
    statistics = TrajectoryStatistics(template)
    csv_paths = {metric: root / f"{metric}_{prefix}.csv" for metric in metrics}
    temporary = {
        metric: path.with_suffix(".csv.part") for metric, path in csv_paths.items()
    }
    with ExitStack() as stack:
        handles = {
            metric: stack.enter_context(temporary[metric].open("wb"))
            for metric in ("rmsd", "rg")
        }
        chunks = stack.enter_context(
            closing(
                XTCFile.read_iter(
                    trajectory_path,
                    atom_i=np.flatnonzero(protein_mask),
                    stack_size=chunk_frames,
                )
            )
        )
        for coordinates, box, time_ps in chunks:
            first = statistics.count == 0
            rmsd, rg = statistics.update(coordinates, box)
            for metric, values in (("rmsd", rmsd), ("rg", rg)):
                pl.DataFrame({"time_ns": time_ps / 1000.0, metric: values}).write_csv(
                    handles[metric], include_header=first, float_precision=5
                )
    rmsf = statistics.rmsf()
    pl.DataFrame({
        "residue_index": np.arange(1, len(rmsf) + 1, dtype=float),
        "rmsf": rmsf,
    }).write_csv(temporary["rmsf"], float_precision=5)
    for metric, path in csv_paths.items():
        temporary[metric].replace(path)
    last_frame = root / f"{prefix}_last_frame.pdb"
    last_frame_part = last_frame.with_suffix(".part.pdb")
    strucio.save_structure(last_frame_part, statistics.last_frame)
    last_frame_part.replace(last_frame)
    if make_figures:
        import matplotlib.pyplot as plt

        for metric, label, color in (
            ("rmsd", "RMSD (Å)", "dimorange"),
            ("rg", "Radius of Gyration (Å)", "dimgreen"),
            ("rmsf", "RMSF (Å)", "dimorange"),
        ):
            count = len(rmsf) if metric == "rmsf" else statistics.count
            samples = plot_samples(csv_paths[metric], count=count)
            x, y = samples.to_numpy().T
            figure, ax = plt.subplots(figsize=(6, 3), dpi=200, layout="constrained")
            try:
                ax.plot(x, y, color=biotite.colors[color])
                if len(x) > 1:
                    ax.set_xlim(x[0], x[-1])
                ax.set_title(run_name)
                xlabel = "Residue Index" if metric == "rmsf" else "Time (ns)"
                if count > _PLOT_POINTS:
                    xlabel += "\nSampled overview; full resolution in CSV"
                ax.set_xlabel(xlabel)
                ax.set_ylabel(label)
                figure.savefig(csv_paths[metric].with_suffix(".png"))
            finally:
                plt.close(figure)
    publication.write(
        tuple(
            ArtifactFile(
                path=name,
                size_bytes=(root / name).stat().st_size,
                content_sha256=sha256_file(root / name),
            )
            for name in paths
        )
    )
