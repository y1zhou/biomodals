"""Single-domain pI/germlines, informational and independent from ranking."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from biomodals.helper.antibody import assign_germlines, sequence_pi
from biomodals.helper.antibody_tables import (
    GERMLINE_ASSIGNMENT_SCHEMA,
    GERMLINE_TABLE_SCHEMA,
)
from biomodals.helper.shell import package_outputs
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


def annotation_tables(candidates: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute each distinct supplied sequence once and bind evidence to every row."""
    sequences = (
        candidates
        .select("vh")
        .unique()
        .with_columns(
            pl
            .col("vh")
            .map_elements(sequence_pi, return_dtype=pl.Float64)
            .alias("vh_pI"),
            pl
            .col("vh")
            .map_elements(
                assign_germlines, return_dtype=pl.Struct(GERMLINE_ASSIGNMENT_SCHEMA)
            )
            .alias("_germline"),
            pl
            .col("vh")
            .map_elements(
                lambda value: hashlib.sha256(value.encode()).hexdigest(),
                return_dtype=pl.String,
            )
            .alias("sequence_sha256"),
        )
        .unnest("_germline")
    )
    joined = candidates.join(
        sequences, on="vh", how="left", validate="m:1", maintain_order="left"
    )
    summary = joined.select(
        "parent_id",
        "candidate_id",
        "vh_pI",
        pl.col("v_gene").alias("vh_v_gene"),
        pl.col("j_gene").alias("vh_j_gene"),
        pl.col("error").alias("annotation_error"),
    )
    germlines = joined.with_columns(pl.lit("vh").alias("chain")).select(
        *GERMLINE_TABLE_SCHEMA
    )
    return summary, germlines


def annotate_nanobody_candidates(csv_bytes: bytes) -> AppRunResult:
    """Publish bounded Parquet evidence on CPU; never trim or impute candidates."""
    if not 0 < len(csv_bytes) <= 4 * 1024 * 1024:
        raise ValueError("Candidate annotation input exceeds its byte limit")
    candidates = pl.read_csv(BytesIO(csv_bytes), infer_schema=False)
    if (
        candidates.columns != ["parent_id", "candidate_id", "vh"]
        or not 1 <= candidates.height <= 5400
    ):
        raise ValueError("Unexpected candidate annotation input")
    summary, germlines = annotation_tables(candidates)
    with TemporaryDirectory(prefix="nanobody-annotation-") as directory:
        root = Path(directory) / "annotations"
        root.mkdir()
        summary.write_parquet(root / "annotations.parquet", compression="zstd")
        germlines.write_parquet(root / "germlines.parquet", compression="zstd")
        archive = package_outputs(root, num_threads=2)
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="annotations",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    filename="annotations.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                    data=archive,
                ),
            )
        ],
    )
