"""Bounded VHH generation and independent VH2/VHH2 evaluation."""

# ruff: noqa: PLC0415 - scientific dependencies live only in the task image.

from __future__ import annotations

from contextlib import ExitStack
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import orjson
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.design.abnativ2_vhh.contracts import (
    SCORE_BATCH_SIZE,
    HumanizationSettings,
)
from biomodals.app.design.abnativ2_vhh.models import (
    MODEL_ROOT,
    RUNTIME_IDENTITY,
    SOURCE_COMMIT,
    assert_assets,
)
from biomodals.app.design.abnativ2_vhh.search import generate
from biomodals.app.design.vhh import VHHInput
from biomodals.helper.shell import package_outputs
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


class ScoreInput(BaseModel):
    """An internal candidate key and the complete, unaligned prepared domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=100, pattern="^[A-Za-z0-9_-]+$")
    sequence: str = Field(
        min_length=1, max_length=152, pattern="^[ACDEFGHIKLMNPQRSTVWY]+$"
    )


@cache
def prepare_runtime() -> None:
    """Verify the staged model bytes once per immutable inference container."""
    assert_assets(MODEL_ROOT)


def native_grid(sequence: str, identifier: str = "vhh") -> str:
    """Use native V2/VHH alignment, rejecting truncation or additional imputation."""
    from abnativ.model.alignment.mybio import (  # type: ignore[ty:unresolved-import]
        anarci_alignments_of_Fv_sequences_iter,
    )
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    heavy, kappa, light, _, _ = anarci_alignments_of_Fv_sequences_iter(
        [SeqRecord(Seq(sequence), id=identifier)],
        isVHH=True,
        verbose=False,
        run_parallel=False,
        check_AHo_CDR_gaps=True,
        del_cyst_misalign=False,
        check_terms=False,
    )
    records = heavy.to_recs()
    if len(records) != 1 or kappa.to_recs() or light.to_recs():
        raise ValueError("AbNatiV2 requires exactly one native-numbered VH domain")
    aligned = str(records[0].seq)
    if len(aligned) != 149 or aligned.replace("-", "") != sequence:
        raise ValueError("Native AbNatiV2 numbering changed the prepared VH")
    return aligned


def allowed_positions(parent: VHHInput, aligned: str) -> list[int]:
    """Map the frozen protection policy to native, one-based framework positions."""
    from abnativ.model.alignment.aho_consensus import (  # type: ignore[ty:unresolved-import]
        fr_aho_indices,
    )

    occupied = [index + 1 for index, residue in enumerate(aligned) if residue != "-"]
    protected = {occupied[index] for index in parent.protected_indices}
    return sorted(set(occupied).intersection(fr_aho_indices) - protected)


def _tables(name: str, frames: dict[str, pl.DataFrame]) -> AppOutput:
    with TemporaryDirectory(prefix="vhh-tables-") as directory:
        root = Path(directory) / name
        root.mkdir()
        for filename, frame in frames.items():
            frame.write_parquet(root / f"{filename}.parquet", compression="zstd")
        archive = package_outputs(root, num_threads=2)
    return AppOutput(
        name=name,
        kind=ArtifactKind.ARCHIVE,
        storage=InlineBytes(
            filename=f"{name}.tar.zst",
            media_type=ZSTD_MEDIA_TYPE,
            data=archive,
        ),
    )


def humanize_vhh(*, parent: dict[str, Any], settings: dict[str, Any]) -> AppRunResult:
    """Keep all accepted alternatives, or a successful unchanged parental endpoint."""
    prepared = VHHInput.model_validate(parent)
    parameters = HumanizationSettings.model_validate(settings)
    prepare_runtime()
    aligned = native_grid(prepared.sequence)
    mutable = allowed_positions(prepared, aligned)

    import matplotlib.pyplot as plt

    with (
        TemporaryDirectory(prefix="abnativ2-vhh-") as directory,
        ExitStack() as cleanup,
    ):
        cleanup.callback(plt.close, "all")
        candidates, search = generate(
            prepared.sequence, aligned, mutable, parameters, Path(directory)
        )
    attempts = []
    occupied = [i for i, aa in enumerate(aligned) if aa != "-"]
    for index, candidate in enumerate(candidates or [prepared.sequence], start=1):
        error = None
        try:
            prepared.validate_candidate(candidate)
            candidate_grid = native_grid(candidate)
            if [i for i, aa in enumerate(candidate_grid) if aa != "-"] != occupied:
                raise ValueError("AbNatiV2 endpoint changed native alignment occupancy")
        except ValueError as exc:
            error = str(exc)
        attempts.append({"attempt_index": index, "sequence": candidate, "error": error})
    report = {
        "schema_version": 2,
        "method": "abnativ2_vhh",
        "input_sequence": prepared.sequence,
        "input_aligned_sequence": aligned,
        "mutable_aho_positions": mutable,
        "settings": parameters.model_dump(),
        "source_commit": SOURCE_COMMIT,
        "runtime_identity": RUNTIME_IDENTITY,
        "search": search.model_dump() if search else None,
        "attempts": attempts,
    }
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="generation",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    filename="generation.json",
                    media_type="application/json",
                    data=orjson.dumps(report),
                ),
            ),
        ],
        metrics={
            "attempts": len(attempts),
            "valid_attempts": sum(row["error"] is None for row in attempts),
        },
    )


def score_vhh(
    *, sequences: list[dict[str, str]], model_type: Literal["VH2", "VHH2"]
) -> AppRunResult:
    """Score a bounded union chunk with one independent native model."""
    if model_type not in {"VH2", "VHH2"} or not 1 <= len(sequences) <= SCORE_BATCH_SIZE:
        raise ValueError(f"Expected VH2 or VHH2 and 1–{SCORE_BATCH_SIZE} candidates")
    inputs = [ScoreInput.model_validate(value) for value in sequences]
    if len({value.id for value in inputs}) != len(inputs):
        raise ValueError("Candidate identities must be unique")
    prepare_runtime()
    from abnativ.model.scoring_functions import (  # type: ignore[ty:unresolved-import]
        abnativ_scoring,
    )
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    records, rows = [], []
    for value in inputs:
        aligned, error = None, None
        try:
            aligned = native_grid(value.sequence, value.id)
            records.append(SeqRecord(Seq(aligned), id=value.id))
        except ValueError as exc:
            error = str(exc)
        rows.append({
            "candidate_id": value.id,
            "sequence": value.sequence,
            "aligned_sequence": aligned,
            "error": error,
        })
    summary = pl.DataFrame(
        rows,
        schema={
            "candidate_id": pl.String,
            "sequence": pl.String,
            "aligned_sequence": pl.String,
            "error": pl.String,
        },
    )
    details = {}
    if records:
        mean, profile = abnativ_scoring(
            model_type,
            records,
            batch_size=32,
            mean_score_only=False,
            do_align=False,
            is_VHH=True,
            verbose=False,
        )
        means, profiles = pl.from_pandas(mean), pl.from_pandas(profile)
        if means["seq_id"].to_list() != [record.id for record in records] or means[
            "aligned_seq"
        ].to_list() != [str(record.seq) for record in records]:
            raise ValueError("AbNatiV2 scores do not match their submitted candidates")
        values = means.select(
            pl.col("seq_id").alias("candidate_id"),
            pl.col(f"AbNatiV {model_type} Score").cast(pl.Float64).alias("score"),
        )
        summary = summary.join(values, on="candidate_id", how="left", validate="1:1")
        summary = summary.with_columns(
            pl.when(pl.col("score").is_finite()).then(pl.col("score")).alias("score"),
            pl
            .when(
                pl.col("error").is_null()
                & ~pl.col("score").is_finite().fill_null(False)
            )
            .then(pl.lit("Native model did not return a finite score"))
            .otherwise(pl.col("error"))
            .alias("error"),
        )
        details.update(native_scores=means, residue_scores=profiles)
    else:
        summary = summary.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[_tables("scores", {"summary": summary, **details})],
        metrics={"candidates": len(inputs), "scored": summary["score"].count()},
    )
