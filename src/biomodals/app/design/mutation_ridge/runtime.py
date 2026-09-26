"""Run-owned CSV publication; native interruption may safely re-enter it."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from biomodals.app.design.mutation_ridge.execution import RUNTIME_IDENTITY, RidgeRequest
from biomodals.app.design.mutation_ridge.inputs import build_dataset, combination_space
from biomodals.app.design.mutation_ridge.regression import write_combinations
from biomodals.helper.app_run import AppRunLayout, volume_app_output
from biomodals.helper.artifacts import (
    file_matches_sha256,
    read_bounded_file_bytes,
    sha256_file,
    write_bytes_atomic,
)
from biomodals.schema import AppRunResult, AppRunStatus, ArtifactFile, ArtifactKind


class RidgePublication(BaseModel):
    """Content-bound final publication, separate from any fitted estimator."""

    model_config = ConfigDict(extra="forbid")
    request_sha256: str
    runtime_identity: str
    csv_sha256: str
    csv_bytes: int
    result: AppRunResult


def run_ridge(
    request_json: str,
    *,
    output_key: str,
    runtime_identity: str,
    volume_root: Path,
    volume_name: str,
) -> AppRunResult:
    """Fit once per owned task and publish only a completed, validated CSV."""
    if runtime_identity != RUNTIME_IDENTITY:
        raise ValueError("Target deployment changed mutation-ridge scientific versions")
    if re.fullmatch(r"[0-9a-f]{64}", output_key) is None:
        raise ValueError("Invalid run-owned output key")
    request = RidgeRequest.model_validate_json(request_json)
    digest = sha256(request.model_dump_json().encode()).hexdigest()
    layout = AppRunLayout.from_run_root(volume_root / "runs" / output_key)
    marker = layout.markers_dir / "result.json"
    csv_path = layout.outputs_dir / "candidates.csv"
    if marker.exists():
        published = RidgePublication.model_validate_json(
            read_bounded_file_bytes(
                marker, field_name="ridge publication", max_bytes=1024 * 1024
            )
        )
        if (
            published.request_sha256 != digest
            or published.runtime_identity != runtime_identity
        ):
            raise ValueError(
                "Output namespace belongs to a different scientific request"
            )
        if not file_matches_sha256(
            csv_path,
            expected_size=published.csv_bytes,
            expected_digest=published.csv_sha256,
        ):
            raise ValueError("Published ridge candidates failed integrity validation")
        return published.result
    dataset = build_dataset(request.measurements_csv.encode(), request.parental_fasta)
    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    # One kernel-owned writer; redelivery overwrites this same partial file.
    # Large sort scratch stays container-local, not in accumulating Volume dirs.
    temporary_csv = layout.prep_dir / "candidates.csv"
    summary = write_combinations(
        dataset,
        temporary_csv,
        max_mutations=request.max_mutations,
        budget=request.candidate_budget,
        alpha=request.alpha,
        higher_is_better=request.higher_is_better,
        seed=request.seed,
    )
    temporary_csv.replace(csv_path)
    csv_digest, size = sha256_file(csv_path), csv_path.stat().st_size
    count = sum(combination_space(dataset, request.max_mutations)[1])
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="candidates",
                kind=ArtifactKind.TABLE,
                remote_path=str(csv_path),
                mount_root=str(volume_root),
                volume_name=volume_name,
                media_type="text/csv",
                files=[
                    ArtifactFile(
                        path=csv_path.name, size_bytes=size, content_sha256=csv_digest
                    )
                ],
                metadata={
                    "validation": summary.model_dump(mode="json"),
                    "candidate_count": count,
                    "chain_columns": {
                        chain: f"sequence_{chain}" for chain in dataset.parents
                    },
                    "request_sha256": digest,
                    "runtime_identity": runtime_identity,
                },
            )
        ],
        metrics={"candidate_count": count, "training_variants": len(dataset.variants)},
    )
    publication = RidgePublication(
        request_sha256=digest,
        runtime_identity=runtime_identity,
        csv_sha256=csv_digest,
        csv_bytes=size,
        result=result,
    )
    write_bytes_atomic(marker, publication.model_dump_json().encode())
    return result
