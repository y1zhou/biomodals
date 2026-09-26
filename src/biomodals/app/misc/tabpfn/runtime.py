"""Validated run-owned prediction publication with no fitted-model persistence."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict

from biomodals.app.misc.tabpfn.execution import TabPFNRequest
from biomodals.app.misc.tabpfn.models import RUNTIME_IDENTITY, native_regressor
from biomodals.app.misc.tabpfn.tables import (
    RegressionTables,
    fit_predict_tables,
    read_table_file,
)
from biomodals.helper.app_run import AppRunLayout, volume_app_output
from biomodals.helper.artifacts import (
    file_matches_sha256,
    read_bounded_file_bytes,
    sha256_file,
    write_bytes_atomic,
)
from biomodals.schema import AppRunResult, AppRunStatus, ArtifactFile, ArtifactKind


class PredictionPublication(BaseModel):
    """Bind completed predictions to their input and foundation-model identity."""

    model_config = ConfigDict(extra="forbid")
    request_sha256: str
    runtime_identity: str
    csv_sha256: str
    csv_bytes: int
    result: AppRunResult


def run_tabpfn(
    request_json: str,
    *,
    output_key: str,
    runtime_identity: str,
    volume_root: Path,
    volume_name: str,
    model_root: Path,
) -> AppRunResult:
    """Reloaded immutable inputs enter native fit only after trust-boundary checks."""
    if runtime_identity != RUNTIME_IDENTITY:
        raise ValueError("Target deployment changed TabPFN scientific versions")
    if re.fullmatch(r"[0-9a-f]{64}", output_key) is None:
        raise ValueError("Invalid run-owned output key")
    request = TabPFNRequest.model_validate_json(request_json)
    digest = sha256(
        orjson.dumps(request.scientific_identity(), option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    layout = AppRunLayout.from_run_root(volume_root / "runs" / output_key)
    marker = layout.markers_dir / "result.json"
    destination = layout.outputs_dir / "predictions.csv"
    if marker.exists():
        saved = PredictionPublication.model_validate_json(
            read_bounded_file_bytes(
                marker, field_name="prediction publication", max_bytes=1024 * 1024
            )
        )
        if saved.request_sha256 != digest or saved.runtime_identity != runtime_identity:
            raise ValueError("Prediction namespace belongs to a different request")
        if not file_matches_sha256(destination, saved.csv_bytes, saved.csv_sha256):
            raise ValueError("Published predictions failed integrity validation")
        return saved.result
    tables = RegressionTables(
        request.table_schema,
        read_table_file(
            request.training.resolve(volume_root), request.table_schema, training=True
        ),
        read_table_file(
            request.inference.resolve(volume_root), request.table_schema, training=False
        ),
    )
    estimator = native_regressor(
        model_root,
        seed=request.seed,
        n_estimators=request.n_estimators,
        categorical_indices=tables.categorical_indices,
    )
    predicted = fit_predict_tables(tables, estimator, batch_size=request.batch_size)
    layout.prep_dir.mkdir(parents=True, exist_ok=True)
    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    temporary = layout.prep_dir / "predictions.csv"
    predicted.write_csv(temporary)
    temporary.replace(destination)
    csv_digest, size = sha256_file(destination), destination.stat().st_size
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="predictions",
                kind=ArtifactKind.TABLE,
                remote_path=str(destination),
                mount_root=str(volume_root),
                volume_name=volume_name,
                media_type="text/csv",
                files=[
                    ArtifactFile(
                        path=destination.name,
                        size_bytes=size,
                        content_sha256=csv_digest,
                    )
                ],
                metadata={
                    "request_sha256": digest,
                    "runtime_identity": runtime_identity,
                    "feature_schema": request.table_schema.model_dump(mode="json"),
                    "training_rows": tables.training.height,
                    "prediction_rows": predicted.height,
                },
            )
        ],
    )
    publication = PredictionPublication(
        request_sha256=digest,
        runtime_identity=runtime_identity,
        csv_sha256=csv_digest,
        csv_bytes=size,
        result=result,
    )
    write_bytes_atomic(marker, publication.model_dump_json().encode())
    return result
