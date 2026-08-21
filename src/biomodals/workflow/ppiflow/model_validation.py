"""PPIFlow checkpoint validation for workflow deployments."""

from pathlib import Path

import orjson

from biomodals.helper.artifacts import sha256_file
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)


def validate_ppiflow_model(
    *,
    model_path: str,
    expected_sha256: str,
) -> AppRunResult:
    """Hash one mounted checkpoint and reject unexpected model bytes."""
    reload_ppiflow_model_volume()
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"PPIFlow model checkpoint not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PPIFlow model digest mismatch for {path.name}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="ppiflow_model_validation",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=orjson.dumps({
                        "model_name": path.name,
                        "sha256": actual_sha256,
                    }),
                    media_type="application/json",
                    filename="ppiflow-model-validation.json",
                ),
            )
        ],
    )


def reload_ppiflow_model_volume() -> None:
    """Refresh the shared model Volume before validation or inference."""
    MODEL_VOLUME.reload()
