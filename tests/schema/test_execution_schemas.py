"""Tests for shared workflow schema contracts."""

# ruff: noqa: D103

from typing import Any

import pytest
from pydantic import ValidationError

from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


def _valid_app_config(**overrides: object) -> AppConfig:
    values: dict[str, Any] = {
        "name": "demo",
        "package_name": "demo-package",
        "version": "1.0.0",
    }
    values.update(overrides)
    return AppConfig(**values)


def test_app_config_validates_source_reproducibility_and_runtime_bounds() -> None:
    with pytest.raises(ValidationError, match="repo_url"):
        AppConfig(name="missing-source", version="1.0.0")

    with pytest.raises(ValidationError, match="repo_commit_hash"):
        AppConfig(name="missing-version", package_name="demo-package")

    with pytest.raises(ValidationError, match="CUDA version must start"):
        _valid_app_config(cuda_version="12.8")

    with pytest.raises(ValidationError, match="CUDA 12.x"):
        _valid_app_config(gpu="B200+", cuda_version="cu128")

    with pytest.raises(ValidationError, match="Timeout must be between"):
        _valid_app_config(timeout=0)

    with pytest.raises(ValidationError, match="Timeout must be between"):
        _valid_app_config(timeout=999_999)


def test_inline_bytes_allows_zstd_binary_data_round_trip() -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="packed",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=b"\xff\x00",
                    filename="packed.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
            )
        ],
    )

    dumped = result.model_dump_json()
    loaded = AppRunResult.model_validate_json(dumped)

    assert "_wA=" in dumped
    assert isinstance(loaded.outputs[0].storage, InlineBytes)
    assert loaded.outputs[0].storage.data == b"\xff\x00"
    assert loaded.outputs[0].storage.media_type == ZSTD_MEDIA_TYPE


def test_volume_path_rejects_absolute_and_traversal_paths() -> None:
    for unsafe_path in ("/absolute/out", "../out", "a/../out", r"a\b"):
        with pytest.raises(ValidationError, match="VolumePath.path"):
            VolumePath(volume_name="Workflow-outputs", path=unsafe_path)
