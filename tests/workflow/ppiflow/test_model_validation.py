"""Tests for PPIFlow checkpoint validation."""

# ruff: noqa: D103

import hashlib
from pathlib import Path

import orjson
import pytest

from biomodals.workflow.ppiflow.model_validation import validate_ppiflow_model


def test_validate_ppiflow_model_binds_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "binder.ckpt"
    checkpoint.write_bytes(b"model-bytes")
    expected = hashlib.sha256(b"model-bytes").hexdigest()

    result = validate_ppiflow_model(
        model_path=str(checkpoint),
        expected_sha256=expected,
    )

    assert orjson.loads(result.outputs[0].storage.data) == {
        "model_name": "binder.ckpt",
        "sha256": expected,
    }


def test_validate_ppiflow_model_rejects_changed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "binder.ckpt"
    checkpoint.write_bytes(b"changed")

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_ppiflow_model(
            model_path=str(checkpoint),
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
        )
