"""Tests for FlowPacker workflow-compatible result contracts."""

# ruff: noqa: D103

from biomodals.app.fold.flowpacker_app import _flowpacker_app_run_result
from biomodals.schema import AppRunStatus, ArtifactKind, InlineBytes


def test_flowpacker_workflow_result_wraps_tarball_bytes() -> None:
    result = _flowpacker_app_run_result(
        run_name="packed",
        tarball_bytes=b"tarball",
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "flowpacker_outputs"
    assert output.kind == ArtifactKind.STRUCTURES
    assert isinstance(output.storage, InlineBytes)
    assert output.storage.data == b"tarball"
    assert output.storage.filename == "packed.tar.zst"
    assert output.storage.archive_format == "tar.zst"
