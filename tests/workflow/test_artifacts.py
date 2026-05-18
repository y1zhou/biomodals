"""Tests for local workflow artifact materialization."""

# ruff: noqa: D103,S603,S607

import shutil
import subprocess
from pathlib import Path

import pytest

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.core.artifacts import materialize_app_run_result


def test_materialize_inline_bytes_writes_raw_and_volume_artifact(
    tmp_path: Path,
) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="summary",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(data=b"ok\n", filename="summary.txt"),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "nodes" / "summary" / "attempts" / "1",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="summary",
    )

    raw_path = tmp_path / "nodes" / "summary" / "attempts" / "1" / "raw_outputs"
    materialized_path = (
        tmp_path
        / "nodes"
        / "summary"
        / "attempts"
        / "1"
        / "materialized_outputs"
        / "summary-summary"
        / "summary.txt"
    )
    assert raw_path.joinpath("summary.txt").read_bytes() == b"ok\n"
    assert materialized_path.read_bytes() == b"ok\n"
    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path=str(materialized_path.parent),
    )
    assert artifacts[0].files[0].path == "summary.txt"
    assert (tmp_path / "artifacts" / "summary-summary.json").exists()


def test_materialized_inline_artifact_path_is_volume_relative(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "demo" / "run-1"
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="summary",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(data=b"ok\n", filename="summary.txt"),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=run_root / "nodes" / "summary" / "attempts" / "attempt-1",
        artifact_dir=run_root / "artifacts",
        producing_node_id="summary",
        volume_root=tmp_path,
    )

    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path=(
            "demo/run-1/nodes/summary/attempts/attempt-1/"
            "materialized_outputs/summary-summary"
        ),
    )


def test_materialize_volume_path_references_existing_remote_output(
    tmp_path: Path,
) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(
                    volume_name="AF3Score-outputs",
                    path="run-1/af3score_metrics.csv",
                ),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="score",
    )

    assert artifacts[0].storage == VolumePath(
        volume_name="AF3Score-outputs",
        path="run-1/af3score_metrics.csv",
    )
    assert (tmp_path / "artifacts" / "score-scores.json").exists()


def test_materialize_tar_zst_preserves_archive_and_extracts_files(
    tmp_path: Path,
) -> None:
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        pytest.skip("tar and zstd are required for tar.zst extraction")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_dir.joinpath("model.pdb").write_text("ATOM\n", encoding="utf-8")
    archive_path = tmp_path / "outputs.tar.zst"
    subprocess.run(
        ["tar", "-I", "zstd", "-cf", str(archive_path), "-C", str(source_dir), "."],
        check=True,
    )

    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="models",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=archive_path.read_bytes(),
                    filename="outputs.tar.zst",
                    archive_format="tar.zst",
                ),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="pack",
    )

    assert (tmp_path / "attempt" / "raw_outputs" / "outputs.tar.zst").exists()
    assert (
        tmp_path / "attempt" / "materialized_outputs" / "pack-models" / "model.pdb"
    ).read_text(encoding="utf-8") == "ATOM\n"
    assert artifacts[0].files[0].path == "model.pdb"
