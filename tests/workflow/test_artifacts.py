"""Tests for local workflow artifact materialization."""

# ruff: noqa: D103

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
        volume_root=tmp_path,
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
        path="nodes/summary/attempts/1/materialized_outputs/summary-summary",
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


def test_materialize_inline_bytes_preserves_output_metadata(
    tmp_path: Path,
) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="summary",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(data=b"ok\n", filename="summary.txt"),
                metadata={"stage": "stage1"},
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="summary",
        volume_root=tmp_path,
    )

    assert artifacts[0].metadata == {"stage": "stage1"}


def test_materialize_app_run_result_persists_log_outputs_under_attempt_logs(
    tmp_path: Path,
) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        logs=[
            AppOutput(
                name="stderr",
                kind=ArtifactKind.LOGS,
                storage=InlineBytes(data=b"warning\n", filename="stderr.log"),
                metadata={"stream": "stderr"},
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="node",
        volume_root=tmp_path,
    )

    log_path = tmp_path / "attempt" / "logs" / "node-logs-stderr" / "stderr.log"
    assert log_path.read_bytes() == b"warning\n"
    assert artifacts[0].kind == ArtifactKind.LOGS
    assert artifacts[0].source_app_output_name == "stderr"
    assert artifacts[0].metadata == {"stream": "stderr"}
    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="attempt/logs/node-logs-stderr",
    )
    assert (tmp_path / "artifacts" / "node-logs-stderr.json").exists()


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


def test_materialize_volume_path_can_copy_from_mounted_volume(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-volume"
    source_dir = source_root / "runs" / "run-1"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("scores.csv").write_text("score\n1\n", encoding="utf-8")
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(
                    volume_name="AF3Score-outputs",
                    path="runs/run-1",
                ),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "workflow" / "attempt",
        artifact_dir=tmp_path / "workflow" / "artifacts",
        producing_node_id="score",
        volume_root=tmp_path / "workflow",
        volume_path_mode="copy",
        volume_roots={"AF3Score-outputs": source_root},
    )

    copied_file = (
        tmp_path
        / "workflow"
        / "attempt"
        / "materialized_outputs"
        / "score-scores"
        / "scores.csv"
    )
    assert copied_file.read_text(encoding="utf-8") == "score\n1\n"
    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="attempt/materialized_outputs/score-scores",
    )
    assert artifacts[0].files[0].path == "scores.csv"


def test_materialize_volume_path_copy_preserves_empty_directories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-volume"
    source_dir = source_root / "runs" / "run-1"
    source_dir.mkdir(parents=True)
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(
                    volume_name="AF3Score-outputs",
                    path="runs/run-1",
                ),
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "workflow" / "attempt",
        artifact_dir=tmp_path / "workflow" / "artifacts",
        producing_node_id="score",
        volume_root=tmp_path / "workflow",
        volume_path_mode="copy",
        volume_roots={"AF3Score-outputs": source_root},
    )

    materialized_dir = (
        tmp_path / "workflow" / "attempt" / "materialized_outputs" / "score-scores"
    )
    assert materialized_dir.is_dir()
    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="attempt/materialized_outputs/score-scores",
    )


def test_materialize_volume_path_copy_rejects_traversal(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-volume"
    source_root.mkdir()
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath.model_construct(
                    kind="volume_path",
                    volume_name="AF3Score-outputs",
                    path="../secret.csv",
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="relative"):
        materialize_app_run_result(
            result=result,
            workflow_volume_name="Workflow-outputs",
            attempt_dir=tmp_path / "workflow" / "attempt",
            artifact_dir=tmp_path / "workflow" / "artifacts",
            producing_node_id="score",
            volume_root=tmp_path / "workflow",
            volume_path_mode="copy",
            volume_roots={"AF3Score-outputs": source_root},
        )


def test_materialize_volume_path_copy_rejects_symlinked_children(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-volume"
    source_dir = source_root / "runs" / "run-1"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("scores.csv").write_text("score\n1\n", encoding="utf-8")
    secret = tmp_path / "secret.csv"
    secret.write_text("secret\n", encoding="utf-8")
    source_dir.joinpath("secret-link.csv").symlink_to(secret)
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(
                    volume_name="AF3Score-outputs",
                    path="runs/run-1",
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="symlink"):
        materialize_app_run_result(
            result=result,
            workflow_volume_name="Workflow-outputs",
            attempt_dir=tmp_path / "workflow" / "attempt",
            artifact_dir=tmp_path / "workflow" / "artifacts",
            producing_node_id="score",
            volume_root=tmp_path / "workflow",
            volume_path_mode="copy",
            volume_roots={"AF3Score-outputs": source_root},
        )


def test_materialize_volume_path_copy_rejects_symlink_path_component(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-volume"
    real_dir = source_root / "real-run"
    real_dir.mkdir(parents=True)
    real_dir.joinpath("scores.csv").write_text("score\n1\n", encoding="utf-8")
    source_root.joinpath("linked-run").symlink_to(real_dir, target_is_directory=True)
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="scores",
                kind=ArtifactKind.SCORES,
                storage=VolumePath(
                    volume_name="AF3Score-outputs",
                    path="linked-run",
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="symlinks"):
        materialize_app_run_result(
            result=result,
            workflow_volume_name="Workflow-outputs",
            attempt_dir=tmp_path / "workflow" / "attempt",
            artifact_dir=tmp_path / "workflow" / "artifacts",
            producing_node_id="score",
            volume_root=tmp_path / "workflow",
            volume_path_mode="copy",
            volume_roots={"AF3Score-outputs": source_root},
        )


def test_materialize_inline_bytes_rejects_non_utf8_bytes(
    tmp_path: Path,
) -> None:
    result = AppRunResult.model_construct(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput.model_construct(
                name="archive",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes.model_construct(
                    data=b"\xff\x00",
                    filename="archive.tar.zst",
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="UTF-8 text"):
        materialize_app_run_result(
            result=result,
            workflow_volume_name="Workflow-outputs",
            attempt_dir=tmp_path / "attempt",
            artifact_dir=tmp_path / "artifacts",
            producing_node_id="pack",
        )


def test_materialize_inline_zstd_archive_preserves_binary_bytes(
    tmp_path: Path,
) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="archive",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=b"\xff\x00",
                    filename="archive.tar.zst",
                    media_type="application/zstd",
                ),
                metadata={"archive_format": "tar.zst"},
            )
        ],
    )

    artifacts = materialize_app_run_result(
        result=result,
        workflow_volume_name="Workflow-outputs",
        attempt_dir=tmp_path / "attempt",
        artifact_dir=tmp_path / "artifacts",
        producing_node_id="pack",
        volume_root=tmp_path,
    )

    raw_path = tmp_path / "attempt" / "raw_outputs" / "archive.tar.zst"
    materialized_path = (
        tmp_path
        / "attempt"
        / "materialized_outputs"
        / "pack-archive"
        / "archive.tar.zst"
    )
    assert raw_path.read_bytes() == b"\xff\x00"
    assert materialized_path.read_bytes() == b"\xff\x00"
    assert artifacts[0].kind == ArtifactKind.ARCHIVE
    assert artifacts[0].storage == VolumePath(
        volume_name="Workflow-outputs",
        path="attempt/materialized_outputs/pack-archive",
        media_type="application/zstd",
    )
    assert artifacts[0].metadata == {"archive_format": "tar.zst"}


def test_archive_outputs_use_volume_path_metadata(tmp_path: Path) -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="models",
                kind=ArtifactKind.ARCHIVE,
                storage=VolumePath(
                    volume_name="FlowPacker-outputs",
                    path="workflow/packed/packed.tar.zst",
                    media_type="application/zstd",
                ),
                metadata={"archive_format": "tar.zst"},
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

    assert artifacts[0].storage == VolumePath(
        volume_name="FlowPacker-outputs",
        path="workflow/packed/packed.tar.zst",
        media_type="application/zstd",
    )
    assert artifacts[0].metadata == {"archive_format": "tar.zst"}
