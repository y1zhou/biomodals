"""Tests for PPIFlow manifest helpers."""

# ruff: noqa: D103

from pathlib import Path

import polars as pl
import pytest

from biomodals.schema import AppRunStatus, ArtifactKind, VolumePath
from biomodals.workflow.ppiflow import manifests


def _manifest_row(candidate_id: str = "candidate-1") -> dict[str, object]:
    return manifests.candidate_manifest_row(
        candidate_id=candidate_id,
        stage_name="Stage2Input",
        stage_role="stage2_input",
        operation_mode="existing_structures",
        candidate_status=AppRunStatus.SUCCEEDED.value,
        source_artifact_id="source-artifact",
        source_path="inputs/model.pdb",
        derived_path="inputs/model.pdb",
        files=[
            manifests.candidate_file_record(
                role="structure",
                volume_name="source-volume",
                app_volume_path="inputs/model.pdb",
                path="model.pdb",
                media_type="chemical/x-pdb",
                size_bytes=5,
            )
        ],
    )


def test_candidate_ids_are_deterministic_and_provenance_sensitive() -> None:
    first = manifests.initial_candidate_id(
        stage_name="PPIFlowStep",
        source_artifact_id="artifact-a",
        source_path="outputs/model.pdb",
    )
    repeated = manifests.initial_candidate_id(
        stage_name="PPIFlowStep",
        source_artifact_id="artifact-a",
        source_path="outputs/model.pdb",
    )
    changed_source = manifests.initial_candidate_id(
        stage_name="PPIFlowStep",
        source_artifact_id="artifact-a",
        source_path="outputs/other.pdb",
    )

    assert first == repeated
    assert first != changed_source
    assert first.startswith("cand_")


def test_derived_and_stage2_candidate_ids() -> None:
    first = manifests.derived_candidate_id(
        parent_candidate_id="candidate-a",
        stage_name="LigandMPNN",
        operation_mode="binder",
        derived_basename="design_1.pdb",
    )
    changed_mode = manifests.derived_candidate_id(
        parent_candidate_id="candidate-a",
        stage_name="LigandMPNN",
        operation_mode="abmpnn",
        derived_basename="design_1.pdb",
    )

    assert first != changed_mode
    assert manifests.stage2_input_candidate_id(1) == "stage2_input_000001"
    with pytest.raises(ValueError, match="1-based"):
        manifests.stage2_input_candidate_id(0)


def test_manifest_round_trips_nested_files_as_parquet(tmp_path: Path) -> None:
    path = manifests.write_manifest([_manifest_row()], tmp_path / "manifest.parquet")

    frame = manifests.read_manifest(path)

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["candidate_id"] == "candidate-1"
    assert row["files"][0]["role"] == "structure"
    assert row["files"][0]["app_volume_path"] == "inputs/model.pdb"


def test_manifest_output_is_table_artifact_with_parquet_file(
    tmp_path: Path,
) -> None:
    manifest_path = manifests.write_manifest(
        [_manifest_row()],
        tmp_path / "candidate_manifest.parquet",
    )

    output = manifests.manifest_artifact_output(
        manifest_path=manifest_path,
        mount_root=str(tmp_path),
        volume_name="workflow-volume",
        stage_name="Stage2Input",
        row_count=1,
    )

    assert output.name == "candidate_manifest"
    assert output.kind == ArtifactKind.TABLE
    assert output.storage == VolumePath(
        volume_name="workflow-volume",
        path="candidate_manifest.parquet",
        media_type=manifests.MANIFEST_MEDIA_TYPE,
    )
    assert output.metadata["files"][0]["path"] == "candidate_manifest.parquet"
    assert output.metadata["files"][0]["role"] == "candidate_manifest"


def test_strict_candidate_join_requires_all_candidate_ids() -> None:
    required = pl.DataFrame({"candidate_id": ["a", "b"], "left": [1, 2]})
    available = pl.DataFrame({"candidate_id": ["a"], "right": [3]})

    with pytest.raises(ValueError, match="Missing required candidate ids"):
        manifests.strict_candidate_join(required, available)

    joined = manifests.strict_candidate_join(
        required,
        available,
        allow_missing_candidates=True,
    )
    assert joined.get_column("candidate_id").to_list() == ["a"]


def test_expected_file_errors_check_workflow_and_app_volume_paths(
    tmp_path: Path,
) -> None:
    workflow_root = tmp_path / "workflow"
    app_root = tmp_path / "app"
    workflow_root.mkdir()
    app_root.mkdir()
    (workflow_root / "nodes" / "manifest.parquet").parent.mkdir()
    (workflow_root / "nodes" / "manifest.parquet").write_text("ok", encoding="utf-8")
    (app_root / "outputs").mkdir()
    (app_root / "outputs" / "model.pdb").write_text("ATOM\n", encoding="utf-8")
    rows = [
        manifests.candidate_manifest_row(
            candidate_id="candidate-a",
            stage_name="Stage",
            stage_role="test",
            operation_mode="test",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            files=[
                manifests.candidate_file_record(
                    role="manifest",
                    workflow_path="nodes/manifest.parquet",
                ),
                manifests.candidate_file_record(
                    role="structure",
                    volume_name="app-volume",
                    app_volume_path="outputs/model.pdb",
                    size_bytes=5,
                ),
            ],
        )
    ]

    assert (
        manifests.expected_file_errors(
            rows,
            volume_roots={
                "workflow-volume": workflow_root,
                "app-volume": app_root,
            },
            workflow_volume_name="workflow-volume",
        )
        == []
    )

    (app_root / "outputs" / "model.pdb").unlink()
    assert manifests.expected_file_errors(
        rows,
        volume_roots={
            "workflow-volume": workflow_root,
            "app-volume": app_root,
        },
        workflow_volume_name="workflow-volume",
    ) == ["candidate-a: missing expected file app-volume:outputs/model.pdb"]


def test_reusable_completed_candidates_require_expected_files(
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "model.pdb").write_text("ATOM\n", encoding="utf-8")
    rows = [
        manifests.candidate_manifest_row(
            candidate_id="complete",
            stage_name="Stage",
            stage_role="test",
            operation_mode="test",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            files=[
                manifests.candidate_file_record(
                    role="structure",
                    volume_name="app-volume",
                    app_volume_path="model.pdb",
                )
            ],
        ),
        manifests.candidate_manifest_row(
            candidate_id="failed",
            stage_name="Stage",
            stage_role="test",
            operation_mode="test",
            candidate_status=AppRunStatus.FAILED.value,
            files=[],
        ),
    ]

    assert manifests.reusable_completed_candidate_ids(
        rows,
        volume_roots={"app-volume": app_root},
        workflow_volume_name="workflow-volume",
    ) == {"complete"}

    (app_root / "model.pdb").unlink()
    assert (
        manifests.reusable_completed_candidate_ids(
            rows,
            volume_roots={"app-volume": app_root},
            workflow_volume_name="workflow-volume",
        )
        == set()
    )
