"""Tests for PPIFlow coordinator helpers."""

# ruff: noqa: D103

from pathlib import Path
from threading import Lock
from time import sleep

import orjson
import polars as pl
import pytest

from biomodals.schema import AppOutput, AppRunStatus, ArtifactKind, VolumePath
from biomodals.workflow.ppiflow import coordinators, manifests


def test_candidate_concurrency_from_config_defaults_and_validates() -> None:
    assert coordinators.candidate_concurrency_from_config({}) == 4
    assert (
        coordinators.candidate_concurrency_from_config(
            {"candidate_concurrency": 2},
            {"candidate_concurrency": 8},
        )
        == 2
    )
    with pytest.raises(ValueError, match="at least 1"):
        coordinators.candidate_concurrency_from_config({"candidate_concurrency": 0})


def test_run_candidate_tasks_respects_concurrency_limit() -> None:
    lock = Lock()
    active = 0
    max_active = 0

    def submit(task: coordinators.CandidateTask) -> coordinators.CandidateOutcome:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        sleep(0.01)
        with lock:
            active -= 1
        return coordinators.CandidateOutcome(
            candidate_id=task.candidate_id,
            status=AppRunStatus.SUCCEEDED,
        )

    outcomes = coordinators.run_candidate_tasks(
        [coordinators.CandidateTask(str(index)) for index in range(5)],
        submit,
        candidate_concurrency=2,
    )

    assert max_active <= 2
    assert [outcome.candidate_id for outcome in outcomes] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


def test_status_from_candidate_outcomes() -> None:
    assert (
        coordinators.status_from_candidate_outcomes([
            coordinators.CandidateOutcome("a", AppRunStatus.SUCCEEDED),
            coordinators.CandidateOutcome("b", AppRunStatus.SUCCEEDED),
        ])
        == AppRunStatus.SUCCEEDED
    )
    assert (
        coordinators.status_from_candidate_outcomes([
            coordinators.CandidateOutcome("a", AppRunStatus.SUCCEEDED),
            coordinators.CandidateOutcome("b", AppRunStatus.FAILED),
        ])
        == AppRunStatus.PARTIAL
    )
    assert (
        coordinators.status_from_candidate_outcomes([
            coordinators.CandidateOutcome("a", AppRunStatus.FAILED)
        ])
        == AppRunStatus.FAILED
    )


def test_pending_tasks_skip_only_reusable_completed_candidates(tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    app_root.mkdir()
    (app_root / "a.pdb").write_text("ATOM\n", encoding="utf-8")
    frame = pl.DataFrame([
        manifests.candidate_manifest_row(
            candidate_id="a",
            stage_name="Stage",
            stage_role="test",
            operation_mode="test",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            files=[
                manifests.candidate_file_record(
                    role="structure",
                    volume_name="app-volume",
                    app_volume_path="a.pdb",
                )
            ],
        ),
        manifests.candidate_manifest_row(
            candidate_id="b",
            stage_name="Stage",
            stage_role="test",
            operation_mode="test",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            files=[
                manifests.candidate_file_record(
                    role="structure",
                    volume_name="app-volume",
                    app_volume_path="missing.pdb",
                )
            ],
        ),
    ])

    reusable = manifests.reusable_completed_candidate_ids(
        frame,
        volume_roots={"app-volume": app_root},
        workflow_volume_name="workflow-volume",
    )
    tasks = coordinators.pending_candidate_tasks(
        frame,
        reusable_candidate_ids=reusable,
    )

    assert reusable == {"a"}
    assert [task.candidate_id for task in tasks] == ["b"]


def test_outcome_rows_and_manifest_merge() -> None:
    reusable = pl.DataFrame([
        manifests.candidate_manifest_row(
            candidate_id="a",
            stage_name="Stage",
            stage_role="test",
            operation_mode="reuse",
            candidate_status=AppRunStatus.SUCCEEDED.value,
            files=[],
        )
    ])
    outcome_rows = coordinators.outcome_manifest_rows(
        stage_name="Stage",
        stage_role="test",
        operation_mode="run",
        outcomes=[
            coordinators.CandidateOutcome(
                "b",
                AppRunStatus.FAILED,
                outputs={"score": 1.0},
                error="boom",
            )
        ],
    )

    merged = coordinators.merge_candidate_manifest_rows(reusable, outcome_rows)

    assert [row["candidate_id"] for row in merged] == ["a", "b"]
    assert merged[1]["candidate_status"] == "failed"
    assert merged[1]["error"] == "boom"


def test_outcome_rows_accept_app_outputs_in_summary() -> None:
    rows = coordinators.outcome_manifest_rows(
        stage_name="Stage",
        stage_role="sequence_design",
        operation_mode="abmpnn",
        outcomes=[
            coordinators.CandidateOutcome(
                "candidate-1",
                AppRunStatus.SUCCEEDED,
                outputs={
                    "app_outputs": [
                        AppOutput(
                            name="structures",
                            kind=ArtifactKind.STRUCTURES,
                            storage=VolumePath(
                                volume_name="PPIFlow-outputs",
                                path="candidate-1",
                            ),
                        )
                    ]
                },
            )
        ],
    )

    summary = orjson.loads(rows[0]["summary_json"])

    assert summary["app_outputs"][0]["name"] == "structures"
