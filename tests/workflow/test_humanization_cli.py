"""Submission contracts: dry runs never contact Modal, limits stay global."""

import pytest

from biomodals.helper.catalog import BiomodalsApp, get_catalog
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.workflow.humanization import workflow


def test_cli_dry_run_validates_native_controls_without_staging(
    tmp_path, monkeypatch, capsys
):
    """Tune method controls and display the graph without any remote launch."""
    source = tmp_path / "pairs.csv"
    source.write_bytes(b"id,vh,vl\na,ACD,EFG\n")
    monkeypatch.setattr(
        workflow,
        "stage_execution_launch",
        lambda *args: pytest.fail("Dry run staged remote work"),
    )
    raw = workflow.submit_humanization_workflow.info.raw_f
    raw(
        input_csv=str(source),
        dry_run=True,
        max_containers=2,
        max_gpu_containers=1,
        hudiff_ab_candidate_count=2,
        humatch_max_edits=5,
    )
    assert "HumanizationEvaluateNode" in capsys.readouterr().out
    with pytest.raises(ValueError, match="vh_target_family"):
        raw(
            input_csv=str(source), dry_run=True, humatch_vh_target_family="not-a-family"
        )
    with pytest.raises(ValueError):
        raw(input_csv=str(source), dry_run=True, max_containers=1, max_gpu_containers=2)


def test_composition_includes_all_generators_and_scorers():
    """Included operations have unique tags and do not include child coordinators."""
    assert workflow.CONF.depends_on_apps == (
        "sapiens",
        "humatch",
        "pabnativ2",
        "hudiff_ab",
    )
    functions = workflow.app._local_state.functions
    for name in (
        "sapiens_humanize",
        "humatch_humanize",
        "pabnativ2_humanize_pair",
        "hudiff_ab_humanize_pair",
        "sapiens_score",
        "humatch_score",
        "pabnativ2_score",
        "annotate_humanization_candidate",
    ):
        assert name in functions
    assert list(workflow.app._local_state.classes) == ["ExecutionCoordinator"]


def test_package_metadata_and_waited_result_locations(tmp_path, monkeypatch, capsys):
    """Package launch keeps canonical imports and exposes durable result locations."""
    from types import SimpleNamespace

    catalog = get_catalog("workflow", use_absolute_paths=True)
    for reference in ("humanization", str(catalog["humanization"])):
        entry = BiomodalsApp(reference, all_apps=catalog)
        assert entry.name == "humanization"
        assert entry.category == "workflow"
        assert entry.module == "biomodals.workflow.humanization.workflow"
    source = tmp_path / "pairs.csv"
    source.write_bytes(b"id,vh,vl\na,ACD,EFG\n")
    publication = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="humanization_results",
                kind=ArtifactKind.DIRECTORY,
                storage=VolumePath(volume_name="test-results", path="run/humanization"),
            )
        ],
    )
    monkeypatch.setattr(workflow, "stage_execution_launch", lambda *args: None)
    monkeypatch.setattr(
        workflow.orchestrator, "execution_coordinator_handle", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        workflow.orchestrator,
        "submit_workflow_run",
        lambda *args, **kwargs: SimpleNamespace(get=lambda: publication),
    )
    workflow.submit_humanization_workflow.info.raw_f(
        input_csv=str(source),
        use_deployed_coordinator=True,
    )
    output = capsys.readouterr().out
    assert "Humanization completed: succeeded" in output
    assert "humanization_results" in output
    assert "test-results" in output
    assert "run/humanization" in output
