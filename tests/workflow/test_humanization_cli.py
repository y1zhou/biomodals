"""Submission contracts: dry runs never contact Modal, limits stay global."""

import pytest

from biomodals.execution import RunStatus
from biomodals.helper.catalog import BiomodalsApp, get_catalog
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.workflow.humanization import workflow


@pytest.mark.parametrize("method", workflow.METHODS)
def test_workflow_calls_public_app_validator(monkeypatch, method):
    """Any workflow can reuse the same app-owned parameter boundary."""
    app = getattr(workflow, f"{method if method != 'hudiff_ab' else 'hudiff'}_app")
    name = "validate_controls" if method == "hudiff_ab" else "validate_parameters"
    validator = getattr(app, name)
    calls = []

    def validate(**parameters):
        calls.append(parameters)
        return validator(**parameters)

    monkeypatch.setattr(app, name, validate)
    workflow.build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n")
    assert len(calls) == 1


@pytest.mark.parametrize("identity", ["pabnativ2", "hudiff_ab.patch"])
def test_app_scientific_identity_changes_workflow_fingerprint(monkeypatch, identity):
    """Patch/wrapper/asset identities must reach the execution reuse boundary."""
    from biomodals.execution.definition_plan import execution_plan

    expected = {
        "pabnativ2": workflow.pabnativ2_app.SCIENTIFIC_RUNTIME_IDENTITY,
        "hudiff_ab.patch": workflow.hudiff_app.patch_identity(),
    }
    assert workflow.SCIENTIFIC_VERSIONS[identity] == expected[identity]
    inputs = b"id,vh,vl\na,ACD,EFG\n"
    original = execution_plan(
        workflow.build_humanization_workflow(inputs).validate(), workload_run_key="test"
    ).workload_plan_fingerprint
    monkeypatch.setitem(workflow.SCIENTIFIC_VERSIONS, identity, "changed-identity")
    assert (
        execution_plan(
            workflow.build_humanization_workflow(inputs).validate(),
            workload_run_key="test",
        ).workload_plan_fingerprint
        != original
    )


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
    assert workflow.CONF.tags["biomodals_tool"] == "humanization"
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
    from uuid import UUID

    run_id = UUID(int=1)
    publication = SimpleNamespace(run=SimpleNamespace(status=RunStatus.SUCCEEDED))
    monkeypatch.setattr(workflow, "uuid4", lambda: run_id)
    monkeypatch.setattr(workflow, "stage_execution_launch", lambda *args: None)
    staged = []
    monkeypatch.setattr(
        workflow, "stage_execution_request", lambda *args: staged.append(args[-1])
    )
    call = SimpleNamespace(get=lambda: publication, object_id="fc-test")
    scientific_result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="humanization_results",
                kind=ArtifactKind.DIRECTORY,
                storage=VolumePath(
                    volume_name="test-results", path="parent-run/humanization"
                ),
            )
        ],
    )
    monkeypatch.setattr(
        workflow,
        "execution_coordinator_handle",
        lambda **kwargs: SimpleNamespace(
            run=SimpleNamespace(spawn=lambda **kwargs: call),
            result=SimpleNamespace(remote=lambda: scientific_result),
        ),
    )
    workflow.submit_humanization_workflow.info.raw_f(
        input_csv=str(source),
        use_deployed_coordinator=True,
    )
    output = capsys.readouterr().out
    assert "Humanization completed: succeeded" in output
    assert "humanization_results" in output
    assert "test-results" in output
    assert "parent-run/humanization/selection.csv" in output
    assert staged[0].pairs[0].id == "a"
    assert staged[0].max_active_gpu_provider_calls == 2
