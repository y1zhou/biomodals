"""Real CLI outcomes with only staging and remote provider boundaries replaced."""

import importlib
from types import SimpleNamespace

import orjson
import pytest
from typer.testing import CliRunner

from biomodals import cli
from biomodals.execution import RunStatus
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)

VHH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"


@pytest.mark.parametrize(
    "name", ["humanization", "nanobody_humanization", "protein_optimization"]
)
@pytest.mark.parametrize(
    ("status", "wait"),
    [
        (RunStatus.FAILED, True),
        (RunStatus.CANCELLED, True),
        (RunStatus.SUSPENDED, True),
        (RunStatus.STATE_UNKNOWN, True),
        (RunStatus.SUCCEEDED, True),
        (RunStatus.PARTIAL, True),
        (RunStatus.FAILED, False),
    ],
)
def test_cli_reports_waited_outcome(tmp_path, monkeypatch, name, status, wait):
    """Retain exit codes/transcripts; accept partial evidence only where intended."""
    workflow = importlib.import_module(f"biomodals.workflow.{name}.workflow")
    source = tmp_path / "input.csv"
    source.write_text(
        {
            "humanization": "id,vh,vl\na,ACD,EFG\n",
            "nanobody_humanization": f"id,vhh\na,{VHH}\n",
            "protein_optimization": "mutations,label\nA:A1C,1\nA:A2D,2\n",
        }[name]
    )
    monkeypatch.setattr(cli, "_resolve_deployment_version", lambda **kwargs: 1)
    monkeypatch.setattr(workflow, "stage_execution_launch", lambda *args: None)
    if name == "protein_optimization":
        monkeypatch.setattr(
            workflow, "REQUEST_FILE", SimpleNamespace(stage=lambda *args: None)
        )
    else:
        monkeypatch.setattr(workflow, "stage_execution_request", lambda *args: None)
    if name == "nanobody_humanization":
        for app_name in ("abnativ2_vhh", "hudiff_nb"):
            module = importlib.import_module(f"biomodals.app.design.{app_name}.app")
            monkeypatch.setattr(
                module,
                f"stage_{app_name}_models",
                SimpleNamespace(remote=lambda: None),
            )
        monkeypatch.setattr(
            workflow.modal.Function,
            "from_name",
            lambda *args, **kwargs: SimpleNamespace(remote=lambda: None),
        )
    reads = []

    def get():
        reads.append("overview")
        return SimpleNamespace(
            run=SimpleNamespace(
                status=status, status_message="Retained diagnostic", status_reason=None
            )
        )

    def result():
        reads.append("result")
        output_name = {
            "humanization": "humanization_results",
            "nanobody_humanization": "nanobody_results",
            "protein_optimization": "optimization_results",
        }[name]
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=output_name,
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(volume_name="results", path="retained/result"),
                )
            ],
        )

    call = SimpleNamespace(get=get, object_id="fc-test")
    monkeypatch.setattr(
        workflow,
        "execution_coordinator_handle",
        lambda **kwargs: SimpleNamespace(
            run=SimpleNamespace(spawn=lambda **kwargs: call),
            result=SimpleNamespace(remote=result),
        ),
    )
    arguments = ["workflow", "run", name, "--", "--input-csv", str(source)]
    if not wait:
        arguments.append("--no-wait")
    outcome = CliRunner().invoke(cli.app, arguments)
    (tmp_path / "cli-outcome.json").write_bytes(
        orjson.dumps({
            "workflow": name,
            "status": status.value,
            "wait": wait,
            "exit_code": outcome.exit_code,
            "transcript": outcome.output,
            "exception": str(outcome.exception) if outcome.exception else None,
        })
    )
    accepted = status == RunStatus.SUCCEEDED or (
        name != "protein_optimization" and status == RunStatus.PARTIAL
    )
    if not wait:
        assert outcome.exit_code == 0, outcome.output
        assert reads == []
    elif accepted:
        assert outcome.exit_code == 0, outcome.output
        assert reads == ["overview", "result"]
        assert "retained/result/" in outcome.output
    else:
        assert outcome.exit_code != 0
        assert status.value in str(outcome.exception)
        assert "Retained diagnostic" in str(outcome.exception)
        assert reads == ["overview"]
