"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import modal

from biomodals.app.design import ppiflow_app
from biomodals.helper.styling import strip_ansi
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.workflow import ppiflow_workflow
from biomodals.workflow.core import NodeRunContext
from biomodals.workflow.ppiflow_workflow import (
    CONF,
    PPIFlowModalNamespace,
    _active_ppiflow_app_steps,
    _stage_ppiflow_app_inputs,
    build_ppiflow_workflow,
)


class _FakeFunctionCall:
    def __init__(self, object_id: str, result: AppRunResult | None = None) -> None:
        self.object_id = object_id
        self.result = result or AppRunResult(status=AppRunStatus.SUCCEEDED)

    def get(self, timeout=None):
        _ = timeout
        return self.result


class _FakePPIFlowFunction:
    def __init__(self) -> None:
        self.kwargs = {}

    def _result(self) -> AppRunResult:
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ppiflow_outputs",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=ppiflow_app.CONF.output_volume_name,
                        path="demo-run",
                    ),
                )
            ],
        )

    def remote(self, **kwargs):
        self.kwargs = kwargs
        return self._result()

    def spawn(self, **kwargs):
        self.kwargs = kwargs
        return _FakeFunctionCall("fc-ppiflow", self._result())


def _task_yaml(*, enabled_steps: str) -> bytes:
    return f"""
task:
  gentype: binder
steps:
{enabled_steps}
""".encode()


def test_ppiflow_workflow_declares_app_dependency() -> None:
    assert CONF.depends_on_apps == ("ppiflow",)
    assert CONF.tags == {"depends_on": "ppiflow"}


def test_ppiflow_app_step_uses_included_modal_namespace(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  PPIFlowStep: true\n"),
        steps_yaml_bytes=b"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: /inputs/demo.pdb
    binder_chain: B
""",
        modal_namespace=namespace,
    )

    definition = workflow.validate()
    spec = definition.nodes["stage1-ppiflow-design"]
    result = spec.node.run(
        NodeRunContext(
            run_id="run-1",
            node_id=spec.node_id,
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert fake_function.kwargs["run_name"] == "demo-run"
    assert isinstance(fake_function.kwargs["args"], ppiflow_app.PPIFlowArgs)
    assert result.outputs[0].storage == VolumePath(
        volume_name=ppiflow_app.CONF.output_volume_name,
        path="demo-run",
    )


def test_ppiflow_app_step_submits_app_function_directly(tmp_path: Path) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  PPIFlowStep: true\n"),
        steps_yaml_bytes=b"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: /inputs/demo.pdb
    binder_chain: B
""",
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-ppiflow-design"]
    submission = spec.node.submit_remote(
        NodeRunContext(
            run_id="run-1",
            node_id=spec.node_id,
            attempt_id="attempt-1",
            cache_dir=tmp_path,
            inputs={},
        )
    )

    assert submission.function_name == "ppiflow_run"
    assert submission.function_call.object_id == "fc-ppiflow"
    assert fake_function.kwargs["run_name"] == "demo-run"
    assert isinstance(fake_function.kwargs["args"], ppiflow_app.PPIFlowArgs)


def test_submit_ppiflow_workflow_dry_run_prints_dag_without_orchestrator(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    task_yaml = tmp_path / "task.yaml"
    steps_yaml = tmp_path / "steps.yaml"
    input_pdb = tmp_path / "demo.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    task_yaml.write_bytes(_task_yaml(enabled_steps="  PPIFlowStep: true\n"))
    steps_yaml.write_text(
        f"""
PPIFlowStep:
  run_name: demo-run
  args:
    name: demo
    specified_hotspots: A1
    input_pdb: {input_pdb}
    binder_chain: B
""",
        encoding="utf-8",
    )

    class UnexpectedWorkflowOrchestrator:
        def __init__(self) -> None:
            raise AssertionError("dry-run should not construct the orchestrator")

    monkeypatch.setattr(
        ppiflow_workflow.orchestrator,
        "WorkflowOrchestrator",
        UnexpectedWorkflowOrchestrator,
    )

    raw_f = ppiflow_workflow.submit_ppiflow_workflow.info.raw_f
    assert raw_f is not None
    raw_f(
        task_yaml=str(task_yaml),
        steps_yaml=str(steps_yaml),
        run_id="demo",
        dry_run=True,
    )

    stdout = strip_ansi(capsys.readouterr().out)
    assert "[workflow] DAG graph: node_id [placement; class] <- dependency" in stdout
    assert (
        "[workflow]   stage1-ppiflow-design [remote; PPIFlowWorkflowNode] <- -"
        in stdout
    )
    assert "ppiflow_workflow.PPIFlowWorkflowNode" not in stdout
    assert "Submitting PPIFlow workflow" not in stdout


def test_ppiflow_unsupported_steps_fail_with_clear_adapter_error(
    tmp_path: Path,
) -> None:
    fake_function = _FakePPIFlowFunction()
    namespace = PPIFlowModalNamespace(
        ppiflow_run=cast(modal.Function, fake_function),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(enabled_steps="  FlowpackerStep_stage1: true\n"),
        steps_yaml_bytes=b"FlowpackerStep_stage1: {}\n",
        modal_namespace=namespace,
    )

    spec = workflow.validate().nodes["stage1-flowpacker"]
    try:
        spec.node.run(
            NodeRunContext(
                run_id="run-1",
                node_id=spec.node_id,
                attempt_id="attempt-1",
                cache_dir=tmp_path,
                inputs={},
            )
        )
    except NotImplementedError as exc:
        assert "workflow-compatible app adapter" in str(exc)
    else:
        raise AssertionError("unsupported PPIFlow step should fail clearly")


def test_ppiflow_entrypoint_stages_local_app_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )

    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/input_pdb/input.pdb"
    )
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_uses_active_stage_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    task_doc = {
        "steps": {
            "PPIFlowStep": True,
            "PartialStep": True,
        }
    }
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "input_pdb": str(input_pdb),
                "binder_chain": "B",
            }
        },
        "PartialStep": {
            "args": {
                "name": "demo-partial",
                "specified_hotspots": "A1",
                "input_pdb": str(tmp_path / "stage2-not-local.pdb"),
                "fixed_positions": "B1",
                "start_t": 0.5,
            }
        },
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=_active_ppiflow_app_steps(task_doc, stage=1),
    )

    assert staged["PPIFlowStep"]["args"]["input_pdb"].endswith(
        "/PPIFlowStep/input_pdb/input.pdb"
    )
    assert staged["PartialStep"]["args"]["input_pdb"].endswith("stage2-not-local.pdb")
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input_pdb/input.pdb")]


def test_ppiflow_staging_keeps_same_basename_inputs_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    antigen_pdb = tmp_path / "antigen" / "input.pdb"
    framework_pdb = tmp_path / "framework" / "input.pdb"
    antigen_pdb.parent.mkdir()
    framework_pdb.parent.mkdir()
    antigen_pdb.write_text("ATOM antigen\n", encoding="utf-8")
    framework_pdb.write_text("ATOM framework\n", encoding="utf-8")
    uploaded = []

    class FakeBatch:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def put_file(self, local_path, remote_path):
            uploaded.append((Path(local_path), remote_path))

    class FakeVolume:
        def batch_upload(self):
            return FakeBatch()

    monkeypatch.setattr(
        ppiflow_app,
        "CONF",
        SimpleNamespace(
            output_volume=FakeVolume(),
            output_volume_mountpoint="/biomodals-outputs",
            output_volume_name="PPIFlow-outputs",
        ),
    )
    steps_doc = {
        "PPIFlowStep": {
            "args": {
                "name": "demo",
                "specified_hotspots": "A1",
                "antigen_pdb": str(antigen_pdb),
                "antigen_chain": "A",
                "framework_pdb": str(framework_pdb),
                "heavy_chain": "H",
            }
        }
    }

    staged = _stage_ppiflow_app_inputs(
        steps_doc=steps_doc,
        run_id="run-1",
        app_steps=("PPIFlowStep",),
    )

    assert staged["PPIFlowStep"]["args"]["antigen_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/antigen_pdb/input.pdb"
    )
    assert staged["PPIFlowStep"]["args"]["framework_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/framework_pdb/input.pdb"
    )
    assert uploaded == [
        (antigen_pdb, "/run-1/PPIFlowStep/antigen_pdb/input.pdb"),
        (framework_pdb, "/run-1/PPIFlowStep/framework_pdb/input.pdb"),
    ]
