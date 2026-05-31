"""Tests for the PPIFlow workflow definition."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import modal

from biomodals.app.design import ppiflow_app
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.workflow.core import NodeRunContext
from biomodals.workflow.ppiflow_workflow import (
    CONF,
    PPIFlowModalNamespace,
    _stage_ppiflow_app_inputs,
    build_ppiflow_workflow,
)


class _FakePPIFlowFunction:
    def __init__(self) -> None:
        self.kwargs = {}

    def remote(self, **kwargs):
        self.kwargs = kwargs
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

    staged = _stage_ppiflow_app_inputs(steps_doc=steps_doc, run_id="run-1")

    assert staged["PPIFlowStep"]["args"]["input_pdb"] == (
        "/biomodals-outputs/run-1/PPIFlowStep/input.pdb"
    )
    assert uploaded == [(input_pdb, "/run-1/PPIFlowStep/input.pdb")]
