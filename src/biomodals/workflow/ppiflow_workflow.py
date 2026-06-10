"""PPIFlow workflow definition built on the reusable workflow runtime."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal
import yaml

from biomodals.app.design import ppiflow_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import volume_path_from_mount_path
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppRunResult,
    ArtifactKind,
    NodeExecutionPolicy,
    NodePlacement,
)
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    RemoteNodeSubmission,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
    print_workflow_dag,
)

PPI_FLOW_OUTPUT_LAYOUT = (
    "stage1/",
    "stage2/",
    "design_output/",
    "design_output/ranked_designs.csv",
    "design_output/design_report.md",
)
PPI_FLOW_APP_STEPS = ("PPIFlowStep", "PartialStep")

DEPENDENCY_APPS = ("ppiflow",)
CONF = AppConfig(
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="PPIFlowWorkflow",
    package_name="biomodals-ppiflow-workflow",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)


@dataclass(frozen=True)
class PPIFlowModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    ppiflow_run: modal.Function


@dataclass
class PPIFlowWorkflowNode(AppBackedNode):
    """Base class for PPIFlow v2 app-backed workflow nodes."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def _app_kwargs(self, context: NodeRunContext) -> dict[str, object]:
        if self.step_name not in PPI_FLOW_APP_STEPS:
            raise NotImplementedError(
                f"PPIFlow workflow step {self.step_name!r} does not yet have a "
                "workflow-compatible app adapter."
            )

        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")

        run_name = sanitize_filename(
            str(self.config.get("run_name") or f"{context.run_id}-{self.step_name}")
        )
        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        return {"args": app_args, "run_name": run_name}

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the PPIFlow app function directly from the orchestrator."""
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ppiflow_run.spawn(
                **self._app_kwargs(context)
            ),
            function_name="ppiflow_run",
        )


@dataclass
class FilterStructuresNode(WorkflowNativeNode):
    """Filter structures using score artifacts."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute filtering logic."""
        raise NotImplementedError


@dataclass
class RankAndReportNode(WorkflowNativeNode):
    """Rank final designs and write report artifacts."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute ranking and report logic."""
        raise NotImplementedError


def build_ppiflow_workflow(
    *,
    task_yaml_bytes: bytes,
    steps_yaml_bytes: bytes,
    stage: int | None = None,
    modal_namespace: PPIFlowModalNamespace | None = None,
) -> Workflow:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    if modal_namespace is None:
        modal_namespace = PPIFlowModalNamespace(
            ppiflow_run=ppiflow_app.ppiflow_run_workflow,
        )

    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    workflow = Workflow("ppiflow-v2")

    stage1_tail = None
    if stage in {None, 1}:
        stage1_tail = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            modal_namespace=modal_namespace,
        )

    if stage in {None, 2}:
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage1_tail if stage is None else None,
            modal_namespace=modal_namespace,
        )

    return workflow


def _add_stage1_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    modal_namespace: PPIFlowModalNamespace,
):
    tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "PPIFlowStep", modal_namespace),
            id="stage1-ppiflow-design",
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage1"):
        mpnn_step = ("stage1-ligandmpnn", "MPNNStep_stage1")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage1"
    ):
        mpnn_step = ("stage1-abmpnn", "AbMPNNStep_stage1")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            _app_step_node(steps, step_name, modal_namespace),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            _app_step_node(steps, "FlowpackerStep_stage1", modal_namespace),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = workflow.add_node(
            _app_step_node(steps, "AF3scoreStep_stage1", modal_namespace),
            id="stage1-af3score",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FilterStep_stage1"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        tail = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage1",
                _step_cfg(steps, "FilterStep_stage1"),
            ),
            id="stage1-filter",
            inputs=inputs,
        )
    return tail


def _add_stage2_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    modal_namespace: PPIFlowModalNamespace,
) -> None:
    tail = upstream
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "RosettaFixStep", modal_namespace),
            id="stage2-rosetta-fix",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "PartialStep", modal_namespace),
            id="stage2-partial-ppiflow",
            inputs=_structure_inputs(tail),
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage2"):
        mpnn_step = ("stage2-ligandmpnn", "MPNNStep_stage2")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage2"
    ):
        mpnn_step = ("stage2-abmpnn", "AbMPNNStep_stage2")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            _app_step_node(steps, step_name, modal_namespace),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            _app_step_node(steps, "FlowpackerStep_stage2", modal_namespace),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = workflow.add_node(
            _app_step_node(steps, "AF3scoreStep_stage2", modal_namespace),
            id="stage2-af3score",
            inputs=_structure_inputs(tail),
        )

    filtered = tail
    if _step_enabled(enabled, "FilterStep_stage2"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        filtered = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage2",
                _step_cfg(steps, "FilterStep_stage2"),
            ),
            id="stage2-filter",
            inputs=inputs,
        )

    refold = None
    if _step_enabled(enabled, "ReFoldStep"):
        refold = workflow.add_node(
            _app_step_node(steps, "ReFoldStep", modal_namespace),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
        )

    if _step_enabled(enabled, "RosettaRelaxStep"):
        workflow.add_node(
            _app_step_node(steps, "RosettaRelaxStep", modal_namespace),
            id="stage2-rosetta-relax",
            inputs=_structure_inputs(filtered),
        )

    dockq = None
    if _step_enabled(enabled, "DockQStep"):
        inputs = _structure_inputs(filtered)
        if refold is not None:
            inputs["models"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        dockq = workflow.add_node(
            _app_step_node(steps, "DockQStep", modal_namespace),
            id="stage2-dockq",
            inputs=inputs,
        )

    if _step_enabled(enabled, "RankStep") or _step_enabled(enabled, "ReportStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        workflow.add_node(
            RankAndReportNode("RankAndReportStep", _rank_report_cfg(steps)),
            id="stage2-rank-report",
            inputs=inputs,
        )


def _structure_inputs(upstream) -> dict[str, Any]:
    if upstream is None:
        return {}
    return {"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)}


def _app_step_node(
    steps: dict[str, Any],
    step_name: str,
    modal_namespace: PPIFlowModalNamespace,
) -> PPIFlowWorkflowNode:
    return PPIFlowWorkflowNode(
        step_name=step_name,
        modal_namespace=modal_namespace,
        config=_step_cfg(steps, step_name),
    )


def _load_yaml_bytes(data: bytes) -> dict[str, Any]:
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return loaded


def _task_section(task_doc: dict[str, Any]) -> dict[str, Any]:
    section = task_doc.get("task", task_doc)
    if not isinstance(section, dict):
        raise ValueError("task.yaml must contain a mapping under 'task'")
    return section


def _enabled_section(task_doc: dict[str, Any]) -> dict[str, bool]:
    enabled = task_doc.get("steps", {})
    if not isinstance(enabled, dict):
        raise ValueError("task.yaml 'steps' section must be a mapping")
    return {str(key): bool(value) for key, value in enabled.items()}


def _step_enabled(enabled: dict[str, bool], step_name: str) -> bool:
    return bool(enabled.get(step_name, False))


def _step_cfg(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    cfg = steps.get(step_name, {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"steps.yaml entry {step_name!r} must be a mapping")
    return cfg


def _rank_report_cfg(steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "RankStep": _step_cfg(steps, "RankStep"),
        "ReportStep": _step_cfg(steps, "ReportStep"),
    }


def _ppiflow_input_fields(args: object) -> tuple[str, ...]:
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyConfig):
        return ("antigen_pdb", "framework_pdb")
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyPartialConfig):
        return ("complex_pdb",)
    if isinstance(
        args,
        (ppiflow_app.SampleBinderConfig, ppiflow_app.SampleBinderPartialConfig),
    ):
        return ("input_pdb",)
    raise TypeError(f"Unsupported PPIFlow args type: {type(args).__name__}")


def _active_ppiflow_app_steps(
    task_doc: dict[str, Any], stage: int | None
) -> tuple[str, ...]:
    """Return PPIFlow app steps that should be staged for the selected run."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    enabled = _enabled_section(task_doc)
    active_steps: list[str] = []
    if stage in {None, 1} and _step_enabled(enabled, "PPIFlowStep"):
        active_steps.append("PPIFlowStep")
    if stage in {None, 2} and _step_enabled(enabled, "PartialStep"):
        active_steps.append("PartialStep")
    return tuple(active_steps)


def _stage_ppiflow_app_inputs(
    *,
    steps_doc: dict[str, Any],
    run_id: str,
    app_steps: tuple[str, ...],
) -> dict[str, Any]:
    """Upload local PPIFlow app inputs and rewrite step args to mounted paths."""
    staged_steps = deepcopy(steps_doc)
    uploads: list[tuple[Path, str]] = []
    volume_root = Path(ppiflow_app.CONF.output_volume_mountpoint)

    for step_name in app_steps:
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        raw_args = cfg.get("args", cfg)
        if not isinstance(raw_args, dict):
            continue

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        for field_name in _ppiflow_input_fields(app_args.args):
            current_value = getattr(app_args.args, field_name)
            current_path = Path(current_value)
            if current_path.is_absolute() and current_path.is_relative_to(volume_root):
                continue

            local_path = current_path.expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PPIFlow {step_name} input {field_name!r} was not found "
                    f"locally or in the mounted output volume: {current_value}"
                )

            remote_rel = (
                Path(run_id)
                / sanitize_filename(step_name)
                / sanitize_filename(field_name)
                / sanitize_filename(local_path.name)
            )
            raw_args[field_name] = str(volume_root / remote_rel)
            uploads.append((local_path, remote_rel.as_posix()))

    if uploads:
        with ppiflow_app.CONF.output_volume.batch_upload() as batch:
            for local_path, remote_rel in uploads:
                remote_storage = volume_path_from_mount_path(
                    str(volume_root / remote_rel),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                print(
                    f"Uploading PPIFlow input '{local_path}' to {remote_storage}",
                    flush=True,
                )
                batch.put_file(local_path, f"/{remote_storage.path}")
    return staged_steps


@app.local_entrypoint()
def submit_ppiflow_workflow(
    task_yaml: str,
    steps_yaml: str,
    run_id: str | None = None,
    stage: int | None = None,
    force: bool = False,
    wait: bool = True,
    max_parallel: int = 16,
    dry_run: bool = False,
) -> None:
    """Build and submit a PPIFlow workflow from task and step YAML files.

    Args:
        task_yaml: Path to the PPIFlow task YAML declaring enabled workflow
            steps and design mode.
        steps_yaml: Path to the YAML file containing per-step app arguments.
        run_id: Stable workflow run id for durable ledger state. Defaults to
            the task YAML filename stem.
        stage: Optional stage selector. Use 1 for stage 1 only, 2 for stage 2
            only, or omit to build both stages.
        force: Replace an existing workflow run ledger before running.
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_parallel: Maximum number of ready workflow nodes to execute
            concurrently in one scheduler wave.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
    """
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    steps_yaml_bytes = steps_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    if dry_run:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=task_yaml_bytes,
            steps_yaml_bytes=steps_yaml_bytes,
            stage=stage,
        )
        print_workflow_dag(workflow.validate())
        return

    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_bytes),
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=task_yaml_bytes,
        steps_yaml_bytes=yaml.safe_dump(steps_doc).encode("utf-8"),
        stage=stage,
    )

    orchestrator_handle = orchestrator.WorkflowOrchestrator()
    orchestrator_kwargs = {
        "workflow": workflow,
        "run_id": resolved_run_id,
        "force": force,
        "max_ready_workers": max_parallel,
    }
    print(
        f"Submitting PPIFlow workflow '{resolved_run_id}' with "
        f"{len(workflow.validate().nodes)} node(s)",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(
            orchestrator_handle.run.remote(**orchestrator_kwargs)
        )
    else:
        function_call = orchestrator_handle.run.spawn(**orchestrator_kwargs)
        result = str(getattr(function_call, "object_id", function_call))
    if isinstance(result, AppRunResult):
        print(f"PPIFlow workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"PPIFlow workflow run submitted. FunctionCall id: {result}", flush=True)
