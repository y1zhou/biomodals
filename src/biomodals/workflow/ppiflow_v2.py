"""PPIFlow DAG definition built on the reusable workflow builder.

This module is intentionally definition-only in the first workflow runtime
slice. It preserves the legacy PPIFlow stage graph and expected archive layout,
but its nodes are not executable until the corresponding workflow-compatible
app functions and native filter/report implementations are wired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from biomodals.schema import (
    AppRunResult,
    ArtifactKind,
    NodeExecutionPolicy,
    NodePlacement,
)
from biomodals.workflow import AppBackedNode, Workflow, WorkflowNativeNode
from biomodals.workflow.core.nodes import NodeRunContext

PPI_FLOW_OUTPUT_LAYOUT = (
    "stage1/",
    "stage2/",
    "design_output/",
    "design_output/ranked_designs.csv",
    "design_output/design_report.md",
)
PPI_FLOW_V2_EXECUTION_STATUS = "definition-only"


@dataclass
class PPIFlowWorkflowNode(AppBackedNode):
    """Base class for PPIFlow v2 app-backed workflow nodes."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE


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
) -> Workflow:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")

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
        )

    if stage in {None, 2}:
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage1_tail if stage is None else None,
        )

    return workflow


def _add_stage1_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
):
    tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "PPIFlowStep"),
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
            _app_step_node(steps, step_name),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            _app_step_node(steps, "FlowpackerStep_stage1"),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = workflow.add_node(
            _app_step_node(steps, "AF3scoreStep_stage1"),
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
) -> None:
    tail = upstream
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "RosettaFixStep"),
            id="stage2-rosetta-fix",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            _app_step_node(steps, "PartialStep"),
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
            _app_step_node(steps, step_name),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            _app_step_node(steps, "FlowpackerStep_stage2"),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = workflow.add_node(
            _app_step_node(steps, "AF3scoreStep_stage2"),
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
            _app_step_node(steps, "ReFoldStep"),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
        )

    if _step_enabled(enabled, "RosettaRelaxStep"):
        workflow.add_node(
            _app_step_node(steps, "RosettaRelaxStep"),
            id="stage2-rosetta-relax",
            inputs=_structure_inputs(filtered),
        )

    dockq = None
    if _step_enabled(enabled, "DockQStep"):
        inputs = _structure_inputs(filtered)
        if refold is not None:
            inputs["models"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        dockq = workflow.add_node(
            _app_step_node(steps, "DockQStep"),
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


def _app_step_node(steps: dict[str, Any], step_name: str) -> PPIFlowWorkflowNode:
    return PPIFlowWorkflowNode(step_name, _step_cfg(steps, step_name))


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
