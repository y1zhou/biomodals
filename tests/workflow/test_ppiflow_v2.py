"""Tests for the PPIFlow v2 workflow definition."""

# ruff: noqa: D103

import pytest

from biomodals.workflow.ppiflow_v2 import (
    PPI_FLOW_OUTPUT_LAYOUT,
    PPI_FLOW_V2_EXECUTION_STATUS,
    PPIFlowWorkflowNode,
    build_ppiflow_workflow,
)


def _task_yaml(*, gentype: str = "binder", stage2: bool = True) -> bytes:
    enabled = {
        "PPIFlowStep": True,
        "MPNNStep_stage1": gentype == "binder",
        "AbMPNNStep_stage1": gentype != "binder",
        "FlowpackerStep_stage1": True,
        "AF3scoreStep_stage1": True,
        "FilterStep_stage1": True,
        "RosettaFixStep": stage2,
        "PartialStep": stage2,
        "MPNNStep_stage2": stage2 and gentype == "binder",
        "AbMPNNStep_stage2": stage2 and gentype != "binder",
        "FlowpackerStep_stage2": stage2,
        "AF3scoreStep_stage2": stage2,
        "FilterStep_stage2": stage2,
        "ReFoldStep": stage2,
        "DockQStep": stage2,
        "RankStep": stage2,
        "ReportStep": stage2,
    }
    step_lines = "\n".join(
        f"  {name}: {str(value).lower()}" for name, value in enabled.items()
    )
    return f"""
task:
  name: smoke
  gentype: {gentype}
steps:
{step_lines}
""".encode()


def test_binder_workflow_models_stage_dependencies() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(gentype="binder"),
        steps_yaml_bytes=b"PPIFlowStep:\n  samples_per_target: 1\n",
    )

    definition = workflow.validate()

    assert "stage1-ligandmpnn" in definition.nodes
    assert "stage1-abmpnn" not in definition.nodes
    assert definition.dependencies["stage1-ligandmpnn"] == {"stage1-ppiflow-design"}
    assert definition.dependencies["stage1-filter"] == {
        "stage1-flowpacker",
        "stage1-af3score",
    }
    assert definition.dependencies["stage2-rosetta-fix"] == {"stage1-filter"}
    assert definition.dependencies["stage2-rank-report"] == {
        "stage2-filter",
        "stage2-dockq",
    }


def test_app_backed_steps_use_data_driven_node_class() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(gentype="binder"),
        steps_yaml_bytes=b"PPIFlowStep:\n  samples_per_target: 1\n",
    )

    definition = workflow.validate()
    ppiflow_node = definition.nodes["stage1-ppiflow-design"].node
    mpnn_node = definition.nodes["stage1-ligandmpnn"].node

    assert type(ppiflow_node) is PPIFlowWorkflowNode
    assert ppiflow_node.step_name == "PPIFlowStep"
    assert type(mpnn_node) is PPIFlowWorkflowNode
    assert mpnn_node.step_name == "MPNNStep_stage1"


def test_antibody_workflow_uses_abmpnn_nodes() -> None:
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(gentype="antibody"),
        steps_yaml_bytes=b"{}",
    )

    definition = workflow.validate()

    assert "stage1-abmpnn" in definition.nodes
    assert "stage1-ligandmpnn" not in definition.nodes
    assert "stage2-abmpnn" in definition.nodes
    assert "stage2-ligandmpnn" not in definition.nodes


def test_stage_selector_limits_workflow_nodes() -> None:
    stage1 = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(stage2=False),
        steps_yaml_bytes=b"{}",
        stage=1,
    ).validate()
    stage2 = build_ppiflow_workflow(
        task_yaml_bytes=_task_yaml(),
        steps_yaml_bytes=b"{}",
        stage=2,
    ).validate()

    assert all(node_id.startswith("stage1-") for node_id in stage1.nodes)
    assert all(node_id.startswith("stage2-") for node_id in stage2.nodes)
    assert stage2.dependencies["stage2-rosetta-fix"] == set()


def test_invalid_stage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stage"):
        build_ppiflow_workflow(
            task_yaml_bytes=_task_yaml(),
            steps_yaml_bytes=b"{}",
            stage=3,
        )


def test_expected_output_layout_matches_legacy_archive_shape() -> None:
    assert PPI_FLOW_OUTPUT_LAYOUT == (
        "stage1/",
        "stage2/",
        "design_output/",
        "design_output/ranked_designs.csv",
        "design_output/design_report.md",
    )


def test_ppiflow_v2_is_explicitly_definition_only() -> None:
    assert PPI_FLOW_V2_EXECUTION_STATUS == "definition-only"
