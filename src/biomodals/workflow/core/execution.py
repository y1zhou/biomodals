"""Execution-kernel plans for reusable workflows."""

from __future__ import annotations

from biomodals.execution import ExecutionPlan, NodeDependency, NodePlan, TaskPlan
from biomodals.workflow.core._runtime.hashing import dag_hash
from biomodals.workflow.core.builder import WorkflowDefinition

_EXECUTION_PLAN_SCHEMA_VERSION = "1"


def execution_plan(
    definition: WorkflowDefinition,
    *,
    workload_run_key: str,
) -> ExecutionPlan:
    """Map one validated workflow definition to immutable execution Nodes."""
    encounter_order = tuple(definition.nodes)
    nodes = tuple(
        NodePlan(
            node_key=node_id,
            dependencies=tuple(
                NodeDependency(node_key=dependency)
                for dependency in encounter_order
                if dependency in definition.dependencies[node_id]
            ),
            aggregation_policy=definition.nodes[node_id].aggregation_policy,
            allow_empty_result=definition.nodes[node_id].allow_empty_result,
        )
        for node_id in encounter_order
    )
    return ExecutionPlan(
        workload_name=f"workflow:{definition.name}",
        workload_run_key=workload_run_key,
        nodes=nodes,
        scientific_payload={"dag_hash": dag_hash(definition)},
        scientific_versions={
            "biomodals.workflow.execution_plan": _EXECUTION_PLAN_SCHEMA_VERSION,
        },
    )


def node_task_plan(node_id: str) -> TaskPlan:
    """Represent one workflow Node invocation as one kernel Task."""
    return TaskPlan(
        task_key="node",
        scientific_payload={"workflow_node_id": node_id},
    )
