"""Map executable graph definitions to durable kernel plans."""

from __future__ import annotations

from biomodals.execution.definition import ExecutionDefinition
from biomodals.execution.hashing import dag_hash
from biomodals.execution.model import ExecutionPlan, NodeDependency, NodePlan, TaskPlan
from biomodals.schema import AppConfig

_EXECUTION_PLAN_SCHEMA_VERSION = "1"


def app_scientific_version(config: AppConfig) -> str:
    """Return the code/package identity that affects an app's publications."""
    version = config.repo_commit_hash or config.version
    if version is None:  # AppConfig validation normally makes this unreachable.
        raise ValueError(f"App {config.name!r} has no scientific version")
    return version


def execution_plan(
    definition: ExecutionDefinition,
    *,
    workload_run_key: str,
) -> ExecutionPlan:
    """Map one validated Execution Definition to its immutable plan."""
    encounter_order = tuple(definition.nodes)
    nodes = tuple(
        NodePlan(
            node_key=node_id,
            dependencies=tuple(
                NodeDependency(
                    node_key=dependency,
                    accept_partial=(
                        dependency in definition.nodes[node_id].partial_dependencies
                    ),
                )
                for dependency in encounter_order
                if dependency in definition.dependencies[node_id]
            ),
            aggregation_policy=definition.nodes[node_id].aggregation_policy,
            allow_empty_result=definition.nodes[node_id].allow_empty_result,
        )
        for node_id in encounter_order
    )
    metadata = definition.plan_metadata
    if metadata is not None:
        return ExecutionPlan(
            workload_name=metadata.workload_name,
            workload_run_key=workload_run_key,
            nodes=nodes,
            scientific_payload=metadata.scientific_payload,
            scientific_versions=dict(metadata.scientific_versions),
        )
    return ExecutionPlan(
        workload_name=f"workflow:{definition.name}",
        workload_run_key=workload_run_key,
        nodes=nodes,
        scientific_payload={"dag_hash": dag_hash(definition)},
        scientific_versions={
            **definition.scientific_versions,
            "biomodals.workflow.execution_plan": _EXECUTION_PLAN_SCHEMA_VERSION,
        },
    )


def node_task_plan(node_id: str) -> TaskPlan:
    """Represent one Execution Node invocation as one kernel Task."""
    return TaskPlan(
        task_key="node",
        scientific_payload={"workflow_node_id": node_id},
    )
