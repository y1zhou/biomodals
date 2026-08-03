"""Small test setup for durable Provider Call behavior."""

from __future__ import annotations

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodeAggregationPolicy,
    NodePlan,
    ProviderBinding,
    SqliteExecutionRepository,
    TaskPlan,
)
from biomodals.execution.scheduler import (
    PullWorkerDispatchDescriptor,
    TaskDispatchDescriptor,
)

RUN_ID = UUID("d4e4744e-aacf-4478-92d6-a58681805162")
GPU_BINDING = ProviderBinding(
    environment="production",
    app_name="biomodals-alphafold3",
    app_version=23,
    function_name="run_inference",
    uses_gpu=True,
    runtime_image_key="alphafold3-gpu",
)
CPU_BINDING = ProviderBinding(
    environment="production",
    app_name="biomodals-alphafold3",
    app_version=23,
    function_name="run_search",
    uses_gpu=False,
    runtime_image_key="alphafold3-search",
)


def create_repository(
    *,
    connection: sqlite3.Connection | None = None,
    task_count: int = 3,
    max_active_provider_calls: int = 3,
    max_active_gpu_provider_calls: int = 2,
    aggregation_policy: NodeAggregationPolicy = NodeAggregationPolicy.COLLECT_ALL,
) -> SqliteExecutionRepository:
    """Return a running Node with cache-missing discovered Tasks."""
    connection = connection or sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="alphafold3",
            nodes=(
                NodePlan(
                    node_key="inference",
                    aggregation_policy=aggregation_policy,
                ),
            ),
        ),
        deployment=DeploymentIdentity(
            "production",
            "biomodals-alphafold3-coordinator",
            23,
        ),
        max_active_provider_calls=max_active_provider_calls,
        max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        now=100,
    )
    repository.start_node(RUN_ID, "inference", now=101)
    repository.discover_tasks(
        RUN_ID,
        "inference",
        tuple(
            TaskPlan(
                task_key=f"seed-{index}",
                scientific_payload={"seed": index},
                execution_payload={"seed": index, "input": "/volume/input.json"},
            )
            for index in range(task_count)
        ),
        now=102,
    )
    for index in range(task_count):
        repository.record_task_result_observation(
            RUN_ID,
            "inference",
            f"seed-{index}",
            AvailabilityStatus.MISSING,
            now=103,
        )
    return repository


def persist_fixed_policy(
    repository: SqliteExecutionRepository,
    task_keys: tuple[str, ...],
    *,
    binding: ProviderBinding,
    compatibility_key: str,
    max_tasks_per_call: int = 1,
    now: int = 109,
) -> None:
    """Bind selected fixture Tasks to one fixed-batch dispatch policy."""
    node = repository.get_node(RUN_ID, "inference")
    tasks = {task.task_key: task for task in repository.list_tasks(RUN_ID, "inference")}
    repository.persist_fixed_dispatch_policy(
        RUN_ID,
        tuple(
            TaskDispatchDescriptor(
                node_key="inference",
                node_ordinal=node.ordinal,
                task_key=task_key,
                task_ordinal=tasks[task_key].ordinal,
                binding=binding,
                compatibility_key=compatibility_key,
                max_tasks_per_call=max_tasks_per_call,
                depth=0,
                unblocking_span=0,
            )
            for task_key in task_keys
        ),
        now=now,
    )


def persist_pull_policy(
    repository: SqliteExecutionRepository,
    *,
    binding: ProviderBinding,
    compatibility_key: str,
    claim_capacity: int,
    max_worker_calls: int = 100,
    now: int = 109,
) -> None:
    """Bind the fixture Node to one pull-worker dispatch policy."""
    node = repository.get_node(RUN_ID, "inference")
    tasks = repository.list_tasks(RUN_ID, "inference")
    repository.persist_pull_worker_dispatch_policy(
        RUN_ID,
        PullWorkerDispatchDescriptor(
            node_key="inference",
            node_ordinal=node.ordinal,
            binding=binding,
            compatibility_key=compatibility_key,
            claim_capacity=claim_capacity,
            max_worker_calls=max_worker_calls,
            unfinished_task_count=sum(not task.status.is_terminal for task in tasks),
            nonterminal_worker_count=0,
            next_worker_ordinal=0,
            depth=0,
            unblocking_span=0,
        ),
        now=now,
    )
