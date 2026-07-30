"""Small test setup for durable Provider Call behavior."""

from __future__ import annotations

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    ProviderBinding,
    SqliteExecutionRepository,
    TaskPlan,
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
    task_count: int = 3,
    max_active_provider_calls: int = 3,
    max_active_gpu_provider_calls: int = 2,
) -> SqliteExecutionRepository:
    """Return a running Node with cache-missing discovered Tasks."""
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=ExecutionPlan(
            workload_name="alphafold3",
            nodes=(NodePlan(node_key="inference"),),
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
