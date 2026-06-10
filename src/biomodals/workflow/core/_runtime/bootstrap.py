"""Workflow run bootstrap and resume validation."""

from __future__ import annotations

from biomodals.schema import RunStatus, WorkflowRun
from biomodals.workflow.core._runtime import hashing
from biomodals.workflow.core._runtime.services import RuntimeServices
from biomodals.workflow.core.builder import WorkflowDefinition


def start_run(
    definition: WorkflowDefinition,
    *,
    run_id: str,
    force: bool,
    services: RuntimeServices,
) -> None:
    """Prepare durable run state before the scheduler loop starts."""
    ledger = services.ledger
    volume_sync = services.volume_sync
    dag_hash = hashing.dag_hash(definition)
    volume_sync.reload()
    run_exists = ledger.run_exists(definition.name, run_id)
    if run_exists and force:
        ledger.reset_run(definition.name, run_id)
        volume_sync.commit()
        ledger.create_run(
            WorkflowRun(
                workflow_name=definition.name,
                run_id=run_id,
                dag_hash=dag_hash,
            )
        )
        volume_sync.commit()
    elif run_exists:
        existing_run = ledger.load_run(definition.name, run_id)
        if existing_run.dag_hash is not None and existing_run.dag_hash != dag_hash:
            raise ValueError(
                "DAG hash does not match existing workflow run; rerun with force"
            )
    else:
        ledger.create_run(
            WorkflowRun(
                workflow_name=definition.name,
                run_id=run_id,
                dag_hash=dag_hash,
            )
        )
        volume_sync.commit()
    ledger.mark_run_status(RunStatus.RUNNING)
    volume_sync.commit()
