"""Provider-neutral pull-worker loop tests."""

# ruff: noqa: D103

from uuid import UUID

from biomodals.execution import PullTaskClaim, WorkerAssignmentRecord
from biomodals.execution.pull_worker import drive_pull_worker

CALL_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _assignment(task_key: str, ordinal: int) -> WorkerAssignmentRecord:
    return WorkerAssignmentRecord(
        execution_run_id=RUN_ID,
        node_key="rosetta",
        task_key=task_key,
        task_fingerprint=f"fingerprint-{task_key}",
        execution_payload={"index": ordinal + 1},
        provider_call_id=CALL_ID,
        request_id=f"claim-{ordinal}",
        ordinal=ordinal,
        created_at=100,
    )


def test_pull_worker_replays_stable_request_ids_and_drains_until_empty() -> None:
    claims = [
        (_assignment("task-0", 0), _assignment("task-1", 1)),
        (_assignment("task-2", 0),),
        (),
    ]
    claim_calls = []
    completion_calls = []

    def claim(request_id: str, capacity: int) -> PullTaskClaim:
        claim_calls.append((request_id, capacity))
        return PullTaskClaim(
            request_id=request_id,
            provider_call_id=CALL_ID,
            assignments=claims[len(claim_calls) - 1],
        )

    summary = drive_pull_worker(
        provider_call_id=CALL_ID,
        claim_capacity=2,
        claim=claim,
        execute=lambda assignment: str(assignment.execution_payload["index"]),
        complete=lambda assignment, request_id, result: completion_calls.append((
            assignment.task_key,
            request_id,
            result,
        )),
        max_parallel=2,
    )

    assert claim_calls == [
        (f"{CALL_ID}:claim:0", 2),
        (f"{CALL_ID}:claim:1", 2),
        (f"{CALL_ID}:claim:2", 2),
    ]
    assert completion_calls == [
        ("task-0", f"{CALL_ID}:complete:fingerprint-task-0", "1"),
        ("task-1", f"{CALL_ID}:complete:fingerprint-task-1", "2"),
        ("task-2", f"{CALL_ID}:complete:fingerprint-task-2", "1"),
    ]
    assert summary.claimed_tasks == 3
    assert summary.claim_requests == 3


def test_pull_worker_checkpoints_each_completed_microbatch_before_reporting() -> None:
    claims = [
        (_assignment("task-0", 0), _assignment("task-1", 1)),
        (_assignment("task-2", 2),),
        (),
    ]
    events: list[str] = []
    claim_count = 0

    def claim(request_id: str, capacity: int) -> PullTaskClaim:
        nonlocal claim_count
        del capacity
        assignments = claims[claim_count]
        claim_count += 1
        return PullTaskClaim(
            request_id=request_id,
            provider_call_id=CALL_ID,
            assignments=assignments,
        )

    drive_pull_worker(
        provider_call_id=CALL_ID,
        claim_capacity=2,
        claim=claim,
        execute=lambda assignment: events.append(f"execute:{assignment.task_key}"),
        checkpoint_batch=lambda: events.append("checkpoint"),
        complete=lambda assignment, request_id, result: events.append(
            f"complete:{assignment.task_key}"
        ),
        max_parallel=1,
    )

    assert events == [
        "execute:task-0",
        "execute:task-1",
        "checkpoint",
        "complete:task-0",
        "complete:task-1",
        "execute:task-2",
        "checkpoint",
        "complete:task-2",
    ]
