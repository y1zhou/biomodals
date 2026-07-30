"""Result-driven DAG boundary tests."""

# ruff: noqa: D103, S106

import sqlite3
from uuid import UUID

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodeDependency,
    NodePlan,
    NodeStatus,
    ProviderBinding,
    RunStatus,
    SqliteExecutionRepository,
    TaskPlan,
    TaskStatus,
    WorkStatusReason,
    required_node_keys,
)

RUN_ID = UUID("d4e4744e-aacf-4478-92d6-a58681805162")
CPU_BINDING = ProviderBinding("prod", "demo", 2, "compute", False)


def _linear_plan() -> ExecutionPlan:
    return ExecutionPlan(
        workload_name="linear",
        nodes=(
            NodePlan(node_key="input"),
            NodePlan(
                node_key="compute",
                dependencies=(NodeDependency(node_key="input"),),
            ),
            NodePlan(
                node_key="summary",
                dependencies=(NodeDependency(node_key="compute"),),
            ),
        ),
    )


def test_required_closure_walks_backward_and_stops_at_available_results() -> None:
    plan = _linear_plan()

    assert (
        required_node_keys(
            plan,
            {
                "summary": AvailabilityStatus.AVAILABLE,
            },
        )
        == ()
    )
    assert required_node_keys(
        plan,
        {
            "summary": AvailabilityStatus.MISSING,
            "compute": AvailabilityStatus.AVAILABLE,
        },
    ) == ("summary",)
    assert required_node_keys(
        plan,
        {
            "summary": AvailabilityStatus.MISSING,
            "compute": AvailabilityStatus.MISSING,
            "input": AvailabilityStatus.MISSING,
        },
    ) == ("input", "compute", "summary")


def test_unknown_result_observation_authorizes_no_work() -> None:
    assert (
        required_node_keys(
            _linear_plan(),
            {
                "summary": AvailabilityStatus.UNKNOWN,
            },
        )
        is None
    )


def _repository() -> SqliteExecutionRepository:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    repository.create_run(
        execution_run_id=RUN_ID,
        plan=_linear_plan(),
        deployment=DeploymentIdentity("prod", "demo-coordinator", 2),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=0,
        now=100,
    )
    return repository


def test_cached_terminal_result_prunes_ancestors_and_finishes_run() -> None:
    repository = _repository()
    repository.record_node_result_observation(
        RUN_ID,
        "summary",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    active_calls = repository.prune_unrequired_nodes(
        RUN_ID,
        required_node_keys=set(),
        now=111,
    )
    run = repository.finalize_run_from_results(RUN_ID, now=112)

    assert active_calls == ()
    assert run.status == RunStatus.SUCCEEDED
    assert {
        node.node_key: (node.status, node.status_reason)
        for node in repository.list_nodes(RUN_ID)
    } == {
        "input": (NodeStatus.SKIPPED, WorkStatusReason.RESULT_ALREADY_SATISFIED),
        "compute": (NodeStatus.SKIPPED, WorkStatusReason.RESULT_ALREADY_SATISFIED),
        "summary": (NodeStatus.SUCCEEDED, None),
    }


def test_running_pruned_ancestor_waits_for_conclusive_call_cancellation() -> None:
    repository = _repository()
    repository.start_node(RUN_ID, "input", now=101)
    repository.discover_tasks(
        RUN_ID,
        "input",
        (
            TaskPlan(task_key="download", scientific_payload={"source": "pdb"}),
            TaskPlan(task_key="unused", scientific_payload={"source": "cache"}),
        ),
        now=102,
    )
    for task_key in ("download", "unused"):
        repository.record_task_result_observation(
            RUN_ID,
            "input",
            task_key,
            AvailabilityStatus.MISSING,
            now=103,
        )
    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "input",
        ("download",),
        submission_token="download",
        binding=CPU_BINDING,
        compatibility_key="pdb",
        now=104,
    )
    assert claim is not None
    repository.attach_provider_call(
        claim.call.provider_call_id,
        provider_call_handle_id="fc-download",
        now=105,
    )
    repository.record_node_result_observation(
        RUN_ID,
        "summary",
        AvailabilityStatus.AVAILABLE,
        now=110,
    )

    active_calls = repository.prune_unrequired_nodes(
        RUN_ID,
        required_node_keys=set(),
        now=111,
    )

    assert active_calls == (claim.call.provider_call_id,)
    pending_sibling = repository.get_task(RUN_ID, "input", "unused")
    assert pending_sibling.status == TaskStatus.SKIPPED
    assert pending_sibling.status_reason == WorkStatusReason.RESULT_ALREADY_SATISFIED
    assert (
        repository.finalize_run_from_results(RUN_ID, now=112).status
        == RunStatus.RUNNING
    )

    repository.cancel_pruned_provider_call(
        claim.call.provider_call_id,
        now=120,
    )
    run = repository.finalize_run_from_results(RUN_ID, now=121)

    task = repository.get_task(RUN_ID, "input", "download")
    node = repository.get_node(RUN_ID, "input")
    assert task.status == TaskStatus.CANCELLED
    assert task.status_reason == WorkStatusReason.RESULT_ALREADY_SATISFIED
    assert node.status == NodeStatus.CANCELLED
    assert node.status_reason == WorkStatusReason.RESULT_ALREADY_SATISFIED
    assert run.status == RunStatus.SUCCEEDED
