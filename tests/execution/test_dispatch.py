"""Fixed-batch construction and policy persistence tests."""

# ruff: noqa: D103, S106

import sqlite3
from dataclasses import replace

import orjson
import pytest

from biomodals.execution import ProviderBinding
from biomodals.execution.scheduler import (
    PullWorkerDispatchDescriptor,
    TaskDispatchDescriptor,
    form_fixed_batches,
    form_pull_worker_candidates,
)

from .provider_call_helpers import RUN_ID, create_repository

GPU = ProviderBinding(
    "production",
    "biomodals-af3",
    23,
    "run_inference",
    True,
    "af3-gpu",
)
CPU = ProviderBinding(
    "production",
    "biomodals-af3",
    23,
    "run_search",
    False,
    "af3-search",
)


def _task(
    task_key: str,
    ordinal: int,
    *,
    binding: ProviderBinding = GPU,
    compatibility_key: str = "same-model",
    max_tasks_per_call: int = 2,
) -> TaskDispatchDescriptor:
    return TaskDispatchDescriptor(
        node_key="inference",
        node_ordinal=2,
        task_key=task_key,
        task_ordinal=ordinal,
        binding=binding,
        compatibility_key=compatibility_key,
        max_tasks_per_call=max_tasks_per_call,
        depth=2,
        unblocking_span=1,
    )


def test_fixed_batches_preserve_compatibility_and_encounter_order() -> None:
    batches = form_fixed_batches((
        _task("seed-0", 0),
        _task("seed-1", 1, compatibility_key="other-model"),
        _task("seed-2", 2),
        _task("seed-3", 3),
        _task("search-0", 4, binding=CPU),
    ))

    assert [
        (
            batch.binding.function_name,
            batch.compatibility_key,
            batch.task_keys,
            batch.task_ordinal,
            batch.max_tasks_per_call,
        )
        for batch in batches
    ] == [
        ("run_inference", "same-model", ("seed-0", "seed-2"), 0, 2),
        ("run_inference", "same-model", ("seed-3",), 3, 2),
        ("run_inference", "other-model", ("seed-1",), 1, 2),
        ("run_search", "same-model", ("search-0",), 4, 2),
    ]


def test_fixed_batches_never_span_nodes() -> None:
    second_node = TaskDispatchDescriptor(
        node_key="ranking",
        node_ordinal=3,
        task_key="seed-1",
        task_ordinal=0,
        binding=GPU,
        compatibility_key="same-model",
        max_tasks_per_call=2,
        depth=3,
        unblocking_span=0,
    )

    batches = form_fixed_batches((_task("seed-0", 0), second_node))

    assert [batch.task_keys for batch in batches] == [("seed-0",), ("seed-1",)]


def test_fixed_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_tasks_per_call must be positive"):
        form_fixed_batches((_task("seed-0", 0, max_tasks_per_call=0),))


def test_fixed_batch_policy_is_durable_and_immutable_within_a_run() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=3)
    descriptors = tuple(
        replace(_task(f"seed-{index}", index), node_ordinal=0) for index in range(3)
    )

    persisted, changed = repository.persist_fixed_dispatch_policy(
        RUN_ID,
        descriptors,
        now=110,
    )

    assert persisted == descriptors
    assert changed is True

    claim = repository.preclaim_fixed_batch(
        RUN_ID,
        "inference",
        ("seed-0", "seed-1"),
        submission_token="batch",
        binding=GPU,
        compatibility_key="same-model",
        max_tasks_per_call=2,
        now=112,
    )

    assert claim is not None
    task_policies = connection.execute(
        """
        SELECT dispatch_policy_json
        FROM execution_tasks
        WHERE execution_run_id = ? AND node_key = ?
        ORDER BY ordinal
        """,
        (str(RUN_ID), "inference"),
    ).fetchall()
    assert [orjson.loads(row[0])["max_tasks_per_call"] for row in task_policies] == [
        2,
        2,
        2,
    ]

    with pytest.raises(
        ValueError,
        match="fixed-batch dispatch policy cannot change within a Run",
    ):
        repository.persist_fixed_dispatch_policy(
            RUN_ID,
            (
                replace(
                    _task("seed-2", 2, max_tasks_per_call=1),
                    node_ordinal=0,
                ),
            ),
            now=113,
        )


def test_fixed_batch_policy_validates_tasks_with_bulk_queries() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=100)
    descriptors = tuple(
        replace(_task(f"seed-{index}", index), node_ordinal=0) for index in range(100)
    )
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    repository.persist_fixed_dispatch_policy(RUN_ID, descriptors, now=110)
    connection.set_trace_callback(None)

    node_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and "FROM execution_nodes" in statement
    ]
    task_selects = [
        statement for statement in statements if "FROM execution_tasks" in statement
    ]
    assert len(node_selects) == 1
    assert len(task_selects) == 1


def test_ready_dispatch_query_uses_the_resource_class_index() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=100)
    descriptors = tuple(
        replace(_task(f"seed-{index}", index), node_ordinal=0) for index in range(100)
    )
    repository.persist_fixed_dispatch_policy(RUN_ID, descriptors, now=110)

    query_plan = connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT task.*, node.ordinal AS node_ordinal
        FROM execution_tasks AS task
        JOIN execution_nodes AS node
            ON node.execution_run_id = task.execution_run_id
            AND node.node_key = task.node_key
        WHERE task.execution_run_id = ?
            AND task.node_key IN (?)
            AND node.status = ?
            AND node.discovery_complete = 1
            AND task.status = ?
            AND task.result_observation = ?
            AND task.dispatch_policy_json IS NOT NULL
            AND json_extract(
                task.dispatch_policy_json,
                '$.binding.uses_gpu'
            ) = ?
        ORDER BY node.ordinal, task.ordinal
        LIMIT ?
        """,
        (str(RUN_ID), "inference", "running", "pending", "missing", 1, 8),
    ).fetchall()

    assert any(
        "execution_tasks_ready_dispatch_resource_idx" in str(row[3])
        for row in query_plan
    )


def test_pull_worker_policy_is_persisted_before_candidate_formation() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=3)
    descriptor = PullWorkerDispatchDescriptor(
        node_key="inference",
        node_ordinal=0,
        binding=GPU,
        compatibility_key="af3-seeds",
        claim_capacity=2,
        unfinished_task_count=3,
        nonterminal_worker_count=0,
        next_worker_ordinal=0,
        depth=0,
        unblocking_span=0,
    )

    persisted, changed = repository.persist_pull_worker_dispatch_policy(
        RUN_ID,
        descriptor,
        now=110,
    )

    assert persisted == descriptor
    assert changed is True
    mode, policy_json = connection.execute(
        """
        SELECT dispatch_mode, dispatch_policy_json
        FROM execution_nodes
        WHERE execution_run_id = ? AND node_key = ?
        """,
        (str(RUN_ID), "inference"),
    ).fetchone()
    assert mode == "pull_worker"
    assert orjson.loads(policy_json)["claim_capacity"] == 2

    with pytest.raises(
        ValueError,
        match="pull-worker dispatch policy cannot change within a Run",
    ):
        repository.persist_pull_worker_dispatch_policy(
            RUN_ID,
            replace(descriptor, claim_capacity=1),
            now=111,
        )


def test_pull_worker_candidates_fill_the_derived_pool_gap() -> None:
    candidates = form_pull_worker_candidates((
        PullWorkerDispatchDescriptor(
            node_key="rosetta",
            node_ordinal=3,
            binding=CPU,
            compatibility_key="rosetta-cpu",
            claim_capacity=2,
            unfinished_task_count=7,
            nonterminal_worker_count=1,
            next_worker_ordinal=4,
            depth=2,
            unblocking_span=1,
        ),
    ))

    assert [candidate.candidate_key for candidate in candidates] == [
        "rosetta:run_search:rosetta-cpu:worker-4",
        "rosetta:run_search:rosetta-cpu:worker-5",
        "rosetta:run_search:rosetta-cpu:worker-6",
    ]
    assert all(candidate.task_keys == () for candidate in candidates)


def test_pull_worker_candidates_reject_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="claim_capacity must be positive"):
        form_pull_worker_candidates((
            PullWorkerDispatchDescriptor(
                node_key="rosetta",
                node_ordinal=0,
                binding=CPU,
                compatibility_key="rosetta-cpu",
                claim_capacity=0,
                unfinished_task_count=1,
                nonterminal_worker_count=0,
                next_worker_ordinal=0,
                depth=0,
                unblocking_span=0,
            ),
        ))
