"""Coordinator-owned SQLite state for execution Runs."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from biomodals.execution.model import (
    ActiveProviderCallCounts,
    AvailabilityStatus,
    DeploymentIdentity,
    DispatchMode,
    ExecutionNodeRecord,
    ExecutionPlan,
    ExecutionRunRecord,
    ExecutionSnapshot,
    ExecutionTaskRecord,
    NodeAggregationPolicy,
    NodeDependency,
    NodePlan,
    NodeStatus,
    ProviderBinding,
    ProviderCallPreclaim,
    ProviderCallRecord,
    ProviderCallStatus,
    PullTaskClaim,
    ResultProvenance,
    RunStatus,
    RunStatusReason,
    TaskPlan,
    TaskStatus,
    WorkerAssignmentRecord,
    WorkStatusReason,
)
from biomodals.execution.scheduler import (
    aggregate_task_outcome,
    propagated_skip_node_keys,
    terminal_run_outcome,
)

EXECUTION_SCHEMA_VERSION = 1


class UnsupportedExecutionSchemaVersionError(RuntimeError):
    """Raised when a repository uses another execution schema version."""


class ExecutionRunNotFoundError(LookupError):
    """Raised when an Execution Run ID is absent from this repository."""


_RUN_TRANSITIONS: Mapping[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.SUSPENDED,
        RunStatus.STATE_UNKNOWN,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }),
    RunStatus.RUNNING: frozenset({
        RunStatus.CANCEL_REQUESTED,
        RunStatus.SUSPENDED,
        RunStatus.STATE_UNKNOWN,
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
    }),
    RunStatus.CANCEL_REQUESTED: frozenset({
        RunStatus.CANCELLED,
        RunStatus.STATE_UNKNOWN,
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
    }),
    RunStatus.SUSPENDED: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.STATE_UNKNOWN,
        RunStatus.FAILED,
    }),
    RunStatus.STATE_UNKNOWN: frozenset({
        RunStatus.RUNNING,
        RunStatus.CANCEL_REQUESTED,
        RunStatus.SUCCEEDED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.PARTIAL: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

_RUN_REASONS: Mapping[RunStatus, frozenset[RunStatusReason]] = {
    RunStatus.SUSPENDED: frozenset({
        RunStatusReason.COORDINATOR_ERROR,
        RunStatusReason.RESULT_VALIDATION_UNKNOWN,
    }),
    RunStatus.STATE_UNKNOWN: frozenset({
        RunStatusReason.SUBMISSION_OUTCOME_UNKNOWN,
        RunStatusReason.PROVIDER_OUTCOME_UNKNOWN,
        RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN,
    }),
    RunStatus.FAILED: frozenset({
        RunStatusReason.REQUIRED_WORK_FAILED,
        RunStatusReason.DEPLOYMENT_UNAVAILABLE,
    }),
}

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE execution_schema (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE execution_runs (
        execution_run_id TEXT PRIMARY KEY,
        predecessor_execution_run_id TEXT,
        workload_name TEXT NOT NULL,
        workload_run_key TEXT,
        workload_plan_fingerprint TEXT NOT NULL,
        plan_json TEXT NOT NULL,
        deployment_environment TEXT NOT NULL,
        deployment_name TEXT NOT NULL,
        deployment_version INTEGER NOT NULL CHECK (deployment_version > 0),
        status TEXT NOT NULL,
        status_reason TEXT,
        status_message TEXT,
        max_active_provider_calls INTEGER NOT NULL
            CHECK (max_active_provider_calls > 0),
        max_active_gpu_provider_calls INTEGER NOT NULL
            CHECK (
                max_active_gpu_provider_calls >= 0
                AND max_active_gpu_provider_calls <= max_active_provider_calls
            ),
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        started_at INTEGER,
        completed_at INTEGER
    )
    """,
    """
    CREATE TABLE execution_nodes (
        execution_run_id TEXT NOT NULL
            REFERENCES execution_runs(execution_run_id) ON DELETE CASCADE,
        node_key TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        aggregation_policy TEXT NOT NULL,
        allow_empty_result INTEGER NOT NULL CHECK (allow_empty_result IN (0, 1)),
        status TEXT NOT NULL,
        status_reason TEXT,
        discovery_complete INTEGER NOT NULL DEFAULT 0
            CHECK (discovery_complete IN (0, 1)),
        result_observation TEXT,
        result_observed_at INTEGER,
        result_provenance TEXT,
        error_message TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        started_at INTEGER,
        completed_at INTEGER,
        PRIMARY KEY (execution_run_id, node_key),
        UNIQUE (execution_run_id, ordinal)
    )
    """,
    """
    CREATE TABLE execution_dispatch_batches (
        dispatch_batch_id TEXT PRIMARY KEY,
        execution_run_id TEXT NOT NULL,
        node_key TEXT NOT NULL,
        mode TEXT NOT NULL,
        compatibility_key TEXT NOT NULL,
        policy_json TEXT NOT NULL,
        claim_capacity INTEGER CHECK (claim_capacity > 0),
        created_at INTEGER NOT NULL,
        FOREIGN KEY (execution_run_id, node_key)
            REFERENCES execution_nodes(execution_run_id, node_key)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE execution_provider_calls (
        provider_call_id TEXT PRIMARY KEY,
        execution_run_id TEXT NOT NULL,
        node_key TEXT NOT NULL,
        dispatch_batch_id TEXT NOT NULL
            REFERENCES execution_dispatch_batches(dispatch_batch_id)
            ON DELETE CASCADE,
        submission_token TEXT NOT NULL,
        preclaim_json TEXT NOT NULL,
        dispatch_mode TEXT NOT NULL,
        provider_environment TEXT NOT NULL,
        provider_app_name TEXT NOT NULL,
        provider_app_version INTEGER NOT NULL
            CHECK (provider_app_version > 0),
        provider_function_name TEXT NOT NULL,
        uses_gpu INTEGER NOT NULL CHECK (uses_gpu IN (0, 1)),
        runtime_image_key TEXT,
        status TEXT NOT NULL,
        provider_call_handle_id TEXT UNIQUE,
        result_envelope_json TEXT,
        error_message TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        attached_at INTEGER,
        started_at INTEGER,
        completed_at INTEGER,
        UNIQUE (execution_run_id, submission_token),
        FOREIGN KEY (execution_run_id, node_key)
            REFERENCES execution_nodes(execution_run_id, node_key)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE execution_tasks (
        execution_run_id TEXT NOT NULL,
        node_key TEXT NOT NULL,
        task_key TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        fingerprint TEXT NOT NULL,
        scientific_payload_json TEXT NOT NULL,
        execution_payload_json TEXT NOT NULL,
        status TEXT NOT NULL,
        status_reason TEXT,
        result_observation TEXT,
        result_observed_at INTEGER,
        result_provenance TEXT,
        dispatch_batch_id TEXT
            REFERENCES execution_dispatch_batches(dispatch_batch_id),
        provider_call_id TEXT
            REFERENCES execution_provider_calls(provider_call_id),
        worker_provider_call_id TEXT
            REFERENCES execution_provider_calls(provider_call_id),
        local_owned INTEGER NOT NULL DEFAULT 0 CHECK (local_owned IN (0, 1)),
        error_message TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        started_at INTEGER,
        completed_at INTEGER,
        CHECK (
            (provider_call_id IS NULL OR worker_provider_call_id IS NULL)
            AND (
                local_owned = 0
                OR (
                    provider_call_id IS NULL
                    AND worker_provider_call_id IS NULL
                )
            )
        ),
        PRIMARY KEY (execution_run_id, node_key, task_key),
        UNIQUE (execution_run_id, node_key, ordinal),
        FOREIGN KEY (execution_run_id, node_key)
            REFERENCES execution_nodes(execution_run_id, node_key)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE execution_task_claim_requests (
        request_id TEXT PRIMARY KEY,
        provider_call_id TEXT NOT NULL
            REFERENCES execution_provider_calls(provider_call_id)
            ON DELETE CASCADE,
        capacity INTEGER NOT NULL CHECK (capacity > 0),
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE execution_worker_assignments (
        execution_run_id TEXT NOT NULL,
        node_key TEXT NOT NULL,
        task_key TEXT NOT NULL,
        provider_call_id TEXT NOT NULL
            REFERENCES execution_provider_calls(provider_call_id),
        request_id TEXT NOT NULL
            REFERENCES execution_task_claim_requests(request_id),
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        created_at INTEGER NOT NULL,
        PRIMARY KEY (execution_run_id, node_key, task_key),
        UNIQUE (request_id, ordinal),
        FOREIGN KEY (execution_run_id, node_key, task_key)
            REFERENCES execution_tasks(execution_run_id, node_key, task_key)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE execution_task_completion_requests (
        request_id TEXT PRIMARY KEY,
        provider_call_id TEXT NOT NULL
            REFERENCES execution_provider_calls(provider_call_id),
        task_key TEXT NOT NULL,
        observation TEXT NOT NULL,
        message TEXT,
        created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE execution_node_dependencies (
        execution_run_id TEXT NOT NULL,
        node_key TEXT NOT NULL,
        dependency_node_key TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        accept_partial INTEGER NOT NULL CHECK (accept_partial IN (0, 1)),
        PRIMARY KEY (execution_run_id, node_key, dependency_node_key),
        UNIQUE (execution_run_id, node_key, ordinal),
        FOREIGN KEY (execution_run_id, node_key)
            REFERENCES execution_nodes(execution_run_id, node_key)
            ON DELETE CASCADE,
        FOREIGN KEY (execution_run_id, dependency_node_key)
            REFERENCES execution_nodes(execution_run_id, node_key)
            ON DELETE CASCADE
    )
    """,
)


class SqliteExecutionRepository:
    """Persist execution state on a host-owned SQLite connection."""

    def __init__(self, connection: sqlite3.Connection):
        """Bind a connection without committing, closing, or selecting its path."""
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def initialize_schema(self) -> None:
        """Create the current schema or reject another recorded version."""
        schema_exists = self._connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'execution_schema'
            """
        ).fetchone()
        if schema_exists is not None:
            row = self._connection.execute(
                "SELECT version FROM execution_schema WHERE singleton = 1"
            ).fetchone()
            version = None if row is None else int(row["version"])
            if version != EXECUTION_SCHEMA_VERSION:
                raise UnsupportedExecutionSchemaVersionError(
                    f"Unsupported execution schema version {version}"
                )
            return

        for statement in _SCHEMA_STATEMENTS:
            self._connection.execute(statement)
        self._connection.execute(
            "INSERT INTO execution_schema (singleton, version) VALUES (1, ?)",
            (EXECUTION_SCHEMA_VERSION,),
        )

    def create_run(
        self,
        *,
        execution_run_id: UUID,
        plan: ExecutionPlan,
        deployment: DeploymentIdentity,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
        predecessor_execution_run_id: UUID | None = None,
        now: int,
    ) -> ExecutionRunRecord:
        """Persist one immutable plan and its initial pending Nodes."""
        _validate_call_limits(
            max_active_provider_calls,
            max_active_gpu_provider_calls,
        )
        plan_json = _dump_plan(plan)
        values = (
            str(execution_run_id),
            (
                None
                if predecessor_execution_run_id is None
                else str(predecessor_execution_run_id)
            ),
            plan.workload_name,
            plan.workload_run_key,
            plan.workload_plan_fingerprint,
            plan_json,
            deployment.environment,
            deployment.deployment_name,
            deployment.deployment_version,
            RunStatus.PENDING.value,
            max_active_provider_calls,
            max_active_gpu_provider_calls,
            now,
            now,
        )
        self._connection.execute(
            """
            INSERT INTO execution_runs (
                execution_run_id,
                predecessor_execution_run_id,
                workload_name,
                workload_run_key,
                workload_plan_fingerprint,
                plan_json,
                deployment_environment,
                deployment_name,
                deployment_version,
                status,
                max_active_provider_calls,
                max_active_gpu_provider_calls,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        for ordinal, node in enumerate(plan.nodes):
            self._connection.execute(
                """
                INSERT INTO execution_nodes (
                    execution_run_id,
                    node_key,
                    ordinal,
                    aggregation_policy,
                    allow_empty_result,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(execution_run_id),
                    node.node_key,
                    ordinal,
                    node.aggregation_policy.value,
                    int(node.allow_empty_result),
                    NodeStatus.PENDING.value,
                    now,
                    now,
                ),
            )
        for node in plan.nodes:
            for ordinal, dependency in enumerate(node.dependencies):
                self._connection.execute(
                    """
                    INSERT INTO execution_node_dependencies (
                        execution_run_id,
                        node_key,
                        dependency_node_key,
                        ordinal,
                        accept_partial
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(execution_run_id),
                        node.node_key,
                        dependency.node_key,
                        ordinal,
                        int(dependency.accept_partial),
                    ),
                )
        return self.get_run(execution_run_id)

    def create_successor_run(
        self,
        *,
        predecessor_execution_run_id: UUID,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
        now: int,
        plan: ExecutionPlan | None = None,
    ) -> ExecutionRunRecord:
        """Create an explicit compatible retry boundary with no copied state."""
        if execution_run_id == predecessor_execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
        predecessor = self.validate_successor_source(predecessor_execution_run_id)
        if (
            plan is not None
            and plan.workload_plan_fingerprint
            != predecessor.plan.workload_plan_fingerprint
        ):
            raise ValueError("Workload Plan Fingerprint does not match predecessor")
        return self.create_run(
            execution_run_id=execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            plan=predecessor.plan,
            deployment=deployment,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            now=now,
        )

    def validate_successor_source(
        self,
        predecessor_execution_run_id: UUID,
    ) -> ExecutionRunRecord:
        """Return a predecessor only when replacement work is conclusively safe."""
        predecessor = self.get_run(predecessor_execution_run_id)
        if not predecessor.status.is_terminal:
            raise ValueError("predecessor Run is not terminal")
        if self.active_provider_call_counts(predecessor_execution_run_id).total:
            raise ValueError("predecessor Run still has active Provider Calls")
        nodes = self.list_nodes(predecessor_execution_run_id)
        tasks = tuple(
            task
            for node in nodes
            for task in self.list_tasks(
                predecessor_execution_run_id,
                node.node_key,
            )
        )
        if not all(node.status.is_terminal for node in nodes) or not all(
            task.status.is_terminal for task in tasks
        ):
            raise ValueError("predecessor execution state is not conclusive")
        return predecessor

    def get_run(self, execution_run_id: UUID) -> ExecutionRunRecord:
        """Load one Execution Run by its opaque identity."""
        row = self._connection.execute(
            """
            SELECT *
            FROM execution_runs
            WHERE execution_run_id = ?
            """,
            (str(execution_run_id),),
        ).fetchone()
        if row is None:
            raise ExecutionRunNotFoundError(str(execution_run_id))
        return _run_from_row(row)

    def snapshot(self, execution_run_id: UUID) -> ExecutionSnapshot:
        """Return a common read-only execution view for host projections."""
        nodes = self.list_nodes(execution_run_id)
        tasks = tuple(
            task
            for node in nodes
            for task in self.list_tasks(execution_run_id, node.node_key)
        )
        return ExecutionSnapshot(
            run=self.get_run(execution_run_id),
            nodes=nodes,
            tasks=tasks,
            provider_calls=self.list_provider_calls(execution_run_id),
            active_provider_calls=self.active_provider_call_counts(execution_run_id),
        )

    def list_nodes(self, execution_run_id: UUID) -> tuple[ExecutionNodeRecord, ...]:
        """Load planned Nodes in their persisted encounter order."""
        rows = self._connection.execute(
            """
            SELECT *
            FROM execution_nodes
            WHERE execution_run_id = ?
            ORDER BY ordinal
            """,
            (str(execution_run_id),),
        ).fetchall()
        return tuple(self._node_from_row(row) for row in rows)

    def get_node(
        self,
        execution_run_id: UUID,
        node_key: str,
    ) -> ExecutionNodeRecord:
        """Load one planned Node."""
        row = self._connection.execute(
            """
            SELECT *
            FROM execution_nodes
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (str(execution_run_id), node_key),
        ).fetchone()
        if row is None:
            raise LookupError(f"Execution Node not found: {node_key}")
        return self._node_from_row(row)

    def start_node(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        now: int,
    ) -> ExecutionNodeRecord:
        """Move one pending Node into execution."""
        run = self.get_run(execution_run_id)
        if run.status == RunStatus.PENDING:
            self._transition_run(
                execution_run_id,
                RunStatus.RUNNING,
                reason=None,
                message=None,
                now=now,
                explicit_resume=False,
            )
        elif run.status != RunStatus.RUNNING:
            raise ValueError(f"cannot start a Node while Run is {run.status.value}")
        node = self.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.PENDING:
            raise ValueError(f"cannot start Node from {node.status.value}")
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                started_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (
                NodeStatus.RUNNING.value,
                now,
                now,
                str(execution_run_id),
                node_key,
            ),
        )
        return self.get_node(execution_run_id, node_key)

    def discover_tasks(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_plans: tuple[TaskPlan, ...],
        *,
        now: int,
    ) -> tuple[ExecutionTaskRecord, ...]:
        """Persist one Node's complete finite Task set exactly once."""
        node = self.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.RUNNING:
            raise ValueError(f"cannot discover Tasks for {node.status.value} Node")
        if node.discovery_complete:
            raise ValueError("Task discovery is already complete")

        seen: set[str] = set()
        prepared: list[tuple[TaskPlan, str, str, str]] = []
        plan_fingerprint = self.get_run(execution_run_id).plan.workload_plan_fingerprint
        for task_plan in task_plans:
            if not task_plan.task_key:
                raise ValueError("Task key cannot be empty")
            if task_plan.task_key in seen:
                raise ValueError(f"duplicate Task key {task_plan.task_key!r}")
            seen.add(task_plan.task_key)
            prepared.append((
                task_plan,
                task_plan.fingerprint(
                    workload_plan_fingerprint=plan_fingerprint,
                    node_key=node_key,
                ),
                _dump_json(task_plan.scientific_payload),
                _dump_json(task_plan.execution_payload),
            ))

        for ordinal, (
            task_plan,
            fingerprint,
            scientific_payload_json,
            execution_payload_json,
        ) in enumerate(prepared):
            self._connection.execute(
                """
                INSERT INTO execution_tasks (
                    execution_run_id,
                    node_key,
                    task_key,
                    ordinal,
                    fingerprint,
                    scientific_payload_json,
                    execution_payload_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(execution_run_id),
                    node_key,
                    task_plan.task_key,
                    ordinal,
                    fingerprint,
                    scientific_payload_json,
                    execution_payload_json,
                    TaskStatus.PENDING.value,
                    now,
                    now,
                ),
            )
        if prepared or node.allow_empty_result:
            status = NodeStatus.RUNNING
            error_message = None
            completed_at = None
        else:
            status = NodeStatus.FAILED
            error_message = "Node discovered no Tasks"
            completed_at = now
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET discovery_complete = 1,
                status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (
                status.value,
                error_message,
                completed_at,
                now,
                str(execution_run_id),
                node_key,
            ),
        )
        return self.list_tasks(execution_run_id, node_key)

    def record_node_result_observation(
        self,
        execution_run_id: UUID,
        node_key: str,
        observation: AvailabilityStatus,
        *,
        now: int,
    ) -> ExecutionNodeRecord:
        """Record caller-owned Node validation without storing its evidence."""
        node = self.get_node(execution_run_id, node_key)
        if node.status.is_terminal:
            raise ValueError(
                f"cannot record a result for terminal Node {node.status.value}"
            )
        status = node.status
        provenance: ResultProvenance | None = None
        completed_at = node.completed_at
        if observation == AvailabilityStatus.AVAILABLE:
            status = NodeStatus.SUCCEEDED
            provenance = (
                ResultProvenance.CACHE
                if node.status == NodeStatus.PENDING
                else ResultProvenance.CURRENT_RUN
            )
            completed_at = now
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                result_observation = ?,
                result_observed_at = ?,
                result_provenance = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (
                status.value,
                observation.value,
                now,
                None if provenance is None else provenance.value,
                completed_at,
                now,
                str(execution_run_id),
                node_key,
            ),
        )
        if observation == AvailabilityStatus.UNKNOWN:
            self._suspend_for_unknown_result(execution_run_id, now=now)
        return self.get_node(execution_run_id, node_key)

    def fail_node(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        message: str,
        now: int,
    ) -> ExecutionNodeRecord:
        """Fail caller-owned discovery or publication after ownership concludes."""
        if not message:
            raise ValueError("Node failure message cannot be empty")
        node = self.get_node(execution_run_id, node_key)
        if node.status == NodeStatus.FAILED and node.error_message == message:
            return node
        if node.status != NodeStatus.RUNNING:
            raise ValueError(f"cannot fail {node.status.value} Node")
        tasks = self.list_tasks(execution_run_id, node_key)
        if any(not task.status.is_terminal for task in tasks):
            raise ValueError("cannot fail Node while Tasks remain active")
        calls = [
            call
            for call in self.list_provider_calls(execution_run_id)
            if call.node_key == node_key and not call.status.is_terminal
        ]
        if calls:
            raise ValueError("cannot fail Node while Provider Calls remain active")
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET discovery_complete = 1,
                status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (
                NodeStatus.FAILED.value,
                message,
                now,
                now,
                str(execution_run_id),
                node_key,
            ),
        )
        return self.get_node(execution_run_id, node_key)

    def record_task_result_observation(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_key: str,
        observation: AvailabilityStatus,
        *,
        now: int,
    ) -> ExecutionTaskRecord:
        """Record caller-owned Task validation and any conclusive cache hit."""
        task = self.get_task(execution_run_id, node_key, task_key)
        if task.status.is_terminal:
            raise ValueError(
                f"cannot record a result for terminal Task {task.status.value}"
            )
        status = task.status
        provenance: ResultProvenance | None = None
        completed_at = task.completed_at
        if observation == AvailabilityStatus.AVAILABLE:
            status = TaskStatus.SUCCEEDED
            provenance = (
                ResultProvenance.CACHE
                if task.status == TaskStatus.PENDING
                else ResultProvenance.CURRENT_RUN
            )
            completed_at = now
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                result_observation = ?,
                result_observed_at = ?,
                result_provenance = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ? AND task_key = ?
            """,
            (
                status.value,
                observation.value,
                now,
                None if provenance is None else provenance.value,
                completed_at,
                now,
                str(execution_run_id),
                node_key,
                task_key,
            ),
        )
        if observation == AvailabilityStatus.UNKNOWN:
            self._suspend_for_unknown_result(execution_run_id, now=now)
        return self.get_task(execution_run_id, node_key, task_key)

    def preclaim_fixed_batch(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_keys: tuple[str, ...],
        *,
        submission_token: str,
        binding: ProviderBinding,
        compatibility_key: str,
        max_tasks_per_call: int = 1,
        now: int,
    ) -> ProviderCallPreclaim | None:
        """Atomically own one fixed Task batch and reserve one call slot."""
        if not submission_token:
            raise ValueError("submission token cannot be empty")
        if max_tasks_per_call <= 0:
            raise ValueError("max_tasks_per_call must be positive")
        if not task_keys:
            raise ValueError("fixed Provider Call batch cannot be empty")
        if len(task_keys) > max_tasks_per_call:
            raise ValueError("fixed Provider Call batch exceeds max_tasks_per_call")
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("fixed Provider Call batch contains duplicate Tasks")
        preclaim_json = _dump_json({
            "binding": _binding_json_value(binding),
            "compatibility_key": compatibility_key,
            "max_tasks_per_call": max_tasks_per_call,
            "node_key": node_key,
            "task_keys": task_keys,
        })
        existing = self._connection.execute(
            """
            SELECT *
            FROM execution_provider_calls
            WHERE execution_run_id = ? AND submission_token = ?
            """,
            (str(execution_run_id), submission_token),
        ).fetchone()
        if existing is not None:
            if existing["preclaim_json"] != preclaim_json:
                raise ValueError("submission token was reused for different work")
            return ProviderCallPreclaim(
                call=self._provider_call_from_row(existing),
                spawn_authorized=False,
            )

        run = self.get_run(execution_run_id)
        if run.status != RunStatus.RUNNING:
            return None
        node = self.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            raise ValueError("Provider Call Node is not ready for admission")
        counts = self.active_provider_call_counts(execution_run_id)
        if counts.total >= run.max_active_provider_calls:
            return None
        if binding.uses_gpu and counts.gpu >= run.max_active_gpu_provider_calls:
            return None

        tasks = [
            self.get_task(execution_run_id, node_key, task_key)
            for task_key in task_keys
        ]
        for task in tasks:
            if (
                task.status != TaskStatus.PENDING
                or task.result_observation != AvailabilityStatus.MISSING
                or task.provider_call_id is not None
                or task.worker_provider_call_id is not None
                or task.local_owned
            ):
                raise ValueError(
                    f"Task {task.task_key!r} is not ready for Provider Call ownership"
                )

        dispatch_batch_id = uuid4()
        provider_call_id = uuid4()
        self._connection.execute(
            """
            INSERT INTO execution_dispatch_batches (
                dispatch_batch_id,
                execution_run_id,
                node_key,
                mode,
                compatibility_key,
                policy_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(dispatch_batch_id),
                str(execution_run_id),
                node_key,
                DispatchMode.FIXED_BATCH.value,
                compatibility_key,
                preclaim_json,
                now,
            ),
        )
        self._connection.execute(
            """
            INSERT INTO execution_provider_calls (
                provider_call_id,
                execution_run_id,
                node_key,
                dispatch_batch_id,
                submission_token,
                preclaim_json,
                dispatch_mode,
                provider_environment,
                provider_app_name,
                provider_app_version,
                provider_function_name,
                uses_gpu,
                runtime_image_key,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(provider_call_id),
                str(execution_run_id),
                node_key,
                str(dispatch_batch_id),
                submission_token,
                preclaim_json,
                DispatchMode.FIXED_BATCH.value,
                binding.environment,
                binding.app_name,
                binding.app_version,
                binding.function_name,
                int(binding.uses_gpu),
                binding.runtime_image_key,
                ProviderCallStatus.SUBMITTING.value,
                now,
                now,
            ),
        )
        for task_key in task_keys:
            self._connection.execute(
                """
                UPDATE execution_tasks
                SET status = ?,
                    dispatch_batch_id = ?,
                    provider_call_id = ?,
                    started_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND task_key = ?
                """,
                (
                    TaskStatus.RUNNING.value,
                    str(dispatch_batch_id),
                    str(provider_call_id),
                    now,
                    now,
                    str(execution_run_id),
                    node_key,
                    task_key,
                ),
            )
        return ProviderCallPreclaim(
            call=self.get_provider_call(provider_call_id),
            spawn_authorized=True,
        )

    def preclaim_pull_worker(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        submission_token: str,
        binding: ProviderBinding,
        compatibility_key: str,
        claim_capacity: int,
        now: int,
    ) -> ProviderCallPreclaim | None:
        """Admit one derived pull-worker call without assigning Tasks yet."""
        if not submission_token:
            raise ValueError("submission token cannot be empty")
        if claim_capacity <= 0:
            raise ValueError("claim_capacity must be positive")
        policy_json = _dump_json({
            "binding": _binding_json_value(binding),
            "claim_capacity": claim_capacity,
            "compatibility_key": compatibility_key,
            "node_key": node_key,
        })
        preclaim_json = _dump_json({
            "mode": DispatchMode.PULL_WORKER.value,
            "policy": json.loads(policy_json),
            "submission_token": submission_token,
        })
        existing = self._connection.execute(
            """
            SELECT *
            FROM execution_provider_calls
            WHERE execution_run_id = ? AND submission_token = ?
            """,
            (str(execution_run_id), submission_token),
        ).fetchone()
        if existing is not None:
            if existing["preclaim_json"] != preclaim_json:
                raise ValueError("submission token was reused for different work")
            return ProviderCallPreclaim(
                call=self._provider_call_from_row(existing),
                spawn_authorized=False,
            )

        run = self.get_run(execution_run_id)
        if run.status != RunStatus.RUNNING:
            return None
        node = self.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            raise ValueError("Pull-worker Node is not ready for admission")
        invalid_task = self._connection.execute(
            """
            SELECT task_key
            FROM execution_tasks
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status = ?
                AND result_observation != ?
            LIMIT 1
            """,
            (
                str(execution_run_id),
                node_key,
                TaskStatus.PENDING.value,
                AvailabilityStatus.MISSING.value,
            ),
        ).fetchone()
        if invalid_task is not None:
            raise ValueError(
                f"Task {invalid_task['task_key']!r} was not cache-validated"
            )

        batch = self._connection.execute(
            """
            SELECT *
            FROM execution_dispatch_batches
            WHERE execution_run_id = ? AND node_key = ? AND mode = ?
            """,
            (
                str(execution_run_id),
                node_key,
                DispatchMode.PULL_WORKER.value,
            ),
        ).fetchone()
        if batch is not None and batch["policy_json"] != policy_json:
            raise ValueError("pull-worker policy cannot change within a Run")

        unfinished = self._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM execution_tasks
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status IN (?, ?)
            """,
            (
                str(execution_run_id),
                node_key,
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        ).fetchone()["count"]
        desired_workers = (unfinished + claim_capacity - 1) // claim_capacity
        existing_workers = (
            0
            if batch is None
            else self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM execution_provider_calls
                WHERE dispatch_batch_id = ?
                    AND status NOT IN (?, ?, ?)
                """,
                (
                    batch["dispatch_batch_id"],
                    ProviderCallStatus.SUCCEEDED.value,
                    ProviderCallStatus.FAILED.value,
                    ProviderCallStatus.CANCELLED.value,
                ),
            ).fetchone()["count"]
        )
        if existing_workers >= desired_workers:
            return None
        counts = self.active_provider_call_counts(execution_run_id)
        if counts.total >= run.max_active_provider_calls:
            return None
        if binding.uses_gpu and counts.gpu >= run.max_active_gpu_provider_calls:
            return None

        if batch is None:
            dispatch_batch_id = uuid4()
            self._connection.execute(
                """
                INSERT INTO execution_dispatch_batches (
                    dispatch_batch_id,
                    execution_run_id,
                    node_key,
                    mode,
                    compatibility_key,
                    policy_json,
                    claim_capacity,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(dispatch_batch_id),
                    str(execution_run_id),
                    node_key,
                    DispatchMode.PULL_WORKER.value,
                    compatibility_key,
                    policy_json,
                    claim_capacity,
                    now,
                ),
            )
        else:
            dispatch_batch_id = UUID(batch["dispatch_batch_id"])

        provider_call_id = uuid4()
        self._connection.execute(
            """
            INSERT INTO execution_provider_calls (
                provider_call_id,
                execution_run_id,
                node_key,
                dispatch_batch_id,
                submission_token,
                preclaim_json,
                dispatch_mode,
                provider_environment,
                provider_app_name,
                provider_app_version,
                provider_function_name,
                uses_gpu,
                runtime_image_key,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(provider_call_id),
                str(execution_run_id),
                node_key,
                str(dispatch_batch_id),
                submission_token,
                preclaim_json,
                DispatchMode.PULL_WORKER.value,
                binding.environment,
                binding.app_name,
                binding.app_version,
                binding.function_name,
                int(binding.uses_gpu),
                binding.runtime_image_key,
                ProviderCallStatus.SUBMITTING.value,
                now,
                now,
            ),
        )
        return ProviderCallPreclaim(
            call=self.get_provider_call(provider_call_id),
            spawn_authorized=True,
        )

    def claim_pull_tasks(
        self,
        provider_call_id: UUID,
        *,
        request_id: str,
        capacity: int,
        now: int,
    ) -> PullTaskClaim:
        """Checkpoint an ordered Task microbatch before returning its payloads."""
        if not request_id:
            raise ValueError("claim request ID cannot be empty")
        if capacity <= 0:
            raise ValueError("claim capacity must be positive")
        call = self.get_provider_call(provider_call_id)
        existing = self._connection.execute(
            """
            SELECT provider_call_id, capacity
            FROM execution_task_claim_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["provider_call_id"] != str(provider_call_id)
                or existing["capacity"] != capacity
            ):
                raise ValueError("claim request ID was reused for another claim")
            return self._load_pull_task_claim(request_id)
        if call.dispatch_mode != DispatchMode.PULL_WORKER:
            raise ValueError("Provider Call is not a pull worker")
        if call.status not in {
            ProviderCallStatus.SUBMITTING,
            ProviderCallStatus.ATTACHED,
            ProviderCallStatus.RUNNING,
            ProviderCallStatus.OUTCOME_UNKNOWN,
            ProviderCallStatus.STATE_UNKNOWN,
        }:
            raise ValueError(f"cannot claim Tasks for {call.status.value} worker")
        batch = self._connection.execute(
            """
            SELECT claim_capacity
            FROM execution_dispatch_batches
            WHERE dispatch_batch_id = ?
            """,
            (str(call.dispatch_batch_id),),
        ).fetchone()
        if capacity > batch["claim_capacity"]:
            raise ValueError("claim capacity exceeds the pull-worker policy")

        self._connection.execute(
            """
            INSERT INTO execution_task_claim_requests (
                request_id,
                provider_call_id,
                capacity,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (request_id, str(provider_call_id), capacity, now),
        )
        rows = self._connection.execute(
            """
            SELECT task_key
            FROM execution_tasks
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status = ?
                AND result_observation = ?
                AND provider_call_id IS NULL
                AND worker_provider_call_id IS NULL
                AND local_owned = 0
            ORDER BY ordinal
            LIMIT ?
            """,
            (
                str(call.execution_run_id),
                call.node_key,
                TaskStatus.PENDING.value,
                AvailabilityStatus.MISSING.value,
                capacity,
            ),
        ).fetchall()
        for ordinal, row in enumerate(rows):
            self._connection.execute(
                """
                INSERT INTO execution_worker_assignments (
                    execution_run_id,
                    node_key,
                    task_key,
                    provider_call_id,
                    request_id,
                    ordinal,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(call.execution_run_id),
                    call.node_key,
                    row["task_key"],
                    str(provider_call_id),
                    request_id,
                    ordinal,
                    now,
                ),
            )
            self._connection.execute(
                """
                UPDATE execution_tasks
                SET status = ?,
                    worker_provider_call_id = ?,
                    started_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND task_key = ?
                """,
                (
                    TaskStatus.RUNNING.value,
                    str(provider_call_id),
                    now,
                    now,
                    str(call.execution_run_id),
                    call.node_key,
                    row["task_key"],
                ),
            )
        return self._load_pull_task_claim(request_id)

    def record_pull_task_completion(
        self,
        provider_call_id: UUID,
        task_key: str,
        *,
        request_id: str,
        observation: AvailabilityStatus,
        message: str | None = None,
        now: int,
    ) -> ExecutionTaskRecord:
        """Apply one idempotent pull-worker completion after publication."""
        if not request_id:
            raise ValueError("completion request ID cannot be empty")
        call = self.get_provider_call(provider_call_id)
        existing = self._connection.execute(
            """
            SELECT provider_call_id, task_key, observation, message
            FROM execution_task_completion_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["provider_call_id"] != str(provider_call_id)
                or existing["task_key"] != task_key
                or existing["observation"] != observation.value
                or existing["message"] != message
            ):
                raise ValueError("completion request ID was reused")
            return self.get_task(call.execution_run_id, call.node_key, task_key)
        assignment = self._connection.execute(
            """
            SELECT 1
            FROM execution_worker_assignments
            WHERE execution_run_id = ?
                AND node_key = ?
                AND task_key = ?
                AND provider_call_id = ?
            """,
            (
                str(call.execution_run_id),
                call.node_key,
                task_key,
                str(provider_call_id),
            ),
        ).fetchone()
        if assignment is None:
            raise ValueError("Task is not assigned to this Provider Call")
        self._connection.execute(
            """
            INSERT INTO execution_task_completion_requests (
                request_id,
                provider_call_id,
                task_key,
                observation,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                str(provider_call_id),
                task_key,
                observation.value,
                message,
                now,
            ),
        )
        if observation == AvailabilityStatus.MISSING:
            return self.fail_task(
                call.execution_run_id,
                call.node_key,
                task_key,
                message=message or "Worker publication was missing",
                now=now,
            )
        return self.record_task_result_observation(
            call.execution_run_id,
            call.node_key,
            task_key,
            observation,
            now=now,
        )

    def acquire_local_task(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_key: str,
        *,
        now: int,
    ) -> bool:
        """Acquire or recover publication-gated coordinator-local work."""
        run = self.get_run(execution_run_id)
        if run.status != RunStatus.RUNNING:
            return False
        task = self.get_task(execution_run_id, node_key, task_key)
        if task.status.is_terminal:
            return False
        if task.result_observation != AvailabilityStatus.MISSING:
            return False
        if task.status == TaskStatus.RUNNING:
            return (
                task.local_owned
                and task.provider_call_id is None
                and task.worker_provider_call_id is None
            )
        if (
            task.status != TaskStatus.PENDING
            or task.provider_call_id is not None
            or task.worker_provider_call_id is not None
        ):
            return False
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                local_owned = 1,
                started_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ? AND task_key = ?
            """,
            (
                TaskStatus.RUNNING.value,
                now,
                now,
                str(execution_run_id),
                node_key,
                task_key,
            ),
        )
        return True

    def fail_task(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_key: str,
        *,
        message: str,
        now: int,
    ) -> ExecutionTaskRecord:
        """Record one conclusive workload or publication failure."""
        task = self.get_task(execution_run_id, node_key, task_key)
        if task.status == TaskStatus.FAILED:
            return task
        if task.status.is_terminal:
            raise ValueError(f"cannot fail terminal Task {task.status.value}")
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ? AND task_key = ?
            """,
            (
                TaskStatus.FAILED.value,
                message,
                now,
                now,
                str(execution_run_id),
                node_key,
                task_key,
            ),
        )
        return self.get_task(execution_run_id, node_key, task_key)

    def reconcile_node_tasks(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        now: int,
    ) -> ExecutionNodeRecord:
        """Apply fail-fast admission stopping and terminal Task aggregation."""
        node = self.get_node(execution_run_id, node_key)
        if node.status.is_terminal:
            return node
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            raise ValueError("Node Tasks cannot be aggregated before discovery")
        self.apply_task_failure_policy(
            execution_run_id,
            node_key,
            now=now,
        )
        tasks = self.list_tasks(execution_run_id, node_key)
        outcome = aggregate_task_outcome(
            node.aggregation_policy,
            tuple(task.status for task in tasks),
        )
        if outcome is None:
            return self.get_node(execution_run_id, node_key)
        error_message = (
            "One or more Tasks failed" if outcome == NodeStatus.FAILED else None
        )
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND node_key = ?
            """,
            (
                outcome.value,
                error_message,
                now,
                now,
                str(execution_run_id),
                node_key,
            ),
        )
        return self.get_node(execution_run_id, node_key)

    def apply_task_failure_policy(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        now: int,
    ) -> tuple[str, ...]:
        """Stop unowned sibling admission after a fail-fast Task failure."""
        node = self.get_node(execution_run_id, node_key)
        if node.status != NodeStatus.RUNNING or not node.discovery_complete:
            raise ValueError("Node Task policy cannot run before discovery")
        tasks = self.list_tasks(execution_run_id, node_key)
        if node.aggregation_policy != NodeAggregationPolicy.FAIL_FAST or not any(
            task.status == TaskStatus.FAILED for task in tasks
        ):
            return ()
        skipped = tuple(
            task.task_key
            for task in tasks
            if task.status == TaskStatus.PENDING
            and task.provider_call_id is None
            and task.worker_provider_call_id is None
            and not task.local_owned
        )
        if not skipped:
            return ()
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status = ?
                AND provider_call_id IS NULL
                AND worker_provider_call_id IS NULL
                AND local_owned = 0
            """,
            (
                TaskStatus.SKIPPED.value,
                now,
                now,
                str(execution_run_id),
                node_key,
                TaskStatus.PENDING.value,
            ),
        )
        return skipped

    def skip_unreachable_nodes(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> tuple[ExecutionNodeRecord, ...]:
        """Persist dependency-derived skips in immutable plan order."""
        run = self.get_run(execution_run_id)
        if run.status != RunStatus.RUNNING:
            return ()
        nodes = self.list_nodes(execution_run_id)
        skipped_keys = propagated_skip_node_keys(
            run.plan,
            {node.node_key: node.status for node in nodes},
        )
        for node_key in skipped_keys:
            self._connection.execute(
                """
                UPDATE execution_nodes
                SET status = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND status = ?
                """,
                (
                    NodeStatus.SKIPPED.value,
                    now,
                    now,
                    str(execution_run_id),
                    node_key,
                    NodeStatus.PENDING.value,
                ),
            )
        return tuple(
            self.get_node(execution_run_id, node_key) for node_key in skipped_keys
        )

    def prune_unrequired_nodes(
        self,
        execution_run_id: UUID,
        *,
        required_node_keys: set[str],
        now: int,
    ) -> tuple[UUID, ...]:
        """Prune unnecessary work and return calls needing cancellation."""
        active_calls: list[UUID] = []
        for node in self.list_nodes(execution_run_id):
            if node.node_key in required_node_keys or node.status.is_terminal:
                continue
            if node.status == NodeStatus.PENDING:
                self._connection.execute(
                    """
                    UPDATE execution_nodes
                    SET status = ?,
                        status_reason = ?,
                        completed_at = ?,
                        updated_at = ?
                    WHERE execution_run_id = ? AND node_key = ?
                    """,
                    (
                        NodeStatus.SKIPPED.value,
                        WorkStatusReason.RESULT_ALREADY_SATISFIED.value,
                        now,
                        now,
                        str(execution_run_id),
                        node.node_key,
                    ),
                )
                continue

            self._connection.execute(
                """
                UPDATE execution_tasks
                SET status = ?,
                    status_reason = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND status = ?
                    AND provider_call_id IS NULL
                    AND local_owned = 0
                """,
                (
                    TaskStatus.SKIPPED.value,
                    WorkStatusReason.RESULT_ALREADY_SATISFIED.value,
                    now,
                    now,
                    str(execution_run_id),
                    node.node_key,
                    TaskStatus.PENDING.value,
                ),
            )
            self._connection.execute(
                """
                UPDATE execution_tasks
                SET status = ?,
                    status_reason = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND status = ?
                    AND (
                        local_owned = 1
                        OR provider_call_id IN (
                            SELECT provider_call_id
                            FROM execution_provider_calls
                            WHERE status IN (?, ?, ?)
                        )
                        OR worker_provider_call_id IN (
                            SELECT provider_call_id
                            FROM execution_provider_calls
                            WHERE status IN (?, ?, ?)
                        )
                    )
                """,
                (
                    TaskStatus.CANCELLED.value,
                    WorkStatusReason.RESULT_ALREADY_SATISFIED.value,
                    now,
                    now,
                    str(execution_run_id),
                    node.node_key,
                    TaskStatus.RUNNING.value,
                    ProviderCallStatus.SUCCEEDED.value,
                    ProviderCallStatus.FAILED.value,
                    ProviderCallStatus.CANCELLED.value,
                    ProviderCallStatus.SUCCEEDED.value,
                    ProviderCallStatus.FAILED.value,
                    ProviderCallStatus.CANCELLED.value,
                ),
            )
            rows = self._connection.execute(
                """
                SELECT provider_call_id
                FROM execution_provider_calls
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND status NOT IN (?, ?, ?)
                ORDER BY created_at, rowid
                """,
                (
                    str(execution_run_id),
                    node.node_key,
                    ProviderCallStatus.SUCCEEDED.value,
                    ProviderCallStatus.FAILED.value,
                    ProviderCallStatus.CANCELLED.value,
                ),
            ).fetchall()
            active_calls.extend(UUID(row["provider_call_id"]) for row in rows)
            self._complete_pruned_node_if_conclusive(
                execution_run_id,
                node.node_key,
                now=now,
            )
        return tuple(active_calls)

    def cancel_pruned_provider_call(
        self,
        provider_call_id: UUID,
        *,
        now: int,
    ) -> ProviderCallRecord:
        """Record conclusive cancellation during result-driven pruning."""
        call = self.cancel_provider_call(
            provider_call_id,
            message="Result was already satisfied",
            now=now,
        )
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status_reason = ?
            WHERE (provider_call_id = ? OR worker_provider_call_id = ?)
                AND status = ?
            """,
            (
                WorkStatusReason.RESULT_ALREADY_SATISFIED.value,
                str(provider_call_id),
                str(provider_call_id),
                TaskStatus.CANCELLED.value,
            ),
        )
        self._complete_pruned_node_if_conclusive(
            call.execution_run_id,
            call.node_key,
            now=now,
        )
        return self.get_provider_call(provider_call_id)

    def finalize_run_from_results(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> ExecutionRunRecord:
        """Persist the strict terminal-boundary outcome after owner cleanup."""
        run = self.get_run(execution_run_id)
        if run.status.is_terminal:
            return run
        nodes = self.list_nodes(execution_run_id)
        if not all(node.status.is_terminal for node in nodes):
            return run
        if self.active_provider_call_counts(execution_run_id).total:
            return run
        outcome = terminal_run_outcome(
            run.plan,
            {node.node_key: node.status for node in nodes},
        )
        if outcome is None:
            return run
        if run.status == RunStatus.PENDING:
            run = self._transition_run(
                execution_run_id,
                RunStatus.RUNNING,
                reason=None,
                message=None,
                now=now,
                explicit_resume=False,
            )
        if run.status not in {
            RunStatus.RUNNING,
            RunStatus.CANCEL_REQUESTED,
            RunStatus.STATE_UNKNOWN,
        }:
            return run
        return self._transition_run(
            execution_run_id,
            outcome,
            reason=(
                RunStatusReason.REQUIRED_WORK_FAILED
                if outcome == RunStatus.FAILED
                else None
            ),
            message=(
                "Required scientific work failed"
                if outcome == RunStatus.FAILED
                else None
            ),
            now=now,
            explicit_resume=False,
        )

    def request_run_cancellation(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> tuple[UUID, ...]:
        """Durably stop admission and identify remote owners to cancel."""
        run = self.get_run(execution_run_id)
        if run.status.is_terminal:
            return ()
        if run.status != RunStatus.CANCEL_REQUESTED:
            self._transition_run(
                execution_run_id,
                RunStatus.CANCEL_REQUESTED,
                reason=None,
                message=None,
                now=now,
                explicit_resume=False,
            )
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                status_reason = NULL,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ?
                AND (
                    (
                        status = ?
                        AND provider_call_id IS NULL
                        AND worker_provider_call_id IS NULL
                        AND local_owned = 0
                    )
                    OR (status = ? AND local_owned = 1)
                )
            """,
            (
                TaskStatus.CANCELLED.value,
                now,
                now,
                str(execution_run_id),
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        )
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                status_reason = NULL,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ? AND status = ?
            """,
            (
                NodeStatus.CANCELLED.value,
                now,
                now,
                str(execution_run_id),
                NodeStatus.PENDING.value,
            ),
        )
        for node in self.list_nodes(execution_run_id):
            if node.status == NodeStatus.RUNNING:
                self._complete_cancelled_node_if_conclusive(
                    execution_run_id,
                    node.node_key,
                    now=now,
                )
        rows = self._connection.execute(
            """
            SELECT provider_call_id
            FROM execution_provider_calls
            WHERE execution_run_id = ?
                AND status NOT IN (?, ?, ?)
            ORDER BY created_at, rowid
            """,
            (
                str(execution_run_id),
                ProviderCallStatus.SUCCEEDED.value,
                ProviderCallStatus.FAILED.value,
                ProviderCallStatus.CANCELLED.value,
            ),
        ).fetchall()
        return tuple(UUID(row["provider_call_id"]) for row in rows)

    def mark_provider_cancellation_unknown(
        self,
        provider_call_id: UUID,
        *,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        """Preserve ownership when a cancellation outcome is inconclusive."""
        call = self.get_provider_call(provider_call_id)
        if call.status in {
            ProviderCallStatus.SUBMITTING,
            ProviderCallStatus.OUTCOME_UNKNOWN,
        }:
            if call.status == ProviderCallStatus.SUBMITTING:
                self._set_provider_call_status(
                    provider_call_id,
                    ProviderCallStatus.OUTCOME_UNKNOWN,
                    message=message,
                    now=now,
                )
        elif call.status in {
            ProviderCallStatus.ATTACHED,
            ProviderCallStatus.RUNNING,
            ProviderCallStatus.STATE_UNKNOWN,
        }:
            if call.status != ProviderCallStatus.STATE_UNKNOWN:
                self._set_provider_call_status(
                    provider_call_id,
                    ProviderCallStatus.STATE_UNKNOWN,
                    message=message,
                    now=now,
                )
        elif call.status.is_terminal:
            return call
        self._project_run_unknown(
            call.execution_run_id,
            reason=RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN,
            message=message,
            now=now,
        )
        return self.get_provider_call(provider_call_id)

    def active_provider_call_counts(
        self,
        execution_run_id: UUID,
    ) -> ActiveProviderCallCounts:
        """Derive occupied total and GPU slots from nonterminal calls."""
        terminal = tuple(
            status.value for status in ProviderCallStatus if status.is_terminal
        )
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(uses_gpu), 0) AS gpu
            FROM execution_provider_calls
            WHERE execution_run_id = ?
                AND status NOT IN (?, ?, ?)
            """,
            (str(execution_run_id), *terminal),
        ).fetchone()
        return ActiveProviderCallCounts(total=row["total"], gpu=row["gpu"])

    def list_provider_calls(
        self,
        execution_run_id: UUID,
    ) -> tuple[ProviderCallRecord, ...]:
        """Load Provider Calls in durable preclaim order."""
        rows = self._connection.execute(
            """
            SELECT *
            FROM execution_provider_calls
            WHERE execution_run_id = ?
            ORDER BY created_at, rowid
            """,
            (str(execution_run_id),),
        ).fetchall()
        return tuple(self._provider_call_from_row(row) for row in rows)

    def get_provider_call(self, provider_call_id: UUID) -> ProviderCallRecord:
        """Load one durable Provider Call."""
        row = self._connection.execute(
            """
            SELECT *
            FROM execution_provider_calls
            WHERE provider_call_id = ?
            """,
            (str(provider_call_id),),
        ).fetchone()
        if row is None:
            raise LookupError(f"Provider Call not found: {provider_call_id}")
        return self._provider_call_from_row(row)

    def attach_provider_call(
        self,
        provider_call_id: UUID,
        *,
        provider_call_handle_id: str,
        now: int,
    ) -> ProviderCallRecord:
        """Durably attach the provider's concrete call identity."""
        if not provider_call_handle_id:
            raise ValueError("provider call handle ID cannot be empty")
        call = self.get_provider_call(provider_call_id)
        if (
            call.status == ProviderCallStatus.ATTACHED
            and call.provider_call_handle_id == provider_call_handle_id
        ):
            return call
        if call.status not in {
            ProviderCallStatus.SUBMITTING,
            ProviderCallStatus.OUTCOME_UNKNOWN,
        }:
            raise ValueError(f"cannot attach {call.status.value} Provider Call")
        self._connection.execute(
            """
            UPDATE execution_provider_calls
            SET status = ?,
                provider_call_handle_id = ?,
                attached_at = ?,
                updated_at = ?
            WHERE provider_call_id = ?
            """,
            (
                ProviderCallStatus.ATTACHED.value,
                provider_call_handle_id,
                now,
                now,
                str(provider_call_id),
            ),
        )
        self._reconcile_run_unknown(call.execution_run_id, now=now)
        return self.get_provider_call(provider_call_id)

    def mark_provider_call_running(
        self,
        provider_call_id: UUID,
        *,
        now: int,
    ) -> ProviderCallRecord:
        """Record a conclusive active provider observation."""
        call = self.get_provider_call(provider_call_id)
        if call.status == ProviderCallStatus.RUNNING:
            return call
        if call.status not in {
            ProviderCallStatus.ATTACHED,
            ProviderCallStatus.STATE_UNKNOWN,
        }:
            raise ValueError(f"cannot mark {call.status.value} Provider Call running")
        self._connection.execute(
            """
            UPDATE execution_provider_calls
            SET status = ?,
                started_at = COALESCE(started_at, ?),
                updated_at = ?
            WHERE provider_call_id = ?
            """,
            (
                ProviderCallStatus.RUNNING.value,
                now,
                now,
                str(provider_call_id),
            ),
        )
        self._reconcile_run_unknown(call.execution_run_id, now=now)
        return self.get_provider_call(provider_call_id)

    def mark_submission_outcome_unknown(
        self,
        provider_call_id: UUID,
        *,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        """Preserve ownership when spawn may have occurred without attachment."""
        call = self.get_provider_call(provider_call_id)
        if call.status == ProviderCallStatus.OUTCOME_UNKNOWN:
            return call
        if call.status != ProviderCallStatus.SUBMITTING:
            raise ValueError(
                f"cannot mark {call.status.value} submission outcome unknown"
            )
        self._set_provider_call_status(
            provider_call_id,
            ProviderCallStatus.OUTCOME_UNKNOWN,
            message=message,
            now=now,
        )
        self._project_run_unknown(
            call.execution_run_id,
            reason=RunStatusReason.SUBMISSION_OUTCOME_UNKNOWN,
            message=message,
            now=now,
        )
        return self.get_provider_call(provider_call_id)

    def mark_provider_call_state_unknown(
        self,
        provider_call_id: UUID,
        *,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        """Preserve attached ownership after an inconclusive provider lookup."""
        call = self.get_provider_call(provider_call_id)
        if call.status == ProviderCallStatus.STATE_UNKNOWN:
            return call
        if call.status not in {
            ProviderCallStatus.ATTACHED,
            ProviderCallStatus.RUNNING,
        }:
            raise ValueError(
                f"cannot mark {call.status.value} Provider Call state unknown"
            )
        self._set_provider_call_status(
            provider_call_id,
            ProviderCallStatus.STATE_UNKNOWN,
            message=message,
            now=now,
        )
        self._project_run_unknown(
            call.execution_run_id,
            reason=RunStatusReason.PROVIDER_OUTCOME_UNKNOWN,
            message=message,
            now=now,
        )
        return self.get_provider_call(provider_call_id)

    def record_provider_call_result(
        self,
        provider_call_id: UUID,
        *,
        result_envelope: Any,
        now: int,
    ) -> ProviderCallRecord:
        """Persist a Result Envelope before releasing the call's slots."""
        if result_envelope is None:
            raise ValueError("Provider Call success requires a Result Envelope")
        envelope_json = _dump_json(result_envelope)
        call = self.get_provider_call(provider_call_id)
        if call.status == ProviderCallStatus.SUCCEEDED:
            if call.result_envelope != result_envelope:
                raise ValueError("Provider Call Result Envelope cannot be replaced")
            return call
        if call.status not in {
            ProviderCallStatus.ATTACHED,
            ProviderCallStatus.RUNNING,
            ProviderCallStatus.OUTCOME_UNKNOWN,
            ProviderCallStatus.STATE_UNKNOWN,
        }:
            raise ValueError(
                f"cannot complete {call.status.value} Provider Call successfully"
            )
        self._connection.execute(
            """
            UPDATE execution_provider_calls
            SET status = ?,
                result_envelope_json = ?,
                error_message = NULL,
                completed_at = ?,
                updated_at = ?
            WHERE provider_call_id = ?
            """,
            (
                ProviderCallStatus.SUCCEEDED.value,
                envelope_json,
                now,
                now,
                str(provider_call_id),
            ),
        )
        if call.dispatch_mode == DispatchMode.PULL_WORKER:
            self._connection.execute(
                """
                UPDATE execution_tasks
                SET status = ?,
                    error_message = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE execution_run_id = ?
                    AND node_key = ?
                    AND worker_provider_call_id = ?
                    AND status IN (?, ?)
                """,
                (
                    TaskStatus.FAILED.value,
                    "Pull worker returned before reporting Task completion",
                    now,
                    now,
                    str(call.execution_run_id),
                    call.node_key,
                    str(provider_call_id),
                    TaskStatus.PENDING.value,
                    TaskStatus.RUNNING.value,
                ),
            )
        self._reconcile_run_unknown(call.execution_run_id, now=now)
        return self.get_provider_call(provider_call_id)

    def fail_provider_call(
        self,
        provider_call_id: UUID,
        *,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        """Record conclusive call failure and fail unfinished owned Tasks."""
        return self._finish_provider_call(
            provider_call_id,
            call_status=ProviderCallStatus.FAILED,
            task_status=TaskStatus.FAILED,
            message=message,
            now=now,
        )

    def cancel_provider_call(
        self,
        provider_call_id: UUID,
        *,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        """Record conclusive cancellation and cancel unfinished owned Tasks."""
        call = self.get_provider_call(provider_call_id)
        if call.status == ProviderCallStatus.SUBMITTING:
            raise ValueError("cannot cancel submitting Provider Call")
        return self._finish_provider_call(
            provider_call_id,
            call_status=ProviderCallStatus.CANCELLED,
            task_status=TaskStatus.CANCELLED,
            message=message,
            now=now,
        )

    def list_tasks(
        self,
        execution_run_id: UUID,
        node_key: str,
    ) -> tuple[ExecutionTaskRecord, ...]:
        """Load one Node's Tasks in persisted encounter order."""
        rows = self._connection.execute(
            """
            SELECT *
            FROM execution_tasks
            WHERE execution_run_id = ? AND node_key = ?
            ORDER BY ordinal
            """,
            (str(execution_run_id), node_key),
        ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def get_task(
        self,
        execution_run_id: UUID,
        node_key: str,
        task_key: str,
    ) -> ExecutionTaskRecord:
        """Load one discovered Task."""
        row = self._connection.execute(
            """
            SELECT *
            FROM execution_tasks
            WHERE execution_run_id = ? AND node_key = ? AND task_key = ?
            """,
            (str(execution_run_id), node_key, task_key),
        ).fetchone()
        if row is None:
            raise LookupError(f"Execution Task not found: {node_key}/{task_key}")
        return _task_from_row(row)

    def transition_run(
        self,
        execution_run_id: UUID,
        status: RunStatus,
        *,
        reason: RunStatusReason | None = None,
        message: str | None = None,
        now: int,
    ) -> ExecutionRunRecord:
        """Apply one legal Run transition and atomically replace diagnostics."""
        return self._transition_run(
            execution_run_id,
            status,
            reason=reason,
            message=message,
            now=now,
            explicit_resume=False,
        )

    def resume_run(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> ExecutionRunRecord:
        """Explicitly resume one suspended coordinator Run."""
        current = self.get_run(execution_run_id)
        if current.status != RunStatus.SUSPENDED:
            raise ValueError("only a suspended Run can be explicitly resumed")
        return self._transition_run(
            execution_run_id,
            RunStatus.RUNNING,
            reason=None,
            message=None,
            now=now,
            explicit_resume=True,
        )

    def _transition_run(
        self,
        execution_run_id: UUID,
        status: RunStatus,
        *,
        reason: RunStatusReason | None,
        message: str | None,
        now: int,
        explicit_resume: bool,
    ) -> ExecutionRunRecord:
        _validate_run_reason(status, reason)
        current = self.get_run(execution_run_id)
        if current.status.is_terminal:
            raise ValueError(f"cannot transition terminal Run {current.status.value}")
        if (
            current.status == RunStatus.SUSPENDED
            and status == RunStatus.RUNNING
            and not explicit_resume
        ):
            raise ValueError("suspended Run requires explicit resume")
        if status not in _RUN_TRANSITIONS[current.status]:
            raise ValueError(
                f"cannot transition Run from {current.status.value} to {status.value}"
            )
        started_at = (
            now
            if status == RunStatus.RUNNING and current.started_at is None
            else current.started_at
        )
        completed_at = now if status.is_terminal else None
        self._connection.execute(
            """
            UPDATE execution_runs
            SET status = ?,
                status_reason = ?,
                status_message = ?,
                updated_at = ?,
                started_at = ?,
                completed_at = ?
            WHERE execution_run_id = ?
            """,
            (
                status.value,
                None if reason is None else reason.value,
                message,
                now,
                started_at,
                completed_at,
                str(execution_run_id),
            ),
        )
        return self.get_run(execution_run_id)

    def _suspend_for_unknown_result(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> None:
        run = self.get_run(execution_run_id)
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise ValueError(
                f"cannot suspend result validation while Run is {run.status.value}"
            )
        self._transition_run(
            execution_run_id,
            RunStatus.SUSPENDED,
            reason=RunStatusReason.RESULT_VALIDATION_UNKNOWN,
            message="Workload result validation was inconclusive",
            now=now,
            explicit_resume=False,
        )

    def _provider_call_from_row(self, row: sqlite3.Row) -> ProviderCallRecord:
        task_rows = self._connection.execute(
            """
            SELECT task_key
            FROM execution_tasks
            WHERE provider_call_id = ? OR worker_provider_call_id = ?
            ORDER BY ordinal
            """,
            (row["provider_call_id"], row["provider_call_id"]),
        ).fetchall()
        return ProviderCallRecord(
            provider_call_id=UUID(row["provider_call_id"]),
            execution_run_id=UUID(row["execution_run_id"]),
            node_key=row["node_key"],
            dispatch_batch_id=UUID(row["dispatch_batch_id"]),
            dispatch_mode=DispatchMode(row["dispatch_mode"]),
            submission_token=row["submission_token"],
            binding=ProviderBinding(
                environment=row["provider_environment"],
                app_name=row["provider_app_name"],
                app_version=row["provider_app_version"],
                function_name=row["provider_function_name"],
                uses_gpu=bool(row["uses_gpu"]),
                runtime_image_key=row["runtime_image_key"],
            ),
            status=ProviderCallStatus(row["status"]),
            provider_call_handle_id=row["provider_call_handle_id"],
            result_envelope=(
                None
                if row["result_envelope_json"] is None
                else json.loads(row["result_envelope_json"])
            ),
            error_message=row["error_message"],
            task_keys=tuple(task["task_key"] for task in task_rows),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            attached_at=row["attached_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def _load_pull_task_claim(self, request_id: str) -> PullTaskClaim:
        request = self._connection.execute(
            """
            SELECT provider_call_id
            FROM execution_task_claim_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if request is None:
            raise LookupError(f"Task claim request not found: {request_id}")
        rows = self._connection.execute(
            """
            SELECT
                assignment.execution_run_id,
                assignment.node_key,
                assignment.task_key,
                assignment.provider_call_id,
                assignment.request_id,
                assignment.ordinal,
                assignment.created_at,
                task.fingerprint,
                task.execution_payload_json
            FROM execution_worker_assignments AS assignment
            JOIN execution_tasks AS task
                ON task.execution_run_id = assignment.execution_run_id
                AND task.node_key = assignment.node_key
                AND task.task_key = assignment.task_key
            WHERE assignment.request_id = ?
            ORDER BY assignment.ordinal
            """,
            (request_id,),
        ).fetchall()
        return PullTaskClaim(
            request_id=request_id,
            provider_call_id=UUID(request["provider_call_id"]),
            assignments=tuple(
                WorkerAssignmentRecord(
                    execution_run_id=UUID(row["execution_run_id"]),
                    node_key=row["node_key"],
                    task_key=row["task_key"],
                    task_fingerprint=row["fingerprint"],
                    execution_payload=json.loads(row["execution_payload_json"]),
                    provider_call_id=UUID(row["provider_call_id"]),
                    request_id=row["request_id"],
                    ordinal=row["ordinal"],
                    created_at=row["created_at"],
                )
                for row in rows
            ),
        )

    def _complete_pruned_node_if_conclusive(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        now: int,
    ) -> None:
        active_call = self._connection.execute(
            """
            SELECT 1
            FROM execution_provider_calls
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status NOT IN (?, ?, ?)
            LIMIT 1
            """,
            (
                str(execution_run_id),
                node_key,
                ProviderCallStatus.SUCCEEDED.value,
                ProviderCallStatus.FAILED.value,
                ProviderCallStatus.CANCELLED.value,
            ),
        ).fetchone()
        unfinished_task = self._connection.execute(
            """
            SELECT 1
            FROM execution_tasks
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status IN (?, ?)
            LIMIT 1
            """,
            (
                str(execution_run_id),
                node_key,
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        ).fetchone()
        if active_call is not None or unfinished_task is not None:
            return
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                status_reason = ?,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status = ?
            """,
            (
                NodeStatus.CANCELLED.value,
                WorkStatusReason.RESULT_ALREADY_SATISFIED.value,
                now,
                now,
                str(execution_run_id),
                node_key,
                NodeStatus.RUNNING.value,
            ),
        )

    def _complete_cancelled_node_if_conclusive(
        self,
        execution_run_id: UUID,
        node_key: str,
        *,
        now: int,
    ) -> None:
        active_call = self._connection.execute(
            """
            SELECT 1
            FROM execution_provider_calls
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status NOT IN (?, ?, ?)
            LIMIT 1
            """,
            (
                str(execution_run_id),
                node_key,
                ProviderCallStatus.SUCCEEDED.value,
                ProviderCallStatus.FAILED.value,
                ProviderCallStatus.CANCELLED.value,
            ),
        ).fetchone()
        unfinished_task = self._connection.execute(
            """
            SELECT 1
            FROM execution_tasks
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status IN (?, ?)
            LIMIT 1
            """,
            (
                str(execution_run_id),
                node_key,
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        ).fetchone()
        if active_call is not None or unfinished_task is not None:
            return
        self._connection.execute(
            """
            UPDATE execution_nodes
            SET status = ?,
                status_reason = NULL,
                completed_at = ?,
                updated_at = ?
            WHERE execution_run_id = ?
                AND node_key = ?
                AND status = ?
            """,
            (
                NodeStatus.CANCELLED.value,
                now,
                now,
                str(execution_run_id),
                node_key,
                NodeStatus.RUNNING.value,
            ),
        )

    def _set_provider_call_status(
        self,
        provider_call_id: UUID,
        status: ProviderCallStatus,
        *,
        message: str | None,
        now: int,
    ) -> None:
        self._connection.execute(
            """
            UPDATE execution_provider_calls
            SET status = ?,
                error_message = ?,
                completed_at = CASE WHEN ? THEN ? ELSE NULL END,
                updated_at = ?
            WHERE provider_call_id = ?
            """,
            (
                status.value,
                message,
                int(status.is_terminal),
                now,
                now,
                str(provider_call_id),
            ),
        )

    def _finish_provider_call(
        self,
        provider_call_id: UUID,
        *,
        call_status: ProviderCallStatus,
        task_status: TaskStatus,
        message: str,
        now: int,
    ) -> ProviderCallRecord:
        call = self.get_provider_call(provider_call_id)
        if call.status.is_terminal:
            if call.status != call_status:
                raise ValueError(
                    f"cannot rewrite terminal Provider Call {call.status.value}"
                )
            return call
        self._set_provider_call_status(
            provider_call_id,
            call_status,
            message=message,
            now=now,
        )
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE provider_call_id = ?
                AND status IN (?, ?)
            """,
            (
                task_status.value,
                message,
                now,
                now,
                str(provider_call_id),
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        )
        self._connection.execute(
            """
            UPDATE execution_tasks
            SET status = ?,
                error_message = ?,
                completed_at = ?,
                updated_at = ?
            WHERE worker_provider_call_id = ?
                AND status IN (?, ?)
            """,
            (
                task_status.value,
                message,
                now,
                now,
                str(provider_call_id),
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
            ),
        )
        self._reconcile_run_unknown(call.execution_run_id, now=now)
        return self.get_provider_call(provider_call_id)

    def _project_run_unknown(
        self,
        execution_run_id: UUID,
        *,
        reason: RunStatusReason,
        message: str,
        now: int,
    ) -> None:
        run = self.get_run(execution_run_id)
        if run.status == RunStatus.STATE_UNKNOWN:
            if run.status_reason != reason or run.status_message != message:
                self._connection.execute(
                    """
                    UPDATE execution_runs
                    SET status_reason = ?,
                        status_message = ?,
                        updated_at = ?
                    WHERE execution_run_id = ?
                    """,
                    (reason.value, message, now, str(execution_run_id)),
                )
            return
        self._transition_run(
            execution_run_id,
            RunStatus.STATE_UNKNOWN,
            reason=reason,
            message=message,
            now=now,
            explicit_resume=False,
        )

    def _reconcile_run_unknown(
        self,
        execution_run_id: UUID,
        *,
        now: int,
    ) -> None:
        run = self.get_run(execution_run_id)
        if run.status != RunStatus.STATE_UNKNOWN:
            return
        row = self._connection.execute(
            """
            SELECT 1
            FROM execution_provider_calls
            WHERE execution_run_id = ?
                AND status IN (?, ?)
            LIMIT 1
            """,
            (
                str(execution_run_id),
                ProviderCallStatus.OUTCOME_UNKNOWN.value,
                ProviderCallStatus.STATE_UNKNOWN.value,
            ),
        ).fetchone()
        if row is None:
            target = (
                RunStatus.CANCEL_REQUESTED
                if run.status_reason == RunStatusReason.CANCELLATION_OUTCOME_UNKNOWN
                else RunStatus.RUNNING
            )
            self._transition_run(
                execution_run_id,
                target,
                reason=None,
                message=None,
                now=now,
                explicit_resume=False,
            )

    def _node_from_row(self, row: sqlite3.Row) -> ExecutionNodeRecord:
        dependencies = self._connection.execute(
            """
            SELECT dependency_node_key, accept_partial
            FROM execution_node_dependencies
            WHERE execution_run_id = ? AND node_key = ?
            ORDER BY ordinal
            """,
            (row["execution_run_id"], row["node_key"]),
        ).fetchall()
        return ExecutionNodeRecord(
            execution_run_id=UUID(row["execution_run_id"]),
            node_key=row["node_key"],
            ordinal=row["ordinal"],
            dependencies=tuple(
                NodeDependency(
                    node_key=dependency["dependency_node_key"],
                    accept_partial=bool(dependency["accept_partial"]),
                )
                for dependency in dependencies
            ),
            aggregation_policy=NodeAggregationPolicy(row["aggregation_policy"]),
            allow_empty_result=bool(row["allow_empty_result"]),
            status=NodeStatus(row["status"]),
            status_reason=(
                None
                if row["status_reason"] is None
                else WorkStatusReason(row["status_reason"])
            ),
            discovery_complete=bool(row["discovery_complete"]),
            result_observation=(
                None
                if row["result_observation"] is None
                else AvailabilityStatus(row["result_observation"])
            ),
            result_observed_at=row["result_observed_at"],
            result_provenance=(
                None
                if row["result_provenance"] is None
                else ResultProvenance(row["result_provenance"])
            ),
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )


def _validate_call_limits(total: int, gpu: int) -> None:
    if total <= 0:
        raise ValueError("max_active_provider_calls must be positive")
    if gpu < 0:
        raise ValueError("max_active_gpu_provider_calls cannot be negative")
    if gpu > total:
        raise ValueError(
            "max_active_gpu_provider_calls cannot exceed max_active_provider_calls"
        )


def _validate_run_reason(
    status: RunStatus,
    reason: RunStatusReason | None,
) -> None:
    allowed = _RUN_REASONS.get(status)
    if allowed is None:
        if reason is not None:
            raise ValueError(f"{status.value} does not accept a status reason")
        return
    if reason is None:
        raise ValueError(f"{status.value} requires a status reason")
    if reason not in allowed:
        raise ValueError(f"{reason.value} is not valid for {status.value}")


def _dump_plan(plan: ExecutionPlan) -> str:
    value = {
        "nodes": [
            {
                "aggregation_policy": node.aggregation_policy.value,
                "allow_empty_result": node.allow_empty_result,
                "dependencies": [
                    {
                        "accept_partial": dependency.accept_partial,
                        "node_key": dependency.node_key,
                    }
                    for dependency in node.dependencies
                ],
                "node_key": node.node_key,
            }
            for node in plan.nodes
        ],
        "scientific_payload": plan.scientific_payload,
        "scientific_versions": plan.scientific_versions,
        "workload_name": plan.workload_name,
        "workload_run_key": plan.workload_run_key,
    }
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _dump_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_plan(value: str) -> ExecutionPlan:
    decoded: dict[str, Any] = json.loads(value)
    return ExecutionPlan(
        workload_name=decoded["workload_name"],
        workload_run_key=decoded["workload_run_key"],
        scientific_payload=decoded["scientific_payload"],
        scientific_versions=decoded["scientific_versions"],
        nodes=tuple(
            NodePlan(
                node_key=node["node_key"],
                dependencies=tuple(
                    NodeDependency(
                        node_key=dependency["node_key"],
                        accept_partial=dependency["accept_partial"],
                    )
                    for dependency in node["dependencies"]
                ),
                aggregation_policy=NodeAggregationPolicy(node["aggregation_policy"]),
                allow_empty_result=node["allow_empty_result"],
            )
            for node in decoded["nodes"]
        ),
    )


def _task_from_row(row: sqlite3.Row) -> ExecutionTaskRecord:
    return ExecutionTaskRecord(
        execution_run_id=UUID(row["execution_run_id"]),
        node_key=row["node_key"],
        task_key=row["task_key"],
        ordinal=row["ordinal"],
        fingerprint=row["fingerprint"],
        scientific_payload=json.loads(row["scientific_payload_json"]),
        execution_payload=json.loads(row["execution_payload_json"]),
        status=TaskStatus(row["status"]),
        status_reason=(
            None
            if row["status_reason"] is None
            else WorkStatusReason(row["status_reason"])
        ),
        result_observation=(
            None
            if row["result_observation"] is None
            else AvailabilityStatus(row["result_observation"])
        ),
        result_observed_at=row["result_observed_at"],
        result_provenance=(
            None
            if row["result_provenance"] is None
            else ResultProvenance(row["result_provenance"])
        ),
        provider_call_id=(
            None if row["provider_call_id"] is None else UUID(row["provider_call_id"])
        ),
        worker_provider_call_id=(
            None
            if row["worker_provider_call_id"] is None
            else UUID(row["worker_provider_call_id"])
        ),
        local_owned=bool(row["local_owned"]),
        error_message=row["error_message"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _run_from_row(row: sqlite3.Row) -> ExecutionRunRecord:
    plan = _load_plan(row["plan_json"])
    if plan.workload_plan_fingerprint != row["workload_plan_fingerprint"]:
        raise RuntimeError("stored Execution Plan fingerprint does not match its data")
    return ExecutionRunRecord(
        execution_run_id=UUID(row["execution_run_id"]),
        predecessor_execution_run_id=(
            None
            if row["predecessor_execution_run_id"] is None
            else UUID(row["predecessor_execution_run_id"])
        ),
        plan=plan,
        deployment=DeploymentIdentity(
            environment=row["deployment_environment"],
            deployment_name=row["deployment_name"],
            deployment_version=row["deployment_version"],
        ),
        status=RunStatus(row["status"]),
        status_reason=(
            None
            if row["status_reason"] is None
            else RunStatusReason(row["status_reason"])
        ),
        status_message=row["status_message"],
        max_active_provider_calls=row["max_active_provider_calls"],
        max_active_gpu_provider_calls=row["max_active_gpu_provider_calls"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _binding_json_value(binding: ProviderBinding) -> dict[str, Any]:
    return {
        "app_name": binding.app_name,
        "app_version": binding.app_version,
        "environment": binding.environment,
        "function_name": binding.function_name,
        "runtime_image_key": binding.runtime_image_key,
        "uses_gpu": binding.uses_gpu,
    }
