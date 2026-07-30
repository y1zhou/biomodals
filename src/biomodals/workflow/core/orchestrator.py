"""Workflow-owned Modal adapter for the shared execution kernel."""

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

import modal

from biomodals.app.config import AppConfig
from biomodals.execution import (
    DeploymentIdentity,
    ExecutionRunNotFoundError,
    ExecutionSnapshot,
    ProviderBinding,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    deployed_execution_coordinator,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import (
    MAX_TIMEOUT,
    WORKFLOW_ORCHESTRATOR_VOLUME,
    WORKFLOW_ORCHESTRATOR_VOLUME_NAME,
)
from biomodals.schema import AppRunResult
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.execution import execution_plan
from biomodals.workflow.core.run_store import WorkflowRunStore
from biomodals.workflow.core.runtime import WorkflowRuntime

CONF = AppConfig(
    tags={"group": "workflow"},
    name="WorkflowOrchestrator",
    package_name="biomodals-workflow-orchestrator",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)
OUT_VOLUME = WORKFLOW_ORCHESTRATOR_VOLUME
OUT_VOLUME_NAME = WORKFLOW_ORCHESTRATOR_VOLUME_NAME
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


@dataclass(frozen=True)
class WorkflowCoordinatorPlan:
    """Trusted workflow definition and operational settings needed for recovery."""

    workflow: Workflow
    workload_run_key: str
    max_active_provider_calls: int = 32
    max_active_gpu_provider_calls: int | None = None
    strict_external_artifact_checks: bool = False
    external_artifact_checker_function_name: str | None = None

    def __post_init__(self) -> None:
        """Reject settings that cannot form a kernel Execution Run."""
        if not isinstance(self.workflow, Workflow):
            raise TypeError("workflow must be a Workflow object")
        if not self.workload_run_key:
            raise ValueError("workload_run_key cannot be empty")
        if self.max_active_provider_calls < 1:
            raise ValueError("max_active_provider_calls must be positive")
        if not 0 <= self.effective_gpu_limit <= self.max_active_provider_calls:
            raise ValueError(
                "max_active_gpu_provider_calls must be between zero and "
                "max_active_provider_calls"
            )
        if (
            self.strict_external_artifact_checks
            and not self.external_artifact_checker_function_name
        ):
            raise ValueError(
                "strict artifact checks require an external checker function name"
            )

    @property
    def effective_gpu_limit(self) -> int:
        """Return the persisted GPU subset limit."""
        if self.max_active_gpu_provider_calls is None:
            return self.max_active_provider_calls
        return self.max_active_gpu_provider_calls

    @property
    def identity(self) -> tuple[object, ...]:
        """Return fields that must not change within one Execution Run."""
        plan = execution_plan(
            self.workflow.validate(),
            workload_run_key=self.workload_run_key,
        )
        return (
            plan.workload_plan_fingerprint,
            self.workload_run_key,
            self.max_active_provider_calls,
            self.effective_gpu_limit,
            self.strict_external_artifact_checks,
            self.external_artifact_checker_function_name,
        )


@app.cls(
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped, single-writer coordinator included in each workflow app."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh durable state before serving this run-scoped pool."""
        self._writer_lock = RLock()
        self._runtime = None
        self._development_function_handles = None
        self._identity()
        OUT_VOLUME.reload()

    @modal.method()
    def run(
        self,
        workflow: Workflow,
        workload_run_key: str,
        max_active_provider_calls: int = 32,
        max_active_gpu_provider_calls: int | None = None,
        strict_external_artifact_checks: bool = False,
        external_artifact_checker_function_name: str | None = None,
        development_function_handles: Mapping[str, Any] | None = None,
    ) -> AppRunResult:
        """Persist one plan and drive its Execution Run until it stops."""
        candidate = WorkflowCoordinatorPlan(
            workflow=workflow,
            workload_run_key=workload_run_key,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            strict_external_artifact_checks=strict_external_artifact_checks,
            external_artifact_checker_function_name=(
                external_artifact_checker_function_name
            ),
        )
        with self._lock():
            plan = self._persist_or_verify_plan(candidate)
            if development_function_handles is not None:
                self._development_function_handles = dict(development_function_handles)
            runtime = self._open_runtime(plan, resolve_external_checker=True)
        try:
            return runtime.run(
                workload_run_key=plan.workload_run_key,
                synchronize=self._lock,
            )
        finally:
            with self._lock():
                self._close_runtime()
                OUT_VOLUME.commit()

    @modal.method()
    def status(self) -> ExecutionSnapshot:
        """Read the current kernel snapshot without advancing the Run."""
        with self._lock():
            return self._verified_snapshot()

    @modal.method()
    def cancel(self) -> ExecutionSnapshot:
        """Idempotently request cancellation of this Execution Run."""
        with self._lock():
            self._require_ledger()
            plan = self._load_plan()
            runtime = self._open_runtime(plan, resolve_external_checker=False)
            runtime.cancel()
            return self._verified_snapshot()

    @modal.method()
    def resume(self) -> AppRunResult:
        """Explicitly resume and drive a suspended Execution Run."""
        with self._lock():
            self._require_ledger()
            plan = self._load_plan()
            runtime = self._open_runtime(plan, resolve_external_checker=True)
        try:
            return runtime.resume(
                workload_run_key=plan.workload_run_key,
                synchronize=self._lock,
            )
        finally:
            with self._lock():
                self._close_runtime()
                OUT_VOLUME.commit()

    @modal.exit()
    def exit(self) -> None:
        """Persist pending workflow state without cancelling child calls."""
        with self._lock():
            self._close_runtime()
            OUT_VOLUME.commit()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        execution_run_id = UUID(self.execution_run_id)
        deployment = DeploymentIdentity(
            environment=self.deployment_environment,
            deployment_name=self.deployment_name,
            deployment_version=self.deployment_version,
        )
        return execution_run_id, deployment

    def _lock(self) -> RLock:
        lock = getattr(self, "_writer_lock", None)
        if lock is None:
            lock = RLock()
            self._writer_lock = lock
        return lock

    def _persist_or_verify_plan(
        self,
        candidate: WorkflowCoordinatorPlan,
    ) -> WorkflowCoordinatorPlan:
        store = self._run_store()
        try:
            if store.workflow_plan_path.exists():
                plan = _decode_plan(store.read_workflow_plan())
                if plan.identity != candidate.identity:
                    raise ValueError(
                        "Workflow coordinator plan does not match Execution Run"
                    )
                return plan
            store.write_workflow_plan(pickle.dumps(candidate))
            OUT_VOLUME.commit()
            return candidate
        finally:
            store.close()

    def _load_plan(self) -> WorkflowCoordinatorPlan:
        store = self._run_store()
        try:
            return _decode_plan(store.read_workflow_plan())
        finally:
            store.close()

    def _open_runtime(
        self,
        plan: WorkflowCoordinatorPlan,
        *,
        resolve_external_checker: bool,
    ) -> WorkflowRuntime:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            return runtime
        execution_run_id, deployment = self._identity()
        driver = self._modal_driver()
        external_checker = None
        if resolve_external_checker and plan.strict_external_artifact_checks:
            function_name = plan.external_artifact_checker_function_name
            if function_name is None:
                raise RuntimeError("Persisted workflow plan has no artifact checker")
            checker = driver.resolve(
                ProviderBinding(
                    environment=deployment.environment,
                    app_name=deployment.deployment_name,
                    app_version=deployment.deployment_version,
                    function_name=function_name,
                    uses_gpu=False,
                )
            )
            external_checker = checker.remote
        runtime = WorkflowRuntime(
            workflow=plan.workflow,
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=Path(CONF.output_volume_mountpoint),
            workflow_volume_name=OUT_VOLUME_NAME,
            workflow_volume=OUT_VOLUME,
            modal_driver=driver,
            max_active_provider_calls=plan.max_active_provider_calls,
            max_active_gpu_provider_calls=plan.effective_gpu_limit,
            strict_external_artifact_checks=(
                plan.strict_external_artifact_checks
                if resolve_external_checker
                else False
            ),
            external_artifact_checker=external_checker,
        )
        self._runtime = runtime
        return runtime

    def _modal_driver(self) -> ModalCallDriver:
        handles = getattr(self, "_development_function_handles", None)
        if handles is None:
            return ModalCallDriver()

        def resolve_development_function(
            _app_name: str,
            function_name: str,
            **_kwargs: object,
        ) -> Any:
            try:
                return handles[function_name]
            except KeyError as error:
                raise ValueError(
                    f"No development function handle for {function_name!r}"
                ) from error

        return ModalCallDriver(function_resolver=resolve_development_function)

    def _verified_snapshot(self) -> ExecutionSnapshot:
        execution_run_id, deployment = self._identity()
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            snapshot = runtime.store.execution.snapshot(execution_run_id)
        else:
            self._require_ledger()
            store = self._run_store()
            try:
                snapshot = store.execution.snapshot(execution_run_id)
            finally:
                store.close()
        if snapshot.run.deployment != deployment:
            raise ValueError("Deployment Identity does not match Execution Run")
        return snapshot

    def _run_store(self) -> WorkflowRunStore:
        execution_run_id, _ = self._identity()
        return WorkflowRunStore(
            Path(CONF.output_volume_mountpoint),
            execution_run_id,
        )

    def _require_ledger(self) -> None:
        store = self._run_store()
        if not store.ledger_path.is_file():
            raise ExecutionRunNotFoundError(self.execution_run_id)

    def _close_runtime(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.close()
            self._runtime = None


def _decode_plan(content: bytes) -> WorkflowCoordinatorPlan:
    """Decode one trusted, deployment-owned workflow plan."""
    plan = pickle.loads(content)  # noqa: S301 - internal Volume state, not user input
    if not isinstance(plan, WorkflowCoordinatorPlan):
        raise TypeError("Stored workflow coordinator plan has an unsupported type")
    return plan


def execution_coordinator_handle(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    use_deployed_coordinator: bool,
    class_resolver: Any | None = None,
) -> Any:
    """Bind one run to either its exact deployment or current development app."""
    if use_deployed_coordinator:
        return deployed_execution_coordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            class_resolver=class_resolver or modal.Cls.from_name,
        )
    return ExecutionCoordinator(
        execution_run_id=str(execution_run_id),
        deployment_environment=deployment.environment,
        deployment_name=deployment.deployment_name,
        deployment_version=deployment.deployment_version,
    )
