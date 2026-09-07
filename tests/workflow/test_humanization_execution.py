"""Staged requests and the service-compatible host use the real offline kernel."""

from dataclasses import replace
from uuid import UUID

import polars as pl
import pytest

from biomodals.execution import DeploymentIdentity, ExecutionOverview, RunStatus
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.modal import (
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.execution import (
    HumanizationExecutionCoordinator,
    HumanizationExecutionRequest,
    load_execution_request,
    persist_execution_request,
    result_directory,
)
from biomodals.workflow.humanization.workflow import build_humanization_workflow


def test_request_roundtrip_preserves_cli_scientific_identity(tmp_path):
    """API staging preserves scientific identity independently of admission ceilings."""
    request = HumanizationExecutionRequest(
        run_name="test", pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),)
    )
    run_id = UUID(int=1)
    persist_execution_request(tmp_path, run_id, request)
    assert load_execution_request(tmp_path, run_id) == request
    assert HumanizationExecutionRequest.from_bytes(request.to_bytes()) == request
    expected = execution_plan(
        build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n").validate(),
        workload_run_key="test",
    )
    assert request.execution_plan == expected
    assert replace(request, max_active_provider_calls=1000).execution_plan == expected
    with pytest.raises(RuntimeError, match="immutable"):
        persist_execution_request(
            tmp_path, run_id, replace(request, run_name="changed")
        )
    with pytest.raises(ValueError, match="max_gpu_containers"):
        replace(request, max_active_provider_calls=1)


def test_staged_host_returns_partial_overview_and_reopens_without_resubmission(
    tmp_path,
):
    """Failed providers still publish the parental table and never silently rerun."""

    class Driver:
        def __init__(self):
            self.calls = []
            self.unknown = True

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.calls.append(operation)
            return f"call-{len(self.calls)}"

        def observe(self, provider_call_handle_id):
            if self.unknown:
                return ProviderCallObservation(
                    ProviderCallObservationKind.STATE_UNKNOWN
                )
            return ProviderCallObservation(
                ProviderCallObservationKind.FAILED, message="model unavailable"
            )

        def cancel(self, provider_call_handle_id):
            pass

    class Volume:
        def commit(self):
            pass

        def reload(self):
            pass

    run_id = UUID(int=2)
    request = HumanizationExecutionRequest(
        run_name="test", pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),)
    )
    persist_execution_request(tmp_path, run_id, request)
    driver = Driver()

    def host():
        return HumanizationExecutionCoordinator(
            execution_run_id=run_id,
            deployment=DeploymentIdentity("main", "HumanizationWorkflow", 1),
            volume_root=tmp_path,
            output_volume=Volume(),
            output_volume_name="workflow",
            provider_driver=driver,
            poll_interval_seconds=0,
        )

    coordinator = host()
    try:
        overview = coordinator.run()
        assert isinstance(overview, ExecutionOverview)
        assert overview.run.status == RunStatus.STATE_UNKNOWN
        driver.unknown = False
        assert coordinator.resume().run.status == RunStatus.PARTIAL
        assert len(driver.calls) == 8
        page = coordinator.provider_calls(limit=2)
        assert len(page.calls) == 2
        assert coordinator.provider_call(page.calls[0].provider_call_id) is not None
    finally:
        coordinator.close()
    reopened = host()
    try:
        assert reopened.status().run.status == RunStatus.PARTIAL
        assert reopened.run().run.status == RunStatus.PARTIAL
        assert len(driver.calls) == 8
    finally:
        reopened.close()
    table = pl.read_csv(tmp_path / result_directory(run_id) / "selection.csv")
    assert table.select("parent_id", "is_parent").rows() == [("a", True)]


def test_staged_host_rejects_different_scientific_version(tmp_path):
    """A changed deployed runtime cannot silently execute an older staged request."""
    request = HumanizationExecutionRequest(
        run_name="test", pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),)
    )
    request = replace(
        request,
        scientific_versions={**request.scientific_versions, "sapiens": "changed"},
    )
    run_id = UUID(int=3)
    persist_execution_request(tmp_path, run_id, request)
    coordinator = HumanizationExecutionCoordinator(
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "HumanizationWorkflow", 1),
        volume_root=tmp_path,
        output_volume=object(),
        output_volume_name="workflow",
        provider_driver=object(),
    )
    with pytest.raises(ValueError, match="scientific"):
        coordinator.run()
