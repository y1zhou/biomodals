"""Immutable preparation survives staging, version drift and coordinator reopen."""

# ruff: noqa: D103

from dataclasses import replace
from uuid import UUID

import pytest

from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import (
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.workflow.nanobody_humanization import workflow
from biomodals.workflow.nanobody_humanization.execution import (
    NanobodyExecutionCoordinator,
    NanobodyExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.workflow.nanobody_humanization.preparation import VHInput, prepare_vh

VHH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"


def test_saved_preparation_and_cli_graph_identity_are_immutable(tmp_path, monkeypatch):
    request = NanobodyExecutionRequest(
        run_name="test", parents=(prepare_vh(VHInput(id="one", vhh="HHHHHH" + VHH)),)
    )
    run_id = UUID(int=1)
    persist_execution_request(tmp_path, run_id, request)
    monkeypatch.setattr(
        "biomodals.workflow.nanobody_humanization.preparation.prepare_vh",
        lambda *args: pytest.fail("Loading saved input must not re-prepare"),
    )
    restored = load_execution_request(tmp_path, run_id)
    assert restored == request
    assert restored.parents[0].original_sequence == "HHHHHH" + VHH
    assert restored.parents[0].sequence == VHH
    expected = request.execution_plan
    assert (
        replace(
            request, max_active_provider_calls=1000, max_active_gpu_provider_calls=40
        ).execution_plan
        == expected
    )
    generation = [
        node for node in expected.nodes if node.node_key.startswith("generate_")
    ]
    assert len(generation) == 2
    assert all(node.dependencies == () for node in generation)
    union = next(node for node in expected.nodes if node.node_key == "union")
    assert {edge.node_key for edge in union.dependencies} == {
        node.node_key for node in generation
    }
    assert all(edge.accept_partial for edge in union.dependencies)
    with pytest.raises(RuntimeError, match="immutable"):
        persist_execution_request(
            tmp_path, run_id, replace(request, run_name="changed")
        )
    with pytest.raises(ValueError, match="scientific versions"):
        _ = replace(request, scientific_versions={}).execution_plan


def test_unknown_generation_reopens_without_replacements(tmp_path):
    request = NanobodyExecutionRequest(
        run_name="test", parents=(prepare_vh(VHInput(id="one", vhh=VHH)),)
    )
    run_id = UUID(int=2)
    persist_execution_request(tmp_path, run_id, request)

    class Driver:
        def __init__(self):
            self.calls = []
            self.unknown = True

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.calls.append(operation)
            return f"call-{len(self.calls)}"

        def observe(self, handle):
            return ProviderCallObservation(
                ProviderCallObservationKind.STATE_UNKNOWN
                if self.unknown
                else ProviderCallObservationKind.FAILED,
                message="Native failure",
            )

        def cancel(self, handle):
            pass

    class Volume:
        def commit(self):
            pass

        def reload(self):
            pass

    driver = Driver()

    def host():
        return NanobodyExecutionCoordinator(
            execution_run_id=run_id,
            deployment=DeploymentIdentity("main", "NanobodyHumanizationWorkflow", 1),
            volume_root=tmp_path,
            output_volume=Volume(),
            output_volume_name="workflow",
            provider_driver=driver,
            poll_interval_seconds=0,
        )

    coordinator = host()
    try:
        assert coordinator.run().run.status == RunStatus.STATE_UNKNOWN
        driver.unknown = False
        assert coordinator.resume().run.status == RunStatus.FAILED
        assert len(driver.calls) == 2
    finally:
        coordinator.close()
    reopened = host()
    try:
        assert reopened.run().run.status == RunStatus.FAILED
        assert len(driver.calls) == 2
        with pytest.raises(LookupError, match="not available"):
            reopened.result()
    finally:
        reopened.close()


def test_cli_dry_run_prepares_native_domains_without_cloud(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.csv"
    source.write_text(f"id,vhh\nexample,HHHHHH{VHH}\n")
    monkeypatch.setattr(
        workflow,
        "stage_execution_request",
        lambda *args: pytest.fail("No staging in dry run"),
    )
    monkeypatch.setattr(
        workflow.modal.Function,
        "from_name",
        lambda *args, **kwargs: pytest.fail("No cloud lookup"),
    )
    workflow.submit_nanobody_humanization_workflow.info.raw_f(
        input_csv=str(source), dry_run=True
    )
    displayed = capsys.readouterr().out
    assert all(
        name in displayed
        for name in (
            "generate_abnativ2_vhh",
            "generate_hudiff_nb",
            "union",
            "evaluate",
            "publish",
        )
    )
    with pytest.raises(ValueError, match="max_gpu_containers"):
        workflow.submit_nanobody_humanization_workflow.info.raw_f(
            input_csv=str(source), dry_run=True, max_containers=1, max_gpu_containers=2
        )
