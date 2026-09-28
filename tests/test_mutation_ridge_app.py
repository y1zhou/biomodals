"""Offline run-owned publication, provider composition and CLI validation."""

from uuid import uuid4

import polars as pl
import pytest

from biomodals.app.design.mutation_ridge import app as ridge_app
from biomodals.app.design.mutation_ridge.execution import (
    OPERATION,
    RUNTIME_IDENTITY,
    MutationRidgeNode,
    RidgeRequest,
    ridge_graph,
)
from biomodals.app.design.mutation_ridge.runtime import run_ridge
from biomodals.execution.nodes import NodeRunContext
from biomodals.schema import AppRunStatus


def _request():
    return RidgeRequest(
        measurements_csv="mutations,label\n,0\nA:A1C,1\nA:A2M,2\n",
        parental_fasta=">A\nAA\n",
    )


def test_run_publication_is_idempotent_and_checks_content(tmp_path, monkeypatch):
    """Redelivery reuses a complete task result, while corrupted output fails."""
    arguments = dict(
        output_key="a" * 64,
        runtime_identity=RUNTIME_IDENTITY,
        volume_root=tmp_path,
        volume_name="test",
    )
    first = run_ridge(_request().model_dump_json(), **arguments)
    assert first.status == AppRunStatus.SUCCEEDED
    output = first.outputs[0]
    path = tmp_path / output.storage.path
    assert pl.read_csv(path)["sequence_A"].to_list() == ["CM"]
    monkeypatch.setattr(
        "biomodals.app.design.mutation_ridge.runtime.write_combinations",
        lambda *a, **k: pytest.fail("Repeated scientific work"),
    )
    assert run_ridge(_request().model_dump_json(), **arguments) == first
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run_ridge(_request().model_dump_json(), **arguments)


def test_unpublished_partial_csv_can_be_replaced_but_foreign_plan_cannot(tmp_path):
    """Partial output is not a completion marker, and keys bind exact requests."""
    path = tmp_path / "runs" / ("a" * 64) / "outputs" / "candidates.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial")
    arguments = dict(
        output_key="a" * 64,
        runtime_identity=RUNTIME_IDENTITY,
        volume_root=tmp_path,
        volume_name="test",
    )
    run_ridge(_request().model_dump_json(), **arguments)
    changed = _request().model_copy(update={"alpha": 2.0})
    with pytest.raises(ValueError, match="different scientific request"):
        run_ridge(changed.model_dump_json(), **arguments)
    assert pl.read_csv(path).height == 1


def test_node_is_a_single_cpu_call_in_its_owning_run(tmp_path):
    """The app operation composes without a nested coordinator or path input."""
    context = NodeRunContext(
        execution_run_id=uuid4(),
        workload_run_key="display",
        node_id="ridge",
        task_key="default",
        work_dir=tmp_path,
        cache_dir=tmp_path,
        inputs={},
    )
    node = MutationRidgeNode(_request())
    call = node.prepare_remote(context)
    assert call.function_name == OPERATION
    assert call.uses_gpu is False
    assert len(call.kwargs["output_key"]) == 64
    assert call == node.prepare_remote(context)
    assert RidgeRequest.model_validate_json(call.kwargs["request_json"]) == _request()
    ridge_graph(_request()).validate()


def test_worker_reload_then_commit_and_reject_wrong_runtime(tmp_path, monkeypatch):
    """Warm workers refresh state before reads and commit before returning."""
    events = []

    class FakeVolume:
        def reload(self):
            events.append("reload")

        def commit(self):
            events.append("commit")

    class FakeConfig:
        output_volume = FakeVolume()
        output_volume_mountpoint = str(tmp_path)
        output_volume_name = "test"

    monkeypatch.setattr(ridge_app, "CONF", FakeConfig())
    result = ridge_app.mutation_ridge_score.get_raw_f()(
        _request().model_dump_json(), "b" * 64, RUNTIME_IDENTITY
    )
    assert result.metrics["candidate_count"] == 1
    assert events == ["reload", "commit"]
    with pytest.raises(ValueError, match="scientific versions"):
        ridge_app.mutation_ridge_score.get_raw_f()(
            _request().model_dump_json(), "b" * 64, "older"
        )


def test_cli_dry_run_requires_no_modal_calls(tmp_path, monkeypatch, capsys):
    """Input and graph validation precede any coordinator lookup."""
    csv = tmp_path / "measurements.csv"
    csv.write_text(_request().measurements_csv)
    monkeypatch.setattr(
        ridge_app.orchestrator,
        "execution_coordinator_handle",
        lambda **kwargs: pytest.fail("Unexpected Modal access"),
    )
    ridge_app.submit_mutation_ridge_task(input_csv=str(csv), dry_run=True)
    assert "fit_score_combinations" in capsys.readouterr().out


def test_request_budgets_and_output_paths_are_checked_before_compute(tmp_path):
    """Unbounded output estimates and path traversal never reach fitting."""
    request = _request().model_dump()
    request["parental_fasta"] = ">A\n" + "A" * 16385
    with pytest.raises(ValueError, match="total parental residues"):
        RidgeRequest(**request)
    with pytest.raises(ValueError, match="output key"):
        run_ridge(
            _request().model_dump_json(),
            output_key="../escape",
            runtime_identity=RUNTIME_IDENTITY,
            volume_root=tmp_path,
            volume_name="test",
        )
