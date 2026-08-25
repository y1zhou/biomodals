"""Tests for AlphaFold3's remote App Run request boundary."""

# ruff: noqa: D101,D102,D107

from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3 import execution_request, inference_inputs
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    load_execution_request,
    persist_execution_request,
    stage_execution_request,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


class FakeVolume:
    def __init__(self, root: Path) -> None:
        self.root = root

    def read_file(self, path: str):
        selected = self.root / path.lstrip("/")
        if not selected.is_file():
            raise FileNotFoundError(path)
        yield selected.read_bytes()

    @contextmanager
    def batch_upload(self, *, force: bool):
        assert force
        root = self.root

        class Batch:
            def put_file(self, source, destination: str) -> None:
                path = root / destination.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read())

        yield Batch()


def _request(
    *,
    max_containers: int = 4,
    max_gpu_containers: int = 2,
    allow_large_inference: bool = False,
):
    config = AF3Config(
        name="example",
        modelSeeds=[2, 1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                )
            )
        ],
    )
    return AlphaFold3ExecutionRequest.prepare(
        config,
        search_msa=True,
        search_protein_templates=True,
        max_active_provider_calls=max_containers,
        max_active_gpu_provider_calls=max_gpu_containers,
        allow_large_inference=allow_large_inference,
        recycle=10,
        sample=5,
    )


def test_execution_request_round_trips_and_revalidates_identity() -> None:
    """Staged state re-derives rather than trusting its invocation record."""
    predecessor = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    request = _request(allow_large_inference=True)
    request = AlphaFold3ExecutionRequest.prepare(
        request.config,
        search_msa=request.search_msa,
        search_protein_templates=request.search_protein_templates,
        max_active_provider_calls=request.max_active_provider_calls,
        max_active_gpu_provider_calls=request.max_active_gpu_provider_calls,
        allow_large_inference=request.allow_large_inference,
        recycle=request.recycle,
        sample=request.sample,
        repair_execution_run_ids=(predecessor,),
    )

    decoded = AlphaFold3ExecutionRequest.from_bytes(request.to_bytes())

    assert decoded.invocation == request.invocation
    assert decoded.execution_plan == request.execution_plan
    assert decoded.max_active_provider_calls == 4
    assert decoded.allow_large_inference
    assert decoded.repair_execution_run_ids == (predecessor,)


def test_execution_envelope_has_independent_metadata_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Envelope metadata should fit beyond an exact-limit staged config."""
    request = _request()
    config_bytes = inference_inputs.serialize_af3_input(request.config)
    monkeypatch.setattr(
        inference_inputs,
        "MAX_STAGED_INPUT_BYTES",
        len(config_bytes),
    )
    envelope = request.to_bytes()
    assert len(envelope) > len(config_bytes)

    monkeypatch.setattr(
        execution_request,
        "MAX_EXECUTION_REQUEST_BYTES",
        len(envelope),
    )
    assert request.to_bytes() == envelope


def test_execution_envelope_retains_one_gib_ceiling() -> None:
    """The internal coordinator envelope retains its independent headroom."""
    assert execution_request.MAX_EXECUTION_REQUEST_BYTES == 1024 * 1024 * 1024


def test_operational_limits_do_not_change_the_scientific_plan() -> None:
    """CPU and GPU call ceilings remain outside result compatibility."""
    first = _request(max_containers=4, max_gpu_containers=2)
    second = _request(max_containers=1, max_gpu_containers=1)

    assert (
        first.execution_plan.workload_plan_fingerprint
        == second.execution_plan.workload_plan_fingerprint
    )
    assert first.max_active_provider_calls == 4
    assert second.max_active_provider_calls == 1


def test_execution_request_rejects_a_forged_invocation() -> None:
    """The remote coordinator does not trust client-supplied identity fields."""
    record = orjson.loads(_request().to_bytes())
    record["invocation"]["invocation_id"] = "a" * 64

    with pytest.raises(ValueError, match="does not match"):
        AlphaFold3ExecutionRequest.from_bytes(orjson.dumps(record))


def test_execution_request_staging_is_immutable_and_remotely_revalidated(
    tmp_path: Path,
) -> None:
    """The client stages bytes once and the coordinator revalidates them."""
    request = _request()
    volume = FakeVolume(tmp_path)

    request_path = stage_execution_request(volume, RUN_ID, request)
    assert stage_execution_request(volume, RUN_ID, request) == request_path
    assert load_execution_request(tmp_path, RUN_ID) == request

    path = tmp_path.joinpath(*request_path.parts)
    path.write_bytes(_request(max_containers=1, max_gpu_containers=1).to_bytes())
    with pytest.raises(RuntimeError, match="conflicts"):
        stage_execution_request(volume, RUN_ID, request)


def test_coordinator_request_persistence_is_idempotent(tmp_path: Path) -> None:
    """Mounted coordinators can stage a successor request before its ledger."""
    request = _request()

    request_path = persist_execution_request(tmp_path, RUN_ID, request)
    assert persist_execution_request(tmp_path, RUN_ID, request) == request_path
    assert load_execution_request(tmp_path, RUN_ID) == request
