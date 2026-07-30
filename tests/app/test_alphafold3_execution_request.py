"""Tests for AlphaFold3's remote App Run request boundary."""

# ruff: noqa: D101,D102,D107

from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    execution_request_path,
    load_execution_request,
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


def _request(*, search_workers: int = 4, gpu_workers: int = 2):
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
        max_parallel_search_workers=search_workers,
        max_num_gpus=gpu_workers,
        recycle=10,
        sample=5,
    )


def test_execution_request_round_trips_and_revalidates_identity() -> None:
    """Staged state re-derives rather than trusting its invocation record."""
    request = _request()

    decoded = AlphaFold3ExecutionRequest.from_bytes(request.to_bytes())

    assert decoded.invocation == request.invocation
    assert decoded.execution_plan == request.execution_plan
    assert decoded.max_active_provider_calls == 4


def test_operational_limits_do_not_change_the_scientific_plan() -> None:
    """CPU and GPU call ceilings remain outside result compatibility."""
    first = _request(search_workers=4, gpu_workers=2)
    second = _request(search_workers=1, gpu_workers=1)

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

    assert stage_execution_request(volume, RUN_ID, request) == (
        execution_request_path(RUN_ID)
    )
    assert stage_execution_request(volume, RUN_ID, request) == (
        execution_request_path(RUN_ID)
    )
    assert load_execution_request(tmp_path, RUN_ID) == request

    path = tmp_path.joinpath(*execution_request_path(RUN_ID).parts)
    path.write_bytes(_request(search_workers=1).to_bytes())
    with pytest.raises(RuntimeError, match="conflicts"):
        stage_execution_request(volume, RUN_ID, request)
