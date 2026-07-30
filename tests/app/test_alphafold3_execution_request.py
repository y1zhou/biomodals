"""Tests for AlphaFold3's remote App Run request boundary."""

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)


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
