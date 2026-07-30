"""Tests for the immutable direct Rosetta App Run request."""

# ruff: noqa: D103

from dataclasses import replace
from hashlib import sha256

from biomodals.app.bioinfo.rosetta.execution_contracts import RosettaTaskSpec
from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
)


def _request() -> RosettaExecutionRequest:
    return RosettaExecutionRequest(
        run_name="example",
        run_id="run-id",
        tasks=(
            RosettaTaskSpec(
                task_key="1",
                index=1,
                binary="relax",
                pdb="inputs/1/input.pdb",
                rosetta_script=None,
                flags_file=None,
                output_dir="outputs/1",
                worker_log="logs/1.log",
                expected_files=(),
                input_sha256=sha256(b"ATOM\n").hexdigest(),
            ),
        ),
        app_version="2025.51",
        max_active_provider_calls=2,
        claim_capacity=4,
        max_parallel_per_worker=4,
    )


def test_request_round_trips_and_excludes_worker_policy_from_fingerprint() -> None:
    request = _request()
    changed_policy = replace(
        request,
        max_active_provider_calls=1,
        claim_capacity=8,
        max_parallel_per_worker=2,
    )

    assert RosettaExecutionRequest.from_bytes(request.to_bytes()) == request
    assert (
        request.execution_plan.workload_plan_fingerprint
        == changed_policy.execution_plan.workload_plan_fingerprint
    )


def test_file_paths_do_not_replace_content_identity() -> None:
    request = _request()
    moved_task = replace(
        request.tasks[0],
        pdb="inputs/1/renamed.pdb",
    )
    moved = replace(request, tasks=(moved_task,))
    changed = replace(
        request,
        tasks=(
            replace(
                request.tasks[0],
                input_sha256=sha256(b"DIFFERENT\n").hexdigest(),
            ),
        ),
    )

    assert (
        request.execution_plan.workload_plan_fingerprint
        == moved.execution_plan.workload_plan_fingerprint
    )
    assert (
        request.execution_plan.workload_plan_fingerprint
        != changed.execution_plan.workload_plan_fingerprint
    )
