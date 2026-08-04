"""Tests for BoltzGen's immutable remote App Run request."""

# ruff: noqa: D101,D102,D103,D107

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from biomodals.app.design.boltzgen.execution_request import (
    BoltzGenExecutionRequest,
    load_execution_request,
    prepare_execution_request,
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
    yaml_content: bytes = b"name: example\n",
    total_calls: int = 2,
    gpu_calls: int = 2,
    replace_claim_owners: tuple[tuple[str, str], ...] = (),
) -> BoltzGenExecutionRequest:
    request = prepare_execution_request(
        run_name="example",
        run_ids=("run-a", "run-b"),
        yaml_content=yaml_content,
        additional_files={"template.cif": b"data"},
        protocol="nanobody-anything",
        num_designs=10,
        budget=5,
        steps=None,
        extra_args=None,
        filter_results=True,
        filter_rmsd_threshold=2.5,
        app_version="0.3.2",
        repo_commit_hash="abc123",
        max_active_provider_calls=total_calls,
        max_active_gpu_provider_calls=gpu_calls,
    )
    if replace_claim_owners:
        return replace(
            request,
            replace_claim_owners=replace_claim_owners,
        )
    return request


def test_operational_limits_and_claim_owners_do_not_change_scientific_plan() -> None:
    first = _request(total_calls=2, gpu_calls=2)
    second = _request(
        total_calls=1,
        gpu_calls=1,
        replace_claim_owners=(("run-a", "old-call"),),
    )

    assert (
        first.execution_plan.workload_plan_fingerprint
        == second.execution_plan.workload_plan_fingerprint
    )
    assert first.max_active_provider_calls == 2
    assert second.max_active_provider_calls == 1


def test_result_affecting_input_changes_scientific_plan() -> None:
    first = _request()
    second = _request(yaml_content=b"name: changed\n")

    assert (
        first.execution_plan.workload_plan_fingerprint
        != second.execution_plan.workload_plan_fingerprint
    )


def test_request_round_trips_and_stages_idempotently(tmp_path: Path) -> None:
    request = _request()
    volume = FakeVolume(tmp_path)

    assert BoltzGenExecutionRequest.from_bytes(request.to_bytes()) == request
    request_relative_path = stage_execution_request(volume, RUN_ID, request)
    assert stage_execution_request(volume, RUN_ID, request) == request_relative_path
    assert load_execution_request(tmp_path, RUN_ID) == request

    request_path = tmp_path.joinpath(*request_relative_path.parts)
    request_path.write_bytes(_request(yaml_content=b"changed: true\n").to_bytes())
    with pytest.raises(RuntimeError, match="conflicts"):
        stage_execution_request(volume, RUN_ID, request)


@pytest.mark.parametrize(
    "changes",
    [
        {"run_name": "../escape"},
        {"run_ids": ("run-a", "../escape")},
    ],
)
def test_request_rejects_path_escaping_run_identity(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="safe filename component"):
        replace(_request(), **changes)
