"""Tests for AlphaFold3's caller-driven execution-kernel adapter."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

import biomodals.app.fold.alphafold3.execution_runtime as execution_runtime
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)
from biomodals.app.fold.alphafold3.execution_runtime import (
    AlphaFold3ExecutionRuntime,
    _result_envelope,
)
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import (
    DeploymentIdentity,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDeploymentUnavailableError,
)
from biomodals.helper.app_execution import AppExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "AlphaFold3", 7)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1

    def read_file(self, path: str):
        del path
        raise FileNotFoundError
        yield b""  # pragma: no cover


class FakeClaims:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(
        self,
        key: str,
        value: object,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)


class NoCallDriver:
    def resolve(self, binding):
        raise AssertionError(f"Unexpected remote binding: {binding}")

    def spawn(self, function, *, args, kwargs):
        del function, args, kwargs
        raise AssertionError("Unexpected Provider Call")

    def observe(self, provider_call_handle_id: str):
        raise AssertionError(provider_call_handle_id)

    def cancel(self, provider_call_handle_id: str) -> None:
        raise AssertionError(provider_call_handle_id)


class RecordingCallDriver:
    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.spawns) + 1}"
        self.spawns.append({
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "handle": handle,
        })
        return handle

    def observe(self, provider_call_handle_id: str):
        return ModalCallObservation(ModalCallObservationKind.RUNNING)

    def cancel(self, provider_call_handle_id: str) -> None:
        raise AssertionError(provider_call_handle_id)


class UnavailableCallDriver(NoCallDriver):
    def resolve(self, binding):
        raise ModalDeploymentUnavailableError(f"{binding} is unavailable")


def _request(
    *,
    seeds: list[int] | None = None,
    max_num_gpus: int = 1,
) -> AlphaFold3ExecutionRequest:
    return AlphaFold3ExecutionRequest.prepare(
        AF3Config(
            name="example",
            modelSeeds=seeds or [1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                    )
                )
            ],
        ),
        search_msa=False,
        search_protein_templates=True,
        max_parallel_search_workers=2,
        max_num_gpus=max_num_gpus,
        recycle=10,
        sample=1,
    )


def _runtime(
    tmp_path: Path,
    *,
    request: AlphaFold3ExecutionRequest | None = None,
    driver: object | None = None,
) -> AlphaFold3ExecutionRuntime:
    output = FakeVolume()
    cache = FakeVolume()
    source = FakeVolume()
    sharded = FakeVolume()
    claims = FakeClaims()
    return AlphaFold3ExecutionRuntime(
        request=request or _request(),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=AppExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver or NoCallDriver(),
        output_volume=output,
        search_runtime=SearchRuntime(
            sharded_volume=cast(Any, sharded),
            cache_volume=cast(Any, cache),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            sharded_root=tmp_path / "sharded",
            cache_root=tmp_path / "cache",
        ),
        template_runtime=TemplateRuntime(
            source_volume=cast(Any, source),
            cache_volume=cast(Any, cache),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            source_root=tmp_path / "source",
            cache_root=tmp_path / "cache",
        ),
        inference_runtime=InferenceRuntime(
            output_root=tmp_path,
            volume=cast(Any, output),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
        poll_interval_seconds=0,
        now=lambda: 10,
    )


def test_no_search_request_publishes_explicit_empty_stage_results(
    tmp_path: Path,
) -> None:
    """No-op scientific stages complete without synthetic Tasks or calls."""
    runtime = _runtime(tmp_path)
    runtime._initialize()

    for _ in range(4):
        runtime.advance_once()

    nodes = {node.node_key: node for node in runtime.store.execution.list_nodes(RUN_ID)}
    assert nodes["stage-request-input"].status == NodeStatus.SUCCEEDED
    for node_key in (
        "raw-database-searches",
        "combined-msa-publications",
        "protein-template-searches",
    ):
        assert nodes[node_key].status == NodeStatus.SUCCEEDED
        assert runtime.store.execution.list_tasks(RUN_ID, node_key) == ()
        assert (
            tmp_path / "execution-publications" / str(RUN_ID) / node_key / "empty.json"
        ).is_file()
    assert runtime.store.execution.list_provider_calls(RUN_ID) == ()
    runtime.close()


def test_malformed_provider_return_is_a_conclusive_diagnostic() -> None:
    """Malformed output is stored as a diagnostic, not provider uncertainty."""
    assert _result_envelope(None) == {"invalid_result": "None"}
    assert _result_envelope({"execution_result": {"path": "result.json"}}) == {
        "execution_result": {"path": "result.json"}
    }


def test_seed_tasks_use_fixed_batches_without_duplicate_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three seed Tasks map durably to two GPU calls and are not resubmitted."""
    driver = RecordingCallDriver()
    runtime = _runtime(
        tmp_path,
        request=_request(seeds=[1, 2, 3], max_num_gpus=2),
        driver=driver,
    )
    monkeypatch.setattr(
        execution_runtime,
        "load_staged_inference_input",
        lambda *args, **kwargs: SimpleNamespace(recycle=10),
    )
    runtime._initialize()

    for _ in range(6):
        runtime.advance_once()

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert len(driver.spawns) == 2
    assert [call.task_keys for call in calls] == [
        ("seed:1", "seed:2"),
        ("seed:3",),
    ]
    assert {call.status for call in calls} == {ProviderCallStatus.ATTACHED}
    assert all(
        cast(Any, spawn["function"]).function_name == "run_inference_pipeline"
        for spawn in driver.spawns
    )

    runtime.advance_once()

    assert len(driver.spawns) == 2
    assert all(
        call.status == ProviderCallStatus.RUNNING
        for call in runtime.store.execution.list_provider_calls(RUN_ID)
    )
    runtime.close()


def test_seed_claim_is_not_acquired_before_deployment_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired exact version cannot strand a scientific generation claim."""
    runtime = _runtime(
        tmp_path,
        request=_request(seeds=[1, 2], max_num_gpus=2),
        driver=UnavailableCallDriver(),
    )
    monkeypatch.setattr(
        execution_runtime,
        "load_staged_inference_input",
        lambda *args, **kwargs: SimpleNamespace(recycle=10),
    )
    runtime._initialize()

    for _ in range(6):
        runtime.advance_once()

    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED
    assert runtime.store.execution.list_provider_calls(RUN_ID) == ()
    claims = cast(FakeClaims, runtime.inference_runtime.claims)
    assert claims.values == {}
    runtime.close()
