"""Tests for AlphaFold3's shared execution graph."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

import biomodals.app.fold.alphafold3.execution_planning as planning_module
from biomodals.app.fold.alphafold3.execution_planning import (
    SEED_PREDICTIONS,
    STAGE_INFERENCE,
    TEMPLATE_SEARCHES,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)
from biomodals.app.fold.alphafold3.execution_runtime import (
    _result_envelope,
    alphafold3_execution_graph,
)
from biomodals.app.fold.alphafold3.generation_claims import GenerationClaim
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    InferenceRuntime,
    SeedClaimPlan,
)
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    GraphExecutionRunStore,
    ProviderCallStatus,
    ProviderDeploymentUnavailableError,
    RunStatus,
    TaskStatus,
)
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)

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


class RecordingDriver:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.spawns: list[dict[str, object]] = []

    def resolve(self, binding):
        if self.unavailable:
            raise ProviderDeploymentUnavailableError(str(binding))
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
        del provider_call_handle_id
        return ProviderCallObservation(ProviderCallObservationKind.RUNNING)

    def cancel(self, provider_call_handle_id: str) -> None:
        del provider_call_handle_id


def _request(
    *,
    seeds: list[int] | None = None,
    max_gpu_containers: int = 1,
    search_msa: bool = False,
) -> AlphaFold3ExecutionRequest:
    return AlphaFold3ExecutionRequest.prepare(
        AF3Config(
            name="example",
            modelSeeds=seeds or [1],
            sequences=[AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))],
        ),
        search_msa=search_msa,
        search_protein_templates=True,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=max_gpu_containers,
        recycle=10,
        sample=1,
    )


def _graph_inputs(tmp_path: Path):
    output = FakeVolume()
    cache = FakeVolume()
    claims = FakeClaims()
    return {
        "output_volume": output,
        "search_runtime": SearchRuntime(
            sharded_volume=cast(Any, FakeVolume()),
            cache_volume=cast(Any, cache),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            sharded_root=tmp_path / "sharded",
            cache_root=tmp_path / "cache",
        ),
        "template_runtime": TemplateRuntime(
            source_volume=cast(Any, FakeVolume()),
            cache_volume=cast(Any, cache),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            source_root=tmp_path / "source",
            cache_root=tmp_path / "cache",
        ),
        "inference_runtime": InferenceRuntime(
            output_root=tmp_path,
            volume=cast(Any, output),
            claims=cast(Any, claims),
            container_id="coordinator",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
    }


def _runtime(
    tmp_path: Path,
    *,
    request: AlphaFold3ExecutionRequest | None = None,
    driver: RecordingDriver | None = None,
) -> tuple[ExecutionGraphRuntime, dict[str, object]]:
    selected = request or _request()
    inputs = _graph_inputs(tmp_path)
    graph = alphafold3_execution_graph(
        selected,
        execution_run_id=RUN_ID,
        **inputs,
    )
    store = GraphExecutionRunStore(tmp_path, RUN_ID)
    runtime = ExecutionGraphRuntime(
        graph=graph,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        artifact_volume_name="AlphaFold3-outputs",
        workload_run_key=selected.execution_plan.workload_run_key,
        request=selected,
        provider_driver=cast(Any, driver or RecordingDriver()),
        storage_sync=ExecutionVolumeSync(
            volume=inputs["output_volume"],
            store=store,
        ),
        max_active_provider_calls=selected.max_active_provider_calls,
        max_active_gpu_provider_calls=selected.max_active_gpu_provider_calls,
        store=store,
        poll_interval_seconds=0,
        now=iter(range(10, 1000)).__next__,
    )
    return runtime, inputs


def _mock_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planning_module, "stage_inference_run", lambda *args: None)
    monkeypatch.setattr(
        planning_module,
        "load_staged_inference_input",
        lambda *args, **kwargs: SimpleNamespace(recycle=10),
    )


def _missing_seed_statuses(*args, **kwargs):
    seeds = args[2]
    return [{"status": "missing", "seed": seed} for seed in seeds]


def _owned_seed_claims(
    runtime,
    run_id,
    seeds,
    *,
    sample_count,
    generation_ids,
    reload_volume,
    allow_large_inference=False,
):
    del runtime, sample_count, reload_volume, allow_large_inference
    return SeedClaimPlan(
        reused_seeds=(),
        owned=tuple(
            ClaimedSeed(
                seed=seed,
                claim=GenerationClaim(
                    scope_key=f"seed:{run_id}:{seed}",
                    generation_id=generation_ids[seed],
                    owner={"identity": {"run_id": run_id, "seed": seed}},
                ),
            )
            for seed in seeds
        ),
        active=(),
    )


def _advance_until_calls(runtime: ExecutionGraphRuntime, count: int) -> None:
    for _ in range(16):
        runtime.advance_once()
        if len(runtime.store.execution.list_provider_calls(RUN_ID)) == count:
            return
    raise AssertionError(f"Expected {count} AlphaFold3 Provider Calls")


def test_graph_preserves_the_staged_execution_plan(tmp_path: Path) -> None:
    request = _request()
    assert request.execution_plan.workload_run_key is not None
    graph = alphafold3_execution_graph(
        request,
        execution_run_id=RUN_ID,
        **_graph_inputs(tmp_path),
    )

    plan = execution_plan(
        graph.validate(),
        workload_run_key=request.execution_plan.workload_run_key,
    )

    assert plan == request.execution_plan


def test_task_result_refresh_reloads_the_template_cache(tmp_path: Path) -> None:
    inputs = _graph_inputs(tmp_path)
    graph = alphafold3_execution_graph(
        _request(),
        execution_run_id=RUN_ID,
        **inputs,
    )
    node = cast(Any, graph.validate().nodes[TEMPLATE_SEARCHES].node)

    node.refresh_result_storage()

    assert inputs["template_runtime"].cache_volume.reloads == 1


def test_staged_inference_preserves_unknown_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = alphafold3_execution_graph(
        _request(),
        execution_run_id=RUN_ID,
        **_graph_inputs(tmp_path),
    )
    node = cast(Any, graph.validate().nodes[STAGE_INFERENCE].node)
    monkeypatch.setattr(
        planning_module,
        "load_staged_inference_input",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )

    assert node.planning.staged_inference_observation() == AvailabilityStatus.UNKNOWN


def test_no_search_stages_complete_without_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_staging(monkeypatch)
    monkeypatch.setattr(
        planning_module,
        "inspect_seed_predictions",
        _missing_seed_statuses,
    )
    runtime, _inputs = _runtime(tmp_path)
    runtime.attach()

    for _ in range(5):
        runtime.advance_once()

    nodes = {node.node_key: node for node in runtime.store.execution.list_nodes(RUN_ID)}
    for node_key in (
        "stage-request-input",
        "raw-database-searches",
        "combined-msa-publications",
        "protein-template-searches",
        "stage-inference-input",
    ):
        assert nodes[node_key].status.is_terminal
    assert {
        call.node_key for call in runtime.store.execution.list_provider_calls(RUN_ID)
    }.issubset({SEED_PREDICTIONS})


def test_seed_tasks_use_balanced_fixed_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_staging(monkeypatch)
    monkeypatch.setattr(
        planning_module,
        "inspect_seed_predictions",
        _missing_seed_statuses,
    )
    monkeypatch.setattr(
        planning_module,
        "claim_seed_predictions",
        _owned_seed_claims,
    )
    driver = RecordingDriver()
    runtime, _inputs = _runtime(
        tmp_path,
        request=_request(seeds=[1, 2, 3], max_gpu_containers=2),
        driver=driver,
    )
    runtime.attach()

    _advance_until_calls(runtime, 2)

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert [call.task_keys for call in calls] == [
        ("seed:1", "seed:2"),
        ("seed:3",),
    ]
    assert all(
        call.status in {ProviderCallStatus.ATTACHED, ProviderCallStatus.RUNNING}
        for call in calls
    )
    assert len(driver.spawns) == 2


def test_seed_claims_follow_deployment_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_staging(monkeypatch)
    monkeypatch.setattr(
        planning_module,
        "inspect_seed_predictions",
        _missing_seed_statuses,
    )
    claim_calls = 0

    def claim(*args, **kwargs):
        nonlocal claim_calls
        claim_calls += 1
        return _owned_seed_claims(*args, **kwargs)

    monkeypatch.setattr(planning_module, "claim_seed_predictions", claim)
    runtime, _inputs = _runtime(
        tmp_path,
        request=_request(seeds=[1, 2], max_gpu_containers=2),
        driver=RecordingDriver(unavailable=True),
    )
    runtime.attach()

    for _ in range(8):
        runtime.advance_once()
        if runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED:
            break

    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED
    assert claim_calls == 0
    assert runtime.store.execution.list_provider_calls(RUN_ID) == ()


def test_overlapping_seed_request_submits_only_missing_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_staging(monkeypatch)

    def inspect(runtime, run_id, seeds, **kwargs):
        del runtime, kwargs
        return [
            {
                "status": "reused" if seed == 1 else "missing",
                "run_id": run_id,
                "seed": seed,
            }
            for seed in seeds
        ]

    monkeypatch.setattr(planning_module, "inspect_seed_predictions", inspect)
    monkeypatch.setattr(
        planning_module,
        "claim_seed_predictions",
        _owned_seed_claims,
    )
    driver = RecordingDriver()
    runtime, _inputs = _runtime(
        tmp_path,
        request=_request(seeds=[1, 2], max_gpu_containers=2),
        driver=driver,
    )
    runtime.attach()

    _advance_until_calls(runtime, 1)

    [call] = runtime.store.execution.list_provider_calls(RUN_ID)
    assert call.task_keys == ("seed:2",)
    assert (
        runtime.store.execution.get_task(
            RUN_ID,
            SEED_PREDICTIONS,
            "seed:1",
        ).status
        == TaskStatus.SUCCEEDED
    )


def test_completed_invocation_prunes_ancestor_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        planning_module,
        "load_invocation_manifest",
        lambda *args, **kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        planning_module,
        "request_manifest_artifacts_available",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        planning_module,
        "load_request_manifest",
        lambda *args, **kwargs: {"status": "complete"},
    )
    runtime, _inputs = _runtime(tmp_path)
    runtime.attach()

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert snapshot.provider_calls == ()
    assert all(node.status.is_terminal for node in snapshot.nodes)


def test_malformed_provider_return_keeps_bounded_diagnostic() -> None:
    assert _result_envelope(None) == {"invalid_result": "None"}
    assert _result_envelope({"execution_result": {"path": "result.json"}}) == {
        "execution_result": {"path": "result.json"}
    }
