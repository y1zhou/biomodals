"""Execution-kernel contract tests for direct OligoFormer runs."""

# ruff: noqa: D101,D102,D103,D107

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

from biomodals.app.score import oligoformer_app
from biomodals.app.score.oligoformer_execution import (
    EFFICACY_NODE,
    EVIDENCE_MERGE_NODE,
    EVIDENCE_PLAN_NODE,
    FINAL_NODE,
    PITA_CANDIDATES_NODE,
    PITA_REFERENCE_NODE,
    PREPARE_NODE,
    PUBLISH_NODE,
    REFERENCE_FINALIZE_NODE,
    REFERENCE_PLAN_NODE,
    REFERENCE_SHARDS_NODE,
    TARGETSCAN_TILES_NODE,
    OligoformerExecutionCoordinator,
    OligoformerExecutionRequest,
    OligoformerExecutionRuntime,
)
from biomodals.execution import AvailabilityStatus, DeploymentIdentity, RunStatus
from biomodals.execution.modal import ModalCallObservation, ModalCallObservationKind
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "OligoFormer", 7)


class FakeVolume:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class FakeClaims:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def put(self, key: str, value: str, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


class CompletingDriver:
    def __init__(self, state: dict[str, bool], plan, evidence_plan=None) -> None:
        self.state = state
        self.plan = plan
        self.evidence_plan = evidence_plan
        self.calls: dict[str, tuple[Any, dict[str, object]]] = {}
        self.spawns: list[str] = []

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.calls) + 1}"
        self.calls[handle] = (function, dict(kwargs))
        self.spawns.append(function.function_name)
        return handle

    def observe(self, provider_call_handle_id: str):
        function, _kwargs = self.calls[provider_call_handle_id]
        name = function.function_name
        if name == "download_oligoformer_models":
            self.state["models"] = True
            result = None
        elif name == "prepare_oligoformer_run":
            result = self.plan
        elif name == "run_oligoformer_efficacy":
            self.state["efficacy"] = True
            result = replace(self.plan, efficacy_ready=True)
        elif name == "plan_oligoformer_off_target_evidence":
            result = self.evidence_plan
        elif name == "prepare_oligoformer_pita_reference":
            result = oligoformer_app.PitaReferencePlan(
                ("/reference/utr.stab",),
                "/reference/ext-utr.stab",
            )
        elif name == "run_oligoformer_pita_candidate":
            self.state["pita"] = True
            spec = cast(oligoformer_app.OffTargetShardSpec, _kwargs["spec"])
            result = oligoformer_app.OffTargetShardResult(
                spec.index,
                "/results/pita.tab",
            )
        elif name == "run_oligoformer_targetscan_tile":
            self.state["targetscan"] = True
            result = "/results/targetscan.tab"
        elif name == "publish_oligoformer_off_target_evidence":
            self.state["evidence"] = True
            result = None
        elif name == "build_oligoformer_final_tables":
            self.state["final"] = True
            result = replace(self.plan, efficacy_ready=True, final_ready=True)
        else:
            path = oligoformer_app._oligoformer_result_archive_path(self.plan)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"archive")
            oligoformer_app._publish_oligoformer_result_record(
                Path(self.plan.run_root).parent,
                str(_kwargs["publication_key"]),
                path,
                model_identity=self.plan.model_identity or "model-content-v1",
                reference_identity=self.plan.reference_identity,
            )
            result = {"result_path": str(path), "size_bytes": 7}
        return ModalCallObservation(ModalCallObservationKind.SUCCEEDED, result=result)

    def cancel(self, provider_call_handle_id: str) -> None:
        pass


def _request(**changes) -> OligoformerExecutionRequest:
    execution = oligoformer_app.DEFAULT_EXECUTION_CONFIG
    request = OligoformerExecutionRequest(
        run_name="example",
        mrna_fasta_bytes=b">target\nAUGCUAGCUAGCUAGCUAGC\n",
        sirna_fasta_bytes=None,
        off_target=False,
        toxicity=False,
        all_human=False,
        utr_bytes=None,
        orf_bytes=None,
        top_n=20,
        functionality_filter=True,
        pita_threshold=-10.0,
        targetscan_threshold=1.0,
        toxicity_threshold=50.0,
        off_target_nodes=execution.off_target_nodes,
        off_target_workers=execution.off_target_workers,
        off_target_process_slots=execution.off_target_process_slots,
        off_target_prep_workers=execution.off_target_prep_workers,
        pita_prepare_nodes=execution.pita_prepare_nodes,
        pita_prepare_workers=execution.pita_prepare_workers,
        pita_prepare_utr_shard_size=execution.pita_prepare_utr_shard_size,
        pita_row_shard_size=execution.pita_row_shard_size,
        pita_row_attempts=execution.pita_row_attempts,
        targetscan_rnaplfold_nodes=execution.targetscan_rnaplfold_nodes,
        targetscan_rnaplfold_workers=execution.targetscan_rnaplfold_workers,
        targetscan_rnaplfold_shard_size=execution.targetscan_rnaplfold_shard_size,
        targetscan_prepare_nodes=execution.targetscan_prepare_nodes,
        targetscan_ref_shard_size=None,
        targetscan_candidate_shard_size=execution.targetscan_candidate_shard_size,
        targetscan_context_nodes=execution.targetscan_context_nodes,
        targetscan_context_workers=execution.targetscan_context_workers,
        targetscan_context_shard_size=execution.targetscan_context_shard_size,
        targetscan_context_attempts=execution.targetscan_context_attempts,
        targetscan_merge_nodes=execution.targetscan_merge_nodes,
        force=False,
        force_generation=None,
        app_version="test-version",
        model_version="model-source-v1",
        reference_version="reference-source-v1",
    )
    return replace(request, **changes)


def test_request_round_trips_without_pickle() -> None:
    request = _request(
        off_target=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
    )

    assert OligoformerExecutionRequest.from_bytes(request.to_bytes()) == request


def test_coordinator_reuses_its_active_runtime(tmp_path: Path) -> None:
    """Concurrent lifecycle calls cannot replace a runtime under its driver."""
    request = _request()
    coordinator = OligoformerExecutionCoordinator(
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        output_volume=FakeVolume(),
        model_volume=FakeVolume(),
        output_claims=FakeClaims(),
        modal_driver=cast(Any, object()),
    )
    runtime = cast(
        OligoformerExecutionRuntime,
        SimpleNamespace(
            request=request,
            predecessor_execution_run_id=None,
        ),
    )
    coordinator._runtime = runtime

    assert coordinator._open_runtime(request) is runtime


def test_efficacy_only_plan_is_minimal() -> None:
    request = _request()

    assert request.execution_plan.node_keys == (
        "download-models",
        PREPARE_NODE,
        EFFICACY_NODE,
        FINAL_NODE,
        PUBLISH_NODE,
    )


def test_model_and_reference_versions_are_scientific_identity() -> None:
    request = _request(off_target=True, all_human=True, reference_version="ref-v1")

    assert (
        replace(
            request, model_version="model-source-v2"
        ).execution_plan.workload_plan_fingerprint
        != request.execution_plan.workload_plan_fingerprint
    )
    assert (
        replace(
            request, reference_version="ref-v2"
        ).execution_plan.workload_plan_fingerprint
        != request.execution_plan.workload_plan_fingerprint
    )


def test_cached_terminal_publication_completes_without_a_run_plan(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request()
    archive = tmp_path / "cached" / "oligoformer.tar.zst"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")
    oligoformer_app._publish_oligoformer_result_record(
        tmp_path,
        request.execution_plan.workload_plan_fingerprint,
        archive,
        model_identity="model-content-v1",
        reference_identity=None,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_oligoformer_model_volume_identity_digest",
        lambda: "model-content-v1",
    )
    driver = CompletingDriver({}, None)
    volume = FakeVolume()
    runtime = OligoformerExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=volume,
        model_volume=volume,
        output_claims=FakeClaims(),
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert driver.spawns == []
    runtime.close()


def test_cached_terminal_publication_rejects_changed_model(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request()
    archive = tmp_path / "cached" / "oligoformer.tar.zst"
    archive.parent.mkdir()
    archive.write_bytes(b"archive")
    oligoformer_app._publish_oligoformer_result_record(
        tmp_path,
        request.execution_plan.workload_plan_fingerprint,
        archive,
        model_identity="model-content-v1",
        reference_identity=None,
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_oligoformer_model_volume_identity_digest",
        lambda: "model-content-v2",
    )
    volume = FakeVolume()
    runtime = OligoformerExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=CompletingDriver({}, None),
        output_volume=volume,
        model_volume=volume,
        output_claims=FakeClaims(),
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    assert runtime._node_observation(PUBLISH_NODE) == AvailabilityStatus.MISSING
    runtime.close()


def test_custom_reference_plan_uses_kernel_off_target_tiles() -> None:
    request = _request(
        off_target=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
    )

    assert request.execution_plan.node_keys == (
        "download-models",
        PREPARE_NODE,
        EFFICACY_NODE,
        EVIDENCE_PLAN_NODE,
        PITA_REFERENCE_NODE,
        PITA_CANDIDATES_NODE,
        TARGETSCAN_TILES_NODE,
        EVIDENCE_MERGE_NODE,
        FINAL_NODE,
        PUBLISH_NODE,
    )


def test_all_human_plan_prepares_reference_shards() -> None:
    request = _request(off_target=True, all_human=True)

    assert request.execution_plan.node_keys[3:6] == (
        REFERENCE_PLAN_NODE,
        REFERENCE_SHARDS_NODE,
        REFERENCE_FINALIZE_NODE,
    )


def test_process_budget_is_split_without_extra_scheduler_state() -> None:
    execution = replace(
        oligoformer_app.DEFAULT_EXECUTION_CONFIG,
        off_target_process_slots=12,
        off_target_nodes=3,
        off_target_workers=8,
        off_target_prep_workers=1,
        pita_prepare_nodes=1,
        pita_prepare_workers=8,
        targetscan_prepare_nodes=2,
        targetscan_context_nodes=2,
        targetscan_context_workers=8,
    )

    assert oligoformer_app._off_target_branch_slots(execution) == (6, 6)
    assert oligoformer_app._pita_local_workers(execution) == (1, 2)
    assert oligoformer_app._targetscan_local_workers(execution) == 3

    runtime = object.__new__(OligoformerExecutionRuntime)
    runtime.request = _request(
        off_target_process_slots=execution.off_target_process_slots,
        off_target_nodes=execution.off_target_nodes,
        pita_prepare_nodes=execution.pita_prepare_nodes,
    )
    assert runtime._node_call_limit(PITA_CANDIDATES_NODE) == 1


def test_oligoformer_has_no_app_owned_modal_queue_or_nested_dispatch() -> None:
    source = Path(oligoformer_app.__file__).read_text(encoding="utf-8")

    assert "modal.Queue" not in source
    assert ".remote(" not in source
    scientific_source = source[: source.index("def submit_oligoformer_task")]
    assert ".spawn(" not in scientific_source


def test_pita_reference_provider_mounts_all_human_references() -> None:
    volumes = oligoformer_app.prepare_oligoformer_pita_reference.spec.volumes

    assert oligoformer_app.CONF.model_volume_mountpoint in volumes


def test_runtime_drives_efficacy_only_run_through_deployed_functions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request()
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="cache",
        efficacy_key="efficacy",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(tmp_path / "efficacy"),
        output_dir=str(tmp_path / "run" / "outputs" / "final"),
        output_stems=("target",),
        config=oligoformer_app.OligoformerRunConfig(),
        postprocess_key="final",
        efficacy_ready=False,
        evidence_ready=False,
        final_ready=False,
        model_identity="model-content-v1",
    )
    state = {"models": False, "efficacy": False, "final": False}
    monkeypatch.setattr(
        oligoformer_app,
        "_oligoformer_models_ready",
        lambda: state["models"],
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_build_plan",
        lambda *_args, **_kwargs: replace(
            plan,
            efficacy_ready=state["efficacy"],
            final_ready=state["final"],
        ),
    )
    volume = FakeVolume()
    driver = CompletingDriver(state, plan)
    runtime = OligoformerExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=volume,
        model_volume=volume,
        output_claims=FakeClaims(),
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert driver.spawns == [
        "download_oligoformer_models",
        "prepare_oligoformer_run",
        "run_oligoformer_efficacy",
        "build_oligoformer_final_tables",
        "publish_oligoformer_outputs",
    ]
    runtime.close()


def test_runtime_dispatches_off_target_scientific_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request(
        off_target=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
    )
    config = oligoformer_app.OligoformerRunConfig(off_target=True)
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="cache",
        efficacy_key="efficacy",
        run_root=str(tmp_path / "run"),
        efficacy_dir=str(tmp_path / "efficacy"),
        output_dir=str(tmp_path / "run" / "outputs" / "final"),
        output_stems=("target",),
        config=config,
        postprocess_key="final",
        efficacy_ready=False,
        evidence_ready=False,
        final_ready=False,
        model_identity="model-content-v1",
    )
    pita_spec = oligoformer_app.OffTargetShardSpec(
        run_root=plan.run_root,
        output_dir=plan.output_dir,
        stem="target",
        index=0,
        record_name="RNA0",
        record_sequence="AUGCUAGCUAGCUAGCUAG",
        utr_path=str(tmp_path / "utr.fa"),
        orf_path=str(tmp_path / "orf.fa"),
        row_shard_size=1000,
    )
    targetscan_spec = oligoformer_app.TargetscanBatchSpec(
        run_root=plan.run_root,
        output_dir=plan.output_dir,
        stem="target",
        ref_shard_size=10,
        shard_index=0,
        sirna_path=str(tmp_path / "sirna.fa"),
        sirna_count=1,
        utr_path=str(tmp_path / "utr.fa"),
        orf_path=str(tmp_path / "orf.fa"),
        rnaplfold_cache_dir="",
    )
    evidence_plan = oligoformer_app.OligoformerEvidencePlan((
        oligoformer_app.OligoformerEvidenceStemPlan(
            "target",
            (pita_spec,),
            (targetscan_spec,),
        ),
    ))
    state = {
        "models": False,
        "efficacy": False,
        "pita": False,
        "targetscan": False,
        "evidence": False,
        "final": False,
    }
    monkeypatch.setattr(
        oligoformer_app,
        "_oligoformer_models_ready",
        lambda: state["models"],
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_build_plan",
        lambda *_args, **_kwargs: replace(
            plan,
            efficacy_ready=state["efficacy"],
            evidence_ready=state["evidence"],
            final_ready=state["final"],
        ),
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_pita_candidate_ready",
        lambda _spec: state["pita"],
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_targetscan_tile_ready",
        lambda _spec: state["targetscan"],
    )
    monkeypatch.setattr(
        oligoformer_app,
        "_raw_off_target_ready",
        lambda *_args, **_kwargs: state["evidence"],
    )
    volume = FakeVolume()
    driver = CompletingDriver(state, plan, evidence_plan)
    runtime = OligoformerExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=volume,
        model_volume=volume,
        output_claims=FakeClaims(),
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert "run_oligoformer_pita_candidate" in driver.spawns
    assert "run_oligoformer_targetscan_tile" in driver.spawns
    assert driver.spawns.count("run_oligoformer_pita_candidate") == 1
    assert driver.spawns.count("run_oligoformer_targetscan_tile") == 1
    runtime.close()
