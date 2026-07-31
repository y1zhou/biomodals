"""ENsiRNA execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from biomodals.app.score import ensirna_app
from biomodals.app.score.ensirna_execution import (
    CHUNKS_NODE,
    DOWNLOAD_MODELS_NODE,
    FINALIZE_NODE,
    INFERENCE_NODE,
    PREPARE_NODE,
    PREPROCESS_NODE,
    EnsirnaExecutionRequest,
    EnsirnaExecutionRuntime,
    EnsirnaPdbChunkSpec,
    EnsirnaPreparationPlan,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import ModalCallObservation, ModalCallObservationKind
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "ENsiRNA", 7)


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
    def __init__(self, request: EnsirnaExecutionRequest) -> None:
        self.request = request
        self.calls: dict[str, tuple[Any, dict[str, object]]] = {}
        self.spawns: list[tuple[str, dict[str, object]]] = []

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.calls) + 1}"
        copied = dict(kwargs)
        self.calls[handle] = (function, copied)
        self.spawns.append((function.function_name, copied))
        return handle

    def observe(self, provider_call_handle_id: str):
        function, kwargs = self.calls[provider_call_handle_id]
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=self._publish(function.function_name, kwargs),
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        pass

    def _publish(self, function_name: str, kwargs: dict[str, object]):
        if function_name == "download_ensirna_models":
            return None
        if function_name == "ensirna_prepare_inputs":
            cache_key = ensirna_app._cache_key_for_fasta(self.request.fasta_content)
            layout = ensirna_app._layout_for_cache_key(cache_key)
            prep_dir = ensirna_app._pdb_prep_dir(layout)
            prep_dir.mkdir(parents=True, exist_ok=True)
            chunk = EnsirnaPdbChunkSpec(
                chunk_name="chunk_0000",
                csv_path=str(prep_dir / "chunk_0000.csv"),
                json_path=str(prep_dir / "chunk_0000.json"),
                pdb_dir=str(layout.outputs_dir / "mrna_pdb"),
            )
            Path(chunk.csv_path).write_text("siRNA\ntarget_0\n", encoding="utf-8")
            return EnsirnaPreparationPlan(
                cache_key=cache_key,
                prepared_dir=str(layout.run_root),
                json_path=str(layout.outputs_dir / "mrna.json"),
                processed_dir=str(layout.outputs_dir / "mrna_processed"),
                candidate_count=1,
                chunk_count=1,
                chunks=[chunk],
                cached=False,
            )
        if function_name == "ensirna_prepare_pdb_chunk":
            chunk = kwargs["chunk"]
            assert isinstance(chunk, EnsirnaPdbChunkSpec)
            Path(chunk.json_path).write_text('{"siRNA":"target_0"}\n')
            return {"chunk_name": chunk.chunk_name, "cached": 1}
        plan = kwargs["plan"] if "plan" in kwargs else None
        if function_name == "ensirna_finalize_prepared_inputs":
            assert isinstance(plan, EnsirnaPreparationPlan)
            json_path = Path(plan.json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text('{"siRNA":"target_0"}\n')
            return plan
        if function_name == "ensirna_preprocess_dataset":
            assert isinstance(plan, EnsirnaPreparationPlan)
            Path(plan.prepared_dir, "prepared.test").write_text("ready")
            return plan
        prepared_dir = Path(str(kwargs["prepared_dir"]))
        result = prepared_dir / "outputs" / "mrna_result.xlsx"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"xlsx")
        return b"xlsx"


def test_ensirna_request_round_trip_preserves_the_staged_dag() -> None:
    request = EnsirnaExecutionRequest(
        run_name="design",
        fasta_content=b">target\nACGUACGUACGUACGUACG\n",
        prepare_workers=4,
        pdb_cores=2,
        preprocess_shard_size=1024,
        force_generation=None,
        app_version="0288243",
    )

    decoded = EnsirnaExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.node_keys == (
        DOWNLOAD_MODELS_NODE,
        PREPARE_NODE,
        CHUNKS_NODE,
        FINALIZE_NODE,
        PREPROCESS_NODE,
        INFERENCE_NODE,
    )
    assert decoded.execution_plan.terminal_node_keys == (INFERENCE_NODE,)


def test_ensirna_concurrency_and_sharding_are_operational() -> None:
    base = EnsirnaExecutionRequest(
        run_name="design",
        fasta_content=b">target\nACGUACGUACGUACGUACG\n",
        prepare_workers=4,
        pdb_cores=2,
        preprocess_shard_size=1024,
        force_generation=None,
        app_version="0288243",
    )
    changed = EnsirnaExecutionRequest(
        run_name=base.run_name,
        fasta_content=base.fasta_content,
        prepare_workers=8,
        pdb_cores=1,
        preprocess_shard_size=256,
        force_generation=base.force_generation,
        app_version=base.app_version,
        replace_claim_owner="old-run",
    )

    assert (
        base.execution_plan.workload_plan_fingerprint
        == changed.execution_plan.workload_plan_fingerprint
    )


def test_runtime_dispatches_the_staged_graph(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = EnsirnaExecutionRequest(
        run_name="design",
        fasta_content=b">target\nACGUACGUACGUACGUACG\n",
        prepare_workers=4,
        pdb_cores=2,
        preprocess_shard_size=1024,
        force_generation=None,
        app_version="0288243",
    )
    volume = FakeVolume()
    monkeypatch.setattr(
        ensirna_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        ensirna_app,
        "_chunk_artifacts_valid",
        lambda chunk: Path(chunk.json_path).is_file(),
    )
    monkeypatch.setattr(
        ensirna_app,
        "_is_prepared",
        lambda layout: Path(layout.run_root, "prepared.test").is_file(),
    )
    monkeypatch.setattr(
        ensirna_app,
        "_result_ready",
        lambda layout, _cache_key: Path(
            layout.outputs_dir,
            "mrna_result.xlsx",
        ).is_file(),
    )
    driver = CompletingDriver(request)
    claims = FakeClaims()
    runtime = EnsirnaExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=volume,
        output_claims=claims,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [name for name, _kwargs in driver.spawns] == [
        "ensirna_prepare_inputs",
        "download_ensirna_models",
        "ensirna_prepare_pdb_chunk",
        "ensirna_finalize_prepared_inputs",
        "ensirna_preprocess_dataset",
        "run_ensirna_inference",
    ]
    assert set(claims.values.values()) == {str(RUN_ID)}
    runtime.close()
