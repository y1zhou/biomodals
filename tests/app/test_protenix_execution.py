"""Protenix execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.fold import protenix_app
from biomodals.app.fold.protenix_execution import (
    DOWNLOAD_NODE,
    FINALIZE_NODE,
    INFERENCE_NODE,
    MSA_NODE,
    PLAN_NODE,
    ProtenixExecutionRequest,
    ProtenixExecutionRuntime,
    ProtenixMsaTaskSpec,
    ProtenixPreparationPlan,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import ModalCallObservation, ModalCallObservationKind
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "Protenix", 7)


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
    def __init__(self, root: Path) -> None:
        self.root = root
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
        if function_name == "download_protenix_data":
            return None
        if function_name == "plan_protenix_inputs":
            tasks = []
            for index in range(2):
                output_dir = self.root / "msa" / str(index)
                output_dir.mkdir(parents=True)
                input_path = output_dir / f"task-{index}.json"
                input_path.write_text("{}")
                tasks.append(
                    ProtenixMsaTaskSpec(
                        task_key=f"task-{index}",
                        input_name=f"task-{index}",
                        query_command="msa",
                        input_json_path=str(input_path),
                        output_dir=str(output_dir),
                        msa_server_mode="protenix",
                        expected_json_path=str(
                            output_dir / f"task-{index}-update-msa.json"
                        ),
                        publication_key=f"key-{index}",
                    )
                )
            return ProtenixPreparationPlan(
                preparation_key="prepare-key",
                prepared_json_path=str(self.root / "prepared" / "input.json"),
                tasks=tuple(tasks),
            )
        if function_name == "query_protenix_msa_server":
            task = kwargs["task"]
            assert isinstance(task, ProtenixMsaTaskSpec)
            expected = Path(task.expected_json_path)
            expected.write_text("{}")
            protenix_app._atomic_write(
                protenix_app._msa_task_marker_path(task),
                orjson.dumps({
                    "publication_key": task.publication_key,
                    "expected_json_path": str(expected),
                    "size": expected.stat().st_size,
                    "sha256": sha256(expected.read_bytes()).hexdigest(),
                }),
            )
            return None
        if function_name == "finalize_protenix_inputs":
            plan = kwargs["plan"]
            assert isinstance(plan, ProtenixPreparationPlan)
            prepared = Path(plan.prepared_json_path)
            protenix_app._atomic_write(prepared, b"{}")
            protenix_app._atomic_write(
                protenix_app._prepared_marker_path(plan),
                orjson.dumps({
                    "preparation_key": plan.preparation_key,
                    "size": prepared.stat().st_size,
                    "sha256": sha256(prepared.read_bytes()).hexdigest(),
                }),
            )
            return {"prepared_json_path": str(prepared), "size": 2}
        return protenix_app._publish_result(
            str(kwargs["result_key"]),
            str(kwargs["run_name"]),
            b"tar",
        )


def _request(**changes) -> ProtenixExecutionRequest:
    values = {
        "run_name": "complex",
        "input_content": b'[{"name":"complex"}]',
        "model_name": "protenix_base_default_v1.0.0",
        "seeds": "101,202",
        "cycle": 10,
        "step": 200,
        "sample": 5,
        "dtype": "bf16",
        "use_msa": True,
        "msa_server_mode": "protenix",
        "use_template": False,
        "use_rna_msa": False,
        "use_tfg_guidance": False,
        "use_fast_layernorm": True,
        "force_redownload": False,
        "extra_args": None,
        "score_only": False,
        "max_active_provider_calls": 4,
        "app_version": "7e1de70",
    }
    values.update(changes)
    return ProtenixExecutionRequest(**values)


def test_request_round_trip_preserves_msa_fanout_graph() -> None:
    request = _request()

    decoded = ProtenixExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.node_keys == (
        DOWNLOAD_NODE,
        PLAN_NODE,
        MSA_NODE,
        FINALIZE_NODE,
        INFERENCE_NODE,
    )
    assert decoded.execution_plan.terminal_node_keys == (INFERENCE_NODE,)


def test_score_only_skips_prediction_preprocessing() -> None:
    request = _request(score_only=True)

    assert request.execution_plan.node_keys == (DOWNLOAD_NODE, INFERENCE_NODE)


def test_operational_limits_do_not_change_scientific_identity() -> None:
    base = _request()
    changed = _request(
        max_active_provider_calls=12,
        force_redownload=True,
        replace_claim_owner="old-run",
    )

    assert (
        base.execution_plan.workload_plan_fingerprint
        == changed.execution_plan.workload_plan_fingerprint
    )


def test_runtime_dispatches_each_msa_input_as_a_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request()
    volume = FakeVolume()
    monkeypatch.setattr(
        protenix_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            repo_commit_hash="7e1de70",
            version="2.0.0",
        ),
    )
    driver = CompletingDriver(tmp_path)
    claims = FakeClaims()
    runtime = ProtenixExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=volume,
        msa_cache_volume=volume,
        output_claims=claims,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [name for name, _kwargs in driver.spawns] == [
        "plan_protenix_inputs",
        "download_protenix_data",
        "query_protenix_msa_server",
        "query_protenix_msa_server",
        "finalize_protenix_inputs",
        "run_protenix",
    ]
    assert set(claims.values.values()) == {str(RUN_ID)}
    runtime.close()
