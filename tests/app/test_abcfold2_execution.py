"""ABCFold2 execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from biomodals.app.fold import abcfold2_app
from biomodals.app.fold.abcfold2_execution import (
    BOLTZ_ARCHIVE_NODE,
    BOLTZ_DOWNLOAD_NODE,
    BOLTZ_SEEDS_NODE,
    CHAI_ARCHIVE_NODE,
    CHAI_DOWNLOAD_NODE,
    CHAI_SEEDS_NODE,
    PREPARE_NODE,
    ABCFold2ExecutionRequest,
    ABCFold2ExecutionRuntime,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import ModalCallObservation, ModalCallObservationKind
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "ABCFold2", 7)


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
        self.root = root / "ab" / "abcdef-no-tmpl"
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
        if function_name == "prepare_abcfold2":
            self.root.mkdir(parents=True, exist_ok=True)
            for model in ("boltz", "chai"):
                model_dir = self.root / f"{model}_models"
                model_dir.mkdir()
                (model_dir / "abcdef-no-tmpl.yaml").write_text("config")
            return {
                "run_id": "abcdef-no-tmpl",
                "workdir": str(self.root),
                "seeds": [1, 2],
                "num_trunk_recycles": 1,
                "num_diffn_timesteps": 2,
                "num_diffn_samples": 3,
                "num_trunk_samples": 4,
                "boltz_additional_cli_args": None,
            }
        if function_name in {"run_abcfold2_boltz", "run_abcfold2_chai"}:
            model = "boltz" if function_name.endswith("boltz") else "chai"
            seed = int(str(kwargs["seed"]))
            name = (
                f"boltz_results_seed-{seed}"
                if model == "boltz"
                else f"chai_seed-{seed}"
            )
            path = self.root / f"{model}_models" / name
            path.mkdir()
            abcfold2_app._write_publication_marker(
                self.root / ".biomodals" / f"{model}-seed-{seed}.json",
                {
                    "publication_key": kwargs["publication_key"],
                    "result_path": str(path),
                },
            )
            return str(path)
        model = "boltz" if "boltz" in function_name else "chai"
        return abcfold2_app._publish_archive(
            self.root,
            model,
            b"tar",
            str(kwargs["publication_key"]),
        )


def _request(**changes) -> ABCFold2ExecutionRequest:
    values = {
        "run_name": "complex-no-tmpl",
        "yaml_content": b"name: complex\nseeds: [1, 2]\n",
        "msa_chains": "A,B",
        "search_templates": False,
        "download_models": True,
        "force_redownload": False,
        "run_boltz": True,
        "run_chai": True,
        "max_active_provider_calls": 4,
        "app_version": "fcfdd49",
        "boltz_version": "cb04aec",
        "chai_version": "0ac6831",
    }
    values.update(changes)
    return ABCFold2ExecutionRequest(**values)


def test_request_round_trip_preserves_parallel_model_branches() -> None:
    request = _request()

    decoded = ABCFold2ExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.node_keys == (
        PREPARE_NODE,
        BOLTZ_DOWNLOAD_NODE,
        CHAI_DOWNLOAD_NODE,
        BOLTZ_SEEDS_NODE,
        BOLTZ_ARCHIVE_NODE,
        CHAI_SEEDS_NODE,
        CHAI_ARCHIVE_NODE,
    )
    assert decoded.execution_plan.terminal_node_keys == (
        BOLTZ_ARCHIVE_NODE,
        CHAI_ARCHIVE_NODE,
    )


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


def test_disabled_models_are_absent_from_the_graph() -> None:
    request = _request(
        download_models=False,
        run_boltz=False,
        run_chai=False,
    )

    assert request.execution_plan.node_keys == (PREPARE_NODE,)
    assert request.execution_plan.terminal_node_keys == (PREPARE_NODE,)


def test_runtime_dispatches_each_seed_without_nested_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request(download_models=False)
    volume = FakeVolume()
    monkeypatch.setattr(
        abcfold2_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            repo_commit_hash="fcfdd49",
            version="0.2.0",
        ),
    )
    driver = CompletingDriver(tmp_path)
    claims = FakeClaims()
    runtime = ABCFold2ExecutionRuntime(
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
        "prepare_abcfold2",
        "run_abcfold2_boltz",
        "run_abcfold2_boltz",
        "run_abcfold2_chai",
        "run_abcfold2_chai",
        "collect_abcfold2_boltz_data",
        "collect_abcfold2_chai_data",
    ]
    assert set(claims.values.values()) == {str(RUN_ID)}
    runtime.close()
