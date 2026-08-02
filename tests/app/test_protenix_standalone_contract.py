"""Tests for standalone Protenix app behavior."""

# ruff: noqa: D101,D102,D103,D107

import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

import orjson

from biomodals.app.fold import protenix_app
from biomodals.execution import RunStatus


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        pass


def test_msa_cache_uses_an_absolute_modal_mount() -> None:
    assert Path(protenix_app.APP_INFO.msa_cache_mountpoint).is_absolute()


def test_publication_validators_reject_same_size_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_output = tmp_path / "msa"
    task_output.mkdir()
    expected = task_output / "updated.json"
    expected.write_bytes(b"{}")
    task = protenix_app.ProtenixMsaTaskSpec(
        task_key="task-0",
        input_name="input",
        query_command="msa",
        input_json_path=str(tmp_path / "input.json"),
        output_dir=str(task_output),
        msa_server_mode="protenix",
        expected_json_path=str(expected),
        publication_key="msa-key",
    )
    protenix_app._atomic_write(
        protenix_app._msa_task_marker_path(task),
        orjson.dumps({
            "publication_key": task.publication_key,
            "expected_json_path": str(expected),
            "size": 2,
            "sha256": sha256(b"{}").hexdigest(),
        }),
    )

    prepared = tmp_path / "prepared.json"
    prepared.write_bytes(b"{}")
    plan = protenix_app.ProtenixPreparationPlan(
        preparation_key="prepared-key",
        prepared_json_path=str(prepared),
        tasks=(task,),
    )
    protenix_app._atomic_write(
        protenix_app._prepared_marker_path(plan),
        orjson.dumps({
            "preparation_key": plan.preparation_key,
            "size": 2,
            "sha256": sha256(b"{}").hexdigest(),
        }),
    )

    monkeypatch.setattr(
        protenix_app,
        "CONF",
        SimpleNamespace(output_volume_mountpoint=str(tmp_path)),
    )
    result = protenix_app._result_path("result-key", "demo")
    result.parent.mkdir(parents=True)
    result.write_bytes(b"ab")
    result.with_suffix(f"{result.suffix}.complete.json").write_bytes(
        orjson.dumps({
            "result_key": "result-key",
            "size": 2,
            "sha256": sha256(b"ab").hexdigest(),
        })
    )

    assert protenix_app._msa_task_ready(task)
    assert protenix_app._prepared_ready(plan)
    assert protenix_app._result_ready("result-key", "demo")

    expected.write_bytes(b"[]")
    prepared.write_bytes(b"[]")
    result.write_bytes(b"cd")

    assert not protenix_app._msa_task_ready(task)
    assert not protenix_app._prepared_ready(plan)
    assert not protenix_app._result_ready("result-key", "demo")


def test_planner_discovers_one_task_per_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    volume = FakeVolume()
    jobs = [
        SimpleNamespace(
            name=f"job-{index}",
            sequences=[
                SimpleNamespace(
                    proteinChain=SimpleNamespace(sequence=f"PROTEIN{index}"),
                    rnaSequence=None,
                )
            ],
        )
        for index in range(2)
    ]

    class FakeConfig:
        def __init__(self, root):
            self.root = list(root)

        @classmethod
        def from_file(cls, _path):
            return cls(jobs)

        def to_files(self, output_dir, name):
            Path(output_dir, f"{name}.json").write_text(name)

    schema = ModuleType("uniaf3.schema")
    schema.ProtenixConfig = FakeConfig  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "uniaf3", ModuleType("uniaf3"))
    monkeypatch.setitem(sys.modules, "uniaf3.schema", schema)
    monkeypatch.setattr(
        protenix_app,
        "APP_INFO",
        replace(protenix_app.APP_INFO, msa_cache_mountpoint=str(tmp_path / "msa")),
    )
    monkeypatch.setattr(protenix_app, "MSA_CACHE_VOLUME", volume)
    monkeypatch.setattr(
        protenix_app,
        "CONF",
        SimpleNamespace(repo_commit_hash="7e1de70", version="2.0.0"),
    )

    plan = protenix_app.plan_protenix_inputs.get_raw_f()(
        b'[{"name":"demo"}]',
        msa_server_mode="protenix",
    )

    assert [task.task_key for task in plan.tasks] == ["0000-job-0", "0001-job-1"]
    assert all(Path(task.input_json_path).is_file() for task in plan.tasks)
    assert volume.commit_count == 1


def test_local_entrypoint_launches_one_execution_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name":"demo"}]')
    execution_run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    captured = {}

    class FakeOutputVolume:
        def read_file(self, path):
            captured["download"] = path
            yield b"tar"

    class FakeMethod:
        def spawn(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(
                object_id="fc-1",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=RunStatus.SUCCEEDED,
                        status_message=None,
                        status_reason=None,
                    )
                ),
            )

    def stage(volume, run_id, request):
        captured.update(volume=volume, run_id=run_id, request=request)

    def coordinator_handle(**kwargs):
        captured["handle_kwargs"] = kwargs
        return SimpleNamespace(run=FakeMethod())

    volume = FakeOutputVolume()
    monkeypatch.setattr(
        protenix_app,
        "CONF",
        SimpleNamespace(
            name="Protenix",
            version="2.0.0",
            repo_commit_hash="7e1de70",
            output_volume=volume,
            output_volume_mountpoint="/protenix-output",
        ),
    )
    monkeypatch.setattr(protenix_app, "uuid4", lambda: execution_run_id)
    monkeypatch.setattr(protenix_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        protenix_app,
        "_execution_coordinator_handle",
        coordinator_handle,
    )
    monkeypatch.setattr(
        protenix_app,
        "write_local_tarball",
        lambda path, data: captured.update(out_file=path, data=data),
    )
    raw = protenix_app.submit_protenix_task.info.raw_f
    assert raw is not None

    raw(
        input_file=str(input_path),
        out_dir=str(tmp_path / "results"),
        run_name="../demo",
        max_parallel_msa=3,
    )

    assert captured["request"].run_name == "demo"
    assert captured["request"].max_active_provider_calls == 3
    assert captured["run_kwargs"] == {"development": True}
    assert captured["data"] == b"tar"
