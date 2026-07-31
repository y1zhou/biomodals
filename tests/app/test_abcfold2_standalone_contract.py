"""Tests for ABCFold2 app run layout behavior."""

# ruff: noqa: D101,D102,D103,D107

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID

from biomodals.app.fold import abcfold2_app
from biomodals.app.fold.abcfold2_execution import ABCFold2RunConfig
from biomodals.execution import RunStatus


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        pass


def test_prepare_abcfold2_uses_hash_partitioned_app_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "abcdef-no-tmpl"
    calls = {}
    output_volume = FakeOutputVolume()
    prepare_module = ModuleType("abcfold.cli.prepare")

    def fake_search_msa(
        conf_file,
        out_dir,
        force,
        chains,
        search_templates,
        template_cache_dir,
    ):
        calls["search_msa"] = {
            "conf_file": conf_file,
            "out_dir": out_dir,
            "force": force,
            "chains": chains,
            "search_templates": search_templates,
            "template_cache_dir": template_cache_dir,
        }
        Path(out_dir).joinpath(f"{run_id}.yaml").write_bytes(
            Path(conf_file).read_bytes()
        )

    def fake_prepare_boltz(conf_file, out_dir):
        calls["prepare_boltz"] = {"conf_file": conf_file, "out_dir": out_dir}
        boltz_yaml = Path(out_dir) / "boltz_models" / f"{run_id}.yaml"
        boltz_yaml.parent.mkdir(parents=True)
        boltz_yaml.write_text("boltz\n", encoding="utf-8")

    def fake_prepare_chai(conf_file, out_dir, ccd_lib_dir):
        calls["prepare_chai"] = {
            "conf_file": conf_file,
            "out_dir": out_dir,
            "ccd_lib_dir": ccd_lib_dir,
        }
        chai_yaml = Path(out_dir) / "chai_models" / f"{run_id}.yaml"
        chai_yaml.parent.mkdir(parents=True)
        chai_yaml.write_text("chai\n", encoding="utf-8")

    prepare_module.search_msa = fake_search_msa
    prepare_module.prepare_boltz = fake_prepare_boltz
    prepare_module.prepare_chai = fake_prepare_chai
    monkeypatch.setitem(sys.modules, "abcfold", ModuleType("abcfold"))
    monkeypatch.setitem(sys.modules, "abcfold.cli", ModuleType("abcfold.cli"))
    monkeypatch.setitem(sys.modules, "abcfold.cli.prepare", prepare_module)
    monkeypatch.setattr(
        abcfold2_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        abcfold2_app,
        "get_run_id",
        SimpleNamespace(local=lambda yaml_str: "abcdef"),
    )
    monkeypatch.setattr(
        abcfold2_app,
        "load_params_from_run_yaml",
        lambda yaml_path: {
            "seeds": [7],
            "num_trunk_recycles": 1,
            "num_diffn_timesteps": 2,
            "num_diffn_samples": 3,
            "num_trunk_samples": 4,
            "boltz_additional_cli_args": None,
        },
    )

    result = abcfold2_app.prepare_abcfold2.get_raw_f()(
        yaml_str=b"name: demo\n",
        search_templates=False,
        msa_chains="A",
    )

    run_root = tmp_path / "ab" / run_id
    search_call = calls["search_msa"]
    assert result["run_id"] == run_id
    assert result["workdir"] == str(run_root)
    assert Path(search_call["conf_file"]).name == f"{run_id}.yaml"
    assert search_call["out_dir"] == run_root
    assert search_call["force"] is True
    assert search_call["chains"] == "A"
    assert search_call["search_templates"] is False
    assert search_call["template_cache_dir"] == tmp_path / ".cache" / "rcsb"
    assert calls["prepare_boltz"]["conf_file"] == run_root / f"{run_id}.yaml"
    assert calls["prepare_boltz"]["out_dir"] == run_root
    assert calls["prepare_chai"]["conf_file"] == run_root / f"{run_id}.yaml"
    assert calls["prepare_chai"]["out_dir"] == run_root
    assert output_volume.commit_count == 3


def test_collectors_only_package_completed_seed_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    volume = FakeOutputVolume()
    workdir = tmp_path / "run"
    boltz_dir = workdir / "boltz_models"
    chai_dir = workdir / "chai_models"
    boltz_dir.mkdir(parents=True)
    chai_dir.mkdir()
    (boltz_dir / "run.yaml").write_text("boltz")
    (chai_dir / "run.yaml").write_text("chai")
    monkeypatch.setattr(
        abcfold2_app,
        "CONF",
        SimpleNamespace(output_volume=volume),
    )
    monkeypatch.setattr(
        abcfold2_app, "package_outputs", lambda *_args, **_kwargs: b"tar"
    )
    run_conf = {"workdir": str(workdir), "run_id": "run"}

    boltz = abcfold2_app.collect_abcfold2_boltz_data.get_raw_f()(
        run_conf,
        publication_key="boltz-key",
    )
    chai = abcfold2_app.collect_abcfold2_chai_data.get_raw_f()(
        run_conf,
        publication_key="chai-key",
    )

    assert Path(str(boltz["archive_path"])).read_bytes() == b"tar"
    assert Path(str(chai["archive_path"])).read_bytes() == b"tar"
    assert volume.commit_count == 2


def test_seed_publication_rejects_an_old_parameter_key(tmp_path: Path) -> None:
    result = tmp_path / "boltz_models" / "boltz_results_seed-1"
    result.mkdir(parents=True)
    abcfold2_app._write_publication_marker(
        tmp_path / ".biomodals" / "boltz-seed-1.json",
        {"publication_key": "old", "result_path": str(result)},
    )

    assert not abcfold2_app._seed_ready(tmp_path, "boltz", 1, "new")


def test_local_entrypoint_launches_one_execution_coordinator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text("name: demo\n")
    output_dir = tmp_path / "results"
    execution_run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    captured = {}

    class FakeVolume:
        def read_file(self, path):
            captured.setdefault("downloads", []).append(path)
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

    volume = FakeVolume()
    monkeypatch.setattr(
        abcfold2_app,
        "CONF",
        SimpleNamespace(
            name="ABCFold2",
            version="0.2.0",
            repo_commit_hash="fcfdd49",
            output_volume=volume,
            output_volume_mountpoint="/abcfold2-output",
        ),
    )
    monkeypatch.setattr(abcfold2_app, "uuid4", lambda: execution_run_id)
    monkeypatch.setattr(abcfold2_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        abcfold2_app,
        "_execution_coordinator_handle",
        coordinator_handle,
    )
    monkeypatch.setattr(
        abcfold2_app,
        "run_config_from_snapshot",
        lambda _snapshot: ABCFold2RunConfig(
            run_id="abcdef-no-tmpl",
            workdir="/abcfold2-output/ab/abcdef-no-tmpl",
            seeds=(1,),
            num_trunk_recycles=1,
            num_diffn_timesteps=2,
            num_diffn_samples=3,
            num_trunk_samples=4,
            boltz_additional_cli_args=None,
        ),
    )
    raw = abcfold2_app.submit_abcfold2_task.info.raw_f
    assert raw is not None

    raw(
        input_yaml=str(input_yaml),
        out_dir=str(output_dir),
        run_name="demo",
        run_boltz=True,
        run_chai=True,
        max_parallel_children=3,
    )

    local = output_dir / "demo-no-tmpl"
    assert captured["request"].max_active_provider_calls == 3
    assert captured["run_kwargs"] == {"development": True}
    assert (local / "run-config.json").is_file()
    assert (local / "boltz_models.tar.zst").read_bytes() == b"tar"
    assert (local / "chai_models.tar.zst").read_bytes() == b"tar"
