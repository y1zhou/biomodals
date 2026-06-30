"""Tests for ABCFold2 app run layout behavior."""

# ruff: noqa: D101,D102,D103,D107

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from biomodals.app.fold import abcfold2_app


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


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
