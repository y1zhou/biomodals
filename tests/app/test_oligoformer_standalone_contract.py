"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.score import oligoformer_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def _fake_conf(tmp_path: Path, volume: FakeVolume, repo_dir: Path | None = None):
    return SimpleNamespace(
        name=oligoformer_app.CONF.name,
        version=oligoformer_app.CONF.version,
        repo_commit_hash=oligoformer_app.CONF.repo_commit_hash,
        output_volume_mountpoint=str(tmp_path / "outputs-volume"),
        git_clone_dir=repo_dir or tmp_path / "repo",
        output_volume=volume,
    )


def test_prepare_oligoformer_run_writes_volume_inputs(tmp_path: Path, monkeypatch):
    volume = FakeVolume()
    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume))

    result = oligoformer_app.prepare_oligoformer_run.get_raw_f()(
        mrna_fasta_bytes=b">target one\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        sirna_fasta_bytes=b">s\nAUGCUAGCUAGCUAGCUAG\n",
        off_target=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
    )

    input_dir = Path(result.input_dir)
    assert result.output_stems == ("target_@_one",)
    assert input_dir.joinpath("mrna.fa").read_bytes().startswith(b">target one")
    assert input_dir.joinpath("sirna.fa").read_bytes().startswith(b">s")
    assert input_dir.joinpath("utr.txt").read_bytes() == b">utr\nAUGC\n"
    assert input_dir.joinpath("orf.txt").read_bytes() == b">orf\nAUGC\n"
    assert result.efficacy_ready is False
    assert result.final_ready is False
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_run_oligoformer_efficacy_builds_gpu_stage_command(tmp_path: Path, monkeypatch):
    captured = {}
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=repo_dir / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=repo_dir / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    info.model_rnafm_redevelop_dir.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.joinpath("model.pt").write_text(
        "weights", encoding="utf-8"
    )
    input_dir = tmp_path / "run" / "inputs"
    efficacy_dir = tmp_path / "run" / "prepare" / "efficacy"
    input_dir.mkdir(parents=True)
    input_dir.joinpath("mrna.fa").write_text(">target\nAUGCUAGCUAGCUAGCUAGC\n")
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root=str(tmp_path / "run"),
        input_dir=str(input_dir),
        efficacy_dir=str(efficacy_dir),
        output_dir=str(tmp_path / "run" / "outputs"),
        output_stems=("target",),
        efficacy_ready=False,
        final_ready=False,
    )

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "_ranked", "_ranked_filtered"):
            out_dir.joinpath(f"target{suffix}.txt").write_text("ok\n")

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)

    result = oligoformer_app.run_oligoformer_efficacy.get_raw_f()(
        plan=plan,
        functionality_filter=False,
    )

    assert captured["cmd"][:5] == ["python", "scripts/main.py", "-i", "1", "-i1"]
    assert "--biomodals_stage" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--biomodals_stage") + 1] == "efficacy"
    assert "--no_func" in captured["cmd"]
    assert "-off" not in captured["cmd"]
    assert "-tox" not in captured["cmd"]
    assert captured["cwd"] == repo_dir
    assert result.efficacy_ready is True
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_run_oligoformer_postprocess_packages_cpu_outputs(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {"commands": []}
    volume = FakeVolume()
    repo_dir = tmp_path / "repo"
    repo_dir.joinpath("toxicity").mkdir(parents=True)
    repo_dir.joinpath("toxicity", "cell_viability.txt").write_text(
        "Seed\tcell_viability\nUGCUA\t40\n",
        encoding="utf-8",
    )
    layout = oligoformer_app.AppRunLayout.from_run_root(tmp_path / "run")
    layout.prep_dir.joinpath("efficacy").mkdir(parents=True)
    layout.inputs_dir.mkdir(parents=True)
    layout.inputs_dir.joinpath("utr.txt").write_text(">utr\nAUGC\n", encoding="utf-8")
    layout.inputs_dir.joinpath("orf.txt").write_text(">orf\nAUGC\n", encoding="utf-8")
    layout.prep_dir.joinpath("efficacy", "target.txt").write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tfilter\n"
        "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t0\n",
        encoding="utf-8",
    )
    for suffix in ("_ranked", "_ranked_filtered"):
        layout.prep_dir.joinpath("efficacy", f"target{suffix}.txt").write_text(
            "placeholder\n", encoding="utf-8"
        )
    layout.markers_dir.mkdir(parents=True)
    oligoformer_app._marker_path(layout, "efficacy.done").write_text(
        "{}", encoding="utf-8"
    )
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root=str(layout.run_root),
        input_dir=str(layout.inputs_dir),
        efficacy_dir=str(layout.prep_dir / "efficacy"),
        output_dir=str(layout.outputs_dir),
        output_stems=("target",),
        efficacy_ready=True,
        final_ready=False,
    )

    def fake_run_command(cmd, *, cwd):
        commands = captured["commands"]
        assert isinstance(commands, list)
        commands.append(cmd)
        infer_dir = repo_dir / "data/infer/target"
        infer_dir.mkdir(parents=True, exist_ok=True)
        if cmd[1] == "scripts/pita.sh":
            infer_dir.joinpath("pita.tab").write_text(
                "microRNA\tScore\nRNA0\t-11\n", encoding="utf-8"
            )
        else:
            infer_dir.joinpath("targetscan.tab").write_text(
                "ref\tRNA0\t2\n", encoding="utf-8"
            )

    def fake_package_outputs(root):
        captured["package_root"] = Path(root)
        return b"archive"

    monkeypatch.setattr(oligoformer_app, "CONF", _fake_conf(tmp_path, volume, repo_dir))
    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    monkeypatch.setattr(oligoformer_app, "package_outputs", fake_package_outputs)

    result = oligoformer_app.run_oligoformer_postprocess.get_raw_f()(
        plan=plan,
        off_target=True,
        toxicity=True,
        top_n=-1,
    )

    final_table = layout.outputs_dir.joinpath("target.txt").read_text(encoding="utf-8")
    assert result == b"archive"
    commands = captured["commands"]
    assert isinstance(commands, list)
    assert commands[0][1] == "scripts/pita.sh"
    assert commands[1][1] == "scripts/targetscan.sh"
    assert captured["package_root"] == layout.outputs_dir
    assert "off_target_filter" in final_table
    assert "toxicity_filter" in final_table
    assert volume.reload_count == 1
    assert volume.commit_count == 1


def test_download_oligoformer_models_writes_to_model_volume(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path(oligoformer_app.__file__).read_text(encoding="utf-8")
    volume = FakeVolume()
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
        repo_ref_dir=tmp_path / "repo" / "off-target" / "ref",
        model_ref_dir=tmp_path / "models" / "off-target" / "ref",
    )
    calls = []

    def fake_download_files(urls, *, force, num_retries, progress_bar_desc):
        calls.append({
            "urls": urls,
            "force": force,
            "num_retries": num_retries,
            "progress_bar_desc": progress_bar_desc,
        })
        for archive_path in urls.values():
            Path(archive_path).parent.mkdir(parents=True, exist_ok=True)
            if str(archive_path).endswith(".tar.gz"):
                with tarfile.open(archive_path, "w:gz") as archive:
                    payload = tmp_path / "model.pt"
                    payload.write_text("weights", encoding="utf-8")
                    archive.add(payload, arcname="RNA-FM/redevelop/model.pt")
            else:
                ref_name = Path(archive_path).name.removesuffix(".zip")
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(ref_name, f">{ref_name}\nAUGC\n")

    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", volume)
    monkeypatch.setattr(oligoformer_app, "download_files", fake_download_files)

    oligoformer_app.download_oligoformer_models.get_raw_f()(force=True)

    assert list(calls[0]["urls"]) == [info.rnafm_archive_url]
    assert calls[0]["force"] is True
    assert calls[0]["num_retries"] == 3
    assert calls[0]["progress_bar_desc"] == "OligoFormer model downloads"
    assert calls[1]["urls"] == info.human_ref_downloads
    assert calls[1]["progress_bar_desc"] == "OligoFormer human ref downloads"
    assert (
        info.model_rnafm_redevelop_dir.joinpath("model.pt").read_text(encoding="utf-8")
        == "weights"
    )
    for ref_path in info.model_human_ref_paths:
        assert ref_path.read_text(encoding="utf-8") == f">{ref_path.name}\nAUGC\n"
    assert volume.commit_count == 1
    assert "min(Args.top_n, RESULT_ranked.shape[0])" in source
    assert "--biomodals_stage" in source


def test_submit_oligoformer_orchestrates_split_run(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    calls = []
    plan = oligoformer_app.OligoformerRunPlan(
        cache_key="abc123",
        run_root="/remote/run",
        input_dir="/remote/run/inputs",
        efficacy_dir="/remote/run/prepare/efficacy",
        output_dir="/remote/run/outputs",
        output_stems=("m",),
        efficacy_ready=False,
        final_ready=False,
    )
    efficacy_plan = replace(plan, efficacy_ready=True)

    class FakeDownloadModels:
        def remote(self, *, force):
            calls.append(("download", force))

    class FakePrepare:
        def remote(self, **kwargs):
            calls.append(("prepare", kwargs))
            return plan

    class FakeEfficacy:
        def remote(self, **kwargs):
            calls.append(("efficacy", kwargs))
            return efficacy_plan

    class FakePostprocess:
        def remote(self, **kwargs):
            calls.append(("postprocess", kwargs))
            return b"archive"

    monkeypatch.setattr(
        oligoformer_app, "download_oligoformer_models", FakeDownloadModels()
    )
    monkeypatch.setattr(oligoformer_app, "prepare_oligoformer_run", FakePrepare())
    monkeypatch.setattr(oligoformer_app, "run_oligoformer_efficacy", FakeEfficacy())
    monkeypatch.setattr(
        oligoformer_app, "run_oligoformer_postprocess", FakePostprocess()
    )
    raw_f = oligoformer_app.submit_oligoformer_task.info.raw_f
    assert raw_f is not None

    raw_f(mrna_fasta=str(input_fasta), out_dir=str(tmp_path), run_name="demo")

    assert calls[0] == ("download", False)
    assert calls[1][0] == "prepare"
    assert calls[1][1]["top_n"] == oligoformer_app.APP_INFO.default_top_n
    assert calls[2] == (
        "efficacy",
        {"plan": plan, "functionality_filter": True},
    )
    assert calls[3][0] == "postprocess"
    assert calls[3][1]["plan"] == efficacy_plan
    assert (tmp_path / "demo_oligoformer.tar.zst").read_bytes() == b"archive"


def test_submit_oligoformer_requires_off_target_references(tmp_path: Path) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    raw_f = oligoformer_app.submit_oligoformer_task.info.raw_f
    assert raw_f is not None

    with pytest.raises(ValueError, match="--utr-file and --orf-file"):
        raw_f(
            mrna_fasta=str(input_fasta),
            out_dir=str(tmp_path),
            off_target=True,
        )
