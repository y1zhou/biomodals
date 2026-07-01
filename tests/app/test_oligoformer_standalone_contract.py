"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

import tarfile
from pathlib import Path

import pytest

from biomodals.app.score import oligoformer_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


def test_run_oligoformer_builds_off_target_and_toxicity_command(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
    )
    info.repo_rnafm_dir.parent.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.mkdir(parents=True)
    info.model_rnafm_redevelop_dir.joinpath("model.pt").write_text(
        "weights", encoding="utf-8"
    )

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        output_arg = cmd[cmd.index("--output_dir") + 1]
        captured["output_arg"] = output_arg
        output_dir = Path(output_arg)
        captured["utr_bytes"] = Path(cmd[cmd.index("--utr") + 1]).read_bytes()
        captured["orf_bytes"] = Path(cmd[cmd.index("--orf") + 1]).read_bytes()
        output_dir.joinpath("result.csv").write_text("ok\n", encoding="utf-8")

    def fake_package_outputs(root):
        captured["root"] = Path(root)
        return b"archive"

    monkeypatch.setattr(oligoformer_app, "run_command", fake_run_command)
    monkeypatch.setattr(oligoformer_app, "package_outputs", fake_package_outputs)
    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)

    result = oligoformer_app.run_oligoformer.get_raw_f()(
        mrna_fasta_bytes=b">m\nAUGCUAGCUAGCUAGCUAGCUAGC\n",
        run_name="demo",
        off_target=True,
        toxicity=True,
        utr_bytes=b">utr\nAUGC\n",
        orf_bytes=b">orf\nAUGC\n",
        top_n=100,
        functionality_filter=False,
    )

    cmd = captured["cmd"]
    assert result == b"archive"
    assert cmd[:5] == ["python", "scripts/main.py", "-i", "1", "-i1"]
    assert "-off" in cmd
    assert "-tox" in cmd
    assert "--no_func" in cmd
    assert cmd[cmd.index("-top") + 1] == "100"
    assert captured["utr_bytes"] == b">utr\nAUGC\n"
    assert captured["orf_bytes"] == b">orf\nAUGC\n"
    assert captured["root"].name == "outputs"
    assert captured["output_arg"].endswith("/")
    assert captured["cwd"] == oligoformer_app.CONF.git_clone_dir
    assert (
        info.repo_rnafm_redevelop_dir.joinpath("model.pt").read_text(encoding="utf-8")
        == "weights"
    )


def test_download_oligoformer_models_writes_to_model_volume(
    tmp_path: Path, monkeypatch
) -> None:
    volume = FakeVolume()
    info = oligoformer_app.AppInfo(
        repo_rnafm_dir=tmp_path / "repo" / "RNA-FM",
        model_rnafm_dir=tmp_path / "models" / "RNA-FM",
    )
    captured = {}

    def fake_download_files(urls, *, force, num_retries, progress_bar_desc):
        captured["urls"] = urls
        captured["force"] = force
        captured["num_retries"] = num_retries
        captured["progress_bar_desc"] = progress_bar_desc
        archive_path = next(iter(urls.values()))
        Path(archive_path).parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as archive:
            payload = tmp_path / "model.pt"
            payload.write_text("weights", encoding="utf-8")
            archive.add(payload, arcname="RNA-FM/redevelop/model.pt")

    monkeypatch.setattr(oligoformer_app, "APP_INFO", info)
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", volume)
    monkeypatch.setattr(oligoformer_app, "download_files", fake_download_files)

    oligoformer_app.download_oligoformer_models.get_raw_f()(force=True)

    assert list(captured["urls"]) == [info.rnafm_archive_url]
    assert captured["force"] is True
    assert captured["num_retries"] == 3
    assert captured["progress_bar_desc"] == "OligoFormer model downloads"
    assert (
        info.model_rnafm_redevelop_dir.joinpath("model.pt").read_text(encoding="utf-8")
        == "weights"
    )
    assert volume.commit_count == 1


def test_submit_oligoformer_downloads_models_before_run(
    tmp_path: Path, monkeypatch
) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    calls = []

    class FakeDownloadModels:
        def remote(self, *, force):
            calls.append(("download", force))

    class FakeRunOligoFormer:
        def remote(self, **kwargs):
            calls.append(("run", kwargs))
            return b"archive"

    monkeypatch.setattr(
        oligoformer_app, "download_oligoformer_models", FakeDownloadModels()
    )
    monkeypatch.setattr(oligoformer_app, "run_oligoformer", FakeRunOligoFormer())
    raw_f = oligoformer_app.submit_oligoformer_task.info.raw_f
    assert raw_f is not None

    raw_f(mrna_fasta=str(input_fasta), out_dir=str(tmp_path), run_name="demo")

    assert calls[0] == ("download", False)
    assert calls[1][0] == "run"
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
