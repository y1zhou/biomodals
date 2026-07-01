"""Tests for standalone ENsiRNA app behavior."""

# ruff: noqa: D101,D102,D103,D107

from dataclasses import replace
from pathlib import Path

from biomodals.app.score import ensirna_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


def test_runtime_image_uses_rosetta_base_build() -> None:
    source = Path(ensirna_app.__file__).read_text(encoding="utf-8")

    assert 'from_registry("rosettacommons/rosetta:serial-420"' in source
    assert ".debian_slim(" not in source
    assert "tanwenchong/ensirna:v2" not in source
    assert '"MAMBA_ROOT_PREFIX": APP_INFO.mamba_root' in source
    assert '"PATH": APP_INFO.mamba_bin_path' in source
    assert "def rosetta_extract_shim" in source
    assert "rna_denovo.static.linuxgccrelease" in source
    assert "def get_pdb_runtime_patch" in source
    assert "expected_len = 61 + len(seq2) + len(seq1) + 1 + 1" in source
    assert "def _fit_secstruct(secstruct, size):" in source
    assert "-out:file:silent" in source
    assert "https://huggingface.co/cuhkaih/rnafm/resolve/main" in source
    assert "RNA-FM_pretrained.pth" in source
    assert "find . -path '*/pkl/*.ckpt' -delete" in source
    assert "download_ensirna_models" in source
    assert "MODEL_VOLUME.commit()" in source
    assert "download_files(" in source
    assert "curl -fL --retry" not in source
    assert "micromamba create" not in source
    assert ".micromamba_install(" in source
    assert "viennarna=2.6.4-0" in source
    assert ".uv_pip_install(*APP_INFO.pip_packages)" in source
    assert ".uv_pip_install(*APP_INFO.torch_packages" in source
    assert "https://download.pytorch.org/whl/cu118" in source
    assert "rna-fm" in source
    assert "ignore_dep_versions=True" in source
    assert 'skip_deps=["uniaf3"]' in source


def test_download_ensirna_models_writes_to_model_volume(monkeypatch) -> None:
    captured = {}
    volume = FakeVolume()

    def fake_download_files(urls, **kwargs):
        captured["urls"] = urls
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ensirna_app, "download_files", fake_download_files)
    monkeypatch.setattr(ensirna_app, "MODEL_VOLUME", volume)

    ensirna_app.download_ensirna_models.get_raw_f()(force=True)

    assert len(captured["urls"]) == 6
    assert ensirna_app.APP_INFO.rnafm_pretrained_url in captured["urls"]
    assert captured["kwargs"]["force"] is True
    assert captured["kwargs"]["num_retries"] == 3
    assert volume.commit_count == 1
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        assert (ensirna_app.APP_INFO.checkpoint_dir / filename) in captured[
            "urls"
        ].values()


def test_run_ensirna_uses_documented_design_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}
    ensirna_dir = tmp_path / "ENsiRNA"
    checkpoint_dir = tmp_path / "models" / "pkl"
    (ensirna_dir / "pkl").mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        (checkpoint_dir / filename).write_bytes(b"checkpoint")

    def fake_run_command(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

    def fake_package_outputs(root):
        captured["root"] = Path(root)
        return b"archive"

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    monkeypatch.setattr(ensirna_app, "package_outputs", fake_package_outputs)
    monkeypatch.setattr(
        ensirna_app,
        "APP_INFO",
        replace(
            ensirna_app.APP_INFO,
            ensirna_dir=ensirna_dir,
            checkpoint_dir=checkpoint_dir,
        ),
    )

    result = ensirna_app.run_ensirna.get_raw_f()(
        mrna_fasta_bytes=b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        run_name="demo",
    )

    assert result == b"archive"
    assert captured["cmd"][:6] == [
        "micromamba",
        "run",
        "-n",
        ensirna_app.APP_INFO.conda_env_name,
        "bash",
        "design.sh",
    ]
    assert Path(captured["cmd"][6]).name == "mrna.fasta"
    assert Path(captured["cmd"][7]).name == "outputs"
    assert captured["cwd"] == ensirna_dir
    assert captured["root"].name == "outputs"
    for filename in ensirna_app.APP_INFO.checkpoint_filenames:
        assert (ensirna_dir / "pkl" / filename).resolve() == checkpoint_dir / filename


def test_submit_ensirna_writes_local_tarball(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    captured = {}

    class FakeDownload:
        def remote(self, *, force: bool):
            captured["download_force"] = force

    class FakeRun:
        def remote(self, **kwargs):
            captured["run"] = kwargs
            return b"archive"

    monkeypatch.setattr(ensirna_app, "download_ensirna_models", FakeDownload())
    monkeypatch.setattr(ensirna_app, "run_ensirna", FakeRun())
    raw_f = ensirna_app.submit_ensirna_task.info.raw_f
    assert raw_f is not None

    raw_f(
        mrna_fasta=str(input_fasta),
        out_dir=str(tmp_path),
        run_name="demo",
    )

    assert captured["download_force"] is False
    assert captured["run"] == {
        "mrna_fasta_bytes": b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        "run_name": "demo",
    }
    assert (tmp_path / "demo_ensirna.tar.zst").read_bytes() == b"archive"
