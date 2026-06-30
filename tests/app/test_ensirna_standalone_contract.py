"""Tests for standalone ENsiRNA app behavior."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.score import ensirna_app


def test_run_ensirna_uses_documented_design_pipeline(monkeypatch) -> None:
    captured = {}
    volume = SimpleNamespace(commit_count=0)

    def commit():
        volume.commit_count += 1

    volume.commit = commit

    def fake_run_command(cmd):
        captured["cmd"] = cmd

    def fake_package_outputs(root):
        captured["root"] = Path(root)
        return b"archive"

    monkeypatch.setattr(ensirna_app, "run_command", fake_run_command)
    monkeypatch.setattr(ensirna_app, "package_outputs", fake_package_outputs)
    monkeypatch.setattr(ensirna_app, "MODEL_VOLUME", volume)

    result = ensirna_app.run_ensirna.get_raw_f()(
        mrna_fasta_bytes=b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        run_name="demo",
    )

    assert result == b"archive"
    assert captured["cmd"][:2] == ["bash", "-lc"]
    assert f"cd {ensirna_app.ENSIRNA_DIR};" in captured["cmd"][2]
    assert "bash design.sh" in captured["cmd"][2]
    assert captured["root"].name == "outputs"
    assert volume.commit_count == 1


def test_submit_ensirna_writes_local_tarball(tmp_path: Path, monkeypatch) -> None:
    input_fasta = tmp_path / "target.fa"
    input_fasta.write_text(">m\nAUGCUAGCUAGCUAGCUAGC\n", encoding="utf-8")
    captured = {}

    class FakeRun:
        def remote(self, **kwargs):
            captured.update(kwargs)
            return b"archive"

    monkeypatch.setattr(ensirna_app, "run_ensirna", FakeRun())
    raw_f = ensirna_app.submit_ensirna_task.info.raw_f
    assert raw_f is not None

    raw_f(
        mrna_fasta=str(input_fasta),
        out_dir=str(tmp_path),
        run_name="demo",
    )

    assert captured == {
        "mrna_fasta_bytes": b">m\nAUGCUAGCUAGCUAGCUAGC\n",
        "run_name": "demo",
    }
    assert (tmp_path / "demo_ensirna.tar.zst").read_bytes() == b"archive"
