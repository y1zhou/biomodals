"""Tests for standalone OligoFormer app behavior."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path

import pytest

from biomodals.app.score import oligoformer_app


class FakeVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


def test_run_oligoformer_builds_off_target_and_toxicity_command(monkeypatch) -> None:
    captured = {}
    volume = FakeVolume()

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
    monkeypatch.setattr(oligoformer_app, "MODEL_VOLUME", volume)

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
    assert volume.commit_count == 1


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
