"""Tests for the BoltzGen compressor app."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.misc import compress_app


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_compress_one_run_uses_app_run_layout_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = {}
    output_volume = FakeOutputVolume()
    run_root = tmp_path / "demo"
    run_root.joinpath("outputs").mkdir(parents=True)
    run_root.joinpath("outputs", "result.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        compress_app,
        "BG_CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
        ),
    )

    def fake_warmup_directory(path):
        calls["warmup_directory"] = path

    def fake_package_outputs(path):
        calls["package_outputs"] = path
        return b"tar.zst"

    monkeypatch.setattr(compress_app, "warmup_directory", fake_warmup_directory)
    monkeypatch.setattr(compress_app, "package_outputs", fake_package_outputs)

    assert compress_app.compress_one_run.get_raw_f()("demo") == "demo.tar.zst"
    assert calls["warmup_directory"] == run_root
    assert calls["package_outputs"] == run_root
    assert run_root.with_suffix(".tar.zst").read_bytes() == b"tar.zst"
    assert output_volume.reload_count == 1
    assert output_volume.commit_count == 1
