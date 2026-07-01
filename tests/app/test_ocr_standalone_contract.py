"""Tests for OCR app archive and staging behavior."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path

from biomodals.app.misc import ocr_app
from biomodals.helper.catalog import get_catalog
from biomodals.helper.shell import run_command


def _tar_zst_names(content: bytes, tmp_path: Path) -> set[str]:
    archive_path = tmp_path / "archive.tar.zst"
    archive_path.write_bytes(content)
    return set(
        run_command(
            ["tar", "-I", "zstd", "-tf", str(archive_path)],
            output_mode="capture",
        )
    )


def test_ocr_replaces_paddleocr_in_app_catalog() -> None:
    apps = get_catalog("app", use_absolute_paths=True)

    assert "ocr" in apps
    assert apps["ocr"].name == "ocr_app.py"
    assert "paddleocr" not in apps


def test_run_mineru_ocr_packages_hybrid_outputs(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_run_command(cmd, *, output_mode, log_file, env):
        calls["cmd"] = cmd
        calls["output_mode"] = output_mode
        calls["log_file"] = log_file
        calls["env"] = env
        out_dir = Path(cmd[cmd.index("-o") + 1])
        pdf_path = Path(cmd[cmd.index("-p") + 1])
        hybrid_dir = out_dir / pdf_path.stem / "hybrid_auto"
        hybrid_dir.mkdir(parents=True)
        (hybrid_dir / f"{pdf_path.stem}_model.json").write_text("[]\n")
        Path(log_file).write_text("mineru log\n")
        return []

    monkeypatch.setattr(ocr_app, "run_command", fake_run_command)

    archive = ocr_app.run_mineru_ocr.get_raw_f()(b"%PDF-1.7\n", "demo.pdf", "high")

    assert calls["cmd"][:5] == ["mineru", "-b", "hybrid-engine", "--effort", "high"]
    assert calls["output_mode"] == "tee"
    assert calls["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    names = _tar_zst_names(archive, tmp_path)
    assert "demo/mineru.log" in names
    assert "demo/hybrid_auto/demo_model.json" in names


def test_run_mineru_popo_stages_mineru_hybrid_as_vlm(monkeypatch, tmp_path) -> None:
    mineru_root = tmp_path / "demo"
    hybrid_dir = mineru_root / "hybrid_auto"
    hybrid_dir.mkdir(parents=True)
    (hybrid_dir / "demo_model.json").write_text("[]\n")
    mineru_archive = ocr_app.package_outputs(mineru_root)
    calls = []
    original_run_command = ocr_app.run_command

    def fake_run_command(cmd, *, output_mode, log_file=None, cwd=None, env=None):
        if cmd[0] == "tar":
            return original_run_command(cmd, output_mode=output_mode)
        calls.append((cmd, log_file, env))
        command_name = Path(cmd[1]).name
        if command_name == "label_normalization.py":
            input_root = Path(cmd[cmd.index("--input-dir") + 1])
            assert (input_root / "demo" / "vlm" / "demo_model.json").is_file()
            output_dir = Path(cmd[cmd.index("--output-dir") + 1]) / "mineru"
            output_dir.mkdir(parents=True)
            (output_dir / "demo.json").write_text('{"pages": {}}\n')
        elif command_name == "run_inference.py":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "demo.json").write_text("[]\n")
        elif command_name == "get_json_tree.py":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            txt_dir = Path(cmd[cmd.index("--txt-dir") + 1])
            output_dir.mkdir(parents=True)
            txt_dir.mkdir(parents=True)
            (output_dir / "demo.json").write_text("{}\n")
            (txt_dir / "demo.txt").write_text("tree\n")
        else:
            raise AssertionError(cmd)
        assert log_file is not None
        Path(log_file).write_text("log\n")
        return []

    monkeypatch.setattr(ocr_app, "run_command", fake_run_command)

    archive = ocr_app.run_mineru_popo.get_raw_f()(
        mineru_archive,
        b"%PDF-1.7\n",
        "demo.pdf",
        32,
    )

    assert calls[1][2]["POPO_MAX_NEW_TOKENS"] == "32"
    names = _tar_zst_names(archive, tmp_path)
    assert "popo-results/outputs/label_normalization/mineru/demo.json" in names
    assert "popo-results/outputs/inference/mineru/demo.json" in names
    assert "popo-results/outputs/build_tree/mineru/demo.json" in names
    assert "popo-results/outputs/build_tree_txt/mineru/demo.txt" in names
    assert "popo-results/logs/inference.log" in names
