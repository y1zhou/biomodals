"""Tests for OCR app archive and staging behavior."""

# ruff: noqa: D101,D102,D103,D107

import sys
import types
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


def test_ocr_app_configs_use_tool_specific_model_store_paths() -> None:
    assert ocr_app.CONF.name == "MinerU"
    assert ocr_app.POPO_CONF.name == "MinerU-Popo"
    assert ocr_app.MINERU_CONFIG_PATH == Path("/biomodals-store/MinerU/mineru.json")
    assert ocr_app.VLLM_CACHE_ROOT == Path("/biomodals-store/MinerU/vllm-cache")
    assert ocr_app.POPO_HF_CACHE_DIR == Path("/biomodals-store/huggingface/hub")
    assert ocr_app.POPO_HF_CACHE_DIR / "models--DreamEternal--MinerU-Popo" == Path(
        "/biomodals-store/huggingface/hub/models--DreamEternal--MinerU-Popo"
    )


def test_run_mineru_ocr_packages_hybrid_outputs(monkeypatch, tmp_path) -> None:
    calls = {}

    class FakeModelVolume:
        def commit(self) -> None:
            calls["committed"] = True

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
    monkeypatch.setattr(ocr_app, "VLLM_CACHE_ROOT", tmp_path / "vllm-cache")
    monkeypatch.setattr(ocr_app, "MODEL_VOLUME", FakeModelVolume())

    archive = ocr_app.run_mineru_ocr.get_raw_f()(b"%PDF-1.7\n", "demo.pdf", "high")

    assert calls["cmd"][:5] == ["mineru", "-b", "hybrid-engine", "--effort", "high"]
    assert calls["output_mode"] == "tee"
    assert calls["env"]["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm-cache")
    assert calls["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert calls["committed"] is True
    names = _tar_zst_names(archive, tmp_path)
    assert "demo/mineru.log" in names
    assert "demo/hybrid_auto/demo_model.json" in names


def test_download_popo_model_weights_passes_hf_token(monkeypatch, tmp_path) -> None:
    calls = {}
    snapshot_path = (
        tmp_path / "hub" / "models--DreamEternal--MinerU-Popo" / "snapshots" / "abc123"
    )

    class FakeHub(types.ModuleType):
        def snapshot_download(self, **kwargs) -> str:
            calls["snapshot_download"] = kwargs
            return str(snapshot_path)

    class FakeModelVolume:
        def commit(self) -> None:
            calls["committed"] = True

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub("huggingface_hub"))
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(ocr_app, "POPO_HF_CACHE_DIR", tmp_path / "hub")
    monkeypatch.setattr(ocr_app, "MODEL_VOLUME", FakeModelVolume())

    ocr_app.download_popo_model_weights.get_raw_f()(force=True)

    assert calls["snapshot_download"] == {
        "repo_id": "DreamEternal/MinerU-Popo",
        "cache_dir": tmp_path / "hub",
        "token": "secret-token",
        "force_download": True,
        "local_files_only": False,
    }
    assert calls["committed"] is True


def test_download_popo_model_weights_allows_missing_hf_token(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {}
    snapshot_path = (
        tmp_path / "hub" / "models--DreamEternal--MinerU-Popo" / "snapshots" / "abc123"
    )

    class FakeHub(types.ModuleType):
        def snapshot_download(self, **kwargs) -> str:
            calls["snapshot_download"] = kwargs
            return str(snapshot_path)

    class FakeModelVolume:
        def commit(self) -> None:
            calls["committed"] = True

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub("huggingface_hub"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(ocr_app, "POPO_HF_CACHE_DIR", tmp_path / "hub")
    monkeypatch.setattr(ocr_app, "MODEL_VOLUME", FakeModelVolume())

    ocr_app.download_popo_model_weights.get_raw_f()(force=False)

    assert calls["snapshot_download"]["token"] is None
    assert calls["snapshot_download"]["cache_dir"] == tmp_path / "hub"
    assert calls["snapshot_download"]["local_files_only"] is False
    assert calls["committed"] is True


def test_run_mineru_popo_stages_mineru_hybrid_as_vlm(monkeypatch, tmp_path) -> None:
    snapshot_path = (
        tmp_path / "hub" / "models--DreamEternal--MinerU-Popo" / "snapshots" / "abc123"
    )
    mineru_root = tmp_path / "demo"
    hybrid_dir = mineru_root / "hybrid_auto"
    hybrid_dir.mkdir(parents=True)
    (hybrid_dir / "demo_model.json").write_text("[]\n")
    mineru_archive = ocr_app.package_outputs(mineru_root)
    calls = []
    original_run_command = ocr_app.run_command

    class FakeHub(types.ModuleType):
        def snapshot_download(self, **kwargs) -> str:
            calls.append(("snapshot_download", kwargs, None))
            return str(snapshot_path)

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
            assert Path(cmd[cmd.index("--model-path") + 1]) == snapshot_path
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
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub("huggingface_hub"))
    monkeypatch.setattr(ocr_app, "POPO_HF_CACHE_DIR", tmp_path / "hub")

    archive = ocr_app.run_mineru_popo.get_raw_f()(
        mineru_archive,
        b"%PDF-1.7\n",
        "demo.pdf",
        32,
    )

    snapshot_call = calls[0]
    assert snapshot_call == (
        "snapshot_download",
        {
            "repo_id": "DreamEternal/MinerU-Popo",
            "cache_dir": tmp_path / "hub",
            "token": None,
            "force_download": False,
            "local_files_only": True,
        },
        None,
    )
    assert calls[2][2]["POPO_MAX_NEW_TOKENS"] == "32"
    names = _tar_zst_names(archive, tmp_path)
    assert "popo-results/label_normalization/mineru/demo.json" in names
    assert "popo-results/inference/mineru/demo.json" in names
    assert "popo-results/build_tree/mineru/demo.json" in names
    assert "popo-results/build_tree_txt/mineru/demo.txt" in names
    assert "popo-results/logs/inference.log" in names


def test_ocr_local_entrypoint_nests_popo_results_without_popo_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "demo.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "demo_mineru.tar.zst").write_bytes(b"stale")
    (out_dir / "demo_popo.tar.zst").write_bytes(b"stale")
    calls = {}

    class FakeMinerUOCR:
        def remote(self, pdf_content: bytes, input_name: str, effort: str) -> bytes:
            calls["mineru"] = {
                "pdf_content": pdf_content,
                "input_name": input_name,
                "effort": effort,
            }
            mineru_root = tmp_path / "remote-mineru" / "demo"
            hybrid_dir = mineru_root / "hybrid_auto"
            hybrid_dir.mkdir(parents=True)
            (hybrid_dir / "demo_model.json").write_text("[]\n")
            (mineru_root / "mineru.log").write_text("mineru log\n")
            return ocr_app.package_outputs(mineru_root)

    class FakePopo:
        def remote(
            self,
            mineru_results_archive: bytes,
            pdf_content: bytes,
            input_name: str,
            max_new_tokens: int,
        ) -> bytes:
            calls["popo"] = {
                "mineru_results_archive": mineru_results_archive,
                "pdf_content": pdf_content,
                "input_name": input_name,
                "max_new_tokens": max_new_tokens,
            }
            popo_root = tmp_path / "remote-popo" / "popo-results"
            inference_dir = popo_root / "inference" / "mineru"
            inference_dir.mkdir(parents=True)
            (inference_dir / "demo.json").write_text("[]\n")
            logs_dir = popo_root / "logs"
            logs_dir.mkdir(parents=True)
            (logs_dir / "inference.log").write_text("popo log\n")
            return ocr_app.package_outputs(popo_root)

    monkeypatch.setattr(ocr_app, "run_mineru_ocr", FakeMinerUOCR())
    monkeypatch.setattr(ocr_app, "run_mineru_popo", FakePopo())

    raw_f = ocr_app.submit_ocr_task.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdf=str(input_pdf),
        out_dir=str(out_dir),
        run_popo=True,
        skip_model_download=True,
        popo_max_new_tokens=32,
    )

    run_dir = out_dir / "demo"
    assert calls["mineru"] == {
        "pdf_content": b"%PDF-1.7\n",
        "input_name": "demo.pdf",
        "effort": "high",
    }
    assert calls["popo"]["pdf_content"] == b"%PDF-1.7\n"
    assert calls["popo"]["input_name"] == "demo.pdf"
    assert calls["popo"]["max_new_tokens"] == 32
    assert not (out_dir / "demo_mineru.tar.zst").exists()
    assert not (out_dir / "demo_popo.tar.zst").exists()
    assert (run_dir / "hybrid_auto" / "demo_model.json").is_file()
    assert (run_dir / "popo-results" / "inference" / "mineru" / "demo.json").is_file()
    assert not (out_dir / "popo-results").exists()
    assert not (run_dir / "mineru.log").exists()
    assert not (run_dir / "popo-results" / "logs").exists()
    assert (run_dir / "logs" / "mineru.log").read_text() == "mineru log\n"
    assert (run_dir / "logs" / "inference.log").read_text() == "popo log\n"


def test_ocr_local_entrypoint_skips_mineru_when_hybrid_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "demo.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n")
    out_dir = tmp_path / "out"
    hybrid_dir = out_dir / "demo" / "hybrid_auto"
    hybrid_dir.mkdir(parents=True)
    (hybrid_dir / "demo_model.json").write_text("[]\n")
    calls = {}

    class FakeMinerUOCR:
        def remote(self, *args, **kwargs) -> bytes:
            raise AssertionError("MinerU should be skipped")

    class FakePopo:
        def remote(
            self,
            mineru_results_archive: bytes,
            pdf_content: bytes,
            input_name: str,
            max_new_tokens: int,
        ) -> bytes:
            calls["popo"] = {
                "mineru_results_archive": mineru_results_archive,
                "pdf_content": pdf_content,
                "input_name": input_name,
                "max_new_tokens": max_new_tokens,
            }
            popo_root = tmp_path / "remote-popo" / "popo-results"
            inference_dir = popo_root / "inference" / "mineru"
            inference_dir.mkdir(parents=True)
            (inference_dir / "demo.json").write_text("[]\n")
            return ocr_app.package_outputs(popo_root)

    monkeypatch.setattr(ocr_app, "run_mineru_ocr", FakeMinerUOCR())
    monkeypatch.setattr(ocr_app, "run_mineru_popo", FakePopo())

    raw_f = ocr_app.submit_ocr_task.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdf=str(input_pdf),
        out_dir=str(out_dir),
        run_popo=True,
        skip_model_download=True,
        popo_max_new_tokens=32,
    )

    names = _tar_zst_names(calls["popo"]["mineru_results_archive"], tmp_path)
    assert "demo/hybrid_auto/demo_model.json" in names
    assert calls["popo"]["pdf_content"] == b"%PDF-1.7\n"
    assert calls["popo"]["input_name"] == "demo.pdf"
    assert calls["popo"]["max_new_tokens"] == 32
    assert (
        out_dir / "demo" / "popo-results" / "inference" / "mineru" / "demo.json"
    ).is_file()


def test_ocr_local_entrypoint_skips_popo_when_results_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "demo.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n")
    run_dir = tmp_path / "out" / "demo"
    hybrid_dir = run_dir / "hybrid_auto"
    popo_results = run_dir / "popo-results" / "inference" / "mineru"
    hybrid_dir.mkdir(parents=True)
    popo_results.mkdir(parents=True)
    (hybrid_dir / "demo_model.json").write_text("[]\n")
    (popo_results / "demo.json").write_text("[]\n")

    class FakeMinerUOCR:
        def remote(self, *args, **kwargs) -> bytes:
            raise AssertionError("MinerU should be skipped")

    class FakePopo:
        def remote(self, *args, **kwargs) -> bytes:
            raise AssertionError("MinerU-Popo should be skipped")

    monkeypatch.setattr(ocr_app, "run_mineru_ocr", FakeMinerUOCR())
    monkeypatch.setattr(ocr_app, "run_mineru_popo", FakePopo())

    raw_f = ocr_app.submit_ocr_task.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdf=str(input_pdf),
        out_dir=str(tmp_path / "out"),
        run_popo=True,
        skip_model_download=True,
    )

    assert (hybrid_dir / "demo_model.json").is_file()
    assert (popo_results / "demo.json").is_file()


def test_ocr_local_entrypoint_force_reruns_existing_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdf = tmp_path / "demo.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n")
    out_dir = tmp_path / "out"
    run_dir = out_dir / "demo"
    (run_dir / "hybrid_auto").mkdir(parents=True)
    (run_dir / "popo-results").mkdir()
    calls = {}

    class FakeMinerUOCR:
        def remote(self, pdf_content: bytes, input_name: str, effort: str) -> bytes:
            calls["mineru"] = {
                "pdf_content": pdf_content,
                "input_name": input_name,
                "effort": effort,
            }
            mineru_root = tmp_path / "remote-mineru" / "demo"
            hybrid_dir = mineru_root / "hybrid_auto"
            hybrid_dir.mkdir(parents=True)
            (hybrid_dir / "demo_model.json").write_text("[]\n")
            return ocr_app.package_outputs(mineru_root)

    class FakePopo:
        def remote(
            self,
            mineru_results_archive: bytes,
            pdf_content: bytes,
            input_name: str,
            max_new_tokens: int,
        ) -> bytes:
            calls["popo"] = {
                "mineru_results_archive": mineru_results_archive,
                "pdf_content": pdf_content,
                "input_name": input_name,
                "max_new_tokens": max_new_tokens,
            }
            popo_root = tmp_path / "remote-popo" / "popo-results"
            inference_dir = popo_root / "inference" / "mineru"
            inference_dir.mkdir(parents=True)
            (inference_dir / "demo.json").write_text("[]\n")
            return ocr_app.package_outputs(popo_root)

    monkeypatch.setattr(ocr_app, "run_mineru_ocr", FakeMinerUOCR())
    monkeypatch.setattr(ocr_app, "run_mineru_popo", FakePopo())

    raw_f = ocr_app.submit_ocr_task.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdf=str(input_pdf),
        out_dir=str(out_dir),
        run_popo=True,
        skip_model_download=True,
        force=True,
        popo_max_new_tokens=32,
    )

    assert calls["mineru"] == {
        "pdf_content": b"%PDF-1.7\n",
        "input_name": "demo.pdf",
        "effort": "high",
    }
    assert calls["popo"]["pdf_content"] == b"%PDF-1.7\n"
    assert calls["popo"]["input_name"] == "demo.pdf"
    assert calls["popo"]["max_new_tokens"] == 32
    assert (run_dir / "hybrid_auto" / "demo_model.json").is_file()
    assert (run_dir / "popo-results" / "inference" / "mineru" / "demo.json").is_file()
