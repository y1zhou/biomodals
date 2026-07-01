"""OCR app using MinerU and MinerU-Popo.

MinerU source repo: <https://github.com/opendatalab/MinerU>.
MinerU-Popo source repo: <https://github.com/opendatalab/MinerU-Popo>.

## Outputs

The local entrypoint writes all outputs under `<pdf-stem>/`. MinerU outputs live
at the top level, logs live under `logs/`, and optional MinerU-Popo outputs live
under `popo-results/`.

Example layout for input `demo.pdf` with `--run-popo`:

```text
demo/
  hybrid_auto/
  logs/
    mineru.log
    label_normalization.log
    inference.log
    build_tree.log
  popo-results/
    label_normalization/
      mineru/
        demo.json
    inference/
      mineru/
        demo.json
    build_tree/
      mineru/
        demo.json
    build_tree_txt/
      mineru/
        demo.txt
```
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import modal

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MAX_TIMEOUT, MODEL_VOLUME
from biomodals.helper.io import resolve_local_output_dir
from biomodals.helper.shell import package_outputs, run_command, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="MinerU",
    repo_url="https://github.com/opendatalab/MinerU",
    repo_commit_hash="3e60291846cb7c3bf8fe7f4f16238f4fc6cce491",
    package_name="mineru",
    version="3.4",
    python_version="3.12",
    cuda_version="cu130",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

POPO_CONF = AppConfig(
    tags=CONF.tags,
    name="MinerU-Popo",
    repo_url="https://github.com/opendatalab/MinerU-Popo",
    repo_commit_hash="75c36a8c0f38adee03c78850366645a58cd5d4af",
    python_version="3.10",
    cuda_version="cu129",
    gpu=os.environ.get("POPO_GPU", os.environ.get("GPU", "L40S")),
    timeout=int(
        os.environ.get("POPO_TIMEOUT", os.environ.get("TIMEOUT", str(MAX_TIMEOUT)))
    ),
)

POPO_REPO_DIR = POPO_CONF.git_clone_dir
MINERU_CONFIG_PATH = Path(CONF.model_volume_mountpoint) / CONF.name / "mineru.json"
VLLM_CACHE_ROOT = Path(CONF.model_volume_mountpoint) / CONF.name / "vllm-cache"
POPO_HF_CACHE_DIR = Path(POPO_CONF.default_env["HF_HOME"]) / "hub"

##########################################
# Image and app definitions
##########################################
common_apt_packages = (
    "build-essential",
    "fonts-dejavu-core",
    "git",
    "libgl1",
    "libglib2.0-0",
)

mineru_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install(*common_apt_packages)
    .env(
        CONF.default_env
        | {
            "MINERU_TOOLS_CONFIG_JSON": str(MINERU_CONFIG_PATH),
            "VLLM_CACHE_ROOT": str(VLLM_CACHE_ROOT),
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .uv_pip_install(f"mineru[core,vllm]=={CONF.version}")
    .pipe(patch_image_for_helper)
)

popo_image = (
    modal.Image
    .debian_slim(python_version=POPO_CONF.python_version)
    .apt_install(*common_apt_packages)
    .env(
        POPO_CONF.default_env
        | {
            "POPO_INFERENCE_BACKEND": "transformers",
        }
    )
    .uv_pip_install("huggingface_hub")
    .run_commands(
        f"git clone {POPO_CONF.repo_url} {POPO_REPO_DIR}",
        f"cd {POPO_REPO_DIR} && git checkout {POPO_CONF.repo_commit_hash}",
        (
            f"grep -Ev '^(cuda-bindings|nvidia-)' {POPO_REPO_DIR}/requirements.txt "
            "| sed -E 's/^(click|gradio_pdf)==[^[:space:]]+$/\\1/' "
            f"> {POPO_REPO_DIR / 'requirements-biomodals.txt'}"
        ),
        (
            "/.uv/uv pip install --python $(command -v python) "
            f"--compile-bytecode -r {POPO_REPO_DIR / 'requirements-biomodals.txt'}"
        ),
    )
    .pipe(patch_image_for_helper, skip_deps=("uniaf3",))
)

app = modal.App(CONF.name, image=mineru_image, tags=CONF.tags)


##########################################
# Archive helpers
##########################################
def _extract_tar_zst_bytes(archive_bytes: bytes, destination: Path) -> None:
    """Extract `.tar.zst` bytes to a local destination."""
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{CONF.name}_extract_") as tmpdir:
        archive_path = Path(tmpdir) / "archive.tar.zst"
        archive_path.write_bytes(archive_bytes)
        run_command(
            ["tar", "-I", "zstd", "-xf", str(archive_path), "-C", str(destination)],
            output_mode="capture",
        )


def _collect_local_run_logs(run_dir: Path) -> None:
    """Move MinerU and Popo logs into the run-level logs directory."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    mineru_log = run_dir / "mineru.log"
    if mineru_log.is_file():
        mineru_log.replace(logs_dir / "mineru.log")

    popo_logs_dir = run_dir / "popo-results" / "logs"
    if not popo_logs_dir.is_dir():
        return

    for path in popo_logs_dir.iterdir():
        if path.is_file():
            path.replace(logs_dir / path.name)
    popo_logs_dir.rmdir()


def _snapshot_download_popo_model(
    *,
    force: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Download or resolve the MinerU-Popo model in Hugging Face's cache layout."""
    from huggingface_hub import snapshot_download  # type: ignore[ty:unresolved-import]

    snapshot_path = snapshot_download(
        repo_id="DreamEternal/MinerU-Popo",
        cache_dir=POPO_HF_CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
        force_download=force,
        local_files_only=local_files_only,
    )
    return Path(snapshot_path)


##########################################
# Fetch model weights
##########################################
@app.function(
    image=mineru_image,
    volumes=CONF.mounts(model_volume=True, model_ro=False, is_huggingface=True),
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=CONF.timeout,
)
def download_mineru_model_weights() -> None:
    """Download MinerU models into the Biomodals model store."""
    MINERU_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("💊 Downloading MinerU model weights...")
    run_command(
        ["mineru-models-download", "--source", "huggingface", "--model_type", "all"],
        env={"MINERU_TOOLS_CONFIG_JSON": str(MINERU_CONFIG_PATH)},
    )
    MODEL_VOLUME.commit()
    print("💊 MinerU model download complete")


@app.function(
    image=popo_image,
    volumes=POPO_CONF.mounts(model_volume=True, model_ro=False, is_huggingface=True),
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=POPO_CONF.timeout,
)
def download_popo_model_weights(force: bool = False) -> None:
    """Download MinerU-Popo checkpoints into the Biomodals model store."""
    print(
        "💊 Downloading MinerU-Popo model weights to "
        f"{POPO_HF_CACHE_DIR / 'models--DreamEternal--MinerU-Popo'}..."
    )
    snapshot_path = _snapshot_download_popo_model(force=force)
    MODEL_VOLUME.commit()
    print(f"💊 MinerU-Popo model download complete: {snapshot_path}")


##########################################
# Inference functions
##########################################
@app.function(
    image=mineru_image,
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 131072),
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_ro=False, is_huggingface=True),
)
def run_mineru_ocr(
    pdf_content: bytes,
    input_name: str,
    effort: Literal["medium", "high"] = "high",
) -> bytes:
    """Run MinerU on one PDF and return the generated directory as `.tar.zst`."""
    if effort not in {"medium", "high"}:
        raise ValueError("effort must be one of: medium, high")

    safe_input_name = sanitize_filename(input_name)
    if Path(safe_input_name).suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file, got: {input_name}")
    pdf_stem = Path(safe_input_name).stem
    with TemporaryDirectory(prefix=f"{CONF.name}_mineru_") as tmpdir:
        workdir = Path(tmpdir)
        input_pdf = workdir / safe_input_name
        out_dir = workdir / "mineru-output"
        log_path = workdir / "mineru.log"
        input_pdf.write_bytes(pdf_content)
        out_dir.mkdir(parents=True, exist_ok=True)
        VLLM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mineru",
            "-b",
            "hybrid-engine",
            "--effort",
            effort,
            "-p",
            str(input_pdf),
            "-o",
            str(out_dir),
        ]
        result_dir = out_dir / pdf_stem
        try:
            run_command(
                cmd,
                output_mode="tee",
                log_file=log_path,
                env={
                    "VLLM_CACHE_ROOT": str(VLLM_CACHE_ROOT),
                    "VLLM_USE_FLASHINFER_SAMPLER": "0",
                },
            )
        finally:
            if log_path.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(log_path), result_dir / "mineru.log")

        hybrid_dir = result_dir / "hybrid_auto"
        if not hybrid_dir.is_dir():
            raise FileNotFoundError(
                f"MinerU did not create expected output: {hybrid_dir}"
            )
        archive = package_outputs(result_dir)
        try:
            MODEL_VOLUME.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"💊 Warning: failed to persist vLLM cache: {exc}")
        return archive


@app.function(
    image=popo_image,
    gpu=POPO_CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(4096, 131072),
    timeout=POPO_CONF.timeout,
    volumes=POPO_CONF.mounts(model_volume=True, is_huggingface=True),
)
def run_mineru_popo(
    mineru_results_archive: bytes,
    pdf_content: bytes,
    input_name: str,
    max_new_tokens: int = 8192,
) -> bytes:
    """Run MinerU-Popo on MinerU outputs and return `popo-results/` as `.tar.zst`."""
    safe_input_name = sanitize_filename(input_name)
    if Path(safe_input_name).suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file, got: {input_name}")
    doc_id = Path(safe_input_name).stem
    with TemporaryDirectory(prefix=f"{POPO_CONF.name}_") as tmpdir:
        workdir = Path(tmpdir)
        mineru_extract_root = workdir / "mineru-results"
        _extract_tar_zst_bytes(mineru_results_archive, mineru_extract_root)
        hybrid_dir = mineru_extract_root / doc_id / "hybrid_auto"
        if not hybrid_dir.is_dir():
            matches = [
                path
                for path in mineru_extract_root.rglob("hybrid_auto")
                if path.is_dir() and path.parent.name == doc_id
            ]
            if len(matches) == 1:
                hybrid_dir = matches[0]
            elif not matches:
                raise FileNotFoundError(
                    f"MinerU archive does not contain {doc_id}/hybrid_auto"
                )
            else:
                raise RuntimeError(
                    f"MinerU archive contains multiple {doc_id}/hybrid_auto dirs"
                )

        post_process_doc_dir = workdir / "post-process" / "mineru" / doc_id
        post_process_doc_dir.mkdir(parents=True, exist_ok=True)
        (post_process_doc_dir / "vlm").symlink_to(
            hybrid_dir,
            target_is_directory=True,
        )

        pdf_dir = workdir / "eval_pdf_dir"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        (pdf_dir / safe_input_name).write_bytes(pdf_content)

        popo_results = workdir / "popo-results"
        logs_dir = popo_results / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        label_dir = popo_results / "label_normalization"
        inference_dir = popo_results / "inference" / "mineru"
        tree_dir = popo_results / "build_tree" / "mineru"
        tree_txt_dir = popo_results / "build_tree_txt" / "mineru"
        model_snapshot_dir = _snapshot_download_popo_model(local_files_only=True)

        run_command(
            [
                sys.executable,
                str(POPO_REPO_DIR / "post_processing" / "label_normalization.py"),
                "--model",
                "mineru",
                "--input-dir",
                str(workdir / "post-process" / "mineru"),
                "--output-dir",
                str(label_dir),
                "--pdf-dir",
                str(pdf_dir),
                "--doc-id",
                doc_id,
                "--doc-limit",
                "0",
            ],
            output_mode="tee",
            log_file=logs_dir / "label_normalization.log",
            cwd=POPO_REPO_DIR,
        )
        normalized_doc = label_dir / "mineru" / f"{doc_id}.json"
        if not normalized_doc.is_file():
            raise FileNotFoundError(f"MinerU-Popo did not create {normalized_doc}")

        run_command(
            [
                sys.executable,
                str(POPO_REPO_DIR / "post_processing" / "run_inference.py"),
                "--input-dir",
                str(label_dir / "mineru"),
                "--model-path",
                str(model_snapshot_dir),
                "--output-dir",
                str(inference_dir),
                "--raw-output-root",
                "",
                "--limit",
                "0",
            ],
            output_mode="tee",
            log_file=logs_dir / "inference.log",
            cwd=POPO_REPO_DIR,
            env={
                "POPO_INFERENCE_BACKEND": "transformers",
                "POPO_MAX_NEW_TOKENS": str(max_new_tokens),
            },
        )
        inference_doc = inference_dir / f"{doc_id}.json"
        if not inference_doc.is_file():
            raise FileNotFoundError(f"MinerU-Popo did not create {inference_doc}")

        run_command(
            [
                sys.executable,
                str(POPO_REPO_DIR / "post_processing" / "get_json_tree.py"),
                "--input-dir",
                str(inference_dir),
                "--output-dir",
                str(tree_dir),
                "--txt-dir",
                str(tree_txt_dir),
            ],
            output_mode="tee",
            log_file=logs_dir / "build_tree.log",
            cwd=POPO_REPO_DIR,
        )
        tree_doc = tree_dir / f"{doc_id}.json"
        if not tree_doc.is_file():
            raise FileNotFoundError(f"MinerU-Popo did not create {tree_doc}")

        return package_outputs(popo_results)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_ocr_task(
    input_pdf: str | None = None,
    out_dir: str | None = None,
    effort: Literal["medium", "high"] = "high",
    run_popo: bool = False,
    download_mineru_models: bool = False,
    download_popo_models: bool = False,
    skip_model_download: bool = False,
    force_popo_redownload: bool = False,
    popo_max_new_tokens: int = 8192,
) -> None:
    """Run MinerU OCR and optionally MinerU-Popo post-processing.

    Args:
        input_pdf: Path to the input PDF. May be omitted when only downloading models.
        out_dir: Local output directory. Defaults to the current working directory.
        effort: MinerU hybrid-engine effort level, either `medium` or `high`.
        run_popo: Whether to run MinerU-Popo after MinerU.
        download_mineru_models: Download MinerU model weights and continue if
            `input_pdf` is also provided.
        download_popo_models: Download MinerU-Popo model weights and continue if
            `input_pdf` is also provided.
        skip_model_download: Skip automatic model download checks before inference.
        force_popo_redownload: Force Hugging Face re-download for MinerU-Popo weights.
        popo_max_new_tokens: Maximum new tokens for MinerU-Popo transformer generation.
    """
    if effort not in {"medium", "high"}:
        raise ValueError("effort must be one of: medium, high")
    if input_pdf is None and not (download_mineru_models or download_popo_models):
        raise ValueError("input_pdf is required unless a model download flag is set")

    if download_mineru_models or (input_pdf is not None and not skip_model_download):
        download_mineru_model_weights.remote()
    if download_popo_models or (
        input_pdf is not None and run_popo and not skip_model_download
    ):
        download_popo_model_weights.remote(force=force_popo_redownload)
    if input_pdf is None:
        return

    input_path = Path(input_pdf).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input must be a PDF file, got: {input_path}")
    safe_input_name = sanitize_filename(input_path.name)
    safe_input_stem = sanitize_filename(input_path.stem)
    output_dir = resolve_local_output_dir(out_dir)
    run_dir = output_dir / safe_input_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_content = input_path.read_bytes()

    print(f"🧬 Running MinerU on {input_path}...")
    mineru_archive = run_mineru_ocr.remote(
        pdf_content,
        safe_input_name,
        effort,
    )
    _extract_tar_zst_bytes(mineru_archive, output_dir)
    _collect_local_run_logs(run_dir)
    print(f"🧬 Extracted MinerU results under {run_dir}")

    if run_popo:
        print("🧬 Running MinerU-Popo post-processing...")
        popo_archive = run_mineru_popo.remote(
            mineru_archive,
            pdf_content,
            safe_input_name,
            popo_max_new_tokens,
        )
        _extract_tar_zst_bytes(popo_archive, run_dir)
        _collect_local_run_logs(run_dir)
        print(f"🧬 Extracted MinerU-Popo results under {run_dir / 'popo-results'}")

    for archive_name in (
        f"{safe_input_stem}_mineru.tar.zst",
        f"{safe_input_stem}_popo.tar.zst",
    ):
        (output_dir / archive_name).unlink(missing_ok=True)
