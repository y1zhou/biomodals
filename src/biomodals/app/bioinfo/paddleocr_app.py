"""Parse document and extract text with PaddleOCR: <https://www.paddleocr.ai/latest/index.html/>."""
# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import package_outputs, softlink_dir

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="PaddleOCR",
    repo_url="https://github.com/PaddlePaddle/PaddleOCR",
    package_name="paddleocr",
    version="3.4.1",
    python_version="3.12",
    cuda_version="cu126",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "86400")),
)


@dataclass
class AppInfo:
    """Container for PaddleOCR-specific configuration and constants."""

    model_weights_path: str = "/root/.paddlex"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    patch_image_for_helper(
        modal.Image.debian_slim(python_version=CONF.python_version)
        .apt_install("git", "build-essential", "libgl1-mesa-glx", "libglib2.0-0")
        .env(CONF.default_env),
        copy_patch_files=True,
    )
    .uv_pip_install(
        "paddlepaddle-gpu==3.2.1",
        index_url=f"https://www.paddlepaddle.org.cn/packages/stable/{CONF.cuda_version}/",
    )
    .uv_pip_install(f"{CONF.package_name}[all]=={CONF.version}")
    .uv_pip_install(
        f"https://paddle-whl.bj.bcebos.com/nightly/{CONF.cuda_version}/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl"
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME},
)
def run_paddleocr(input_content: bytes, input_name: str) -> bytes:
    """Run PaddleOCR on the input PDF content and return extracted markdown and images."""
    from tempfile import mkdtemp

    from paddleocr import PaddleOCRVL

    # PaddleOCR hardcodes the model cache directory
    model_cache_dir = Path(CONF.model_volume_mountpoint) / CONF.name
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    softlink_dir(model_cache_dir, Path(APP_INFO.model_weights_path))

    run_dir = ".".join(input_name.split(".")[:-1])
    workdir = Path(mkdtemp(prefix=f"{CONF.name}_")) / run_dir
    workdir.mkdir()
    input_file = workdir / input_name
    input_file.write_bytes(input_content)

    pipeline = PaddleOCRVL()

    markdown_list = []
    markdown_images = []
    for res in pipeline.predict_iter(input=str(input_file)):
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    mkd_file_path = workdir / f"{run_dir}.md"
    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    for item in markdown_images:
        if not item:
            continue
        for path, image in item.items():
            file_path = workdir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)

    return package_outputs(workdir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_paddleocr_task(input: str, output_dir: str | None = None) -> None:
    """Run PaddleOCR on Modal and save results to a local directory.

    Args:
        input: Path to the input PDF or image file.
        output_dir: Path to the directory where results will be saved. OCR
            results will be saved with the same filename as the input, but
            with a `.tar.zst` extension.
    """
    # Load input file
    input_file_path = Path(input).expanduser().resolve()
    input_content = input_file_path.read_bytes()
    input_filename = input_file_path.name

    # Run PaddleOCR
    tarball_bytes = run_paddleocr.remote(input_content, input_filename)
    if output_dir is None:
        out_dir_path = Path.cwd()
    else:
        out_dir_path = Path(output_dir).expanduser().resolve()
    out_path = out_dir_path / f"{input_file_path.stem}.tar.zst"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tarball_bytes)
    print(f"Saved OCR results to {out_path}")
