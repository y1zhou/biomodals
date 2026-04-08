"""Parse document and extract text with PaddleOCR: <https://www.paddleocr.ai/latest/index.html/>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | **Required** | Path to the input PDF or image file. |

## Outputs

* TODO
"""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "86400"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "PaddleOCR")

MODEL_WEIGHTS_PATH = "/root/.paddlex"
MODEL_WEIGHTS_VOLUME = Volume.from_name("paddleocr-models", create_if_missing=True)

OUT_VOLUME_NAME = "paddleocr-output"
OUT_VOLUME = Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True, version=2)
OUT_VOLUME_PATH = f"/{OUT_VOLUME_NAME}"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu126",  # find best torch and CUDA versions
        }
    )
    .uv_pip_install(
        "paddlepaddle-gpu==3.2.1",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .uv_pip_install("paddleocr[all]")
    .uv_pip_install(
        "https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl"
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=8,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={OUT_VOLUME_PATH: OUT_VOLUME, MODEL_WEIGHTS_PATH: MODEL_WEIGHTS_VOLUME},
)
def run_paddleocr(input_content: bytes, run_name: str) -> str:
    """Run PaddleOCR on the input PDF content and return extracted markdown and images."""
    from paddleocr import PaddleOCRVL

    run_dir = ".".join(run_name.split(".")[:-1])
    workdir = Path(OUT_VOLUME_PATH) / run_dir
    if workdir.exists():
        return str(workdir)

    workdir.mkdir(parents=True, exist_ok=True)
    input_file = workdir / run_name
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

    return str(workdir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_paddleocr_task(input: str) -> None:
    """Run PaddleOCR on Modal and save results to a local directory."""
    # Load input PDB
    input_file_path = Path(input).expanduser().resolve()
    input_content = input_file_path.read_bytes()
    run_name = input_file_path.name

    # Run PaddleOCR
    _ = run_paddleocr.remote(input_content, run_name)
    print("See PaddleOCR results with:\n")
    print(f"  modal volume ls {OUT_VOLUME_NAME} {run_name}")
