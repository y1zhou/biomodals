"""PPIFlow on Modal - runnable scaffold.

Usage
=====
# 1) (Optional) download model weights into persistent Volume (run once)
modal run ppiflow_app.py --download-models --force-redownload

# 2) Run inference (binder design)
modal run ppiflow_app.py \
  --run-name test1 \
  --input-pdb ./target.pdb \
  --target-chain B \
  --binder-chain A \
  --config configs/inference_binder.yaml \
  --specified-hotspots "B119,B141,B200" \
  --samples-min-length 75 \
  --samples-max-length 76 \
  --samples-per-target 5 \
  --model-weights models/binder.ckpt
"""

from __future__ import annotations

import argparse
import os
import shlex
import tarfile
import tempfile
from pathlib import Path

from modal import App, Image, Volume

# -------------------------
# Modal configs
# -------------------------
APP_NAME = os.environ.get("MODAL_APP", "ppiflow")
GPU = os.environ.get("GPU", "A10G")  # e.g. A10G, A100
TIMEOUT = int(os.environ.get("TIMEOUT", "7200"))

# Persistent Volumes
MODELS_VOL = Volume.from_name("ppiflow-models", create_if_missing=True)
RUNS_VOL = Volume.from_name("ppiflow-runs", create_if_missing=True)

MODELS_DIR = Path("/models")
RUNS_DIR = Path("/runs")

# -------------------------
# Image definition
# -------------------------
# TODO: pin versions according to your repo requirements (cuda/torch/etc.)
runtime_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", 
        "curl", 
        "ca-certificates",
        "build-essential",
        "python3-dev",
    )
    .uv_pip_install(
        # TODO: replace with your real deps
        "torch",
        "pyyaml",
        "numpy",
        "pandas",
        "omegaconf", 
        "hydra-core",
        "lightning",      # ← 必加（这是 PyTorch Lightning 新名字）
        "torchmetrics",
        "scipy",
        "tqdm",
        "einops",
        "biopython",
        "scikit-learn",
        "dm-tree",   # ← 关键：提供 import tree   
        "gputil",
        "tmtools",
        "optree",
    )
    .run_commands(
    "rm -rf /ppiflow && git clone --depth 1 https://github.com/Mingchenchen/PPIFlow.git /ppiflow",
    )    
    .run_commands(
        "pip install -U pip",

        # 1) 明确安装 CUDA 版 PyTorch（示例：cu121）
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",

        # 2) 你之前已经补过的通用依赖
        "pip install pyyaml numpy pandas omegaconf hydra-core lightning torchmetrics dm-tree scipy tqdm einops biopython scikit-learn",

        # 3) 安装匹配 torch+cuda 的 torch_scatter wheel（关键）
        #    这里的 torch-2.4.0+cu121 要与你实际 torch 版本一致
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",

        "pip install mdtraj",
        "pip install freesasa",

    )

    # TODO: if you need your ppiflow repo code, choose ONE approach:
    # (A) bake into image at build time by git clone (good for stable repo)
    # .run_commands("git clone --depth 1 https://github.com/<org>/PPIFlow.git /ppiflow")
)

app = App(APP_NAME)


# -------------------------
# Helpers
# -------------------------
def _run(cmd: str, cwd: str | None = None) -> None:
    """Run a shell command with strict error handling."""
    import subprocess

    print(f"[cmd] {cmd}")
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合并输出，保证 Modal 日志里能看到
    )
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")

    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def _tar_dir(src_dir: Path, out_tar_gz: Path) -> None:
    """Create tar.gz from a directory."""
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


# -------------------------
# Step 1: download weights into Volume (run once)
# -------------------------
@app.function(
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={str(MODELS_DIR): MODELS_VOL},
)
def download_models_task(force_redownload: bool = False) -> str:
    """Download model checkpoints into MODELS_DIR (persistent).
    Return a short summary string.
    """
    # TODO: replace with your real model URLs / filenames
    # Example:
    #   binder.ckpt, backbone.ckpt, etc.
    CHECKPOINT_URLS = {
        "binder.ckpt": "https://example.com/path/to/binder.ckpt",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = []
    skipped = []

    for fname, url in CHECKPOINT_URLS.items():
        dst = MODELS_DIR / fname
        if dst.exists() and not force_redownload:
            skipped.append(fname)
            continue

        # Use curl for simplicity; switch to aria2c/aiohttp if you need concurrency
        _run(f'curl -L "{url}" -o "{dst}"')
        downloaded.append(fname)

    # Persist changes to Volume
    MODELS_VOL.commit()

    return f"downloaded={downloaded}, skipped={skipped}, models_dir={MODELS_DIR}"


# -------------------------
# Step 2: inference (remote GPU job)
# -------------------------
@app.function(
    gpu=GPU,
    cpu=(2, 8),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={
        str(MODELS_DIR): MODELS_VOL,
        str(RUNS_DIR): RUNS_VOL,
    },
)
def run_ppiflow(
    run_name: str,
    input_pdb_bytes: bytes,
    target_chain: str,
    binder_chain: str,
    config_path: str,
    specified_hotspots: str,
    samples_min_length: int,
    samples_max_length: int,
    samples_per_target: int,
    model_weights_path: str,
) -> bytes:
    """Run PPIFlow inference, write outputs to RUNS_DIR/<run_name>,
    tar.gz it, and return tarball bytes to local.
    """
    # Working dirs
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write input pdb into run_dir
    input_pdb = run_dir / "target.pdb"
    input_pdb.write_bytes(input_pdb_bytes)

    # Resolve model weights inside container
    # If user passes "models/binder.ckpt", we map it to /models/binder.ckpt
    mw = Path(model_weights_path)
    model_ckpt = MODELS_DIR / mw.name if not str(mw).startswith(str(MODELS_DIR)) else mw
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    # Resolve config path:
    # Option 1: your repo is baked into image at /ppiflow, so configs live there
    # Option 2: you pass a config via CLI and we write it too (not implemented here)
    #
    # TODO: Adjust according to your actual repo layout.
    config = Path(config_path)
    if not config.exists():
        # if repo is at /ppiflow and you pass a relative path, try resolve there
        repo_guess = Path("/ppiflow") / config_path
        if repo_guess.exists():
            config = repo_guess

    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config_path} (tried {config})")

    # -------------------------
    # TODO: Replace THIS with your real PPIFlow inference command
    # Example based on your earlier CLI:
    #
    # python sample_binder.py \
    #   --input_pdb <...> \
    #   --target_chain B \
    #   --binder_chain A \
    #   --config <...> \
    #   --specified_hotspots "B119,B141,B200" \
    #   --samples_min_length 75 \
    #   --samples_max_length 76 \
    #   --samples_per_target 5 \
    #   --model_weights <...> \
    #   --output_dir <...> \
    #   --name <...>
    #
    # If your repo isn't in the image, you must clone/install it in runtime_image.
    # -------------------------
    cmd = " ".join(
        [
            "python",
            # TODO: set correct path to entry script
            "/ppiflow/sample_binder.py",
            "--input_pdb",
            shlex.quote(str(input_pdb)),
            "--target_chain",
            shlex.quote(target_chain),
            "--binder_chain",
            shlex.quote(binder_chain),
            "--config",
            shlex.quote(str(config)),
            "--specified_hotspots",
            shlex.quote(specified_hotspots),
            "--samples_min_length",
            str(samples_min_length),
            "--samples_max_length",
            str(samples_max_length),
            "--samples_per_target",
            str(samples_per_target),
            "--model_weights",
            shlex.quote(str(model_ckpt)),
            "--output_dir",
            shlex.quote(str(run_dir / "outputs")),
            "--name",
            shlex.quote(run_name),
        ]
    )

    _run(cmd)

    # Persist run outputs to Volume
    RUNS_VOL.commit()

    # Pack outputs and return
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"{run_name}.tar.gz"
        _tar_dir(run_dir, tar_path)
        return tar_path.read_bytes()


# -------------------------
# Local entrypoint
# -------------------------
@app.local_entrypoint()
def main(
    #  """CLI entrypoint for running PPIFlow on Modal."""
    run_name: str = "test1",
    input_pdb: str | None = None,
    target_chain: str = "B",
    binder_chain: str = "A",
    config: str = "configs/inference_binder.yaml",
    specified_hotspots: str = "",
    samples_min_length: int = 75,
    samples_max_length: int = 76,
    samples_per_target: int = 5,
    model_weights: str = "models/binder.ckpt",
    download_models: bool = False,
    force_redownload: bool = False,
    out_dir: str = "./ppiflow_outputs",
) -> None:
    if download_models:
        msg = download_models_task.remote(force_redownload=force_redownload)
        print(msg)
        return

    if input_pdb is None:
        raise ValueError("--input-pdb is required for inference")

    pdb_bytes = Path(input_pdb).read_bytes()

    tar_bytes = run_ppiflow.remote(
        run_name=run_name,
        input_pdb_bytes=pdb_bytes,
        target_chain=target_chain,
        binder_chain=binder_chain,
        config_path=config,
        specified_hotspots=specified_hotspots,
        samples_min_length=samples_min_length,
        samples_max_length=samples_max_length,
        samples_per_target=samples_per_target,
        model_weights_path=model_weights,
    )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir_p / f"{run_name}.tar.gz"
    out_tar.write_bytes(tar_bytes)
    print(f"[ok] saved: {out_tar}")
