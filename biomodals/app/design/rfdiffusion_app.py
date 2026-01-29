"""
RFdiffusion on Modal - minimal runnable scaffold (fixed).

Typical usage:

  # 1) download models into the Modal Volume (run once)
  modal run rfdiffusion_app.py --download-models --force-redownload

  # 2) run inference (binder design / scaffold etc.)
  modal run rfdiffusion_app.py \
    --run-name test1 \
    --input-pdb ~/outputs/RFdiffusion/RBD_wt.pdb \
    --rfd-args "contigmap.contigs='[100-150/0 E333-526]' inference.num_designs=2 ppi.hotspot_res=[E405,E408]"
"""

import os
import shlex
from pathlib import Path

from modal import App, Image, Volume


# -------------------------
# Modal configs
# -------------------------
GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", "36000"))
APP_NAME = os.environ.get("MODAL_APP", "RFdiffusion")

RFD_VOLUME = Volume.from_name("rfdiffusion-models", create_if_missing=True)
RFD_OUT_VOLUME = Volume.from_name("rfdiffusion-outputs", create_if_missing=True)

RFD_REPO_DIR = "/root/RFdiffusion"
RFD_MODELS_DIR = f"{RFD_REPO_DIR}/models"
RFD_OUT_DIR = "/root/rfdiffusion_outputs"

# -------------------------
# RFdiffusion checkpoints (hardcoded from upstream download_models.sh)
# https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
# -------------------------
RFD_CHECKPOINT_URLS: dict[str, str] = {
    "Base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "Complex_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
    "Complex_Fold_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
    "InpaintSeq_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
    "InpaintSeq_Fold_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
    "ActiveSite_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
    "Base_epoch8_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt",
}


# -------------------------
# Image definition
# -------------------------
# NOTE:
# A newer, modern CUDA/PyTorch-style environment 
# (https://github.com/JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090/blob/main/RTX-5090.dockerfile) 

runtime_image = (
    Image.micromamba(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "ca-certificates",
        "zstd",
        "tar",
        "fd-find",  # prefer fd over find
    )
    .run_commands(
        # CHANGED: ensure the 'fd' binary is available (Ubuntu often exposes it as 'fdfind').
        "bash -lc 'command -v fd >/dev/null 2>&1 || (command -v fdfind >/dev/null 2>&1 && ln -sf $(command -v fdfind) /usr/local/bin/fd) || true'"
    )
    .run_commands(
       f"git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git {RFD_REPO_DIR}"
    )
    .env(
        {
            "PYTHONPATH": RFD_REPO_DIR,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        # install CUDA-enabled PyTorch from official index (avoid accidental CPU-only wheels).
        # Pin torch < 2.6 to avoid the torch.load(weights_only=...) default behavior change.
        "python -m pip install --no-cache-dir -U pip && "
        "python -m pip install --no-cache-dir "
        "--index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
    )
    .uv_pip_install(
        "numpy",
        "scipy",
        "tqdm",
        "pyyaml",
        "omegaconf",
        "hydra-core",
        "biopython",
        "pandas",
        "einops",
        "opt_einsum",
        "dm-tree",
        "pyrsistent",   # RFdiffusion symmetry 
        "aiohttp",  # async checkpoint download
        "torchdata>=0.7",  # DGL / datapipes support
    )
    .run_commands(
    # build/install NVIDIA SE3Transformer in one chained step.
       f"cd {RFD_REPO_DIR}/env/SE3Transformer && "
        "python -m pip install --no-cache-dir -r requirements.txt && "
        "python setup.py install"
    )
    .run_commands(
        # ensure DGL CUDA wheel. Uninstall any CPU build first.
        "python -m pip uninstall -y dgl || true && "
        'python -m pip install --no-cache-dir "dgl==1.1.3" -f https://data.dgl.ai/wheels/cu121/repo.html'
    )


app = App(APP_NAME, image=runtime_image)


# -------------------------
# Helpers
# -------------------------
def run_command(
    cmd: list[str],
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and stream stdout/stderr."""
    import subprocess as sp

    print("Running:", " ".join(cmd))
    with sp.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        bufsize=1,
        encoding="utf-8",
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
        rc = p.wait()
        if rc != 0:
            raise sp.CalledProcessError(rc, cmd)

def build_runtime_env() -> dict[str, str]:
    """
     Build a consistent runtime environment.
    - Keep the env dict explicit so subprocesses inherit expected settings.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = RFD_REPO_DIR
    # (Optional) quiet DGL backend warning
    env.setdefault("DGLBACKEND", "pytorch")
    return env

def package_dir_to_tar_zst(dir_path: str) -> bytes:
    """Package a directory into tar.zst and return bytes."""
    import subprocess as sp

    dp = Path(dir_path)
    parent = str(dp.parent)
    name = dp.name
    cmd = ["tar", "--zstd", "-cf", "-", name]
    return sp.check_output(cmd, cwd=parent)

def collect_outputs_for_bundle(root_dir: str) -> list[Path]:
    """
    Collect the subset of output files that are typically useful to download.
    This avoids bundling unnecessary large intermediate files.
    """
    import subprocess as sp
    root = Path(root_dir)
    # Prefer fd (multi-threaded); fallback to find.
    has_fd = sp.run(["bash", "-lc", "command -v fd >/dev/null 2>&1"]).returncode == 0

    if has_fd:
        cmd = (
            f"cd {root} && "
            r"fd -t f '\.(pdb|trb|json|yaml|yml|log|txt|csv)$' ."
        )
    else:
        cmd = (
            f"cd {root} && "
            r"find . -type f \( "
            r"-name '*.pdb' -o -name '*.trb' -o -name '*.json' -o "
            r"-name '*.yaml' -o -name '*.yml' -o -name '*.log' -o "
            r"-name '*.txt' -o -name '*.csv' "
            r"\)"
        )

    out = sp.check_output(["bash", "-lc", cmd], text=True).strip().splitlines()
    files = [root / p.lstrip("./") for p in out if p.strip()]
    return files

def package_files_to_tar_zst(files: list[Path], base_dir: str) -> bytes:
    """Create a tar.zst containing only selected files, preserving relative paths."""
    import subprocess as sp
    import tempfile

    base = Path(base_dir)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for p in files:
            rel = p.relative_to(base)
            f.write(str(rel) + "\n")
        filelist = f.name

    cmd = ["bash", "-lc", f"tar --zstd -cf - -C {base} -T {filelist}"]
    return sp.check_output(cmd)


# -------------------------
# Step 1: download model weights into the models Volume
# -------------------------
async def _download_file(session, url: str, local_path: Path) -> None:
    """Download a file asynchronously via aiohttp streaming."""
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while True:
                chunk = await response.content.read(8192)
                if not chunk:
                    break
                f.write(chunk)

@app.function(
    timeout=TIMEOUT * 2,
    volumes={RFD_MODELS_DIR: RFD_VOLUME},
)
async def download_rfdiffusion_models(force: bool = False) -> None:
    """
    Download RFdiffusion checkpoints into the persistent models Volume.

    URLs are copied verbatim from:
    https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
    """
    import asyncio
    import aiohttp

    headers = {"User-Agent": "Mozilla/5.0"}
    model_dir = Path(RFD_MODELS_DIR)

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for fname, url in RFD_CHECKPOINT_URLS.items():
            dst = model_dir / fname
            if force or not dst.exists():
                print(f"downloading {fname} -> {dst}")
                tasks.append(_download_file(session, url, dst))

        if tasks:
            await asyncio.gather(*tasks)
        else:
            print("All RFdiffusion checkpoints already present; nothing to download.")

    # Commit so checkpoints are visible immediately for remote inference jobs.
    RFD_VOLUME.commit()
    print("RFdiffusion checkpoints downloaded and committed.")


# -------------------------
# Step 2: inference function (remote GPU job)
# -------------------------
@app.function(
    gpu=GPU,
    cpu=(2, 4),
    memory=(4096, 65536),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={
      RFD_MODELS_DIR: RFD_VOLUME.read_only(),
      # output cache volume.
      RFD_OUT_DIR: RFD_OUT_VOLUME,
    },
)
def rfdiffusion_infer(
    input_pdb_bytes: bytes,
    input_pdb_name: str,
    run_name: str,
    rfd_args: str,
) -> bytes:
    """
    Run RFdiffusion inference inside the container and return a .tar.zst bundle.

    - Outputs are cached under /root/rfdiffusion_outputs/<run_name> on a persistent Volume.
    - The returned archive contains the most useful artifacts (PDB/TRB/log/JSON/YAML by default).
    """
    from tempfile import TemporaryDirectory

    env = build_runtime_env()  
  
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        input_pdb = tmp / input_pdb_name
        input_pdb.write_bytes(input_pdb_bytes)

        local_out_dir = tmp / f"{run_name}_outputs"
        local_out_dir.mkdir(parents=True, exist_ok=True)

        cached_run_dir = Path(RFD_OUT_DIR) / run_name
        cached_run_dir.mkdir(parents=True, exist_ok=True)

        run_infer_py = f"{RFD_REPO_DIR}/scripts/run_inference.py"
      
        # hydra overrides are passed as a single string, split safely.
        extra_tokens = shlex.split(hydra_overrides) if hydra_overrides else []

        cmd = [
            "python",
            run_infer_py,
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={local_out_dir}/rfout",
            *extra_tokens,
        ]

        cmd = [
            "python",
            run_infer_py,
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={local_out_dir}/rfout",
            *extra_tokens,
        ]

        run_command(cmd, cwd=RFD_REPO_DIR, env=env)

        # cache outputs to the output Volume for later reuse/inspection.
        run_command(["bash", "-lc", f"cp -a '{local_out_dir}/.' '{cached_run_dir}/'"], env=env)
        RFD_OUT_VOLUME.commit()

        # bundle only the interesting artifacts (smaller/faster downloads).
        selected = collect_outputs_for_bundle(str(local_out_dir))
        if selected:
            tar_bytes = package_files_to_tar_zst(selected, base_dir=str(local_out_dir))
        else:
            # Fallback: if matching fails for some reason, bundle everything.
            tar_bytes = package_dir_to_tar_zst(str(local_out_dir))

        return tar_bytes




# -------------------------
# Local entrypoint (CLI)
# -------------------------
@app.local_entrypoint()
def main(
    run_name: str | None = None,
    input_pdb: str | None = None,
    rfd_args: str = "",
    download_models: bool = False,
    force_redownload: bool = False,
    out_dir: str | None = None,
):
    if download_models:
        download_rfdiffusion_models.remote(force=force_redownload)
        return

    if run_name is None:
        raise ValueError("Missing required --run-name")
    if input_pdb is None:
        raise ValueError("Missing required --input-pdb (path to local .pdb)")

    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    pdb_bytes = input_path.read_bytes()

    local_out = Path(out_dir).expanduser().resolve() if out_dir else Path.cwd()
    local_out.mkdir(parents=True, exist_ok=True)
    out_file = local_out / f"{run_name}_rfdiffusion_outputs.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output already exists: {out_file}")

    tar_bytes = rfdiffusion_infer.remote(
        input_pdb_bytes=pdb_bytes,
        input_pdb_name=input_path.name,
        run_name=run_name,
        rfd_args=rfd_args,
    )
    out_file.write_bytes(tar_bytes)
    print(f"Done. Saved: {out_file}")
