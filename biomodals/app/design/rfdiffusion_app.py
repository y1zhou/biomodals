"""RFdiffusion source repo: <https://github.com/RosettaCommons/RFdiffusion>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--run-name` | **Required** | Unique name used for output tarball name and output-volume cache key. |
| `--input-pdb` | **Required** | Local path to the input PDB file (uploaded to the Modal worker). |
| `--out-dir` | `$CWD` | Optional local output directory for the returned `.tar.zst` bundle. |
| `--contigs` | `None` | Convenience wrapper for `contigmap.contigs` (Hydra override). Example: `"100-150/0 E333-526"`. |
| `--num-designs` | `1` | Convenience wrapper for `inference.num_designs` (Hydra override). |
| `--hotspot-res` | `None` | Convenience wrapper for `ppi.hotspot_res` (Hydra override). Example: `"E405,E408"`. |
| `--rfd-args` | `""` | Additional Hydra overrides as a raw string (escape hatch / backward-compatible). |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download checkpoint weights and skip running inference. |
| `--force-redownload`/`--no-force-redownload` | `--no-force-redownload` | Force re-download checkpoints even if they already exist in the models volume. |

For a complete set of RFdiffusion Hydra override keys, see RFdiffusion docs and `scripts/run_inference.py`.

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `RFdiffusion` | Name of the Modal app to use. |
| `GPU` | `A100` | Type of GPU to use. See https://modal.com/pricing for details. |
| `TIMEOUT` | `36000` | Timeout for the inference Modal function in seconds. |

## Notes

- Checkpoint URLs are hardcoded from the upstream script:
  https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
- Checkpoints are stored in a persistent Modal volume (`rfdiffusion-models`).
- Outputs are cached in a persistent Modal volume (`rfdiffusion-outputs`) under:
  `/root/rfdiffusion_outputs/<run-name>/`
- The returned tarball bundles only “useful” artifacts by default (e.g. `.pdb`, `.trb`, `.log`, `.json`, `.yaml/.yml`, `.csv`).
  If the selection step yields nothing, it falls back to bundling the entire local output directory.

## Outputs

* A `.tar.zst` archive will be written to `--out-dir` (or `$CWD`) named:
  `<run-name>_rfdiffusion_outputs.tar.zst`.
* The same run outputs are cached on the output volume for later inspection/reuse
  under the run name key.

## Typical usage:

  # 1) Download checkpoints into the persistent models volume (run once)
  modal run rfdiffusion_app.py --download-models --force-redownload

  # 2) Run inference (binder design / scaffold etc.)
  modal run rfdiffusion_app.py \
    --run-name demo1 \
    --input-pdb ~/outputs/rfdiffusion_app/RBD_wt.pdb \
    --contigs "100-150/0 E333-526" \
    --num-designs 2 \
    --hotspot-res "E405,E408"
"""

from __future__ import annotations

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
# For an example of a newer, modern CUDA/PyTorch-style Docker environment, see:
# https://github.com/JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090/blob/main/RTX-5090.dockerfile
# The runtime image below is defined directly with Modal and is not built from that Dockerfile.
runtime_image = (
    Image.debian_slim(python_version="3.10")
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
    .run_commands("ln -s /usr/bin/fdfind /usr/local/bin/fd")
    .run_commands(
        f"git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git {RFD_REPO_DIR}"
    )
    .env(
        {
            "PYTHONPATH": RFD_REPO_DIR,
            "PYTHONUNBUFFERED": "1",
            "DGLBACKEND": "pytorch",
            "UV_TORCH_BACKEND": "cu121",
        }
    )
    # install CUDA-enabled PyTorch from official index (avoid accidental CPU-only wheels).
    # Pin torch < 2.6 to avoid the torch.load(weights_only=...) default behavior change.
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
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
        "pyrsistent",  # RFdiffusion symmetry
        "aiohttp",  # async checkpoint download
        "torchdata>=0.7",  # DGL / datapipes support
        "dgl==1.1.3",  # DGL CUDA wheel
        # Where to find the CUDA wheels for DGL
        find_links="https://data.dgl.ai/wheels/cu121/repo.html",
    )
    .run_commands(
        # build/install NVIDIA SE3Transformer in one chained step.
        f"cd {RFD_REPO_DIR}/env/SE3Transformer && "
        "python -m pip install --no-cache-dir -r requirements.txt && "
        "python setup.py install"
    )
)


app = App(APP_NAME, image=runtime_image)


# -------------------------
# Helpers
# -------------------------

import re
import shutil

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_run_name(run_name: str) -> str:
    """Validate run_name to prevent path traversal and keep filenames predictable.

    Allowed: [A-Za-z0-9][A-Za-z0-9._-]{0,63}
    - must start with alnum
    - max length 64
    - no slashes, whitespace, or ".."
    """
    if not _RUN_NAME_RE.match(run_name):
        raise ValueError(
            "Invalid --run-name. Use 1-64 chars of letters/digits plus . _ - "
            "(must start with a letter/digit)."
        )
    if ".." in run_name:
        raise ValueError("Invalid --run-name: '..' is not allowed.")
    return run_name


def run_command(cmd: list[str], cwd: str | Path | None = None, **kwargs) -> None:
    """Run a command and stream stdout/stderr.

    This is intentionally a thin wrapper around `subprocess.Popen` to:
    - avoid shell invocation (argv list only)
    - surface live logs in Modal
    - raise on non-zero exit

    Args:
        cmd: Command argv list (no shell).
        cwd: Optional working directory.
        **kwargs: Passed through to `subprocess.Popen` (e.g., env=...).

    """
    import shlex
    import subprocess as sp

    cwd_str = str(cwd) if cwd is not None else None
    print("Running:", shlex.join(cmd), f"(cwd={cwd_str})" if cwd_str else "")

    # Default streaming settings; callers may override via kwargs.
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, cwd=cwd_str, **kwargs) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
        rc = p.wait()
        if rc != 0:
            raise sp.CalledProcessError(rc, cmd)


def _require_fd() -> None:
    """Fail fast if `fd` is not available (required in the runtime image)."""
    if shutil.which("fd") is None:
        raise RuntimeError(
            "fd (fd-find) is required but was not found on PATH. "
            "Install fd-find in the runtime image and expose it as `fd`."
        )


def warmup_directory(
    dir_path: str | Path, file_pattern: str = ".", jobs: int = 256
) -> None:
    """Warm up the disk cache for files in a directory using `fd`.

    This is optional, and mostly useful when subsequent steps will re-read many
    files (e.g., large tarballs or model weights).
    """
    _require_fd()
    dir_path = Path(dir_path)

    cmd = [
        "fd",
        "-t",
        "f",
        file_pattern,
        str(dir_path),
        "-j",
        str(jobs),
        "-x",
        "dd",
        "if={}",
        "of=/dev/null",
        "bs=1M",
        "status=none",
    ]
    run_command(cmd)


def collect_outputs_for_bundle(root_dir: str | Path) -> list[Path]:
    """Collect the output artifacts we typically want to download.

    Uses `fd` only (no shell). Returns absolute Paths under `root_dir`, sorted for
    reproducible bundling.
    """
    _require_fd()
    import subprocess as sp

    root = Path(root_dir)
    pattern = r"\.(pdb|trb|json|ya?ml|log|txt|csv)$"

    root = Path(root_dir)
    exts = (".pdb", ".trb", ".json", ".yaml", ".yml", ".log", ".txt", ".csv")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    return sorted(files, key=lambda p: str(p))


def package_dir_to_tar_zst(dir_path: str | Path) -> bytes:
    """Package an entire directory into a tar.zst and return bytes."""
    import subprocess as sp

    dp = Path(dir_path)
    cmd = ["tar", "--zstd", "-cf", "-", dp.name]
    return sp.check_output(cmd, cwd=str(dp.parent))


def package_files_to_tar_zst(files: list[Path], base_dir: str | Path) -> bytes:
    """Create a tar.zst containing only selected files (relative to base_dir)."""
    import subprocess as sp
    import tempfile

    base = Path(base_dir)

    # Ensure all files are under base (avoid tar errors + path traversal).
    rel_paths: list[str] = []
    for p in files:
        rel_paths.append(str(Path(p).relative_to(base)))

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for rel in rel_paths:
            f.write(rel + "\n")
        filelist = f.name

    try:
        cmd = ["tar", "--zstd", "-cf", "-", "-C", str(base), "-T", filelist]
        return sp.check_output(cmd)
    finally:
        Path(filelist).unlink(missing_ok=True)


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
    """Download RFdiffusion checkpoints into the persistent models Volume.

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
    hydra_overrides: str,
) -> bytes:
    """Run RFdiffusion inference inside the container and return a .tar.zst bundle.

    - Outputs are written directly to /root/rfdiffusion_outputs/<run_name> on a persistent Volume.
      Partial results are preserved if the run is interrupted.
    - A SUCCESS marker file is written only after successful completion.
    """
    import shlex
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ---- input pdb (tmp is fine) ----
        input_pdb = tmp / input_pdb_name
        input_pdb.write_bytes(input_pdb_bytes)

        # ---- cached output dir (persistent volume) ----
        cached_run_dir = Path(RFD_OUT_DIR) / run_name
        cached_run_dir.mkdir(parents=True, exist_ok=True)

        # optional: keep actual RFdiffusion outputs in a subdir
        run_dir = cached_run_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        run_infer_py = f"{RFD_REPO_DIR}/scripts/run_inference.py"

        # hydra overrides are passed as a single string, split safely.
        extra_tokens = shlex.split(hydra_overrides) if hydra_overrides else []

        out_prefix = run_dir / "rfout"

        cmd = [
            "python",
            run_infer_py,
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={out_prefix}",
            *extra_tokens,
        ]

        # ---- run inference (writes directly into cache volume) ----
        run_command(cmd, cwd=RFD_REPO_DIR)

        # NOTE:
        # This SUCCESS marker is intentionally not consumed by the current code path.
        # It serves as a lightweight completion flag for external inspection
        # (e.g. humans, debugging, or future cache-reuse logic) to distinguish
        # completed runs from partial or interrupted ones.
        # ---- mark success (only reached if inference didn't error) ----
        success_marker = cached_run_dir / "SUCCESS"
        success_marker.write_text("ok\n", encoding="utf-8")

        # ---- commit cached outputs ----
        RFD_OUT_VOLUME.commit()

        # ---- bundle outputs for return ----
        selected = collect_outputs_for_bundle(str(run_dir))
        if selected:
            tar_bytes = package_files_to_tar_zst(selected, base_dir=str(run_dir))
        else:
            tar_bytes = package_dir_to_tar_zst(str(run_dir))

        return tar_bytes


# -------------------------
# Local entrypoint (CLI)
# -------------------------
@app.local_entrypoint()
def submit_rfdiffusion_task(
    run_name: str | None = None,
    input_pdb: str | None = None,
    contigs: str | None = None,
    num_designs: int = 1,
    hotspot_res: str | None = None,
    # Backwards-compatible "raw overrides" input (deprecated, but kept for convenience).
    rfd_args: str = "",
    download_models: bool = False,
    force_redownload: bool = False,
    out_dir: str | None = None,
):
    """Submit an RFdiffusion inference job to Modal.

    Parameters
    ----------
    run_name : str
        Unique name for this run. Used as the output-volume cache key and as part
        of the returned output archive filename.
    input_pdb : str
        Path to the input PDB file on the local machine. The file will be uploaded
        to the Modal worker before inference starts.
    contigs : str | None
        Convenience wrapper for `contigmap.contigs` (Hydra override). This argument
        simplifies common RFdiffusion use cases such as binder or scaffold design.
    num_designs : int
        Convenience wrapper for `inference.num_designs` (Hydra override).
    hotspot_res : str | None
        Convenience wrapper for `ppi.hotspot_res` (Hydra override), typically used
        for binder design.
    rfd_args : str
        Raw RFdiffusion Hydra overrides passed directly to the inference script.
        This acts as an escape hatch for advanced or unsupported options.
    download_models : bool
        If set, download RFdiffusion checkpoint weights into the persistent models
        volume and exit without running inference.
    force_redownload : bool
        Force re-download checkpoint weights even if they already exist in the
        models volume.
    out_dir : str | None
        Optional local directory where the output `.tar.zst` archive will be written.
        Defaults to the current working directory.

    Notes
    -----
    - For longer jobs, increase TIMEOUT via environment variable:
        TIMEOUT=360000 modal run rfdiffusion_app.py ...

    """
    if download_models:
        download_rfdiffusion_models.remote(force=force_redownload)
        return

    if run_name is None:
        raise ValueError("Missing required --run-name")

        run_name = validate_run_name(run_name)

    if input_pdb is None:
        raise ValueError("Missing required --input-pdb (path to local .pdb)")

    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    # Build Hydra overrides string from structured arguments.
    overrides: list[str] = []

    if contigs:
        overrides.append(f'contigmap.contigs="[{contigs}]"')  # keep as a single token
    if num_designs:
        overrides.append(f"inference.num_designs={int(num_designs)}")
    if hotspot_res:
        # Accept "E405,E408" or "E405 E408"
        hs = hotspot_res.replace(" ", ",")
        overrides.append(f"ppi.hotspot_res=[{hs}]")

    # Prefer extra_overrides; keep rfd_args as a deprecated escape hatch.
    if rfd_args.strip():
        overrides.extend(shlex.split(rfd_args))

    hydra_overrides = " ".join(overrides)

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
        hydra_overrides=hydra_overrides,
    )
    out_file.write_bytes(tar_bytes)
    print(f"Done. Saved: {out_file}")
