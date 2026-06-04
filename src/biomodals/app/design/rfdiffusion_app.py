r"""RFdiffusion source repo: <https://github.com/RosettaCommons/RFdiffusion>.

## Notes

* Checkpoint URLs are hardcoded from the upstream script:
  https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
* For a complete set of RFdiffusion Hydra override keys, see RFdiffusion docs and `scripts/run_inference.py`.
* Checkpoints are stored in a persistent Modal volume (`rfdiffusion-models`).
* Outputs are cached in a persistent Modal volume (`rfdiffusion-outputs`) under:
  `/root/rfdiffusion_outputs/<run-name>/`
* The returned tarball bundles only “useful” artifacts by default (e.g. `.pdb`, `.trb`, `.log`, `.json`, `.yaml/.yml`, `.csv`).
  If the selection step yields nothing, it falls back to bundling the entire local output directory.

## Outputs

* A `.tar.zst` archive will be written to `--out-dir` (or `$CWD`) named:
  `<run-name>_rfdiffusion_outputs.tar.zst`.
* The same run outputs are cached on the output volume for later inspection/reuse
  under the run name key.

## Typical usage:

```bash
# 1) Download checkpoints into the persistent models volume (run once)
modal run rfdiffusion_app.py --download-models --force-redownload

# 2) Run inference (binder design / scaffold etc.)
modal run rfdiffusion_app.py \
    --run-name demo1 \
    --input-pdb ~/outputs/rfdiffusion_app/RBD_wt.pdb \
    --contigs "100-150/0 E333-526" \
    --num-designs 2 \
    --hotspot-res "E405,E408"
```
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from modal import App, Image

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.shell import (
    find_with_fd,
    package_outputs,
    run_command_with_log,
    sanitize_filename,
    warmup_directory,
)
from biomodals.helper.volume_run import volume_path_from_mount_path
from biomodals.helper.web import download_files

# -------------------------
# Modal configs
# -------------------------
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="RFdiffusion",
    repo_url="https://github.com/RosettaCommons/RFdiffusion",
    repo_commit_hash="2d0c003df46b9db41d119321f15403dec3716cd9",
    python_version="3.10",
    cuda_version="cu121",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "36000")),
    model_volume_mountpoint="/opt/RFdiffusion/models",
)

# -------------------------
# Image definition
# -------------------------
# Ref: example of a newer, modern CUDA/PyTorch-style Docker environment, see:
# https://github.com/JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090/blob/main/RTX-5090.dockerfile
runtime_image = (
    Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential")
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
        ))
    )
    .env(CONF.default_env | {"DGLBACKEND": "pytorch"})
    # Pin torch < 2.6 to avoid the torch.load(weights_only=...) default behavior change.
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1")
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
        "torchdata>=0.7",  # DGL / datapipes support
        "dgl==1.1.3",  # DGL CUDA wheel
        # Where to find the CUDA wheels for DGL
        find_links=f"https://data.dgl.ai/wheels/{CONF.cuda_version}/repo.html",
    )
    .run_commands(
        # build/install NVIDIA SE3Transformer in one chained step.
        " && ".join((
            f"cd {CONF.git_clone_dir}/env/SE3Transformer",
            "uv pip install --system --no-build-isolation --no-cache-dir -r requirements.txt",
            "python setup.py install",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .env({"PYTHONPATH": str(CONF.git_clone_dir)})
    .pipe(patch_image_for_helper, skip_deps=["uniaf3"])
)
app = App(CONF.name, image=runtime_image, tags=CONF.tags)


@app.function(
    timeout=CONF.timeout, volumes=CONF.mounts(model_volume=True, model_ro=False)
)
def download_rfdiffusion_models(force: bool = False) -> None:
    """Download RFdiffusion checkpoints into the persistent models Volume."""
    # RFdiffusion checkpoints (hardcoded from upstream)
    # https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
    base_url = "http://files.ipd.uw.edu/pub/RFdiffusion"
    checkpoint_urls: dict[str, str] = {
        "Base_ckpt.pt": f"{base_url}/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
        "Complex_base_ckpt.pt": f"{base_url}/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
        "Complex_Fold_base_ckpt.pt": f"{base_url}/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
        "InpaintSeq_ckpt.pt": f"{base_url}/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
        "InpaintSeq_Fold_ckpt.pt": f"{base_url}/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
        "ActiveSite_ckpt.pt": f"{base_url}/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
        "Base_epoch8_ckpt.pt": f"{base_url}/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt",
    }

    model_dir = Path(CONF.model_volume_mountpoint)
    try:
        download_files(
            {v: model_dir / k for k, v in checkpoint_urls.items()},
            force=force,
            progress_bar_desc="💊 RFD checkpoints",
        )
    finally:
        MODEL_VOLUME.commit()

    print("💊 RFdiffusion checkpoints downloaded and committed.")


def _bundle_outputs(run_dir: str) -> bytes:
    """Bundle the outputs of the run into a .tar.zst archive."""
    remote_run_dir = volume_path_from_mount_path(
        run_dir, CONF.output_volume_mountpoint, CONF.output_volume_name
    )
    print(f"💊 RFdiffusion cached outputs: {remote_run_dir}", flush=True)
    warmup_directory(run_dir)
    exts = "|".join(("pdb", "trb", "json", "yaml", "yml", "log", "txt", "csv"))
    selected = find_with_fd(run_dir, rf"\.({exts})$", "-tf")
    return package_outputs(run_dir, paths_to_bundle=selected or None)


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def rfdiffusion_infer(
    input_pdb_bytes: bytes, input_pdb_name: str, run_name: str, hydra_overrides: str
) -> bytes:
    """Run RFdiffusion inference inside the container and return a .tar.zst bundle.

    - Partial results are preserved if the run is interrupted.
    - A SUCCESS marker file is written only after successful completion.
    """
    from tempfile import TemporaryDirectory

    # ---- cached output dir (persistent volume) ----
    cached_run_dir = Path(CONF.output_volume_mountpoint) / run_name
    cached_run_dir.mkdir(parents=True, exist_ok=True)

    # ---- compute hash based on all inputs ----
    input_hash = hash_string(
        ":".join((
            input_pdb_bytes.decode("utf-8"),
            hydra_overrides,
        ))
    )
    success_marker = cached_run_dir / "SUCCESS"
    if success_marker.exists():
        marker_hash = success_marker.read_text().strip()
        if marker_hash == input_hash:
            print(f"💊 Found RFdiffusion cached output: hash={marker_hash}")
            return _bundle_outputs(str(cached_run_dir))

    # optional: keep actual RFdiffusion outputs in a subdir
    rfd_out_dir = cached_run_dir / "rfd-scaffolds"
    rfd_out_dir.mkdir(exist_ok=True)
    out_prefix = rfd_out_dir / run_name

    with TemporaryDirectory() as tmpdir:
        # ---- input pdb (tmp is fine) ----
        input_pdb = Path(tmpdir) / input_pdb_name
        input_pdb.write_bytes(input_pdb_bytes)

        # hydra overrides are passed as a single string, split safely.
        cmd = [
            sys.executable,
            f"{CONF.git_clone_dir}/scripts/run_inference.py",
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={out_prefix}",
            *shlex.split(hydra_overrides),
        ]

        # ---- run inference (writes directly into cache volume) ----
        try:
            run_command_with_log(
                cmd,
                log_file=cached_run_dir / f"{run_name}-{CONF.name}.log",
                verbose=True,
                cwd=CONF.git_clone_dir,
            )
            success_marker.write_text(input_hash)
        finally:
            CONF.output_volume.commit()

    return _bundle_outputs(str(cached_run_dir))


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
    rfd_args: str = "",
    download_models: bool = False,
    force_redownload: bool = False,
    out_dir: str | None = None,
):
    """Submit an RFdiffusion inference job to Modal.

    Args:
        run_name: Unique name for this run. Used as the output-volume cache key
            and as part of the returned output archive filename.
        input_pdb: Path to the input PDB file on the local machine. The file
            will be uploaded to the Modal worker before inference starts.
        contigs: Convenience wrapper for `contigmap.contigs` (Hydra override),
            e.g. `"100-150/0 E333-526"`. This simplifies common RFdiffusion
            use cases such as binder or scaffold design.
        num_designs: Convenience wrapper for `inference.num_designs`.
        hotspot_res: Convenience wrapper for `ppi.hotspot_res`,
            e.g. `"E405,E408"`. Typically used for binder design.
        rfd_args: Raw RFdiffusion Hydra overrides passed directly to the
            inference script. This is an escape hatch for advanced options.
        download_models: If set, download RFdiffusion checkpoint weights into
            the persistent models volume and exit without running inference.
        force_redownload: Force re-download checkpoint weights even if they
            already exist in the models volume.
        out_dir: Optional local directory where the output `.tar.zst` archive
            will be written. Defaults to the current working directory.

    """
    if download_models:
        download_rfdiffusion_models.remote(force=force_redownload)
        return

    if run_name is None:
        raise ValueError("Missing required --run-name")

    run_name = sanitize_filename(run_name)

    if input_pdb is None:
        raise ValueError("Missing required --input-pdb (path to local .pdb)")

    input_path = Path(input_pdb)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

    # Build Hydra overrides string from structured arguments.
    overrides: list[str] = []

    if contigs:
        overrides.append(f"contigmap.contigs=[{contigs}]")  # keep as a single token
    if num_designs:
        overrides.append(f"inference.num_designs={int(num_designs)}")
    if hotspot_res:
        # Accept "E405,E408" or "E405 E408"
        hs = hotspot_res.replace(" ", ",")
        overrides.append(f"ppi.hotspot_res=[{hs}]")

    # Prefer extra_overrides; keep rfd_args as a deprecated escape hatch.
    if rfd_args.strip():
        overrides.extend(shlex.split(rfd_args))

    if not overrides:
        raise ValueError(
            "At least one of 'contigs', 'num_designs', 'hotspot_res' or 'rfd_args' is required"
        )
    hydra_overrides = shlex.join(overrides)
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
    print(f"🧬 Done. Saved: {out_file}")
