r"""RFdiffusion source repo: <https://github.com/RosettaCommons/RFdiffusion>.

## Notes

* Checkpoint URLs are hardcoded from the upstream script:
  https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh
* For a complete set of RFdiffusion Hydra override keys, see RFdiffusion docs and `scripts/run_inference.py`.

## Outputs

* A `.tar.zst` archive will be written to `--out-dir` (or `$CWD`) named:
  `<run-name>_RFdiffusion.tar.zst`.
* The same run outputs are cached on the output volume for later inspection/reuse
  under the run name key.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from modal import App, Image

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.app_run import volume_path_from_mount_path
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.shell import (
    find_with_fd,
    package_outputs,
    run_command,
    sanitize_filename,
    softlink_dir,
    warmup_directory,
)
from biomodals.helper.web import download_files
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)

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

    model_dir = Path(CONF.model_volume_mountpoint) / "models"
    try:
        download_files(
            {v: model_dir / k for k, v in checkpoint_urls.items()},
            force=force,
            progress_bar_desc="💊 RFD checkpoints",
        )
    finally:
        MODEL_VOLUME.commit()

    print("💊 RFdiffusion checkpoints downloaded and committed.")


def _rfdiffusion_infer(
    input_pdb_bytes: bytes, input_pdb_name: str, run_name: str, hydra_overrides: str
) -> str:
    """Run RFdiffusion inference and return the cached output directory."""
    import shutil
    from tempfile import TemporaryDirectory

    # ---- cached output dir (persistent volume) ----
    cached_run_dir = Path(CONF.output_volume_mountpoint) / run_name
    cached_run_dir.mkdir(parents=True, exist_ok=True)

    # ---- compute hash based on all inputs ----
    input_hash = hash_string(
        ":".join((input_pdb_bytes.decode("utf-8"), hydra_overrides))
    )
    success_marker = cached_run_dir / "SUCCESS"
    if success_marker.exists():
        marker_hash = success_marker.read_text().strip()
        if marker_hash == input_hash:
            print(f"💊 Found RFdiffusion cached output: hash={marker_hash}")
            return str(cached_run_dir)

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

        # Symlink to model and IGSO3 cache directories
        for subdir in ("models", "schedules"):
            softlink_dir(
                Path(CONF.model_volume_mountpoint) / subdir, CONF.git_clone_dir / subdir
            )

        # ---- run inference (writes directly into cache volume) ----
        try:
            run_command(
                cmd,
                output_mode="tee",
                log_file=cached_run_dir / f"{run_name}-{CONF.name}.log",
                cwd=CONF.git_clone_dir,
            )
            CONF.output_volume.commit()

            # If trajectories are generated, compress them to save space
            traj_dir = rfd_out_dir / "traj"
            traj_tarball = rfd_out_dir / f"{run_name}_traj.tar.zst"
            if not traj_tarball.exists() and traj_dir.exists():
                traj_tarball.write_bytes(package_outputs(traj_dir))
            if traj_tarball.exists() and traj_dir.exists():
                shutil.rmtree(traj_dir)

            success_marker.write_text(input_hash)
        finally:
            CONF.output_volume.commit()

    return str(cached_run_dir)


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 32768),
    timeout=CONF.timeout,
    # Do not mount as read-only because we may need to write IGSO3 cache
    volumes=CONF.mounts(output_volume=True, model_volume=True, model_ro=False),
)
def rfdiffusion_infer(
    input_pdb_bytes: bytes, input_pdb_name: str, run_name: str, hydra_overrides: str
) -> AppRunResult:
    """Run RFdiffusion and return a workflow-compatible output directory."""
    safe_run_name = sanitize_filename(run_name)
    run_dir = _rfdiffusion_infer(
        input_pdb_bytes=input_pdb_bytes,
        input_pdb_name=input_pdb_name,
        run_name=safe_run_name,
        hydra_overrides=hydra_overrides,
    )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name=f"{CONF.name}_outputs",
                kind=ArtifactKind.DIRECTORY,
                storage=volume_path_from_mount_path(
                    remote_path=f"{run_dir}/rfd-scaffolds",
                    mount_root=CONF.output_volume_mountpoint,
                    volume_name=CONF.output_volume_name,
                ),
                metadata={"run_name": safe_run_name},
            )
        ],
        logs=[
            AppOutput(
                name=f"{CONF.name}_log",
                kind=ArtifactKind.LOGS,
                storage=volume_path_from_mount_path(
                    remote_path=f"{run_dir}/{safe_run_name}-{CONF.name}.log",
                    mount_root=CONF.output_volume_mountpoint,
                    volume_name=CONF.output_volume_name,
                ),
            )
        ],
    )


@app.function(timeout=CONF.timeout, volumes=CONF.mounts(output_volume=True))
def bundle_rfdiffusion_outputs(run_name: str) -> bytes:
    """Bundle an RFdiffusion output directory from the app output volume."""
    run_dir = Path(CONF.output_volume_mountpoint) / run_name
    remote_run_dir = volume_path_from_mount_path(
        str(run_dir), CONF.output_volume_mountpoint, CONF.output_volume_name
    )
    print(f"💊 RFdiffusion cached outputs: {remote_run_dir}", flush=True)
    warmup_directory(run_dir)

    exts = "|".join(("pdb", "trb", "log", "tar.zst"))
    selected = find_with_fd(run_dir, rf"\.({exts})$", "-tf")
    return package_outputs(run_dir, paths_to_bundle=selected or None)


# -------------------------
# Local entrypoint (CLI)
# -------------------------
def build_rfdiffusion_hydra_overrides(
    *,
    contigs: str | None = None,
    num_designs: int = 1,
    hotspot_res: str | None = None,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
    rfd_args: str = "",
) -> str:
    """Build RFdiffusion Hydra override args from entrypoint/workflow settings."""
    overrides: list[str] = [
        f"inference.num_designs={int(num_designs)}",
        f"denoiser.noise_scale_ca={noise_scale_ca}",
        f"denoiser.noise_scale_frame={noise_scale_frame}",
    ]
    if contigs:
        overrides.append(f"contigmap.contigs=[{contigs}]")
    if hotspot_res:
        overrides.append(f"ppi.hotspot_res=[{hotspot_res.replace(' ', ',')}]")
    if rfd_args.strip():
        overrides.extend(shlex.split(rfd_args))
    return shlex.join(overrides)


@app.local_entrypoint()
def submit_rfdiffusion_task(
    run_name: str | None = None,
    input_pdb: str | None = None,
    contigs: str | None = None,
    num_designs: int = 1,
    hotspot_res: str | None = None,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
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
        noise_scale_ca: Convenience wrapper for `denoiser.noise_scale_ca`.
        noise_scale_frame: Convenience wrapper for `denoiser.noise_scale_frame`.
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

    hydra_overrides = build_rfdiffusion_hydra_overrides(
        contigs=contigs,
        num_designs=num_designs,
        hotspot_res=hotspot_res,
        noise_scale_ca=noise_scale_ca,
        noise_scale_frame=noise_scale_frame,
        rfd_args=rfd_args,
    )
    pdb_bytes = input_path.read_bytes()

    local_out = Path(out_dir).expanduser().resolve() if out_dir else Path.cwd()
    local_out.mkdir(parents=True, exist_ok=True)
    out_file = local_out / f"{run_name}_{CONF.name}.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output already exists: {out_file}")

    result = AppRunResult.model_validate(
        rfdiffusion_infer.remote(
            input_pdb_bytes=pdb_bytes,
            input_pdb_name=input_path.name,
            run_name=run_name,
            hydra_overrides=hydra_overrides,
        )
    )
    output = next(
        output for output in result.outputs if output.name == f"{CONF.name}_outputs"
    )
    if not isinstance(output.storage, VolumePath):
        raise TypeError(f"{CONF.name} outputs should use volume path storage")
    tar_bytes = bundle_rfdiffusion_outputs.remote(run_name=output.metadata["run_name"])
    out_file.write_bytes(tar_bytes)
    print(f"🧬 Done. Saved: {out_file}")
