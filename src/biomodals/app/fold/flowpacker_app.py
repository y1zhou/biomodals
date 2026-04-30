"""FlowPacker source repo: <https://gitlab.com/mjslee0921/flowpacker>.

FlowPacker performs protein side-chain packing with flow matching. The upstream
inference script operates on a folder of PDB files; this Biomodals wrapper accepts
a single `.pdb`/`.cif` file or a folder containing `.pdb`/`.cif` files and runs
the same batch inference path on Modal.

See the upstream entrypoint and paper for input constraints:

* <https://gitlab.com/mjslee0921/flowpacker/-/blob/main/sampler_pdb.py>
* <https://www.biorxiv.org/content/10.1101/2024.07.05.602280v1>

## Additional notes

* The upstream checkpoints are stored with Git LFS. The first run downloads them
  into the shared Biomodals model volume.
* FlowPacker recommends `cluster.pth` for most use cases; `bc40.pth` is also
  available.
* Very large structures can require more GPU memory. Override the GPU with the
  `GPU` environment variable, for example `GPU=A100-80G`.

## Outputs

Results are saved locally as `<run-name>.tar.zst`. The archive contains
FlowPacker's `run_*` sampled PDB folders, `output_dict.pth`, the generated
inference config, and `flowpacker.log`. When `--use-confidence` is enabled, the
archive also includes FlowPacker's `best_run` folder.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.helper import patch_image_for_helper
from biomodals.helper.shell import package_outputs, run_command_with_log

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="FlowPacker",
    repo_url="https://gitlab.com/mjslee0921/flowpacker",
    repo_commit_hash="03421c7fdda73862994aa54fb3077f3f6561408c",
    package_name="flowpacker",
    python_version="3.11",  # numpy 1.22.4 requires 3.10
    cuda_version="cu121",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)


@dataclass(frozen=True)
class AppInfo:
    """Container for FlowPacker-specific constants."""

    supported_models: Sequence[str] = ("cluster", "bc40")
    checkpoint_names: Sequence[str] = ("cluster", "bc40", "confidence")
    default_sample_coeff: float = 5.0
    default_num_steps: int = 10
    dependencies: Sequence[str] = (
        "torch==2.3.0",
        "tqdm",
        "tensorboard",
        "pyyaml",
        "easydict",
        "biotite",
        "dm-tree",
        "biopython",
        "modelcif",
        "torch_geometric",
        "numpy<2",
        "torch_cluster",
        "pandas",
        "e3nn",
    )
    dependency_cutoff: str = "2025-02-10"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "git-lfs", "build-essential")
    .env(CONF.default_env | {"GIT_LFS_SKIP_SMUDGE": "1"})
    .run_commands(
        " && ".join(
            (
                f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
                f"cd {CONF.git_clone_dir}",
                f"git checkout {CONF.repo_commit_hash}",
            )
        )
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(
        *APP_INFO.dependencies,
        find_links=f"https://data.pyg.org/whl/torch-2.3.0+{CONF.cuda_version}.html",
        extra_options=f"--exclude-newer {APP_INFO.dependency_cutoff}",
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
def _checkpoint_path(name: str) -> Path:
    """Return the shared-volume path for a FlowPacker checkpoint."""
    return CONF.model_dir / "checkpoints" / f"{name}.pth"


@app.function(
    cpu=(0.125, 8.125),
    timeout=CONF.timeout,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME},
)
def download_flowpacker_checkpoints(force: bool = False) -> None:
    """Download FlowPacker Git LFS checkpoints into the model volume."""
    from biomodals.helper.shell import run_command

    checkpoint_dir = CONF.model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wanted = [_checkpoint_path(name) for name in APP_INFO.checkpoint_names]
    if not force and all(path.exists() for path in wanted):
        print("💊 FlowPacker checkpoints already exist in the model volume")
        return

    if force:
        _ = [path.unlink(missing_ok=True) for path in wanted]

    include_paths = ",".join(
        f"checkpoints/{name}.pth" for name in APP_INFO.checkpoint_names
    )
    run_command(["git", "lfs", "install", "--skip-repo"], cwd=CONF.git_clone_dir)
    run_command(
        ["git", "lfs", "pull", "--include", include_paths, "--exclude", ""],
        cwd=CONF.git_clone_dir,
        env={"GIT_LFS_SKIP_SMUDGE": "0"},
    )

    for name in APP_INFO.checkpoint_names:
        src = CONF.git_clone_dir / "checkpoints" / f"{name}.pth"
        dst = _checkpoint_path(name)
        shutil.copy2(src, dst)
        MODEL_VOLUME.commit()
    print("💊 FlowPacker checkpoint download complete")


##########################################
# Inference functions
##########################################
def _ensure_checkpoint_symlink() -> None:
    """Link the repo checkpoint directory to the mounted model volume."""
    checkpoint_src = CONF.model_dir / "checkpoints"
    checkpoint_link = CONF.git_clone_dir / "checkpoints"
    if checkpoint_link.is_symlink() and checkpoint_link.resolve() == checkpoint_src:
        return

    if checkpoint_link.is_dir() and not checkpoint_link.is_symlink():
        shutil.rmtree(checkpoint_link)
    elif checkpoint_link.exists() or checkpoint_link.is_symlink():
        checkpoint_link.unlink()

    checkpoint_link.symlink_to(checkpoint_src, target_is_directory=True)


def _write_flowpacker_config(
    config_path: Path,
    *,
    input_dir: Path,
    model_name: str,
    use_confidence: bool,
    n_samples: int,
    num_steps: int,
    sample_coeff: float,
) -> None:
    """Write the upstream FlowPacker inference config for one Modal run."""
    import yaml

    conf_ckpt = "./checkpoints/confidence.pth" if use_confidence else None
    config = {
        "mode": "vf",
        "data": {
            "data": model_name,
            "train_path": None,
            "cluster_path": None,
            "test_path": str(input_dir),
            "min_length": 1,
            "max_length": 4096,
            "edge_type": "knn",
            "max_radius": 16.0,
            "max_neighbors": 30,
        },
        "ckpt": f"./checkpoints/{model_name}.pth",
        "conf_ckpt": conf_ckpt,
        "sample": {
            "batch_size": 1,
            "n_samples": n_samples,
            "use_ema": True,
            "eps": 2.0e-3,
            "save_trajectory": False,
            "coeff": sample_coeff,
            "num_steps": num_steps,
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    # Cannot mount as ro as FlowPacker mkdirs there for whatever reason
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME},
)
def run_flowpacker(
    input_files: list[tuple[str, bytes]],
    run_name: str,
    model_name: str = "cluster",
    use_confidence: bool = False,
    n_samples: int = 1,
    num_steps: int = APP_INFO.default_num_steps,
    sample_coeff: float = APP_INFO.default_sample_coeff,
    use_gt_masks: bool = False,
    inpaint: str | None = None,
    save_traj: bool = False,
    seed: int = 42,
) -> bytes:
    """Run FlowPacker inference and return packaged outputs."""
    import sys
    from tempfile import mkdtemp

    if model_name not in APP_INFO.supported_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {APP_INFO.supported_models}"
        )
    ckpt_path = _checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FlowPacker checkpoint is missing: {ckpt_path}")

    confidence_ckpt_path = _checkpoint_path("confidence")
    if use_confidence and not confidence_ckpt_path.exists():
        raise FileNotFoundError(
            f"FlowPacker confidence checkpoint is missing: {confidence_ckpt_path}"
        )

    _ensure_checkpoint_symlink()

    input_dir = Path(mkdtemp(prefix="flowpacker_inputs_"))
    sample_dir = CONF.git_clone_dir / "samples" / run_name
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    for file_name, content in input_files:
        dst = input_dir / Path(file_name).name
        dst.write_bytes(content)

    config_path = CONF.git_clone_dir / "config" / "inference" / "biomodals.yaml"
    _write_flowpacker_config(
        config_path,
        input_dir=input_dir,
        model_name=model_name,
        use_confidence=use_confidence,
        n_samples=n_samples,
        num_steps=num_steps,
        sample_coeff=sample_coeff,
    )

    cmd = [
        sys.executable,
        "sampler_pdb.py",
        "biomodals",
        run_name,
        "--seed",
        str(seed),
    ]
    if save_traj:
        cmd.extend(["--save_traj", "True"])
    if use_gt_masks:
        cmd.extend(["--use_gt_masks", "True"])
    if inpaint:
        cmd.extend(["--inpaint", inpaint])

    print(
        f"💊 Running FlowPacker with model '{model_name}' on {len(input_files)} input(s)"
    )
    log_path = sample_dir / "flowpacker.log"
    run_command_with_log(cmd, log_file=log_path, cwd=CONF.git_clone_dir, verbose=True)

    if not sample_dir.exists():
        raise RuntimeError(
            f"FlowPacker did not create expected output directory: {sample_dir}"
        )
    shutil.copy2(config_path, sample_dir / "biomodals_inference.yaml")
    return package_outputs(sample_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
def _collect_input_files(input_path: Path) -> list[tuple[str, bytes]]:
    """Collect FlowPacker structure inputs as `(file name, bytes)` tuples."""
    allowed_suffixes = {".pdb", ".cif"}
    if input_path.is_file():
        if input_path.suffix.lower() not in allowed_suffixes:
            raise ValueError("FlowPacker input file must end in .pdb or .cif")
        return [(input_path.name, input_path.read_bytes())]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    structure_files = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_suffixes
    )
    if not structure_files:
        raise FileNotFoundError(
            f"No .pdb or .cif files found directly under input directory: {input_path}"
        )
    return [(p.name, p.read_bytes()) for p in structure_files]


@app.local_entrypoint()
def submit_flowpacker_task(
    input_path: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    model_name: str = "cluster",
    use_confidence: bool = False,
    n_samples: int = 1,
    num_steps: int = APP_INFO.default_num_steps,
    sample_coeff: float = APP_INFO.default_sample_coeff,
    use_gt_masks: bool = False,
    inpaint: str | None = None,
    save_traj: bool = False,
    seed: int = 42,
    download_models: bool = False,
    force_redownload: bool = False,
) -> None:
    """Run FlowPacker side-chain packing on Modal and fetch results.

    Args:
        input_path: Path to a `.pdb`/`.cif` file or a directory containing
            `.pdb`/`.cif` files. Directory inputs are read non-recursively.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the input
            file or directory name.
        model_name: FlowPacker checkpoint to use. Choose `cluster` for the
            recommended PDB-S40 model or `bc40` for the BC40 model.
        use_confidence: Use FlowPacker's confidence model to populate `best_run`.
        n_samples: Number of side-chain samples to generate for each input.
        num_steps: Number of flow-matching decode steps.
        sample_coeff: FlowPacker sampling coefficient.
        use_gt_masks: Generate samples for all chi angles instead of only the
            missing-coordinate mask detected from the input structure.
        inpaint: Optional residue design mask such as `A_10-25/B_40/C`, which
            designs res 10-25 of chain A, 40 of chain B and all of chain C.
        save_traj: Save the full sampling trajectory instead of only final samples.
        seed: Random seed for FlowPacker inference.
        download_models: Download FlowPacker checkpoints and exit without inference.
        force_redownload: Force checkpoint redownload even when cached files exist.
    """
    if model_name not in APP_INFO.supported_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {APP_INFO.supported_models}"
        )
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if sample_coeff <= 0:
        raise ValueError("sample_coeff must be positive")

    print("🧬 Ensuring FlowPacker checkpoints are available")
    download_flowpacker_checkpoints.remote(force=force_redownload)
    if download_models:
        print("🧬 FlowPacker checkpoint download complete")
        return

    resolved_input = Path(input_path).expanduser().resolve()
    input_files = _collect_input_files(resolved_input)
    if run_name is None:
        run_name = (
            resolved_input.stem if resolved_input.is_file() else resolved_input.name
        )

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    print(f"🧬 Submitting FlowPacker run '{run_name}' with {len(input_files)} input(s)")
    tarball_bytes = run_flowpacker.remote(
        input_files=input_files,
        run_name=run_name,
        model_name=model_name,
        use_confidence=use_confidence,
        n_samples=n_samples,
        num_steps=num_steps,
        sample_coeff=sample_coeff,
        use_gt_masks=use_gt_masks,
        inpaint=inpaint,
        save_traj=save_traj,
        seed=seed,
    )

    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 Run complete! Results saved to {out_file}")
