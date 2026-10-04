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
import orjson

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout, volume_app_output
from biomodals.helper.artifacts import sha256_bytes, sha256_file
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.pdb import validate_pdb_content
from biomodals.helper.shell import (
    copy_files,
    package_outputs,
    run_command,
    sanitize_filename,
    softlink_dir,
)
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

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

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "git-lfs", "build-essential")
    .env(CONF.default_env | {"GIT_LFS_SKIP_SMUDGE": "1"})
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(
        *APP_INFO.dependencies,
        find_links=f"https://data.pyg.org/whl/torch-2.3.0+{CONF.cuda_version}.html",
        extra_options=f"--exclude-newer {APP_INFO.dependency_cutoff}",
    )
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
@app.function(
    cpu=(0.125, 8.125),
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_ro=False),
)
def download_flowpacker_checkpoints(force: bool = False) -> None:
    """Download FlowPacker Git LFS checkpoints into the model volume."""
    from biomodals.helper.shell import run_command

    cache_dir = Path(CONF.model_volume_mountpoint)
    all_checkpoints = [cache_dir / f"{name}.pth" for name in APP_INFO.checkpoint_names]
    missing_checkpoints = [
        f for f in all_checkpoints if not (f.exists() and f.stat().st_size > 1024)
    ]
    if not force and len(missing_checkpoints) == 0:
        print("💊 FlowPacker checkpoints already exist in the model volume")
        return

    if force:
        _ = [path.unlink(missing_ok=True) for path in all_checkpoints]
        missing_checkpoints = all_checkpoints

    include_paths = ",".join(f"checkpoints/{f.name}" for f in missing_checkpoints)
    run_command(["git", "lfs", "install", "--skip-repo"], cwd=CONF.git_clone_dir)
    run_command(
        ["git", "lfs", "pull", "--include", include_paths],
        cwd=CONF.git_clone_dir,
        env={"GIT_LFS_SKIP_SMUDGE": "0"},
    )
    checkpoint_dir = CONF.git_clone_dir / "checkpoints"
    files_to_copy: dict[str | Path, str | Path] = {}
    for f in missing_checkpoints:
        source_path = checkpoint_dir / f.name
        if not source_path.exists():
            raise FileNotFoundError(
                f"FlowPacker Git LFS checkpoint was not downloaded: {source_path}"
            )
        files_to_copy[source_path] = f
    copy_files(files_to_copy)
    MODEL_VOLUME.commit()
    print("💊 FlowPacker checkpoint download complete")


##########################################
# Inference functions
##########################################
def _checkpoint_path(checkpoint_name: str) -> Path:
    """Get the path to the checkpoint file in the Git LFS directory."""
    return CONF.git_clone_dir / "checkpoints" / f"{checkpoint_name}.pth"


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

    conf_ckpt = str(_checkpoint_path("confidence")) if use_confidence else None
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
        "ckpt": str(_checkpoint_path(model_name)),
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


def validate_flowpacker_outputs(
    sample_dir: Path,
    *,
    input_stems: Sequence[str],
    n_samples: int,
    use_confidence: bool,
) -> list[dict[str, object]]:
    """Require every requested packed structure and confidence-selected copy."""
    folders = [f"run_{index + 1}" for index in range(n_samples)]
    if use_confidence:
        folders.append("best_run")
    records = []
    for stem in input_stems:
        sample_hashes = set()
        for folder in folders:
            path = sample_dir / folder / f"{stem}.pdb"
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Missing FlowPacker structure: {path}")
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError(f"FlowPacker structure is too large: {path}")
            content = path.read_bytes()
            validate_pdb_content(content, max_bytes=128 * 1024 * 1024)
            digest = sha256_bytes(content)
            if folder == "best_run" and digest not in sample_hashes:
                raise ValueError(
                    f"Confidence selection is not a sampled structure: {path}"
                )
            sample_hashes.add(digest)
            records.append({
                "path": str(path.relative_to(sample_dir)),
                "size_bytes": len(content),
                "sha256": digest,
            })
    metrics = sample_dir / "output_dict.pth"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        raise FileNotFoundError("FlowPacker did not publish output_dict.pth")
    return records


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    # Cannot mount as read-only because FlowPacker runs mkdir for some reason
    volumes=CONF.mounts(output_volume=True, model_volume=True, model_ro=False),
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
    from tempfile import TemporaryDirectory

    if model_name not in APP_INFO.supported_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {APP_INFO.supported_models}"
        )
    ckpt_dir = CONF.git_clone_dir / "checkpoints"
    if ckpt_dir.exists() and not ckpt_dir.is_symlink():
        shutil.rmtree(ckpt_dir)
        softlink_dir(CONF.model_volume_mountpoint, ckpt_dir)
    ckpt_path = _checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FlowPacker checkpoint is missing: {ckpt_path}")

    confidence_ckpt_path = _checkpoint_path("confidence")
    if use_confidence and not confidence_ckpt_path.exists():
        raise FileNotFoundError(
            f"FlowPacker confidence checkpoint is missing: {confidence_ckpt_path}"
        )

    if n_samples < 1 or num_steps < 1:
        raise ValueError("FlowPacker sample and step counts must be positive")
    names = [Path(name).name for name, _ in input_files]
    stems = [Path(name).stem for name in names]
    if not stems or len(set(stems)) != len(stems):
        raise ValueError(
            "FlowPacker requires nonempty inputs with unique structure stems"
        )
    if any(Path(name).suffix not in {".pdb", ".cif"} for name in names):
        raise ValueError("FlowPacker inputs must end in .pdb or .cif")
    run_name = sanitize_filename(run_name)
    layout = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / "workflow" / run_name
    )
    log_path = layout.logs_dir / "flowpacker.log"
    with TemporaryDirectory(prefix="flowpacker_") as scratch:
        input_dir = Path(scratch)
        scratch_name = input_dir.name
        sample_dir = CONF.git_clone_dir / "samples" / scratch_name
        config_path = (
            CONF.git_clone_dir / "config" / "inference" / f"{scratch_name}.yaml"
        )
        try:
            for file_name, content in input_files:
                dst = input_dir / Path(file_name).name
                dst.write_bytes(content)

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
                scratch_name,
                scratch_name,
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
            run_command(
                cmd,
                output_mode="stream",
                log_file=log_path,
                cwd=CONF.git_clone_dir,
            )

            structures = validate_flowpacker_outputs(
                sample_dir,
                input_stems=stems,
                n_samples=n_samples,
                use_confidence=use_confidence,
            )
            manifest = {
                "upstream_commit": CONF.repo_commit_hash,
                "inputs": [
                    {"name": name, "sha256": sha256_bytes(content)}
                    for name, content in input_files
                ],
                "structures": structures,
                "checkpoint_sha256": sha256_file(ckpt_path),
                "confidence_checkpoint_sha256": sha256_file(confidence_ckpt_path)
                if use_confidence
                else None,
                "seed": seed,
                "n_samples": n_samples,
                "num_steps": num_steps,
                "sample_coeff": sample_coeff,
                "use_confidence": use_confidence,
                "use_gt_masks": use_gt_masks,
                "inpaint": inpaint,
                "save_traj": save_traj,
            }
            (sample_dir / "validation.json").write_bytes(
                orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
            )
            shutil.copy2(log_path, sample_dir / "flowpacker.log")
            shutil.copy2(config_path, sample_dir / "biomodals_inference.yaml")
            return package_outputs(sample_dir)
        finally:
            # The log is volume-backed even when model execution/validation fails.
            try:
                CONF.output_volume.commit()
            finally:
                shutil.rmtree(sample_dir, ignore_errors=True)
                config_path.unlink(missing_ok=True)


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True, model_ro=False),
)
def run_flowpacker_workflow(
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
) -> AppRunResult:
    """Run FlowPacker and return a workflow-compatible app result."""
    safe_run_name = sanitize_filename(run_name)
    tarball_bytes = run_flowpacker.local(
        input_files=input_files,
        run_name=safe_run_name,
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
    archive_filename = f"{safe_run_name}.tar.zst"
    layout = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / "workflow" / safe_run_name
    )
    archive_path = layout.outputs_dir / archive_filename
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(tarball_bytes)
    CONF.output_volume.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name="flowpacker_outputs",
                kind=ArtifactKind.ARCHIVE,
                remote_path=str(archive_path),
                mount_root=CONF.output_volume_mountpoint,
                volume_name=CONF.output_volume_name,
                media_type=ZSTD_MEDIA_TYPE,
                metadata={"archive_format": "tar.zst", "filename": archive_filename},
            )
        ],
    )


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
    safe_run_name = sanitize_filename(run_name)

    local_out_dir = resolve_local_output_dir(out_dir)
    out_file = build_local_output_path(local_out_dir, run_name=safe_run_name)

    print(
        f"🧬 Submitting FlowPacker run '{safe_run_name}' "
        f"with {len(input_files)} input(s)"
    )
    tarball_bytes = run_flowpacker.remote(
        input_files=input_files,
        run_name=safe_run_name,
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

    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 Run complete! Results saved to {out_file}")
