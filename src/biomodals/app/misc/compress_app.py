"""Compress old BoltzGen outputs to save Volume space."""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.design import boltzgen_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import package_outputs, run_command, warmup_directory

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Compressor",
    package_name="compress_app",
    version="1.0.0",
    python_version="3.13",
)

##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .add_local_python_source("biomodals.app", copy=True)
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
app = include_dependency_apps(app, ["boltzgen"])


@dataclass(frozen=True)
class BoltzgenConf:
    """Hydrated BoltzGen configuration."""

    volume_mounts = boltzgen_app.CONF.mounts(output_volume=True)
    output_volume = boltzgen_app.CONF.output_volume
    output_volume_mountpoint = boltzgen_app.CONF.output_volume_mountpoint


BG_CONF = BoltzgenConf()


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=MAX_TIMEOUT,
    volumes=BG_CONF.volume_mounts,
    nonpreemptible=True,
    retries=2,
)
def compress_one_run(dir_name: str) -> str:
    """Compress a previous BoltzGen run and return the .tar.zst file name."""
    import subprocess

    dir_path = Path(BG_CONF.output_volume_mountpoint) / dir_name
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    tar_path = dir_path.with_suffix(".tar.zst")

    # Remove old failed tarballs if they exist
    if tar_path.exists():
        try:
            run_command(["zstd", "-t", str(tar_path)])
            return tar_path.name
        except subprocess.CalledProcessError:
            tar_path.unlink()

    if not tar_path.exists():
        BG_CONF.output_volume.reload()
        warmup_directory(dir_path)
        tar_content = package_outputs(dir_path)
        tar_path.write_bytes(tar_content)
        BG_CONF.output_volume.commit()
        print(f"Compressed BoltzGen run to {tar_path.name}")
    return tar_path.name


@app.local_entrypoint()
def compress_all_runs(*args) -> None:
    """Compress all BoltzGen runs in the output volume."""
    tasks: list[modal.FunctionCall] = []
    for f in BG_CONF.output_volume.iterdir("/", recursive=False):
        if f.type == modal.volume.FileEntryType.DIRECTORY:
            print(f"Compressing BoltzGen run {f.path}")
            tasks.append(compress_one_run.spawn(f.path))

    _ = modal.FunctionCall.gather(*tasks)


if __name__ == "__main__":
    with modal.enable_output():
        with app.run(detach=True):
            compress_all_runs()
