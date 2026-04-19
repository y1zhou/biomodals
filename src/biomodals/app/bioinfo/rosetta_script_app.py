"""Rosetta source repo: <https://github.com/RosettaCommons/rosetta>.

Use for commercial purposes requires purchase of a separate license.
Please see <https://els2.comotion.uw.edu/product/rosetta> or email
license@uw.edu for more information.

See <https://docs.rosettacommons.org/docs/latest/Home> for documentation.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.helper import patch_image_for_helper

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="rosetta_script",
    repo_url="https://github.com/RosettaCommons/rosetta",
    package_name="rosetta",
    version="2026.15+release.b166af21f0",
    python_version="3.12",
    timeout=int(os.environ.get("TIMEOUT", "14400")),
)
OUT_VOLUME = CONF.get_out_volume()

##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    modal.Image.micromamba(python_version=CONF.python_version)
    .env(CONF.default_env)
    .micromamba_install(
        f"{CONF.package_name}={CONF.version}",
        channels=["https://conda.rosettacommons.org/", "conda-forge"],
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(1.125, 30.125),  # Each pod can run 1-30 jobs
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=CONF.timeout,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def run_rosetta_script():
    """Run Rosetta scripts."""
    pass


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_rosetta_script_task(
    input_pdb: str | None = None,
    input_rosetta_script: str | None = None,
    input_flags_file: str | None = None,
    input_csv: str | None = None,
    out_dir: str | None = None,
    num_pods: int = 1,
    num_cpu_per_pod: int = 1,
) -> None:
    """Run Rosetta scripts on Modal and fetch results to `out_dir`.

    Args:
        input_pdb: Path to input PDB file. Needs to be provided together with
            `input_rosetta_script`.
        input_rosetta_script: Path to input Rosetta script file. Needs to be
            provided together with `input_pdb`.
        input_flags_file: Path to input Rosetta flag file, if additional flags
            are needed. See <https://docs.rosettacommons.org/docs/latest/development_documentation/code_structure/namespaces/namespace-utility-options#flagsfile>.
        input_csv: Path to an input CSV file, if `input_pdb`, `input_rosetta_script`,
            and `input_flags_file` are not provided. The CSV file should have columns
            "pdb", "rosetta_script", and optionally "flags_file" that specify the
            *local* file paths to the respective files for each run. This allows
            batch processing of multiple Rosetta script runs in one command.
        out_dir: Optional output directory. If not provided, results will only
            be saved to the Modal output volume and not downloaded locally. If
            provided, results will be saved to `out_dir` with the same filename as
            the input PDB file but with a `.tar.zst` extension.
        num_pods: Number of parallel pods to run. Only applicable when `input_csv`
            is provided. Default is 1.
        num_cpu_per_pod: Number of CPUs to allocate per Modal pod. Default is 1.
            Note that a maximum of 30 CPUs can be allocated per pod.
    """
    import polars as pl

    # Validate and read input
    if input_csv is not None:
        input_csv_path = Path(input_csv).expanduser().resolve()
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

        run_name = input_csv_path.stem
        df = pl.read_csv(input_csv_path)
        cols = ["pdb", "rosetta_script"]
        if "flags_file" in df.columns:
            df = df.select(*cols, "flags_file")
        else:
            df = df.select(*cols).with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("flags_file")
            )
    else:
        if input_pdb is None or input_rosetta_script is None:
            raise ValueError(
                "Both input_pdb and input_rosetta_script need to be provided together"
            )
        run_name = Path(input_pdb).stem
        df = pl.DataFrame(
            {"pdb": [input_pdb], "rosetta_script": [input_rosetta_script]}
        ).with_columns(pl.lit(input_flags_file, dtype=pl.Utf8).alias("flags_file"))

    print(f"🧬 Running {CONF.name} inference pipeline...")
    tarball_bytes = run_rosetta_script.remote()

    # Save results locally
    if out_dir is not None:
        local_out_dir = Path(out_dir).expanduser().resolve()
        local_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = local_out_dir / f"{run_name}.tar.zst"
        out_file.write_bytes(tarball_bytes)
        print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
