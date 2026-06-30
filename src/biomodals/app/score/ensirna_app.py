"""ENsiRNA source repo: <https://github.com/tanwenchong/ENsiRNA>.

ENsiRNA designs siRNA candidates from an mRNA FASTA file. This wrapper uses the
upstream Docker image and runs the documented `design.sh` pipeline.

## Outputs

Results are saved locally as `<run-name>_ensirna.tar.zst`. The archive contains
the upstream ENsiRNA output directory, including `mrna_result.xlsx` when the run
completes successfully.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import package_outputs, run_command, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="ENsiRNA",
    repo_url="https://github.com/tanwenchong/ENsiRNA",
    repo_commit_hash="028824341635903f3c661f5d1cc737de106493d5",
    package_name="ensirna",
    version="2",
    python_version="3.10",
    cuda_version="cu118",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "7200")),
)
ENSIRNA_DIR = "/app/ENsiRNA-main/ENsiRNA"


##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .from_registry(
        "tanwenchong/ensirna:v2",
        add_python=CONF.python_version,
    )
    .entrypoint([])
    .apt_install("zstd")
    .env(CONF.default_env)
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_ro=False),
)
def run_ensirna(mrna_fasta_bytes: bytes, run_name: str) -> bytes:
    """Run ENsiRNA design and return packaged outputs."""
    safe_run_name = sanitize_filename(run_name)
    with TemporaryDirectory(prefix=f"ensirna_{safe_run_name}_") as tmpdir:
        run_root = Path(tmpdir) / safe_run_name
        input_dir = run_root / "inputs"
        output_dir = run_root / "outputs"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()

        mrna_fasta = input_dir / "mrna.fasta"
        mrna_fasta.write_bytes(mrna_fasta_bytes)
        shell_cmd = " ".join((
            "set -euo pipefail;",
            "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true;",
            "conda activate my_environment_name 2>/dev/null ||",
            "source activate my_environment_name;",
            f"cd {shlex.quote(ENSIRNA_DIR)};",
            "bash design.sh",
            shlex.quote(str(mrna_fasta)),
            shlex.quote(str(output_dir)),
        ))
        run_command(["bash", "-lc", shell_cmd])
        MODEL_VOLUME.commit()
        return package_outputs(output_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_ensirna_task(
    mrna_fasta: str,
    out_dir: str | None = None,
    run_name: str | None = None,
) -> None:
    """Run ENsiRNA siRNA candidate design.

    Args:
        mrna_fasta: Local mRNA FASTA file to design siRNA candidates for.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the mRNA FASTA
            filename stem.
    """
    input_path = Path(mrna_fasta).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"mRNA FASTA not found: {input_path}")
    run_name = run_name or input_path.stem
    out_file = build_local_output_path(
        resolve_local_output_dir(out_dir),
        run_name=run_name,
        suffix="ensirna",
    )

    print(f"🧬 Submitting ENsiRNA run '{run_name}'")
    tarball_bytes = run_ensirna.remote(
        mrna_fasta_bytes=input_path.read_bytes(),
        run_name=run_name,
    )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 ENsiRNA run complete! Results saved to {out_file}")
