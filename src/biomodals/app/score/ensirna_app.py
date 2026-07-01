"""ENsiRNA source repo: <https://github.com/tanwenchong/ENsiRNA>.

ENsiRNA designs siRNA candidates from an mRNA FASTA file. This wrapper uses the
upstream Linux runtime requirements, builds on the Biomodals Rosetta base image,
and runs the documented `design.sh` pipeline.

## Additional notes

The upstream Linux instructions require Rosetta for PDB generation. The runtime
uses the same public Rosetta base image as `rosetta_app.py`; commercial use may
require a separate Rosetta license. Model weights and checkpoints are stored in
the standard Biomodals model volume.

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
from dataclasses import dataclass
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
from biomodals.helper.web import download_files

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


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Container for ENsiRNA-specific paths, dependencies, and patches."""

    conda_env_name: str = "base"
    mamba_root: str = "/root/micromamba"
    ensirna_dir: Path = CONF.git_clone_dir / "ENsiRNA"
    rosetta_compat_root: Path = Path(
        "/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371"
    )
    rnafm_pretrained_url: str = (
        "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
    )
    rnafm_cache_path: Path = (
        Path(CONF.default_env["TORCH_HOME"]) / "hub/checkpoints/RNA-FM_pretrained.pth"
    )
    checkpoint_filenames: tuple[str, ...] = tuple(
        f"checkpoint_{idx}.ckpt" for idx in range(1, 6)
    )
    checkpoint_dir: Path = Path(CONF.model_volume_mountpoint) / "pkl"
    conda_packages: tuple[str, ...] = (
        f"python={CONF.python_version}",
        "pip",
        "viennarna=2.6.4-0",
    )
    conda_channels: tuple[str, ...] = ("conda-forge", "bioconda")
    pip_packages: tuple[str, ...] = (
        "biopython",
        "numpy",
        "pandas",
        "scipy",
        "tensorboard",
        "tqdm",
        "openpyxl",
        "rdkit",
        "scikit-learn",
        "xgboost",
    )
    torch_packages: tuple[str, ...] = ("torch", "torchvision", "torchaudio")
    torch_index_url: str = "https://download.pytorch.org/whl/cu118"
    extra_pip_packages: tuple[str, ...] = ("torch-geometric", "rna-fm")

    @property
    def mamba_bin_path(self) -> str:
        """Return PATH with micromamba's base environment first."""
        return f"{self.mamba_root}/bin:/root/.local/bin:$PATH"

    @property
    def rosetta_rna_denovo(self) -> Path:
        """Return the ENsiRNA-expected Rosetta rna_denovo path."""
        return self.rosetta_compat_root / (
            "main/source/bin/rna_denovo.static.linuxgccrelease"
        )

    @property
    def rosetta_extract(self) -> Path:
        """Return the ENsiRNA-expected Rosetta extract script path."""
        return self.rosetta_compat_root / (
            "main/tools/rna_tools/silent_util/extract_lowscore_decoys.py"
        )

    @property
    def model_downloads(self) -> dict[str, Path]:
        """Return ENsiRNA model URLs mapped to model-volume paths."""
        return {
            self.rnafm_pretrained_url: self.rnafm_cache_path,
            **{
                (
                    f"{CONF.repo_url}/raw/{CONF.repo_commit_hash}/ENsiRNA/pkl/"
                    f"{filename}"
                ): self.checkpoint_dir / filename
                for filename in self.checkpoint_filenames
            },
        }

    @property
    def rosetta_extract_shim(self) -> str:
        """Return the Rosetta extract shim written into the image."""
        return """#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

silent_file = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
if silent_file is None:
    raise SystemExit("usage: extract_lowscore_decoys.py <silent-file> ...")

before = {p.name for p in Path.cwd().glob("*.pdb")}
subprocess.run(["extract_pdbs", "-in:file:silent", silent_file], check=True)
created = [p for p in Path.cwd().glob("*.pdb") if p.name not in before]
if not created:
    raise SystemExit("extract_pdbs did not produce a PDB")

target = Path(f"{silent_file}.1.pdb")
if created[0] != target:
    created[0].replace(target)
"""

    @property
    def rosetta_extract_writer(self) -> str:
        """Return a Python one-liner that writes the Rosetta extract shim."""
        return (
            "from pathlib import Path; "
            f"Path({str(self.rosetta_extract)!r}).write_text("
            f"{self.rosetta_extract_shim!r})"
        )

    @property
    def get_pdb_runtime_patch(self) -> str:
        """Return the source patch for ENsiRNA's PDB generation helper."""
        return f"""from pathlib import Path

path = Path({str(self.ensirna_dir / "data/get_pdb.py")!r})
text = path.read_text()
old = '''        if len(sec_pos) != 61+len(seq2)+len(seq1)+1+1:
            print('!=',data['siRNA'],len(sec_pos),len(seq2))
            return None
        return sec_pos,chain
'''
new = '''        expected_len = 61 + len(seq2) + len(seq1) + 1 + 1
        if len(sec_pos) != expected_len:
            print('!=', data['siRNA'], len(sec_pos), len(seq2))
            while len(sec_pos) < expected_len:
                sec_pos.append(sec_pos[-1] - 1 if sec_pos else 0)
                chain.append(3)
            if len(sec_pos) > expected_len:
                sec_pos = sec_pos[:expected_len]
                chain = chain[:expected_len]
        return sec_pos, chain
'''
secstruct_old = '''        secondary_seq = secondary_seq1 + ' ' + secondary_seq2
'''
secstruct_new = '''        def _fit_secstruct(secstruct, size):
            if len(secstruct) < size:
                return secstruct + '.' * (size - len(secstruct))
            while len(secstruct) > size and secstruct.startswith('.'):
                secstruct = secstruct[1:]
            while len(secstruct) > size and secstruct.endswith('.'):
                secstruct = secstruct[:-1]
            return secstruct[:size]

        secondary_seq1 = _fit_secstruct(secondary_seq1, len(seq1))
        secondary_seq2 = _fit_secstruct(secondary_seq2, len(seq2))
        secondary_seq = secondary_seq1 + ' ' + secondary_seq2
'''
rosetta_cmd_old = '''        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna'])
'''
rosetta_cmd_new = '''        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna','-out:file:silent','default.out'])
'''
if old not in text:
    raise SystemExit("expected ENsiRNA get_pdb length check not found")
if secstruct_old not in text:
    raise SystemExit("expected ENsiRNA secondary structure block not found")
if rosetta_cmd_old not in text:
    raise SystemExit("expected ENsiRNA Rosetta command not found")
text = text.replace(old, new)
text = text.replace(secstruct_old, secstruct_new)
text = text.replace(rosetta_cmd_old, rosetta_cmd_new)
path.write_text(text)
"""

    @property
    def get_pdb_runtime_patch_runner(self) -> str:
        """Return a Python one-liner that applies the PDB helper patch."""
        return f"exec({self.get_pdb_runtime_patch!r})"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    modal.Image
    .from_registry("rosettacommons/rosetta:serial-420", add_python=CONF.python_version)
    .apt_install("git", "curl", "ca-certificates", "build-essential", "zstd")
    .env(
        CONF.default_env
        | {"MAMBA_ROOT_PREFIX": APP_INFO.mamba_root, "PATH": APP_INFO.mamba_bin_path}
    )
    .run_commands("curl -L micro.mamba.pm/install.sh | bash")
    .micromamba_install(
        *APP_INFO.conda_packages, channels=list(APP_INFO.conda_channels)
    )
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "find . -path '*/pkl/*.ckpt' -delete",
        ))
    )
    .run_commands(
        " && ".join((
            f"mkdir -p {APP_INFO.rosetta_rna_denovo.parent} "
            f"{APP_INFO.rosetta_extract.parent}",
            f"ln -sf /usr/local/bin/rna_denovo {APP_INFO.rosetta_rna_denovo}",
            f"python -c {shlex.quote(APP_INFO.rosetta_extract_writer)}",
            f"chmod +x {APP_INFO.rosetta_extract}",
        ))
    )
    .run_commands(f"python -c {shlex.quote(APP_INFO.get_pdb_runtime_patch_runner)}")
    .uv_pip_install(*APP_INFO.pip_packages)
    .uv_pip_install(*APP_INFO.torch_packages, index_url=APP_INFO.torch_index_url)
    .uv_pip_install(*APP_INFO.extra_pip_packages)
    .pipe(patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3"])
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=CONF.timeout
)
def download_ensirna_models(force: bool = False) -> None:
    """Download ENsiRNA model files into the standard model volume."""
    download_files(
        APP_INFO.model_downloads,
        force=force,
        num_retries=3,
        progress_bar_desc="ENsiRNA model downloads",
    )
    MODEL_VOLUME.commit()


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True),
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

        checkpoint_dir = APP_INFO.ensirna_dir / "pkl"
        checkpoint_dir.mkdir(exist_ok=True)
        for filename in APP_INFO.checkpoint_filenames:
            checkpoint = APP_INFO.checkpoint_dir / filename
            if not checkpoint.exists():
                raise FileNotFoundError(f"ENsiRNA checkpoint not found: {checkpoint}")
            link = checkpoint_dir / filename
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(checkpoint)

        mrna_fasta = input_dir / "mrna.fasta"
        mrna_fasta.write_bytes(mrna_fasta_bytes)
        run_command(
            [
                "micromamba",
                "run",
                "-n",
                APP_INFO.conda_env_name,
                "bash",
                "design.sh",
                str(mrna_fasta),
                str(output_dir),
            ],
            cwd=APP_INFO.ensirna_dir,
        )
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
    download_ensirna_models.remote(force=False)
    tarball_bytes = run_ensirna.remote(
        mrna_fasta_bytes=input_path.read_bytes(),
        run_name=run_name,
    )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 ENsiRNA run complete! Results saved to {out_file}")
