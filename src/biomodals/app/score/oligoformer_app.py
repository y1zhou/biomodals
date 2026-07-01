"""OligoFormer source repo: <https://github.com/lulab/OligoFormer>.

OligoFormer predicts siRNA efficacy from an mRNA FASTA file. This wrapper builds
the runtime from the upstream README and supports OligoFormer's off-target and
toxicity options for standalone runs.

## Off-target prediction

When `--off-target` is set, provide both `--utr-file` and `--orf-file`, or set
`--all-human` to use the upstream bundled human references.

## Outputs

Results are saved locally as `<run-name>_oligoformer.tar.zst`.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
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
    name="OligoFormer",
    repo_url="https://github.com/lulab/OligoFormer",
    repo_commit_hash="e2f53ad63387bbe166bf123949151e2bc9bf6ec3",
    package_name="oligoformer",
    version="1.0",
    python_version="3.10",
    cuda_version="cu118",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "7200")),
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    """OligoFormer runtime paths and pinned dependencies."""

    requirements: tuple[str, ...] = (
        "bio==1.6.2",
        "matplotlib==3.7.5",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "prefetch-generator==1.0.3",
        "ptflops==0.7.3",
        "pytorch-ignite==0.5.0.post2",
        "scikit-learn==1.3.1",
        "scipy==1.10.1",
        "torch==2.2.1",
        "torchvision==0.17.1",
        "tqdm==4.66.1",
        "yacs==0.1.8",
    )
    rnafm_archive_url: str = (
        "https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1"
    )
    repo_rnafm_dir: Path = CONF.git_clone_dir / "RNA-FM"
    model_rnafm_dir: Path = Path(CONF.model_volume_mountpoint) / "RNA-FM"

    @property
    def model_rnafm_redevelop_dir(self) -> Path:
        """Return the RNA-FM redevelop directory expected by OligoFormer."""
        return self.model_rnafm_dir / "redevelop"

    @property
    def repo_rnafm_redevelop_dir(self) -> Path:
        """Return the runtime RNA-FM redevelop directory inside OligoFormer."""
        return self.repo_rnafm_dir / "redevelop"


APP_INFO = AppInfo()


##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "ca-certificates",
        "unzip",
        "perl",
        "libstatistics-lite-perl",
        "libbio-perl-perl",
        "zstd",
    )
    .env(CONF.default_env)
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "cd off-target/pita",
            "make install",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(*APP_INFO.requirements)
    .pipe(
        patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3", "modal"]
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=CONF.timeout
)
def download_oligoformer_models(force: bool = False) -> None:
    """Download RNA-FM weights into the standard model volume."""
    import shutil

    if APP_INFO.model_rnafm_redevelop_dir.is_dir() and not force:
        print("🧬 OligoFormer RNA-FM weights already available")
        return

    if APP_INFO.model_rnafm_dir.exists():
        shutil.rmtree(APP_INFO.model_rnafm_dir)

    with TemporaryDirectory(prefix="oligoformer_models_") as tmpdir:
        archive_path = Path(tmpdir) / "RNA-FM.tar.gz"
        download_files(
            {APP_INFO.rnafm_archive_url: archive_path},
            force=True,
            num_retries=3,
            progress_bar_desc="OligoFormer model downloads",
        )
        APP_INFO.model_rnafm_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command([
            "tar",
            "-xzf",
            str(archive_path),
            "-C",
            str(APP_INFO.model_rnafm_dir.parent),
        ])

    if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        raise FileNotFoundError(
            "OligoFormer RNA-FM weights were not extracted to "
            f"{APP_INFO.model_rnafm_redevelop_dir}"
        )

    MODEL_VOLUME.commit()
    print("🧬 OligoFormer RNA-FM weights downloaded and committed")


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
def run_oligoformer(
    mrna_fasta_bytes: bytes,
    run_name: str,
    sirna_fasta_bytes: bytes | None = None,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    utr_bytes: bytes | None = None,
    orf_bytes: bytes | None = None,
    top_n: int = -1,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
) -> bytes:
    """Run OligoFormer inference and return packaged outputs."""
    if top_n < -1:
        raise ValueError("top_n must be -1 or a non-negative integer")
    if off_target and not all_human and (utr_bytes is None or orf_bytes is None):
        raise ValueError(
            "OligoFormer off-target mode requires both UTR and ORF references "
            "unless all_human is enabled."
        )
    if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        raise FileNotFoundError(
            "OligoFormer RNA-FM weights are missing. Run "
            "download_oligoformer_models first."
        )

    import shutil

    if APP_INFO.repo_rnafm_dir.is_symlink():
        APP_INFO.repo_rnafm_dir.unlink()
    elif (
        APP_INFO.repo_rnafm_dir.exists()
        and not APP_INFO.repo_rnafm_redevelop_dir.is_dir()
    ):
        shutil.rmtree(APP_INFO.repo_rnafm_dir)
    if not APP_INFO.repo_rnafm_redevelop_dir.is_dir():
        shutil.copytree(APP_INFO.model_rnafm_dir, APP_INFO.repo_rnafm_dir)

    safe_run_name = sanitize_filename(run_name)
    with TemporaryDirectory(prefix=f"oligoformer_{safe_run_name}_") as tmpdir:
        run_root = Path(tmpdir) / safe_run_name
        input_dir = run_root / "inputs"
        output_dir = run_root / "outputs"
        input_dir.mkdir(parents=True)
        output_dir.mkdir()

        mrna_fasta = input_dir / "mrna.fa"
        mrna_fasta.write_bytes(mrna_fasta_bytes)
        output_prefix = f"{output_dir}/"
        cmd = [
            "python",
            "scripts/main.py",
            "-i",
            "1",
            "-i1",
            str(mrna_fasta),
            "--output_dir",
            output_prefix,
        ]

        if sirna_fasta_bytes is not None:
            sirna_fasta = input_dir / "sirna.fa"
            sirna_fasta.write_bytes(sirna_fasta_bytes)
            cmd.extend(["-i2", str(sirna_fasta)])
        if top_n != -1:
            cmd.extend(["-top", str(top_n)])
        if not functionality_filter:
            cmd.append("--no_func")
        if off_target:
            cmd.append("-off")
            if all_human:
                cmd.append("-a")
            else:
                utr_file = input_dir / "utr.fa"
                orf_file = input_dir / "orf.fa"
                utr_file.write_bytes(utr_bytes or b"")
                orf_file.write_bytes(orf_bytes or b"")
                cmd.extend(["--utr", str(utr_file), "--orf", str(orf_file)])
            cmd.extend([
                "--pita_threshold",
                str(pita_threshold),
                "--targetscan_threshold",
                str(targetscan_threshold),
            ])
        if toxicity:
            cmd.extend(["-tox", "--toxicity_threshold", str(toxicity_threshold)])

        run_command(cmd, cwd=CONF.git_clone_dir)
        return package_outputs(output_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_oligoformer_task(
    mrna_fasta: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    sirna_fasta: str | None = None,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    utr_file: str | None = None,
    orf_file: str | None = None,
    top_n: int = -1,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
) -> None:
    """Run OligoFormer siRNA efficacy prediction.

    Args:
        mrna_fasta: Local mRNA FASTA file to scan for siRNA candidates.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the mRNA FASTA
            filename stem.
        sirna_fasta: Optional FASTA file of specific siRNAs to score instead of
            traversing the mRNA with OligoFormer's default 19 nt window.
        off_target: Enable OligoFormer off-target prediction.
        toxicity: Enable OligoFormer toxicity prediction.
        all_human: Use upstream bundled human ORF and UTR references for
            off-target prediction.
        utr_file: Local UTR reference file for off-target prediction.
        orf_file: Local ORF reference file for off-target prediction.
        top_n: Number of top siRNAs to use for off-target prediction, or -1 for
            all candidates.
        functionality_filter: Keep upstream functionality filtering enabled.
        pita_threshold: PITA threshold used by off-target prediction.
        targetscan_threshold: TargetScan threshold used by off-target prediction.
        toxicity_threshold: Toxicity filter threshold.
    """
    input_path = Path(mrna_fasta).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"mRNA FASTA not found: {input_path}")
    run_name = run_name or input_path.stem
    out_file = build_local_output_path(
        resolve_local_output_dir(out_dir),
        run_name=run_name,
        suffix="oligoformer",
    )

    if off_target and not all_human and (utr_file is None or orf_file is None):
        raise ValueError(
            "Set --utr-file and --orf-file for off-target prediction, or pass "
            "--all-human."
        )

    sirna_fasta_bytes = None
    if sirna_fasta is not None:
        sirna_path = Path(sirna_fasta).expanduser().resolve()
        if not sirna_path.exists():
            raise FileNotFoundError(f"siRNA FASTA not found: {sirna_path}")
        sirna_fasta_bytes = sirna_path.read_bytes()

    utr_bytes = None
    if utr_file is not None:
        utr_path = Path(utr_file).expanduser().resolve()
        if not utr_path.exists():
            raise FileNotFoundError(f"UTR reference not found: {utr_path}")
        utr_bytes = utr_path.read_bytes()

    orf_bytes = None
    if orf_file is not None:
        orf_path = Path(orf_file).expanduser().resolve()
        if not orf_path.exists():
            raise FileNotFoundError(f"ORF reference not found: {orf_path}")
        orf_bytes = orf_path.read_bytes()

    print(f"🧬 Submitting OligoFormer run '{run_name}'")
    download_oligoformer_models.remote(force=False)
    tarball_bytes = run_oligoformer.remote(
        mrna_fasta_bytes=input_path.read_bytes(),
        run_name=run_name,
        sirna_fasta_bytes=sirna_fasta_bytes,
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        utr_bytes=utr_bytes,
        orf_bytes=orf_bytes,
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
    )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 OligoFormer run complete! Results saved to {out_file}")
