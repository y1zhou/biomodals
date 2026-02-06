"""
PPIFlow on Modal - multi-task runner scaffold (binder / binder_partial / monomer / ab / ab_partial)

Examples
========
# (Optional) download models to Volume (run once)
modal run ppiflow_app.py --download-models --force-redownload

# 1) Binder (de novo)
modal run ppiflow_app.py \
  --task binder \
  --run-name test1 \
  --input-pdb ./target.pdb \
  --target-chain B \
  --binder-chain A \
  --config configs/inference_binder.yaml \
  --specified-hotspots "B119,B141,B200" \
  --samples-min-length 75 \
  --samples-max-length 76 \
  --samples-per-target 5 \
  --model-weights models/binder.ckpt

# 2) Binder partial (motif/partial diffusion)
modal run ppiflow_app.py \
  --task binder_partial \
  --run-name bp1 \
  --input-pdb ./complex.pdb \
  --target-chain B \
  --binder-chain A \
  --config configs/inference_binder_partial.yaml \
  --start-t 0.7 \
  --samples-per-target 20 \
  --model-weights models/binder.ckpt \
  --motif-contig "A19-27,A31"

# 3) Monomer unconditional (length subset)
modal run ppiflow_app.py \
  --task monomer \
  --run-name mono1 \
  --config configs/inference_unconditional.yaml \
  --model-weights models/monomer.ckpt \
  --length-subset "[50, 100]" \
  --samples-num 5

# 4) Antibody / Nanobody (framework + CDR length sampling)
modal run ppiflow_app.py \
  --task antibody \
  --run-name ab1 \
  --antigen-pdb ./antigen.pdb \
  --framework-pdb ./framework.pdb \
  --antigen-chain C \
  --heavy-chain A \
  --light-chain B \
  --specified-hotspots "C11,C14" \
  --cdr-length "CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11" \
  --samples-per-target 5 \
  --config configs/inference_nanobody.yaml \
  --model-weights models/antibody.ckpt

# 5) Antibody partial (complex + fixed positions + optional CDR)
modal run ppiflow_app.py \
  --task antibody_partial \
  --run-name abp1 \
  --complex-pdb ./ab_ag_complex.pdb \
  --antigen-chain A \
  --heavy-chain H \
  --light-chain L \
  --fixed-positions "H26,H27,H28,L50-63" \
  --cdr-position "H26-32,H45-56,H97-113" \
  --specified-hotspots "A56,A58" \
  --start-t 0.8 \
  --samples-per-target 20 \
  --retry-limit 10 \
  --config configs/test_antibody.yaml \
  --model-weights models/antibody.ckpt
"""

from __future__ import annotations

import argparse
import os
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import List

from modal import App, Image, Volume

# -------------------------
# Modal configs
# -------------------------
APP_NAME = os.environ.get("MODAL_APP", "ppiflow")
GPU = os.environ.get("GPU", "A10G")  # e.g. A10G, A100
TIMEOUT = int(os.environ.get("TIMEOUT", "7200"))

MODELS_VOL = Volume.from_name("ppiflow-models", create_if_missing=True)
RUNS_VOL = Volume.from_name("ppiflow-runs", create_if_missing=True)

MODELS_DIR = Path("/models")
RUNS_DIR = Path("/runs")
REPO_DIR = Path("/ppiflow")

# -------------------------
# Image definition
# -------------------------
runtime_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates", "build-essential", "python3-dev")
    .uv_pip_install(
        # keep minimal here; install torch + cuda explicitly below
        "pyyaml",
        "numpy",
        "pandas",
        "omegaconf",
        "hydra-core",
        "lightning",
        "torchmetrics",
        "scipy",
        "tqdm",
        "einops",
        "biopython",
        "scikit-learn",
        "dm-tree",
        "gputil",
        "tmtools",
    )
    .run_commands(
        "rm -rf /ppiflow && git clone --depth 1 https://github.com/Mingchenchen/PPIFlow.git /ppiflow",
    )
    .run_commands(
        "pip install -U pip",
        # pin torch/cuda as needed (example cu121)
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        # core deps again (safe even if already installed)
        "pip install pyyaml numpy pandas omegaconf hydra-core lightning torchmetrics dm-tree scipy tqdm einops biopython scikit-learn",
        # IMPORTANT: torch_scatter must match torch+cuda build
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html",
        "pip install mdtraj freesasa",
    )
)

app = App(APP_NAME)


# -------------------------
# Helpers
# -------------------------
def _run(cmd: str, cwd: str | None = None) -> None:
    """Run a shell command once, streaming combined output into Modal logs."""
    import subprocess

    print(f"[cmd] {cmd}")
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def _tar_dir(src_dir: Path, out_tar_gz: Path) -> None:
    """Create tar.gz from a directory."""
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


def _resolve_ckpt(model_weights_path: str) -> Path:
    """Map user path like 'models/binder.ckpt' -> '/models/binder.ckpt'."""
    mw = Path(model_weights_path)
    if str(mw).startswith(str(MODELS_DIR)):
        ckpt = mw
    else:
        ckpt = MODELS_DIR / mw.name
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    return ckpt


def _resolve_config(config_path: str) -> Path:
    """Resolve config path: prefer explicit existing path; else /ppiflow/<config_path>."""
    c = Path(config_path)
    if c.exists():
        return c
    guess = REPO_DIR / config_path
    if guess.exists():
        return guess
    raise FileNotFoundError(f"Config not found: {config_path} (also tried {guess})")


def _write_input_file(run_dir: Path, filename: str, data: bytes | None) -> Path | None:
    if data is None:
        return None
    p = run_dir / filename
    p.write_bytes(data)
    return p


def _cmd(args: List[str]) -> str:
    """Join args into a safe shell command string (we still quote values explicitly)."""
    return " ".join(args)


# -------------------------
# Step 1: download weights into Volume (run once)
# -------------------------
@app.function(timeout=TIMEOUT, image=runtime_image, volumes={str(MODELS_DIR): MODELS_VOL})
def download_models_task(force_redownload: bool = False) -> str:
    """Download model checkpoints into MODELS_DIR (persistent)."""
    # TODO: replace with real URLs (official checkpoints)
    CHECKPOINT_URLS = {
        "binder.ckpt": "https://example.com/path/to/binder.ckpt",
        "monomer.ckpt": "https://example.com/path/to/monomer.ckpt",
        "antibody.ckpt": "https://example.com/path/to/antibody.ckpt",
        "nanobody.ckpt": "https://example.com/path/to/nanobody.ckpt",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded, skipped = [], []
    for fname, url in CHECKPOINT_URLS.items():
        dst = MODELS_DIR / fname
        if dst.exists() and not force_redownload:
            skipped.append(fname)
            continue
        _run(f'curl -L "{url}" -o "{dst}"')
        downloaded.append(fname)

    MODELS_VOL.commit()
    return f"downloaded={downloaded}, skipped={skipped}, models_dir={MODELS_DIR}"


# -------------------------
# Step 2: inference (remote GPU job)
# -------------------------
@app.function(
    gpu=GPU,
    cpu=(2, 8),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={str(MODELS_DIR): MODELS_VOL, str(RUNS_DIR): RUNS_VOL},
)
def run_ppiflow_task(
    task: str,
    run_name: str,
    # binder/binder_partial
    input_pdb_bytes: bytes | None,
    target_chain: str | None,
    binder_chain: str | None,
    specified_hotspots: str | None,
    samples_min_length: int | None,
    samples_max_length: int | None,
    samples_per_target: int | None,
    motif_contig: str | None,
    interface_dist: float | None,
    start_t: float | None,
    # monomer
    length_subset: str | None,
    motif_csv_bytes: bytes | None,
    motif_names: str | None,
    samples_num: int | None,
    # antibody
    antigen_pdb_bytes: bytes | None,
    framework_pdb_bytes: bytes | None,
    antigen_chain: str | None,
    heavy_chain: str | None,
    light_chain: str | None,
    cdr_length: str | None,
    # antibody_partial
    complex_pdb_bytes: bytes | None,
    fixed_positions: str | None,
    cdr_position: str | None,
    retry_limit: int | None,
    # shared
    config_path: str,
    model_weights_path: str,
) -> bytes:
    """Run a selected PPIFlow task and return a tar.gz of the whole run directory."""
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # resolve resources
    config = _resolve_config(config_path)
    ckpt = _resolve_ckpt(model_weights_path)

    # Write provided inputs into run_dir
    input_pdb = _write_input_file(run_dir, "input.pdb", input_pdb_bytes)
    antigen_pdb = _write_input_file(run_dir, "antigen.pdb", antigen_pdb_bytes)
    framework_pdb = _write_input_file(run_dir, "framework.pdb", framework_pdb_bytes)
    motif_csv = _write_input_file(run_dir, "motif.csv", motif_csv_bytes)
    complex_pdb = _write_input_file(run_dir, "complex.pdb", complex_pdb_bytes)

    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # build command by task
    if task == "binder":
        if input_pdb is None:
            raise ValueError("binder requires input_pdb")
        if target_chain is None or binder_chain is None:
            raise ValueError("binder requires target_chain and binder_chain")

        cmd = _cmd(
            [
                "python",
                str(REPO_DIR / "sample_binder.py"),
                "--input_pdb",
                shlex.quote(str(input_pdb)),
                "--target_chain",
                shlex.quote(target_chain),
                "--binder_chain",
                shlex.quote(binder_chain),
                "--config",
                shlex.quote(str(config)),
                # allow empty hotspots (script can auto-generate if binder_chain provided)
                *(
                    ["--specified_hotspots", shlex.quote(specified_hotspots)]
                    if specified_hotspots
                    else []
                ),
                "--samples_min_length",
                str(samples_min_length if samples_min_length is not None else 50),
                "--samples_max_length",
                str(samples_max_length if samples_max_length is not None else 100),
                "--samples_per_target",
                str(samples_per_target if samples_per_target is not None else 100),
                "--model_weights",
                shlex.quote(str(ckpt)),
                "--output_dir",
                shlex.quote(str(out_dir)),
                "--name",
                shlex.quote(run_name),
            ]
        )

    elif task == "binder_partial":
        if input_pdb is None:
            raise ValueError("binder_partial requires input_pdb")
        if target_chain is None or binder_chain is None:
            raise ValueError("binder_partial requires target_chain and binder_chain")

        cmd_args = [
            "python",
            str(REPO_DIR / "sample_binder_partial.py"),
            "--input_pdb",
            shlex.quote(str(input_pdb)),
            "--target_chain",
            shlex.quote(target_chain),
            "--binder_chain",
            shlex.quote(binder_chain),
            "--config",
            shlex.quote(str(config)),
            "--samples_per_target",
            str(samples_per_target if samples_per_target is not None else 100),
            "--model_weights",
            shlex.quote(str(ckpt)),
            "--output_dir",
            shlex.quote(str(out_dir)),
            "--name",
            shlex.quote(run_name),
        ]
        if specified_hotspots:
            cmd_args += ["--specified_hotspots", shlex.quote(specified_hotspots)]
        if motif_contig:
            cmd_args += ["--motif_contig", shlex.quote(motif_contig)]
        if interface_dist is not None:
            cmd_args += ["--interface_dist", str(interface_dist)]
        if start_t is not None:
            cmd_args += ["--start_t", str(start_t)]
        cmd = _cmd(cmd_args)

    elif task == "monomer":
        # monomer has no PDB unless motif scaffolding CSV is provided
        cmd_args = [
            "python",
            str(REPO_DIR / "sample_monomer.py"),
            "--config",
            shlex.quote(str(config)),
            "--model_weights",
            shlex.quote(str(ckpt)),
            "--output_dir",
            shlex.quote(str(out_dir)),
        ]
        # choose mode
        if length_subset:
            cmd_args += ["--length_subset", shlex.quote(length_subset)]
        elif motif_csv is not None:
            cmd_args += ["--motif_csv", shlex.quote(str(motif_csv))]
            if motif_names:
                cmd_args += ["--motif_names", shlex.quote(motif_names)]
        else:
            raise ValueError("monomer requires either --length_subset or motif_csv")
        cmd_args += ["--samples_num", str(samples_num if samples_num is not None else 100)]
        cmd = _cmd(cmd_args)

    elif task == "antibody":
        if antigen_pdb is None or framework_pdb is None:
            raise ValueError("antibody requires antigen_pdb and framework_pdb")
        if antigen_chain is None or heavy_chain is None:
            raise ValueError("antibody requires antigen_chain and heavy_chain")
        if not specified_hotspots:
            raise ValueError("antibody requires specified_hotspots")

        cmd_args = [
            "python",
            str(REPO_DIR / "sample_antibody_nanobody.py"),
            "--antigen_pdb",
            shlex.quote(str(antigen_pdb)),
            "--framework_pdb",
            shlex.quote(str(framework_pdb)),
            "--antigen_chain",
            shlex.quote(antigen_chain),
            "--heavy_chain",
            shlex.quote(heavy_chain),
            "--specified_hotspots",
            shlex.quote(specified_hotspots),
            "--samples_per_target",
            str(samples_per_target if samples_per_target is not None else 100),
            "--config",
            shlex.quote(str(config)),
            "--model_weights",
            shlex.quote(str(ckpt)),
            "--output_dir",
            shlex.quote(str(out_dir)),
            "--name",
            shlex.quote(run_name),
        ]
        if light_chain:
            cmd_args += ["--light_chain", shlex.quote(light_chain)]
        if cdr_length:
            cmd_args += ["--cdr_length", shlex.quote(cdr_length)]
        cmd = _cmd(cmd_args)

    elif task == "antibody_partial":
        if complex_pdb is None:
            raise ValueError("antibody_partial requires complex_pdb")
        if antigen_chain is None or heavy_chain is None:
            raise ValueError("antibody_partial requires antigen_chain and heavy_chain")
        if not fixed_positions:
            raise ValueError("antibody_partial requires fixed_positions")
        if start_t is None:
            raise ValueError("antibody_partial requires start_t")
        if samples_per_target is None:
            raise ValueError("antibody_partial requires samples_per_target")

        cmd_args = [
            "python",
            str(REPO_DIR / "sample_antibody_nanobody_partial.py"),
            "--complex_pdb",
            shlex.quote(str(complex_pdb)),
            "--antigen_chain",
            shlex.quote(antigen_chain),
            "--heavy_chain",
            shlex.quote(heavy_chain),
            "--fixed_positions",
            shlex.quote(fixed_positions),
            "--start_t",
            str(start_t),
            "--samples_per_target",
            str(samples_per_target),
            "--output_dir",
            shlex.quote(str(out_dir)),
            "--config",
            shlex.quote(str(config)),
            "--model_weights",
            shlex.quote(str(ckpt)),
            "--name",
            shlex.quote(run_name),
        ]
        if light_chain:
            cmd_args += ["--light_chain", shlex.quote(light_chain)]
        if cdr_position:
            cmd_args += ["--cdr_position", shlex.quote(cdr_position)]
        if specified_hotspots:
            cmd_args += ["--specified_hotspots", shlex.quote(specified_hotspots)]
        if retry_limit is not None:
            cmd_args += ["--retry_Limit", str(retry_limit)]
        cmd = _cmd(cmd_args)

    else:
        raise ValueError(f"Unknown task: {task}")

    _run(cmd, cwd=str(REPO_DIR))

    RUNS_VOL.commit()

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"{run_name}.tar.gz"
        _tar_dir(run_dir, tar_path)
        return tar_path.read_bytes()


# -------------------------
# Local entrypoint
# -------------------------
@app.local_entrypoint()
def main(
    # Core
    task: str = "binder",
    run_name: str = "test1",
    config: str = "configs/inference_binder.yaml",
    model_weights: str = "models/binder.ckpt",
    # Download
    download_models: bool = False,
    force_redownload: bool = False,
    # Local output
    out_dir: str = "./ppiflow_outputs",
    # ---------- binder / binder_partial ----------
    input_pdb: str | None = None,
    target_chain: str | None = None,
    binder_chain: str | None = None,
    specified_hotspots: str = "",
    samples_min_length: int = 75,
    samples_max_length: int = 76,
    samples_per_target: int = 5,
    motif_contig: str = "",
    interface_dist: float | None = None,
    start_t: float | None = None,
    # ---------- monomer ----------
    length_subset: str = "",
    motif_csv: str = "",
    motif_names: str = "",
    samples_num: int = 100,
    # ---------- antibody ----------
    antigen_pdb: str = "",
    framework_pdb: str = "",
    antigen_chain: str = "",
    heavy_chain: str = "",
    light_chain: str = "",
    cdr_length: str = "",
    # ---------- antibody_partial ----------
    complex_pdb: str = "",
    fixed_positions: str = "",
    cdr_position: str = "",
    retry_limit: int | None = None,
) -> None:
    if download_models:
        msg = download_models_task.remote(force_redownload=force_redownload)
        print(msg)
        return

    # read optional file bytes (only when user provides a path)
    def read_bytes(p: str) -> bytes | None:
        if not p:
            return None
        return Path(p).read_bytes()

    tar_bytes = run_ppiflow_task.remote(
        task=task,
        run_name=run_name,
        # binder/binder_partial
        input_pdb_bytes=read_bytes(input_pdb) if input_pdb else None,
        target_chain=target_chain,
        binder_chain=binder_chain,
        specified_hotspots=specified_hotspots,
        samples_min_length=samples_min_length,
        samples_max_length=samples_max_length,
        samples_per_target=samples_per_target,
        motif_contig=motif_contig,
        interface_dist=interface_dist,
        start_t=start_t,
        # monomer
        length_subset=length_subset,
        motif_csv_bytes=read_bytes(motif_csv) if motif_csv else None,
        motif_names=motif_names,
        samples_num=samples_num,
        # antibody
        antigen_pdb_bytes=read_bytes(antigen_pdb) if antigen_pdb else None,
        framework_pdb_bytes=read_bytes(framework_pdb) if framework_pdb else None,
        antigen_chain=antigen_chain or None,
        heavy_chain=heavy_chain or None,
        light_chain=light_chain or None,
        cdr_length=cdr_length or None,
        # antibody_partial
        complex_pdb_bytes=read_bytes(complex_pdb) if complex_pdb else None,
        fixed_positions=fixed_positions or None,
        cdr_position=cdr_position or None,
        retry_limit=retry_limit,
        # shared
        config_path=config,
        model_weights_path=model_weights,
    )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir_p / f"{run_name}.tar.gz"
    out_tar.write_bytes(tar_bytes)
    print(f"[ok] saved: {out_tar}")
