"""PPIFlow source repo: <https://github.com/zhuqianhui2-hash/PPIFlow>.

This file (`ppiflow_app.py`) is a **single Modal entrypoint** that routes to multiple upstream
PPIFlow sampling scripts (binder / antibody / nanobody / monomer / partial-flow variants),
while enforcing a **stable output layout** and **inference-safe config override**.

## What this wrapper guarantees

- **One CLI** (`--task`) for multiple PPIFlow scripts in `/ppiflow/sample_*.py`.
- **Role-based uploads**: local input files are uploaded with stable filenames (e.g. `binder_input.pdb`,
  `antigen.pdb`, `framework.pdb`, `complex.pdb`, `motif.csv`) so the remote worker never guesses ordering.
- **Forced outputs** (remote side):
  - `--output_dir` is forced to `/runs/<task>/<run_name>/outputs`
  - `--name` is forced to `<run_name>`
- **Effective config** (remote side):
  - If a `--config` is provided, an `effective_config.yaml` is written under the run directory with:
    `model.use_deepspeed_evo_attention = False`
  - This makes inference **portable** (does not require deepspeed/nvcc kernels).

## Configuration

### Primary flags (local entrypoint)

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `binder` | Task router. One of: `binder`, `antibody`, `nanobody`, `monomer`, `scaffolding`, `ab_partial_flow`, `nb_partial_flow`, `binder_partial_flow`. |
| `--run-name` | `test1` | Unique run identifier. Controls output directory name and tarball name (`<task>.<run-name>.tar.gz`). |
| `--out-dir` | `./ppiflow_outputs` | Local directory to write the returned run bundle (`.tar.gz`). |
| `--model-weights` | **Required** | Local/remote path to a checkpoint. Remote resolves to `/models/<basename>` unless already under `/models/`. |
| `--config` | `None` | YAML config path (absolute in container or repo-relative). If provided, it will be rewritten to `effective_config.yaml` with deepspeed evo attention disabled. |

### Input file flags (local -> uploaded to Modal)

| Task | Required local file flags | Uploaded filename on worker |
|------|---------------------------|-----------------------------|
| `binder` | `--binder-input-pdb` | `binder_input.pdb` |
| `binder_partial_flow` | `--binder-input-pdb` | `binder_input.pdb` |
| `antibody` / `nanobody` | `--ab-antigen-pdb`, `--ab-framework-pdb` | `antigen.pdb`, `framework.pdb` |
| `ab_partial_flow` / `nb_partial_flow` | `--pf-complex-pdb` | `complex.pdb` |
| `scaffolding` | `--scaffold-motif-csv` | `motif.csv` |
| `monomer` | *(no file upload required)* | *(none)* |

### Binder args (sample_binder.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--binder-target-chain` | `B` | Target chain ID passed to `--target_chain`. |
| `--binder-binder-chain` | `A` | Binder chain ID passed to `--binder_chain`. |
| `--binder-specified-hotspots` | `None` | Hotspots string, e.g. `"B119,B141,B200"`. |
| `--binder-samples-min-length` | `75` | Minimum binder length. |
| `--binder-samples-max-length` | `76` | Maximum binder length. |
| `--binder-samples-per-target` | `5` | Number of samples per target. |

### Antibody / Nanobody args (sample_antibody_nanobody.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--ab-antigen-chain` | `None` | **Required** for `antibody/nanobody`. Passed to `--antigen_chain`. |
| `--ab-heavy-chain` | `None` | **Required** for `antibody/nanobody`. Passed to `--heavy_chain`. |
| `--ab-light-chain` | `None` | Optional light chain. Passed to `--light_chain` when provided. |
| `--ab-specified-hotspots` | `None` | Optional hotspot residues on antigen, e.g. `"A56,A58"`. |
| `--ab-cdr-length` | `None` | Optional CDR length override (string format per upstream script). |
| `--ab-samples-per-target` | `5` | Samples per target for antibody/nanobody. |

### Monomer unconditional args (sample_monomer.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--mono-length-subset` | `None` | **Required** for `monomer`. String list, e.g. `"[60, 80, 100]"`. |
| `--mono-samples-num` | `5` | Number of unconditional samples. |

### Scaffolding args (sample_monomer.py motif mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--scaffold-motif-names` | `None` | Optional motif name filter passed as `--motif_names`. |
| `--scaffold-samples-num` | `5` | Number of scaffolding samples. |

### Partial flow antibody / nanobody args (sample_antibody_nanobody_partial.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--pf-fixed-positions` | `None` | **Required**. Fixed positions string, e.g. `"H26,H27,H28,L50-63"`. |
| `--pf-cdr-position` | `None` | **Required**. CDR ranges string, e.g. `"H26-32,H45-56,H97-113"`. |
| `--pf-start-t` | `None` | **Required**. Partial flow start time (float). |
| `--pf-samples-per-target` | `None` | **Required**. Samples per target. |
| `--pf-retry-limit` | `10` | Passed as `--retry_Limit` (upstream spelling). |
| `--pf-specified-hotspots` | `None` | Optional hotspots for partial flow. |
| `--pf-antigen-chain` | `None` | **Required**. Passed to `--antigen_chain`. |
| `--pf-heavy-chain` | `None` | **Required**. Passed to `--heavy_chain`. |
| `--pf-light-chain` | `None` | Optional. Passed to `--light_chain` when provided. |

### Partial flow binder args (sample_binder_partial.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--bpf-target-chain` | `B` | Target chain passed to `--target_chain`. |
| `--bpf-binder-chain` | `A` | Binder chain passed to `--binder_chain`. |
| `--bpf-start-t` | `0.7` | Partial flow start time passed to `--start_t`. |

## Environment variables (Modal)

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `ppiflow` | Name of the Modal app. |
| `GPU` | `L40S` | GPU type for the worker (e.g. `A10G`, `A100`, `L40S`). |
| `TIMEOUT` | `36000` | Modal function timeout (seconds). |

## Persistent volumes & paths

- Models volume: `ppiflow-models` mounted at `/models`
- Runs volume: `ppiflow-runs` mounted at `/ppiflow-runs`

Expected checkpoint layout (one-time upload examples):
  1) ppiflow
  modal volume put ppiflow-models /models/antibody.ckpt antibody.ckpt
  modal volume put ppiflow-models /models/binder.ckpt   binder.ckpt
  modal volume put ppiflow-models /models/monomer.ckpt  monomer.ckpt
  modal volume put ppiflow-models /models/nanobody.ckpt nanobody.ckpt
  2) proteinmpnn
  # Upload ProteinMPNN weights to persistent Volume (one-time)
  modal volume put ppiflow-models /models/proteinmpnn_v_48_002.pt proteinmpnn_v_48_002.pt
  modal volume put ppiflow-models /models/proteinmpnn_v_48_010.pt proteinmpnn_v_48_010.pt
  modal volume put ppiflow-models /models/proteinmpnn_v_48_020.pt proteinmpnn_v_48_020.pt

## Outputs

- Each run is stored under the runs volume at:
  `/runs/<task>/<run_name>/`
  with:
  - `inputs/`  (uploaded inputs)
  - `outputs/` (upstream script outputs; forced `--output_dir`)
  - `effective_config.yaml` (if `--config` provided)
  - `cmd.txt` (exact executed command)
  - `stdout.log` (combined stdout/stderr)
  - `artifacts/` (best-effort collected: metrics/config + any `.csv`)

- The local CLI saves a `.tar.gz` bundle to:
  `<out-dir>/<task>.<run-name>.tar.gz`

## Typical usage

  # Binder (de novo)
  modal run ppiflow_app.py --task binder -- \
    --binder-input-pdb ~/target.pdb \
    --binder-target-chain B \
    --binder-binder-chain A \
    --binder-specified-hotspots "B119,B141,B200" \
    --binder-samples-min-length 75 \
    --binder-samples-max-length 76 \
    --binder-samples-per-target 5 \
    --config /ppiflow/configs/inference_binder.yaml \
    --model-weights /models/binder.ckpt \
    --run-name test1 \
    --out-dir ./ppiflow_outputs

  # Antibody partial flow
  modal run ppiflow_app.py --task ab_partial_flow -- \
    --pf-complex-pdb ~/complex.pdb \
    --pf-fixed-positions "H26,H27,H28,L50-63" \
    --pf-cdr-position "H26-32,H45-56,H97-113" \
    --pf-start-t 0.8 \
    --pf-samples-per-target 5 \
    --pf-antigen-chain A \
    --pf-heavy-chain H \
    --pf-light-chain L \
    --model-weights /models/antibody.ckpt \
    --run-name abp1
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from modal import App, Image, Volume

# -------------------------
# Modal configs
# -------------------------
APP_NAME = os.environ.get("MODAL_APP", "ppiflow")
GPU = os.environ.get("GPU", "L40S")  # e.g. A10G, A100, L40S
TIMEOUT = int(os.environ.get("TIMEOUT", "36000"))

# Persistent Volumes
MODELS_VOL = Volume.from_name("ppiflow-models", create_if_missing=True)
RUNS_VOL = Volume.from_name("ppiflow-runs", create_if_missing=True)

MODELS_DIR = Path("/models")
RUNS_DIR = Path("/ppiflow-runs")

# -------------------------
# Image definition
# -------------------------
PPIFLOW_REPO = "https://github.com/zhuqianhui2-hash/PPIFlow.git"  # updated at 2026-02-10-18:00
PPIFLOW_DIR = "/ppiflow"

PYTORCH_CU121_INDEX = "https://download.pytorch.org/whl/cu121"
PYG_WHL = "https://data.pyg.org/whl/torch-2.3.0+cu121.html"

TORCH_PKGS = [
    "torch==2.3.1+cu121",
    "torchvision==0.18.1+cu121",
    "torchaudio==2.3.1+cu121",
]

PYG_PKGS = [
    "pyg-lib==0.4.0+pt23cu121",
    "torch-scatter==2.1.2+pt23cu121",
    "torch-sparse==0.6.18+pt23cu121",
    "torch-cluster==1.6.3+pt23cu121",
    "torch-spline-conv==1.2.2+pt23cu121",
    "torch-geometric==2.6.1",
]

INFER_PKGS = [
    "numpy==1.26.3",
    "scipy==1.15.2",
    "pandas==2.2.3",
    "scikit-learn==1.2.2",
    "pyyaml==6.0.2",
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "hydra-submitit-launcher==1.2.0",
    "submitit==1.5.3",
    "tqdm==4.67.1",
    "lightning==2.5.0.post0",
    "pytorch-lightning==2.5.0.post0",
    "torchmetrics==1.6.2",
    "lightning-utilities==0.14.0",
    "einops==0.8.1",
    "dm-tree==0.1.6",
    "optree==0.14.1",
    "opt-einsum==3.4.0",
    "opt-einsum-fx==0.1.4",
    "e3nn==0.5.6",
    "fair-esm==2.0.0",
    "biopython==1.83",
    "biotite==1.0.1",
    "biotraj==1.2.2",
    "gemmi==0.6.5",
    "ihm==2.2",
    "modelcif==0.7",
    "tmtools==0.2.0",
    "freesasa==2.2.1",
    "mdtraj==1.10.3",
    "requests==2.32.3",
    "packaging==24.2",
    "typing-extensions==4.12.2",
    "protobuf==3.20.2",
    "tensorboard==2.19.0",
    "tensorboard-data-server==0.7.2",
    "grpcio==1.72.1",
    "gputil==1.4.0",
    "gpustat==1.1.1",
    "hjson==3.1.0",
    "ninja==1.11.1.3",
]

runtime_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "python3-dev",
        "pkg-config",
        "gfortran",
        "libopenblas-dev",
        "liblapack-dev",
        "libhdf5-dev",
        "libnetcdf-dev",
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": PPIFLOW_DIR})
    .run_commands(f"rm -rf {PPIFLOW_DIR} && git clone --depth 1 {PPIFLOW_REPO} {PPIFLOW_DIR}")
    .pip_install(*TORCH_PKGS, extra_index_url=PYTORCH_CU121_INDEX)
    .uv_pip_install(*PYG_PKGS, find_links=PYG_WHL)
    .uv_pip_install(*INFER_PKGS)
)

app = App(APP_NAME)

# -------------------------
# Task routing (PPIFlow sampling scripts only)
# -------------------------
TASK_TO_SCRIPT: dict[str, str] = {
    "binder": f"{PPIFLOW_DIR}/sample_binder.py",
    "antibody": f"{PPIFLOW_DIR}/sample_antibody_nanobody.py",
    "nanobody": f"{PPIFLOW_DIR}/sample_antibody_nanobody.py",
    "monomer": f"{PPIFLOW_DIR}/sample_monomer.py",
    "scaffolding": f"{PPIFLOW_DIR}/sample_monomer.py",
    "ab_partial_flow": f"{PPIFLOW_DIR}/sample_antibody_nanobody_partial.py",
    "nb_partial_flow": f"{PPIFLOW_DIR}/sample_antibody_nanobody_partial.py",
    "binder_partial_flow": f"{PPIFLOW_DIR}/sample_binder_partial.py",
}

MPNN_TASKS = {"mpnn_stage1", "mpnn_stage2"}

# -------------------------
# Helpers
# -------------------------
def _tar_dir(src_dir: Path, out_tar_gz: Path) -> None:
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _write_effective_config(src_config: Path, dst_config: Path) -> None:
    import yaml

    cfg = yaml.safe_load(src_config.read_text()) or {}
    cfg.setdefault("model", {})
    cfg["model"]["use_deepspeed_evo_attention"] = False
    dst_config.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _collect_artifacts(run_dir: Path) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    want_exact = {"metrics.csv", "config.yml", "config.yaml"}
    for f in _iter_files(run_dir):
        if f.name in want_exact:
            dst = artifacts / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())

    for f in _iter_files(run_dir):
        if f.suffix.lower() == ".csv":
            dst = artifacts / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())


def _script_help_text(script: Path) -> str:
    p = subprocess.run(
        ["python", str(script), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.stdout or ""


def _script_supports_flag(script: Path, flag: str, help_text: str | None = None) -> bool:
    txt = help_text if help_text is not None else _script_help_text(script)
    return flag in txt


def _find_abmpnn_run_script() -> Path:
    """
    Find ABMPNN/ProteinMPNN runnable script under /ppiflow/ProteinMPNN.
    We try common names, then fallback to a glob search.
    """
    base = Path(PPIFLOW_DIR) / "ProteinMPNN"
    candidates = [
        base / "protein_mpnn_run.py",
        base / "run.py",
        base / "protein_mpnn_run_main.py",
        Path(PPIFLOW_DIR) / "protein_mpnn_run.py",
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = sorted(base.rglob("*mpnn*run*.py"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"ABMPNN run script not found under {base}. "
        "Expected something like ProteinMPNN/protein_mpnn_run.py"
    )


def _resolve_abmpnn_ckpt(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
    else:
        p = Path(PPIFLOW_DIR) / "ProteinMPNN" / "model_weights" / "abmpnn.pt"
    if not p.exists():
        raise FileNotFoundError(f"ABMPNN checkpoint not found: {p}")
    return p


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    path.write_text(buf.getvalue())


def _collect_pdbs_for_mpnn(src_outputs: Path) -> list[Path]:
    pdbs = sorted(src_outputs.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No .pdb found under: {src_outputs}")
    return pdbs


def _gather_mpnn_fastas(mpnn_out_root: Path) -> list[Path]:
    # accept .fa / .fasta / .faa etc. under seqs/
    hits: list[Path] = []
    for pat in ("seqs/*.fa", "seqs/*.fasta", "seqs/*.faa", "seqs/*.fa.gz", "seqs/*.fasta.gz"):
        hits.extend(mpnn_out_root.rglob(pat))
    return sorted(set(hits))

def _detect_abmpnn_cli_flags(mpnn_script: Path) -> dict[str, str]:
    ht = _script_help_text(mpnn_script)

    # folder input (this is required for ProteinMPNN/protein_mpnn_run.py)
    folder_flag = "--folder_with_pdbs_path" if _script_supports_flag(mpnn_script, "--folder_with_pdbs_path", ht) else ""
    if not folder_flag and _script_supports_flag(mpnn_script, "--pdb_dir", ht):
        folder_flag = "--pdb_dir"
    if not folder_flag and _script_supports_flag(mpnn_script, "--input_folder", ht):
        folder_flag = "--input_folder"

    # single pdb input (optional, many scripts do NOT support it)
    pdb_flag = "--pdb_path" if _script_supports_flag(mpnn_script, "--pdb_path", ht) else ""
    if not pdb_flag and _script_supports_flag(mpnn_script, "--input_pdb", ht):
        pdb_flag = "--input_pdb"

    # out dir
    out_flag = "--out_folder" if _script_supports_flag(mpnn_script, "--out_folder", ht) else ""
    if not out_flag and _script_supports_flag(mpnn_script, "--output_dir", ht):
        out_flag = "--output_dir"
    if not out_flag and _script_supports_flag(mpnn_script, "--out_dir", ht):
        out_flag = "--out_dir"

    # checkpoint (ABMPNN single file OR ProteinMPNN model dir system)
    ckpt_candidates = [
        "--checkpoint_path",
        "--checkpoint",
        "--model_path",
        "--weights",
        "--ckpt",
        "--checkpoint_file",
    ]
    ckpt_flag = ""
    for f in ckpt_candidates:
        if _script_supports_flag(mpnn_script, f, ht):
            ckpt_flag = f
            break

    # ProteinMPNN classic interface (dir + model_name)
    weights_dir_flag = "--path_to_model_weights" if _script_supports_flag(mpnn_script, "--path_to_model_weights", ht) else ""
    model_name_flag = "--model_name" if _script_supports_flag(mpnn_script, "--model_name", ht) else ""

    nseq_flag = "--num_seq_per_target" if _script_supports_flag(mpnn_script, "--num_seq_per_target", ht) else ""
    if not nseq_flag and _script_supports_flag(mpnn_script, "--num_seqs", ht):
        nseq_flag = "--num_seqs"

    temp_flag = "--sampling_temp" if _script_supports_flag(mpnn_script, "--sampling_temp", ht) else ""
    seed_flag = "--seed" if _script_supports_flag(mpnn_script, "--seed", ht) else ""
    batch_flag = "--batch_size" if _script_supports_flag(mpnn_script, "--batch_size", ht) else ""

    if not out_flag:
        raise RuntimeError(
            f"Cannot detect output flag for {mpnn_script}. "
            "Expected one of: --out_folder / --output_dir / --out_dir"
        )

    # For protein_mpnn_run.py, folder_flag is required
    if "protein_mpnn_run.py" in str(mpnn_script) and not folder_flag:
        raise RuntimeError(
            f"{mpnn_script} appears to require folder input, but folder flag not detected. "
            "Expected --folder_with_pdbs_path."
        )

    return {
        "folder_flag": folder_flag,
        "pdb_flag": pdb_flag,
        "out_flag": out_flag,
        "ckpt_flag": ckpt_flag,
        "weights_dir_flag": weights_dir_flag,
        "model_name_flag": model_name_flag,
        "nseq_flag": nseq_flag,
        "temp_flag": temp_flag,
        "seed_flag": seed_flag,
        "batch_flag": batch_flag,
    }


def _run_abmpnn_on_folder(
    *,
    mpnn_script: Path,
    flags: dict[str, str],
    pdb_folder: Path,
    out_folder: Path,
    ckpt_path: Path,
    num_seqs: int,
    sampling_temp: float,
    seed: int,
    batch_size: int,
    model_name: str | None,
    log_path: Path,
) -> None:
    """
    Ultra-safe runner for ProteinMPNN / ABMPNN scripts.

    Fixes two classes of issues:
    1) Upstream bug: fixed_positions_dict may be printed without initialization.
       -> Patch `main(args)` to define `fixed_positions_dict = {}` first.
    2) Import error when executing a copied script elsewhere (protein_mpnn_utils not found).
       -> Write patched script NEXT TO the original script and run with cwd=original_dir.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Step 0: patch script IN PLACE (same directory as original)
    # -------------------------
    run_script = mpnn_script
    patched = False
    patch_dir = mpnn_script.parent  # e.g. /ppiflow/ProteinMPNN
    try:
        txt = mpnn_script.read_text(encoding="utf-8", errors="ignore")

        # Only patch if it looks like the ProteinMPNN entry script
        if mpnn_script.name == "protein_mpnn_run.py" and "def main(args):" in txt:
            if "fixed_positions_dict = {}  # PATCH" not in txt:
                lines = txt.splitlines(True)
                out_lines: list[str] = []
                inserted = False
                for line in lines:
                    out_lines.append(line)
                    if (not inserted) and line.strip() == "def main(args):":
                        out_lines.append("    fixed_positions_dict = {}  # PATCH: avoid UnboundLocalError\n")
                        inserted = True

                if inserted:
                    run_script = patch_dir / "_patched_protein_mpnn_run.py"
                    run_script.write_text("".join(out_lines), encoding="utf-8")
                    patched = True
            else:
                # already patched earlier
                run_script = patch_dir / "_patched_protein_mpnn_run.py"
                if run_script.exists():
                    patched = True
    except Exception:
        # fall back to original script
        run_script = mpnn_script
        patched = False

    # -------------------------
    # Step 1: build argv
    # -------------------------
    argv: list[str] = ["python", str(run_script)]

    folder_flag = flags.get("folder_flag") or ""
    pdb_flag = flags.get("pdb_flag") or ""
    if folder_flag:
        argv += [folder_flag, str(pdb_folder)]
    elif pdb_flag:
        raise RuntimeError(
            "MPNN runner is configured for folder mode, but folder_flag is missing and pdb_flag exists. "
            "Implement a single-PDB runner or ensure folder_flag is detected correctly."
        )
    else:
        raise RuntimeError("Neither folder_flag nor pdb_flag detected for MPNN script.")

    out_flag = flags.get("out_flag") or ""
    if not out_flag:
        raise RuntimeError("No out_flag detected for MPNN script.")
    argv += [out_flag, str(out_folder)]

    ckpt_flag = flags.get("ckpt_flag") or ""
    if ckpt_flag:
        argv += [ckpt_flag, str(ckpt_path)]
    else:
        weights_dir_flag = flags.get("weights_dir_flag") or ""
        model_name_flag = flags.get("model_name_flag") or ""
        if not weights_dir_flag or not model_name_flag:
            raise RuntimeError(
                "MPNN script does not expose a checkpoint flag and does not expose "
                "--path_to_model_weights/--model_name. Cannot pass weights."
            )
        weights_dir = Path("/models")
        argv += [weights_dir_flag, str(weights_dir)]

        if not model_name:
            raise ValueError(
                "mpnn_model_name is required for this ProteinMPNN script (--model_name). "
                "You passed None/empty."
            )
        argv += [model_name_flag, str(model_name)]

        # --- sanity check for classic ProteinMPNN ---
        expected_pt = weights_dir / f"{model_name}.pt"
        if not expected_pt.exists():
            available = sorted(p.name for p in weights_dir.glob("*.pt"))
            raise FileNotFoundError(
                f"ProteinMPNN weight not found: {expected_pt}\n"
                f"Available under /models: {available}"
            )
        # --- auto-detect CA-only checkpoint and toggle --ca_only ---
        try:
            import torch
            ckpt = torch.load(str(expected_pt), map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            w = sd.get("features.edge_embedding.weight", None)
            if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                if int(w.shape[1]) == 167:
                    argv += ["--ca_only"]
        except Exception:
            pass




    nseq_flag = flags.get("nseq_flag") or ""
    temp_flag = flags.get("temp_flag") or ""
    seed_flag = flags.get("seed_flag") or ""
    batch_flag = flags.get("batch_flag") or ""

    if nseq_flag:
        argv += [nseq_flag, str(int(num_seqs))]
    if temp_flag:
        argv += [temp_flag, str(float(sampling_temp))]
    if seed_flag:
        argv += [seed_flag, str(int(seed))]
    if batch_flag:
        argv += [batch_flag, str(int(batch_size))]

    # -------------------------
    # Step 2: DO NOT pass --fixed_positions by default.
    #
    # Reason:
    # - After we patch `fixed_positions_dict = {}` at the start of main(),
    #   passing an empty fixed_positions file can trigger KeyError in tied_featurize
    #   (it expects per-sample keys like b['name'] == 'vhh1_0').
    # -------------------------
    # (intentionally disabled)


    # -------------------------
    # Step 3: run with cwd = original ProteinMPNN dir to satisfy relative imports
    # -------------------------
    p = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(patch_dir),  # critical: make `protein_mpnn_utils` importable
    )

    dbg: list[str] = []
    dbg.append("=== runner ===")
    dbg.append(f"original_script: {mpnn_script}")
    dbg.append(f"executed_script: {run_script}")
    dbg.append(f"patched: {patched}")
    dbg.append(f"cwd: {patch_dir}")
    dbg.append("")
    dbg.append("=== argv ===")
    dbg.append(" ".join(argv))
    dbg.append("")
    dbg.append("=== output ===")
    dbg.append(p.stdout or "")
    log_path.write_text("\n".join(dbg))

    if p.returncode != 0:
        raise RuntimeError(f"ABMPNN/ProteinMPNN failed (exit {p.returncode}). See {log_path}")







def _stage_pdb_folder(pdbs: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in pdbs:
        # keep stable filenames
        (dst_dir / p.name).write_bytes(p.read_bytes())


# -------------------------
# Remote GPU job: unified runner
# -------------------------
@app.function(
    gpu=GPU,
    cpu=(2, 8),
    timeout=TIMEOUT,
    image=runtime_image,
    volumes={str(MODELS_DIR): MODELS_VOL, str(RUNS_DIR): RUNS_VOL},
)
def run_ppiflow_structured(
    # ---------- common ----------
    task: str,
    run_name: str,
    input_files: list[tuple[str, bytes]],
    model_weights: str | None,
    config: str | None,
    # ---------- binder (sample_binder.py) ----------
    binder_target_chain: str,
    binder_binder_chain: str,
    binder_specified_hotspots: str | None,
    binder_samples_min_length: int,
    binder_samples_max_length: int,
    binder_samples_per_target: int,
    # ---------- antibody/nanobody (sample_antibody_nanobody.py) ----------
    ab_antigen_chain: str | None,
    ab_heavy_chain: str | None,
    ab_light_chain: str | None,
    ab_specified_hotspots: str | None,
    ab_cdr_length: str | None,
    ab_samples_per_target: int,
    # ---------- monomer (sample_monomer.py unconditional) ----------
    mono_length_subset: str | None,
    mono_samples_num: int,
    # ---------- scaffolding (sample_monomer.py motif) ----------
    scaffold_motif_names: str | None,
    scaffold_samples_num: int,
    # ---------- partial flow antibody/nanobody ----------
    pf_fixed_positions: str | None,
    pf_cdr_position: str | None,
    pf_specified_hotspots: str | None,
    pf_start_t: float | None,
    pf_samples_per_target: int | None,
    pf_retry_limit: int,
    pf_antigen_chain: str | None,
    pf_heavy_chain: str | None,
    pf_light_chain: str | None,
    # ---------- partial flow binder ----------
    bpf_target_chain: str,
    bpf_binder_chain: str,
    bpf_start_t: float,
    # ---------- mpnn ----------
    mpnn_source_task: str | None,
    mpnn_source_run: str | None,
    mpnn_num_seq_per_target_stage1: int,
    mpnn_num_seq_per_target_stage2: int,
    mpnn_temp_stage1: float,
    mpnn_temp_stage2: float,
    mpnn_model_name: str | None,
    mpnn_batch_size: int,
    mpnn_seed: int,
    mpnn_ckpt_path: str | None,
) -> bytes:
    # -------------------------
    # Branch 1: MPNN tasks (operate on existing run dir)
    # -------------------------
    if task in MPNN_TASKS:
        if not mpnn_source_task or not mpnn_source_run:
            raise ValueError("mpnn_stage1/2 requires --mpnn-source-task and --mpnn-source-run")

        src_run_dir = RUNS_DIR / mpnn_source_task / mpnn_source_run
        src_outputs = src_run_dir / "outputs"
        if not src_outputs.exists():
            raise FileNotFoundError(f"Source outputs missing: {src_outputs}")

        pdbs = _collect_pdbs_for_mpnn(src_outputs)

        mpnn_script = _find_abmpnn_run_script()
        ckpt_path = _resolve_abmpnn_ckpt(mpnn_ckpt_path)

        # detect flags once
        flags = _detect_abmpnn_cli_flags(mpnn_script)
        (src_run_dir / f"{task}.mpnn_help.txt").write_text(_script_help_text(mpnn_script))

        mpnn_dir = src_run_dir / task
        mpnn_out = mpnn_dir / "out"
        mpnn_dir.mkdir(parents=True, exist_ok=True)

        if task == "mpnn_stage1":
            num_seq = int(mpnn_num_seq_per_target_stage1)
            temp = float(mpnn_temp_stage1)
            manifest_name = "candidates_stage1.csv"
        else:
            num_seq = int(mpnn_num_seq_per_target_stage2)
            temp = float(mpnn_temp_stage2)
            manifest_name = "candidates_stage2.csv"

        mpnn_dir = src_run_dir / task
        mpnn_out = mpnn_dir / "out"
        pdb_folder = mpnn_dir / "pdbs"
        mpnn_dir.mkdir(parents=True, exist_ok=True)

        _stage_pdb_folder(pdbs, pdb_folder)

        log_path = mpnn_dir / "mpnn_folder.log"
        _run_abmpnn_on_folder(
         mpnn_script=mpnn_script,
        flags=flags,
        pdb_folder=pdb_folder,
        out_folder=mpnn_out,
        ckpt_path=ckpt_path,
        num_seqs=num_seq,
        sampling_temp=temp,
        seed=int(mpnn_seed),
        batch_size=int(mpnn_batch_size),
        model_name=mpnn_model_name,   # 你要把这个参数加回 signature & local entrypoint
        log_path=log_path,
        )

        fastas = _gather_mpnn_fastas(mpnn_out)
        if not fastas:
            raise RuntimeError(f"No fasta outputs found under {mpnn_out} (expected */seqs/*.fa*)")

        rows: list[dict[str, Any]] = []
        for fa in fastas:
            # .../out/<pdb_stem>/seqs/<something>.fa
            try:
                pdb_stem = fa.parent.parent.name
            except Exception:
                pdb_stem = fa.stem
            rows.append(
                {
                    "pdb_path": str((src_outputs / f"{pdb_stem}.pdb").resolve()),
                    "seq_fasta": str(fa.resolve()),
                    "iptm": "",
                    "ptm": "",
                    "passed": "",
                }
            )

        _write_csv(src_run_dir / manifest_name, rows, ["pdb_path", "seq_fasta", "iptm", "ptm", "passed"])

        (src_run_dir / f"{task}.meta.json").write_text(
            json.dumps(
                {
                    "mpnn_script": str(mpnn_script),
                    "ckpt_path": str(ckpt_path),
                    "num_seq_per_target": num_seq,
                    "sampling_temp": temp,
                    "batch_size": int(mpnn_batch_size),
                    "seed": int(mpnn_seed),
                    "detected_flags": flags,
                },
                indent=2,
            )
        )

        RUNS_VOL.commit()

        # Return a tar of the SOURCE run dir (so you get both backbone outputs and mpnn outputs)
        with tempfile.TemporaryDirectory() as td:
            tar_path = Path(td) / f"{task}.{run_name}.tar.gz"
            _tar_dir(src_run_dir, tar_path)
            return tar_path.read_bytes()

    # -------------------------
    # Branch 2: PPIFlow sampling tasks
    # -------------------------
    if task not in TASK_TO_SCRIPT:
        raise ValueError(f"Unknown task={task}. Choose from {sorted(TASK_TO_SCRIPT) + sorted(MPNN_TASKS)}")

    script = Path(TASK_TO_SCRIPT[task])
    if not script.exists():
        raise FileNotFoundError(f"Script not found in image: {script}")

    run_dir = RUNS_DIR / task / run_name
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---- write inputs (role-based filenames) ----
    for fname, content in input_files:
        (inputs_dir / Path(fname).name).write_bytes(content)

    # ---- resolve model checkpoint (required for sampling tasks) ----
    if not model_weights:
        raise ValueError("--model-weights is required for PPIFlow sampling tasks")
    mw = Path(model_weights)
    model_ckpt = mw if str(mw).startswith(str(MODELS_DIR)) else (MODELS_DIR / mw.name)
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    # ---- resolve config + write effective config (optional) ----
    effective_config: Path | None = None
    if config:
        cfg_path = Path(config)
        if not cfg_path.is_absolute():
            cfg_guess = Path(PPIFLOW_DIR) / config
            cfg_path = cfg_guess if cfg_guess.exists() else cfg_path
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {config} (resolved: {cfg_path})")
        effective_config = run_dir / "effective_config.yaml"
        _write_effective_config(cfg_path, effective_config)

    def p_in(name: str) -> Path:
        p = inputs_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")
        return p

    argv: list[str] = ["python", str(script)]

    if task == "binder":
        input_pdb = p_in("binder_input.pdb")
        argv += ["--input_pdb", str(input_pdb)]
        argv += ["--target_chain", binder_target_chain]
        argv += ["--binder_chain", binder_binder_chain]
        if effective_config:
            argv += ["--config", str(effective_config)]
        if binder_specified_hotspots:
            argv += ["--specified_hotspots", binder_specified_hotspots]
        argv += [
            "--samples_min_length",
            str(binder_samples_min_length),
            "--samples_max_length",
            str(binder_samples_max_length),
            "--samples_per_target",
            str(binder_samples_per_target),
            "--model_weights",
            str(model_ckpt),
            "--output_dir",
            str(outputs_dir),
            "--name",
            run_name,
        ]

    elif task in {"antibody", "nanobody"}:
        antigen_pdb = p_in("antigen.pdb")
        framework_pdb = p_in("framework.pdb")
        argv += ["--antigen_pdb", str(antigen_pdb), "--framework_pdb", str(framework_pdb)]

        if not ab_antigen_chain:
            raise ValueError("antibody/nanobody requires --ab-antigen-chain")
        if not ab_heavy_chain:
            raise ValueError("antibody/nanobody requires --ab-heavy-chain")

        argv += ["--antigen_chain", ab_antigen_chain, "--heavy_chain", ab_heavy_chain]
        if ab_light_chain:
            argv += ["--light_chain", ab_light_chain]
        if ab_specified_hotspots:
            argv += ["--specified_hotspots", ab_specified_hotspots]
        if ab_cdr_length:
            argv += ["--cdr_length", ab_cdr_length]
        if effective_config:
            argv += ["--config", str(effective_config)]
        argv += [
            "--samples_per_target",
            str(ab_samples_per_target),
            "--model_weights",
            str(model_ckpt),
            "--output_dir",
            str(outputs_dir),
            "--name",
            run_name,
        ]

    elif task == "monomer":
        if mono_length_subset is None:
            raise ValueError("monomer requires --mono-length-subset")
        if effective_config:
            argv += ["--config", str(effective_config)]
        argv += [
            "--model_weights",
            str(model_ckpt),
            "--output_dir",
            str(outputs_dir),
            "--length_subset",
            mono_length_subset,
            "--samples_num",
            str(mono_samples_num),
        ]

    elif task == "scaffolding":
        motif_csv = p_in("motif.csv")
        if effective_config:
            argv += ["--config", str(effective_config)]
        argv += ["--model_weights", str(model_ckpt), "--output_dir", str(outputs_dir), "--motif_csv", str(motif_csv)]
        if scaffold_motif_names:
            argv += ["--motif_names", scaffold_motif_names]
        argv += ["--samples_num", str(scaffold_samples_num)]

    elif task in {"ab_partial_flow", "nb_partial_flow"}:
        complex_pdb = p_in("complex.pdb")

        if not pf_fixed_positions:
            raise ValueError(f"{task} requires --pf-fixed-positions")
        if not pf_cdr_position:
            raise ValueError(f"{task} requires --pf-cdr-position")
        if pf_start_t is None:
            raise ValueError(f"{task} requires --pf-start-t")
        if pf_samples_per_target is None:
            raise ValueError(f"{task} requires --pf-samples-per-target")
        if not pf_antigen_chain:
            raise ValueError(f"{task} requires --pf-antigen-chain")
        if not pf_heavy_chain:
            raise ValueError(f"{task} requires --pf-heavy-chain")

        argv += [
            "--complex_pdb",
            str(complex_pdb),
            "--fixed_positions",
            pf_fixed_positions,
            "--cdr_position",
            pf_cdr_position,
            "--start_t",
            str(pf_start_t),
            "--samples_per_target",
            str(pf_samples_per_target),
            "--output_dir",
            str(outputs_dir),
            "--retry_Limit",
            str(pf_retry_limit),
        ]
        if pf_specified_hotspots:
            argv += ["--specified_hotspots", pf_specified_hotspots]
        if effective_config:
            argv += ["--config", str(effective_config)]

        argv += [
            "--model_weights",
            str(model_ckpt),
            "--antigen_chain",
            pf_antigen_chain,
            "--heavy_chain",
            pf_heavy_chain,
        ]
        if pf_light_chain:
            argv += ["--light_chain", pf_light_chain]
        argv += ["--name", run_name]

    elif task == "binder_partial_flow":
        input_pdb = p_in("binder_input.pdb")
        argv += ["--input_pdb", str(input_pdb)]
        if effective_config:
            argv += ["--config", str(effective_config)]
        argv += [
            "--target_chain",
            bpf_target_chain,
            "--binder_chain",
            bpf_binder_chain,
            "--start_t",
            str(bpf_start_t),
            "--output_dir",
            str(outputs_dir),
        ]

    else:
        raise ValueError(f"Task routed but not implemented: {task}")

    # ---- run ----
    (run_dir / "cmd.txt").write_text(" ".join(argv) + "\n")
    run_cwd = inputs_dir if task == "scaffolding" else None

    p = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(run_cwd) if run_cwd else None,
    )
    (run_dir / "stdout.log").write_text(p.stdout or "")

    # if fail, but outputs exist, keep; else raise
    pdbs = sorted((run_dir / "outputs").glob("*.pdb"))
    if p.returncode != 0 and not pdbs:
        raise RuntimeError(f"PPIFlow failed (exit {p.returncode}). See {run_dir}/stdout.log")

    _collect_artifacts(run_dir)
    RUNS_VOL.commit()

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"{task}.{run_name}.tar.gz"
        _tar_dir(run_dir, tar_path)
        return tar_path.read_bytes()


# -------------------------
# Local entrypoint
# -------------------------
@app.local_entrypoint()
def submit_ppiflow(
    # ---------- common ----------
    task: str = "binder",
    run_name: str = "test1",
    out_dir: str = "./ppiflow_outputs",
    # ---------- local input files ----------
    binder_input_pdb: str | None = None,
    ab_antigen_pdb: str | None = None,
    ab_framework_pdb: str | None = None,
    pf_complex_pdb: str | None = None,
    scaffold_motif_csv: str | None = None,
    # ---------- model weights (sampling tasks only) ----------
    model_weights: str | None = None,
    # ---------- config ----------
    config: str | None = None,
    # ---------- binder args ----------
    binder_target_chain: str = "B",
    binder_binder_chain: str = "A",
    binder_specified_hotspots: str | None = None,
    binder_samples_min_length: int = 75,
    binder_samples_max_length: int = 76,
    binder_samples_per_target: int = 5,
    # ---------- antibody/nanobody args ----------
    ab_antigen_chain: str | None = None,
    ab_heavy_chain: str | None = None,
    ab_light_chain: str | None = None,
    ab_specified_hotspots: str | None = None,
    ab_cdr_length: str | None = None,
    ab_samples_per_target: int = 5,
    # ---------- monomer unconditional ----------
    mono_length_subset: str | None = None,
    mono_samples_num: int = 5,
    # ---------- scaffolding ----------
    scaffold_motif_names: str | None = None,
    scaffold_samples_num: int = 5,
    # ---------- partial flow antibody/nanobody ----------
    pf_fixed_positions: str | None = None,
    pf_cdr_position: str | None = None,
    pf_specified_hotspots: str | None = None,
    pf_start_t: float | None = None,
    pf_samples_per_target: int | None = None,
    pf_retry_limit: int = 10,
    pf_antigen_chain: str | None = None,
    pf_heavy_chain: str | None = None,
    pf_light_chain: str | None = None,
    # ---------- partial flow binder ----------
    bpf_target_chain: str = "B",
    bpf_binder_chain: str = "A",
    bpf_start_t: float = 0.7,
    # ---------- mpnn ----------
    mpnn_source_task: str | None = None,
    mpnn_source_run: str | None = None,
    mpnn_num_seq_per_target_stage1: int = 8,
    mpnn_num_seq_per_target_stage2: int = 4,
    mpnn_temp_stage1: float = 0.5,
    mpnn_temp_stage2: float = 0.1,
    mpnn_model_name: str = "v_48_020",
    mpnn_batch_size: int = 1,
    mpnn_seed: int = 0,
    mpnn_ckpt_path: str | None = None,
) -> None:
    """
    Unified Modal CLI.
    - Sampling tasks upload inputs (role-based).
    - MPNN tasks do NOT upload inputs; they operate on an existing run:
        --mpnn-source-task <task> --mpnn-source-run <run_name>
    """
    allowed = set(TASK_TO_SCRIPT) | set(MPNN_TASKS)
    if task not in allowed:
        raise ValueError(f"--task must be one of {sorted(allowed)}")

    def _read_file_as(role_name: str, path: str | None) -> tuple[str, bytes] | None:
        if not path:
            return None
        pp = Path(path).expanduser()
        if not pp.exists():
            raise FileNotFoundError(f"Local file not found: {pp}")
        return (role_name, pp.read_bytes())

    input_files: list[tuple[str, bytes]] = []

    # Build uploads only for sampling tasks
    if task in TASK_TO_SCRIPT:
        if task in {"binder", "binder_partial_flow"}:
            if not binder_input_pdb:
                raise ValueError(f"{task} requires --binder-input-pdb")
            item = _read_file_as("binder_input.pdb", binder_input_pdb)
            if item:
                input_files.append(item)

        if task in {"antibody", "nanobody"}:
            if not ab_antigen_pdb or not ab_framework_pdb:
                raise ValueError(f"{task} requires --ab-antigen-pdb and --ab-framework-pdb")
            input_files.append(_read_file_as("antigen.pdb", ab_antigen_pdb))  # type: ignore[arg-type]
            input_files.append(_read_file_as("framework.pdb", ab_framework_pdb))  # type: ignore[arg-type]

        if task in {"ab_partial_flow", "nb_partial_flow"}:
            if not pf_complex_pdb:
                raise ValueError(f"{task} requires --pf-complex-pdb")
            input_files.append(_read_file_as("complex.pdb", pf_complex_pdb))  # type: ignore[arg-type]

        if task == "scaffolding":
            if not scaffold_motif_csv:
                raise ValueError("scaffolding requires --scaffold-motif-csv")

            csv_path = Path(scaffold_motif_csv).expanduser()
            if not csv_path.exists():
                raise FileNotFoundError(f"Local file not found: {csv_path}")

            csv_text = csv_path.read_text(encoding="utf-8-sig")
            reader = csv.DictReader(StringIO(csv_text))
            required_cols = {"target", "length", "contig", "motif_path"}
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(f"motif.csv must have columns {sorted(required_cols)}, got {reader.fieldnames}")

            rows = list(reader)
            if not rows:
                raise ValueError("motif.csv has no data rows")

            csv_dir = csv_path.parent
            motif_files: dict[str, Path] = {}
            for r in rows:
                mp = (r.get("motif_path") or "").strip()
                if not mp:
                    raise ValueError(f"motif_path is empty in row: {r}")

                mp_path = Path(mp).expanduser() if mp.startswith("~") else Path(mp)
                if not mp_path.is_absolute():
                    mp_path = (csv_dir / mp_path).resolve()
                if not mp_path.exists():
                    raise FileNotFoundError(f"motif_path file not found: {mp_path} (from motif_path={mp!r})")

                stable_name = mp_path.name
                if stable_name in motif_files and motif_files[stable_name] != mp_path:
                    stable_name = f"{mp_path.stem}.{len(motif_files) + 1}{mp_path.suffix}"
                motif_files[stable_name] = mp_path
                r["motif_path"] = stable_name

            out_buf = StringIO()
            writer = csv.DictWriter(out_buf, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            input_files.append(("motif.csv", out_buf.getvalue().encode("utf-8")))

            for stable_name, p in motif_files.items():
                input_files.append((stable_name, p.read_bytes()))

        if task == "scaffolding" and scaffold_motif_names:
            s = scaffold_motif_names.strip()
            if not s.startswith("["):
                scaffold_motif_names = json.dumps([s])

    # dispatch
    tar_bytes = run_ppiflow_structured.remote(
        task=task,
        run_name=run_name,
        input_files=input_files,
        model_weights=model_weights,
        config=config,
        binder_target_chain=binder_target_chain,
        binder_binder_chain=binder_binder_chain,
        binder_specified_hotspots=binder_specified_hotspots,
        binder_samples_min_length=binder_samples_min_length,
        binder_samples_max_length=binder_samples_max_length,
        binder_samples_per_target=binder_samples_per_target,
        ab_antigen_chain=ab_antigen_chain,
        ab_heavy_chain=ab_heavy_chain,
        ab_light_chain=ab_light_chain,
        ab_specified_hotspots=ab_specified_hotspots,
        ab_cdr_length=ab_cdr_length,
        ab_samples_per_target=ab_samples_per_target,
        mono_length_subset=mono_length_subset,
        mono_samples_num=mono_samples_num,
        scaffold_motif_names=scaffold_motif_names,
        scaffold_samples_num=scaffold_samples_num,
        pf_fixed_positions=pf_fixed_positions,
        pf_cdr_position=pf_cdr_position,
        pf_specified_hotspots=pf_specified_hotspots,
        pf_start_t=pf_start_t,
        pf_samples_per_target=pf_samples_per_target,
        pf_retry_limit=pf_retry_limit,
        pf_antigen_chain=pf_antigen_chain,
        pf_heavy_chain=pf_heavy_chain,
        pf_light_chain=pf_light_chain,
        bpf_target_chain=bpf_target_chain,
        bpf_binder_chain=bpf_binder_chain,
        bpf_start_t=bpf_start_t,
        mpnn_source_task=mpnn_source_task,
        mpnn_source_run=mpnn_source_run,
        mpnn_num_seq_per_target_stage1=mpnn_num_seq_per_target_stage1,
        mpnn_num_seq_per_target_stage2=mpnn_num_seq_per_target_stage2,
        mpnn_temp_stage1=mpnn_temp_stage1,
        mpnn_temp_stage2=mpnn_temp_stage2,
        mpnn_batch_size=mpnn_batch_size,
        mpnn_seed=mpnn_seed,
        mpnn_ckpt_path=mpnn_ckpt_path,
        mpnn_model_name=mpnn_model_name,
    )

    out_dir_p = Path(out_dir).expanduser()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir_p / f"{task}.{run_name}.tar.gz"
    out_tar.write_bytes(tar_bytes)
    print(f"[ok] saved: {out_tar}")
