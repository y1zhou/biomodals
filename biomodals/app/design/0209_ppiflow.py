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

  modal volume put ppiflow-models /models/antibody.ckpt antibody.ckpt
  modal volume put ppiflow-models /models/binder.ckpt   binder.ckpt
  modal volume put ppiflow-models /models/monomer.ckpt  monomer.ckpt
  modal volume put ppiflow-models /models/nanobody.ckpt nanobody.ckpt

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

import os
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

from modal import App, Image, Volume

# -------------------------
# Modal configs
# -------------------------
APP_NAME = os.environ.get("MODAL_APP", "ppiflow")
GPU = os.environ.get("GPU", "L40S")  # e.g. A10G, A100
TIMEOUT = int(os.environ.get("TIMEOUT", "36000"))

# Persistent Volumes
MODELS_VOL = Volume.from_name("ppiflow-models", create_if_missing=True)
RUNS_VOL = Volume.from_name("ppiflow-runs", create_if_missing=True)

MODELS_DIR = Path("/models")
RUNS_DIR = Path("/ppiflow-runs")

# -------------------------
# Image definition
# -------------------------
# TODO: pin versions according to your repo requirements (cuda/torch/etc.)


PPIFLOW_REPO = "https://github.com/zhuqianhui2-hash/PPIFlow.git"
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

# Inference-safe superset for binder/nanobody/monomer pipelines
INFER_PKGS = [
    # config/runtime
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
    # lightning stack (often imported by model code even for inference)
    "lightning==2.5.0.post0",
    "pytorch-lightning==2.5.0.post0",
    "torchmetrics==1.6.2",
    "lightning-utilities==0.14.0",
    # geometry / embeddings frequently used in protein models
    "einops==0.8.1",
    "dm-tree==0.1.6",
    "optree==0.14.1",
    "opt-einsum==3.4.0",
    "opt-einsum-fx==0.1.4",
    "e3nn==0.5.6",
    "fair-esm==2.0.0",
    # bio/structure IO
    "biopython==1.83",
    "biotite==1.0.1",
    "biotraj==1.2.2",
    "gemmi==0.6.5",
    "ihm==2.2",
    "modelcif==0.7",
    "tmtools==0.2.0",
    "freesasa==2.2.1",
    "mdtraj==1.10.3",
    # misc util commonly present in repos
    "requests==2.32.3",
    "packaging==24.2",
    "typing-extensions==4.12.2",
    # keep these pins aligned with env.yml (reduces drift)
    "protobuf==3.20.2",
    "tensorboard==2.19.0",
    "tensorboard-data-server==0.7.2",
    "grpcio==1.72.1",
    # optional but harmless (and env.yml had them)
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
        # build toolchain (freesasa / possible wheels fallback)
        "build-essential",
        "python3-dev",
        "pkg-config",
        # mdtraj / netcdf/hdf5 related stack (keeps mdtraj/netcdf robust)
        "gfortran",
        "libopenblas-dev",
        "liblapack-dev",
        "libhdf5-dev",
        "libnetcdf-dev",
        # compression libs for various wheels/extensions
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
    )
    .env({"PYTHONUNBUFFERED": "1", "PYTHONPATH": PPIFLOW_DIR})
    .run_commands(
        f"rm -rf {PPIFLOW_DIR} && git clone --depth 1 {PPIFLOW_REPO} {PPIFLOW_DIR}"
    )
    # torch/cu121 via pip extra index (fixes your uv unsatisfiable)
    .pip_install(*TORCH_PKGS, extra_index_url=PYTORCH_CU121_INDEX)
    # pyg via find-links
    .uv_pip_install(*PYG_PKGS, find_links=PYG_WHL)
    # rest
    .uv_pip_install(*INFER_PKGS)
)


# -------------------------
app = App(APP_NAME)

# -------------------------
# Task routing
# -------------------------
TASK_TO_SCRIPT = {
    # README: binder
    "binder": f"{PPIFLOW_DIR}/sample_binder.py",
    # README: antibody & nanobody (the same sample_antibody_nanobody.py)
    "antibody": f"{PPIFLOW_DIR}/sample_antibody_nanobody.py",
    "nanobody": f"{PPIFLOW_DIR}/sample_antibody_nanobody.py",
    # README: monomer unconditional / motif scaffolding (the same sample_monomer.py)
    "monomer": f"{PPIFLOW_DIR}/sample_monomer.py",
    "scaffolding": f"{PPIFLOW_DIR}/sample_monomer.py",
    # README: partial flow (antibody & nanobody)
    "ab_partial_flow": f"{PPIFLOW_DIR}/sample_antibody_nanobody_partial.py",
    "nb_partial_flow": f"{PPIFLOW_DIR}/sample_antibody_nanobody_partial.py",
    # README: partial flow (binder)
    "binder_partial_flow": f"{PPIFLOW_DIR}/sample_binder_partial.py",
}


# -------------------------
# Helpers
# -------------------------
def _tar_dir(src_dir: Path, out_tar_gz: Path) -> None:
    """Create tar.gz from a directory."""
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _rewrite_flag_value(args: list[str], flag: str, new_value: str) -> list[str]:
    """Replace occurrences of:  --flag VALUE
    If flag is missing, append it.
    """
    out: list[str] = []
    i = 0
    found = False
    while i < len(args):
        if args[i] == flag and i + 1 < len(args):
            out.extend([flag, new_value])
            i += 2
            found = True
            continue
        out.append(args[i])
        i += 1
    if not found:
        out.extend([flag, new_value])
    return out


def _remove_flag_and_value(args: list[str], flag: str) -> list[str]:
    """Remove occurrences of: --flag VALUE
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == flag and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _resolve_config_path(config_arg: str) -> Path:
    """Resolve config path inside container:
    - If absolute and exists: use it.
    - Else try /ppiflow/<config_arg>
    - Else try <config_arg> as relative to cwd.
    """
    p = Path(config_arg)
    if p.is_absolute() and p.exists():
        return p

    repo_guess = Path(PPIFLOW_DIR) / config_arg
    if repo_guess.exists():
        return repo_guess

    if p.exists():
        return p

    # fallthrough: return as-is for nicer error message upstream
    return p


def _resolve_model_ckpt_path(model_arg: str) -> Path:
    """If user passes 'models/binder.ckpt' or 'binder.ckpt', map to /models/<basename>.
    If user already passed an absolute /models/... path, keep it.
    """
    p = Path(model_arg)
    if str(p).startswith(str(MODELS_DIR)):
        return p
    return MODELS_DIR / p.name


def _write_effective_config(src_config: Path, dst_config: Path) -> None:
    """Create effective config yaml with forced override:
    model.use_deepspeed_evo_attention = False
    """
    import yaml

    cfg = yaml.safe_load(src_config.read_text()) or {}
    cfg.setdefault("model", {})
    cfg["model"]["use_deepspeed_evo_attention"] = False
    dst_config.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _collect_artifacts(run_dir: Path) -> None:
    """Best-effort copy of key files into run_dir/artifacts for quick inspection.
    We don't assume exact filenames, but we try to surface the usual suspects.
    """
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Common names
    want_exact = {"metrics.csv", "config.yml", "config.yaml"}
    for f in _iter_files(run_dir):
        if f.name in want_exact:
            dst = artifacts / f.name
            if dst.exists():
                continue
            dst.write_bytes(f.read_bytes())

    # Also collect any CSVs at shallow depth (often contains metrics)
    for f in _iter_files(run_dir):
        if f.suffix.lower() == ".csv":
            dst = artifacts / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())


# -------------------------
# Remote GPU job: unified runner (README-aligned)
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
    # ---------- partial flow antibody/nanobody (sample_antibody_nanobody_partial.py) ----------
    pf_fixed_positions: str | None,
    pf_cdr_position: str | None,
    pf_specified_hotspots: str | None,
    pf_start_t: float | None,
    pf_samples_per_target: int | None,
    pf_retry_limit: int,
    pf_antigen_chain: str | None,
    pf_heavy_chain: str | None,
    pf_light_chain: str | None,
    # ---------- partial flow binder (sample_binder_partial.py) ----------
    bpf_target_chain: str,
    bpf_binder_chain: str,
    bpf_start_t: float,
) -> bytes:
    import subprocess
    from pathlib import Path

    if task not in TASK_TO_SCRIPT:
        raise ValueError(f"Unknown task={task}. Choose from {sorted(TASK_TO_SCRIPT)}")

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

    # ---- resolve checkpoint ----
    if not model_weights:
        raise ValueError("--model-weights is required")
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
            raise FileNotFoundError(
                f"Config not found: {config} (resolved: {cfg_path})"
            )
        effective_config = run_dir / "effective_config.yaml"
        _write_effective_config(cfg_path, effective_config)

    def p_in(name: str) -> Path:
        """Return input path under inputs_dir; error if missing."""
        p = inputs_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")
        return p

    argv: list[str] = ["python", str(script)]

    # -------------------------
    # Task: binder (sample_binder.py)
    # expects: --input_pdb, --target_chain, --binder_chain, --config, --specified_hotspots,
    #          --samples_min_length, --samples_max_length, --samples_per_target,
    #          --model_weights, --output_dir, --name
    # -------------------------
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

    # -------------------------
    # Task: antibody / nanobody (sample_antibody_nanobody.py)
    # expects: --antigen_pdb, --framework_pdb, --antigen_chain, --heavy_chain,
    #          [--light_chain], [--specified_hotspots], [--cdr_length],
    #          --samples_per_target, --config, --model_weights, --output_dir, --name
    # -------------------------
    elif task in {"antibody", "nanobody"}:
        antigen_pdb = p_in("antigen.pdb")
        framework_pdb = p_in("framework.pdb")
        argv += [
            "--antigen_pdb",
            str(antigen_pdb),
            "--framework_pdb",
            str(framework_pdb),
        ]

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

    # -------------------------
    # Task: monomer unconditional (sample_monomer.py)
    # expects: --config, --model_weights, --output_dir, --length_subset, --samples_num
    # -------------------------
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

    # -------------------------
    # Task: motif scaffolding (sample_monomer.py)
    # expects: --config inference_scaffolding.yaml, --motif_csv, --motif_names, --samples_num
    # -------------------------
    elif task == "scaffolding":
        motif_csv = p_in("motif.csv")
        if effective_config:
            argv += ["--config", str(effective_config)]
        argv += [
            "--model_weights",
            str(model_ckpt),
            "--output_dir",
            str(outputs_dir),
            "--motif_csv",
            str(motif_csv),
        ]
        if scaffold_motif_names:
            argv += ["--motif_names", scaffold_motif_names]
        argv += ["--samples_num", str(scaffold_samples_num)]

    # -------------------------
    # Task: partial flow antibody/nanobody (sample_antibody_nanobody_partial.py)
    # expects: --complex_pdb, --fixed_positions, --cdr_position,
    #          [--specified_hotspots], --start_t, --samples_per_target, --output_dir,
    #          --retry_Limit, --config, --model_weights, --antigen_chain, --heavy_chain,
    #          [--light_chain], --name
    # -------------------------
    elif task in {"ab_partial_flow", "nb_partial_flow"}:
        complex_pdb = p_in("complex.pdb")

        # required args (README-aligned)
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

    # -------------------------
    # Task: partial flow binder (sample_binder_partial.py)
    # expects: --input_pdb, --config inference_binder_partial.yaml, --target_chain, --binder_chain,
    #          --start_t, --output_dir
    # -------------------------
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

    outputs_dir = run_dir / "outputs"
    pdbs = sorted(outputs_dir.glob("*.pdb"))

    if p.returncode != 0 and pdbs:
        # Generation produced outputs; metrics/test failed -> keep outputs
        # Optionally append note into stdout.log
        pass
    elif p.returncode != 0:
        raise RuntimeError(
            f"PPIFlow failed (exit {p.returncode}). See {run_dir}/stdout.log"
        )

    _collect_artifacts(run_dir)
    RUNS_VOL.commit()

    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"{task}.{run_name}.tar.gz"
        _tar_dir(run_dir, tar_path)
        return tar_path.read_bytes()


# -------------------------
# Local entrypoint: unified CLI (README-aligned, role-based uploads)
# -------------------------
@app.local_entrypoint()
def submit_ppiflow(
    # ---------- common ----------
    task: str = "binder",
    run_name: str = "test1",
    out_dir: str = "./ppiflow_outputs",
    # ---------- local input files ----------
    # binder / binder_partial_flow
    binder_input_pdb: str | None = None,
    # antibody/nanobody
    ab_antigen_pdb: str | None = None,
    ab_framework_pdb: str | None = None,
    # partial flow antibody/nanobody
    pf_complex_pdb: str | None = None,
    # scaffolding
    scaffold_motif_csv: str | None = None,
    # ---------- model weights ----------
    model_weights: str | None = None,
    # ---------- config (path or repo-relative under /ppiflow) ----------
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
) -> None:
    """Unified Modal CLI aligned with README sample_*.py CLIs.
    Upload is role-based (stable filenames) so remote runner doesn't guess ordering.
    """
    from pathlib import Path

    allowed = {
        "binder",
        "antibody",
        "nanobody",
        "monomer",
        "scaffolding",
        "ab_partial_flow",
        "nb_partial_flow",
        "binder_partial_flow",
    }
    if task not in allowed:
        raise ValueError(f"--task must be one of {sorted(allowed)}")

    def _read_file_as(role_name: str, path: str | None) -> tuple[str, bytes] | None:
        if not path:
            return None
        pp = Path(path).expanduser()
        if not pp.exists():
            raise FileNotFoundError(f"Local file not found: {pp}")
        return (role_name, pp.read_bytes())

    # ---- build role-based uploads ----
    input_files: list[tuple[str, bytes]] = []

    # binder / binder_partial_flow
    if task in {"binder", "binder_partial_flow"}:
        if not binder_input_pdb:
            raise ValueError(f"{task} requires --binder-input-pdb")
        item = _read_file_as("binder_input.pdb", binder_input_pdb)
        if item:
            input_files.append(item)

    # antibody/nanobody
    if task in {"antibody", "nanobody"}:
        if not ab_antigen_pdb or not ab_framework_pdb:
            raise ValueError(f"{task} requires --ab-antigen-pdb and --ab-framework-pdb")
        input_files.append(_read_file_as("antigen.pdb", ab_antigen_pdb))  # type: ignore[arg-type]
        input_files.append(_read_file_as("framework.pdb", ab_framework_pdb))  # type: ignore[arg-type]

    # partial flow antibody/nanobody
    if task in {"ab_partial_flow", "nb_partial_flow"}:
        if not pf_complex_pdb:
            raise ValueError(f"{task} requires --pf-complex-pdb")
        input_files.append(_read_file_as("complex.pdb", pf_complex_pdb))  # type: ignore[arg-type]

    # scaffolding motif csv (+ all motif_path pdbs)
    if task == "scaffolding":
        if not scaffold_motif_csv:
            raise ValueError("scaffolding requires --scaffold-motif-csv")

        import csv
        from io import StringIO

        csv_path = Path(scaffold_motif_csv).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Local file not found: {csv_path}")

        # 1) read and parse csv
        csv_text = csv_path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(StringIO(csv_text))
        required_cols = {"target", "length", "contig", "motif_path"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"motif.csv must have columns {sorted(required_cols)}, got {reader.fieldnames}"
            )

        rows = list(reader)
        if not rows:
            raise ValueError("motif.csv has no data rows")

        # 2) collect motif pdb files referenced by motif_path
        #    resolve relative paths relative to the CSV directory
        csv_dir = csv_path.parent
        motif_files: dict[str, Path] = {}
        for r in rows:
            mp = (r.get("motif_path") or "").strip()
            if not mp:
                raise ValueError(f"motif_path is empty in row: {r}")

            mp_path = Path(mp)
            if mp.startswith("~"):
                mp_path = Path(mp).expanduser()
            elif not mp_path.is_absolute():
                mp_path = (csv_dir / mp_path).resolve()

            if not mp_path.exists():
                raise FileNotFoundError(
                    f"motif_path file not found: {mp_path} (from motif_path={mp!r})"
                )

            # stable filename inside container inputs/
            stable_name = mp_path.name
            if stable_name in motif_files and motif_files[stable_name] != mp_path:
                stable_name = f"{mp_path.stem}.{len(motif_files) + 1}{mp_path.suffix}"

            motif_files[stable_name] = mp_path
            r["motif_path"] = stable_name  # rewrite row motif_path to stable name

        # 3) write rewritten csv (so container can resolve motif_path by filename in inputs/)
        out_buf = StringIO()
        writer = csv.DictWriter(out_buf, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        rewritten_csv_bytes = out_buf.getvalue().encode("utf-8")

        # 4) upload rewritten motif.csv + all motif pdbs
        input_files.append(("motif.csv", rewritten_csv_bytes))
        for stable_name, p in motif_files.items():
            input_files.append((stable_name, p.read_bytes()))

    if task == "scaffolding" and scaffold_motif_names:
        s = scaffold_motif_names.strip()
        if not s.startswith("["):
            import json

            scaffold_motif_names = json.dumps([s])

    # filter None (safety)
    input_files = [x for x in input_files if x is not None]  # type: ignore[comparison-overlap]

    # ---- dispatch to remote ----
    tar_bytes = run_ppiflow_structured.remote(
        task=task,
        run_name=run_name,
        input_files=input_files,
        model_weights=model_weights,
        config=config,
        # binder
        binder_target_chain=binder_target_chain,
        binder_binder_chain=binder_binder_chain,
        binder_specified_hotspots=binder_specified_hotspots,
        binder_samples_min_length=binder_samples_min_length,
        binder_samples_max_length=binder_samples_max_length,
        binder_samples_per_target=binder_samples_per_target,
        # antibody/nanobody
        ab_antigen_chain=ab_antigen_chain,
        ab_heavy_chain=ab_heavy_chain,
        ab_light_chain=ab_light_chain,
        ab_specified_hotspots=ab_specified_hotspots,
        ab_cdr_length=ab_cdr_length,
        ab_samples_per_target=ab_samples_per_target,
        # monomer
        mono_length_subset=mono_length_subset,
        mono_samples_num=mono_samples_num,
        # scaffolding
        scaffold_motif_names=scaffold_motif_names,
        scaffold_samples_num=scaffold_samples_num,
        # partial flow ab/nb
        pf_fixed_positions=pf_fixed_positions,
        pf_cdr_position=pf_cdr_position,
        pf_specified_hotspots=pf_specified_hotspots,
        pf_start_t=pf_start_t,
        pf_samples_per_target=pf_samples_per_target,
        pf_retry_limit=pf_retry_limit,
        pf_antigen_chain=pf_antigen_chain,
        pf_heavy_chain=pf_heavy_chain,
        pf_light_chain=pf_light_chain,
        # binder partial flow
        bpf_target_chain=bpf_target_chain,
        bpf_binder_chain=bpf_binder_chain,
        bpf_start_t=bpf_start_t,
    )

    out_dir_p = Path(out_dir).expanduser()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir_p / f"{task}.{run_name}.tar.gz"
    out_tar.write_bytes(tar_bytes)
    print(f"[ok] saved: {out_tar}")
