"""AF3Score source repo: <https://github.com/Mingchenchen/AF3Score>.

## Overview

- Modal wrapper around AF3Score for scoring existing protein structures.
- Input can be a single structure file or a directory of structures.
- Supports manual batch splitting for large jobs and shared-output resume runs.
- Preserves AF3Score-style output directories and generates the aggregate metrics CSV.
- Designed to be portable across servers: the image bootstraps AF3Score from a pinned remote repository instead of depending on a local checkout layout.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | **Required** | Path to a single structure file or a directory of structures. |
| `--output-dir-name` | `af3score_run` | Remote run directory name under the `af3score-outputs` Modal volume root. |
| `--output-dir` | not set | Optional local output directory used by the explicit download step. |
| `--local-output-dir-name` | value of `--output-dir-name` | Optional local folder name under `--output-dir`. |
| `--num-batches` | `1` | Total number of manual batches to split the sorted input set into. |
| `--batch-index` | `0` | Which batch to process when not using `--run-all-batches`. |
| `--run-all-batches`/`--no-run-all-batches` | `--no-run-all-batches` | Whether to submit all batches from one command. |
| `--download-after-run`/`--no-download-after-run` | `--no-download-after-run` | Whether to download results locally immediately after the remote run finishes. |
| `--download-only`/`--no-download-only` | `--no-download-only` | Skip remote computation and only download an already completed remote run. |
| `--download-archive-only`/`--no-download-archive-only` | `--no-download-archive-only` | Download the remote result bundle to a local `.tar.gz` file but do not extract it. |
| `--extract-only`/`--no-extract-only` | `--no-extract-only` | Skip remote calls and only extract an existing local `.tar.gz` bundle. |
| `--num-jobs` | `8` | Parallelism for `01_prepare_get_json.py`. |
| `--prepare-workers` | host CPU dependent | Worker count for preprocessing. |
| `--jax-workers` | `1` | Requested JAX worker count argument. Internal runtime currently constrains the actual heavy JAX prep worker count for stability. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `AF3Score` | Name of the Modal app. |
| `GPU` | `A100-80GB` | GPU type to request from Modal. |
| `TIMEOUT` | `86400` | Timeout for Modal functions in seconds. |
| `AF3SCORE_REPO_URL` | pinned GitHub repo URL | AF3Score repository URL cloned during image build. |
| `AF3SCORE_REPO_COMMIT` | pinned commit | AF3Score commit checked out during image build. |

## Input Support

- Supported suffixes: `.pdb`, `.ent`, `.cif`, `.mmcif`, `.pdbx`.
- Directory input is collected with a stable sorted order before batching.
- Resume behavior is based on the existence of official AF3Score output files for each structure.

## Outputs

- Outputs are persisted in the Modal volume `af3score-outputs`, mounted at `/mnt/data`.
- Each run writes to `/mnt/data/<output_dir_name>`.
- Official AF3Score per-structure directories are written under `/mnt/data/<output_dir_name>/af3score_outputs`.
- Aggregate metrics are written to `/mnt/data/<output_dir_name>/af3score_metrics.csv`.
- A copy of the aggregate metrics CSV is also written to `/mnt/data/<output_dir_name>/af3score_outputs/af3score_metrics.csv`.
- Local downloads intentionally exclude transient runtime directories such as `work/` and `metrics_view/`.

## Batching Model

- There are two batching layers in this wrapper.
- Outer batching is controlled by this script through `--num-batches`, `--batch-index`, and `--run-all-batches`.
- The outer batch layer is for large-job scheduling: splitting a big input set across multiple Modal jobs / GPUs, improving throughput, and enabling resume-friendly partial reruns.
- Default outer batching is:
  - `--num-batches=1`
  - `--batch-index=0`
  - `--no-run-all-batches`
- In that default mode, the command runs exactly one logical batch: "batch 0 out of 1", which means "process the whole input set together".
- If `--num-batches` is greater than `1` but `--run-all-batches` is still not set, only the selected `--batch-index` is processed.
- Inner batching happens inside upstream `01_prepare_get_json.py`.
- The inner batch layer groups samples by total complex length and writes them into subdirectories such as `batch_0_259`.
- Those inner subdirectories are then processed separately by downstream AF3Score steps.
- In practice:
  - outer batching = distributed workload management
  - inner batching = per-worker efficiency / bucketed AF3 execution

## CLI

Single structure:

```bash
modal run af3score_app.py \
  --input-dir test.pdb \
  --output-dir-name af3score_run_single
```

Batch directory:

```bash
modal run af3score_app.py \
  --input-dir test_pdbs \
  --output-dir-name af3score_run_batch \
  --num-batches 4 \
  --run-all-batches
```

Batch directory with local download:

```bash
modal run af3score_app.py \
  --input-dir test_pdbs \
  --output-dir-name af3score_run_batch \
  --output-dir . \
  --num-batches 4 \
  --run-all-batches
```

Single batch only:

```bash
modal run af3score_app.py \
  --input-dir test_pdbs \
  --output-dir-name af3score_run_batch \
  --num-batches 4 \
  --batch-index 0
```

"""

from __future__ import annotations

import io
import hashlib
import json
import os
import shlex
import shlex as shell_lex
import shutil
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# Modal app level defaults. Most of these can still be overridden by env vars.
GPU = os.environ.get("GPU", "A100-80GB")
TIMEOUT = int(os.environ.get("TIMEOUT", str(24 * 60 * 60)))
APP_NAME = os.environ.get("MODAL_APP", "AF3Score")

# Read AF3 model weights from a dedicated shared Modal volume.
AF3_MODEL_VOLUME_NAME = "ppiflow-models"
AF3_MODEL_VOLUME = Volume.from_name(AF3_MODEL_VOLUME_NAME)
AF3_MODEL_DIR = "/af3-models"
AF3_MODEL_WEIGHTS_CANDIDATES = (
    Path(AF3_MODEL_DIR) / "models" / "af3.bin",
    Path(AF3_MODEL_DIR) / "af3.bin",
)

OUTPUTS_VOLUME_NAME = "af3score-outputs"
OUTPUTS_VOLUME = Volume.from_name(
    OUTPUTS_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
# All persistent remote outputs live under the mounted volume root `/mnt/data`.
MOUNT_ROOT = Path("/mnt/data")
INPUT_STAGE_ROOT = MOUNT_ROOT / "af3score_inputs"
INPUT_STAGE_VOLUME_ROOT = Path("af3score_inputs")
DEFAULT_OUTPUT_DIR_NAME = "af3score_run"

# AF3Score source is cloned into the image at build time so this wrapper can run
# even on machines that do not have a local AF3Score checkout.
REPO_DIR = "/root/AF3Score"
REPO_ROOT = Path(REPO_DIR)
REPO_URL = os.environ.get(
    "AF3SCORE_REPO_URL",
    "https://github.com/Mingchenchen/AF3Score.git",
)
REPO_COMMIT = os.environ.get(
    "AF3SCORE_REPO_COMMIT",
    "b0764aaa4101f8a22a5f404faef7acc13ee52d06",
)
PDB_LIKE_SUFFIXES = {".pdb", ".ent"}
MMCIF_LIKE_SUFFIXES = {".cif", ".mmcif", ".pdbx"}
SUPPORTED_INPUT_SUFFIXES = PDB_LIKE_SUFFIXES | MMCIF_LIKE_SUFFIXES

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "ninja-build",
        "pkg-config",
        "zlib1g-dev",
    )
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    # Build a self-contained image: clone AF3Score, pin the repo commit, install
    # runtime deps, and precompute AF3 data assets during image build.
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        f"git clone {shlex.quote(REPO_URL)} {shlex.quote(REPO_DIR)}",
        f"git -C {shlex.quote(REPO_DIR)} checkout --detach {shlex.quote(REPO_COMMIT)}",
        f"cd {REPO_DIR} && pip install --no-deps -e .",
        (
            "pip install "
            "absl-py "
            "chex "
            "dm-haiku==0.0.13 "
            "dm-tree "
            "'jax[cuda12]==0.4.34' "
            "jax-triton==0.2.0 "
            "jaxtyping==0.2.34 "
            "rdkit==2024.3.5 "
            "tqdm "
            "triton==3.1.0 "
            "typeguard==2.13.3 "
            "zstandard "
            "biopython "
            "h5py "
            "pandas"
        ),
        f"cd {REPO_DIR} && build_data",
    )
    .workdir(REPO_DIR)
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def log_info(message: str) -> None:
    print(message, flush=True)


def _casefold_key(value: str) -> str:
    return value.casefold()


def _description_name(path: Path) -> str:
    return path.stem.casefold()


def split_batches(files: list[Path], num_batches: int, batch_index: int) -> list[Path]:
    # Stable outer batch slicing used by the CLI and remote workers. This is the
    # user-controlled job-level split for concurrency and resume behavior, not the
    # inner length-based batching done later by `01_prepare_get_json.py`.
    total = len(files)
    batch_size = (total + num_batches - 1) // num_batches
    start = batch_index * batch_size
    end = min(start + batch_size, total)
    return files[start:end]


def _run_root(output_dir_name: str) -> Path:
    # Root directory for one logical AF3Score run inside the output volume.
    return MOUNT_ROOT / output_dir_name


def _official_output_root(output_dir_name: str) -> Path:
    return _run_root(output_dir_name) / "af3score_outputs"


def _metrics_input_dir(output_dir_name: str) -> Path:
    return _run_root(output_dir_name) / "metric_inputs"


def _metrics_view_dir(output_dir_name: str) -> Path:
    return _run_root(output_dir_name) / "metrics_view"


def _metrics_csv_path(output_dir_name: str) -> Path:
    return _run_root(output_dir_name) / "af3score_metrics.csv"


def _metrics_csv_in_output_dir(output_dir_name: str) -> Path:
    return _official_output_root(output_dir_name) / "af3score_metrics.csv"


def _work_root(output_dir_name: str) -> Path:
    return _run_root(output_dir_name) / "work"


def _is_pdb_like(path: Path) -> bool:
    return path.suffix.lower() in PDB_LIKE_SUFFIXES


def _is_mmcif_like(path: Path) -> bool:
    return path.suffix.lower() in MMCIF_LIKE_SUFFIXES


def _collect_structure_files(input_dir: str) -> list[Path]:
    # Accept either a single structure file or a directory of structures.
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(
                "Unsupported structure file suffix. "
                f"Supported suffixes: {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}"
            )
        all_files = [root]
    else:
        all_files = sorted(
            [
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
            ],
            key=lambda path: path.name.casefold(),
        )

    if not all_files:
        raise ValueError(
            "No supported structure files were found in the provided input path. "
            f"Supported suffixes: {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}"
        )

    unique: dict[str, Path] = {}
    for structure_path in all_files:
        # Final result names are stem-based, so stems must stay unique.
        stem_key = _casefold_key(structure_path.stem)
        if stem_key in unique and unique[stem_key] != structure_path:
            raise ValueError(
                "Duplicate input structure stems are not supported because output names "
                "must stay stable across resume runs: "
                f"{unique[stem_key]} and {structure_path}"
            )
        unique[stem_key] = structure_path

    return sorted(unique.values(), key=lambda path: path.name.casefold())


def _ensure_local_output_base(path: str) -> Path:
    base = Path(path).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    if not os.access(base, os.W_OK):
        raise PermissionError(f"Local output path is not writable: {base}")
    return base


def _safe_extract_tar(tar_bytes: bytes, destination: Path) -> None:
    # Local extraction guard to avoid path traversal when unpacking Modal results.
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(f"{destination}{os.sep}") and member_path != destination:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _build_output_archive(output_dir_name: str) -> bytes:
    """Bundle final user-facing outputs for optional local download."""
    # Only package final user-facing artifacts for local download. Runtime helper
    # directories like `work/` and `metrics_view/` are intentionally excluded.
    run_root = _run_root(output_dir_name)
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as tar:
        include_paths = [
            _official_output_root(output_dir_name),
            _metrics_csv_path(output_dir_name),
        ]
        for include_path in include_paths:
            if include_path.exists():
                tar.add(include_path, arcname=str(Path(output_dir_name) / include_path.relative_to(run_root)))
    return archive_buffer.getvalue()


def _local_archive_path(output_dir: str, local_output_dir_name: str) -> Path:
    """Return the local tar.gz path used to cache downloaded remote outputs."""
    local_base = _ensure_local_output_base(output_dir)
    return local_base / f"{local_output_dir_name}.tar.gz"


def _write_local_archive(archive_path: Path, tar_bytes: bytes) -> Path:
    """Atomically write a downloaded tar.gz bundle to local disk."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive_path.parent,
        delete=False,
        suffix=".tar.gz.part",
    ) as handle:
        temp_path = Path(handle.name)
    try:
        temp_path.write_bytes(tar_bytes)
        temp_path.replace(archive_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return archive_path


def _extract_local_archive_to_directory(
    archive_path: Path,
    output_dir: str,
    local_output_dir_name: str,
    *,
    extracted_root_name: str,
) -> Path:
    """Extract a previously downloaded local archive into the final folder."""
    local_base = _ensure_local_output_base(output_dir)
    local_target = local_base / local_output_dir_name
    if local_target.exists():
        shutil.rmtree(local_target)
    _safe_extract_tar(archive_path.read_bytes(), local_base)
    extracted_root = local_base / extracted_root_name
    if extracted_root != local_target:
        extracted_root.replace(local_target)
    print(f"[LOCAL] archive_path: {archive_path}", flush=True)
    print(f"[LOCAL] output_dir: {local_target}", flush=True)
    return local_target


def _download_outputs_to_local(
    *,
    output_dir_name: str,
    output_dir: str,
    local_output_dir_name: str,
    extract_after_download: bool = True,
) -> Path:
    """Download an already completed remote run into a local directory."""
    # This helper intentionally allows a second independent "download-only" call
    # so local network / terminal interruptions do not require recomputing remote
    # AF3Score results.
    archive_path = _local_archive_path(output_dir, local_output_dir_name)
    bundle = af3score_download_bundle.remote(output_dir_name=output_dir_name)
    _write_local_archive(archive_path, bundle)
    print(f"[LOCAL] archive_path: {archive_path}", flush=True)
    if not extract_after_download:
        return archive_path
    return _extract_local_archive_to_directory(
        archive_path,
        output_dir,
        local_output_dir_name,
        extracted_root_name=output_dir_name,
    )


def _parse_mmcif_atom_site_rows(cif_path: Path) -> list[dict[str, str]]:
    # AF3Score preprocessing expects PDB-like inputs, so CIF files are converted
    # through a minimal atom_site parser rather than relying on extra local tools.
    lines = cif_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if line != "loop_":
            index += 1
            continue

        index += 1
        columns: list[str] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if candidate.startswith("_atom_site."):
                columns.append(candidate)
                index += 1
                continue
            break

        if not columns or not all(column.startswith("_atom_site.") for column in columns):
            continue

        rows: list[dict[str, str]] = []
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate == "#":
                break
            if candidate.startswith("loop_") or candidate.startswith("_"):
                break

            parts = shell_lex.split(candidate, posix=True)
            if len(parts) != len(columns):
                raise ValueError(
                    f"Unexpected atom_site row width in {cif_path}: "
                    f"expected {len(columns)} columns, got {len(parts)} in line `{candidate}`"
                )
            rows.append(dict(zip(columns, parts, strict=True)))
            index += 1

        if rows:
            return rows

    raise ValueError(f"Could not find a populated _atom_site loop in CIF file: {cif_path}")


def _sanitize_mmcif_value(value: str, default: str = "") -> str:
    if value in {".", "?", ""}:
        return default
    return value


def _format_pdb_atom_line(
    record_name: str,
    serial: int,
    atom_name: str,
    alt_loc: str,
    residue_name: str,
    chain_id: str,
    residue_seq: int,
    insertion_code: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    b_factor: float,
    element: str,
) -> str:
    atom_name = atom_name[:4]
    if len(atom_name) < 4:
        atom_name = atom_name.rjust(4)
    residue_name = residue_name[:3].rjust(3)
    chain_id = (chain_id or "A")[:1]
    alt_loc = (alt_loc or " ")[:1]
    insertion_code = (insertion_code or " ")[:1]
    element = (element or atom_name.strip()[:1]).strip()[:2].rjust(2)
    return (
        f"{record_name:<6}{serial:>5} "
        f"{atom_name}{alt_loc}"
        f"{residue_name} {chain_id}"
        f"{residue_seq:>4}{insertion_code}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{occupancy:>6.2f}{b_factor:>6.2f}          "
        f"{element:>2}"
    )


def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    # Convert one mmCIF structure into a simple PDB file that downstream scripts
    # in the AF3Score pipeline can consume.
    atom_rows = _parse_mmcif_atom_site_rows(cif_path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=pdb_path.parent,
        delete=False,
        suffix=".pdb",
    ) as handle:
        temp_path = Path(handle.name)

    try:
        current_model: str | None = None
        serial = 1
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in atom_rows:
                record_name = _sanitize_mmcif_value(row.get("_atom_site.group_PDB", "ATOM"), "ATOM").upper()
                if record_name not in {"ATOM", "HETATM"}:
                    continue

                model_num = _sanitize_mmcif_value(
                    row.get("_atom_site.pdbx_PDB_model_num", ""),
                    "1",
                )
                if current_model is None:
                    current_model = model_num
                if model_num != current_model:
                    break

                atom_name = _sanitize_mmcif_value(
                    row.get("_atom_site.auth_atom_id", row.get("_atom_site.label_atom_id", "")),
                    "X",
                )
                residue_name = _sanitize_mmcif_value(
                    row.get("_atom_site.auth_comp_id", row.get("_atom_site.label_comp_id", "")),
                    "UNK",
                )
                chain_id = _sanitize_mmcif_value(
                    row.get("_atom_site.auth_asym_id", row.get("_atom_site.label_asym_id", "")),
                    "A",
                )
                residue_seq_raw = _sanitize_mmcif_value(
                    row.get("_atom_site.auth_seq_id", row.get("_atom_site.label_seq_id", "")),
                    "0",
                )
                insertion_code = _sanitize_mmcif_value(
                    row.get("_atom_site.pdbx_PDB_ins_code", ""),
                    " ",
                )
                alt_loc = _sanitize_mmcif_value(
                    row.get("_atom_site.label_alt_id", ""),
                    " ",
                )
                element = _sanitize_mmcif_value(
                    row.get("_atom_site.type_symbol", ""),
                    atom_name.strip()[:1],
                )
                occupancy = float(
                    _sanitize_mmcif_value(row.get("_atom_site.occupancy", ""), "1.0")
                )
                b_factor = float(
                    _sanitize_mmcif_value(row.get("_atom_site.B_iso_or_equiv", ""), "20.0")
                )
                residue_seq = int(float(residue_seq_raw))
                x = float(_sanitize_mmcif_value(row.get("_atom_site.Cartn_x", ""), "0.0"))
                y = float(_sanitize_mmcif_value(row.get("_atom_site.Cartn_y", ""), "0.0"))
                z = float(_sanitize_mmcif_value(row.get("_atom_site.Cartn_z", ""), "0.0"))

                handle.write(
                    _format_pdb_atom_line(
                        record_name=record_name,
                        serial=serial,
                        atom_name=atom_name,
                        alt_loc=alt_loc,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_seq=residue_seq,
                        insertion_code=insertion_code,
                        x=x,
                        y=y,
                        z=z,
                        occupancy=occupancy,
                        b_factor=b_factor,
                        element=element,
                    )
                    + "\n"
                )
                serial += 1
            handle.write("END\n")
        temp_path.replace(pdb_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _extract_chain_ids_from_pdb(pdb_path: Path) -> list[str]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 22:
                continue
            chain_id = line[21].strip() or "_"
            if chain_id not in seen:
                seen.add(chain_id)
                chain_ids.append(chain_id)
    return chain_ids


def _coerce_optional_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_result_json(pdb_path: Path, sample_dir: Path) -> dict[str, object]:
    summary_path = sample_dir / "summary_confidences.json"
    confidences_path = sample_dir / "confidences.json"

    if not summary_path.exists() or not confidences_path.exists():
        raise FileNotFoundError(f"Missing AF3 output files: {sample_dir}")

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with confidences_path.open(encoding="utf-8") as handle:
        confidences = json.load(handle)

    chains = _extract_chain_ids_from_pdb(pdb_path)
    atom_plddts = confidences.get("atom_plddts", [])
    atom_chain_ids = confidences.get("atom_chain_ids", [])
    chain_ptm = summary.get("chain_ptm", [])
    chain_iptm = summary.get("chain_iptm", [])

    chain_metrics: dict[str, dict[str, float]] = {}
    for index, chain_id in enumerate(chains):
        chain_plddts = [
            float(plddt)
            for plddt, atom_chain_id in zip(atom_plddts, atom_chain_ids)
            if atom_chain_id == chain_id
        ]
        metrics: dict[str, float] = {}
        if chain_plddts:
            metrics["plddt"] = sum(chain_plddts) / len(chain_plddts)
        if index < len(chain_ptm):
            metrics["ptm"] = _coerce_optional_float(chain_ptm[index])
        if index < len(chain_iptm):
            metrics["iptm"] = _coerce_optional_float(chain_iptm[index])
        chain_metrics[chain_id] = metrics

    return {
        "description": pdb_path.stem,
        "source_file": pdb_path.name,
        "ptm": _coerce_optional_float(summary.get("ptm", 0.0)),
        "iptm": _coerce_optional_float(summary.get("iptm", 0.0)),
        "ranking_score": _coerce_optional_float(summary.get("ranking_score", 0.0)),
        "chains": chain_metrics,
        "summary_confidences": summary,
    }


def _modal_env() -> dict[str, str]:
    # JAX/XLA environment defaults used inside Modal containers.
    env = os.environ.copy()
    env.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    env.setdefault("XLA_CLIENT_MEM_FRACTION", "0.95")
    return env


def _commit_output_volume() -> None:
    # Best-effort commit so outputs are persisted promptly in the Modal volume.
    try:
        OUTPUTS_VOLUME.commit()
    except Exception as exc:  # pragma: no cover
        log_info(f"[WARN] Volume commit warning: {exc}")


def _resolve_model_weights_path() -> Path:
    # Support either `<volume>/models/af3.bin` or `<volume>/af3.bin`.
    for candidate in AF3_MODEL_WEIGHTS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "AF3 weights not found. Checked: "
        + ", ".join(str(path) for path in AF3_MODEL_WEIGHTS_CANDIDATES)
        + f". Expected volume `{AF3_MODEL_VOLUME_NAME}` to contain af3.bin."
    )


def run_command(cmd: list[str], **kwargs) -> None:
    # Stream subprocess output directly so long AF3 steps remain visible in logs.
    import subprocess as sp

    log_info(f"[CMD] {' '.join(shlex.quote(part) for part in cmd)}")
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("cwd", REPO_DIR)

    with sp.Popen(cmd, **kwargs) as process:
        if process.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (
            buffered_output := process.stdout.readline()
        ) != "" or process.poll() is None:
            print(buffered_output, end="", flush=True)

        if process.returncode != 0:
            raise sp.CalledProcessError(process.returncode, cmd, buffered_output)


def _write_output_json(output_file: Path, result: dict[str, object]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    tmp_file.rename(output_file)


def _write_error_file(err_file: Path, message: str) -> None:
    err_file.parent.mkdir(parents=True, exist_ok=True)
    with open(err_file, "w", encoding="utf-8") as f:
        f.write(message)


def _copy_atomic(source_path: Path, dest_path: Path) -> None:
    # Copy through a temp file so interrupted writes do not leave partial files.
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        delete=False,
        suffix=dest_path.suffix,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copy2(source_path, temp_path)
        temp_path.replace(dest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _official_sample_dir(output_dir: Path, source_path: Path) -> Path:
    return output_dir / _description_name(source_path) / "seed-10_sample-0"


def _official_summary_file(output_dir: Path, source_path: Path) -> Path:
    return _official_sample_dir(output_dir, source_path) / "summary_confidences.json"


def _official_confidences_file(output_dir: Path, source_path: Path) -> Path:
    return _official_sample_dir(output_dir, source_path) / "confidences.json"


def _ensure_metric_input_pdb(source_path: Path, metrics_input_dir: Path) -> Path:
    # `04_get_metrics.py` expects a PDB for each description, so CIF-like inputs
    # are normalized into a stable per-run PDB cache.
    metrics_input_dir.mkdir(parents=True, exist_ok=True)
    metric_pdb = metrics_input_dir / f"{_description_name(source_path)}.pdb"
    if metric_pdb.exists():
        return metric_pdb

    if _is_pdb_like(source_path):
        _copy_atomic(source_path, metric_pdb)
        return metric_pdb

    temp_dir = metrics_input_dir / ".tmp_metric_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    staged_cif = temp_dir / source_path.name
    _copy_atomic(source_path, staged_cif)
    _convert_cif_to_pdb(staged_cif, metric_pdb)
    if staged_cif.exists():
        staged_cif.unlink()
    return metric_pdb


def _materialize_batch_inputs(
    source_files: list[Path],
    pending_input_dir: Path,
    metrics_input_dir: Path,
) -> list[Path]:
    # Batch preprocessing always receives PDB files, regardless of original input
    # format, to keep the downstream AF3Score scripts simple and consistent.
    pending_input_dir.mkdir(parents=True, exist_ok=True)
    pending_pdbs: list[Path] = []
    for source_path in source_files:
        metric_pdb = _ensure_metric_input_pdb(source_path, metrics_input_dir)
        staged_pdb = pending_input_dir / f"{source_path.stem}.pdb"
        shutil.copy2(metric_pdb, staged_pdb)
        pending_pdbs.append(staged_pdb)

    return sorted(pending_pdbs, key=lambda path: path.name.casefold())


def _collect_pending_files(batch_files: list[Path], output_dir: Path) -> tuple[list[Path], int]:
    # Resume logic: if official AF3Score outputs already exist for a structure,
    # skip recomputing it.
    pending_files: list[Path] = []
    skipped = 0
    for pdb in batch_files:
        summary_file = _official_summary_file(output_dir, pdb)
        confidences_file = _official_confidences_file(output_dir, pdb)
        if summary_file.exists() and confidences_file.exists():
            print(f"[SKIP] {pdb.name}", flush=True)
            skipped += 1
            continue
        print(f"[BATCH] Processing {pdb.name}", flush=True)
        pending_files.append(pdb)
    return pending_files, skipped


def _run_batch_inference(
    pending_files: list[Path],
    model_weights_path: Path,
    batch_index: int,
    output_dir_name: str,
    num_jobs: int,
    prepare_workers: int,
    jax_workers: int,
) -> tuple[Path, list[Path]]:
    # One remote outer batch runs the original AF3Score preprocessing + inference
    # steps inside a private working directory, but writes final official outputs
    # into a shared run output root. Inside this outer batch, upstream
    # `01_prepare_get_json.py` will still create additional inner length-based
    # batch directories for efficiency.
    batch_root = _work_root(output_dir_name) / f"batch_{batch_index}"
    if batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)

    env = _modal_env()
    pending_input_dir = batch_root / "inputs"
    output_dir_cif = batch_root / "single_chain_cif"
    save_csv = batch_root / "single_seq.csv"
    output_dir_json = batch_root / "json"
    batch_dir = batch_root / "af3_input_batch"
    output_dir_jax = batch_root / "jax"
    af3_output_dir = _official_output_root(output_dir_name)
    # Keep the heavy JAX preprocessing stage conservative for stability.
    effective_jax_workers = 1

    pending_pdbs = _materialize_batch_inputs(
        pending_files,
        pending_input_dir,
        _metrics_input_dir(output_dir_name),
    )

    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "01_prepare_get_json.py"),
            "--input_dir",
            str(pending_input_dir),
            "--output_dir_cif",
            str(output_dir_cif),
            "--save_csv",
            str(save_csv),
            "--output_dir_json",
            str(output_dir_json),
            "--batch_dir",
            str(batch_dir),
            "--num_jobs",
            str(max(1, num_jobs)),
            "--num_workers",
            str(max(1, prepare_workers)),
        ],
        env=env,
    )

    batch_pdb_root = batch_dir / "pdb"
    batch_json_root = batch_dir / "json"
    if not batch_pdb_root.exists() or not batch_json_root.exists():
        raise RuntimeError("Batch directories were not created by 01_prepare_get_json.py.")

    for batch_pdb_dir in sorted(path for path in batch_pdb_root.iterdir() if path.is_dir()):
        batch_h5_dir = output_dir_jax / batch_pdb_dir.name
        batch_h5_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "02_prepare_pdb2jax.py"),
                "--pdb_folder",
                str(batch_pdb_dir),
                "--output_folder",
                str(batch_h5_dir),
                "--num_workers",
                str(effective_jax_workers),
            ],
            env=env,
        )

    batch_failures: list[str] = []
    for batch_json_dir in sorted(path for path in batch_json_root.iterdir() if path.is_dir()):
        batch_h5_dir = output_dir_jax / batch_json_dir.name
        bucket = batch_json_dir.name.rsplit("_", 1)[-1]
        if not batch_h5_dir.exists():
            raise FileNotFoundError(f"Missing H5 batch directory: {batch_h5_dir}")

        try:
            run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_af3score.py"),
                    "--model_dir",
                    str(model_weights_path.parent),
                    "--batch_json_dir",
                    str(batch_json_dir),
                    "--batch_h5_dir",
                    str(batch_h5_dir),
                    "--output_dir",
                    str(af3_output_dir),
                    "--run_data_pipeline=False",
                    "--run_inference=true",
                    "--init_guess=true",
                    "--num_samples=1",
                    f"--buckets={bucket}",
                    "--write_cif_model=False",
                    "--write_summary_confidences=true",
                    "--write_full_confidences=true",
                    "--write_best_model_root=false",
                    "--write_ranking_scores_csv=false",
                    "--write_terms_of_use_file=false",
                    "--write_fold_input_json_file=false",
                ],
                env=env,
            )
        except Exception as exc:
            batch_failures.append(f"{batch_json_dir.name}: {exc}")
            print(f"[ERROR] Batch chunk failed: {batch_json_dir.name}: {exc}", flush=True)

    if batch_failures:
        raise RuntimeError("; ".join(batch_failures))

    return af3_output_dir, pending_pdbs


def _build_metrics_view(output_dir: Path, metrics_view_dir: Path) -> list[str]:
    # `04_get_metrics.py` expects a directory of per-description subfolders, so a
    # temporary view is created that mirrors the official AF3Score output layout.
    if metrics_view_dir.exists():
        shutil.rmtree(metrics_view_dir)
    metrics_view_dir.mkdir(parents=True, exist_ok=True)

    descriptions: list[str] = []
    for candidate in sorted(output_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not candidate.is_dir():
            continue
        sample_dir = candidate / "seed-10_sample-0"
        if not sample_dir.exists():
            continue
        view_path = metrics_view_dir / candidate.name
        view_path.symlink_to(candidate, target_is_directory=True)
        descriptions.append(candidate.name)
    return descriptions


def _write_metrics_csv(output_dir_name: str) -> dict[str, int | str]:
    """Generate the aggregate AF3Score metrics CSV for one logical run."""
    # Final aggregation step: scan all completed AF3Score outputs and build the
    # run-level metrics CSV expected by the official AF3Score pipeline.
    OUTPUTS_VOLUME.reload()
    output_dir = _official_output_root(output_dir_name)
    metrics_input_dir = _metrics_input_dir(output_dir_name)
    metrics_view_dir = _metrics_view_dir(output_dir_name)
    metrics_csv = _metrics_csv_path(output_dir_name)
    metrics_csv_in_output_dir = _metrics_csv_in_output_dir(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_input_dir.mkdir(parents=True, exist_ok=True)

    descriptions = _build_metrics_view(output_dir, metrics_view_dir)
    if not descriptions:
        return {
            "metrics_csv": str(metrics_csv),
            "metrics_csv_in_output_dir": str(metrics_csv_in_output_dir),
            "metrics_rows": 0,
            "failed_records_txt": str(output_dir / "failed_records.txt"),
        }

    temp_metrics_csv = metrics_csv.with_suffix(".tmp")
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "04_get_metrics.py"),
            "--input_pdb_dir",
            str(metrics_input_dir),
            "--af3score_output_dir",
            str(metrics_view_dir),
            "--save_metric_csv",
            str(temp_metrics_csv),
            "--num_workers",
            str(max(1, min(16, os.cpu_count() or 4))),
        ],
    )
    temp_metrics_csv.replace(metrics_csv)
    shutil.copy2(metrics_csv, metrics_csv_in_output_dir)
    _commit_output_volume()

    metrics_rows = 0
    with metrics_csv.open(encoding="utf-8") as handle:
        metrics_rows = max(0, sum(1 for _ in handle) - 1)

    return {
        "metrics_csv": str(metrics_csv),
        "metrics_csv_in_output_dir": str(metrics_csv_in_output_dir),
        "metrics_rows": metrics_rows,
        "failed_records_txt": str(output_dir / "failed_records.txt"),
    }


##########################################
# Inference function
##########################################
@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={
        "/mnt/data": OUTPUTS_VOLUME,
        AF3_MODEL_DIR: AF3_MODEL_VOLUME.read_only(),
    },
    image=runtime_image,
)
def af3score_run(
    staged_input_dir: str,
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    num_batches: int = 1,
    batch_index: int = 0,
    num_jobs: int = 8,
    prepare_workers: int = 8,
    jax_workers: int = 1,
) -> dict[str, int | str]:
    """Run one remote AF3Score batch and validate official outputs per structure."""
    # Remote worker for exactly one batch.
    OUTPUTS_VOLUME.reload()
    model_weights_path = _resolve_model_weights_path()

    staged_dir = Path(staged_input_dir)
    if not staged_dir.exists():
        raise FileNotFoundError(
            f"Staged input directory not found: {staged_dir}. "
            "Run this script through `modal run af3score_app.py ...` so the local entrypoint can upload inputs."
        )

    output_dir = _official_output_root(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = output_dir / "failed_records"
    failed_dir.mkdir(exist_ok=True)
    _metrics_input_dir(output_dir_name).mkdir(parents=True, exist_ok=True)
    _work_root(output_dir_name).mkdir(parents=True, exist_ok=True)

    all_files = sorted(
        [
            path
            for path in staged_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        ],
        key=lambda path: path.name.casefold(),
    )

    print(f"[INFO] Total files: {len(all_files)}", flush=True)
    print(f"[INFO] Batch {batch_index}/{num_batches}", flush=True)
    print(f"[INFO] Processing {len(all_files)} files", flush=True)
    print(f"[INFO] Output root: {_run_root(output_dir_name)}", flush=True)

    for source_path in all_files:
        # Ensure metric extraction inputs exist before any skip/resume logic.
        _ensure_metric_input_pdb(source_path, _metrics_input_dir(output_dir_name))

    processed = 0
    skipped = 0
    failed = 0
    total_done = 0

    pending_files, skipped = _collect_pending_files(all_files, output_dir)
    if not pending_files:
        print(
            f"[INFO] Batch {batch_index} already complete: {skipped}/{len(all_files)} done",
            flush=True,
        )
        _commit_output_volume()
        return {
            "output_dir": str(output_dir),
            "failed_dir": str(failed_dir),
            "total": len(all_files),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        }

    af3_output_dir = None
    pending_pdbs: list[Path] = []
    batch_exception: Exception | None = None

    try:
        af3_output_dir, pending_pdbs = _run_batch_inference(
            pending_files=pending_files,
            model_weights_path=model_weights_path,
            batch_index=batch_index,
            output_dir_name=output_dir_name,
            num_jobs=num_jobs,
            prepare_workers=prepare_workers,
            jax_workers=jax_workers,
        )
    except Exception as exc:
        # If a batch-level failure happens, keep going and record per-structure
        # failures instead of aborting the whole run.
        batch_exception = exc
        fallback_input_dir = _work_root(output_dir_name) / f"batch_{batch_index}" / "inputs"
        if fallback_input_dir.exists():
            pending_pdbs = sorted(fallback_input_dir.glob("*.pdb"), key=lambda path: path.name.casefold())
        af3_output_dir = _official_output_root(output_dir_name)

    for source_path in pending_files:
        err_file = failed_dir / f"{source_path.stem}.err"
        sample_dir = None if af3_output_dir is None else _official_sample_dir(af3_output_dir, source_path)

        try:
            if sample_dir is None:
                raise RuntimeError("Batch inference did not produce a readable sample directory.")
            summary_path = sample_dir / "summary_confidences.json"
            confidences_path = sample_dir / "confidences.json"
            if not summary_path.exists() or not confidences_path.exists():
                raise FileNotFoundError(f"Missing AF3 output files: {sample_dir}")
            if err_file.exists():
                err_file.unlink()
            processed += 1
            total_done = skipped + processed + failed
            print(
                f"[PROGRESS] Batch {batch_index}: completed {total_done}/{len(all_files)} "
                f"(processed={processed}, skipped={skipped}, failed={failed})",
                flush=True,
            )
        except Exception as exc:
            message = str(exc)
            if batch_exception is not None:
                message = f"{message}\nBatch error: {batch_exception}"
            _write_error_file(err_file, message)
            failed += 1
            total_done = skipped + processed + failed
            print(
                f"[PROGRESS] Batch {batch_index}: completed {total_done}/{len(all_files)} "
                f"(processed={processed}, skipped={skipped}, failed={failed})",
                flush=True,
            )

    batch_root = _work_root(output_dir_name) / f"batch_{batch_index}"
    if batch_root.exists():
        shutil.rmtree(batch_root)

    print(
        f"[INFO] Finished batch_{batch_index}: processed={processed}, "
        f"skipped={skipped}, failed={failed}, total={len(all_files)}",
        flush=True,
    )
    _commit_output_volume()
    return {
        "output_root": str(_run_root(output_dir_name)),
        "output_dir": str(output_dir),
        "failed_dir": str(failed_dir),
        "metrics_input_dir": str(_metrics_input_dir(output_dir_name)),
        "total": len(all_files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }


@app.function(
    timeout=TIMEOUT,
    volumes={
        "/mnt/data": OUTPUTS_VOLUME,
        AF3_MODEL_DIR: AF3_MODEL_VOLUME.read_only(),
    },
    image=runtime_image,
)
def af3score_finalize(output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME) -> dict[str, int | str]:
    """Finalize one run by writing the aggregate metrics CSV."""
    # Run the official metrics aggregation after all requested batches finish.
    return _write_metrics_csv(output_dir_name)


@app.function(
    timeout=TIMEOUT,
    volumes={
        "/mnt/data": OUTPUTS_VOLUME,
        AF3_MODEL_DIR: AF3_MODEL_VOLUME.read_only(),
    },
    image=runtime_image,
)
def af3score_download_bundle(output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME) -> bytes:
    """Return a compressed archive of final outputs for local extraction."""
    # Build a small download bundle containing only final user-facing outputs.
    OUTPUTS_VOLUME.reload()
    return _build_output_archive(output_dir_name)


##########################################
# Local entrypoint
##########################################
@app.local_entrypoint()
def submit_af3score_task(
    input_dir: str = "",  # Required.
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,  # Remote folder name.
    output_dir: str = "",  # Local output base dir.
    local_output_dir_name: str = "",  # Local folder name.
    num_batches: int = 1,  # Total outer batches.
    batch_index: int = 0,  # Current outer batch.
    num_jobs: int = 8,  # Preprocess jobs.
    prepare_workers: int = max(1, (os.cpu_count() or 4) - 2),  # Preprocess CPU workers.
    jax_workers: int = 1,  # Requested JAX prep workers.
    run_all_batches: bool = False,  # Submit all batches.
    download_after_run: bool = False,  # Run, then download.
    download_only: bool = False,  # Download and extract only.
    download_archive_only: bool = False,  # Download archive only.
    extract_only: bool = False,  # Extract local archive only.
) -> None:
    """Stage local inputs, launch remote AF3Score work, and optionally download results.

    This local CLI entrypoint is the main place where user-facing defaults matter.

    Default behavior:
    - `input_dir` must still be provided explicitly.
    - `output_dir_name` defaults to `af3score_run`.
    - `num_batches=1`, `batch_index=0`, and `run_all_batches=False` mean
      "run one logical batch containing the full input set".
    - `num_jobs=8` controls preprocessing job splitting inside
      `01_prepare_get_json.py`; it is not a GPU count.
    - `prepare_workers` controls preprocessing CPU workers.
    - `jax_workers` is the requested JAX-prep worker count, although the remote
      runtime currently constrains the heavy JAX prep step for stability.
    - `download_after_run=False`, `download_only=False`,
      `download_archive_only=False`, and `extract_only=False` mean the default
      command only runs remotely and writes outputs to the Modal volume.
    - Passing `--output-dir` alone does not trigger a local download; it only
      defines the local destination for explicit download / extraction modes.

    Practical guidance:
    - For a single PDB or a very small directory, the default preprocessing
      parallelism is safe but more generous than necessary.
    - For large directories, the defaults are a more reasonable starting point.
    """
    # Local CLI entrypoint. It stages inputs into the Modal volume, launches one
    # or many batch workers, finalizes metrics, and optionally downloads results.
    #
    # Important behavior:
    # - `num_batches`, `batch_index`, and `run_all_batches` control the outer,
    #   user-visible batch layer used for multi-GPU / multi-job scheduling.
    # - `num_jobs`, `prepare_workers`, and `jax_workers` control per-batch
    #   preprocessing parallelism inside one remote worker; they do not request
    #   additional GPUs by themselves.
    # - By default, the command focuses on finishing remote work and exits without
    #   forcing a local download step.
    # - In other words, `--output-dir` is not enough on its own to produce a
    #   local folder; an explicit download or extraction mode is required.
    # - `--download-after-run` keeps the old "run then download immediately"
    #   behavior in one command.
    # - `--download-only` is the recovery-friendly mode: it downloads an already
    #   completed remote run later, without rerunning AF3Score.
    output_dir_name = output_dir_name.strip().strip("/")
    if not output_dir_name:
        raise ValueError("`--output-dir-name` must not be empty.")
    local_output_dir_name = (local_output_dir_name or output_dir_name).strip().strip("/")
    if output_dir and not local_output_dir_name:
        raise ValueError("`--local-output-dir-name` must not be empty when `--output-dir` is set.")
    if download_only and not output_dir:
        raise ValueError("`--download-only` requires `--output-dir`.")
    if download_after_run and not output_dir:
        raise ValueError("`--download-after-run` requires `--output-dir`.")
    if download_archive_only and not output_dir:
        raise ValueError("`--download-archive-only` requires `--output-dir`.")
    if extract_only and not output_dir:
        raise ValueError("`--extract-only` requires `--output-dir`.")
    mode_flags = [download_only, download_archive_only, extract_only]
    if sum(bool(flag) for flag in mode_flags) > 1:
        raise ValueError(
            "`--download-only`, `--download-archive-only`, and `--extract-only` are mutually exclusive."
        )
    if download_only and input_dir:
        print("[INFO] `--download-only` ignores `--input-dir` and only fetches existing outputs.", flush=True)
    if download_archive_only and input_dir:
        print(
            "[INFO] `--download-archive-only` ignores `--input-dir` and only fetches an existing remote bundle.",
            flush=True,
        )
    if extract_only and input_dir:
        print("[INFO] `--extract-only` ignores `--input-dir` and only unpacks a local archive.", flush=True)
    if download_only:
        # Fast path for post hoc retrieval of a completed remote run.
        _download_outputs_to_local(
            output_dir_name=output_dir_name,
            output_dir=output_dir,
            local_output_dir_name=local_output_dir_name,
            extract_after_download=True,
        )
        return
    if download_archive_only:
        _download_outputs_to_local(
            output_dir_name=output_dir_name,
            output_dir=output_dir,
            local_output_dir_name=local_output_dir_name,
            extract_after_download=False,
        )
        return
    if extract_only:
        archive_path = _local_archive_path(output_dir, local_output_dir_name)
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Local archive not found for extraction: {archive_path}. "
                "Run with `--download-archive-only` or `--download-only` first."
            )
        _extract_local_archive_to_directory(
            archive_path,
            output_dir,
            local_output_dir_name,
            extracted_root_name=output_dir_name,
        )
        return
    if num_batches < 1:
        raise ValueError("`--num-batches` must be >= 1.")
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError("`--batch-index` must satisfy 0 <= batch_index < num_batches.")
    if not input_dir:
        raise ValueError("`--input-dir` is required unless `--download-only` is set.")

    all_files = sorted(_collect_structure_files(input_dir), key=lambda path: path.name.casefold())
    print(f"[INFO] Total files: {len(all_files)}", flush=True)
    print(f"[INFO] Output root: {_run_root(output_dir_name)}", flush=True)
    dataset_hash = hashlib.sha1(str(Path(input_dir).expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    upload_id = uuid.uuid4().hex

    batch_specs: list[tuple[int, list[Path], Path, Path]] = []
    if run_all_batches:
        # Precompute staging locations for every batch so they can all be uploaded
        # once and then launched in parallel.
        for current_batch_index in range(num_batches):
            batch_files = split_batches(all_files, num_batches, current_batch_index)
            stage_root = (
                INPUT_STAGE_VOLUME_ROOT
                / dataset_hash
                / f"run_{upload_id}"
                / f"batch_{current_batch_index}_of_{num_batches}"
                / "inputs"
            )
            remote_stage_root = MOUNT_ROOT / stage_root
            batch_specs.append((current_batch_index, batch_files, stage_root, remote_stage_root))
    else:
        batch_files = split_batches(all_files, num_batches, batch_index)
        stage_root = (
            INPUT_STAGE_VOLUME_ROOT
            / dataset_hash
            / f"run_{upload_id}"
            / f"batch_{batch_index}_of_{num_batches}"
            / "inputs"
        )
        remote_stage_root = MOUNT_ROOT / stage_root
        batch_specs.append((batch_index, batch_files, stage_root, remote_stage_root))

    with OUTPUTS_VOLUME.batch_upload(force=True) as batch:
        for current_batch_index, batch_files, stage_root, _ in batch_specs:
            print(f"[INFO] Batch {current_batch_index}/{num_batches}", flush=True)
            print(f"[INFO] Processing {len(batch_files)} files", flush=True)
            for pdb_path in batch_files:
                batch.put_file(str(pdb_path), str(stage_root / pdb_path.name))

    if run_all_batches:
        # Submit all batches concurrently, then wait for each result.
        function_calls: list[tuple[int, object]] = []
        for current_batch_index, _, _, remote_stage_root in batch_specs:
            function_call = af3score_run.spawn(
                staged_input_dir=str(remote_stage_root),
                output_dir_name=output_dir_name,
                num_batches=num_batches,
                batch_index=current_batch_index,
                num_jobs=max(1, num_jobs),
                prepare_workers=max(1, prepare_workers),
                jax_workers=max(1, jax_workers),
            )
            function_calls.append((current_batch_index, function_call))

        for current_batch_index, function_call in function_calls:
            result = function_call.get()
            print(f"[RESULT] batch={current_batch_index} {result}", flush=True)
            print(f"[INFO] finished batch_{current_batch_index}", flush=True)
    else:
        result = af3score_run.remote(
            staged_input_dir=str(remote_stage_root),
            output_dir_name=output_dir_name,
            num_batches=num_batches,
            batch_index=batch_index,
            num_jobs=max(1, num_jobs),
            prepare_workers=max(1, prepare_workers),
            jax_workers=max(1, jax_workers),
        )

        for key, value in result.items():
            print(f"[RESULT] {key}: {value}", flush=True)
        print(f"[INFO] finished batch_{batch_index}", flush=True)

    metrics_result = af3score_finalize.remote(output_dir_name=output_dir_name)
    for key, value in metrics_result.items():
        print(f"[METRICS] {key}: {value}", flush=True)

    total_processed = metrics_result.get("metrics_rows")
    if isinstance(total_processed, int):
        print(f"[INFO] {total_processed}/{len(all_files)} done", flush=True)

    if download_after_run:
        # Optional local download step after remote processing has finished. This
        # is convenient when the client connection is stable and immediate local
        # access to outputs is desired.
        _download_outputs_to_local(
            output_dir_name=output_dir_name,
            output_dir=output_dir,
            local_output_dir_name=local_output_dir_name,
            extract_after_download=True,
        )
    elif output_dir:
        # If a local output directory was provided but immediate download was not
        # requested, print an explicit recovery-friendly follow-up command.
        print(
            "[INFO] Remote run finished. Local download was skipped so the command can exit "
            "quickly. Re-download later with either: "
            f"`modal run af3score_app.py --output-dir-name {shlex.quote(output_dir_name)} "
            f"--output-dir {shlex.quote(output_dir)} --local-output-dir-name "
            f"{shlex.quote(local_output_dir_name)} --download-only` "
            "or download the archive first with "
            f"`modal run af3score_app.py --output-dir-name {shlex.quote(output_dir_name)} "
            f"--output-dir {shlex.quote(output_dir)} --local-output-dir-name "
            f"{shlex.quote(local_output_dir_name)} --download-archive-only` "
            "and extract it later with "
            f"`modal run af3score_app.py --output-dir-name {shlex.quote(output_dir_name)} "
            f"--output-dir {shlex.quote(output_dir)} --local-output-dir-name "
            f"{shlex.quote(local_output_dir_name)} --extract-only`",
            flush=True,
        )
