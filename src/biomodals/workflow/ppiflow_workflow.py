"""Modal workflow orchestrator for the PPIFlow two-stage design pipeline.

This workflow preserves the upstream PPIFlow scheduler inputs:

    modal run src/biomodals/workflow/ppiflow_workflow.py \
      --task task.yaml \
      --steps steps.yaml \
      --stage 1

It calls already deployed Biomodals apps with ``modal.Function.from_name`` and
packages a local ``.tar.zst`` containing the canonical upstream layout:
``stage1/``, ``stage2/``, and ``design_output/``.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import modal
import yaml
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import package_outputs, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": "workflow"},
    name="PPIFlowWorkflow",
    repo_url="https://github.com/Mingchenchen/PPIFlow",
    package_name="biomodals-ppiflow-workflow",
    version="0.1.0",
    python_version="3.12",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)
OUT_VOLUME = CONF.get_out_volume()
MOUNT_ROOT = Path(CONF.output_volume_mountpoint)

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version).env(CONF.default_env)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Data models and constants
##########################################
STAGE1_STEPS = (
    "PPIFlowStep",
    "MPNNStep_stage1",
    "AbMPNNStep_stage1",
    "FlowpackerStep_stage1",
    "AF3scoreStep_stage1",
    "FilterStep_stage1",
)
STAGE2_STEPS = (
    "RosettaFixStep",
    "PartialStep",
    "MPNNStep_stage2",
    "AbMPNNStep_stage2",
    "FlowpackerStep_stage2",
    "AF3scoreStep_stage2",
    "FilterStep_stage2",
    "ReFoldStep",
    "DockQStep",
    "RosettaRelaxStep",
    "RankStep",
    "ReportStep",
)
STRUCTURE_SUFFIXES = {".pdb", ".cif"}
LOCAL_INPUT_SUFFIXES = {".pdb", ".cif", ".csv", ".json", ".yaml", ".yml", ".txt"}


@dataclass(frozen=True)
class WorkflowState:
    """Paths and config shared across workflow steps."""

    run_root: Path
    run_name: str
    task: dict[str, Any]
    enabled: dict[str, bool]
    steps: dict[str, Any]
    app_names: dict[str, str]
    max_workers: int

    @property
    def gentype(self) -> str:
        """Return the upstream PPIFlow design type."""
        return str(self.task.get("gentype") or self.task.get("design_mode") or "binder")

    @property
    def stage1_dir(self) -> Path:
        """Return the canonical Stage 1 output directory."""
        return self.run_root / "stage1"

    @property
    def stage2_dir(self) -> Path:
        """Return the canonical Stage 2 output directory."""
        return self.run_root / "stage2"

    @property
    def design_output_dir(self) -> Path:
        """Return the canonical final design output directory."""
        return self.run_root / "design_output"


##########################################
# Generic helpers
##########################################
def _env_app_name(*keys: str, default: str) -> str:
    for key in keys:
        if value := os.environ.get(key):
            return value
    return default


def _default_app_names() -> dict[str, str]:
    return {
        "ppiflow": _env_app_name("PPIFLOW_APP_NAME", "PPIFLOW_APP", default="PPIFlow"),
        "ligandmpnn": _env_app_name(
            "LIGANDMPNN_APP_NAME", "LIGANDMPNN_APP", default="LigandMPNN"
        ),
        "flowpacker": _env_app_name(
            "FLOWPACKER_APP_NAME", "FLOWPACKER_APP", default="FlowPacker"
        ),
        "af3score": _env_app_name(
            "AF3SCORE_APP_NAME", "AF3SCORE_APP", default="AF3Score"
        ),
        "alphafold3": _env_app_name(
            "ALPHAFOLD3_APP_NAME", "ALPHAFOLD3_APP", default="AlphaFold3"
        ),
        "rosetta": _env_app_name("ROSETTA_APP_NAME", "ROSETTA_APP", default="Rosetta"),
        "dockq": _env_app_name("DOCKQ_APP_NAME", "DOCKQ_APP", default="DockQ"),
    }


def _remote_function(app_names: dict[str, str], key: str, function_name: str):
    return modal.Function.from_name(app_names[key], function_name)


def _load_yaml_bytes(data: bytes) -> dict[str, Any]:
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return loaded


def _task_section(task_doc: dict[str, Any]) -> dict[str, Any]:
    section = task_doc.get("task", task_doc)
    if not isinstance(section, dict):
        raise ValueError("task.yaml must contain a mapping under 'task'")
    return section


def _enabled_section(task_doc: dict[str, Any]) -> dict[str, bool]:
    enabled = task_doc.get("steps", {})
    if not isinstance(enabled, dict):
        raise ValueError("task.yaml 'steps' section must be a mapping")
    return {str(key): bool(value) for key, value in enabled.items()}


def _step_enabled(enabled: dict[str, bool], step_name: str) -> bool:
    return bool(enabled.get(step_name, False))


def _step_cfg(state: WorkflowState, step_name: str) -> dict[str, Any]:
    cfg = state.steps.get(step_name, {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"steps.yaml entry {step_name!r} must be a mapping")
    return cfg


def _ensure_layout(run_root: Path) -> None:
    for path in (
        run_root / "inputs",
        run_root / "stage1",
        run_root / "stage2",
        run_root / "design_output",
        run_root / "_logs",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_warning(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message.rstrip() + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_structure_files(
    source_files: list[Path], dst_dir: Path, *, prefix: str | None = None
) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for idx, src in enumerate(source_files, start=1):
        stem = sanitize_filename(src.stem)
        name = f"{sanitize_filename(prefix)}_{stem}{src.suffix}" if prefix else src.name
        dst = dst_dir / name
        if dst.exists():
            dst = dst_dir / f"{stem}_{idx}{src.suffix}"
        copied.append(_copy_file(src, dst))
    return copied


def _collect_structure_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in STRUCTURE_SUFFIXES
        and not any(part.startswith(".") for part in path.parts)
    )


def _collect_pdb_files(root: Path) -> list[Path]:
    return [
        path for path in _collect_structure_files(root) if path.suffix.lower() == ".pdb"
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _extract_tar_zst_bytes(tarball: bytes, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_bin = shutil.which("tar")
    if tar_bin is None:
        raise FileNotFoundError("tar is not available in PATH")
    with TemporaryDirectory(prefix="ppiflow_extract_") as tmpdir:
        archive_path = Path(tmpdir) / "bundle.tar.zst"
        archive_path.write_bytes(tarball)
        subprocess.run(  # noqa: S603
            [tar_bin, "-I", "zstd", "-xf", str(archive_path), "-C", str(out_dir)],
            check=True,
        )
    return out_dir


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value]
    return [value]


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


##########################################
# Local input staging
##########################################
def _iter_file_refs(obj: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key).lower()
            if isinstance(value, str):
                suffix = Path(value).suffix.lower()
                if suffix in LOCAL_INPUT_SUFFIXES or key_text.endswith(
                    ("pdb", "cif", "csv", "json", "yaml", "yml", "file", "path")
                ):
                    refs.append(value)
            else:
                refs.extend(_iter_file_refs(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(_iter_file_refs(item))
    return refs


def _collect_local_inputs(
    task_doc: dict[str, Any], task_path: Path
) -> list[tuple[str, bytes]]:
    """Collect local files referenced by task.yaml for remote staging."""
    base_dir = task_path.parent
    staged: dict[str, bytes] = {}
    for ref in _iter_file_refs(task_doc):
        if not ref or ref.startswith(("/biomodals-", "/opt/", "s3://", "gs://")):
            continue
        candidate = Path(ref).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if not candidate.exists() or not candidate.is_file():
            continue
        staged[str(Path(ref).name)] = candidate.read_bytes()
    return sorted(staged.items())


def _stage_input_files(run_root: Path, input_files: list[tuple[str, bytes]]) -> None:
    input_dir = run_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for file_name, content in input_files:
        (input_dir / sanitize_filename(Path(file_name).name)).write_bytes(content)


##########################################
# PPIFlow helpers
##########################################
def _gentype_to_ppiflow_mode(gentype: str) -> str:
    if gentype == "binder":
        return "binder"
    if gentype in {"antibody", "nanobody", "antibody_nanobody"}:
        return "antibody_nanobody"
    raise ValueError(f"Unsupported PPIFlow gentype: {gentype!r}")


def _base_ppiflow_config(state: WorkflowState, cfg: dict[str, Any]) -> dict[str, Any]:
    task = state.task
    conf = dict(task)
    conf.update({key: value for key, value in cfg.items() if value is not None})
    conf["name"] = str(task.get("name") or state.run_name)
    conf["specified_hotspots"] = str(conf.get("specified_hotspots") or "")
    conf["samples_per_target"] = _as_int(
        conf.get("samples_per_target"), _as_int(cfg.get("samples_per_target"), 100)
    )
    return conf


def _call_ppiflow(
    state: WorkflowState,
    config: dict[str, Any],
    design_mode: str,
    input_files: list[tuple[str, bytes]],
    out_dir: Path,
    run_label: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ppiflow_fn = _remote_function(state.app_names, "ppiflow", "ppiflow_run_bytes")
    tarball = ppiflow_fn.remote(
        config_dict=config,
        design_mode=design_mode,
        input_files=input_files,
        run_name=run_label,
    )
    _extract_tar_zst_bytes(tarball, out_dir)
    return _collect_pdb_files(out_dir)


def _run_ppiflow_step(
    state: WorkflowState, input_files: list[tuple[str, bytes]]
) -> list[Path]:
    out_dir = state.stage1_dir / "ppiflow_output"
    if not _step_enabled(state.enabled, "PPIFlowStep"):
        _write_warning(out_dir / "SKIPPED.txt", "PPIFlowStep is disabled")
        return _collect_pdb_files(out_dir)

    cfg = _step_cfg(state, "PPIFlowStep")
    config = _base_ppiflow_config(state, cfg)
    gentype = state.gentype
    if gentype == "binder":
        config["input_pdb"] = config.get("input_pdb") or config.get("complex_pdb")
        config["target_chain"] = config.get("target_chain", "B")
        config["binder_chain"] = config.get("binder_chain", "A")
    else:
        config["antigen_chain"] = config.get("antigen_chain", "C")
        config["heavy_chain"] = config.get("heavy_chain", "A")
        if gentype == "nanobody":
            config["light_chain"] = None

    return _call_ppiflow(
        state,
        config=config,
        design_mode=_gentype_to_ppiflow_mode(gentype),
        input_files=input_files,
        out_dir=out_dir,
        run_label=f"{state.run_name}-stage1-ppiflow",
    )


def _partial_config_for_pdb(
    state: WorkflowState,
    cfg: dict[str, Any],
    pdb_path: Path,
    fixed_positions: str,
) -> tuple[str, dict[str, Any]]:
    conf = _base_ppiflow_config(state, cfg)
    conf["samples_per_target"] = _as_int(
        cfg.get("samples_per_target"), _as_int(state.task.get("samples_per_target"), 10)
    )
    conf["start_t"] = _as_float(cfg.get("start_t"), 0.15)
    conf["fixed_positions"] = fixed_positions
    if state.gentype == "binder":
        conf["input_pdb"] = pdb_path.name
        conf["target_chain"] = cfg.get(
            "target_chain", state.task.get("target_chain", "B")
        )
        conf["binder_chain"] = cfg.get(
            "binder_chain", state.task.get("binder_chain", "A")
        )
        conf["sample_hotspot_rate_min"] = _as_float(
            cfg.get("sample_hotspot_rate_min"),
            _as_float(state.task.get("sample_hotspot_rate_min"), 0.2),
        )
        conf["sample_hotspot_rate_max"] = _as_float(
            cfg.get("sample_hotspot_rate_max"),
            _as_float(state.task.get("sample_hotspot_rate_max"), 0.5),
        )
        return "binder_partial_flow", conf

    cdr_position = cfg.get("cdr_position") or state.task.get("cdr_position")
    if not cdr_position:
        raise ValueError(
            "PartialStep for antibody/nanobody requires 'cdr_position' in "
            "steps.yaml PartialStep or task.yaml task"
        )
    conf["complex_pdb"] = pdb_path.name
    conf["cdr_position"] = cdr_position
    conf["antigen_chain"] = cfg.get(
        "antigen_chain", state.task.get("antigen_chain", "C")
    )
    conf["heavy_chain"] = cfg.get("heavy_chain", state.task.get("heavy_chain", "A"))
    conf["light_chain"] = cfg.get("light_chain", state.task.get("light_chain"))
    conf["retry_Limit"] = _as_int(cfg.get("retry_Limit"), 10)
    return "antibody_nanobody_partial", conf


##########################################
# LigandMPNN and FlowPacker queue helpers
##########################################
def _seeds_from_cfg(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("seeds", cfg.get("seed", 0))
    seeds: list[int] = []
    for value in _as_list(raw):
        if value == "":
            continue
        seeds.append(int(value))
    return seeds or [0]


def _ligandmpnn_cli_args(
    state: WorkflowState,
    cfg: dict[str, Any],
    *,
    fixed_residues: str | None = None,
) -> dict[str, str | int | float | bool]:
    model_type = str(
        cfg.get("model_type")
        or ("abmpnn" if state.gentype in {"antibody", "nanobody"} else "soluble_mpnn")
    )
    run_model_type = "protein_mpnn" if model_type == "abmpnn" else model_type
    cli_args: dict[str, str | int | float | bool] = {
        "--model_type": run_model_type,
        "--batch_size": str(_as_int(cfg.get("batch_size"), 1)),
        "--number_of_batches": str(
            _as_int(
                cfg.get("number_of_batches"),
                _as_int(cfg.get("num_seq_per_target"), 1),
            )
        ),
        "--temperature": str(
            _as_float(cfg.get("temperature", cfg.get("sampling_temp")), 0.1)
        ),
        "--save_stats": "1",
        "--pack_side_chains": _as_bool(cfg.get("pack_side_chains"), True),
        "--number_of_packs_per_design": str(
            _as_int(cfg.get("number_of_packs_per_design"), 1)
        ),
        "--sc_num_denoising_steps": str(_as_int(cfg.get("sc_num_denoising_steps"), 3)),
        "--sc_num_samples": str(_as_int(cfg.get("sc_num_samples"), 8)),
        "--repack_everything": _as_bool(cfg.get("repack_everything"), False),
        "--pack_with_ligand_context": _as_bool(
            cfg.get("pack_with_ligand_context"), True
        ),
        "--ligand_mpnn_use_atom_context": _as_bool(
            cfg.get("ligand_mpnn_use_atom_context"), True
        ),
        "--ligand_mpnn_cutoff_for_score": str(
            _as_float(cfg.get("ligand_mpnn_cutoff_for_score"), 8.0)
        ),
        "--ligand_mpnn_use_side_chain_context": _as_bool(
            cfg.get("ligand_mpnn_use_side_chain_context"), False
        ),
        "--global_transmembrane_label": _as_bool(
            cfg.get("global_transmembrane_label"), False
        ),
        "--parse_atoms_with_zero_occupancy": _as_bool(
            cfg.get("parse_atoms_with_zero_occupancy"), False
        ),
    }
    checkpoint = cfg.get("checkpoint")
    if checkpoint is not None:
        cli_args[f"--checkpoint_{model_type}"] = str(checkpoint)
    elif model_type == "abmpnn":
        cli_args["--checkpoint_protein_mpnn"] = (
            "/biomodals-store/LigandMPNN/model_params/abmpnn.pt"
        )

    for cfg_key, cli_key in (
        ("chains_to_design", "--chains_to_design"),
        ("parse_these_chains_only", "--parse_these_chains_only"),
        ("omit_aa", "--omit_AA"),
        ("bias_aa", "--bias_AA"),
        ("redesigned_residues", "--redesigned_residues"),
    ):
        if cfg.get(cfg_key) is not None:
            cli_args[cli_key] = str(cfg[cfg_key])
    if fixed_residues:
        cli_args["--fixed_residues"] = fixed_residues
    elif cfg.get("fixed_residues") is not None:
        cli_args["--fixed_residues"] = str(cfg["fixed_residues"])
    return cli_args


def _normalize_fixed_positions_for_mpnn(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() == "NONE":
        return None
    positions: list[str] = []
    for token in text.split(","):
        token = token.strip()
        match = re.fullmatch(r"[A-Za-z]?(\d+)", token)
        if match:
            positions.append(match.group(1))
    return " ".join(positions) if positions else None


def _collect_mpnn_sequences(mpnn_out: Path, output_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    for fasta in sorted(mpnn_out.rglob("*.fa")) + sorted(mpnn_out.rglob("*.fasta")):
        seqs: list[str] = []
        parts: list[str] = []
        for line in fasta.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    seqs.append("".join(parts))
                    parts = []
            else:
                parts.append(line)
        if parts:
            seqs.append("".join(parts))
        for idx, seq in enumerate(seqs[1:] or seqs):
            rows.append(
                {
                    "link_name": f"{fasta.stem}.pdb".lower(),
                    "sequence_dict": str({"A": seq.split("/")[0]}),
                    "seq_idx": idx,
                }
            )
    _write_csv_rows(output_csv, ["link_name", "sequence_dict", "seq_idx"], rows)


def _run_queue(
    state: WorkflowState,
    queue_label: str,
    tasks: list[dict[str, Any]],
) -> None:
    if not tasks:
        return
    queue_name = f"{CONF.name}-{state.run_name}-{queue_label}-{uuid4().hex}"
    queue = modal.Queue.from_name(queue_name, create_if_missing=True)
    for task in tasks:
        queue.put(task)
    worker_count = max(1, min(state.max_workers, len(tasks)))
    calls = [
        ppiflow_queue_worker.spawn(queue_name, state.app_names)
        for _ in range(worker_count)
    ]
    modal.FunctionCall.gather(*calls)
    modal.Queue.objects.delete(queue_name)
    OUT_VOLUME.reload()


def _run_mpnn_stage(
    state: WorkflowState,
    source_pdbs: list[Path],
    *,
    stage_dir: Path,
    step_name: str,
    output_subdir: str,
    fixed_positions_csv: Path | None = None,
) -> tuple[list[Path], Path]:
    cfg = _step_cfg(state, step_name)
    mpnn_pdbs_dir = stage_dir / "mpnn_pdbs"
    mpnn_out = stage_dir / output_subdir
    copied_pdbs = _copy_structure_files(source_pdbs, mpnn_pdbs_dir)
    fixed_map: dict[str, str | None] = {}
    if fixed_positions_csv and fixed_positions_csv.exists():
        for row in _read_csv_rows(fixed_positions_csv):
            stem = Path(row.get("filename", "")).stem.lower()
            fixed_map[stem] = _normalize_fixed_positions_for_mpnn(
                row.get("fixed_positions")
            )

    tasks: list[dict[str, Any]] = []
    for idx, pdb_path in enumerate(copied_pdbs, start=1):
        fixed_residues = fixed_map.get(pdb_path.stem.lower())
        tasks.append(
            {
                "kind": "ligandmpnn",
                "run_name": f"{state.run_name}-{step_name}-{idx}",
                "input_path": str(pdb_path),
                "output_dir": str(mpnn_out / pdb_path.stem),
                "seeds": _seeds_from_cfg(cfg),
                "cli_args": _ligandmpnn_cli_args(
                    state, cfg, fixed_residues=fixed_residues
                ),
            }
        )
    _run_queue(state, step_name.lower(), tasks)
    _collect_mpnn_sequences(mpnn_out, mpnn_pdbs_dir / "mpnn_seqs.csv")
    mpnn_pdbs = _collect_pdb_files(mpnn_out)
    if not mpnn_pdbs:
        _write_warning(
            mpnn_out / "NO_PACKED_PDBS.txt",
            "LigandMPNN did not emit packed PDBs; downstream steps use input backbones.",
        )
        mpnn_pdbs = copied_pdbs
    return mpnn_pdbs, mpnn_pdbs_dir / "mpnn_seqs.csv"


def _run_flowpacker_stage(
    state: WorkflowState,
    source_pdbs: list[Path],
    *,
    stage_dir: Path,
    step_name: str,
) -> list[Path]:
    cfg = _step_cfg(state, step_name)
    out_dir = stage_dir / "flowpacker_output"
    tasks = [
        {
            "kind": "flowpacker",
            "run_name": f"{state.run_name}-{step_name}-{idx}",
            "input_path": str(pdb_path),
            "output_dir": str(out_dir / pdb_path.stem),
            "model_name": str(cfg.get("model_name") or cfg.get("model") or "cluster"),
            "use_confidence": _as_bool(cfg.get("use_confidence"), False),
            "n_samples": _as_int(cfg.get("n_samples"), 1),
            "num_steps": _as_int(cfg.get("num_steps"), 10),
            "sample_coeff": _as_float(cfg.get("sample_coeff"), 5.0),
            "use_gt_masks": _as_bool(cfg.get("use_gt_masks"), False),
            "inpaint": cfg.get("inpaint"),
            "save_traj": _as_bool(cfg.get("save_traj"), False),
            "seed": _as_int(cfg.get("seed"), 42),
        }
        for idx, pdb_path in enumerate(source_pdbs, start=1)
    ]
    _run_queue(state, step_name.lower(), tasks)
    packed = _collect_pdb_files(out_dir)
    if not packed:
        _write_warning(
            out_dir / "NO_FLOWPACKER_PDBS.txt",
            "FlowPacker produced no PDB files; downstream steps use input structures.",
        )
        packed = source_pdbs
    return packed


##########################################
# AF3Score and filtering
##########################################
def _run_af3score_stage(
    state: WorkflowState,
    source_pdbs: list[Path],
    *,
    stage_dir: Path,
    step_name: str,
) -> Path:
    cfg = _step_cfg(state, step_name)
    out_dir = stage_dir / "af3score_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source_pdbs:
        raise ValueError(f"{step_name} requires at least one PDB input")
    af3score_fn = _remote_function(state.app_names, "af3score", "af3score_run_bytes")
    tarball = af3score_fn.remote(
        input_files=[(path.name, path.read_bytes()) for path in source_pdbs],
        run_name=f"{state.run_name}-{step_name}",
        prepare_workers=_as_int(cfg.get("prepare_workers"), 8),
        max_batches=_as_int(cfg.get("max_batches"), 10),
    )
    _extract_tar_zst_bytes(tarball, out_dir)
    return out_dir


def _parse_filter_clause(clause: str) -> tuple[str, float]:
    match = re.fullmatch(r"\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*", clause)
    if not match:
        raise ValueError(f"Invalid filter clause: {clause!r}")
    return match.group(1), float(match.group(2))


def _metric_passes(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"Unsupported filter operator: {op}")


def _row_passes_filters(row: dict[str, str], filters: dict[str, Any]) -> bool:
    for metric, condition in filters.items():
        if metric not in row or row[metric] == "":
            return False
        value = float(row[metric])
        clauses = condition if isinstance(condition, list) else [condition]
        for clause in clauses:
            if isinstance(clause, dict):
                for op, threshold in clause.items():
                    if not _metric_passes(value, str(op), float(threshold)):
                        return False
            elif isinstance(clause, str):
                op, threshold = _parse_filter_clause(clause)
                if not _metric_passes(value, op, threshold):
                    return False
            else:
                if value < float(clause):
                    return False
    return True


def _default_filter(stage: int) -> dict[str, str]:
    return {"iptm": "> 0.7"} if stage == 1 else {"iptm": "> 0.8"}


def _filter_stage(
    state: WorkflowState,
    source_pdbs: list[Path],
    *,
    stage_dir: Path,
    step_name: str,
    out_name: str,
    output_csv_name: str,
    default_stage: int,
) -> list[Path]:
    cfg = _step_cfg(state, step_name)
    out_dir = stage_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = _first_existing(
        sorted((stage_dir / "af3score_output").rglob("af3score_metrics.csv"))
    )
    if metrics_csv is None:
        _write_warning(
            out_dir / "FILTER_WARNING.txt",
            "AF3Score metrics CSV was not found; all input PDBs were retained.",
        )
        retained = _copy_structure_files(source_pdbs, out_dir)
        _write_csv_rows(
            out_dir / output_csv_name,
            ["filename"],
            [{"filename": p.name} for p in retained],
        )
        return retained

    rows = _read_csv_rows(metrics_csv)
    filters = cfg.get("filters") or _default_filter(default_stage)
    passing_rows = [row for row in rows if _row_passes_filters(row, filters)]
    if not passing_rows:
        _write_warning(
            out_dir / "FILTER_WARNING.txt",
            "No structures passed filters; all input PDBs were retained.",
        )
        passing_rows = rows

    pdb_by_stem = {path.stem.lower(): path for path in source_pdbs}
    retained: list[Path] = []
    out_rows: list[dict[str, Any]] = []
    for row in passing_rows:
        names = [
            row.get("description", ""),
            row.get("filename", ""),
            row.get("pdb", ""),
            row.get("name", ""),
        ]
        stem = next((Path(name).stem.lower() for name in names if name), "")
        src = pdb_by_stem.get(stem)
        if src is None and len(source_pdbs) == 1:
            src = source_pdbs[0]
        if src is None:
            continue
        dst = _copy_file(src, out_dir / src.name)
        retained.append(dst)
        out_rows.append({"filename": dst.name, **row})

    if not retained:
        retained = _copy_structure_files(source_pdbs, out_dir)
        out_rows = [{"filename": p.name} for p in retained]
    fieldnames = list(dict.fromkeys(["filename", *(rows[0].keys() if rows else [])]))
    _write_csv_rows(out_dir / output_csv_name, fieldnames, out_rows)
    return retained


##########################################
# Rosetta, partial flow, AF3 refold, DockQ
##########################################
def _rosetta_jobs(
    pdbs: list[Path],
    cfg: dict[str, Any],
    *,
    binary_default: str = "relax",
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for idx, pdb_path in enumerate(pdbs, start=1):
        jobs.append(
            {
                "index": idx,
                "pdb_name": pdb_path.name,
                "pdb_bytes": pdb_path.read_bytes(),
                "binary": str(
                    cfg.get("binary") or cfg.get("rosetta_binary") or binary_default
                ),
                "rosetta_script_text": cfg.get("rosetta_script_text"),
                "rosetta_script_name": cfg.get("rosetta_script_name", "protocol.xml"),
                "flags_text": cfg.get("flags_text"),
                "flags_name": cfg.get("flags_name", "flags.txt"),
            }
        )
    return jobs


def _run_rosetta_batch(
    state: WorkflowState,
    source_pdbs: list[Path],
    *,
    step_name: str,
    out_dir: Path,
    binary_default: str = "relax",
) -> list[Path]:
    cfg = _step_cfg(state, step_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source_pdbs:
        return []
    rosetta_fn = _remote_function(state.app_names, "rosetta", "run_rosetta_batch_bytes")
    tarball = rosetta_fn.remote(
        jobs=_rosetta_jobs(source_pdbs, cfg, binary_default=binary_default),
        run_name=f"{state.run_name}-{step_name}",
        num_cpu_per_pod=max(1, min(30, len(source_pdbs))),
    )
    _extract_tar_zst_bytes(tarball, out_dir)
    return _collect_pdb_files(out_dir) or source_pdbs


def _write_fixed_positions_csv(
    state: WorkflowState, source_pdbs: list[Path], output_csv: Path
) -> None:
    cfg = _step_cfg(state, "PartialStep")
    explicit = cfg.get("fixed_positions") or state.task.get("fixed_positions")
    rows = [
        {
            "filename": pdb.name,
            "fixed_positions": str(explicit or "NONE"),
        }
        for pdb in source_pdbs
    ]
    _write_csv_rows(output_csv, ["filename", "fixed_positions"], rows)
    if not explicit:
        _write_warning(
            output_csv.with_suffix(".warning.txt"),
            "No residue-energy CSV was available from RosettaFixStep; "
            "fixed_positions was set to NONE for each structure.",
        )


def _run_partial_step(state: WorkflowState, source_pdbs: list[Path]) -> list[Path]:
    cfg = _step_cfg(state, "PartialStep")
    partial_out = state.stage2_dir / "partial_output"
    before_partial = state.stage2_dir / "before_partial_pdbs"
    fixed_csv = state.stage2_dir / "fixed_positions.csv"
    copied = _copy_structure_files(source_pdbs, before_partial)
    rows = _read_csv_rows(fixed_csv)
    fixed_by_name = {
        row["filename"]: row.get("fixed_positions", "NONE") for row in rows
    }

    results: list[Path] = []
    for idx, pdb_path in enumerate(copied, start=1):
        fixed_positions = fixed_by_name.get(pdb_path.name, "NONE")
        per_target_out = partial_out / pdb_path.stem
        if not fixed_positions or fixed_positions.upper() == "NONE":
            _copy_file(pdb_path, per_target_out / pdb_path.name)
            _write_warning(
                per_target_out / "PARTIAL_SKIPPED.txt",
                "No fixed positions were provided; copied input structure unchanged.",
            )
            results.append(per_target_out / pdb_path.name)
            continue
        mode, config = _partial_config_for_pdb(state, cfg, pdb_path, fixed_positions)
        input_files = [(pdb_path.name, pdb_path.read_bytes())]
        results.extend(
            _call_ppiflow(
                state,
                config=config,
                design_mode=mode,
                input_files=input_files,
                out_dir=per_target_out,
                run_label=f"{state.run_name}-partial-{idx}",
            )
        )
    return results or _collect_pdb_files(partial_out)


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _pdb_to_sequences(path: Path) -> list[tuple[str, str]]:
    residues: dict[str, list[tuple[tuple[str, str], str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        resname = line[17:20].strip().upper()
        chain = (line[21].strip() or "A")[:1]
        res_id = line[22:27].strip()
        key = (chain, res_id, resname)
        if key in seen:
            continue
        seen.add(key)
        residues.setdefault(chain, []).append(
            ((res_id, resname), AA3_TO_1.get(resname, "X"))
        )
    return [
        (chain, "".join(aa for _, aa in chain_residues))
        for chain, chain_residues in residues.items()
        if chain_residues
    ]


def _af3_json_for_pdb(pdb_path: Path, cfg: dict[str, Any]) -> bytes:
    sequences = [
        AF3SequenceEntry(protein=AF3Protein(id=chain, sequence=seq))
        for chain, seq in _pdb_to_sequences(pdb_path)
    ]
    if not sequences:
        raise ValueError(f"No protein sequences could be parsed from {pdb_path}")
    seed = _as_int(cfg.get("model_seed", cfg.get("seed")), 1)
    conf = AF3Config(
        name=sanitize_filename(pdb_path.stem),
        modelSeeds=[seed],
        sequences=sequences,
    )
    return conf.model_dump_json(indent=2).encode("utf-8")


def _run_refold_and_relax(state: WorkflowState, source_pdbs: list[Path]) -> None:
    refold_cfg = _step_cfg(state, "ReFoldStep")
    relax_cfg = _step_cfg(state, "RosettaRelaxStep")
    tasks: list[dict[str, Any]] = []
    if _step_enabled(state.enabled, "ReFoldStep"):
        for idx, pdb_path in enumerate(source_pdbs, start=1):
            output_dir = state.stage2_dir / "refold_output" / pdb_path.stem
            json_path = output_dir / "input.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_bytes(_af3_json_for_pdb(pdb_path, refold_cfg))
            tasks.append(
                {
                    "kind": "af3_refold",
                    "run_name": f"{state.run_name}-refold-{idx}",
                    "input_path": str(pdb_path),
                    "output_dir": str(output_dir),
                    "json_path": str(json_path),
                    "recycle": _as_int(refold_cfg.get("recycle"), 10),
                    "sample": _as_int(
                        refold_cfg.get("sample", refold_cfg.get("seed_num")), 5
                    ),
                }
            )
    if _step_enabled(state.enabled, "RosettaRelaxStep"):
        for idx, pdb_path in enumerate(source_pdbs, start=1):
            tasks.append(
                {
                    "kind": "rosetta_relax",
                    "run_name": f"{state.run_name}-relax-{idx}",
                    "input_path": str(pdb_path),
                    "output_dir": str(
                        state.stage2_dir / "rosetta_relax_output" / pdb_path.stem
                    ),
                    "rosetta_cfg": relax_cfg,
                }
            )
    _run_queue(state, "refold-relax", tasks)


def _pair_refold_models(
    refold_out: Path, references: list[Path]
) -> list[dict[str, object]]:
    model_files = sorted(
        path
        for path in refold_out.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdb", ".cif"}
    )
    pairs: list[dict[str, object]] = []
    for idx, model in enumerate(model_files, start=1):
        model_context = "/".join(
            part.lower() for part in model.relative_to(refold_out).parts
        )
        ref = next(
            (
                candidate
                for candidate in references
                if candidate.stem.lower() in model_context
            ),
            references[0] if len(references) == 1 else None,
        )
        if ref is None:
            continue
        pairs.append(
            {
                "id": f"{ref.stem}_{idx}",
                "model_name": model.name,
                "model_bytes": model.read_bytes(),
                "reference_name": ref.name,
                "reference_bytes": ref.read_bytes(),
            }
        )
    return pairs


def _run_dockq(state: WorkflowState, references: list[Path]) -> Path:
    out_dir = state.stage2_dir / "dockq_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = _pair_refold_models(state.stage2_dir / "refold_output", references)
    if not pairs:
        _write_warning(out_dir / "DOCKQ_SKIPPED.txt", "No refold/reference pairs found")
        return out_dir
    cfg = _step_cfg(state, "DockQStep")
    dockq_args = cfg.get("dockq_args", ["--short"])
    if isinstance(dockq_args, str):
        dockq_args = [part for part in dockq_args.split(" ") if part]
    dockq_fn = _remote_function(state.app_names, "dockq", "run_dockq_batch")
    tarball = dockq_fn.remote(
        pairs=pairs,
        run_name=f"{state.run_name}-dockq",
        dockq_args=dockq_args,
    )
    _extract_tar_zst_bytes(tarball, out_dir)
    return out_dir


##########################################
# Ranking and report
##########################################
def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _read_optional_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    return _read_csv_rows(path)


def _rank_and_report(state: WorkflowState, filtered_pdbs: list[Path]) -> None:
    out_dir = state.design_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage2_filter_csv = _first_existing(
        sorted((state.stage2_dir / "filtered_iptm08").glob("filtered_iptm08.csv"))
    )
    dockq_csv = _first_existing(
        sorted((state.stage2_dir / "dockq_output").rglob("dockq_results.csv"))
    )
    af_rows = _read_optional_csv(stage2_filter_csv)
    dockq_rows = _read_optional_csv(dockq_csv)
    dockq_by_id = {row.get("id", ""): row for row in dockq_rows}
    dockq_by_ref = {
        Path(row.get("reference", "")).stem.lower(): row for row in dockq_rows
    }

    ranked_rows: list[dict[str, Any]] = []
    for pdb_path in filtered_pdbs:
        af_row = next(
            (
                row
                for row in af_rows
                if Path(row.get("filename", row.get("description", ""))).stem.lower()
                == pdb_path.stem.lower()
            ),
            {},
        )
        dockq_row = dockq_by_ref.get(pdb_path.stem.lower()) or dockq_by_id.get(
            pdb_path.stem, {}
        )
        ranked_rows.append(
            {
                "design": pdb_path.stem,
                "pdb": str(pdb_path.relative_to(state.run_root)),
                "iptm": af_row.get("iptm", af_row.get("ranking_score", "")),
                "dockq": dockq_row.get("dockq", ""),
                "irmsd": dockq_row.get("irmsd", ""),
                "lrmsd": dockq_row.get("lrmsd", ""),
                "fnat": dockq_row.get("fnat", ""),
            }
        )

    ranked_rows.sort(
        key=lambda row: (_safe_float(row.get("dockq")), _safe_float(row.get("iptm"))),
        reverse=True,
    )
    _write_csv_rows(
        out_dir / "ranked_designs.csv",
        ["design", "pdb", "iptm", "dockq", "irmsd", "lrmsd", "fnat"],
        ranked_rows,
    )

    top_dir = out_dir / "top_models"
    for row in ranked_rows[: min(10, len(ranked_rows))]:
        src = state.run_root / str(row["pdb"])
        if src.exists():
            _copy_file(src, top_dir / src.name)

    report = [
        "# PPIFlow Workflow Report",
        "",
        f"- Run: {state.run_name}",
        f"- Gentype: {state.gentype}",
        f"- Ranked designs: {len(ranked_rows)}",
        f"- Stage 1 enabled steps: {', '.join(k for k in STAGE1_STEPS if _step_enabled(state.enabled, k)) or 'none'}",
        f"- Stage 2 enabled steps: {', '.join(k for k in STAGE2_STEPS if _step_enabled(state.enabled, k)) or 'none'}",
        "",
        "See `ranked_designs.csv` for sortable metrics.",
    ]
    (out_dir / "design_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (out_dir / "design_report.html").write_text(
        "<html><body><pre>"
        + "\n".join(report).replace("&", "&amp;").replace("<", "&lt;")
        + "</pre></body></html>\n",
        encoding="utf-8",
    )


##########################################
# Queue worker
##########################################
@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def ppiflow_queue_worker(queue_name: str, app_names: dict[str, str]) -> None:
    """Process independent per-structure workflow tasks from a Modal queue."""
    queue = modal.Queue.from_name(queue_name)
    while True:
        task = queue.get(block=False)
        if task is None:
            return

        kind = task["kind"]
        output_dir = Path(task["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if kind == "ligandmpnn":
            input_path = Path(task["input_path"])
            ligandmpnn_fn = _remote_function(app_names, "ligandmpnn", "ligandmpnn_run")
            tarball = ligandmpnn_fn.remote(
                task["run_name"],
                "run",
                input_path.read_bytes(),
                task["seeds"],
                task["cli_args"],
                None,
                None,
            )
            _extract_tar_zst_bytes(tarball, output_dir)
        elif kind == "flowpacker":
            input_path = Path(task["input_path"])
            flowpacker_fn = _remote_function(app_names, "flowpacker", "run_flowpacker")
            tarball = flowpacker_fn.remote(
                input_files=[(input_path.name, input_path.read_bytes())],
                run_name=task["run_name"],
                model_name=task["model_name"],
                use_confidence=task["use_confidence"],
                n_samples=task["n_samples"],
                num_steps=task["num_steps"],
                sample_coeff=task["sample_coeff"],
                use_gt_masks=task["use_gt_masks"],
                inpaint=task["inpaint"],
                save_traj=task["save_traj"],
                seed=task["seed"],
            )
            _extract_tar_zst_bytes(tarball, output_dir)
        elif kind == "af3_refold":
            data_fn = _remote_function(app_names, "alphafold3", "run_data_pipeline")
            inference_fn = _remote_function(
                app_names, "alphafold3", "run_inference_pipeline"
            )
            json_path = data_fn.remote(Path(task["json_path"]).read_bytes())
            tarball = inference_fn.remote(
                json_path,
                recycle=task["recycle"],
                sample=task["sample"],
            )
            _extract_tar_zst_bytes(tarball, output_dir)
        elif kind == "rosetta_relax":
            input_path = Path(task["input_path"])
            rosetta_fn = _remote_function(
                app_names, "rosetta", "run_rosetta_batch_bytes"
            )
            tarball = rosetta_fn.remote(
                jobs=[
                    _rosetta_jobs(
                        [input_path],
                        task.get("rosetta_cfg", {}),
                        binary_default="relax",
                    )[0]
                ],
                run_name=task["run_name"],
                num_cpu_per_pod=1,
            )
            _extract_tar_zst_bytes(tarball, output_dir)
        else:
            raise ValueError(f"Unsupported queue task kind: {kind}")

        _write_json(output_dir / "workflow_task.json", task)
        OUT_VOLUME.commit()


##########################################
# Stage runners
##########################################
def _run_stage1(
    state: WorkflowState, input_files: list[tuple[str, bytes]]
) -> list[Path]:
    ppiflow_pdbs = _run_ppiflow_step(state, input_files)
    if not ppiflow_pdbs:
        ppiflow_pdbs = _collect_pdb_files(state.run_root / "inputs")
    if not ppiflow_pdbs:
        raise RuntimeError("Stage 1 produced no PDB files")

    mpnn_pdbs = ppiflow_pdbs
    if state.gentype == "binder" and _step_enabled(state.enabled, "MPNNStep_stage1"):
        mpnn_pdbs, _ = _run_mpnn_stage(
            state,
            ppiflow_pdbs,
            stage_dir=state.stage1_dir,
            step_name="MPNNStep_stage1",
            output_subdir="mpnn_output",
        )
    elif state.gentype in {"antibody", "nanobody"} and _step_enabled(
        state.enabled, "AbMPNNStep_stage1"
    ):
        mpnn_pdbs, _ = _run_mpnn_stage(
            state,
            ppiflow_pdbs,
            stage_dir=state.stage1_dir,
            step_name="AbMPNNStep_stage1",
            output_subdir="abmpnn_output",
        )

    packed_pdbs = mpnn_pdbs
    if _step_enabled(state.enabled, "FlowpackerStep_stage1"):
        packed_pdbs = _run_flowpacker_stage(
            state,
            mpnn_pdbs,
            stage_dir=state.stage1_dir,
            step_name="FlowpackerStep_stage1",
        )

    scored_pdbs = packed_pdbs
    if _step_enabled(state.enabled, "AF3scoreStep_stage1"):
        _run_af3score_stage(
            state,
            packed_pdbs,
            stage_dir=state.stage1_dir,
            step_name="AF3scoreStep_stage1",
        )
    if _step_enabled(state.enabled, "FilterStep_stage1"):
        scored_pdbs = _filter_stage(
            state,
            packed_pdbs,
            stage_dir=state.stage1_dir,
            step_name="FilterStep_stage1",
            out_name="filtered_iptm07",
            output_csv_name="filtered_iptm07.csv",
            default_stage=1,
        )
    return scored_pdbs


def _stage2_seed_inputs(state: WorkflowState) -> list[Path]:
    candidates = _collect_pdb_files(state.stage1_dir / "filtered_iptm07")
    if candidates:
        return candidates
    input_pdbs = _collect_pdb_files(state.run_root / "inputs")
    if input_pdbs:
        out_dir = state.stage1_dir / "filtered_iptm07"
        _write_warning(
            out_dir / "STAGE2_INPUT_WARNING.txt",
            "Stage 2 was requested without Stage 1 filtered outputs; using staged input PDBs.",
        )
        return _copy_structure_files(input_pdbs, out_dir)
    raise RuntimeError("Stage 2 requires Stage 1 filtered PDBs or input PDBs")


def _run_stage2(state: WorkflowState) -> list[Path]:
    stage2_inputs = _stage2_seed_inputs(state)

    rosetta_fixed = stage2_inputs
    if _step_enabled(state.enabled, "RosettaFixStep"):
        rosetta_fixed = _run_rosetta_batch(
            state,
            stage2_inputs,
            step_name="RosettaFixStep",
            out_dir=state.stage2_dir / "rosetta_fix_output",
            binary_default="relax",
        )

    fixed_csv = state.stage2_dir / "fixed_positions.csv"
    if _step_enabled(state.enabled, "PartialStep"):
        _write_fixed_positions_csv(state, stage2_inputs, fixed_csv)
        partial_pdbs = _run_partial_step(state, stage2_inputs)
    else:
        partial_pdbs = rosetta_fixed

    mpnn_pdbs = partial_pdbs
    if state.gentype == "binder" and _step_enabled(state.enabled, "MPNNStep_stage2"):
        mpnn_pdbs, _ = _run_mpnn_stage(
            state,
            partial_pdbs,
            stage_dir=state.stage2_dir,
            step_name="MPNNStep_stage2",
            output_subdir="mpnn_output",
            fixed_positions_csv=fixed_csv if fixed_csv.exists() else None,
        )
    elif state.gentype in {"antibody", "nanobody"} and _step_enabled(
        state.enabled, "AbMPNNStep_stage2"
    ):
        mpnn_pdbs, _ = _run_mpnn_stage(
            state,
            partial_pdbs,
            stage_dir=state.stage2_dir,
            step_name="AbMPNNStep_stage2",
            output_subdir="abmpnn_output",
            fixed_positions_csv=fixed_csv if fixed_csv.exists() else None,
        )

    packed_pdbs = mpnn_pdbs
    if _step_enabled(state.enabled, "FlowpackerStep_stage2"):
        packed_pdbs = _run_flowpacker_stage(
            state,
            mpnn_pdbs,
            stage_dir=state.stage2_dir,
            step_name="FlowpackerStep_stage2",
        )

    filtered_pdbs = packed_pdbs
    if _step_enabled(state.enabled, "AF3scoreStep_stage2"):
        _run_af3score_stage(
            state,
            packed_pdbs,
            stage_dir=state.stage2_dir,
            step_name="AF3scoreStep_stage2",
        )
    if _step_enabled(state.enabled, "FilterStep_stage2"):
        filtered_pdbs = _filter_stage(
            state,
            packed_pdbs,
            stage_dir=state.stage2_dir,
            step_name="FilterStep_stage2",
            out_name="filtered_iptm08",
            output_csv_name="filtered_iptm08.csv",
            default_stage=2,
        )

    _run_refold_and_relax(state, filtered_pdbs)
    if _step_enabled(state.enabled, "DockQStep"):
        _run_dockq(state, filtered_pdbs)
    if _step_enabled(state.enabled, "RankStep") or _step_enabled(
        state.enabled, "ReportStep"
    ):
        _rank_and_report(state, filtered_pdbs)
    return filtered_pdbs


##########################################
# Main remote function and local entrypoint
##########################################
@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.output_volume_mountpoint: OUT_VOLUME},
)
def run_ppiflow_workflow(
    task_yaml_bytes: bytes,
    steps_yaml_bytes: bytes,
    input_files: list[tuple[str, bytes]],
    run_name: str,
    stage: int | None,
    max_workers: int,
    app_names: dict[str, str],
    force: bool,
) -> bytes:
    """Run the PPIFlow workflow on Modal and return a packaged archive."""
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    safe_run_name = sanitize_filename(run_name)
    run_root = MOUNT_ROOT / safe_run_name
    if run_root.exists():
        if force:
            shutil.rmtree(run_root)
        else:
            raise FileExistsError(
                f"Workflow volume run already exists: {run_root}. "
                "Use --force or choose a different --run-name."
            )
    _ensure_layout(run_root)
    _stage_input_files(run_root, input_files)
    _write_json(run_root / "inputs" / "task.yaml.json", task_doc)
    _write_json(run_root / "inputs" / "steps.yaml.json", steps_doc)
    _write_json(run_root / "inputs" / "deployed_apps.json", app_names)
    OUT_VOLUME.commit()

    state = WorkflowState(
        run_root=run_root,
        run_name=safe_run_name,
        task=task,
        enabled=enabled,
        steps=steps_doc,
        app_names=app_names,
        max_workers=max(1, max_workers),
    )

    if stage not in {None, 1, 2}:
        raise ValueError("--stage must be omitted, 1, or 2")
    if stage in {None, 1}:
        _run_stage1(state, input_files)
        OUT_VOLUME.commit()
    if stage in {None, 2}:
        _run_stage2(state)
        OUT_VOLUME.commit()

    return package_outputs(run_root)


@app.local_entrypoint()
def submit_ppiflow_workflow(
    task: str,
    steps: str,
    stage: int | None = None,
    out_dir: str | None = None,
    run_name: str | None = None,
    max_workers: int = 4,
    force: bool = False,
) -> None:
    """Run the upstream-YAML-compatible PPIFlow workflow.

    Args:
        task: Path to upstream-style task.yaml. The file should contain a
            top-level ``task:`` mapping and optional ``steps:`` enable map.
        steps: Path to upstream-style steps.yaml with per-step runtime options.
        stage: Optional stage selector. Use ``1`` for Stage 1 only, ``2`` for
            Stage 2 only, or omit to run both stages.
        out_dir: Optional local output directory for the final ``.tar.zst``.
            Defaults to ``task.output_base_dir`` when present, otherwise CWD.
        run_name: Optional run name. Defaults to ``task.name`` or the task file
            stem. The name also becomes the top-level directory in the archive.
        max_workers: Maximum Modal queue workers for independent per-structure
            tasks.
        force: Replace an existing local tarball or remote workflow run with
            the same name.
    """
    task_path = Path(task).expanduser().resolve()
    steps_path = Path(steps).expanduser().resolve()
    if not task_path.exists():
        raise FileNotFoundError(f"Task YAML not found: {task_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Steps YAML not found: {steps_path}")

    task_doc = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    task_section = _task_section(task_doc)
    resolved_run_name = run_name or str(task_section.get("name") or task_path.stem)
    if out_dir is None:
        out_dir = str(task_section.get("output_base_dir") or Path.cwd())
    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = local_out_dir / f"{sanitize_filename(resolved_run_name)}.tar.zst"
    if out_file.exists() and not force:
        raise FileExistsError(f"Output file already exists: {out_file}")

    input_files = _collect_local_inputs(task_doc, task_path)
    app_names = _default_app_names()
    print(f"🧬 Submitting PPIFlow workflow '{resolved_run_name}'")
    print(f"🧬 Staged {len(input_files)} local input file(s)")
    tarball = run_ppiflow_workflow.remote(
        task_yaml_bytes=task_path.read_bytes(),
        steps_yaml_bytes=steps_path.read_bytes(),
        input_files=input_files,
        run_name=resolved_run_name,
        stage=stage,
        max_workers=max_workers,
        app_names=app_names,
        force=force,
    )
    out_file.write_bytes(tarball)
    print(f"🧬 PPIFlow workflow complete. Results saved to {out_file}")
