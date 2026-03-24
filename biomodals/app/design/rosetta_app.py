"""Rosetta source repo: <https://github.com/RosettaCommons/rosetta>.

Relax docs: <https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax>
InterfaceAnalyzer docs: <https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer>

Generic Modal launcher for existing Rosetta executables. This script does not
clone or depend on a Rosetta source checkout at runtime. Each command runs one
official Rosetta application already present in the Modal image: `relax`,
or `interface_analyzer`. This launcher currently runs Rosetta on CPU, not GPU.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-path` | **Required** | Path to one input `.pdb` file, or a directory containing multiple `.pdb` files for batch processing. |
| `--protocol` | **Required** | Rosetta application to run. One of: `relax` or `interface_analyzer`. |
| `--output-dir` | **Required** | Local download directory. After the Modal job finishes, results are downloaded here. |
| `--output-dir-name` | stem name of `--output-dir` | Optional run name. Remote results are written under `rosetta_outputs/<protocol>/<run_name>`, and local downloads land in `<output-dir>/<run_name>`. |
| `--rosetta-bin` | `/rosetta_source_bin` | Preferred Rosetta executable directory. If that path is not present, the launcher falls back to common in-container binary locations such as `/usr/local/bin`. |
| `--rosetta-database` | `/rosetta_database` | Preferred Rosetta database directory. The launcher checks this first, then falls back to common in-container database locations if needed. |
| `--volume-name` | `rosetta_outputs` | Modal volume name used for remote output storage. Large Rosetta outputs are written here first. |
| `--resume` | off | Reuse existing outputs for the same run name and skip completed items. If not set, old `results/`, `logs/`, `summary.csv`, `aggregated_scores.csv`, and failed-record files under the same remote run name are cleared before a new run starts. |
| `--failed-record-file` | `failed_records.txt` under the run directory | Optional file used to record failed inputs and error messages during batch runs. |
| `--extra-args` | `None` | Extra Rosetta CLI flags passed through safely with `shlex.split`. Example: `--extra-args '-nstruct 10'`. |
| `--parallel-batches` | `1` | Split a directory input into this many parallel Modal workers automatically. Each worker uses one CPU and writes into the same run directory. |
| `--batch-index` | `None` | Optional zero-based batch index when splitting a directory input across multiple Modal runs. |
| `--num-batches` | `None` | Optional total number of batches when splitting a directory input. Must be used together with `--batch-index`. |
| `--interface` | `None` | `interface_analyzer` grouping string such as `A_B` or `AB_C`. Optional for two-chain complexes. Required for multichain complexes unless `--fixedchains` is used. |
| `--fixedchains` | `None` | `interface_analyzer` multichain grouping option such as `'A B'`. Optional for two-chain complexes. Required for multichain complexes unless `--interface` is used. |
| `--relax-movable-chains` | `None` | Optional `relax` restriction. Only the listed chains are allowed to repack/minimize, for example `A,B` or `A B`. |
| `--relax-interface-groups` | `None` | Optional `relax` restriction. Only residues near the specified interface groups are allowed to move, for example `A_B` or `AB_C`. |
| `--relax-interface-cutoff` | `8.0` | Distance cutoff in Angstrom for `--relax-interface-groups`. Residues with any cross-interface heavy-atom contact within this cutoff are selected. |
| `--relax-movable-ranges` | `None` | Optional `relax` restriction. Only the listed PDB residue ranges are allowed to move, for example `A:25-40,B:10-12`. |

For `interface_analyzer`, this launcher automatically adds the official Rosetta
flags `-add_regular_scores_to_scorefile` and `-compute_packstat`, so
`aggregated_scores.csv` includes both interface metrics and regular whole-complex
energy terms. If you care about packstat stability, consider:
`--extra-args '-packstat::oversample 100'`.

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| None | n/a | This launcher does not require any custom environment variables for normal use. Configure behavior with CLI flags instead. |

## Example CLI

Relax one structure:

```bash
modal run rosetta_app.py \
  --input-path ./input.pdb \
  --protocol relax \
  --output-dir ./runs \
  --output-dir-name relax_test
```

Default `relax` performs local refinement on the full input pose. For a
complex, that means both partners may move slightly unless one of the optional
`--relax-*` restriction flags is provided.

Relax only chain A:

```bash
modal run rosetta_app.py \
  --input-path ./complex.pdb \
  --protocol relax \
  --relax-movable-chains A \
  --output-dir ./runs \
  --output-dir-name relax_chain_a
```

Relax only interface residues between chains A and B:

```bash
modal run rosetta_app.py \
  --input-path ./complex.pdb \
  --protocol relax \
  --relax-interface-groups A_B \
  --relax-interface-cutoff 4.0 \
  --output-dir ./runs \
  --output-dir-name relax_interface_ab
```

Parallelize a directory relax run into 4 automatic batches:

```bash
modal run rosetta_app.py \
  --input-path ./pdb_dir \
  --protocol relax \
  --parallel-batches 4 \
  --output-dir ./runs \
  --output-dir-name relax_parallel
```

Run `interface_analyzer` on a standard two-chain complex:

```bash
modal run rosetta_app.py \
  --input-path ./complex.pdb \
  --protocol interface_analyzer \
  --output-dir ./runs \
  --output-dir-name iface_test
```

Run `interface_analyzer` on a multichain complex:

```bash
modal run rosetta_app.py \
  --input-path ./complex.pdb \
  --protocol interface_analyzer \
  --fixedchains 'A B' \
  --output-dir ./runs \
  --output-dir-name iface_multi
```

## Outputs

Remote outputs are always written to:
`rosetta_outputs/<protocol>/<run_name>`

Local downloads are written to:
`<output-dir>/<run_name>`

Typical run contents:

* `results/<item_id>/score.sc`: original Rosetta score file written by the Rosetta executable.
* `results/<item_id>/*.pdb`: Rosetta output structures for that input item.
* `results/<item_id>/SUCCESS.json`: success marker and metadata for resume logic.
* `results/<item_id>/relax_controls/`: only present for restricted `relax` runs.
  `relax.movemap` controls allowed backbone/side-chain motion, and
  `relax.resfile` controls allowed side-chain repacking.
* `logs/*.stdout.log`: captured Rosetta standard output for each input item.
* `logs/*.stderr.log`: captured Rosetta standard error for each input item.
* `logs/*.command.txt`: exact Rosetta command used for each input item.
* `summary.csv`: per-item run summary with success, failure, skip, and return-code information.
* `aggregated_scores.csv`: merged score table across all processed inputs in the current run.

For `interface_analyzer`, `aggregated_scores.csv` is a wide table where:

* `inter_*` columns are interface-specific metrics such as `inter_dG_separated`,
  `inter_dSASA_int`, `inter_delta_unsatHbonds`, `inter_packstat`, and
  `inter_sc_value`.
* `ener_*` columns are regular Rosetta whole-complex energy terms such as
  `ener_total_score`, `ener_fa_atr`, `ener_fa_rep`, and `ener_fa_sol`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


DEFAULT_ROSETTA_BIN = "/rosetta_source_bin"
DEFAULT_ROSETTA_DATABASE = "/rosetta_database"
DEFAULT_VOLUME_NAME = "rosetta_outputs"
# This is the in-container mount point for the Modal volume that stores all
# remote outputs. The user-facing volume name is still configurable via
# --volume-name; this path is where that volume appears inside the container.
OUTPUT_VOLUME_MOUNT = Path("/modal_volumes/rosetta_outputs")
DEFAULT_ROSETTA_IMAGE = "rosettacommons/rosetta:serial"
ROSETTA_DATABASE_CANDIDATES = (
    Path("/rosetta_database"),
    Path("/Rosetta/main/database"),
    Path("/rosetta/database"),
    Path("/database"),
    Path("/home/rosetta/database"),
    Path("/opt/rosetta/database"),
    Path("/usr/local/lib/python3.8/dist-packages/pyrosetta/database"),
    Path("/usr/local/lib/python3.9/dist-packages/pyrosetta/database"),
    Path("/usr/local/lib/python3.10/dist-packages/pyrosetta/database"),
    Path("/usr/local/lib/python3.11/dist-packages/pyrosetta/database"),
    Path("/usr/local/lib/python3.12/dist-packages/pyrosetta/database"),
)
ROSETTA_BIN_CANDIDATES = (
    Path("/rosetta_source_bin"),
    Path("/Rosetta/main/source/bin"),
    Path("/rosetta/source/bin"),
    Path("/source/bin"),
    Path("/home/rosetta/source/bin"),
    Path("/opt/rosetta/source/bin"),
    Path("/usr/local/bin"),
)

PROTOCOL_TO_EXECUTABLE = {
    "relax": "relax.default.linuxgccrelease",
    "interface_analyzer": "InterfaceAnalyzer.default.linuxgccrelease",
}


def _preparse_modal_options(argv: list[str]) -> str:
    # Modal volumes are bound when the module is imported, before the normal
    # argparse entrypoint runs. We therefore pre-parse only --volume-name here
    # so the remote output volume can be mounted correctly.
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    known, _ = parser.parse_known_args(argv)
    return known.volume_name


OUTPUT_VOLUME = modal.Volume.from_name(
    _preparse_modal_options(sys.argv[1:]),
    create_if_missing=True,
)
runtime_image = modal.Image.from_registry(DEFAULT_ROSETTA_IMAGE, add_python="3.11")
app = modal.App("rosetta-generic-launcher", image=runtime_image)


class RosettaLauncherError(RuntimeError):
    """User-facing launcher error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_item_id(source_path: str) -> str:
    # Use the original source path, not file contents, so output folder names
    # remain stable across resume/retry runs for the same input path.
    path_obj = Path(source_path)
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
    stem = path_obj.stem or path_obj.name or "input"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return f"{safe_stem}-{digest}"


def _resolve_run_name(output_dir: str, output_dir_name: str | None) -> str:
    # --output-dir is the local download destination. --output-dir-name is the
    # explicit remote/local run name. If omitted, use the final path component
    # of --output-dir so a simple command still gets a stable run directory.
    if output_dir_name:
        return output_dir_name
    candidate = Path(output_dir).name.strip()
    return candidate or "run"


def _resolve_remote_output_root(protocol: str, output_dir: str, output_dir_name: str | None) -> Path:
    # Remote results are always grouped by protocol first, then run name. This
    # keeps different protocol outputs separated inside the shared Modal volume.
    run_name = _resolve_run_name(output_dir, output_dir_name)
    return (OUTPUT_VOLUME_MOUNT / protocol / run_name).resolve(strict=False)


def _resolve_remote_volume_subpath(protocol: str, output_dir: str, output_dir_name: str | None) -> str:
    run_name = _resolve_run_name(output_dir, output_dir_name)
    return str(Path(protocol) / run_name)


def _resolve_local_output_root(output_dir: str, output_dir_name: str | None) -> Path:
    # Local downloads mirror the remote run directory. If a run name is given,
    # download into output_dir/run_name; otherwise output_dir itself is the run
    # directory.
    root = Path(output_dir).expanduser()
    if output_dir_name:
        root = root / output_dir_name
    return root.resolve(strict=False)


def _resolve_local_download_parent(output_dir: str, output_dir_name: str | None) -> Path:
    # `modal volume get` downloads into the parent directory of the desired run
    # folder. This helper computes that parent so the final on-disk layout is
    # predictable for both named and unnamed runs.
    output_root = Path(output_dir).expanduser()
    if output_dir_name:
        return output_root.resolve(strict=False)
    return output_root.parent.resolve(strict=False)


def _resolve_failed_record_path(root: Path, failed_record_file: str | None) -> Path:
    if failed_record_file:
        path = Path(failed_record_file)
        if not path.is_absolute():
            path = root / path
        return path.resolve(strict=False)
    return (root / "failed_records.txt").resolve(strict=False)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    _ensure_parent(path)
    path.write_bytes(content)


def _prepare_output_root(
    output_root: Path,
    results_root: Path,
    logs_root: Path,
    summary_path: Path,
    failed_record_file: Path,
) -> None:
    # A fresh non-resume run should not mix with historical files from the same
    # run name. Clear the old results/logs and top-level summaries first, but
    # only inside this run directory.
    if results_root.exists():
        shutil.rmtree(results_root)
    if logs_root.exists():
        shutil.rmtree(logs_root)
    for path in (
        summary_path,
        output_root / "aggregated_scores.csv",
        failed_record_file,
    ):
        if path.exists():
            path.unlink()


def _flatten_worker_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result.get("rows", []))
    return rows


def _summarize_worker_results(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "processed": sum(int(result.get("processed", 0)) for result in results),
        "succeeded": sum(int(result.get("succeeded", 0)) for result in results),
        "skipped": sum(int(result.get("skipped", 0)) for result in results),
        "failed": sum(int(result.get("failed", 0)) for result in results),
    }


def _validate_protocol_specific_args(
    protocol: str,
    interface: str,
    fixedchains: str,
    relax_movable_chains: str,
    relax_interface_groups: str,
    relax_movable_ranges: str,
    relax_interface_cutoff: float,
) -> None:
    if protocol not in PROTOCOL_TO_EXECUTABLE:
        supported = ", ".join(sorted(PROTOCOL_TO_EXECUTABLE))
        raise RosettaLauncherError(
            f"Unsupported --protocol '{protocol}'. Supported values: {supported}."
        )
    if protocol == "interface_analyzer" and interface and fixedchains:
        raise RosettaLauncherError(
            "--interface and --fixedchains are mutually exclusive for interface_analyzer."
        )
    if protocol != "relax" and (
        relax_movable_chains or relax_interface_groups or relax_movable_ranges
    ):
        raise RosettaLauncherError(
            "Relax restriction flags are only valid when --protocol relax is used."
        )
    if relax_interface_cutoff <= 0:
        raise RosettaLauncherError("--relax-interface-cutoff must be greater than 0.")


def _resolve_existing_directory(requested: Path, candidates: tuple[Path, ...], label: str) -> Path:
    # Prefer the user-requested path, but fall back to known locations used by
    # common Rosetta container images so the launcher works without hard-coding
    # one single image layout.
    if requested.exists():
        return requested
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise RosettaLauncherError(
        f"{label} path does not exist: {requested}. "
        f"Checked common container locations as well: {candidate_list}. "
        "Check your Modal volume/image mount."
    )


def _extract_chain_ids_from_pdb_bytes(content: bytes) -> list[str]:
    # Read chain IDs from PDB records.
    chain_ids: list[str] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        if not (raw_line.startswith(b"ATOM  ") or raw_line.startswith(b"HETATM")):
            continue
        chain_id = raw_line[21:22].decode("utf-8", errors="ignore").strip() or "_"
        if chain_id not in seen:
            seen.add(chain_id)
            chain_ids.append(chain_id)
    return chain_ids


def _parse_pdb_residues(content: bytes) -> list[dict[str, Any]]:
    """Parse residues and atom coordinates from a PDB byte string."""
    # pose_index is Rosetta's continuous residue numbering across the full pose.
    residues: list[dict[str, Any]] = []
    current_key: tuple[str, int, str] | None = None
    current_residue: dict[str, Any] | None = None
    pose_index = 0

    for raw_line in content.splitlines():
        if not (raw_line.startswith(b"ATOM  ") or raw_line.startswith(b"HETATM")):
            continue
        if len(raw_line) < 54:
            continue
        altloc = raw_line[16:17].decode("utf-8", errors="ignore").strip()
        if altloc not in {"", "A"}:
            continue
        chain_id = raw_line[21:22].decode("utf-8", errors="ignore").strip() or "_"
        resseq_text = raw_line[22:26].decode("utf-8", errors="ignore").strip()
        if not resseq_text:
            continue
        try:
            resseq = int(resseq_text)
            x = float(raw_line[30:38].decode("utf-8", errors="ignore").strip())
            y = float(raw_line[38:46].decode("utf-8", errors="ignore").strip())
            z = float(raw_line[46:54].decode("utf-8", errors="ignore").strip())
        except ValueError:
            continue
        icode = raw_line[26:27].decode("utf-8", errors="ignore").strip()
        atom_name = raw_line[12:16].decode("utf-8", errors="ignore").strip()
        element = raw_line[76:78].decode("utf-8", errors="ignore").strip().upper()
        if not element:
            element = atom_name[:1].upper()
        key = (chain_id, resseq, icode)
        if key != current_key:
            pose_index += 1
            current_residue = {
                "pose_index": pose_index,
                "chain_id": chain_id,
                "resseq": resseq,
                "icode": icode,
                "atoms": [],
                "heavy_atoms": [],
            }
            residues.append(current_residue)
            current_key = key
        if current_residue is None:
            continue
        coord = (x, y, z)
        current_residue["atoms"].append(coord)
        if element != "H":
            current_residue["heavy_atoms"].append(coord)
    return residues


def _split_chain_tokens(value: str) -> list[str]:
    normalized = value.replace(",", " ")
    return [token.strip() for token in normalized.split() if token.strip()]


def _parse_relax_range_spec(spec: str) -> list[tuple[str, int, int]]:
    # Parse CHAIN:START-END selections in PDB numbering.
    selections: list[tuple[str, int, int]] = []
    for token in [piece.strip() for piece in spec.split(",") if piece.strip()]:
        if ":" not in token:
            raise RosettaLauncherError(
                f"Invalid --relax-movable-ranges entry '{token}'. Use CHAIN:START-END."
            )
        chain_id, residue_span = token.split(":", 1)
        chain_id = chain_id.strip()
        if not chain_id:
            raise RosettaLauncherError(
                f"Invalid --relax-movable-ranges entry '{token}'. Missing chain ID."
            )
        if "-" in residue_span:
            start_text, end_text = residue_span.split("-", 1)
        else:
            start_text = residue_span
            end_text = residue_span
        try:
            start = int(start_text.strip())
            end = int(end_text.strip())
        except ValueError as exc:
            raise RosettaLauncherError(
                f"Invalid residue range '{token}'. Use integer PDB residue numbers."
            ) from exc
        if start > end:
            start, end = end, start
        selections.append((chain_id, start, end))
    return selections


def _parse_interface_groups(spec: str) -> tuple[set[str], set[str]]:
    # Parse interface groups such as A_B or AB_C.
    if "_" not in spec:
        raise RosettaLauncherError(
            f"Invalid --relax-interface-groups value '{spec}'. Use GROUP1_GROUP2, e.g. A_B."
        )
    left_text, right_text = spec.split("_", 1)
    left = {char for char in left_text.replace(",", "").replace(" ", "") if char}
    right = {char for char in right_text.replace(",", "").replace(" ", "") if char}
    if not left or not right:
        raise RosettaLauncherError(
            f"Invalid --relax-interface-groups value '{spec}'. Both sides must contain chains."
        )
    return left, right


def _residue_tag(residue: dict[str, Any]) -> str:
    icode = residue["icode"]
    return f"{residue['resseq']}{icode}" if icode else str(residue["resseq"])


def _within_cutoff(residue_a: dict[str, Any], residue_b: dict[str, Any], cutoff_sq: float) -> bool:
    atoms_a = residue_a["heavy_atoms"] or residue_a["atoms"]
    atoms_b = residue_b["heavy_atoms"] or residue_b["atoms"]
    for ax, ay, az in atoms_a:
        for bx, by, bz in atoms_b:
            dx = ax - bx
            dy = ay - by
            dz = az - bz
            if dx * dx + dy * dy + dz * dz <= cutoff_sq:
                return True
    return False


def _select_relax_residue_indices(
    pdb_bytes: bytes,
    relax_movable_chains: str,
    relax_interface_groups: str,
    relax_interface_cutoff: float,
    relax_movable_ranges: str,
) -> set[int]:
    # Build the relax selection in Rosetta pose numbering.
    residues = _parse_pdb_residues(pdb_bytes)
    if not residues:
        raise RosettaLauncherError("Unable to parse residues from the input PDB for relax selection.")

    selected: set[int] = set()
    chain_tokens = set(_split_chain_tokens(relax_movable_chains))
    if chain_tokens:
        selected.update(
            residue["pose_index"] for residue in residues if residue["chain_id"] in chain_tokens
        )

    for chain_id, start, end in _parse_relax_range_spec(relax_movable_ranges):
        selected.update(
            residue["pose_index"]
            for residue in residues
            if residue["chain_id"] == chain_id and start <= residue["resseq"] <= end
        )

    if relax_interface_groups:
        left_chains, right_chains = _parse_interface_groups(relax_interface_groups)
        left_residues = [residue for residue in residues if residue["chain_id"] in left_chains]
        right_residues = [residue for residue in residues if residue["chain_id"] in right_chains]
        if not left_residues or not right_residues:
            raise RosettaLauncherError(
                "--relax-interface-groups did not match chains present in the input PDB."
            )
        cutoff_sq = relax_interface_cutoff * relax_interface_cutoff
        for left_residue in left_residues:
            for right_residue in right_residues:
                if _within_cutoff(left_residue, right_residue, cutoff_sq):
                    selected.add(left_residue["pose_index"])
                    selected.add(right_residue["pose_index"])

    return selected


def _write_relax_selection_files(
    control_dir: Path,
    pdb_bytes: bytes,
    relax_movable_chains: str,
    relax_interface_groups: str,
    relax_interface_cutoff: float,
    relax_movable_ranges: str,
) -> tuple[Path, Path] | tuple[None, None]:
    # Write Rosetta control files for restricted relax.
    if not (relax_movable_chains or relax_interface_groups or relax_movable_ranges):
        return None, None

    residues = _parse_pdb_residues(pdb_bytes)
    selected_pose_indices = _select_relax_residue_indices(
        pdb_bytes,
        relax_movable_chains,
        relax_interface_groups,
        relax_interface_cutoff,
        relax_movable_ranges,
    )
    if not selected_pose_indices:
        raise RosettaLauncherError(
            "Relax restrictions selected zero residues. Check chain IDs, interface groups, or ranges."
        )

    control_dir.mkdir(parents=True, exist_ok=True)
    movemap_path = control_dir / "relax.movemap"
    resfile_path = control_dir / "relax.resfile"

    # MoveMap uses Rosetta pose indices, not chain-local PDB residue numbers.
    movemap_lines = ["RESIDUE * NO", "JUMP * NO"]
    for residue in residues:
        if residue["pose_index"] in selected_pose_indices:
            movemap_lines.append(f"RESIDUE {residue['pose_index']} BBCHI")
    _write_text(movemap_path, "\n".join(movemap_lines) + "\n")

    # Resfile keeps PDB residue numbering plus chain IDs.
    resfile_lines = ["NATRO", "start"]
    for residue in residues:
        if residue["pose_index"] in selected_pose_indices:
            resfile_lines.append(f"{_residue_tag(residue)} {residue['chain_id']} NATAA")
    _write_text(resfile_path, "\n".join(resfile_lines) + "\n")
    return movemap_path, resfile_path


def _split_fixedchains(fixedchains: str) -> list[str]:
    normalized = fixedchains.replace(",", " ")
    return [token for token in normalized.split() if token]


def _validate_interface_analyzer_inputs(
    items: list[dict[str, Any]],
    interface: str,
    fixedchains: str,
) -> None:
    # Require explicit grouping for multichain inputs.
    if not items:
        return

    multichain_sources: list[str] = []
    for item in items:
        chain_ids = _extract_chain_ids_from_pdb_bytes(item["content"])
        if len(chain_ids) > 2:
            multichain_sources.append(item["source_path"])

    if multichain_sources and not (interface or fixedchains):
        preview = ", ".join(multichain_sources[:3])
        if len(multichain_sources) > 3:
            preview += ", ..."
        raise RosettaLauncherError(
            "Multichain input detected for interface_analyzer. "
            "Provide --interface or --fixedchains. "
            f"Examples: {preview}"
        )


def _build_rosetta_command(
    executable: Path,
    database: Path,
    protocol: str,
    input_pdb: Path,
    output_dir: Path,
    interface: str,
    fixedchains: str,
    extra_args: str,
    relax_movemap: Path | None,
    relax_resfile: Path | None,
) -> list[str]:
    """Build one Rosetta CLI for a staged input."""
    # Build the exact Rosetta command.
    command = [
        str(executable),
        "-database",
        str(database),
        "-s",
        str(input_pdb),
        "-out:path:all",
        str(output_dir),
        "-out:file:scorefile",
        str(output_dir / "score.sc"),
        "-overwrite",
    ]

    if protocol == "interface_analyzer":
        # Two-chain can be inferred; multichain needs grouping.
        if interface:
            command.extend(["-interface", interface])
        elif fixedchains:
            fixedchain_tokens = _split_fixedchains(fixedchains)
            if not fixedchain_tokens:
                raise RosettaLauncherError(
                    "--fixedchains was provided but no chain IDs were parsed from it."
                )
            command.extend(["-fixedchains", *fixedchain_tokens])
    # Relax uses the full pose unless restriction files are supplied.
    if protocol == "relax" and relax_movemap and relax_resfile:
        command.extend(
            [
                "-in:file:movemap",
                str(relax_movemap),
                "-packing:resfile",
                str(relax_resfile),
                "-relax:respect_resfile",
            ]
        )

    if extra_args:
        # Safely split extra Rosetta flags.
        command.extend(shlex.split(extra_args))
    if protocol == "interface_analyzer":
        # Add regular scores and packstat by default.
        if "-add_regular_scores_to_scorefile" not in command:
            command.append("-add_regular_scores_to_scorefile")
        if "-compute_packstat" not in command:
            command.append("-compute_packstat")
    return command


def _expected_outputs_exist(protocol: str, output_dir: Path) -> bool:
    # Resume checks outputs, not just directory existence.
    success_marker = output_dir / "SUCCESS.json"
    scorefile = output_dir / "score.sc"
    pdb_outputs = list(output_dir.glob("*.pdb"))

    if not success_marker.exists():
        return False
    if protocol == "interface_analyzer":
        return scorefile.exists()
    if protocol == "relax":
        return scorefile.exists() or bool(pdb_outputs)
    return False


def _load_pdb_inputs_from_remote(path_str: str) -> list[dict[str, Any]]:
    # Load inputs from a path inside the Modal container.
    input_path = Path(path_str)
    if not input_path.exists():
        raise RosettaLauncherError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdb":
            raise RosettaLauncherError(f"Input file must be a .pdb file: {input_path}")
        return [
            {
                "source_path": str(input_path.resolve(strict=False)),
                "display_name": input_path.name,
                "content": input_path.read_bytes(),
            }
        ]

    pdb_files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdb")
    if not pdb_files:
        raise RosettaLauncherError(f"No .pdb files found in input directory: {input_path}")
    return [
        {
            "source_path": str(pdb.resolve(strict=False)),
            "display_name": pdb.name,
            "content": pdb.read_bytes(),
        }
        for pdb in pdb_files
    ]


def _chunk_items(items: list[dict[str, Any]], num_chunks: int) -> list[list[dict[str, Any]]]:
    chunks = [[] for _ in range(num_chunks)]
    for index, item in enumerate(items):
        chunks[index % num_chunks].append(item)
    return [chunk for chunk in chunks if chunk]


def _slice_batch(items: list[dict[str, Any]], batch_index: int, num_batches: int) -> list[dict[str, Any]]:
    # Stable modulo slicing for batch runs.
    if num_batches < 1:
        raise RosettaLauncherError("--num-batches must be at least 1.")
    if batch_index < 0 or batch_index >= num_batches:
        raise RosettaLauncherError(
            f"--batch-index must be in [0, {num_batches - 1}] when --num-batches={num_batches}."
        )
    return [item for idx, item in enumerate(items) if idx % num_batches == batch_index]


def _stage_input_file(staging_dir: Path, item: dict[str, Any]) -> Path:
    # Copy one input into the remote temp working area.
    item_id = _stable_item_id(item["source_path"])
    filename = Path(item["display_name"]).name or f"{item_id}.pdb"
    staged_path = staging_dir / f"{item_id}_{filename}"
    _write_bytes(staged_path, item["content"])
    return staged_path


def _append_failure_record(failed_record_file: Path, source_path: str, message: str) -> None:
    _ensure_parent(failed_record_file)
    with failed_record_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()}\t{source_path}\t{message}\n")


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    # Write one summary row per input item.
    _ensure_parent(summary_path)
    fieldnames = [
        "source_path",
        "item_id",
        "protocol",
        "status",
        "return_code",
        "output_dir",
        "stdout_log",
        "stderr_log",
        "message",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_rosetta_scorefile(scorefile: Path) -> list[dict[str, str]]:
    # Parse Rosetta SCORE: lines into dictionaries.
    if not scorefile.exists():
        return []

    lines = [
        line.strip()
        for line in scorefile.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip().startswith("SCORE:")
    ]
    if len(lines) < 2:
        return []

    header_tokens = lines[0].split()[1:]
    records: list[dict[str, str]] = []
    for line in lines[1:]:
        value_tokens = line.split()[1:]
        if len(value_tokens) != len(header_tokens):
            continue
        records.append(dict(zip(header_tokens, value_tokens)))
    return records


INTERFACE_ANALYZER_INTERFACE_COLUMNS = {
    "complex_normalized",
    "dG_cross",
    "dG_cross/dSASAx100",
    "dG_separated",
    "dG_separated/dSASAx100",
    "dSASA_hphobic",
    "dSASA_int",
    "dSASA_polar",
    "delta_unsatHbonds",
    "hbond_E_fraction",
    "hbonds_int",
    "nres_all",
    "nres_int",
    "packstat",
    "per_residue_energy_int",
    "sc_value",
    "side1_normalized",
    "side1_score",
    "side2_normalized",
    "side2_score",
    "yhh_planarity",
}

ROSETTA_ENERGY_COLUMNS = {
    "total_score",
    "dslf_fa13",
    "fa_atr",
    "fa_dun",
    "fa_elec",
    "fa_intra_rep",
    "fa_intra_sol_xover4",
    "fa_rep",
    "fa_sol",
    "hbond_bb_sc",
    "hbond_lr_bb",
    "hbond_sc",
    "hbond_sr_bb",
    "lk_ball_wtd",
    "omega",
    "p_aa_pp",
    "pro_close",
    "rama_prepro",
    "ref",
}


def _categorize_score_column(protocol: str, column: str) -> str:
    if column == "description":
        return "output"
    if protocol == "interface_analyzer":
        if column in INTERFACE_ANALYZER_INTERFACE_COLUMNS:
            return "interface_metrics"
        if column in ROSETTA_ENERGY_COLUMNS:
            return "energy_terms"
    if column in ROSETTA_ENERGY_COLUMNS:
        return "energy_terms"
    return "other_metrics"


def _prefixed_score_column(protocol: str, column: str) -> str:
    if protocol == "interface_analyzer":
        if column in INTERFACE_ANALYZER_INTERFACE_COLUMNS:
            return f"inter_{column}"
        if column in ROSETTA_ENERGY_COLUMNS:
            return f"ener_{column}"
    return column


def _ordered_score_columns(protocol: str, columns: list[str]) -> list[str]:
    if protocol != "interface_analyzer":
        return columns

    # Put key interface metrics before regular energy terms.
    preferred_interface_columns = [
        "dG_separated",
        "dSASA_int",
        "delta_unsatHbonds",
        "packstat",
        "sc_value",
    ]
    preferred_energy_columns = ["total_score"]

    interface_columns = [column for column in columns if column in INTERFACE_ANALYZER_INTERFACE_COLUMNS]
    energy_columns = [column for column in columns if column in ROSETTA_ENERGY_COLUMNS]
    output_columns = [column for column in columns if column == "description"]
    other_columns = [
        column
        for column in columns
        if column not in INTERFACE_ANALYZER_INTERFACE_COLUMNS
        and column not in ROSETTA_ENERGY_COLUMNS
        and column != "description"
    ]

    ordered_interface_columns = [
        *[column for column in preferred_interface_columns if column in interface_columns],
        *[column for column in interface_columns if column not in preferred_interface_columns],
    ]
    ordered_energy_columns = [
        *[column for column in preferred_energy_columns if column in energy_columns],
        *[column for column in energy_columns if column not in preferred_energy_columns],
    ]
    return [*ordered_interface_columns, *ordered_energy_columns, *other_columns, *output_columns]


def _write_aggregated_scores(output_root: Path, rows: list[dict[str, Any]]) -> Path | None:
    # Merge per-item scorefiles into one run-level CSV.
    aggregated_rows: list[dict[str, str]] = []
    score_columns: list[str] = []
    column_name_map: dict[str, str] = {}

    for row in rows:
        if row["status"] not in {"success", "skipped"}:
            continue
        scorefile = Path(row["output_dir"]) / "score.sc"
        for parsed_row in _parse_rosetta_scorefile(scorefile):
            aggregated_row = {
                "source_path": row["source_path"],
                "item_id": row["item_id"],
                "protocol": row["protocol"],
                "scorefile": str(scorefile),
            }
            for key, value in parsed_row.items():
                mapped_key = _prefixed_score_column(row["protocol"], key)
                aggregated_row[mapped_key] = value
                if key not in column_name_map:
                    column_name_map[key] = mapped_key
            aggregated_rows.append(aggregated_row)
            for key in parsed_row:
                if key not in score_columns:
                    score_columns.append(key)

    if not aggregated_rows:
        return None

    aggregate_path = output_root / "aggregated_scores.csv"
    ordered_score_columns = _ordered_score_columns(aggregated_rows[0]["protocol"], score_columns)
    fieldnames = [
        "item_id",
        *(column_name_map[column] for column in ordered_score_columns),
        "source_path",
        "scorefile",
        "protocol",
    ]
    # Keep item_id first, then scores, then provenance fields.
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregated_rows)
    return aggregate_path


@app.function(
    volumes={str(OUTPUT_VOLUME_MOUNT): OUTPUT_VOLUME},
    # One worker = one CPU. Use --parallel-batches to scale directory runs out.
    cpu=1,
    timeout=24 * 60 * 60,
    image=runtime_image,
)
def prepare_rosetta_run(job: dict[str, Any]) -> dict[str, str]:
    protocol = job["protocol"]
    output_root = _resolve_remote_output_root(protocol, job["output_dir"], job.get("output_dir_name"))
    results_root = output_root / "results"
    logs_root = output_root / "logs"
    failed_record_file = _resolve_failed_record_path(output_root, job.get("failed_record_file"))
    summary_path = output_root / "summary.csv"

    output_root.mkdir(parents=True, exist_ok=True)
    if not job.get("resume"):
        _prepare_output_root(
            output_root=output_root,
            results_root=results_root,
            logs_root=logs_root,
            summary_path=summary_path,
            failed_record_file=failed_record_file,
        )
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    OUTPUT_VOLUME.commit()
    return {"output_root": str(output_root)}


@app.function(
    volumes={str(OUTPUT_VOLUME_MOUNT): OUTPUT_VOLUME},
    cpu=1,
    timeout=24 * 60 * 60,
    image=runtime_image,
)
def finalize_rosetta_run(job: dict[str, Any]) -> dict[str, str]:
    protocol = job["protocol"]
    output_root = _resolve_remote_output_root(protocol, job["output_dir"], job.get("output_dir_name"))
    summary_path = output_root / "summary.csv"
    rows = job["rows"]
    _write_summary(summary_path, rows)
    aggregate_scores_path = _write_aggregated_scores(output_root, rows)
    OUTPUT_VOLUME.commit()
    return {
        "summary_csv": str(summary_path),
        "aggregated_scores_csv": str(aggregate_scores_path) if aggregate_scores_path else "",
    }


@app.function(
    volumes={str(OUTPUT_VOLUME_MOUNT): OUTPUT_VOLUME},
    # One worker = one CPU. Use --parallel-batches to scale directory runs out.
    cpu=1,
    timeout=24 * 60 * 60,
    image=runtime_image,
)
def run_rosetta_job(job: dict[str, Any]) -> dict[str, Any]:
    # Remote runner: resolve Rosetta, execute it, and save outputs.
    protocol = job["protocol"]
    interface = job.get("interface", "")
    fixedchains = job.get("fixedchains", "")
    relax_movable_chains = job.get("relax_movable_chains", "")
    relax_interface_groups = job.get("relax_interface_groups", "")
    relax_movable_ranges = job.get("relax_movable_ranges", "")
    relax_interface_cutoff = float(job.get("relax_interface_cutoff", 8.0))
    _validate_protocol_specific_args(
        protocol,
        interface,
        fixedchains,
        relax_movable_chains,
        relax_interface_groups,
        relax_movable_ranges,
        relax_interface_cutoff,
    )

    requested_rosetta_bin = Path(job["rosetta_bin"])
    requested_rosetta_database = Path(job["rosetta_database"])
    output_root = _resolve_remote_output_root(protocol, job["output_dir"], job.get("output_dir_name"))
    results_root = output_root / "results"
    logs_root = output_root / "logs"
    failed_record_file = _resolve_failed_record_path(output_root, job.get("failed_record_file"))
    summary_path = output_root / "summary.csv"

    output_root.mkdir(parents=True, exist_ok=True)
    if job.get("prepare_output_root", True):
        if not job.get("resume"):
            # Start clean when reusing a run name without --resume.
            _prepare_output_root(
                output_root=output_root,
                results_root=results_root,
                logs_root=logs_root,
                summary_path=summary_path,
                failed_record_file=failed_record_file,
            )
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    rosetta_database = _resolve_existing_directory(
        requested_rosetta_database,
        ROSETTA_DATABASE_CANDIDATES,
        "Rosetta database",
    )
    rosetta_bin = _resolve_existing_directory(
        requested_rosetta_bin,
        ROSETTA_BIN_CANDIDATES,
        "Rosetta bin",
    )

    executable = rosetta_bin / PROTOCOL_TO_EXECUTABLE[protocol]
    if not executable.exists():
        raise RosettaLauncherError(
            "Rosetta executable does not exist: "
            f"{executable}. Check --rosetta-bin and your Modal mount/image."
        )

    if not os.access(executable, os.X_OK):
        raise RosettaLauncherError(f"Rosetta executable is not executable: {executable}")

    if job.get("staged_inputs"):
        # Local inputs uploaded by the caller.
        input_items = job["staged_inputs"]
    else:
        # Input path already exists in the container.
        input_items = _load_pdb_inputs_from_remote(job["input_path"])
    input_items = _slice_batch(input_items, job["batch_index"], job["num_batches"])
    if protocol == "interface_analyzer":
        _validate_interface_analyzer_inputs(input_items, interface, fixedchains)

    if not input_items:
        if job.get("allow_empty_batches"):
            return {
                "protocol": protocol,
                "output_root": str(output_root),
                "remote_volume_subpath": _resolve_remote_volume_subpath(
                    protocol,
                    job["output_dir"],
                    job.get("output_dir_name"),
                ),
                "summary_csv": str(summary_path),
                "aggregated_scores_csv": "",
                "failed_record_file": str(failed_record_file),
                "processed": 0,
                "succeeded": 0,
                "skipped": 0,
                "failed": 0,
                "rows": [],
            }
        raise RosettaLauncherError(
            f"No input items selected for batch {job['batch_index']} / {job['num_batches']}."
        )

    rows: list[dict[str, Any]] = []
    processed = 0
    success_count = 0
    skipped_count = 0
    failed_count = 0

    with tempfile.TemporaryDirectory(prefix="rosetta_launcher_") as tmpdir_name:
        # Stage inputs in a temporary remote working area.
        tmpdir = Path(tmpdir_name)
        staged_inputs_dir = tmpdir / "inputs"
        staged_inputs_dir.mkdir(parents=True, exist_ok=True)

        for item in input_items:
            processed += 1
            source_path = item["source_path"]
            item_id = _stable_item_id(source_path)
            item_output_dir = results_root / item_id
            stdout_log = logs_root / f"{item_id}.stdout.log"
            stderr_log = logs_root / f"{item_id}.stderr.log"
            command_log = logs_root / f"{item_id}.command.txt"
            success_marker = item_output_dir / "SUCCESS.json"

            row = {
                "source_path": source_path,
                "item_id": item_id,
                "protocol": protocol,
                "status": "pending",
                "return_code": "",
                "output_dir": str(item_output_dir),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "message": "",
            }

            if job.get("resume") and _expected_outputs_exist(protocol, item_output_dir):
                # Skip only completed items.
                row["status"] = "skipped"
                row["message"] = "Skipped because resume detected completed outputs."
                rows.append(row)
                skipped_count += 1
                continue

            item_output_dir.mkdir(parents=True, exist_ok=True)
            staged_pdb = _stage_input_file(staged_inputs_dir, item)
            relax_control_dir = item_output_dir / "relax_controls"
            relax_movemap, relax_resfile = _write_relax_selection_files(
                relax_control_dir,
                item["content"],
                relax_movable_chains,
                relax_interface_groups,
                relax_interface_cutoff,
                relax_movable_ranges,
            )
            # Build the per-item Rosetta command.
            command = _build_rosetta_command(
                executable=executable,
                database=rosetta_database,
                protocol=protocol,
                input_pdb=staged_pdb,
                output_dir=item_output_dir,
                interface=interface,
                fixedchains=fixedchains,
                extra_args=job.get("extra_args", ""),
                relax_movemap=relax_movemap,
                relax_resfile=relax_resfile,
            )

            _write_text(command_log, " ".join(shlex.quote(part) for part in command) + "\n")
            # Run the Rosetta executable.
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(item_output_dir),
            )
            _write_text(stdout_log, completed.stdout)
            _write_text(stderr_log, completed.stderr)
            # Always save stdout/stderr.

            row["return_code"] = str(completed.returncode)

            if completed.returncode != 0:
                message = (
                    f"Rosetta exited with return code {completed.returncode}. "
                    f"See {stderr_log} and {stdout_log}."
                )
                row["status"] = "failed"
                row["message"] = message
                rows.append(row)
                failed_count += 1
                _append_failure_record(failed_record_file, source_path, message)
                OUTPUT_VOLUME.commit()
                continue

            success_payload = {
                "protocol": protocol,
                "source_path": source_path,
                "input_cli": job.get("input_cli", ""),
                "completed_at": _utc_now(),
                "return_code": completed.returncode,
                "output_dir": str(item_output_dir),
            }
            # SUCCESS.json is used by resume.
            _write_text(success_marker, json.dumps(success_payload, indent=2, sort_keys=True) + "\n")

            if not _expected_outputs_exist(protocol, item_output_dir):
                message = (
                    "Rosetta finished with exit code 0 but expected outputs were not found. "
                    f"Inspect {item_output_dir}, {stdout_log}, and {stderr_log}."
                )
                row["status"] = "failed"
                row["message"] = message
                rows.append(row)
                failed_count += 1
                _append_failure_record(failed_record_file, source_path, message)
                OUTPUT_VOLUME.commit()
                continue

            row["status"] = "success"
            row["message"] = "Completed successfully."
            rows.append(row)
            success_count += 1
            OUTPUT_VOLUME.commit()

    aggregate_scores_path: Path | None = None
    if job.get("finalize_output_root", True):
        _write_summary(summary_path, rows)
        # Write the run-level score summary.
        aggregate_scores_path = _write_aggregated_scores(output_root, rows)
        OUTPUT_VOLUME.commit()
    return {
        "protocol": protocol,
        "output_root": str(output_root),
        "remote_volume_subpath": _resolve_remote_volume_subpath(
            protocol,
            job["output_dir"],
            job.get("output_dir_name"),
        ),
        "summary_csv": str(summary_path),
        "aggregated_scores_csv": str(aggregate_scores_path) if aggregate_scores_path else "",
        "failed_record_file": str(failed_record_file),
        "processed": processed,
        "succeeded": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "rows": rows,
    }


def _load_local_input_payload(input_path: Path) -> list[dict[str, Any]]:
    # Read local inputs for upload to Modal.
    if not input_path.exists():
        raise RosettaLauncherError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdb":
            raise RosettaLauncherError(f"Input file must be a .pdb file: {input_path}")
        return [
            {
                "source_path": str(input_path.resolve(strict=False)),
                "display_name": input_path.name,
                "content": input_path.read_bytes(),
            }
        ]

    pdb_files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdb")
    if not pdb_files:
        raise RosettaLauncherError(f"No .pdb files found in input directory: {input_path}")
    return [
        {
            "source_path": str(pdb.resolve(strict=False)),
            "display_name": pdb.name,
            "content": pdb.read_bytes(),
        }
        for pdb in pdb_files
    ]


@app.local_entrypoint()
def main(
    input_path: str,
    protocol: str,
    output_dir: str = "runs",
    output_dir_name: str = "",
    rosetta_bin: str = DEFAULT_ROSETTA_BIN,
    rosetta_database: str = DEFAULT_ROSETTA_DATABASE,
    volume_name: str = DEFAULT_VOLUME_NAME,
    resume: bool = False,
    failed_record_file: str = "",
    extra_args: str = "",
    parallel_batches: int = 1,
    interface: str = "",
    fixedchains: str = "",
    relax_movable_chains: str = "",
    relax_interface_groups: str = "",
    relax_interface_cutoff: float = 8.0,
    relax_movable_ranges: str = "",
    batch_index: int = 0,
    num_batches: int = 1,
) -> None:
    # Local CLI entrypoint.
    if parallel_batches < 1:
        raise SystemExit("--parallel-batches must be at least 1.")
    if parallel_batches > 1 and (batch_index != 0 or num_batches != 1):
        raise SystemExit(
            "--parallel-batches cannot be combined with manual --batch-index/--num-batches."
        )
    _validate_protocol_specific_args(
        protocol,
        interface,
        fixedchains,
        relax_movable_chains,
        relax_interface_groups,
        relax_movable_ranges,
        relax_interface_cutoff,
    )

    staged_inputs: list[dict[str, Any]] | None = None
    input_path_obj = Path(input_path)
    if input_path_obj.exists():
        # Upload local inputs automatically.
        staged_inputs = _load_local_input_payload(input_path_obj)
        if parallel_batches == 1:
            staged_inputs = _slice_batch(staged_inputs, batch_index, num_batches)
            if not staged_inputs:
                raise SystemExit(
                    f"No local input items selected for batch {batch_index} / {num_batches}."
                )
        if protocol == "interface_analyzer" and staged_inputs:
            _validate_interface_analyzer_inputs(staged_inputs, interface, fixedchains)

    # Package the remote job payload.
    job = {
        "input_path": input_path,
        "staged_inputs": staged_inputs,
        "protocol": protocol,
        "input_cli": " ".join(["modal", "run", Path(__file__).name, *[shlex.quote(arg) for arg in sys.argv[1:]]]),
        "output_dir": output_dir,
        "output_dir_name": output_dir_name,
        "rosetta_bin": rosetta_bin,
        "rosetta_database": rosetta_database,
        "resume": resume,
        "failed_record_file": failed_record_file,
        "extra_args": extra_args,
        "interface": interface,
        "fixedchains": fixedchains,
        "relax_movable_chains": relax_movable_chains,
        "relax_interface_groups": relax_interface_groups,
        "relax_interface_cutoff": relax_interface_cutoff,
        "relax_movable_ranges": relax_movable_ranges,
        "batch_index": 0 if staged_inputs is not None else batch_index,
        "num_batches": 1 if staged_inputs is not None else num_batches,
    }

    if parallel_batches == 1:
        result = run_rosetta_job.remote(job)
        worker_rows = result.pop("rows", [])
        if worker_rows:
            result["processed"] = len(worker_rows)
    else:
        prepare_rosetta_run.remote(job)
        worker_jobs: list[dict[str, Any]] = []
        if staged_inputs is not None:
            for chunk in _chunk_items(staged_inputs, parallel_batches):
                worker_job = dict(job)
                worker_job["staged_inputs"] = chunk
                worker_job["prepare_output_root"] = False
                worker_job["finalize_output_root"] = False
                worker_job["allow_empty_batches"] = True
                worker_jobs.append(worker_job)
        else:
            for worker_batch_index in range(parallel_batches):
                worker_job = dict(job)
                worker_job["prepare_output_root"] = False
                worker_job["finalize_output_root"] = False
                worker_job["batch_index"] = worker_batch_index
                worker_job["num_batches"] = parallel_batches
                worker_jobs.append(worker_job)

        worker_calls = [run_rosetta_job.spawn(worker_job) for worker_job in worker_jobs]
        worker_results = modal.FunctionCall.gather(*worker_calls)
        rows = _flatten_worker_rows(worker_results)
        finalize_result = finalize_rosetta_run.remote(
            {
                "protocol": protocol,
                "output_dir": output_dir,
                "output_dir_name": output_dir_name,
                "rows": rows,
            }
        )
        counts = _summarize_worker_results(worker_results)
        result = {
            "protocol": protocol,
            "output_root": str(
                _resolve_remote_output_root(protocol, output_dir, output_dir_name)
            ),
            "remote_volume_subpath": _resolve_remote_volume_subpath(
                protocol,
                output_dir,
                output_dir_name,
            ),
            "summary_csv": finalize_result["summary_csv"],
            "aggregated_scores_csv": finalize_result["aggregated_scores_csv"],
            "failed_record_file": str(
                _resolve_failed_record_path(
                    _resolve_remote_output_root(protocol, output_dir, output_dir_name),
                    failed_record_file,
                )
            ),
            **counts,
            "parallel_batches": parallel_batches,
        }
    local_output_root = _resolve_local_output_root(output_dir, output_dir_name)
    local_download_parent = _resolve_local_download_parent(output_dir, output_dir_name)
    local_download_parent.mkdir(parents=True, exist_ok=True)
    # Download the finished run directory from the Modal volume.
    download_command = [
        "modal",
        "volume",
        "get",
        volume_name,
        result["remote_volume_subpath"],
        str(local_download_parent),
        "--force",
    ]
    download_completed = subprocess.run(download_command, capture_output=True, text=True)
    if download_completed.returncode != 0:
        message = (
            "Rosetta job finished, but downloading outputs from the Modal volume failed. "
            f"Command: {' '.join(shlex.quote(part) for part in download_command)}\n"
            f"stdout:\n{download_completed.stdout}\n"
            f"stderr:\n{download_completed.stderr}"
        )
        raise SystemExit(message)

    result["local_output_dir"] = str(local_output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed"] > 0:
        raise SystemExit(1)
