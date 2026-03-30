"""Rosetta source repo: <https://github.com/RosettaCommons/rosetta>.

## Overview

- Modal wrapper around Rosetta `relax` and `interface_analyzer`.
- Input can be a single `.pdb` file or a directory of `.pdb` files.
- Supports skip/resume for completed items under the same `--output-dir-name`.
- Rosetta runs on CPU and does not require GPU.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | **Required** | Path to one `.pdb` file or a directory of `.pdb` files. |
| `--protocol` | **Required** | `relax` or `interface_analyzer`. |
| `--output-dir` | **Required** | Local directory used for the final download. |
| `--output-dir-name` | **Required** | Run name for remote and local outputs. |
| `--parallel-batches` | `1` | Split a directory input across this many Modal workers. |
| `--interface` | `None` | `interface_analyzer` grouping such as `A_B` or `AB_C`. Optional for two-chain inputs; multichain inputs must provide `--interface`. |
| `--relax-selection` | `None` | `relax` only. One selection spec: `chains:A,B`, `interface:A_B` or `interface:A_B@4.0`, or `ranges:A:25-40,B:10-12`. |

For the full Rosetta application details behind `relax` and `interface_analyzer`,
see <https://docs.rosettacommons.org/docs/latest/application_documentation/structure_prediction/relax>
and <https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer>.

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `Rosetta` | Name of the Modal app to use. |
| `TIMEOUT` | `86400` | Timeout for each Modal function in seconds. |

## Outputs

- Remote outputs are written to `/rosetta-outputs/<protocol>/<run_name>`.
- `relax` uses a fixed RosettaScripts XML stored at `/rosetta-data/relax.xml` in `ROSETTA_DATA`.
- Each successful item writes its own Rosetta `score.sc` under `results/<item_id>/`.
- The main run writes `summary.csv`; run `python score_postprocess.py <run_dir>` later to generate `aggregated_scores.csv`.
- The final run directory is downloaded to `<output-dir>/<run_name>`.
- Re-running the same command with the same `--output-dir-name` resumes from completed items and saves results to the same output directory.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from modal import App, Image, Volume


##########################################
# Modal configs
##########################################
TIMEOUT = int(os.environ.get("TIMEOUT", str(24 * 60 * 60)))
APP_NAME = os.environ.get("MODAL_APP", "Rosetta")

OUTPUTS_VOLUME_NAME = "rosetta-outputs"
OUTPUTS_VOLUME = Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)
OUTPUTS_DIR = Path("/rosetta-outputs")
ROSETTA_DATA_VOLUME_NAME = "rosetta-data"
ROSETTA_DATA = Volume.from_name(ROSETTA_DATA_VOLUME_NAME, create_if_missing=True)
ROSETTA_DATA_DIR = Path("/rosetta-data")
ROSETTA_RELAX_XML_PATH = ROSETTA_DATA_DIR / "relax.xml"

ROSETTA_IMAGE = "rosettacommons/rosetta:serial"
ROSETTA_BIN_DIR = Path("/usr/local/bin")
ROSETTA_DATABASE_DIR = Path("/usr/local/database")
PROTOCOL_TO_EXECUTABLE = {
    "interface_analyzer": "InterfaceAnalyzer.default.linuxgccrelease",
    "relax": "rosetta_scripts.default.linuxgccrelease",
}
# Poll remote FunctionCall completion in bounded waits so we can renew the run lock.
FUNCTION_CALL_POLL_TIMEOUT = 30.0
FUNCTION_CALL_POLL_INTERVAL = 5.0

##########################################
# Image and app definitions
##########################################
runtime_image = Image.from_registry(ROSETTA_IMAGE, add_python="3.11")
app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
#---------------------------------------------------------------
# Build the standard run-level output paths for one Rosetta run.
#---------------------------------------------------------------
def _run_paths(protocol: str, run_name: str) -> dict[str, Path]:
    run_root = OUTPUTS_DIR / protocol / run_name
    return {
        "run_root": run_root,
        "lock_dir": run_root / ".run.lock",
        "lock_file": run_root / ".run.lock" / "lease.json",
        "results_root": run_root / "results",
        "logs_root": run_root / "logs",
        "summary_csv": run_root / "summary.csv",
        "failed_records": run_root / "failed_records.txt",
    }


#---------------------------------------------------------------
# Translate one relax-selection string into RosettaScripts vars.
#---------------------------------------------------------------
def _build_relax_script_vars(relax_selection: str) -> dict[str, str]:
    base = {"chain_ids": "A", "interface_cutoff": "4.0", "left_chains": "A", "pdb_ranges": "1A", "right_chains": "B"}
    norm = lambda t: ",".join(s.strip() for s in t.replace(",", " ").split() if s.strip())  # noqa: E731
    
    if not (rs := relax_selection.strip()):
        return {**base, "active_selector": "all_selector", "inactive_selector": "false_selector"}
    
    kind, _, val = rs.partition(":")
    kind, val = kind.strip(), val.strip()
    if kind not in {"chains", "interface", "ranges"}:
        raise RuntimeError(f"Unsupported --relax-selection kind '{kind}'. Use chains, interface, or ranges.")
    
    result = {**base}
    
    if kind == "chains":
        cid = norm(val)
        if not cid:
            raise RuntimeError("Relax chains selection matched zero chain IDs. Check the chains expression.")
        result.update({"active_selector": "chains_selector", "chain_ids": cid, "inactive_selector": "not_chains_selector"})
    
    elif kind == "ranges":
        ranges = []
        for tok in (s.strip() for s in val.split(",") if s.strip()):
            if ":" not in tok:
                raise RuntimeError(f"Invalid ranges selection entry '{tok}'. Use CHAIN:START-END.")
            cid, span = tok.split(":", 1)
            cid = cid.strip()
            if not cid:
                raise RuntimeError(f"Invalid ranges selection entry '{tok}'. Missing chain ID.")
            st, et = (span.split("-", 1) if "-" in span else (span, span))
            try:
                s, e = int(st.strip()), int(et.strip())
            except ValueError as e:
                raise RuntimeError(f"Invalid residue range '{tok}'. Use integer PDB residue numbers.") from e
            ranges.append(f"{min(s,e)}{cid}-{max(s,e)}{cid}")
        if not ranges:
            raise RuntimeError("Relax ranges selection matched zero ranges. Check the ranges expression.")
        result.update({"active_selector": "ranges_selector", "inactive_selector": "not_ranges_selector", "pdb_ranges": ",".join(ranges)})
    
    else:  # interface
        cutoff = 4.0
        if "@" in val:
            val, ct = val.rsplit("@", 1)
            try:
                cutoff = float(ct.strip())
            except ValueError as e:
                raise RuntimeError(f"Invalid interface cutoff '{ct}'. Use a number such as 8.0.") from e
            if cutoff <= 0:
                raise RuntimeError("Interface cutoff must be greater than 0.")
        if "_" not in val:
            raise RuntimeError(f"Invalid interface selection '{val}'. Use interface:A_B or interface:A_B@4.0.")
        lt, rt = (p.strip() for p in val.split("_", 1))
        lc = norm(lt) or ",".join(c for c in lt if not c.isspace())
        rc = norm(rt) or ",".join(c for c in rt if not c.isspace())
        if not lc or not rc:
            raise RuntimeError(f"Invalid interface selection '{val}'. Use interface:A_B or interface:A_B@4.0.")
        result.update({"active_selector": "interface_selector", "inactive_selector": "not_interface_selector", "interface_cutoff": str(cutoff), "left_chains": lc, "right_chains": rc})
    
    return result


#---------------------------------------------------------------
# Build one Rosetta CLI command for relax or interface_analyzer.
#---------------------------------------------------------------
def _build_rosetta_command(
    protocol: str,
    input_pdb: Path,
    output_dir: Path,
    interface: str,
    relax_script_xml: Path | None,
    relax_script_vars: dict[str, str] | None,
) -> list[str]:
    executable = ROSETTA_BIN_DIR / PROTOCOL_TO_EXECUTABLE[protocol]
    command = [
        str(executable),
        "-database",
        str(ROSETTA_DATABASE_DIR),
        "-s",
        str(input_pdb),
        "-out:path:all",
        str(output_dir),
        "-out:file:scorefile",
        str(output_dir / "score.sc"),
        "-overwrite",
    ]

    if protocol == "interface_analyzer":
        if interface:
            command.extend(["-interface", interface])
        command.extend(["-add_regular_scores_to_scorefile", "-compute_packstat"])

    if protocol == "relax":
        if relax_script_xml:
            command.extend(["-parser:protocol", str(relax_script_xml)])
        if relax_script_vars:
            for key, value in relax_script_vars.items():
                command.extend(["-parser:script_vars", f"{key}={value}"])
        command.extend(["-nstruct", "1"])

    return command

#---------------------------------------------------------------
# Item output state helpers.
#---------------------------------------------------------------
# Check whether one item already has complete resumable outputs
def _outputs_exist(protocol: str, output_dir: Path) -> bool:
    if not (output_dir / "SUCCESS.json").exists():
        return False
    if not (output_dir / "score.sc").exists():
        return False
    return protocol != "relax" or any(output_dir.glob("*.pdb"))


#---------------------------------------------------------------
# Run-level reporting helpers.
#---------------------------------------------------------------
# Write the run-level summary.csv for all processed items
def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
##########################################
# Inference functions
##########################################
#------------------------------------------
# Manage one exclusive run-level lock.
#------------------------------------------
@app.function(
    volumes={str(OUTPUTS_DIR): OUTPUTS_VOLUME},
    cpu=1,
    timeout=TIMEOUT,
    image=runtime_image,
)
def manage_rosetta_run_lock(
    job: dict[str, Any],
    acquire: bool = True,
    action: str | None = None,
    owner_id: str = "",
) -> str | None:
    # Resolve the shared run-level lock paths under the outputs volume.
    paths = _run_paths(job["protocol"], job["output_dir_name"])
    run_root = paths["run_root"]
    lock_dir = paths["lock_dir"]
    lock_file = paths["lock_file"]
    assert isinstance(run_root, Path)
    assert isinstance(lock_dir, Path)
    assert isinstance(lock_file, Path)

    if action is None:
        action = "acquire" if acquire else "release"

    run_root.mkdir(parents=True, exist_ok=True)

    # Acquire a new lease unless an unexpired owner already holds it.
    if action == "acquire":
        new_owner_id = uuid4().hex
        now = time.time()
        if lock_file.exists():
            payload = json.loads(lock_file.read_text(encoding="utf-8"))
            if float(payload.get("expires_at", 0.0)) > now:
                raise RuntimeError(f"Run name is already in use: {run_root}")
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(
            json.dumps({"owner_id": new_owner_id, "expires_at": time.time() + TIMEOUT}) + "\n",
            encoding="utf-8",
        )
        OUTPUTS_VOLUME.commit()
        return new_owner_id

    # Missing lock files are treated as already released.
    if not lock_file.exists():
        return None

    payload = json.loads(lock_file.read_text(encoding="utf-8"))
    current_owner_id = str(payload.get("owner_id", ""))
    if owner_id and current_owner_id != owner_id:
        raise RuntimeError(f"Lock owner mismatch for {run_root}: {current_owner_id} != {owner_id}")

    # Renew the current lease so long-running local polling keeps ownership.
    if action == "renew":
        lock_file.write_text(
            json.dumps({"owner_id": current_owner_id, "expires_at": time.time() + TIMEOUT}) + "\n",
            encoding="utf-8",
        )
        OUTPUTS_VOLUME.commit()
        return current_owner_id

    # Release only the lease owned by this caller.
    if action == "release":
        shutil.rmtree(lock_dir)
        OUTPUTS_VOLUME.commit()
        return None

    raise RuntimeError(f"Unsupported lock action: {action}")

#--------------------------------------
# Batch-level remote worker entrypoint
#--------------------------------------
@app.function(
    volumes={str(OUTPUTS_DIR): OUTPUTS_VOLUME, str(ROSETTA_DATA_DIR): ROSETTA_DATA},
    cpu=1,
    timeout=TIMEOUT,
    image=runtime_image,
)
def run_rosetta_job(job: dict[str, Any]) -> dict[str, Any]:
    # 01. Protocol config and validation
    protocol = job["protocol"]
    interface = job.get("interface", "")
    relax_selection = job.get("relax_selection", "")
    if protocol not in PROTOCOL_TO_EXECUTABLE:
        supported = ", ".join(sorted(PROTOCOL_TO_EXECUTABLE))
        raise RuntimeError(f"Unsupported --protocol '{protocol}'. Supported values: {supported}.")
    if protocol != "relax" and relax_selection:
        raise RuntimeError("--relax-selection requires --protocol relax.")

    # 02. Run paths
    paths = _run_paths(protocol, job["output_dir_name"])
    run_root = paths["run_root"]
    results_root = paths["results_root"]
    logs_root = paths["logs_root"]
    summary_csv = paths["summary_csv"]
    failed_records = paths["failed_records"]
    batch_failed_records = (
        failed_records
        if job.get("finalize_output_root", True)
        else failed_records.with_name(
            f"{failed_records.stem}.batch_{int(job.get('batch_index', 0))}{failed_records.suffix}"
        )
    )
    remote_volume_subpath = str(Path(protocol) / job["output_dir_name"])

    run_root.mkdir(parents=True, exist_ok=True)
    if job.get("prepare_output_root", True):
        results_root.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)

    # 03. Batch inputs
    input_items = job["staged_inputs"] or []
    if not input_items:
        if job.get("allow_empty_batches"):
            return {
                "failed": 0,
                "failed_record_file": str(batch_failed_records),
                "output_root": str(run_root),
                "processed": 0,
                "protocol": protocol,
                "remote_volume_subpath": remote_volume_subpath,
                "rows": [],
                "skipped": 0,
                "succeeded": 0,
                "summary_csv": str(summary_csv),
            }
        raise RuntimeError(f"No input items selected for batch {job['batch_index']} / {job['num_batches']}.")

    # 04. Batch execution
    rows: list[dict[str, Any]] = []
    processed = 0
    success_count = 0
    skipped_count = 0
    failed_count = 0

    # interface_analyzer-only validation
    if protocol == "interface_analyzer" and not interface:
        multichain_sources: list[str] = []
        for item in input_items:
            chain_ids: list[str] = []
            seen: set[str] = set()
            for raw_line in item["content"].splitlines():
                if not (raw_line.startswith(b"ATOM  ") or raw_line.startswith(b"HETATM")):
                    continue
                chain_id = raw_line[21:22].decode("utf-8", errors="ignore").strip() or "_"
                if chain_id not in seen:
                    seen.add(chain_id)
                    chain_ids.append(chain_id)
            if len(chain_ids) > 2:
                multichain_sources.append(item["source_path"])

        if multichain_sources:
            preview = ", ".join(multichain_sources[:3])
            if len(multichain_sources) > 3:
                preview += ", ..."
            raise RuntimeError(
                "Multichain input detected for interface_analyzer. "
                "Provide --interface. "
                f"Examples: {preview}"
            )

    with tempfile.TemporaryDirectory(prefix="rosetta_launcher_") as tmpdir_name:
        staged_inputs_dir = Path(tmpdir_name) / "inputs"
        staged_inputs_dir.mkdir(parents=True, exist_ok=True)

        # Per-item execution
        def fail_item(message: str) -> None:
            nonlocal failed_count
            row["status"] = "failed"
            row["message"] = message
            batch_failed_records.parent.mkdir(parents=True, exist_ok=True)
            with batch_failed_records.open("a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{item['source_path']}\t{message}\n")
            OUTPUTS_VOLUME.commit()
            rows.append(row)
            failed_count += 1

        for item in input_items:
            processed += 1
            # Item paths and row
            item_id = Path(item["source_path"]).stem
            item_output_dir = results_root / item_id
            stdout_log = logs_root / f"{item_id}.stdout.log"
            stderr_log = logs_root / f"{item_id}.stderr.log"
            command_log = logs_root / f"{item_id}.command.txt"
            success_marker = item_output_dir / "SUCCESS.json"
            row = {
                "item_id": item_id,
                "message": "",
                "output_dir": str(item_output_dir),
                "protocol": protocol,
                "return_code": "",
                "source_path": item["source_path"],
                "status": "pending",
                "stderr_log": str(stderr_log),
                "stdout_log": str(stdout_log),
            }

            # Resume completed item
            if _outputs_exist(protocol, item_output_dir):
                row["status"] = "skipped"
                row["message"] = "Skipped because completed outputs already exist."
                rows.append(row)
                skipped_count += 1
                continue

            # Stage input PDB and clear stale outputs
            item_output_dir.mkdir(parents=True, exist_ok=True)
            for p in [item_output_dir / "SUCCESS.json", item_output_dir / "score.sc", *item_output_dir.glob("*.pdb")]:
                if p.exists():
                    p.unlink()
            staged_pdb = staged_inputs_dir / f"{item_id}_{item['display_name'] or f'{item_id}.pdb'}"
            staged_pdb.write_bytes(item["content"])

            # relax-only setup
            relax_script_xml = None
            relax_script_vars = None
            if protocol == "relax":
                relax_script_vars = _build_relax_script_vars(relax_selection)
                if not ROSETTA_RELAX_XML_PATH.exists():
                    raise RuntimeError("Missing /rosetta-data/relax.xml. Run the temporary relax XML initializer first.")
                ctrl_dir = item_output_dir / "relax_controls"
                ctrl_dir.mkdir(parents=True, exist_ok=True)
                relax_script_xml = ctrl_dir / "relax.xml"
                shutil.copy2(ROSETTA_RELAX_XML_PATH, relax_script_xml)

            # Build and run Rosetta command
            command = _build_rosetta_command(
                protocol=protocol,
                input_pdb=staged_pdb,
                output_dir=item_output_dir,
                interface=interface,
                relax_script_xml=relax_script_xml,
                relax_script_vars=relax_script_vars,
            )
            command_log.parent.mkdir(parents=True, exist_ok=True)
            command_log.write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")

            # Launch failure handling
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=str(item_output_dir),
                )
            except Exception as exc:
                stdout_log.write_text("", encoding="utf-8")
                launch_error = f"{type(exc).__name__}: {exc}"
                stderr_log.write_text(launch_error + "\n", encoding="utf-8")
                message = f"Failed to launch Rosetta process: {launch_error}. See {stderr_log}."
                fail_item(message)
                continue

            # Process return code and outputs
            stdout_log.write_text(completed.stdout, encoding="utf-8")
            stderr_log.write_text(completed.stderr, encoding="utf-8")
            row["return_code"] = str(completed.returncode)

            if completed.returncode != 0:
                message = (
                    f"Rosetta exited with return code {completed.returncode}. "
                    f"See {stderr_log} and {stdout_log}."
                )
                fail_item(message)
                continue

            if not (item_output_dir / "score.sc").exists() or (
                protocol == "relax" and not any(item_output_dir.glob("*.pdb"))
            ):
                message = (
                    "Rosetta finished with exit code 0 but expected outputs were not found. "
                    f"Inspect {item_output_dir}, {stdout_log}, and {stderr_log}."
                )
                fail_item(message)
                continue

            # Success marker
            success_payload = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "input_cli": job.get("input_cli", ""),
                "output_dir": str(item_output_dir),
                "protocol": protocol,
                "return_code": completed.returncode,
                "source_path": item["source_path"],
            }
            success_marker.write_text(
                json.dumps(success_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            row["status"] = "success"
            row["message"] = "Completed successfully."
            OUTPUTS_VOLUME.commit()
            rows.append(row)
            success_count += 1

    # 05. Run summary
    if job.get("finalize_output_root", True):
        _write_summary(summary_csv, rows)
        OUTPUTS_VOLUME.commit()

    # 06. Run result
    return {
        "failed": failed_count,
        "failed_record_file": str(batch_failed_records),
        "output_root": str(run_root),
        "processed": processed,
        "protocol": protocol,
        "remote_volume_subpath": remote_volume_subpath,
        "rows": rows,
        "skipped": skipped_count,
        "succeeded": success_count,
        "summary_csv": str(summary_csv),
    }


#--------------------------------------
# Finalize one run-level summary remotely
#--------------------------------------
@app.function(
    volumes={str(OUTPUTS_DIR): OUTPUTS_VOLUME},
    cpu=1,
    timeout=TIMEOUT,
    image=runtime_image,
)
def finalize_rosetta_run(job: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    paths = _run_paths(job["protocol"], job["output_dir_name"])
    summary_csv = paths["summary_csv"]
    failed_records = paths["failed_records"]
    run_root = paths["run_root"]
    failed_records.parent.mkdir(parents=True, exist_ok=True)
    with failed_records.open("w", encoding="utf-8") as out:
        for part in sorted(run_root.glob("failed_records.batch_*.txt")):
            out.write(part.read_text(encoding="utf-8"))
    _write_summary(summary_csv, rows)
    OUTPUTS_VOLUME.commit()
    return {
        "failed_record_file": str(failed_records),
        "output_root": str(run_root),
        "summary_csv": str(summary_csv),
    }


#--------------------------------------
# Package one run directory for local download
#--------------------------------------
@app.function(
    volumes={str(OUTPUTS_DIR): OUTPUTS_VOLUME},
    cpu=1,
    timeout=TIMEOUT,
    image=runtime_image,
)
def package_rosetta_run(job: dict[str, Any]) -> bytes:
    paths = _run_paths(job["protocol"], job["output_dir_name"])
    run_root = paths["run_root"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(run_root, arcname=run_root.name)
    return buffer.getvalue()


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def main(
    input_dir: str,
    protocol: str,
    output_dir: str = "",
    output_dir_name: str = "",
    parallel_batches: int = 1,
    interface: str = "",
    relax_selection: str = "",
) -> None:
    # 01. Local input normalization
    if not (output_dir := output_dir.strip()):
        raise SystemExit("--output-dir is required.")
    if not (run_name := output_dir_name.strip()):
        raise SystemExit("--output-dir-name is required.")
    if "/" in (normalized := run_name.replace("\\", "/")) or any(s in {".", ".."} for s in normalized.split("/") if s):
        raise SystemExit("--output-dir-name must be a simple run name. Path separators and '.' or '..' are not allowed.")
    if parallel_batches < 1:
        raise SystemExit("--parallel-batches must be at least 1.")

    # Resolve local input path and collect PDB files
    local_input = Path(input_dir)
    if local_input.is_file():
        if local_input.suffix.lower() != ".pdb":
            raise SystemExit(f"Input file must be a .pdb file: {local_input}")
        pdb_files = [local_input]
    elif local_input.is_dir():
        pdb_files = sorted(path for path in local_input.iterdir() if path.is_file() and path.suffix.lower() == ".pdb")
    else:
        raise SystemExit(f"Input path does not exist: {local_input}")

    if not pdb_files:
        raise SystemExit(f"No .pdb files found in input directory: {local_input}")

    # Build stable staged_inputs payload from local PDB files
    unique_by_stem: dict[str, Path] = {}
    for pdb_file in pdb_files:
        stem_key = pdb_file.stem.casefold()
        if stem_key in unique_by_stem:
            raise SystemExit(
                "Duplicate input structure stems are not supported: "
                f"{unique_by_stem[stem_key]} and {pdb_file}"
            )
        unique_by_stem[stem_key] = pdb_file

    staged_inputs = [
        {
            "source_path": str(path.resolve(strict=False)),
            "display_name": path.name,
            "content": path.read_bytes(),
        }
        for path in sorted(unique_by_stem.values(), key=lambda item: item.name.casefold())
    ]

    # 02. Submit remote workers
    # Build the base remote worker job payload
    job = {
        "batch_index": 0,
        "input_cli": " ".join(
            ["modal", "run", Path(__file__).name, *[shlex.quote(arg) for arg in sys.argv[1:]]]
        ),
        "input_dir": input_dir,
        "interface": interface,
        "num_batches": 1,
        "output_dir_name": run_name,
        "parallel_batches": parallel_batches,
        "protocol": protocol,
        "relax_selection": relax_selection,
        "staged_inputs": staged_inputs,
    }

    # Acquire one run-level lock for this output_dir_name
    lock_owner = manage_rosetta_run_lock.remote(job, action="acquire")
    try:
        # Single worker
        if parallel_batches == 1:
            fc = run_rosetta_job.spawn(job)
            while True:
                try:
                    result = fc.get(timeout=FUNCTION_CALL_POLL_TIMEOUT)
                    break
                except TimeoutError:
                    manage_rosetta_run_lock.remote(job, action="renew", owner_id=lock_owner or "")
                    time.sleep(FUNCTION_CALL_POLL_INTERVAL)
            result["processed"] = len(result.pop("rows", []))
        else:
            # Multi-worker setup
            remote_paths = _run_paths(protocol, run_name)
            worker_jobs: list[dict[str, Any]] = []

            # Split staged_inputs into worker chunks
            worker_chunks = [[] for _ in range(parallel_batches)]
            for index, item in enumerate(staged_inputs):
                worker_chunks[index % parallel_batches].append(item)
            for batch_index, chunk in enumerate(worker_chunks):
                if not chunk:
                    continue
                worker_job = dict(job)
                worker_job["allow_empty_batches"] = True
                worker_job["batch_index"] = batch_index
                worker_job["finalize_output_root"] = False
                worker_job["num_batches"] = parallel_batches
                worker_job["prepare_output_root"] = True
                worker_job["staged_inputs"] = chunk
                worker_jobs.append(worker_job)

            # 03. Wait for worker results
            function_calls = [run_rosetta_job.spawn(wj) for wj in worker_jobs]
            worker_results = []
            for fc in function_calls:
                while True:
                    try:
                        wr = fc.get(timeout=FUNCTION_CALL_POLL_TIMEOUT)
                        break
                    except TimeoutError:
                        manage_rosetta_run_lock.remote(job, action="renew", owner_id=lock_owner or "")
                        time.sleep(FUNCTION_CALL_POLL_INTERVAL)
                worker_results.append(wr)
            rows: list[dict[str, Any]] = []
            for worker_result in worker_results:
                rows.extend(worker_result.get("rows", []))

            finalize_result = finalize_rosetta_run.remote(job, rows)
            sum_wr = lambda k: sum(int(r.get(k, 0)) for r in worker_results)  # noqa: E731
            result = {
                "failed": sum_wr("failed"),
                "failed_record_file": finalize_result["failed_record_file"],
                "output_root": finalize_result["output_root"],
                "parallel_batches": parallel_batches,
                "processed": sum_wr("processed"),
                "protocol": protocol,
                "remote_volume_subpath": str(Path(protocol) / run_name),
                "skipped": sum_wr("skipped"),
                "succeeded": sum_wr("succeeded"),
                "summary_csv": finalize_result["summary_csv"],
            }

        # 04. Download run outputs
        local_download_parent = Path(output_dir).expanduser().resolve(strict=False)
        archive_bytes = package_rosetta_run.remote(job)
        local_download_parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            archive.extractall(path=local_download_parent)
    finally:
        manage_rosetta_run_lock.remote(job, action="release", owner_id=lock_owner or "")

    # 05. Print run summary
    result["local_output_dir"] = str((local_download_parent / run_name).resolve(strict=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed"] > 0:
        raise SystemExit(1)
