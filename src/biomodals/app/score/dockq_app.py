"""DockQ source repo: <https://github.com/y1zhou/DockQ>.

DockQ compares a predicted/model complex against a reference/native complex and
reports continuous docking-quality metrics. This wrapper accepts a CSV of
model/reference file pairs for standalone use and exposes a deployed batch
function for workflows that already have structure bytes in memory.

The upstream DockQ documentation is available at <https://github.com/wallnerlab/DockQ>.

## Outputs

Results are saved locally as `<run-name>.tar.zst`. The archive contains
`dockq_results.csv` and one raw `.log` file per scored pair.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import csv
import os
import re
import shlex
import subprocess
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.shell import package_outputs, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="DockQ",
    repo_url="https://github.com/y1zhou/DockQ",
    package_name="DockQ",
    version="2.1.3",
    python_version="3.12",
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)


##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install("zstd")
    .env(CONF.default_env)
    .uv_pip_install(f"{CONF.package_name}=={CONF.version}", "pandas")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
METRIC_KEYS = {
    "dockq": "dockq",
    "irmsd": "irmsd",
    "lrmsd": "lrmsd",
    "fnat": "fnat",
    "fnonnat": "fnonnat",
    "f1": "f1",
    "clashes": "clashes",
}


def _parse_short_metrics(output: str) -> dict[str, str]:
    """Parse DockQ short-output key/value metrics."""
    metrics: dict[str, str] = {}
    tokens = output.replace("\t", " ").split()
    for idx, token in enumerate(tokens[:-1]):
        normalized = token.rstrip(":").lower()
        if normalized in METRIC_KEYS:
            value = tokens[idx + 1]
            if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
                metrics[METRIC_KEYS[normalized]] = value

    mapping_match = re.search(r"\bmapping\s+(\S+)", output)
    if mapping_match:
        metrics["mapping"] = mapping_match.group(1)
    return metrics


def _safe_structure_name(name: str, fallback: str) -> str:
    """Return a safe structure filename preserving common structure suffixes."""
    raw = Path(name).name or fallback
    safe = sanitize_filename(raw)
    suffix = Path(safe).suffix.lower()
    if suffix not in {".pdb", ".cif", ".gz"}:
        safe = f"{safe}.pdb"
    return safe


def _row_from_pair(
    pair: dict[str, object],
    *,
    pair_idx: int,
    workdir: Path,
    dockq_args: list[str],
) -> dict[str, str]:
    """Run DockQ for one model/reference pair and return a CSV row."""
    pair_id = str(pair.get("id") or f"pair_{pair_idx}")
    pair_dir = workdir / sanitize_filename(pair_id)
    pair_dir.mkdir(parents=True, exist_ok=True)

    model_name = _safe_structure_name(str(pair.get("model_name") or ""), "model.pdb")
    reference_name = _safe_structure_name(
        str(pair.get("reference_name") or ""), "reference.pdb"
    )
    model_path = pair_dir / model_name
    reference_path = pair_dir / reference_name

    model_bytes = pair.get("model_bytes")
    reference_bytes = pair.get("reference_bytes")
    if not isinstance(model_bytes, bytes):
        raise TypeError(f"DockQ pair {pair_id!r} is missing model_bytes")
    if not isinstance(reference_bytes, bytes):
        raise TypeError(f"DockQ pair {pair_id!r} is missing reference_bytes")

    model_path.write_bytes(model_bytes)
    reference_path.write_bytes(reference_bytes)

    cmd = ["DockQ", str(model_path), str(reference_path), *dockq_args]
    if mapping := pair.get("mapping"):
        cmd.extend(["--mapping", str(mapping)])

    print(f"💊 Running DockQ for {pair_id}")
    completed = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    raw_output = completed.stdout or ""
    log_path = pair_dir / "dockq.log"
    log_path.write_text(raw_output)

    metrics = _parse_short_metrics(raw_output)
    row = {
        "id": pair_id,
        "model": model_name,
        "reference": reference_name,
        "dockq": metrics.get("dockq", ""),
        "irmsd": metrics.get("irmsd", ""),
        "lrmsd": metrics.get("lrmsd", ""),
        "fnat": metrics.get("fnat", ""),
        "fnonnat": metrics.get("fnonnat", ""),
        "f1": metrics.get("f1", ""),
        "clashes": metrics.get("clashes", ""),
        "mapping": metrics.get("mapping", str(pair.get("mapping") or "")),
        "returncode": str(completed.returncode),
        "log": str(log_path.relative_to(workdir)),
    }
    if completed.returncode != 0:
        row["error"] = raw_output.strip().splitlines()[-1] if raw_output.strip() else ""
    else:
        row["error"] = ""
    return row


def _write_results_csv(rows: Iterable[dict[str, str]], csv_path: Path) -> None:
    """Write DockQ result rows to CSV."""
    fieldnames = [
        "id",
        "model",
        "reference",
        "dockq",
        "irmsd",
        "lrmsd",
        "fnat",
        "fnonnat",
        "f1",
        "clashes",
        "mapping",
        "returncode",
        "error",
        "log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 16.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
)
def run_dockq_batch(
    pairs: list[dict[str, object]],
    run_name: str,
    dockq_args: list[str] | None = None,
) -> bytes:
    """Run DockQ on model/reference pairs and return packaged outputs."""
    dockq_args = dockq_args or ["--short"]
    if not pairs:
        raise ValueError("At least one DockQ pair is required")

    safe_run_name = sanitize_filename(run_name)
    with TemporaryDirectory(prefix=f"dockq_{safe_run_name}_") as tmpdir:
        out_dir = Path(tmpdir) / safe_run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            _row_from_pair(
                pair,
                pair_idx=idx,
                workdir=out_dir,
                dockq_args=dockq_args,
            )
            for idx, pair in enumerate(pairs, start=1)
        ]
        _write_results_csv(rows, out_dir / "dockq_results.csv")
        return package_outputs(out_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
def _pairs_from_csv(input_csv: Path) -> list[dict[str, object]]:
    """Read standalone DockQ pair specs from a local CSV."""
    text = input_csv.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    required = {"model", "reference"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ValueError(
            f"DockQ input CSV must contain columns {sorted(required)}, "
            f"got {reader.fieldnames}"
        )

    pairs: list[dict[str, object]] = []
    for idx, row in enumerate(reader, start=1):
        model = Path(row["model"]).expanduser()
        reference = Path(row["reference"]).expanduser()
        if not model.is_absolute():
            model = input_csv.parent / model
        if not reference.is_absolute():
            reference = input_csv.parent / reference
        if not model.exists():
            raise FileNotFoundError(f"Model structure not found: {model}")
        if not reference.exists():
            raise FileNotFoundError(f"Reference structure not found: {reference}")
        pairs.append(
            {
                "id": row.get("id") or f"pair_{idx}",
                "model_name": model.name,
                "model_bytes": model.read_bytes(),
                "reference_name": reference.name,
                "reference_bytes": reference.read_bytes(),
                "mapping": row.get("mapping") or None,
            }
        )
    return pairs


@app.local_entrypoint()
def submit_dockq_task(
    input_csv: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    dockq_args: str = "--short",
) -> None:
    """Run DockQ scoring on model/reference pairs from a CSV.

    Args:
        input_csv: CSV with `model` and `reference` columns, plus optional
            `id` and `mapping` columns. Relative paths resolve from the CSV
            parent directory.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the input
            CSV filename stem.
        dockq_args: Extra DockQ CLI arguments, shell-split before execution.
            Defaults to `--short`.
    """
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if run_name is None:
        run_name = input_path.stem

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    pairs = _pairs_from_csv(input_path)
    print(f"🧬 Submitting DockQ run '{run_name}' with {len(pairs)} pair(s)")
    tarball_bytes = run_dockq_batch.remote(
        pairs=pairs,
        run_name=run_name,
        dockq_args=shlex.split(dockq_args),
    )

    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 DockQ run complete! Results saved to {out_file}")
