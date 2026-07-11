#!/usr/bin/env python3
"""Verify Biomodals OligoFormer outputs against direct upstream commands.

This is an engineering verifier, not a fast pytest. The local mode launches the
Biomodals Modal app to produce candidate artifacts, then launches a Modal
Sandbox in the same OligoFormer runtime image and invokes upstream
`scripts/main.py` directly. It compares final tables only because normal app
runs delete raw off-target intermediates after final outputs are written. The
app is never used as the oracle for upstream OligoFormer, PITA, or TargetScan
behavior.
"""

# This verifier intentionally stages files under a private Modal Sandbox /tmp and
# runs pinned upstream commands inside the OligoFormer runtime image.
# ruff: noqa: S108,S603,S607

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MRNA_FASTA = REPO_ROOT / "examples/data/sirna_target.fa"
DEFAULT_UTR_REF = REPO_ROOT / "examples/data/oligoformer_offtarget_utr.fa"
DEFAULT_ORF_REF = REPO_ROOT / "examples/data/oligoformer_offtarget_orf.fa"
FINAL_TABLE_SUFFIXES = ("", "_ranked", "_ranked_filtered")
VERIFIER_VERSION = "oligoformer-upstream-equivalence-v4"
SANDBOX_APP_NAME = "oligoformer-upstream-equivalence"

FLOAT_TOLERANCES: dict[str, float] = {
    "efficacy": 1e-1,
    "pita_score": 1e-6,
    "targetscan_score": 1e-6,
    "cell_viability": 1e-6,
    "Score": 1e-6,
}
INT_COLUMNS = {
    "pos",
    "func_filter",
    "off_target_filter",
    "toxicity_filter",
    "filter",
    "Sites",
}
SORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "final": ("pos", "siRNA"),
    "pita": ("RefSeq", "microRNA"),
    "targetscan": ("refseq", "siRNA"),
}


@dataclass(frozen=True, slots=True)
class TableComparison:
    """Canonical table comparison result."""

    name: str
    passed: bool
    row_count: int
    message: str
    max_deltas: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result."""
        return {
            "name": self.name,
            "passed": self.passed,
            "row_count": self.row_count,
            "message": self.message,
            "max_deltas": self.max_deltas,
        }


def _json_dumps(data: object) -> bytes:
    """Serialize JSON with orjson when available."""
    try:
        import orjson

        return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    except ImportError:
        return json.dumps(data, indent=2, sort_keys=True).encode("utf-8")


def _write_json(path: Path, data: object) -> None:
    """Write JSON bytes to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_dumps(data) + b"\n")


def _hash_path(path: Path) -> str:
    """Return a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verifier_cache_key(
    *,
    mrna_fasta: Path,
    sirna_fasta: Path | None,
    utr_ref: Path,
    orf_ref: Path,
    all_human: bool,
    reference_identity: str | None,
    top_n: int,
    targetscan_ref_shard_size: int | None,
) -> str:
    """Return a stable cache key for verifier inputs."""
    if all_human and reference_identity is None:
        raise ValueError("full-human verification requires a reference identity")
    payload = {
        "version": VERIFIER_VERSION,
        "mrna": _hash_path(mrna_fasta),
        "sirna": _hash_path(sirna_fasta) if sirna_fasta is not None else None,
        "references": (
            {"all_human": True, "identity": reference_identity}
            if all_human
            else {"utr": _hash_path(utr_ref), "orf": _hash_path(orf_ref)}
        ),
        "top_n": top_n,
        "targetscan_ref_shard_size": targetscan_ref_shard_size,
    }
    return hashlib.sha256(_json_dumps(payload)).hexdigest()


def _read_table(path: Path, kind: str) -> pl.DataFrame:
    """Read a verifier table with stable column names."""
    if kind == "targetscan":
        if not path.exists() or path.stat().st_size == 0:
            return pl.DataFrame(
                schema={
                    "refseq": pl.String,
                    "siRNA": pl.String,
                    "targetscan_score": pl.Float64,
                }
            )
        return pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=["refseq", "siRNA", "targetscan_score"],
            infer_schema_length=0,
        )
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def canonical_rows(
    path: Path,
    kind: str,
    *,
    source: str = "app",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return canonicalized table columns and rows."""
    frame = _read_table(path, kind)
    for column in frame.columns:
        if column in FLOAT_TOLERANCES:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Float64, strict=False).alias(column)
            )
        elif column in INT_COLUMNS:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Int64, strict=False).alias(column)
            )
        else:
            frame = frame.with_columns(pl.col(column).cast(pl.String).alias(column))

    if kind == "final" and source == "upstream" and "filter" in frame.columns:
        filter_terms = []
        if "func_filter" in frame.columns:
            filter_terms.append(pl.col("func_filter") != 0)
        if "off_target_filter" in frame.columns:
            filter_terms.append(pl.col("off_target_filter") != 0)
        if "toxicity_filter" in frame.columns:
            filter_terms.append(pl.col("toxicity_filter") != 0)
        filter_expr = pl.lit(0)
        for term in filter_terms:
            filter_expr = filter_expr + term.cast(pl.Int64)
        frame = frame.with_columns(filter_expr.alias("filter"))

    sort_columns = [column for column in SORT_COLUMNS[kind] if column in frame.columns]
    if sort_columns:
        frame = frame.sort(sort_columns)
    rows = frame.to_dicts()
    return list(frame.columns), rows


def compare_tables(
    *,
    name: str,
    app_path: Path,
    upstream_path: Path,
    kind: str,
    canonical_dir: Path | None = None,
) -> TableComparison:
    """Compare two canonicalized verifier tables."""
    app_columns, app_rows = canonical_rows(app_path, kind, source="app")
    upstream_columns, upstream_rows = canonical_rows(
        upstream_path, kind, source="upstream"
    )

    if canonical_dir is not None:
        _write_json(
            canonical_dir / "app" / f"{name}.json",
            {"columns": app_columns, "rows": app_rows},
        )
        _write_json(
            canonical_dir / "upstream" / f"{name}.json",
            {"columns": upstream_columns, "rows": upstream_rows},
        )

    if app_columns != upstream_columns:
        return TableComparison(
            name=name,
            passed=False,
            row_count=0,
            message=f"column mismatch: app={app_columns}, upstream={upstream_columns}",
            max_deltas={},
        )
    if len(app_rows) != len(upstream_rows):
        return TableComparison(
            name=name,
            passed=False,
            row_count=min(len(app_rows), len(upstream_rows)),
            message=f"row count mismatch: app={len(app_rows)}, upstream={len(upstream_rows)}",
            max_deltas={},
        )

    max_deltas: dict[str, float] = {
        column: 0.0 for column in app_columns if column in FLOAT_TOLERANCES
    }
    for index, (app_row, upstream_row) in enumerate(
        zip(app_rows, upstream_rows, strict=True)
    ):
        for column in app_columns:
            left = app_row[column]
            right = upstream_row[column]
            if column in FLOAT_TOLERANCES:
                if left is None or right is None:
                    if left != right:
                        return TableComparison(
                            name=name,
                            passed=False,
                            row_count=len(app_rows),
                            message=f"row {index} column {column} null mismatch",
                            max_deltas=max_deltas,
                        )
                    continue
                delta = abs(float(left) - float(right))
                max_deltas[column] = max(max_deltas[column], delta)
                if delta > FLOAT_TOLERANCES[column]:
                    return TableComparison(
                        name=name,
                        passed=False,
                        row_count=len(app_rows),
                        message=(
                            f"row {index} column {column} delta {delta} exceeds "
                            f"{FLOAT_TOLERANCES[column]}"
                        ),
                        max_deltas=max_deltas,
                    )
            elif left != right:
                return TableComparison(
                    name=name,
                    passed=False,
                    row_count=len(app_rows),
                    message=(
                        f"row {index} column {column} mismatch: "
                        f"app={left!r}, upstream={right!r}"
                    ),
                    max_deltas=max_deltas,
                )

    return TableComparison(
        name=name,
        passed=True,
        row_count=len(app_rows),
        message="ok",
        max_deltas=max_deltas,
    )


def _copy_required(src: Path, dst: Path) -> None:
    """Copy a required artifact, failing clearly when it is absent."""
    if not src.exists():
        raise FileNotFoundError(f"required verifier artifact is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_remove_artifact_root(path: Path, output_mount: str) -> None:
    """Remove only verifier-owned output-volume paths."""
    output_root = Path(output_mount).resolve()
    resolved = path.resolve()
    if output_root not in resolved.parents or "upstream_equivalence" not in path.parts:
        raise ValueError(f"refusing to remove non-verifier path: {path}")
    shutil.rmtree(path, ignore_errors=True)


def _ensure_rnafm_runtime(config: dict[str, Any]) -> None:
    """Copy RNA-FM from the model volume into the writable upstream repo."""
    src = Path(config["model_rnafm_dir"])
    dst = Path(config["repo_rnafm_dir"])
    redevelop = Path(config["repo_rnafm_redevelop_dir"])
    weights = redevelop / "pretrained/RNA-FM_pretrained.pth"
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists() and not weights.is_file():
        shutil.rmtree(dst)
    if not weights.is_file():
        shutil.copytree(src, dst)


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
) -> None:
    """Run a command with stdout and stderr redirected to log files."""
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_log.open("w", encoding="utf-8") as stdout,
        stderr_log.open("w", encoding="utf-8") as stderr,
    ):
        result = sp.run(
            cmd,
            cwd=cwd,
            check=False,
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "upstream command failed with exit code "
            f"{result.returncode}; see {stdout_log} and {stderr_log}"
        )


def _copy_app_artifacts(config: dict[str, Any], artifact_root: Path) -> None:
    """Copy Biomodals app outputs into verifier artifact storage."""
    output_dir = Path(config["output_dir"])
    app_output_dir = artifact_root / "app/outputs"
    for stem in config["output_stems"]:
        for suffix in FINAL_TABLE_SUFFIXES:
            name = f"{stem}{suffix}.txt"
            _copy_required(output_dir / name, app_output_dir / name)


def _run_upstream(config: dict[str, Any], artifact_root: Path) -> None:
    """Run direct upstream OligoFormer/PITA/TargetScan commands in the sandbox."""
    repo_dir = Path(config["repo_dir"])
    upstream_output_dir = artifact_root / "upstream/outputs"
    upstream_done = artifact_root / "upstream/upstream.done"
    if upstream_done.exists() and not config["force"]:
        return

    shutil.rmtree(upstream_output_dir, ignore_errors=True)
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_rnafm_runtime(config)

    for stem in config["output_stems"]:
        shutil.rmtree(repo_dir / "data/infer" / stem, ignore_errors=True)

    cmd = [
        "python",
        "scripts/main.py",
        "-i",
        "1",
        "-i1",
        config["mrna_fasta"],
    ]
    if config["sirna_fasta"] is not None:
        cmd.extend(["-i2", config["sirna_fasta"]])
    cmd.extend([
        "--output_dir",
        f"{upstream_output_dir}/",
        "-off",
        "-tox",
        "-top",
        str(config["top_n"]),
        "--utr",
        config["utr_ref"],
        "--orf",
        config["orf_ref"],
    ])
    _run_logged(
        cmd,
        cwd=repo_dir,
        stdout_log=artifact_root / "logs/upstream_main.stdout.log",
        stderr_log=artifact_root / "logs/upstream_main.stderr.log",
    )

    upstream_done.write_text("done\n", encoding="utf-8")


def _compare_artifacts(
    config: dict[str, Any], artifact_root: Path
) -> list[dict[str, Any]]:
    """Compare app and upstream verifier artifacts."""
    comparisons: list[TableComparison] = []
    for stem in config["output_stems"]:
        for suffix in FINAL_TABLE_SUFFIXES:
            table_name = f"{stem}{suffix or '_original'}"
            comparisons.append(
                compare_tables(
                    name=table_name,
                    app_path=artifact_root / "app/outputs" / f"{stem}{suffix}.txt",
                    upstream_path=artifact_root
                    / "upstream/outputs"
                    / f"{stem}{suffix}.txt",
                    kind="final",
                    canonical_dir=artifact_root / "canonical",
                )
            )
    return [comparison.as_dict() for comparison in comparisons]


def run_remote_worker(config_path: Path) -> int:
    """Run direct upstream oracle and comparisons inside a Modal Sandbox."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_root = Path(config["artifact_root"])
    if config["force"]:
        _safe_remove_artifact_root(artifact_root, config["output_volume_mountpoint"])
    artifact_root.mkdir(parents=True, exist_ok=True)

    input_dir = artifact_root / "inputs"
    _copy_required(Path(config["mrna_fasta"]), input_dir / "mrna.fa")
    if config["sirna_fasta"] is not None:
        _copy_required(Path(config["sirna_fasta"]), input_dir / "sirna.fa")
    if not config["all_human"]:
        _copy_required(Path(config["utr_ref"]), input_dir / "utr.fa")
        _copy_required(Path(config["orf_ref"]), input_dir / "orf.fa")
    _copy_app_artifacts(config, artifact_root)
    _run_upstream(config, artifact_root)
    comparisons = _compare_artifacts(config, artifact_root)

    passed = all(bool(item["passed"]) for item in comparisons)
    summary = {
        "passed": passed,
        "artifact_root": str(artifact_root),
        "run_root": config["run_root"],
        "output_stems": config["output_stems"],
        "comparisons": comparisons,
    }
    _write_json(artifact_root / "summary.json", summary)
    _write_json(Path("/tmp/oligoformer_upstream_equivalence_summary.json"), summary)
    sp.run(["sync", config["output_volume_mountpoint"]], check=True)

    print(f"OligoFormer upstream-equivalence artifacts: {artifact_root}")
    print(f"OligoFormer upstream-equivalence passed: {passed}")
    for item in comparisons:
        print(f"  {item['name']}: {item['message']}")
    return 0 if passed else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse verifier command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrna-fasta", type=Path, default=DEFAULT_MRNA_FASTA)
    parser.add_argument("--sirna-fasta", type=Path)
    parser.add_argument("--utr-ref", type=Path, default=DEFAULT_UTR_REF)
    parser.add_argument("--orf-ref", type=Path, default=DEFAULT_ORF_REF)
    parser.add_argument("--all-human", action="store_true")
    parser.add_argument("--top-n", type=int, default=-1)
    parser.add_argument("--targetscan-ref-shard-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("oligoformer_upstream_equivalence_summary.json"),
    )
    parser.add_argument("--remote-worker", type=Path)
    return parser.parse_args(argv)


def _run_biomodals_app(args: argparse.Namespace):
    """Run or reuse Biomodals OligoFormer artifacts."""
    from biomodals.app.score import oligoformer_app

    oligoformer_app.download_oligoformer_models.remote(force=False)
    mrna_fasta_bytes = args.mrna_fasta.read_bytes()
    sirna_fasta_bytes = (
        args.sirna_fasta.read_bytes() if args.sirna_fasta is not None else None
    )
    utr_bytes = None if args.all_human else args.utr_ref.read_bytes()
    orf_bytes = None if args.all_human else args.orf_ref.read_bytes()
    plan = oligoformer_app.prepare_oligoformer_run.remote(
        mrna_fasta_bytes=mrna_fasta_bytes,
        sirna_fasta_bytes=sirna_fasta_bytes,
        off_target=True,
        toxicity=True,
        all_human=args.all_human,
        utr_bytes=utr_bytes,
        orf_bytes=orf_bytes,
        top_n=args.top_n,
        force=args.force,
    )
    if not plan.final_ready:
        if plan.efficacy_ready:
            efficacy_plan = plan
        else:
            efficacy_plan = oligoformer_app.run_oligoformer_efficacy.remote(plan=plan)
        oligoformer_app.run_oligoformer_postprocess.remote(
            plan=efficacy_plan,
            off_target=True,
            toxicity=True,
            all_human=args.all_human,
            top_n=args.top_n,
            targetscan_ref_shard_size=args.targetscan_ref_shard_size,
        )
        plan = oligoformer_app.prepare_oligoformer_run.remote(
            mrna_fasta_bytes=mrna_fasta_bytes,
            sirna_fasta_bytes=sirna_fasta_bytes,
            off_target=True,
            toxicity=True,
            all_human=args.all_human,
            utr_bytes=utr_bytes,
            orf_bytes=orf_bytes,
            top_n=args.top_n,
            force=False,
        )
    return plan


def _run_local(args: argparse.Namespace) -> int:
    """Run app artifacts and sandbox oracle from the local machine."""
    import modal
    from modal.stream_type import StreamType

    from biomodals.app.score import oligoformer_app

    input_paths = [args.mrna_fasta]
    if args.sirna_fasta is not None:
        input_paths.append(args.sirna_fasta)
    if not args.all_human:
        input_paths.extend((args.utr_ref, args.orf_ref))
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.top_n != -1 and args.top_n < 1:
        raise ValueError("top_n must be -1 or a positive integer")

    with modal.enable_output():
        with oligoformer_app.app.run():
            plan = _run_biomodals_app(args)
            key = verifier_cache_key(
                mrna_fasta=args.mrna_fasta,
                sirna_fasta=args.sirna_fasta,
                utr_ref=args.utr_ref,
                orf_ref=args.orf_ref,
                all_human=args.all_human,
                reference_identity=plan.reference_identity,
                top_n=args.top_n,
                targetscan_ref_shard_size=args.targetscan_ref_shard_size,
            )
            artifact_root = (
                Path(oligoformer_app.CONF.output_volume_mountpoint)
                / "upstream_equivalence"
                / key
            )
            print(f"OligoFormer upstream-equivalence key: {key}", flush=True)
            config = {
                "force": args.force,
                "artifact_root": str(artifact_root),
                "run_root": plan.run_root,
                "output_dir": plan.output_dir,
                "output_stems": list(plan.output_stems),
                "top_n": args.top_n,
                "all_human": args.all_human,
                "mrna_fasta": "/tmp/oligoformer_verifier/mrna.fa",
                "sirna_fasta": (
                    "/tmp/oligoformer_verifier/sirna.fa"
                    if args.sirna_fasta is not None
                    else None
                ),
                "utr_ref": (
                    str(oligoformer_app.APP_INFO.model_ref_dir / "human_UTR.txt")
                    if args.all_human
                    else "/tmp/oligoformer_verifier/utr.fa"
                ),
                "orf_ref": (
                    str(oligoformer_app.APP_INFO.model_ref_dir / "human_ORF.txt")
                    if args.all_human
                    else "/tmp/oligoformer_verifier/orf.fa"
                ),
                "repo_dir": str(oligoformer_app.CONF.git_clone_dir),
                "output_volume_mountpoint": oligoformer_app.CONF.output_volume_mountpoint,
                "model_rnafm_redevelop_dir": str(
                    oligoformer_app.APP_INFO.model_rnafm_redevelop_dir
                ),
                "model_rnafm_dir": str(oligoformer_app.APP_INFO.model_rnafm_dir),
                "repo_rnafm_dir": str(oligoformer_app.APP_INFO.repo_rnafm_dir),
                "repo_rnafm_redevelop_dir": str(
                    oligoformer_app.APP_INFO.repo_rnafm_redevelop_dir
                ),
            }
            sandbox_volumes = {
                str(path): volume
                for path, volume in oligoformer_app.CONF.mounts(
                    output_volume=True,
                    model_volume=True,
                ).items()
            }
            sandbox_app = modal.App.lookup(
                SANDBOX_APP_NAME,
                create_if_missing=True,
            )
            sandbox = modal.Sandbox.create(
                "sleep",
                "86400",
                app=sandbox_app,
                name=f"oligoformer-upstream-{key[:12]}",
                tags={"verifier_key": key},
                image=oligoformer_app.runtime_image,
                volumes=cast(Any, sandbox_volumes),
                timeout=args.timeout,
                cpu=(0.125, 16.125),
                gpu=oligoformer_app.CONF.gpu,
                memory=(1024, 32768),
            )
            process_started = False
            process_finished = False
            try:
                print(
                    f"OligoFormer upstream-equivalence Sandbox: {sandbox.object_id}",
                    flush=True,
                )
                mkdir = sandbox.exec("mkdir", "-p", "/tmp/oligoformer_verifier")
                mkdir.wait()
                if mkdir.returncode != 0:
                    raise RuntimeError("failed to create sandbox verifier directory")
                sandbox.filesystem.copy_from_local(
                    __file__, "/tmp/verify_oligoformer_upstream_equivalence.py"
                )
                sandbox.filesystem.copy_from_local(
                    args.mrna_fasta, "/tmp/oligoformer_verifier/mrna.fa"
                )
                if args.sirna_fasta is not None:
                    sandbox.filesystem.copy_from_local(
                        args.sirna_fasta, "/tmp/oligoformer_verifier/sirna.fa"
                    )
                if not args.all_human:
                    sandbox.filesystem.copy_from_local(
                        args.utr_ref, "/tmp/oligoformer_verifier/utr.fa"
                    )
                    sandbox.filesystem.copy_from_local(
                        args.orf_ref, "/tmp/oligoformer_verifier/orf.fa"
                    )
                sandbox.filesystem.write_text(
                    json.dumps(config), "/tmp/oligoformer_verifier/config.json"
                )
                process = sandbox.exec(
                    "python",
                    "/tmp/verify_oligoformer_upstream_equivalence.py",
                    "--remote-worker",
                    "/tmp/oligoformer_verifier/config.json",
                    timeout=args.timeout,
                    workdir=str(oligoformer_app.CONF.git_clone_dir),
                    stdout=StreamType.DEVNULL,
                    stderr=StreamType.DEVNULL,
                )
                process_started = True
                process.wait()
                process_finished = True
                if process.returncode != 0:
                    raise RuntimeError(
                        "sandbox verifier failed with exit code "
                        f"{process.returncode}; artifacts: {artifact_root}"
                    )
                args.summary_out.parent.mkdir(parents=True, exist_ok=True)
                sandbox.filesystem.copy_to_local(
                    "/tmp/oligoformer_upstream_equivalence_summary.json",
                    args.summary_out,
                )
            finally:
                if process_finished or not process_started:
                    try:
                        sandbox.terminate()
                    finally:
                        sandbox.detach()
                else:
                    print(
                        "OligoFormer verifier client exited before process "
                        f"completion; preserving Sandbox {sandbox.object_id}",
                        flush=True,
                    )
                    sandbox.detach()

    print(f"OligoFormer upstream-equivalence summary: {args.summary_out}")
    print(f"OligoFormer upstream-equivalence artifacts: {artifact_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the verifier CLI."""
    args = _parse_args(argv)
    if args.remote_worker is not None:
        return run_remote_worker(args.remote_worker)
    return _run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
