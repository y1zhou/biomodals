"""Pure table helpers for the PPIFlow workflow."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import orjson
import polars as pl

from biomodals.schema import AppRunStatus


def candidate_key(file_name: str) -> str:
    """Return the original structure stem from a collision-safe artifact name."""
    stem = Path(str(file_name)).stem.lower()
    stem = re.sub(r"_seed-\d+_sample-\d+_model$", "", stem)
    for suffix in (
        "_summary_confidences",
        "_confidences",
        "_ranking_scores",
        "_scores",
        "_model",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = stem.rsplit("__", 1)[-1]
    if "_refold-" in stem:
        stem = stem.rsplit("_refold-", 1)[-1]
    return stem


def row_passes_filters(
    row: Mapping[str, object], filters: Mapping[str, object]
) -> bool:
    """Return whether a score-table row satisfies all configured filters."""
    comparisons = {
        ">": lambda value, threshold: value > threshold,
        ">=": lambda value, threshold: value >= threshold,
        "<": lambda value, threshold: value < threshold,
        "<=": lambda value, threshold: value <= threshold,
        "==": lambda value, threshold: value == threshold,
        "!=": lambda value, threshold: value != threshold,
    }
    for metric, condition in filters.items():
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing filter metric {metric!r}") from exc

        clauses: list[tuple[str, float]] = []
        if isinstance(condition, Mapping):
            clauses = [
                (str(op), float(threshold)) for op, threshold in condition.items()
            ]
        elif isinstance(condition, str):
            for raw_clause in condition.split(","):
                match = re.fullmatch(
                    r"\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*",
                    raw_clause,
                )
                if match is None:
                    raise ValueError(f"Invalid filter clause: {raw_clause!r}")
                clauses.append((match.group(1), float(match.group(2))))
        else:
            clauses = [(">=", float(condition))]

        for op, threshold in clauses:
            comparison = comparisons.get(op)
            if comparison is None:
                raise ValueError(f"Unsupported filter operator: {op}")
            if not comparison(value, threshold):
                return False
    return True


def mpnn_sequence_rows_from_fasta_files(
    files: Sequence[tuple[str, bytes]],
    *,
    stage_name: str,
    parent_candidate_id: str | None = None,
) -> list[dict[str, object]]:
    """Extract `mpnn_seqs.csv` rows from FASTA-like LigandMPNN files."""
    rows = []
    for file_name, data in files:
        if Path(file_name).suffix.lower() not in {".fa", ".faa", ".fasta"}:
            continue
        for header, sequence in parse_fasta_records(data.decode("utf-8")):
            rows.append({
                "candidate_id": candidate_key(header or file_name),
                "parent_candidate_id": parent_candidate_id,
                "stage_name": stage_name,
                "source_file": file_name,
                "sequence_id": header,
                "sequence": sequence,
            })
    return rows


def parse_fasta_records(text: str) -> list[tuple[str, str]]:
    """Parse FASTA records into `(header, sequence)` pairs."""
    records = []
    header: str | None = None
    chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif header is not None:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def refold_metric_rows_from_json_files(
    files: Sequence[tuple[str, bytes]],
    *,
    stage_name: str,
) -> list[dict[str, object]]:
    """Extract candidate-keyed AlphaFold3 ReFold metrics from JSON files."""
    rows = []
    for file_name, data in files:
        path = Path(file_name)
        if path.suffix.lower() != ".json":
            continue
        payload = orjson.loads(data)
        if not isinstance(payload, Mapping):
            continue
        metric_values = {
            key: value
            for key, value in payload.items()
            if value is None or isinstance(value, str | int | float | bool)
        }
        if not metric_values:
            continue
        rows.append({
            "candidate_id": candidate_key(path.name),
            "stage_name": stage_name,
            "source_file": file_name,
            **metric_values,
        })
    return rows


def score_table_status(
    *,
    requested_count: int,
    usable_rows: int,
    failed_count: int = 0,
) -> AppRunStatus:
    """Classify a candidate-wide score table result."""
    if requested_count < 1:
        return AppRunStatus.FAILED
    if usable_rows >= requested_count and failed_count == 0:
        return AppRunStatus.SUCCEEDED
    if usable_rows > 0:
        return AppRunStatus.PARTIAL
    return AppRunStatus.FAILED


def score_frame_with_candidate_ids(
    score_frame: pl.DataFrame,
    *,
    filename_col: str,
    manifest_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return a score frame with candidate ids derived or mapped from filenames."""
    if "candidate_id" in score_frame.columns:
        return score_frame
    if filename_col not in score_frame.columns:
        raise ValueError(f"Score table is missing candidate column {filename_col!r}")

    frame = score_frame.with_columns(
        pl
        .col(filename_col)
        .cast(pl.Utf8)
        .map_elements(candidate_key, return_dtype=pl.Utf8)
        .alias("_candidate_key")
    )
    if manifest_frame is None or manifest_frame.is_empty():
        return frame.rename({"_candidate_key": "candidate_id"})

    key_pairs = manifest_candidate_key_pairs(manifest_frame)
    if not key_pairs:
        return frame.rename({"_candidate_key": "candidate_id"})

    lookup = pl.DataFrame(key_pairs).unique("_candidate_key", keep="first")
    return (
        frame
        .join(lookup, on="_candidate_key", how="left")
        .with_columns(
            pl.coalesce("candidate_id", "_candidate_key").alias("candidate_id")
        )
        .drop("_candidate_key")
    )


def filter_candidates(
    *,
    manifest_frame: pl.DataFrame,
    score_frame: pl.DataFrame,
    filters: Mapping[str, object],
    filename_col: str,
    stage_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Filter candidates and return retained manifest, scores, and audit rows."""
    scores = score_frame_with_candidate_ids(
        score_frame,
        filename_col=filename_col,
        manifest_frame=manifest_frame,
    )
    manifest_ids = set(manifest_frame.get_column("candidate_id").to_list())
    score_by_id = {
        str(row["candidate_id"]): row for row in scores.iter_rows(named=True)
    }
    audit_rows = []
    retained_ids = []
    for row in manifest_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        score_row = score_by_id.get(candidate_id)
        if score_row is None:
            audit_rows.append({
                "candidate_id": candidate_id,
                "stage_name": stage_name,
                "passed": False,
                "reason": "missing_score",
            })
            continue
        passed = row_passes_filters(score_row, filters)
        if passed:
            retained_ids.append(candidate_id)
        audit_rows.append({
            "candidate_id": candidate_id,
            "stage_name": stage_name,
            "passed": passed,
            "reason": "passed" if passed else "filtered",
            **_filter_metric_values(score_row, filters),
        })

    missing_manifest_ids = sorted(set(score_by_id).difference(manifest_ids))
    for candidate_id in missing_manifest_ids:
        audit_rows.append({
            "candidate_id": candidate_id,
            "stage_name": stage_name,
            "passed": False,
            "reason": "missing_manifest",
        })

    retained_manifest = manifest_frame.filter(
        pl.col("candidate_id").is_in(retained_ids)
    )
    retained_scores = scores.filter(pl.col("candidate_id").is_in(retained_ids))
    return retained_manifest, retained_scores, pl.DataFrame(audit_rows)


def candidate_attrition_rows(
    *,
    stage_name: str,
    manifest_frame: pl.DataFrame,
    audit_frame: pl.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Return aggregate retained/rejected/failed/skipped counts for one stage."""
    status_counts = _value_counts(manifest_frame, "candidate_status")
    passed = 0
    rejected = 0
    if audit_frame is not None and "passed" in audit_frame.columns:
        passed = audit_frame.filter(pl.col("passed") == True).height  # noqa: E712
        rejected = audit_frame.filter(pl.col("passed") == False).height  # noqa: E712
    return [
        {
            "stage_name": stage_name,
            "input_candidates": manifest_frame.height,
            "retained": passed,
            "rejected": rejected,
            "failed": int(status_counts.get(AppRunStatus.FAILED.value, 0)),
            "partial": int(status_counts.get(AppRunStatus.PARTIAL.value, 0)),
            "succeeded": int(status_counts.get(AppRunStatus.SUCCEEDED.value, 0)),
        }
    ]


def ranked_design_rows(
    *,
    structures: Sequence[tuple[str, bytes]],
    score_frames: Sequence[pl.DataFrame],
    gentype: str,
    dockq_threshold: float,
) -> list[dict[str, object]]:
    """Rank retained structures using available DockQ/AF3/Rosetta score rows."""
    csv_rows = [row for frame in score_frames for row in frame.iter_rows(named=True)]
    dockq_by_key = {
        _row_key(row): row for row in csv_rows if _has_nonempty(row, "dockq")
    }
    af3_by_key = {
        _row_key(row): row
        for row in csv_rows
        if any("iptm" in column.lower() for column in row)
    }
    rosetta_by_key = {
        _row_key(row): row for row in csv_rows if _has_nonempty(row, "interface_score")
    }
    rank_columns = (
        ("iptm", "ranking_score")
        if gentype == "binder"
        else ("iptm_A_C", "chain_A_iptm", "iptm", "ranking_score")
    )
    ranked = []
    for file_name, _ in structures:
        key = candidate_key(file_name)
        dockq_row = dockq_by_key.get(key, {})
        af3_row = af3_by_key.get(key, {})
        rosetta_row = rosetta_by_key.get(key, {})
        dockq = float(dockq_row["dockq"]) if _has_nonempty(dockq_row, "dockq") else None
        if dockq is not None and dockq <= dockq_threshold:
            continue
        rank_metric = next(
            (
                float(af3_row[column])
                for column in rank_columns
                if _has_nonempty(af3_row, column)
            ),
            None,
        )
        interface_score = (
            float(rosetta_row["interface_score"])
            if _has_nonempty(rosetta_row, "interface_score")
            else None
        )
        score = (
            100 * rank_metric - interface_score
            if rank_metric is not None and interface_score is not None
            else dockq
            if dockq is not None
            else rank_metric
        )
        if score is None:
            continue
        ranked.append({
            "design": key,
            "filename": file_name,
            "rank_score": score,
            "dockq": dockq,
            "iptm": rank_metric,
            "interface_score": interface_score,
        })
    ranked.sort(key=lambda row: float(row["rank_score"]), reverse=True)
    return ranked


def render_report_markdown(
    *,
    step_name: str,
    artifact_count: int,
    ranked_rows: Sequence[Mapping[str, object]],
    attrition_rows: Sequence[Mapping[str, object]] = (),
    max_rows: int = 25,
) -> str:
    """Render a compact PPIFlow Markdown report."""
    lines = [
        "# PPIFlow Workflow Report",
        "",
        f"- Step: {step_name}",
        f"- Input artifacts: {artifact_count}",
        f"- Ranked designs: {len(ranked_rows)}",
        "",
    ]
    if attrition_rows:
        lines.extend(["## Candidate Attrition", ""])
        lines.extend(_markdown_table(attrition_rows))
        lines.append("")
    if ranked_rows:
        lines.extend(["## Ranked Designs", ""])
        lines.extend(_markdown_table(ranked_rows[:max_rows]))
    return "\n".join(lines) + "\n"


def render_report_html(markdown: str) -> str:
    """Render the Markdown report as simple escaped HTML."""
    import html

    return (
        "<!doctype html><html><body><pre>"
        + html.escape(markdown)
        + ("</pre></body></html>\n")
    )


def manifest_candidate_key_pairs(
    manifest_frame: pl.DataFrame,
) -> list[dict[str, str]]:
    """Return lookup rows mapping known candidate filenames to candidate ids."""
    rows = []
    for row in manifest_frame.iter_rows(named=True):
        candidate_id = str(row["candidate_id"])
        keys = {
            candidate_key(str(row.get("source_path") or "")),
            candidate_key(str(row.get("derived_path") or "")),
        }
        for file_record in row.get("files") or []:
            if isinstance(file_record, Mapping):
                for field_name in ("path", "app_volume_path", "workflow_path"):
                    if file_record.get(field_name):
                        keys.add(candidate_key(str(file_record[field_name])))
        rows.extend(
            {"_candidate_key": key, "candidate_id": candidate_id} for key in keys if key
        )
    return rows


def _filter_metric_values(
    row: Mapping[str, object],
    filters: Mapping[str, object],
) -> dict[str, object]:
    return {f"metric_{metric}": row.get(metric) for metric in filters}


def _value_counts(frame: pl.DataFrame, column: str) -> dict[object, int]:
    if column not in frame.columns or frame.is_empty():
        return {}
    counts = frame.group_by(column).len()
    return {row[column]: int(row["len"]) for row in counts.iter_rows(named=True)}


def _row_key(row: Mapping[str, object]) -> str:
    for column in (
        "candidate_id",
        "target_name",
        "reference",
        "reference_pdb",
        "pdb_name",
        "description",
        "filename",
        "name",
        "id",
    ):
        if row.get(column):
            return candidate_key(str(row[column]))
    return ""


def _has_nonempty(row: Mapping[str, object], column: str) -> bool:
    return row.get(column) not in (None, "")


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row.get(column, "")).replace("|", "\\|") for column in columns
            )
            + " |"
        )
    return lines
