"""Tests for PPIFlow table helpers."""

# ruff: noqa: D103

import polars as pl
import pytest

from biomodals.schema import AppRunStatus
from biomodals.workflow.ppiflow import tables


def test_candidate_key_recovers_original_structure_stem() -> None:
    assert tables.candidate_key("artifact__nested__design-1.pdb") == "design-1"
    assert tables.candidate_key("design-2.cif") == "design-2"
    assert (
        tables.candidate_key(
            "ppiflow_5b8c_vhh_refold-ppiflow_5b8c_vhh_stage2_abmpnn-sample0_1_model.cif"
        )
        == "ppiflow_5b8c_vhh_stage2_abmpnn-sample0_1"
    )
    assert (
        tables.candidate_key(
            "ppiflow_5b8c_vhh_refold-ppiflow_5b8c_vhh_stage2_abmpnn-sample0_1_seed-1_sample-0_model.cif"
        )
        == "ppiflow_5b8c_vhh_stage2_abmpnn-sample0_1"
    )


def test_row_passes_filters_supports_numeric_clauses() -> None:
    row = {"iptm": "0.81", "dockq": 0.55}

    assert tables.row_passes_filters(row, {"iptm": "> 0.8", "dockq": ">= 0.5"})
    assert not tables.row_passes_filters(row, {"iptm": "> 0.9"})


def test_row_passes_filters_rejects_bad_clause() -> None:
    with pytest.raises(ValueError, match="Invalid filter clause"):
        tables.row_passes_filters({"iptm": 0.8}, {"iptm": "roughly 0.8"})


def test_mpnn_sequence_rows_from_fasta_files() -> None:
    rows = tables.mpnn_sequence_rows_from_fasta_files(
        [("outputs/seqs/design.fa", b">design-a\nACD\n>design-b\nEFG\n")],
        stage_name="MPNNStep_stage1",
        parent_candidate_id="parent",
    )

    assert rows == [
        {
            "candidate_id": "design-a",
            "parent_candidate_id": "parent",
            "stage_name": "MPNNStep_stage1",
            "source_file": "outputs/seqs/design.fa",
            "sequence_id": "design-a",
            "sequence": "ACD",
        },
        {
            "candidate_id": "design-b",
            "parent_candidate_id": "parent",
            "stage_name": "MPNNStep_stage1",
            "source_file": "outputs/seqs/design.fa",
            "sequence_id": "design-b",
            "sequence": "EFG",
        },
    ]


def test_refold_metric_rows_from_json_files() -> None:
    rows = tables.refold_metric_rows_from_json_files(
        [
            (
                "outputs/design-a_summary_confidences.json",
                b'{"ranking_score":0.7,"iptm":0.8,"nested":{"skip":true}}',
            )
        ],
        stage_name="ReFoldStep",
    )

    assert rows == [
        {
            "candidate_id": "design-a",
            "stage_name": "ReFoldStep",
            "source_file": "outputs/design-a_summary_confidences.json",
            "ranking_score": 0.7,
            "iptm": 0.8,
        }
    ]


def test_score_table_status_classifies_candidate_outcomes() -> None:
    assert (
        tables.score_table_status(requested_count=2, usable_rows=2)
        == AppRunStatus.SUCCEEDED
    )
    assert (
        tables.score_table_status(requested_count=2, usable_rows=1, failed_count=1)
        == AppRunStatus.PARTIAL
    )
    assert (
        tables.score_table_status(requested_count=2, usable_rows=0, failed_count=2)
        == AppRunStatus.FAILED
    )


def test_filter_candidates_returns_retained_manifest_scores_and_audit() -> None:
    manifest = pl.DataFrame({
        "candidate_id": ["candidate-a", "candidate-b"],
        "candidate_status": ["succeeded", "succeeded"],
        "source_path": ["models/design-a.pdb", "models/design-b.pdb"],
        "derived_path": ["models/design-a.pdb", "models/design-b.pdb"],
        "files": [[], []],
    })
    scores = pl.DataFrame({
        "description": ["design-a.pdb", "design-b.pdb"],
        "iptm": [0.9, 0.5],
    })

    retained_manifest, retained_scores, audit = tables.filter_candidates(
        manifest_frame=manifest,
        score_frame=scores,
        filters={"iptm": "> 0.7"},
        filename_col="description",
        stage_name="FilterStep_stage1",
    )

    assert retained_manifest.get_column("candidate_id").to_list() == ["candidate-a"]
    assert retained_scores.get_column("candidate_id").to_list() == ["candidate-a"]
    assert audit.select("candidate_id", "passed", "reason").to_dicts() == [
        {"candidate_id": "candidate-a", "passed": True, "reason": "passed"},
        {"candidate_id": "candidate-b", "passed": False, "reason": "filtered"},
    ]


def test_candidate_attrition_counts_statuses_and_filter_results() -> None:
    manifest = pl.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "candidate_status": ["succeeded", "failed", "partial"],
    })
    audit = pl.DataFrame({"candidate_id": ["a", "b"], "passed": [True, False]})

    assert tables.candidate_attrition_rows(
        stage_name="FilterStep",
        manifest_frame=manifest,
        audit_frame=audit,
    ) == [
        {
            "stage_name": "FilterStep",
            "input_candidates": 3,
            "retained": 1,
            "rejected": 1,
            "failed": 1,
            "partial": 1,
            "succeeded": 1,
        }
    ]


def test_ranked_design_rows_exclude_below_threshold_and_missing_scores() -> None:
    rows = tables.ranked_design_rows(
        structures=[
            ("artifact__design-a.pdb", b"ATOM A\n"),
            ("artifact__design-b.pdb", b"ATOM B\n"),
            ("artifact__design-c.pdb", b"ATOM C\n"),
        ],
        score_frames=[
            pl.DataFrame({"reference": ["design-a", "design-b"], "dockq": [0.8, 0.2]}),
            pl.DataFrame({"description": ["design-a"], "iptm": [0.9]}),
        ],
        gentype="binder",
        dockq_threshold=0.49,
    )

    assert rows == [
        {
            "design": "design-a",
            "filename": "artifact__design-a.pdb",
            "rank_score": 0.8,
            "dockq": 0.8,
            "iptm": 0.9,
            "interface_score": None,
        }
    ]


def test_render_report_markdown_includes_attrition_and_ranked_rows() -> None:
    markdown = tables.render_report_markdown(
        step_name="ReportStep",
        artifact_count=3,
        ranked_rows=[{"design": "design-a", "rank_score": 1.0}],
        attrition_rows=[{"stage_name": "FilterStep", "retained": 1, "rejected": 2}],
    )

    assert "## Candidate Attrition" in markdown
    assert "| FilterStep | 1 | 2 |" in markdown
    assert "## Ranked Designs" in markdown
