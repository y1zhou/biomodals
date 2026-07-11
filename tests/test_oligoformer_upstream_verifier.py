"""Tests for the OligoFormer upstream-equivalence verifier helpers."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_oligoformer_upstream_equivalence.py"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("oligoformer_verifier", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_final_table(path: Path, rows: list[str]) -> None:
    path.write_text(
        "pos\tsense\tsiRNA\tefficacy\tfunc_filter\tpita_score\t"
        "targetscan_score\toff_target_filter\tSeed\tcell_viability\t"
        "toxicity_filter\tfilter\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_default_fixture_paths_exist():
    verifier = _load_verifier()

    assert verifier.DEFAULT_MRNA_FASTA.name == "sirna_target.fa"
    assert verifier.DEFAULT_MRNA_FASTA.exists()
    assert verifier.DEFAULT_UTR_REF.exists()
    assert verifier.DEFAULT_ORF_REF.exists()
    args = verifier._parse_args([])
    assert args.top_n == -1
    assert args.sirna_fasta is None
    assert not args.all_human


def test_verifier_cache_key_includes_custom_sirnas_and_human_ref_identity(
    tmp_path: Path,
):
    verifier = _load_verifier()
    mrna = tmp_path / "mrna.fa"
    sirna = tmp_path / "sirna.fa"
    unused_ref = tmp_path / "unused.fa"
    mrna.write_text(">target\nAUGC\n", encoding="utf-8")
    sirna.write_text(">guide\nAUGC\n", encoding="utf-8")

    base = verifier.verifier_cache_key(
        mrna_fasta=mrna,
        sirna_fasta=sirna,
        utr_ref=unused_ref,
        orf_ref=unused_ref,
        all_human=True,
        reference_identity="refs-v1",
        top_n=-1,
        targetscan_ref_shard_size=1000,
    )
    changed = verifier.verifier_cache_key(
        mrna_fasta=mrna,
        sirna_fasta=sirna,
        utr_ref=unused_ref,
        orf_ref=unused_ref,
        all_human=True,
        reference_identity="refs-v2",
        top_n=-1,
        targetscan_ref_shard_size=1000,
    )

    assert base != changed


def test_copy_app_artifacts_uses_final_variant_output_dir(tmp_path: Path):
    verifier = _load_verifier()
    output_dir = tmp_path / "run" / "outputs" / "postprocess-key"
    output_dir.mkdir(parents=True)
    for suffix in verifier.FINAL_TABLE_SUFFIXES:
        output_dir.joinpath(f"target{suffix}.txt").write_text(
            f"{suffix or 'original'}\n", encoding="utf-8"
        )

    verifier._copy_app_artifacts(
        {
            "output_dir": str(output_dir),
            "output_stems": ["target"],
        },
        tmp_path / "artifacts",
    )

    copied = tmp_path / "artifacts" / "app" / "outputs"
    assert {path.name for path in copied.iterdir()} == {
        "target.txt",
        "target_ranked.txt",
        "target_ranked_filtered.txt",
    }


def test_compare_tables_canonicalizes_order_and_float_format(tmp_path: Path):
    verifier = _load_verifier()
    app_path = tmp_path / "app.txt"
    upstream_path = tmp_path / "upstream.txt"
    _write_final_table(
        app_path,
        [
            "2\tCG\tUGCUAGCUAGCUAGCUAGC\t0.668\t0\t-2.0\t0.5\t0\tGCUAGC\t60\t0\t0",
            "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t-11\t2\t1\tUGCUAG\t60\t0\t1",
        ],
    )
    _write_final_table(
        upstream_path,
        [
            "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8000000\t0\t-11.0000002\t2.0000002\t1\tUGCUAG\t60\t0\t5",
            "2\tCG\tUGCUAGCUAGCUAGCUAGC\t0.6\t0\t-2.0\t0.5\t0\tGCUAGC\t60\t0\t0",
        ],
    )

    result = verifier.compare_tables(
        name="final",
        app_path=app_path,
        upstream_path=upstream_path,
        kind="final",
        canonical_dir=tmp_path / "canonical",
    )

    assert result.passed
    assert result.max_deltas["efficacy"] > 0
    assert (tmp_path / "canonical" / "app" / "final.json").exists()
    assert (tmp_path / "canonical" / "upstream" / "final.json").exists()


def test_compare_tables_rejects_filter_flag_mismatch(tmp_path: Path):
    verifier = _load_verifier()
    app_path = tmp_path / "app.txt"
    upstream_path = tmp_path / "upstream.txt"
    _write_final_table(
        app_path,
        [
            "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t-11\t2\t1\tUGCUAG\t60\t0\t1",
        ],
    )
    _write_final_table(
        upstream_path,
        [
            "1\tGC\tAUGCUAGCUAGCUAGCUAG\t0.8\t0\t-11\t2\t0\tUGCUAG\t60\t0\t1",
        ],
    )

    result = verifier.compare_tables(
        name="final",
        app_path=app_path,
        upstream_path=upstream_path,
        kind="final",
    )

    assert not result.passed
    assert "off_target_filter" in result.message


def test_compare_targetscan_no_header_table(tmp_path: Path):
    verifier = _load_verifier()
    app_path = tmp_path / "app.targetscan.tab"
    upstream_path = tmp_path / "upstream.targetscan.tab"
    app_path.write_text("tx2\tRNA1\t0.2\ntx1\tRNA0\t0.1000001\n", encoding="utf-8")
    upstream_path.write_text("tx1\tRNA0\t0.1\ntx2\tRNA1\t0.2\n", encoding="utf-8")

    result = verifier.compare_tables(
        name="targetscan",
        app_path=app_path,
        upstream_path=upstream_path,
        kind="targetscan",
    )

    assert result.passed
    assert result.row_count == 2
