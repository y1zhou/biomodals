"""Scientific contracts for local antibody metrics and native annotations."""

from types import SimpleNamespace

import pytest

from biomodals.helper.antibody import (
    SCHEMES,
    analyze_chain,
    analyze_pair,
    assign_germlines,
    combined_pi,
    normalize_sequence,
    protein_metrics,
    sequence_detail,
    sequence_liabilities,
)

VH = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGGYFDYWGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


def test_full_sequence_metrics_and_continuous_pair_pi():
    """Pin physical units and concatenation rather than averaging chain pIs."""
    metrics = protein_metrics("ACMWYC")
    assert metrics["extinction_reduced"] == 6990
    assert metrics["extinction_oxidized"] == 7115
    assert metrics["gravy"] == pytest.approx((1.8 + 2.5 + 1.9 - 0.9 - 1.3 + 2.5) / 6)
    # Average free-amino-acid masses minus five waters for peptide bonds.
    assert metrics["molecular_weight_kda"] == pytest.approx(0.7759581)
    heavy = analyze_chain(VH)
    assert heavy["metrics"]["pi"] == pytest.approx(9.201266288757324)
    assert combined_pi(VH, VL) == protein_metrics(VH + VL)["pi"]
    assert combined_pi(VH, VL) != pytest.approx(
        (protein_metrics(VH)["pi"] + protein_metrics(VL)["pi"]) / 2
    )
    assert protein_metrics(VH + "HHHHHH") != heavy["metrics"]


@pytest.mark.parametrize("sequence", ["", "AXC", "A-C", "*", "A" * 513])
def test_invalid_input_is_not_silently_trimmed(sequence):
    """One canonical alphabet and length policy before scientific computation."""
    with pytest.raises(ValueError):
        analyze_chain(sequence)
    assert normalize_sequence(" ac\n d\t") == "ACD"


def test_annotation_failure_keeps_valid_physical_metrics():
    """A short canonical peptide is not assigned a false antibody germline."""
    result = analyze_chain("ACM")
    assert result["metrics"]["molecular_weight_kda"] > 0
    assert result["germlines"]["error"]
    assert result["germlines"]["v_gene"] is None
    detail = sequence_detail("ACM")
    assert detail["error"]
    assert any(item["kind"] == "methionine" for item in detail["liabilities"])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_native_numbering_preserves_input_order_and_cdrs(scheme):
    """All five conventions are native; unnumbered tails retain coordinates."""
    sequence = "GGG" + VH + "HHHHHH"
    result = sequence_detail(sequence, scheme)
    assert result["error"] is None
    assert result["cdr_definition"] == scheme
    assert result["sequence"] == sequence
    start, end = result["domain_span"]
    indices = [row["input_index"] for row in result["residues"]]
    assert indices == list(range(start, end))
    assert start > 0 and end < len(sequence)
    assert {"CDR1", "CDR2", "CDR3"} <= {row["region"] for row in result["residues"]}


def test_native_germline_evidence_and_chain_roles():
    """Pinned package produces species-qualified evidence for actual domains."""
    heavy, light = assign_germlines(VH), assign_germlines(VL)
    assert heavy["chain_type"] == "H"
    assert light["chain_type"] == "K"
    assert heavy["v_gene"] == "IGHV1-2"
    assert heavy["v"][0]["species"] == "Homo sapiens"
    assert heavy["v"][0]["known_pairs"] >= heavy["v"][0]["known_matches"] > 0
    assert heavy["j"] and light["j"]


def test_importable_pair_api_is_plain_data_and_does_not_swap_roles():
    """Standalone Python consumers get the same scientific definitions."""
    pair = analyze_pair(VH, VL)
    assert pair["vh"]["sequence"] == VH
    assert pair["vl"]["sequence"] == VL
    assert pair["vh_vl_pi"] == combined_pi(VH, VL)
    assert pair["errors"] == []
    swapped = analyze_pair(VL, VH)
    assert len(swapped["errors"]) == 2
    assert swapped["vh_vl_pi"] is None


def test_gene_display_collapses_alleles_but_preserves_all_tied_evidence(monkeypatch):
    """Display is compact while evidence keeps cross-species/gene/allele ties."""
    refs = [
        SimpleNamespace(
            id=str(i), species=species, gene=gene, allele=allele, accession="x"
        )
        for i, (species, gene, allele) in enumerate([
            ("human", "IGHV1", "01"),
            ("human", "IGHV1", "02"),
            ("mouse", "IGHV1", "01"),
            ("human", "IGHV2", "01"),
        ])
    ]
    hit = SimpleNamespace(
        references=refs,
        known_pairs=10,
        known_matches=9,
        known_fr4_pairs=0,
        reference_coverage=1.0,
        query_coverage=0.9,
        query_input_start=0,
        imgt_span=(1, 105),
    )
    monkeypatch.setattr(
        "arpeggia.number_antibody",
        lambda *a, **kw: SimpleNamespace(
            chain="H",
            diagnostics=[],
            v_match=SimpleNamespace(score=100.0, hits=[hit]),
            j_match=None,
        ),
    )
    result = assign_germlines(VH)
    assert result["v_gene"] == "IGHV1/IGHV2"
    assert len(result["v"]) == 4


def test_liability_rules_include_overlaps_and_correct_intervals():
    """Follow active LAMBS rules without W oxidation or modification claims."""
    sequence = "QCMNNSTDGSDPLLLLLLLW"
    found = {(r["kind"], r["start"], r["end"]) for r in sequence_liabilities(sequence)}
    assert ("n_terminal_glutamine", 0, 1) in found
    assert ("odd_cysteine_count", 1, 2) in found
    assert ("methionine", 2, 3) in found
    assert ("n_glycosylation", 3, 6) in found
    assert ("n_glycosylation", 4, 7) in found
    assert ("asn_deamidation", 3, 5) in found
    assert ("asn_deamidation", 4, 6) in found
    assert ("asp_isomerization", 7, 9) in found
    assert ("acid_cleavage", 10, 12) in found
    assert ("hydrophobic_patch", 12, 19) in found
    assert not sequence_liabilities("CCNPTAW")
