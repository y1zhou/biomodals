"""Scientific contracts for local antibody metrics and native annotations."""

from types import SimpleNamespace

import pytest

from biomodals.helper.antibody import (
    SCHEMES,
    _germline_pi,
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
    # Black–Mould values from Biopython's pinned hydrophobicity scale.
    assert metrics["gravy"] == pytest.approx(
        (0.616 + 0.68 + 0.738 + 0.878 + 0.88 + 0.68) / 6
    )
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
    assert result["germline_pi"] is None
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


@pytest.mark.parametrize("scheme", SCHEMES)
def test_native_germline_alignment_retains_operations_and_full_input_offsets(scheme):
    """Indels stay gapped, while query coordinates include unnumbered prefixes."""
    from arpeggia import number_antibody

    sequence = "GGG" + VH[:35] + "GG" + VH[35:62] + VH[64:] + "HHHHHH"
    result = sequence_detail(sequence, scheme)
    native = number_antibody(sequence)
    assert result["error"] is None
    alignments = result["germline_alignments"]
    assert [row["segment"] for row in alignments] == ["v", "j"]
    for row, match in zip(alignments, (native.v_match, native.j_match), strict=True):
        hit = match.hits[0]
        alignment = hit.alignment
        assert row["reference_ids"] == [ref.id for ref in hit.references]
        assert row["tied_reference_count"] == sum(len(h.references) for h in match.hits)
        assert row["aligned_reference"] == alignment.aligned_reference
        assert row["aligned_query"] == alignment.aligned_query
        assert row["operations"] == alignment.operations
        assert row["reference_start"] == alignment.reference_span[0]
        assert (
            row["query_input_start"] == hit.query_input_start + alignment.query_span[0]
        )
        query = row["aligned_query"].replace("-", "")
        start = row["query_input_start"]
        assert sequence[start : start + len(query)] == query
        assert (
            len(row["aligned_reference"])
            == len(row["aligned_query"])
            == len(row["operations"])
        )
    assert {"+", "-", "x"} <= set(alignments[0]["operations"])
    assert alignments[0]["query_input_start"] == 3


@pytest.mark.parametrize("chain", [VH, VL])
def test_germline_pi_uses_native_representatives_without_repeating_matching(
    chain, monkeypatch
):
    """The pI uses full V/J references; display numbering and query tails do not alter it."""
    import arpeggia
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    native = arpeggia.number_antibody(chain)
    reference = (
        native.v_match.hits[0].alignment.reference
        + native.j_match.hits[0].alignment.reference
    )
    calls = []
    original = arpeggia.number_antibody

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(arpeggia, "number_antibody", counted)
    result = analyze_chain(chain)
    assert calls == [chain]
    assert result["germline_pi"] == ProteinAnalysis(reference).isoelectric_point()
    assert (
        analyze_chain("GGG" + chain + "HHHHHH")["germline_pi"] == result["germline_pi"]
    )


def test_germline_pi_includes_unaligned_reference_residues_and_keeps_tie_order():
    """Use the displayed first hit, not only local matches or a tie average."""
    from arpeggia import align_seqs
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    v = align_seqs("KACDE", "ACDE", mode="local")
    j = align_seqs("RFG", "RFG", mode="local")
    numbered = SimpleNamespace(
        v_match=SimpleNamespace(
            hits=[
                SimpleNamespace(alignment=v),
                SimpleNamespace(alignment=align_seqs("DACDE", "ACDE", mode="local")),
            ]
        ),
        j_match=SimpleNamespace(hits=[SimpleNamespace(alignment=j)]),
    )
    assert v.reference_span[0] == 1
    assert _germline_pi(numbered) == ProteinAnalysis("KACDERFG").isoelectric_point()
    assert _germline_pi(numbered) != ProteinAnalysis("ACDERFG").isoelectric_point()
    numbered.j_match = None
    assert _germline_pi(numbered) is None
    numbered.j_match = SimpleNamespace(
        hits=[SimpleNamespace(alignment=SimpleNamespace(reference="XFG"))]
    )
    assert _germline_pi(numbered) is None


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
    """Selected motifs overlap and still work on unnumberable sequences."""
    sequence = "QCMNNSTDGSDPLLLLLLLW"
    found = {(r["kind"], r["start"], r["end"]) for r in sequence_liabilities(sequence)}
    assert found == {
        ("odd_cysteine_count", 1, 2),
        ("methionine", 2, 3),
        ("n_glycosylation", 3, 6),
        ("n_glycosylation", 4, 7),
        ("asn_deamidation", 3, 5),
        ("asn_deamidation", 4, 6),
        ("asp_isomerization", 7, 9),
        ("acid_cleavage", 10, 12),
    }
    assert not sequence_liabilities("CCNPTAW")


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("chain", [VH, VL])
@pytest.mark.parametrize("extra_cysteine", ["tail", "cdr"])
def test_odd_cysteine_markers_exclude_conserved_numbered_positions(
    scheme, chain, extra_cysteine
):
    """Scheme changes and input tails cannot move conserved-Cys exclusions."""
    if extra_cysteine == "tail":
        sequence = "GGG" + chain + "HHHHC"
        marked_index = len(sequence) - 1
    else:
        sequence = "GGG" + chain[:30] + "C" + chain[31:] + "HHHHHH"
        marked_index = 33
    result = sequence_detail(sequence, scheme)
    assert result["error"] is None
    assert sequence.count("C") == 3
    assert [r for r in result["liabilities"] if r["kind"] == "odd_cysteine_count"] == [
        {"kind": "odd_cysteine_count", "start": marked_index, "end": marked_index + 1}
    ]
