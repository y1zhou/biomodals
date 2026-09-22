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
    references = result["germlines"]
    assert [row["segment"] for row in references] == ["v", "j"]
    merged = result["alignment"]
    assert merged["input"].replace("-", "") == sequence
    assert [i for i in merged["input_indices"] if i is not None] == list(
        range(len(sequence))
    )
    assert (
        len(merged["germline"])
        == len(merged["input"])
        == len(merged["germline_diffs"])
        == len(merged["input_indices"])
    )
    columns = {
        input_index: index
        for index, input_index in enumerate(merged["input_indices"])
        if input_index is not None
    }
    covered = set()
    for row, match in zip(references, (native.v_match, native.j_match), strict=True):
        hit = match.hits[0]
        alignment = hit.alignment
        assert row["reference_ids"] == [ref.id for ref in hit.references]
        assert row["tied_reference_count"] == sum(len(h.references) for h in match.hits)
        start = hit.query_input_start + alignment.query_span[0]
        # The complete local block survives contiguously, including true deletions.
        offset = columns[start]
        while offset and merged["input_indices"][offset - 1] is None:
            offset -= 1
        end = offset + len(alignment.aligned_query)
        assert merged["input"][offset:end] == alignment.aligned_query
        assert merged["germline"][offset:end] == alignment.aligned_reference
        assert merged["germline_diffs"][offset:end] == alignment.operations
        covered.update(range(offset, end))
    assert {"+", "-", "x"} <= set(merged["germline_diffs"])
    assert all(
        op == " " for i, op in enumerate(merged["germline_diffs"]) if i not in covered
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_parental_and_germline_comparisons_share_input_without_invented_homology(
    scheme,
):
    """Native global parent differences survive independent reference-only gaps."""
    from arpeggia import align_seqs, number_antibody

    sequence = "GGG" + VH[:35] + "GG" + VH[35:62] + VH[64:] + "HHHHHH"
    parent = "GG" + VH[:40] + "CC" + VH[40:] + "HHHHHH"
    native = align_seqs(parent, sequence, mode="global")
    detail = sequence_detail(sequence, scheme, parent)
    alignment = detail["alignment"]
    assert (
        len({len(value) for key, value in alignment.items() if key != "input_indices"})
        == 1
    )
    assert len(alignment["input_indices"]) == len(alignment["input"])
    assert alignment["parental"].replace("-", "").replace(" ", "") == parent
    assert alignment["input"].replace("-", "").replace(" ", "") == sequence
    # Each native comparison survives after unrelated reference-only columns
    # are removed; independent gaps never assert four-way homology.
    parent_columns = [
        i
        for i, ref in enumerate(alignment["parental"])
        if ref != " " and alignment["input"][i] != " "
    ]
    for key, expected in (
        ("parental", native.aligned_reference),
        ("input", native.aligned_query),
        ("parental_diffs", native.operations),
    ):
        assert "".join(alignment[key][i] for i in parent_columns) == expected
    for original, ref_key, query_key, diff_key, references in (
        (sequence, "germline", "input", "germline_diffs", detail["germlines"]),
        (
            parent,
            "parental_germline",
            "parental",
            "parental_germline_diffs",
            detail["parental_germlines"],
        ),
    ):
        numbered = number_antibody(original, scheme=scheme)
        projected = [
            (ref, query, op)
            for ref, query, op in zip(
                alignment[ref_key],
                alignment[query_key],
                alignment[diff_key],
                strict=True,
            )
            if ref != " " and query != " "
        ]
        for match, identity in zip(
            (numbered.v_match, numbered.j_match), references, strict=True
        ):
            hit = match.hits[0]
            assert identity["reference_ids"] == [ref.id for ref in hit.references]
            expected = list(
                zip(
                    hit.alignment.aligned_reference,
                    hit.alignment.aligned_query,
                    hit.alignment.operations,
                    strict=True,
                )
            )
            assert any(
                projected[start : start + len(expected)] == expected
                for start in range(len(projected))
            )
    same = sequence_detail(VH, scheme, VH)["alignment"]
    assert same["parental"].replace(" ", "").replace("-", "") == VH
    assert set(same["parental_diffs"]) == {" "}


def test_parent_germline_failure_preserves_candidate_and_parent_comparison():
    """An unnumberable retained parent does not hide usable candidate evidence."""
    detail = sequence_detail(VH, parental_sequence="ACDE")
    assert detail["error"] is None and detail["germlines"]
    assert detail["parental_germline_error"]
    assert detail["alignment"]["parental"].replace("-", "").replace(" ", "") == "ACDE"
    assert detail["alignment"]["input"].replace("-", "") == VH


def test_candidate_and_parent_assign_germlines_independently():
    """Before/after references are not copied from the humanized assignment."""
    okt3 = "QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCARYYDDHYCLDYWGQGTTLTVSS"
    detail = sequence_detail(VH, parental_sequence=okt3)
    assert detail["germlines"] != detail["parental_germlines"]
    assert "Homo sapiens" in detail["germlines"][0]["reference_names"][0]
    assert "Mus musculus" in detail["parental_germlines"][0]["reference_names"][0]


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
