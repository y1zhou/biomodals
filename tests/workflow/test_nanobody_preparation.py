"""Public sequences exercise native preparation without any model/provider."""

import pytest

from biomodals.workflow.nanobody_humanization.preparation import (
    VHInput,
    preparation_digest,
    prepare_batch,
    prepare_vh,
)

VHH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


def test_preparation_trims_flanks_without_mutating_observed_domain():
    """The baseline is the detected domain, with original offsets retained."""
    original = "HHHHHH" + VHH + "GGG"
    parent = prepare_vh(VHInput(id="one", vhh=original))
    assert parent.sequence == VHH
    assert parent.original_sequence == original
    assert parent.original_indices == tuple(range(6, 6 + len(VHH)))
    assert all(not refs for refs in parent.imputation_references)
    assert (
        len(parent.sequence) == len(parent.imgt_positions) == len(parent.aho_positions)
    )


def test_imputation_uses_native_representatives_and_keeps_original_evidence():
    """Native reference selection fills only supported terminal residues."""
    from arpeggia import number_antibody

    original = VHH[5:-3]
    native = number_antibody(original)
    v = native.v_match.hits[0].references[0].id
    j = native.j_match.hits[0].references[0].id
    expected = native.impute(v_reference=v, j_reference=j)
    parent = prepare_vh(VHInput(id="partial", vhh=original))
    assert parent.sequence == expected.sequence
    assert parent.original_sequence == original
    assert (parent.v_reference, parent.j_reference) == (v, j)
    assert parent.original_indices == tuple(r.input_index for r in expected.residues)
    assert (
        "".join(
            aa
            for aa, i in zip(parent.sequence, parent.original_indices, strict=True)
            if i is not None
        )
        == original
    )
    assert any(parent.imputation_references)


@pytest.mark.parametrize("sequence", ["ACDE", VL, VHH + VHH, VHH[:30] + "X" + VHH[31:]])
def test_invalid_domains_return_addressable_errors_without_discarding_siblings(
    sequence,
):
    """Single-VH eligibility does not discard another valid preview row."""
    parents, issues = prepare_batch([
        VHInput(id="bad", vhh=sequence),
        VHInput(id="good", vhh=VHH),
    ])
    assert [parent.id for parent in parents] == ["good"]
    assert [(issue.row_index, issue.field, issue.code) for issue in issues] == [
        (0, "vhh", "domain_invalid")
    ]


def test_frozen_mask_protects_union_cysteines_and_hallmarks():
    """Map both scientific masks onto residue indices before taking their union."""
    parent = prepare_vh(VHInput(id="one", vhh=VHH))
    protected = set(parent.protected_indices)
    for index, (aa, imgt, aho) in enumerate(
        zip(parent.sequence, parent.imgt_positions, parent.aho_positions, strict=True)
    ):
        number = int(imgt.rstrip("ABCDEFGHIJKL"))
        expected = (
            aa == "C"
            or number in (42, 49, 50, 52)
            or any(
                start <= number <= end
                for start, end in ((27, 38), (55, 66), (105, 117))
            )
            or any(
                start <= aho <= end for start, end in ((27, 42), (57, 69), (108, 138))
            )
        )
        assert (index in protected) == expected
    assert 0 not in protected


def test_preparation_reuses_sequences_and_digest_binds_reviewed_identity(monkeypatch):
    """Deduplicate native work, not parental identity or the reviewed request."""
    import biomodals.workflow.nanobody_humanization.preparation as module

    original = module.prepare_vh
    calls = []

    def counted(record):
        calls.append(record.vhh)
        return original(record)

    monkeypatch.setattr(module, "prepare_vh", counted)
    parents, issues = prepare_batch([VHInput(id=key, vhh=VHH) for key in ("a", "b")])
    assert issues == [] and calls == [VHH]
    assert [parent.id for parent in parents] == ["a", "b"]
    assert preparation_digest(parents) != preparation_digest(tuple(reversed(parents)))
    assert preparation_digest(parents) != preparation_digest([
        parents[0].model_copy(update={"preparation_version": "next"}),
        parents[1],
    ])
    _, issues = prepare_batch([VHInput(id=key, vhh=VHH) for key in ("a b", "a_b")])
    assert [issue.row_index for issue in issues] == [0, 1]
