"""Native generation reports cannot silently change the reviewed scientific input."""

# ruff: noqa: D103

import pytest

from biomodals.workflow.nanobody_humanization.artifacts import generation_frame
from biomodals.workflow.nanobody_humanization.preparation import VHInput, prepare_vh
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings

SEQUENCE = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSV"
    "KGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"
)


def test_report_identity_budget_seed_and_mask_are_bound_to_prepared_parent():
    parent = prepare_vh(VHInput(id="source", vhh=SEQUENCE))
    settings = NanobodySettings(hudiff_nb_candidate_count=1, root_seed=123)
    report = {
        "schema_version": 1,
        "method": "hudiff_nb",
        "input_sequence": parent.sequence,
        "seed": 123,
        "attempts": [{"attempt_index": 1, "sequence": parent.sequence, "error": None}],
    }
    result = generation_frame(parent, "hudiff_nb", settings, report)
    assert result["vh"].to_list() == [parent.sequence]
    assert result["seed"].to_list() == [123]
    assert result["parent_id"].to_list() == ["source"]
    for changed, message in (
        ({"seed": 0}, "seed"),
        ({"input_sequence": "changed"}, "identity"),
        ({"attempts": []}, "budget"),
    ):
        with pytest.raises(ValueError, match=message):
            generation_frame(parent, "hudiff_nb", settings, report | changed)
    index = parent.protected_indices[0]
    mutated = (
        parent.sequence[:index]
        + ("A" if parent.sequence[index] != "A" else "E")
        + parent.sequence[index + 1 :]
    )
    changed = {"attempts": [{"attempt_index": 1, "sequence": mutated, "error": None}]}
    with pytest.raises(ValueError, match="protected"):
        generation_frame(parent, "hudiff_nb", settings, report | changed)
    changed["attempts"][0]["error"] = "Native sample rejected"
    result = generation_frame(parent, "hudiff_nb", settings, report | changed)
    assert result["error"].to_list() == ["Native sample rejected"]
