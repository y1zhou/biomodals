"""Native generation reports cannot silently change the reviewed scientific input."""

# ruff: noqa: D103

import pytest

from biomodals.app.design.abnativ2_vhh.models import RUNTIME_IDENTITY, SOURCE_COMMIT
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


def test_exploration_report_binds_counts_seed_settings_and_successful_noop():
    parent = prepare_vh(VHInput(id="source", vhh=SEQUENCE))
    settings = NanobodySettings(abnativ2_explore=True, abnativ2_candidate_budget=3)
    report = {
        "schema_version": 2,
        "method": "abnativ2_vhh",
        "input_sequence": parent.sequence,
        "settings": settings.abnativ_parameters(
            parent.id, parent.sequence, parent.protected_indices
        ).model_dump(),
        "source_commit": SOURCE_COMMIT,
        "runtime_identity": RUNTIME_IDENTITY,
        "search": {
            "possible_candidates": "7",
            "evaluated_candidates": 3,
            "accepted_candidates": 0,
            "coverage": "sampled",
        },
        "attempts": [{"attempt_index": 1, "sequence": parent.sequence, "error": None}],
    }
    assert generation_frame(parent, "abnativ2_vhh", settings, report)[
        "vh"
    ].to_list() == [parent.sequence]
    for changed in (
        {"settings": report["settings"] | {"sampling_seed": 42}},
        {"runtime_identity": "old"},
        {"search": report["search"] | {"evaluated_candidates": 4}},
        {"search": report["search"] | {"coverage": "complete"}},
        {"attempts": []},
    ):
        with pytest.raises(ValueError):
            generation_frame(parent, "abnativ2_vhh", settings, report | changed)
