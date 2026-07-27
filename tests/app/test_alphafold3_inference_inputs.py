"""Tests for AlphaFold3 inference input serialization."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3 import inference_inputs
from biomodals.app.fold.alphafold3.inference_inputs import (
    MAX_MODEL_SEEDS,
    materialize_local_input,
    normalize_model_seeds,
    serialize_af3_input,
    validate_inference_parameters,
    validate_inference_worker_budget,
    validate_upstream_af3_input,
)


def test_serialize_af3_input_emits_one_chain_type_per_sequence() -> None:
    """Null sibling chain types must not reach the strict upstream parser."""
    config = AF3Config(
        name="two-proteins",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                    unpairedMsa="",
                    pairedMsa="",
                    templates=[],
                )
            ),
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="H",
                    sequence="FGHI",
                    unpairedMsa=">query\nFGHI\n",
                    pairedMsa="",
                    templates=[],
                )
            ),
        ],
    )

    document = orjson.loads(serialize_af3_input(config))

    assert [list(entry) for entry in document["sequences"]] == [
        ["protein"],
        ["protein"],
    ]
    assert document["sequences"][0]["protein"]["unpairedMsa"] == ""
    assert document["sequences"][0]["protein"]["pairedMsa"] == ""
    assert document["sequences"][0]["protein"]["templates"] == []


def test_no_search_resolution_returns_a_validated_config() -> None:
    """The local coordinator should keep models typed until remote staging."""
    config = AF3Config(
        name="single-protein",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                )
            )
        ],
    )

    resolved = alphafold3_app._search_msa_and_templates(
        config,
        search_msa=False,
    )

    assert isinstance(resolved, AF3Config)
    protein = resolved.sequences[0].protein
    assert protein is not None
    assert protein.unpairedMsa == ""
    assert protein.pairedMsa == ""
    assert protein.templates == []
    original = config.sequences[0].protein
    assert original is not None
    assert original.unpairedMsa is None
    assert original.pairedMsa is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            AF3Config(
                name="duplicate",
                modelSeeds=[1],
                sequences=[
                    AF3SequenceEntry(
                        protein=AF3Protein(id=["A", "B"], sequence="ACDE")
                    ),
                    AF3SequenceEntry(protein=AF3Protein(id="B", sequence="FGHI")),
                ],
            ),
            "duplicate IDs",
        ),
        (
            AF3Config(
                name="bad-sequence",
                modelSeeds=[1],
                sequences=[
                    AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACD-E")),
                ],
            ),
            "only letters",
        ),
        (
            AF3Config(
                name="bad-seed",
                modelSeeds=[2**32],
                sequences=[
                    AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
                ],
            ),
            "32-bit unsigned",
        ),
    ],
)
def test_upstream_preflight_rejects_invalid_inputs(
    config: AF3Config,
    message: str,
) -> None:
    """Invalid upstream inputs should fail during the local preflight."""
    with pytest.raises(ValueError, match=message):
        validate_upstream_af3_input(config)


def test_search_preflight_rejects_invalid_input_before_remote_work() -> None:
    """The coordinator should reject malformed chains before its first remote call."""
    config = AF3Config(
        name="bad-sequence",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACD-E")),
        ],
    )

    with pytest.raises(ValueError, match="only letters"):
        alphafold3_app._search_msa_and_templates(config)


def _write_path_backed_msa_input(tmp_path: Path, msa_path: str) -> Path:
    input_path = tmp_path / "input.json"
    input_path.write_text(
        AF3Config(
            name="bounded",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsaPath=msa_path,
                    )
                )
            ],
        ).model_dump_json(exclude_none=True),
        encoding="utf-8",
    )
    return input_path


def test_local_materialization_bounds_path_backed_msa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Path-backed MSA reads should enforce their byte budget."""
    msa_path = tmp_path / "input.a3m"
    msa_path.write_text(">query\nACDE\n", encoding="utf-8")
    input_path = _write_path_backed_msa_input(tmp_path, msa_path.name)
    monkeypatch.setattr(inference_inputs, "MAX_LOCAL_MSA_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        materialize_local_input(input_path)


def test_local_materialization_rejects_path_backed_msa_symlink(
    tmp_path: Path,
) -> None:
    """Caller-local artifacts should not traverse a final symbolic link."""
    target_path = tmp_path / "target.a3m"
    target_path.write_text(">query\nACDE\n", encoding="utf-8")
    link_path = tmp_path / "input.a3m"
    link_path.symlink_to(target_path)
    input_path = _write_path_backed_msa_input(tmp_path, link_path.name)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        materialize_local_input(input_path)


def test_inference_parameters_are_resource_bounded() -> None:
    """Inference fan-out and sampling controls should have finite bounds."""
    with pytest.raises(ValueError, match="between 0 and"):
        validate_inference_parameters(101, 1)
    with pytest.raises(ValueError, match="between 1 and"):
        validate_inference_parameters(1, 101)
    with pytest.raises(ValueError, match="between 1 and"):
        validate_inference_worker_budget(101)


def test_seed_cap_applies_to_requests_not_accumulated_serialization() -> None:
    """An accumulated run may legitimately contain more seeds than one request."""
    seeds = list(range(MAX_MODEL_SEEDS + 1))
    config = AF3Config(
        name="accumulated",
        modelSeeds=seeds,
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )

    assert orjson.loads(serialize_af3_input(config))["modelSeeds"] == seeds
    with pytest.raises(ValueError, match="no more than"):
        normalize_model_seeds(seeds)
