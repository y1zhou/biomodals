"""Tests for AlphaFold3 inference input serialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
import pytest
from uniaf3.schema.alphafold3 import (
    AF3Config,
    AF3Ligand,
    AF3Protein,
    AF3SequenceEntry,
    AF3Template,
)

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3 import inference_inputs
from biomodals.app.fold.alphafold3.inference_inputs import (
    MAX_MODEL_SEEDS,
    MAX_SEED_SAMPLE_PAIRS,
    materialize_local_input,
    normalize_model_seeds,
    prepare_inference_run,
    serialize_af3_input,
    validate_inference_parameters,
    validate_inference_worker_budget,
    validate_inference_workload,
    validate_upstream_af3_input,
)


def _chain_id(index: int) -> str:
    """Return an Excel-style uppercase chain ID."""
    value = ""
    while True:
        index, remainder = divmod(index, 26)
        value = chr(ord("A") + remainder) + value
        if index == 0:
            return value
        index -= 1


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


def test_upstream_preflight_bounds_expanded_entities_and_polymer_residues() -> None:
    """Entity expansion and polymer tokens should have explicit support limits."""
    entity_ids = [_chain_id(index) for index in range(5_121)]
    too_many_entities = AF3Config(
        name="too-many-entities",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                ligand=AF3Ligand(id=entity_ids, ccdCodes=["ATP"]),
            )
        ],
    )
    too_many_residues = AF3Config(
        name="too-many-residues",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(id="A", sequence="A" * 5_121),
            )
        ],
    )

    with pytest.raises(ValueError, match="5,120 expanded entities"):
        validate_upstream_af3_input(too_many_entities)
    with pytest.raises(ValueError, match="5,120 total polymer residues"):
        validate_upstream_af3_input(too_many_residues)


def test_inference_staging_bounds_the_serialized_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final request JSON should be bounded before Volume publication."""
    monkeypatch.setattr(inference_inputs, "MAX_STAGED_INPUT_BYTES", 128)
    config = AF3Config(
        name="bounded-staging",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                    unpairedMsa=">query\nACDE\n",
                    pairedMsa="",
                    templates=[],
                )
            )
        ],
    )

    with pytest.raises(ValueError, match="staged input exceeds the 128-byte limit"):
        prepare_inference_run(
            config,
            recycle=1,
            sample=1,
        )


def test_inference_staging_bounds_the_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable identity document should share the staged-input ceiling."""
    config = AF3Config(
        name="bounded-identity",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(
                protein=AF3Protein(
                    id="A",
                    sequence="ACDE",
                    unpairedMsa=">query\nACDE\n",
                    pairedMsa="",
                    templates=[],
                )
            )
        ],
    )
    prepared = prepare_inference_run(
        config,
        recycle=1,
        sample=1,
    )
    identity_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "identity.json"
    )
    monkeypatch.setattr(
        inference_inputs,
        "MAX_STAGED_INPUT_BYTES",
        len(identity_upload.content) - 1,
    )

    with pytest.raises(ValueError, match="run identity exceeds"):
        prepare_inference_run(
            config,
            recycle=1,
            sample=1,
        )


def test_run_identity_hashes_large_text_while_input_remains_runnable(
    tmp_path: Path,
) -> None:
    """Identity evidence should not duplicate large runnable input fields."""
    unpaired_msa = ">query\nACDE\n"
    user_ccd = "data_custom\n#\n"
    prepared = prepare_inference_run(
        AF3Config(
            name="compact-identity",
            modelSeeds=[1],
            userCCD=user_ccd,
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa=unpaired_msa,
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        ),
        recycle=1,
        sample=1,
    )
    uploads = {
        upload.relative_path.name: upload.content for upload in prepared.payload_uploads
    }
    identity = orjson.loads(uploads["identity.json"])
    identity_input = identity["input"]

    assert unpaired_msa.encode() not in uploads["identity.json"]
    assert user_ccd.encode() not in uploads["identity.json"]
    assert identity_input["sequences"][0]["protein"]["unpairedMsa"] == {
        "sha256": hashlib.sha256(unpaired_msa.encode()).hexdigest(),
        "size_bytes": len(unpaired_msa),
    }
    assert identity_input["sequences"][0]["protein"]["pairedMsa"] == {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": 0,
    }
    assert identity_input["userCCD"] == {
        "sha256": hashlib.sha256(user_ccd.encode()).hexdigest(),
        "size_bytes": len(user_ccd),
    }
    runnable = AF3Config.model_validate_json(uploads["input.json"])
    protein = runnable.sequences[0].protein
    assert protein is not None
    assert protein.unpairedMsa == unpaired_msa
    assert runnable.userCCD == user_ccd


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


def _write_path_backed_template_input(
    tmp_path: Path,
    contents: tuple[bytes, ...],
) -> Path:
    template_paths = [
        tmp_path / f"template-{index}.cif" for index in range(len(contents))
    ]
    for template_path, content in zip(template_paths, contents, strict=True):
        template_path.write_bytes(content)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        AF3Config(
            name="bounded-templates",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        templates=[
                            AF3Template(
                                mmcifPath=template_path.name,
                                queryIndices=[0],
                                templateIndices=[0],
                            )
                            for template_path in template_paths
                        ],
                    )
                )
            ],
        ).model_dump_json(exclude_none=True),
        encoding="utf-8",
    )
    return input_path


def test_local_materialization_bounds_all_path_backed_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate limits should stop before reading a later template file."""
    input_path = _write_path_backed_template_input(
        tmp_path,
        (b"12345", b"67890", b"unread"),
    )
    (tmp_path / "template-2.cif").unlink()
    monkeypatch.setattr(inference_inputs, "MAX_TEMPLATE_TOTAL_BYTES", 8)

    with pytest.raises(ValueError, match="templates exceed the 8-byte limit"):
        materialize_local_input(input_path)


def test_local_materialization_checks_template_count_before_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Template-count rejection should precede referenced-file reads."""
    input_path = _write_path_backed_template_input(tmp_path, (b"first", b"second"))
    (tmp_path / "template-0.cif").unlink()
    monkeypatch.setattr(inference_inputs, "MAX_PROTEIN_TEMPLATES", 1)

    with pytest.raises(ValueError, match="1-template limit"):
        materialize_local_input(input_path)


def test_local_materialization_rejects_empty_template_files(
    tmp_path: Path,
) -> None:
    """Empty template files should fail at the local trust boundary."""
    input_path = _write_path_backed_template_input(tmp_path, (b"",))

    with pytest.raises(ValueError, match="must contain nonempty inline mmcif"):
        materialize_local_input(input_path)


def test_local_materialization_inlines_path_backed_templates(
    tmp_path: Path,
) -> None:
    """Caller template files should become self-contained input JSON fields."""
    content = b"data_inline\n#\n"
    input_path = _write_path_backed_template_input(tmp_path, (content,))

    materialized = materialize_local_input(input_path)

    assert isinstance(materialized, AF3Config)
    protein = materialized.sequences[0].protein
    assert protein is not None
    template = protein.templates[0]
    assert template.mmcif == content.decode()
    assert template.mmcifPath is None


def test_inference_parameters_are_resource_bounded() -> None:
    """Inference fan-out and sampling controls should have finite bounds."""
    with pytest.raises(ValueError, match="between 0 and"):
        validate_inference_parameters(101, 1)
    with pytest.raises(ValueError, match="between 1 and"):
        validate_inference_parameters(1, 101)
    with pytest.raises(ValueError, match="between 1 and"):
        validate_inference_worker_budget(101)
    assert (
        validate_inference_workload(
            list(range(MAX_SEED_SAMPLE_PAIRS // 5)),
            5,
        )
        == MAX_SEED_SAMPLE_PAIRS
    )
    with pytest.raises(ValueError, match="modelSeeds × sample"):
        validate_inference_workload(
            list(range(MAX_SEED_SAMPLE_PAIRS // 5 + 1)),
            5,
        )


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
