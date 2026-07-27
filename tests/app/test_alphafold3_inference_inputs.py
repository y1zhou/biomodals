"""Tests for AlphaFold3 inference input serialization."""

from __future__ import annotations

import orjson
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.inference_inputs import serialize_af3_input


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
