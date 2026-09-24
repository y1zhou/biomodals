"""Native chemistry preflight without any remote call or model prediction."""

# ruff: noqa: D103

import hashlib
import os

import orjson
import pytest

from biomodals.app.fold.alphafold3 import chemistry


def glycan_document():
    return {
        "name": "glycan",
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 4,
        "sequences": [
            {"protein": {"id": "A", "sequence": "ANST"}},
            {"ligand": {"id": "B", "ccdCodes": ["NAG", "NAG", "BMA", "MAN", "MAN"]}},
        ],
        "bondedAtomPairs": [
            [["A", 2, "ND2"], ["B", 1, "C1"]],
            [["B", 1, "O4"], ["B", 2, "C1"]],
            [["B", 2, "O4"], ["B", 3, "C1"]],
            [["B", 3, "O3"], ["B", 4, "C1"]],
            [["B", 3, "O6"], ["B", 5, "C1"]],
        ],
    }


def test_preview_keeps_copy_ids_sites_and_native_bonds():
    document = glycan_document()
    document["sequences"][0]["protein"]["modifications"] = [
        {"ptmType": "SEP", "ptmPosition": 3}
    ]
    preview = chemistry.ChemistryPreview.model_validate(
        chemistry.chemistry_summary(document)
    )
    assert preview.modifications[0].model_dump() == {
        "ids": ["A"],
        "entity_type": "protein",
        "position": 3,
        "ccd_code": "SEP",
    }
    assert preview.bonds[0] == (("A", 2, "ND2"), ("B", 1, "C1"))
    assert preview.ligands[0].ccd_codes == ["NAG", "NAG", "BMA", "MAN", "MAN"]


@pytest.mark.parametrize(
    "value",
    [
        {"userCCD": "/etc/passwd"},
        {"sequences": [{"protein": {"unpairedMsaPath": "msa"}}]},
    ],
)
def test_inline_boundary_blocks_paths(value):
    with pytest.raises(ValueError):
        chemistry.require_inline_inputs(value)


@pytest.fixture
def native_python(monkeypatch):
    executable = os.environ.get("BIOMODALS_AF3_PYTHON")
    if not executable:
        pytest.skip("Set BIOMODALS_AF3_PYTHON to the pinned native AF3 environment")
    monkeypatch.setattr(chemistry.sys, "executable", executable)


def test_native_glycan_and_ptm_receipt(native_python):
    document = glycan_document()
    document["sequences"][0]["protein"]["modifications"] = [
        {"ptmType": "SEP", "ptmPosition": 3}
    ]
    content = orjson.dumps(document)
    receipt = chemistry.ChemistryReceipt.model_validate(
        chemistry.check_chemistry_isolated(content)["receipt"]
    )
    assert receipt.input_sha256 == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("unknown_component", "unknown CCD"),
        ("unknown_atom", "atom"),
        ("self_bond", "itself"),
        ("reversed_bond", "Duplicate"),
        ("polymer_bond", "Polymer-polymer"),
        ("duplicate_ptm", "one modification"),
    ],
)
def test_native_rejects_invalid_or_unsupported_chemistry(native_python, case, expected):
    document = glycan_document()
    bonds = document["bondedAtomPairs"]
    if case == "unknown_component":
        document["sequences"][1]["ligand"]["ccdCodes"][0] = "UNKNOWN_COMPONENT"
    elif case == "unknown_atom":
        bonds[0][0][2] = "NOT_AN_ATOM"
    elif case == "self_bond":
        bonds.append([bonds[0][0], bonds[0][0]])
    elif case == "reversed_bond":
        bonds.append(list(reversed(bonds[0])))
    elif case == "polymer_bond":
        bonds.append([["A", 1, "CA"], ["A", 2, "CA"]])
    elif case == "duplicate_ptm":
        document["sequences"][0]["protein"]["modifications"] = [
            {"ptmType": "SEP", "ptmPosition": 3},
            {"ptmType": "SEP", "ptmPosition": 3},
        ]
    result = chemistry.check_chemistry_isolated(orjson.dumps(document))
    assert expected.lower() in result["error"].lower()
