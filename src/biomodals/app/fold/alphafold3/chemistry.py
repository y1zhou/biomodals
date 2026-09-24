"""Isolated native CCD validation; no searches, conformers or inference."""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.fold.alphafold3.inference_inputs import MAX_INPUT_JSON_BYTES
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.helper.artifacts import file_size_sha256

CHEMISTRY_TIMEOUT_SECONDS = 120
CHEMISTRY_PROTOCOL = 1


class ChemistryReceipt(BaseModel):
    """Evidence for exactly the bytes checked in the pinned CCD environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: int = Field(default=CHEMISTRY_PROTOCOL, ge=1, le=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ccd_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_commit: str = Field(pattern=f"^{ALPHAFOLD3_COMMIT}$")


class ChemistryModification(BaseModel):
    """One input-indexed replacement, shared by every listed chain copy."""

    ids: list[str]
    entity_type: Literal["protein", "rna", "dna"]
    position: int
    ccd_code: str


class ChemistryLigand(BaseModel):
    """Named components of a ligand; empty for a SMILES ligand."""

    ids: list[str]
    ccd_codes: list[str]


class ChemistryPreview(BaseModel):
    """Confirmation data with native one-based residue and atom endpoints."""

    modifications: list[ChemistryModification]
    ligands: list[ChemistryLigand]
    bonds: list[tuple[tuple[str, int, str], tuple[str, int, str]]]
    custom_ccd: bool


def chemistry_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Describe original components/sites without expanding symmetric copies."""
    modifications = []
    ligands = []
    for entry in document["sequences"]:
        kind, chain = next(iter(entry.items()))
        ids = chain["id"]
        ids = [ids] if isinstance(ids, str) else ids
        if kind == "ligand":
            ligands.append({"ids": ids, "ccd_codes": chain.get("ccdCodes", [])})
        for modification in chain.get("modifications", []):
            modifications.append({
                "ids": ids,
                "entity_type": kind,
                "position": modification.get(
                    "ptmPosition", modification.get("basePosition")
                ),
                "ccd_code": modification.get(
                    "ptmType", modification.get("modificationType")
                ),
            })
    return {
        "modifications": modifications,
        "ligands": ligands,
        "bonds": document.get("bondedAtomPairs", []) or [],
        "custom_ccd": bool(document.get("userCCD")),
    }


def require_inline_inputs(value: object) -> None:
    """Reject server-local paths before either the schema or native parser."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower().endswith("path"):
                raise ValueError(
                    f"Path-valued field is not supported by the API: {key}"
                )
            if key == "userCCD" and item is not None:
                if not isinstance(item, str) or not item.lstrip().startswith("data_"):
                    raise ValueError("userCCD must contain inline CCD mmCIF data")
            require_inline_inputs(item)
    elif isinstance(value, list):
        for item in value:
            require_inline_inputs(item)


def validate_native_chemistry(content: bytes) -> ChemistryReceipt:
    """Validate in a fresh process only: native Ccd mutates a cached dictionary."""
    from alphafold3.common import folding_input  # noqa: PLC0415
    from alphafold3.constants import chemical_components  # noqa: PLC0415

    if not 0 < len(content) <= MAX_INPUT_JSON_BYTES:
        raise ValueError("AlphaFold3 document has an invalid size")
    document = orjson.loads(content)
    require_inline_inputs(document)
    if document.get("userCCD"):
        # Native from_json treats short strings as candidate filesystem paths.
        # Padding inline CIF whitespace disables that branch without changing CCD.
        document["userCCD"] += "\n" + " " * 256
    native = folding_input.Input.from_json(orjson.dumps(document).decode())
    for entry in document["sequences"]:
        chain = next(iter(entry.values()))
        positions = [
            item.get("ptmPosition", item.get("basePosition"))
            for item in chain.get("modifications", [])
        ]
        if len(set(positions)) != len(positions):
            raise ValueError("Only one modification is allowed at each chain position")
    ccd = chemical_components.Ccd(user_ccd=native.user_ccd)
    chains = {chain.id: chain for chain in native.chains}
    for chain in native.chains:
        components = (
            chain.ccd_ids or ()
            if isinstance(chain, folding_input.Ligand)
            else chain.to_ccd_sequence()
        )
        for component in components:
            if component not in ccd:
                raise ValueError(f"Chain {chain.id}: unknown CCD component {component}")
    seen = set()
    for left, right in native.bonded_atom_pairs or ():
        if left == right:
            raise ValueError("A covalent bond cannot join an atom to itself")
        bond = frozenset((left, right))
        if bond in seen:
            raise ValueError("Duplicate covalent bond (including reversed endpoints)")
        seen.add(bond)
        if not any(
            isinstance(chains[endpoint[0]], folding_input.Ligand)
            for endpoint in (left, right)
        ):
            raise ValueError(
                "Polymer-polymer bonds, including disulfides, are not supported "
                "by the pinned AlphaFold3 model"
            )
    native.to_structure(ccd)
    # This is the same build_data dictionary used by prediction, not an API copy.
    _, ccd_digest = file_size_sha256(Path(chemical_components._CCD_PICKLE_FILE))
    return ChemistryReceipt(
        input_sha256=hashlib.sha256(content).hexdigest(),
        ccd_sha256=ccd_digest,
        upstream_commit=ALPHAFOLD3_COMMIT,
    )


def check_chemistry_isolated(content: bytes) -> dict[str, Any]:
    """Bound a native check and return small structured evidence or input error."""
    if not 0 < len(content) <= MAX_INPUT_JSON_BYTES:
        raise ValueError("AlphaFold3 document has an invalid size")
    result = subprocess.run(  # noqa: S603 -- fixed module, untrusted bytes on stdin only
        [sys.executable, "-m", __name__],
        input=content,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=CHEMISTRY_TIMEOUT_SECONDS,
        check=True,
    )
    if len(result.stdout) > 8192:
        raise ValueError("Native chemistry response exceeds its limit")
    return orjson.loads(result.stdout)


def _main() -> None:
    content = sys.stdin.buffer.read(MAX_INPUT_JSON_BYTES + 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            receipt = validate_native_chemistry(content)
        result = {"receipt": receipt.model_dump(mode="json")}
    except (ValueError, KeyError, TypeError) as error:
        result = {"error": str(error)[:1000]}
    sys.stdout.buffer.write(orjson.dumps(result))


if __name__ == "__main__":
    _main()
