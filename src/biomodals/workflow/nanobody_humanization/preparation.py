"""Native terminal preparation and immutable parent policy, without providers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.helper.antibody import (
    ARPEGGIA_VERSION,
    GERMLINE_REFERENCE,
    MAX_CHAIN_LENGTH,
    normalize_sequence,
)
from biomodals.helper.io import fasta_identifier

PREPARATION_VERSION = f"1|arpeggia={ARPEGGIA_VERSION}|{GERMLINE_REFERENCE}"
# Exact supported labels from HuDiff bb7636f, dataset/preprocess.py.
HUDIFF_POSITIONS = (
    *(str(n) for n in range(1, 112)),
    *(f"111{letter}" for letter in "ABCDEFGHIJKL"),
    *(f"112{letter}" for letter in "LKJIHGFEDCBA"),
    *(str(n) for n in range(112, 129)),
)


class VHInput(BaseModel):
    """An editable original construct; scientific errors remain row-addressable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(max_length=200)
    vhh: str = Field(max_length=MAX_CHAIN_LENGTH)

    @field_validator("vhh")
    @classmethod
    def normalize(cls, value: str) -> str:
        """Normalize presentation only, leaving invalid residues for row errors."""
        return "".join(value.split()).upper()


class PreparedVH(BaseModel):
    """Saved scientific parent with original correspondence, never re-imputed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    original_sequence: str
    sequence: str
    original_indices: tuple[int | None, ...]
    imputation_references: tuple[tuple[str, ...], ...]
    v_reference: str
    j_reference: str
    imgt_positions: tuple[str, ...]
    aho_positions: tuple[int, ...]
    protected_indices: tuple[int, ...]
    preparation_version: str = PREPARATION_VERSION


class PreparationIssue(BaseModel):
    """Safe row/field explanation, without echoing private sequence content."""

    row_index: int
    field: str
    code: str
    message: str


def prepare_vh(record: VHInput) -> PreparedVH:
    """Trim one detected VH and use native representative terminal imputation."""
    from arpeggia import number_antibody

    original = normalize_sequence(record.vhh)
    numbered = number_antibody(original, scheme="imgt")
    if numbered.chain != "H":
        raise ValueError("Provide one heavy-chain variable domain")
    if numbered.v_match is None or numbered.j_match is None:
        raise ValueError("V and J reference coverage is insufficient for preparation")
    v_reference = numbered.v_match.hits[0].references[0].id
    j_reference = numbered.j_match.hits[0].references[0].id
    prepared = numbered.impute(v_reference=v_reference, j_reference=j_reference)
    sequence = prepared.sequence
    # Native imputation retains diagnostics of the original fragment. Re-number
    # the actual prepared domain instead of treating those as its new status.
    imgt = number_antibody(sequence, scheme="imgt", match_germlines=False)
    aho = number_antibody(sequence, scheme="aho", match_germlines=False)
    if any("PARTIAL_DOMAIN" in diagnostic for diagnostic in imgt.diagnostics):
        raise ValueError("Closest references cannot complete the terminal frameworks")
    if imgt.sequence != sequence or aho.sequence != sequence:
        raise ValueError("Prepared domain is not fully covered by model numbering")
    imgt_positions = tuple(
        f"{r.position.number}{r.position.insertion or ''}" for r in imgt.residues
    )
    if not set(imgt_positions) <= set(HUDIFF_POSITIONS):
        raise ValueError("Prepared domain has positions outside HuDiff-Nb's IMGT grid")
    if any(
        r.position.insertion or not 1 <= r.position.number <= 149 for r in aho.residues
    ):
        raise ValueError("Prepared domain has positions outside AbNatiV2's AHo grid")
    protected = {index for index, residue in enumerate(sequence) if residue == "C"}
    for numbered, regions in (
        (imgt, ((27, 38), (55, 66), (105, 117))),
        (aho, ((27, 42), (57, 69), (108, 138))),
    ):
        protected.update(
            r.input_index
            for r in numbered.residues
            if r.input_index is not None
            and any(start <= r.position.number <= end for start, end in regions)
        )
    protected.update(
        r.input_index
        for r in imgt.residues
        if r.input_index is not None and r.position.number in (42, 49, 50, 52)
    )
    return PreparedVH(
        id=record.id,
        original_sequence=original,
        sequence=sequence,
        original_indices=tuple(r.input_index for r in prepared.residues),
        imputation_references=tuple(tuple(r.imputed_from) for r in prepared.residues),
        v_reference=v_reference,
        j_reference=j_reference,
        imgt_positions=imgt_positions,
        aho_positions=tuple(r.position.number for r in aho.residues),
        protected_indices=tuple(sorted(protected)),
    )


def prepare_batch(
    records: Sequence[VHInput],
) -> tuple[tuple[PreparedVH, ...], list[PreparationIssue]]:
    """Prepare unique sequences once but preserve each original parent identity."""
    issues = []
    seen: dict[str, int] = {}
    duplicate_indices: set[int] = set()
    prepared = []
    sequences: dict[str, PreparedVH | str] = {}
    for index, record in enumerate(records):
        if (
            not record.id
            or record.id != record.id.strip()
            or any(ord(c) < 32 or c == ">" for c in record.id)
        ):
            issues.append(
                PreparationIssue(
                    row_index=index,
                    field="id",
                    code="id_invalid",
                    message="ID must be nonempty, unpadded, and contain no controls or >",
                )
            )
        key = fasta_identifier(record.id)
        if key in seen:
            duplicate_indices.update((index, seen[key]))
        else:
            seen[key] = index
        if record.vhh not in sequences:
            try:
                sequences[record.vhh] = prepare_vh(record)
            except (ValueError, RuntimeError) as error:
                sequences[record.vhh] = str(error)
        value = sequences[record.vhh]
        if isinstance(value, str):
            issues.append(
                PreparationIssue(
                    row_index=index, field="vhh", code="domain_invalid", message=value
                )
            )
        else:
            prepared.append(value.model_copy(update={"id": record.id}))
    issues.extend(
        PreparationIssue(
            row_index=index,
            field="id",
            code="id_duplicate",
            message="IDs must be unique after FASTA whitespace normalization",
        )
        for index in sorted(duplicate_indices)
    )
    return tuple(prepared), issues


def preparation_digest(parents: Sequence[PreparedVH]) -> str:
    """Bind exact reviewed inputs, order, references and preparation policy."""
    return hashlib.sha256(
        ("[" + ",".join(parent.model_dump_json() for parent in parents) + "]").encode()
    ).hexdigest()
