"""Provider-neutral antibody sequence metrics and annotations.

Optional scientific imports stay local: API and workflow annotation images own
their dependencies. This module never downloads references or retains inputs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from arpeggia import GermlineMatch, NumberedAntibody

ANALYSIS_VERSION = "3"
ARPEGGIA_VERSION = "0.10.1"
GERMLINE_REFERENCE = "IMGT-202636-7+llama-supplement"
MAX_CHAIN_LENGTH = 512
SCHEMES = ("imgt", "kabat", "chothia", "martin", "aho")
Scheme = Literal["imgt", "kabat", "chothia", "martin", "aho"]
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ProteinMetrics(TypedDict):
    """Full supplied-chain metrics, with Black–Mould mean hydrophobicity."""

    pi: float
    molecular_weight_kda: float
    gravy: float


class GermlineEvidence(TypedDict):
    """One tied reference with its native similarity/coverage evidence."""

    reference_id: str
    species: str
    gene: str
    allele: str
    accession: str
    score: float
    known_pairs: int
    known_matches: int
    known_fr4_pairs: int
    reference_coverage: float
    query_coverage: float
    query_input_start: int
    imgt_span: tuple[int, int]


class GermlineAssignment(TypedDict):
    """All best V/J reference ties, never an arbitrary representative."""

    chain_type: str | None
    v_gene: str | None
    j_gene: str | None
    v: list[GermlineEvidence]
    j: list[GermlineEvidence]
    diagnostics: list[str]
    error: str | None


class ChainAnalysis(TypedDict):
    """Normalized sequence, independent metrics and germline assignment."""

    sequence: str
    metrics: ProteinMetrics
    germlines: GermlineAssignment
    germline_pi: float | None


class PairAnalysis(TypedDict):
    """Ordinary Python result for an explicitly ordered VH/VL pair."""

    vh: ChainAnalysis
    vl: ChainAnalysis
    vh_vl_pi: float | None
    errors: list[str]


class NumberedResidue(TypedDict):
    """One supplied residue in native order; no imputed positions."""

    input_index: int
    label: str
    region: str


class Liability(TypedDict):
    """Potential sequence motif, using zero-based half-open input intervals."""

    kind: str
    start: int
    end: int


class GermlineAlignment(TypedDict):
    """Native local alignment for one display-representative V/J sequence.

    Starts are zero-based ungapped coordinates; query_input_start already
    includes the local alignment offset within the full supplied input.
    """

    segment: Literal["v", "j"]
    reference_ids: list[str]
    reference_names: list[str]
    tied_reference_count: int
    reference_start: int
    query_input_start: int
    aligned_reference: str
    aligned_query: str
    operations: str


class SequenceDetail(TypedDict):
    """Lazy numbering/CDR and liability overlays on the untrimmed sequence."""

    sequence: str
    scheme: Scheme
    cdr_definition: str
    chain_type: str | None
    domain_span: tuple[int, int] | None
    residues: list[NumberedResidue]
    germline_alignments: list[GermlineAlignment]
    liabilities: list[Liability]
    diagnostics: list[str]
    error: str | None


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and uppercase; reject rather than trim bad residues."""
    normalized = "".join(sequence.split()).upper()
    if not normalized or len(normalized) > MAX_CHAIN_LENGTH:
        raise ValueError(f"Each chain must contain 1–{MAX_CHAIN_LENGTH} residues")
    if not set(normalized) <= AMINO_ACIDS:
        raise ValueError("Use only the 20 canonical amino-acid letters")
    return normalized


def protein_metrics(sequence: str) -> ProteinMetrics:
    """Compute average mass, pI and Black–Mould GRAVY on all input residues."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    protein = ProteinAnalysis(normalize_sequence(sequence))
    return {
        "pi": protein.isoelectric_point(),
        "molecular_weight_kda": protein.molecular_weight() / 1000,
        "gravy": protein.gravy(scale="BlackMould"),
    }


def combined_pi(vh: str, vl: str) -> float:
    """Treat literal VH+VL as one continuous chain, with no linker."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return ProteinAnalysis(
        normalize_sequence(vh) + normalize_sequence(vl)
    ).isoelectric_point()


def _evidence(match: GermlineMatch | None) -> list[GermlineEvidence]:
    if match is None:
        return []
    return [
        {
            "reference_id": ref.id,
            "species": ref.species,
            "gene": ref.gene,
            "allele": ref.allele,
            "accession": ref.accession,
            "score": match.score,
            "known_pairs": hit.known_pairs,
            "known_matches": hit.known_matches,
            "known_fr4_pairs": hit.known_fr4_pairs,
            "reference_coverage": hit.reference_coverage,
            "query_coverage": hit.query_coverage,
            "query_input_start": hit.query_input_start,
            "imgt_span": hit.imgt_span,
        }
        for hit in match.hits
        for ref in hit.references
    ]


def _match_germlines(
    sequence: str,
) -> tuple[GermlineAssignment, NumberedAntibody | None]:
    """Keep the native result for metrics without repeating germline matching."""
    from arpeggia import number_antibody

    sequence = normalize_sequence(sequence)
    result: GermlineAssignment = {
        "chain_type": None,
        "v_gene": None,
        "j_gene": None,
        "v": [],
        "j": [],
        "diagnostics": [],
        "error": None,
    }
    try:
        numbered = number_antibody(sequence, scheme="imgt")
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
        return result, None
    result["chain_type"] = numbered.chain
    result["diagnostics"] = numbered.diagnostics
    result["v"] = _evidence(numbered.v_match)
    result["j"] = _evidence(numbered.j_match)
    result["v_gene"] = "/".join(sorted({hit["gene"] for hit in result["v"]})) or None
    result["j_gene"] = "/".join(sorted({hit["gene"] for hit in result["j"]})) or None
    return result, numbered


def assign_germlines(sequence: str) -> GermlineAssignment:
    """Search all bundled species using fixed IMGT matching; preserve ties."""
    assignment, _ = _match_germlines(sequence)
    return assignment


def _germline_pi(numbered: NumberedAntibody | None) -> float | None:
    """Compute pI of full representative V+J segments, without junction or linker."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if numbered is None or numbered.v_match is None or numbered.j_match is None:
        return None
    sequence = (
        numbered.v_match.hits[0].alignment.reference
        + numbered.j_match.hits[0].alignment.reference
    )
    # Reference segments may contain unresolved residues. Do not remove them
    # or estimate their charge as if the full reference sequence were known.
    if not set(sequence) <= AMINO_ACIDS:
        return None
    return ProteinAnalysis(sequence).isoelectric_point()


def _germline_alignments(numbered: NumberedAntibody) -> list[GermlineAlignment]:
    """Expose native strings, without re-alignment or invented junction edits."""
    result: list[GermlineAlignment] = []
    for segment, match in (("v", numbered.v_match), ("j", numbered.j_match)):
        if match is None:
            continue
        hit = match.hits[0]
        alignment = hit.alignment
        result.append({
            "segment": segment,
            "reference_ids": [ref.id for ref in hit.references],
            "reference_names": [
                f"{ref.species} {ref.gene}*{ref.allele}" for ref in hit.references
            ],
            "tied_reference_count": sum(len(hit.references) for hit in match.hits),
            "reference_start": alignment.reference_span[0],
            "query_input_start": hit.query_input_start + alignment.query_span[0],
            "aligned_reference": alignment.aligned_reference,
            "aligned_query": alignment.aligned_query,
            "operations": alignment.operations,
        })
    return result


def analyze_chain(sequence: str) -> ChainAnalysis:
    """Analyze one variable-domain sequence without I/O or a Job runtime."""
    sequence = normalize_sequence(sequence)
    assignment, numbered = _match_germlines(sequence)
    return {
        "sequence": sequence,
        "metrics": protein_metrics(sequence),
        "germlines": assignment,
        "germline_pi": _germline_pi(numbered),
    }


def analyze_pair(vh: str, vl: str) -> PairAnalysis:
    """Analyze an explicit pair without swapping chains or creating a Job.

    Batch consumers may call the lower-level chain/pI functions to share work
    across repeated sequences within their request.
    """
    heavy, light = analyze_chain(vh), analyze_chain(vl)
    errors = [
        f"{role}: detected chain role {analysis['germlines']['chain_type']}"
        for role, analysis, allowed in (
            ("vh", heavy, (None, "H")),
            ("vl", light, (None, "K", "L")),
        )
        if analysis["germlines"]["chain_type"] not in allowed
    ]
    return {
        "vh": heavy,
        "vl": light,
        "vh_vl_pi": None
        if errors
        else combined_pi(heavy["sequence"], light["sequence"]),
        "errors": errors,
    }


def sequence_liabilities(sequence: str) -> list[Liability]:
    """Find potential motifs, including overlaps and unnumbered tails.

    Selected rules adapted from LAMBS v0.12.0 (Daniel Croote et al., Apache-2.0):
    https://github.com/dcroote/lambs/blob/v0.12.0/index.html
    Odd counts include every C, but markers exclude the conserved framework
    cysteines at IMGT 23/104. Use native input correspondence, independently of
    the display scheme. If numbering fails, no conserved positions are inferred.
    This is sequence screening, not evidence of modification or developability.
    """
    sequence = normalize_sequence(sequence)
    motifs = (
        ("methionine", "M"),
        ("n_glycosylation", "N[^P][ST]"),
        ("asn_deamidation", "N[GSTNH]"),
        ("asp_isomerization", "D[GS]"),
        ("acid_cleavage", "DP"),
    )
    result: list[Liability] = [
        {"kind": kind, "start": match.start(), "end": match.start() + len(match[1])}
        for kind, motif in motifs
        for match in re.finditer(f"(?=({motif}))", sequence)
    ]
    if sequence.count("C") % 2:
        from arpeggia import number_antibody

        conserved = set()
        try:
            numbered = number_antibody(sequence, scheme="imgt", match_germlines=False)
        except (ValueError, RuntimeError):
            pass  # Unnumberable input still receives full-sequence motif screening.
        else:
            conserved = {
                residue.input_index
                for residue in numbered.residues
                if residue.position.number in (23, 104)
                and not residue.position.insertion
            }
        result.extend(
            {"kind": "odd_cysteine_count", "start": i, "end": i + 1}
            for i, residue in enumerate(sequence)
            if residue == "C" and i not in conserved
        )
    return sorted(result, key=lambda row: (row["start"], row["end"], row["kind"]))


def sequence_detail(sequence: str, scheme: Scheme = "imgt") -> SequenceDetail:
    """Number on demand with matching CDR convention; never impute or trim."""
    from arpeggia import number_antibody

    sequence = normalize_sequence(sequence)
    if scheme not in SCHEMES:
        raise ValueError("Unsupported numbering scheme")
    result: SequenceDetail = {
        "sequence": sequence,
        "scheme": scheme,
        "cdr_definition": scheme,
        "chain_type": None,
        "domain_span": None,
        "residues": [],
        "germline_alignments": [],
        "liabilities": sequence_liabilities(sequence),
        "diagnostics": [],
        "error": None,
    }
    try:
        numbered = number_antibody(sequence, scheme=scheme)
    except (ValueError, RuntimeError) as error:
        result["error"] = str(error)
        return result
    result.update(
        chain_type=numbered.chain,
        domain_span=numbered.domain_span,
        cdr_definition=numbered.cdr_definition,
        diagnostics=numbered.diagnostics,
        germline_alignments=_germline_alignments(numbered),
        residues=[
            {
                "input_index": residue.input_index,
                "label": f"{residue.position.number}{residue.position.insertion or ''}",
                "region": residue.region,
            }
            for residue in numbered.residues
            if residue.input_index is not None
        ],
    )
    return result
