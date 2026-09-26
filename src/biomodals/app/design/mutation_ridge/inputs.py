"""Measurement parsing shared by standalone ridge and protein optimization.

Positions refer to one-based, untrimmed parental sequences. Display IDs never
participate in variant identity or become paths. Invalid rows remain addressable;
only an error-free review may become a scientific dataset.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations, product

import polars as pl

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_INPUT_BYTES = 10 * 1024 * 1024
MAX_MEASUREMENT_ROWS = 10_000
MAX_MUTATION_TOKENS = 100_000
_TOKEN = re.compile(
    r"([^\s:,]+):([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])"
)


@dataclass(frozen=True, order=True)
class Substitution:
    """One checked replacement, sorted by chain and numeric position."""

    chain: str
    position: int
    original: str
    replacement: str

    @property
    def token(self) -> str:
        """Return the canonical input representation."""
        return f"{self.chain}:{self.original}{self.position}{self.replacement}"

    @property
    def site(self) -> tuple[str, int]:
        """Return the mutually exclusive replacement site."""
        return self.chain, self.position


Variant = tuple[Substitution, ...]


@dataclass(frozen=True)
class InputIssue:
    """A safe diagnostic; row indices exclude the CSV header and start at zero."""

    row_index: int | None
    field: str
    code: str
    message: str


@dataclass(frozen=True)
class MeasurementReview:
    """Parsed observations and all errors, without silently discarding rows."""

    observations: pl.DataFrame
    variants: tuple[Variant | None, ...]
    required_chains: tuple[str, ...]
    issues: tuple[InputIssue, ...]


@dataclass(frozen=True)
class MutationDataset:
    """Validated parents, canonical mean labels, and replicate evidence."""

    parents: dict[str, str]
    measurements: pl.DataFrame
    observations: pl.DataFrame
    variants: tuple[Variant, ...]

    @property
    def vocabulary(self) -> tuple[Substitution, ...]:
        """Return only substitutions represented in measured variants."""
        return tuple(sorted({mutation for row in self.variants for mutation in row}))


def _chain_id(value: str) -> str:
    if (
        not value
        or len(value) > 100
        or any(not 33 <= ord(c) <= 126 or c in ":,>" for c in value)
    ):
        raise ValueError(
            "Chain IDs must be printable ASCII tokens without spaces, ':', ',' or '>'"
        )
    return value


def parse_mutations(value: str) -> Variant:
    """Parse a substitution list; an empty cell is the measured parent."""
    if not value.strip():
        return ()
    result = []
    seen = set()
    for token in value.split(","):
        match = _TOKEN.fullmatch(token.strip())
        if match is None:
            raise ValueError(
                "Use chain:OriginalPositionReplacement, for example A:Y52F"
            )
        chain, original, position, replacement = match.groups()
        mutation = Substitution(_chain_id(chain), int(position), original, replacement)
        if original == replacement:
            raise ValueError("A substitution must change the original amino acid")
        if mutation.site in seen:
            raise ValueError("Specify at most one substitution per chain position")
        seen.add(mutation.site)
        result.append(mutation)
    return tuple(sorted(result))


def variant_key(variant: Variant) -> str:
    """Serialize a canonical variant, independently of uploaded display IDs."""
    return ",".join(mutation.token for mutation in variant)


def review_measurements(
    content: bytes, *, max_rows: int = MAX_MEASUREMENT_ROWS
) -> MeasurementReview:
    """Read bounded CSV with Polars and retain diagnostics for every invalid row."""
    if not content or len(content) > MAX_INPUT_BYTES:
        raise ValueError("Measurement CSV must be nonempty and at most 10 MiB")
    try:
        frame = pl.read_csv(
            io.BytesIO(content),
            infer_schema=False,
            empty_string_is_null=False,
            n_rows=max_rows + 1,
            raise_if_empty=True,
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError("Measurement CSV is not a valid UTF-8 table") from exc
    if not {"mutations", "label"} <= set(frame.columns) or set(frame.columns) - {
        "id",
        "mutations",
        "label",
    }:
        raise ValueError("Use required mutations,label columns and optional id only")
    if not 1 <= frame.height <= max_rows:
        raise ValueError(f"Provide between 1 and {max_rows} measurement rows")
    if "id" not in frame.columns:
        frame = frame.with_columns(pl.lit("").alias("id"))
    token_count = frame.select(
        pl
        .when(pl.col("mutations").str.strip_chars().str.len_chars() > 0)
        .then(pl.col("mutations").str.count_matches(",") + 1)
        .otherwise(0)
        .sum()
    ).item()
    if token_count > MAX_MUTATION_TOKENS:
        raise ValueError(
            f"Measurement table exceeds {MAX_MUTATION_TOKENS} mutation tokens"
        )
    frame = frame.with_row_index("row_index").with_columns(
        pl.col("label").str.strip_chars().cast(pl.Float64, strict=False).alias("value")
    )
    issues = [
        InputIssue(
            index,
            "label",
            "invalid_label",
            "Provide one finite numeric measurement; censored values are not exact labels",
        )
        for index in frame.filter(~pl.col("value").is_finite().fill_null(False))[
            "row_index"
        ]
    ]
    variants: list[Variant | None] = []
    chains: set[str] = set()
    original_claims: dict[tuple[str, int], dict[str, list[int]]] = {}
    for index, text in enumerate(frame["mutations"]):
        try:
            variant = parse_mutations(text)
        except ValueError as exc:
            issues.append(InputIssue(index, "mutations", "invalid_mutations", str(exc)))
            variants.append(None)
            continue
        variants.append(variant)
        for mutation in variant:
            chains.add(mutation.chain)
            original_claims.setdefault(mutation.site, {}).setdefault(
                mutation.original, []
            ).append(index)
    for claims in original_claims.values():
        if len(claims) > 1:
            for indices in claims.values():
                issues.extend(
                    InputIssue(
                        index,
                        "mutations",
                        "conflicting_original",
                        "Rows disagree on the original residue at the same chain position",
                    )
                    for index in indices
                )
    frame = frame.with_columns(
        pl.Series(
            "canonical_mutations",
            [None if v is None else variant_key(v) for v in variants],
            dtype=pl.String,
        )
    )
    return MeasurementReview(
        frame, tuple(variants), tuple(sorted(chains)), tuple(issues)
    )


def parse_parents(content: str) -> dict[str, str]:
    """Read ordinary chain-ID FASTA, normalizing presentation but not residues."""
    if len(content.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Parental FASTA must be at most 10 MiB")
    records: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].split()
            current = _chain_id(header[0] if header else "")
            if current in records:
                raise ValueError("Parental FASTA chain IDs must be unique")
            records[current] = []
        elif current is None:
            raise ValueError("Parental sequences require FASTA chain-ID headers")
        else:
            records[current].append("".join(line.split()).upper())
    parents = {chain: "".join(parts) for chain, parts in sorted(records.items())}
    if not parents or any(
        not seq or set(seq) - set(AMINO_ACIDS) for seq in parents.values()
    ):
        raise ValueError(
            "Each parental chain must contain only the standard 20 amino acids"
        )
    return parents


def parent_issues(
    review: MeasurementReview, parents: dict[str, str]
) -> tuple[InputIssue, ...]:
    """Associate missing chains and original-residue mismatches with their rows."""
    issues = []
    for index, variant in enumerate(review.variants):
        if variant is None:
            continue
        for mutation in variant:
            sequence = parents.get(mutation.chain)
            if sequence is None:
                code, message = (
                    "missing_chain",
                    f"Provide parental chain {mutation.chain}",
                )
            elif mutation.position > len(sequence):
                code, message = (
                    "position_out_of_range",
                    "Mutation position exceeds parental chain length",
                )
            elif sequence[mutation.position - 1] != mutation.original:
                code, message = (
                    "original_mismatch",
                    "Mutation original residue does not match the parental sequence",
                )
            else:
                continue
            issues.append(InputIssue(index, "mutations", code, message))
    return tuple(issues)


def build_dataset(content: bytes, parental_fasta: str) -> MutationDataset:
    """Validate all rows and aggregate identical variants using the supplied scale."""
    review = review_measurements(content)
    parents = parse_parents(parental_fasta)
    return validated_dataset(review, parents)


def validated_dataset(
    review: MeasurementReview, parents: dict[str, str]
) -> MutationDataset:
    """Reuse a local review without reparsing the user's table or retaining a cache."""
    issues = review.issues + parent_issues(review, parents)
    if issues:
        first = issues[0]
        raise ValueError(f"Measurement row {first.row_index}: {first.message}")
    measurements = (
        review.observations
        .group_by("canonical_mutations")
        .agg(
            pl.col("value").mean().alias("label"),
            pl.len().alias("replicate_count"),
            pl.col("value").std(ddof=1).alias("replicate_std"),
        )
        .rename({"canonical_mutations": "mutations"})
        .sort("mutations")
    )
    # Finite individual observations can still overflow an aggregate.
    if not measurements["label"].is_finite().all():
        raise ValueError("Replicate aggregation produced a nonfinite label")
    return MutationDataset(
        parents,
        measurements,
        review.observations,
        tuple(parse_mutations(value) for value in measurements["mutations"]),
    )


def combination_space(
    dataset: MutationDataset, max_mutations: int
) -> tuple[tuple[tuple[Substitution, ...], ...], tuple[int, ...]]:
    """Group incompatible site alternatives and count novel candidates per size."""
    if max_mutations < 1:
        raise ValueError("Maximum mutations must be positive")
    sites: dict[tuple[str, int], list[Substitution]] = {}
    for mutation in dataset.vocabulary:
        sites.setdefault(mutation.site, []).append(mutation)
    groups = tuple(tuple(alternatives) for alternatives in sites.values())
    counts = [1] + [0] * min(max_mutations, len(groups))
    for alternatives in groups:
        for size in range(len(counts) - 1, 0, -1):
            counts[size] += counts[size - 1] * len(alternatives)
    counts[0] = 0
    for variant in dataset.variants:
        if 0 < len(variant) < len(counts):
            counts[len(variant)] -= 1
    return groups, tuple(counts)


def iter_combinations(
    dataset: MutationDataset, *, max_mutations: int, budget: int
) -> Iterator[Variant]:
    """Enumerate every compatible novel variant, rejecting oversize before yield."""
    groups, counts = combination_space(dataset, max_mutations)
    if budget < 1 or sum(counts) > budget:
        raise ValueError(
            f"Novel combination count {sum(counts)} exceeds candidate budget {budget}"
        )
    measured = set(dataset.variants)
    for size in range(1, len(counts)):
        for sites in combinations(groups, size):
            for variant in product(*sites):
                if variant not in measured:
                    yield variant


def variant_sequences(parents: dict[str, str], variant: Variant) -> dict[str, str]:
    """Reconstruct complete chains from a validated parent-relative variant."""
    chains = {chain: list(sequence) for chain, sequence in parents.items()}
    for mutation in variant:
        if chains[mutation.chain][mutation.position - 1] != mutation.original:
            raise ValueError("Variant does not match its parent or repeats a position")
        chains[mutation.chain][mutation.position - 1] = mutation.replacement
    return {chain: "".join(sequence) for chain, sequence in chains.items()}
