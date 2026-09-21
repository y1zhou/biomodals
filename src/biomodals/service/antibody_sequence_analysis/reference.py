"""Build-once, public approved-therapeutic gene usage reference cache."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import cached_property
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Literal

import niquests
import polars as pl
from pydantic import BaseModel

from biomodals.helper.antibody import (
    ARPEGGIA_VERSION,
    GERMLINE_REFERENCE,
    GermlineAssignment,
    assign_germlines,
    normalize_sequence,
)
from biomodals.service.antibody_sequence_analysis.contracts import (
    GeneUsage,
    GermlinePresentation,
    ReferenceInfo,
)

SOURCE_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv"
POLICY_VERSION = "approved-exact-unique-chain-tied-species-gene/1"
MAX_REFERENCE_BYTES = 16 * 1024 * 1024


class UsageRecord(BaseModel):
    """One species-qualified gene's weighted unique-chain fraction."""

    role: str
    segment: str
    species: str
    gene: str
    frequency: float


class ReferenceSnapshot(BaseModel):
    """Atomic public reference publication; never contains user inputs."""

    info: ReferenceInfo
    usage: list[UsageRecord]

    @cached_property
    def lookup(self) -> dict[tuple[str, str, str, str], float]:
        """Index this immutable reference once, not for every candidate chain."""
        return {
            (row.role, row.segment, row.species, row.gene): row.frequency
            for row in self.usage
        }


def download_reference() -> bytes:
    """Fetch the fixed public CSV URL with bounded bytes and network timeouts."""
    with niquests.get(SOURCE_URL, timeout=(5, 20), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_REFERENCE_BYTES:
                raise ValueError("Therapeutic reference exceeds the byte limit")
        return bytes(content)


def build_reference(content: bytes) -> ReferenceSnapshot:
    """Count exact unique chains, splitting credit over species/gene ties."""
    frame = pl.read_csv(BytesIO(content), infer_schema_length=0)
    approved = frame.filter(pl.col("Highest_Clin_Trial (Feb '25)") == "Approved")
    chains = (
        pl
        .concat([
            approved.select(
                pl.lit(role).alias("role"), pl.col(column).alias("sequence")
            )
            for role, columns in (
                ("vh", ("HeavySequence", "HeavySequence(ifbispec)")),
                ("vl", ("LightSequence", "LightSequence(ifbispec)")),
            )
            for column in columns
        ])
        .with_columns(pl.col("sequence").str.replace_all(r"\s+", "").str.to_uppercase())
        .filter(pl.col("sequence").str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$"))
        .unique()
        .sort("role", "sequence")
    )
    if chains.is_empty():
        raise ValueError("Therapeutic reference contains no approved canonical chains")

    # Exactly supplied sequences define uniqueness, including unnumbered tails.
    def annotate(sequence: str) -> GermlineAssignment | None:
        try:
            normalize_sequence(sequence)
        except ValueError:
            return None
        return assign_germlines(sequence)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assignments = list(pool.map(annotate, chains["sequence"]))
    counts = dict(chains.group_by("role").len().iter_rows())
    credits = []
    for role, assignment in zip(chains["role"], assignments, strict=True):
        if assignment is None:
            continue
        if assignment["chain_type"] not in (("H",) if role == "vh" else ("K", "L")):
            continue
        for segment in ("v", "j"):
            genes = sorted({
                (hit["species"], hit["gene"]) for hit in assignment[segment]
            })
            for species, gene in genes:
                credits.append((
                    role,
                    segment,
                    species,
                    gene,
                    1 / len(genes) / counts[role],
                ))
    usage = (
        pl
        .DataFrame(
            credits,
            schema={
                "role": pl.String,
                "segment": pl.String,
                "species": pl.String,
                "gene": pl.String,
                "frequency": pl.Float64,
            },
            orient="row",
        )
        .group_by("role", "segment", "species", "gene")
        .agg(pl.col("frequency").sum())
        .sort("role", "segment", "species", "gene")
    )
    return ReferenceSnapshot(
        info=ReferenceInfo(
            status="available",
            source_url=SOURCE_URL,
            downloaded_at=datetime.now(UTC).isoformat(),
            source_sha256=hashlib.sha256(content).hexdigest(),
            engine_version=ARPEGGIA_VERSION,
            germline_reference=GERMLINE_REFERENCE,
            policy_version=POLICY_VERSION,
            heavy_sequences=counts.get("vh", 0),
            light_sequences=counts.get("vl", 0),
        ),
        usage=[UsageRecord.model_validate(row) for row in usage.iter_rows(named=True)],
    )


class TherapeuticReference:
    """Share one local build/load; never refresh or replace an existing file."""

    def __init__(
        self, path: Path, *, download: Callable[[], bytes] = download_reference
    ) -> None:
        """Configure the public cache without network or filesystem writes."""
        self.path = path
        self.download = download
        self._lock = Lock()
        self._snapshot: ReferenceSnapshot | None = None

    def get(self) -> ReferenceSnapshot:
        """Build only when absent; leave invalid existing data for operator repair."""
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            if self.path.exists():
                snapshot = ReferenceSnapshot.model_validate_json(self.path.read_bytes())
                if (
                    snapshot.info.status != "available"
                    or snapshot.info.engine_version != ARPEGGIA_VERSION
                    or snapshot.info.germline_reference != GERMLINE_REFERENCE
                    or snapshot.info.policy_version != POLICY_VERSION
                ):
                    raise ValueError(
                        "Existing therapeutic cache uses incompatible reference policy"
                    )
            else:
                snapshot = build_reference(self.download())
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    dir=self.path.parent, suffix=".tmp", delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    try:
                        temporary.write(snapshot.model_dump_json().encode())
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        temporary_path.replace(self.path)
                    finally:
                        temporary_path.unlink(missing_ok=True)
            self._snapshot = snapshot
            return snapshot


def present_germlines(
    assignment: GermlineAssignment,
    role: str,
    snapshot: ReferenceSnapshot | None,
) -> GermlinePresentation:
    """Join gene-level frequencies without modifying frozen native evidence."""
    lookup = snapshot.lookup if snapshot else {}

    def usages(segment: Literal["v", "j"]) -> list[GeneUsage]:
        return [
            GeneUsage(
                species=species,
                gene=gene,
                frequency=lookup.get((role, segment, species, gene), 0.0)
                if snapshot
                else None,
            )
            for species, gene in sorted({
                (hit["species"], hit["gene"]) for hit in assignment[segment]
            })
        ]

    return GermlinePresentation(
        assignment=assignment, v_usage=usages("v"), j_usage=usages("j")
    )
