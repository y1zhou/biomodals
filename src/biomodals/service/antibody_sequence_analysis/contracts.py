"""Typed, ephemeral analysis requests and scientific response metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from biomodals.helper.antibody import (
    ANALYSIS_VERSION,
    ARPEGGIA_VERSION,
    MAX_CHAIN_LENGTH,
    SCHEMES,
    GermlineAssignment,
    ProteinMetrics,
    Scheme,
)

MAX_ENTRIES = 1000
MAX_REQUEST_BYTES = 4 * 1024 * 1024


class AnalysisOptions(BaseModel):
    """Authoritative admission limits and supported conventions."""

    max_groups: int = 2
    max_entries_per_group: int = MAX_ENTRIES
    max_chain_length: int = MAX_CHAIN_LENGTH
    max_request_bytes: int = MAX_REQUEST_BYTES
    schemes: tuple[Scheme, ...] = SCHEMES
    default_scheme: Scheme = "imgt"
    analysis_version: str = ANALYSIS_VERSION
    arpeggia_version: str = ARPEGGIA_VERSION


class FastaGroup(BaseModel):
    """One independent input group; identifiers are scoped to this group."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100)
    fasta: str = Field(max_length=MAX_REQUEST_BYTES)


class AnalysisRequest(BaseModel):
    """Analyze one or two bounded FASTA groups without creating a Job."""

    model_config = ConfigDict(extra="forbid")
    groups: list[FastaGroup] = Field(min_length=1, max_length=2)


class SequenceRequest(BaseModel):
    """Number only the single chain currently opened in a sequence dialog."""

    model_config = ConfigDict(extra="forbid")
    sequence: str = Field(min_length=1, max_length=MAX_REQUEST_BYTES)
    scheme: Scheme = "imgt"


class AnalysisIssue(BaseModel):
    """Safe record/group validation diagnostic, without discarding other rows."""

    code: str
    detail: str


class GeneUsage(BaseModel):
    """Weighted fraction of unique approved chains assigned to this gene."""

    species: str
    gene: str
    frequency: float | None


class GermlinePresentation(BaseModel):
    """Frozen assignment plus current local reference frequencies."""

    assignment: GermlineAssignment
    v_usage: list[GeneUsage]
    j_usage: list[GeneUsage]


class AnalyzedChain(BaseModel):
    """Untrimmed sequence and independent chain-level scientific results."""

    sequence: str
    metrics: ProteinMetrics
    germlines: GermlinePresentation
    germline_pi: float | None


class AnalysisEntry(BaseModel):
    """One pair or standalone domain, with no inherited ranking information."""

    id: str
    vh: AnalyzedChain | None = None
    vl: AnalyzedChain | None = None
    unassigned: AnalyzedChain | None = None
    vh_vl_pi: float | None = None
    issues: list[AnalysisIssue] = Field(default_factory=list)


class AnalysisGroup(BaseModel):
    """Independent valid/error entries in original FASTA order."""

    id: str
    entries: list[AnalysisEntry]
    issues: list[AnalysisIssue] = Field(default_factory=list)


class ReferenceInfo(BaseModel):
    """Shared FAQ provenance, never repeated per gene cell."""

    status: Literal["available", "unavailable"]
    source_url: str
    downloaded_at: str | None = None
    source_sha256: str | None = None
    engine_version: str | None = None
    germline_reference: str | None = None
    policy_version: str | None = None
    heavy_sequences: int | None = None
    light_sequences: int | None = None
    detail: str | None = None


class AnalysisResponse(BaseModel):
    """Ephemeral metrics and one common therapeutic reference provenance block."""

    groups: list[AnalysisGroup]
    reference: ReferenceInfo
