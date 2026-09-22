"""Bounded selection-table reads and deterministic scientific result archives."""

from __future__ import annotations

from pathlib import Path
from typing import IO, BinaryIO, Literal
from uuid import UUID

import polars as pl
from pydantic import BaseModel, Field, model_validator

from biomodals.service.antibody_sequence_analysis.contracts import (
    GermlinePresentation,
    ReferenceInfo,
)
from biomodals.service.selection import (
    NativenessRange,
    SelectionColumn,
    selection_page_values,
)
from biomodals.service.table_archive import (
    BuiltResultArchive,
    ManifestFile,
    build_table_archive,
    validate_manifest_paths,
)
from biomodals.workflow.humanization.germlines import GENE_COLUMNS, SEQUENCE_COLUMNS
from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import selection_table

SELECTION_SCHEMA = {
    **selection_table((), (), ()).schema,
    "quality_tier": pl.Int64,
    "panel_order": pl.Int64,
}
ANNOTATED_SELECTION_SCHEMA = {**SELECTION_SCHEMA, **GENE_COLUMNS}
CURRENT_SELECTION_SCHEMA = {**SELECTION_SCHEMA, **SEQUENCE_COLUMNS}
LEGACY_PI_SELECTION_SCHEMA = {
    name.removesuffix("_pI") + "_pi" if name.endswith("_pI") else name: dtype
    for name, dtype in CURRENT_SELECTION_SCHEMA.items()
}
READ_SELECTION_SCHEMA = {**LEGACY_PI_SELECTION_SCHEMA, **CURRENT_SELECTION_SCHEMA}


class CandidateGermlines(BaseModel):
    """Page-bounded heavy/light assignments and therapeutic usage."""

    vh: GermlinePresentation
    vl: GermlinePresentation


class SelectionPage(BaseModel):
    """One bounded page with the full table's available parent filters."""

    columns: list[SelectionColumn]
    rows: list[dict[str, str | int | float | bool | None]]
    total_rows: int
    offset: int
    limit: int
    parent_ids: list[str]
    default_hidden_columns: list[str] = Field(
        description="Data-dependent defaults computed over the full result before filtering or pagination."
    )
    nativeness_ranges: dict[str, NativenessRange] = Field(
        description="Full-result finite min/max for the three original p-AbNatiV2 nativeness columns, before filtering or pagination. Both zero when no finite values exist. Original scores and their deltas share this range; visualization only."
    )
    germlines: dict[str, CandidateGermlines] | None = Field(
        default=None,
        description="Page-only evidence keyed by candidate_id; null for historical publications.",
    )
    reference: ReferenceInfo | None = Field(
        default=None,
        description="One therapeutic reference provenance block for new germline-annotated publications.",
    )


def query_selection(
    path: Path | IO[bytes],
    *,
    offset: int = 0,
    limit: int = 50,
    parent_id: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> SelectionPage:
    """Keep parsing, filtering and sorting native; serialize only the page."""
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("offset must be nonnegative and limit between 1 and 200")
    if sort_by is not None and sort_by not in READ_SELECTION_SCHEMA:
        raise ValueError(f"Unknown selection column: {sort_by}")
    table = pl.read_csv(path, schema_overrides=READ_SELECTION_SCHEMA)
    if set(table.columns) not in (
        set(SELECTION_SCHEMA),
        set(ANNOTATED_SELECTION_SCHEMA),
        set(CURRENT_SELECTION_SCHEMA),
        set(LEGACY_PI_SELECTION_SCHEMA),
    ):
        raise ValueError("Selection table columns do not match the workflow schema")
    if sort_by is not None and sort_by not in table.columns:
        raise ValueError(
            f"Selection column not available in this publication: {sort_by}"
        )
    return SelectionPage(
        **selection_page_values(
            table,
            offset=offset,
            limit=limit,
            parent_id=parent_id,
            sort_by=sort_by,
            descending=descending,
        )
    )


class HumanizationManifest(BaseModel):
    """Publication identity and safe membership, shared by download and packaging."""

    schema_version: Literal[2, 3, 4, 5, 6]
    execution_run_id: UUID
    parameters: HumanizationSettings
    scientific_versions: dict[str, str]
    files: list[ManifestFile]

    @model_validator(mode="after")
    def validate_members(self) -> HumanizationManifest:
        """Reject unsafe or ambiguous file destinations before any downloads."""
        validate_manifest_paths(self.files)
        required = {"selection.csv", "imgt_mutations.parquet"}
        if self.schema_version >= 3:
            required.add("generation.parquet")
        if self.schema_version >= 4:
            required.add("germlines.parquet")
        if not required <= {record.path for record in self.files}:
            raise ValueError("Scientific publication is missing required tables")
        return self


def build_humanization_archive(
    root: Path,
    destination: BinaryIO,
) -> BuiltResultArchive:
    """Verify the workflow manifest while streaming a reproducible ZIP."""
    manifest = HumanizationManifest.model_validate_json(
        (root / "manifest.json").read_bytes()
    )
    return build_table_archive(root, destination, manifest.files)
