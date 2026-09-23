"""Single-domain result schema with shared bounded table and archive mechanics."""

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
from biomodals.workflow.nanobody_humanization.export import SELECTION_COLUMNS
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings

SELECTION_SCHEMA = {
    name: pl.Float64
    if name.endswith(("_pI", "_nativeness", "_delta"))
    else pl.Int64
    if name in {"panel_order", "quality_tier", "vh_mutations"}
    else pl.Boolean
    if name in {"is_parent", "evaluation_complete"}
    else pl.String
    for name in SELECTION_COLUMNS
}


class NanobodySelectionPage(BaseModel):
    """A bounded page; visibility defaults and score ranges cover the whole result."""

    columns: list[SelectionColumn]
    rows: list[dict[str, str | int | float | bool | None]]
    total_rows: int
    nonparent_count: int = Field(
        ge=0,
        description="Number of unique nonparent candidates in the whole result, before filtering or paging. Zero means no new designs were produced.",
    )
    offset: int
    limit: int
    parent_ids: list[str]
    default_hidden_columns: list[str]
    nativeness_ranges: dict[str, NativenessRange] = Field(
        description="Finite full-result VH2/VHH2 min/max before filtering and paging; zero/zero when unavailable. Visualization only; preserve raw scores and share each original score range with its delta."
    )
    germlines: dict[str, GermlinePresentation] = Field(
        default_factory=dict,
        description="Page-only VH germline evidence keyed by candidate_id; missing when annotation failed.",
    )
    reference: ReferenceInfo | None = None


def query_selection(
    source: Path | IO[bytes],
    *,
    offset: int = 0,
    limit: int = 50,
    parent_id: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> NanobodySelectionPage:
    """Preserve native order and numeric types without serializing unseen rows."""
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("offset must be nonnegative and limit between 1 and 200")
    if sort_by is not None and sort_by not in SELECTION_SCHEMA:
        raise ValueError("Unknown selection column")
    table = pl.read_csv(source, schema_overrides=SELECTION_SCHEMA)
    if table.columns != list(SELECTION_COLUMNS):
        raise ValueError("Selection columns do not match the nanobody publication")
    return NanobodySelectionPage(
        nonparent_count=table.filter(~pl.col("is_parent")).height,
        **selection_page_values(
            table,
            offset=offset,
            limit=limit,
            parent_id=parent_id,
            sort_by=sort_by,
            descending=descending,
        ),
    )


class NanobodyManifest(BaseModel):
    """Bind safe result members to the exact request and reviewed preparation."""

    schema_version: Literal[1, 2]
    execution_run_id: UUID
    parameters: NanobodySettings
    scientific_versions: dict[str, str]
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[ManifestFile] = Field(min_length=4, max_length=8)

    @model_validator(mode="after")
    def validate_members(self) -> "NanobodyManifest":
        """Permit only consolidated tables, not native trees or recursive manifests."""
        validate_manifest_paths(self.files)
        required = {
            "selection.csv",
            "generation.parquet",
            "imgt_mutations.parquet",
            "germlines.parquet",
        }
        optional = {
            f"scores/{model}_{kind}.parquet"
            for model in ("VH2", "VHH2")
            for kind in ("native_scores", "residue_scores")
        }
        if not required <= {file.path for file in self.files} <= required | optional:
            raise ValueError("Unexpected nanobody result membership")
        return self


def build_nanobody_archive(root: Path, destination: BinaryIO) -> BuiltResultArchive:
    """Verify every consolidated member through the shared archive builder."""
    manifest = NanobodyManifest.model_validate_json(
        (root / "manifest.json").read_bytes()
    )
    return build_table_archive(root, destination, manifest.files)
