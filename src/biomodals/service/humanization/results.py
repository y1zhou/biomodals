"""Bounded selection-table reads and deterministic scientific result archives."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, BinaryIO, Literal
from uuid import UUID

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import selection_table

SELECTION_SCHEMA = {
    **selection_table((), (), ()).schema,
    "quality_tier": pl.Int64,
    "panel_order": pl.Int64,
}


class SelectionColumn(BaseModel):
    """Scalar column type independent of the values on the requested page."""

    name: str
    type: Literal["string", "integer", "number", "boolean"]


class NativenessRange(BaseModel):
    """Full-result finite bounds for one original nativeness score."""

    min: float
    max: float


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
    if sort_by is not None and sort_by not in SELECTION_SCHEMA:
        raise ValueError(f"Unknown selection column: {sort_by}")
    table = pl.read_csv(path, schema_overrides=SELECTION_SCHEMA)
    if set(table.columns) != set(SELECTION_SCHEMA):
        raise ValueError("Selection table columns do not match the workflow schema")
    parent_ids = table.get_column("parent_id").unique().sort().to_list()
    hidden = table.select(
        *(
            pl.col(name).is_null().all().alias(name)
            for name in table.columns
            if name.endswith("_error")
        ),
        pl
        .col("evaluation_complete")
        .eq(False)
        .any()
        .not_()
        .alias("evaluation_complete"),
    ).row(0, named=True)
    nativeness_ranges = table.select(
        pl.struct(
            pl
            .col(name)
            .filter(pl.col(name).is_finite())
            .min()
            .fill_null(0.0)
            .alias("min"),
            pl
            .col(name)
            .filter(pl.col(name).is_finite())
            .max()
            .fill_null(0.0)
            .alias("max"),
        ).alias(name)
        for name in table.columns
        if name.endswith("_nativeness")
    ).row(0, named=True)
    if parent_id is not None:
        table = table.filter(pl.col("parent_id") == parent_id)
    total_rows = table.height
    if sort_by is not None:
        ties = [name for name in ("parent_id", "candidate_id") if name != sort_by]
        table = table.sort(
            sort_by,
            *ties,
            descending=[descending, *[False for _ in ties]],
            nulls_last=True,
        )
    types: dict[pl.DataType, Literal["string", "integer", "number", "boolean"]] = {
        pl.String(): "string",
        pl.Int64(): "integer",
        pl.Float64(): "number",
        pl.Boolean(): "boolean",
    }
    return SelectionPage(
        columns=[
            SelectionColumn(name=name, type=types[dtype])
            for name, dtype in table.schema.items()
        ],
        rows=table.slice(offset, limit).to_dicts(),
        total_rows=total_rows,
        offset=offset,
        limit=limit,
        parent_ids=parent_ids,
        default_hidden_columns=[name for name, hide in hidden.items() if hide],
        nativeness_ranges=nativeness_ranges,
    )


class BuiltHumanizationArchive(BaseModel):
    """Metadata for publishing an immutable service-built ZIP."""

    size_bytes: int
    sha256: str


class HumanizationManifestFile(BaseModel):
    """One scientific publication member with its expected content identity."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HumanizationManifest(BaseModel):
    """Publication identity and safe membership, shared by download and packaging."""

    schema_version: Literal[2]
    execution_run_id: UUID
    parameters: HumanizationSettings
    scientific_versions: dict[str, str]
    files: list[HumanizationManifestFile]

    @model_validator(mode="after")
    def validate_members(self) -> HumanizationManifest:
        """Reject unsafe or ambiguous file destinations before any downloads."""
        names = [record.path for record in self.files]
        if len(names) != len(set(names)) or "manifest.json" in names:
            raise ValueError("Duplicate or recursive manifest entries")
        for name in names:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or "\\" in name
            ):
                raise ValueError(f"Unsafe result path: {name}")
        if not {"selection.csv", "imgt_mutations.parquet"} <= set(names):
            raise ValueError("Scientific publication is missing required tables")
        return self


def build_humanization_archive(
    root: Path,
    destination: BinaryIO,
) -> BuiltHumanizationArchive:
    """Verify the workflow manifest while streaming a reproducible ZIP."""
    manifest = HumanizationManifest.model_validate_json(
        (root / "manifest.json").read_bytes()
    )
    records = {record.path: record for record in manifest.files}
    paths = sorted(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Result directory contains a symbolic link")
    files = {
        path.relative_to(root).as_posix(): path for path in paths if path.is_file()
    }
    if set(files) != {*records, "manifest.json"}:
        raise ValueError("Result directory does not match the workflow manifest")
    destination.seek(0)
    destination.truncate()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            digest = hashlib.sha256()
            size = 0
            with (
                path.open("rb") as source,
                archive.open(info, "w", force_zip64=True) as member,
            ):
                while chunk := source.read(1024 * 1024):
                    member.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if name in records:
                record = records[name]
                if (size, digest.hexdigest()) != (
                    record.size_bytes,
                    record.content_sha256,
                ):
                    raise ValueError(f"Result file does not match its manifest: {name}")
    size_bytes = destination.tell()
    destination.seek(0)
    digest = hashlib.sha256()
    while chunk := destination.read(1024 * 1024):
        digest.update(chunk)
    destination.seek(0)
    return BuiltHumanizationArchive(size_bytes=size_bytes, sha256=digest.hexdigest())
