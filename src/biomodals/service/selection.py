"""Polars-native table presentation shared by the two humanization Tools."""

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel


class SelectionColumn(BaseModel):
    """Scalar column type independent of the values on the requested page."""

    name: str
    type: Literal["string", "integer", "number", "boolean"]


class NativenessRange(BaseModel):
    """Full-result finite bounds for one original nativeness score."""

    min: float
    max: float


def selection_page_values(
    table: pl.DataFrame,
    *,
    offset: int,
    limit: int,
    parent_id: str | None,
    sort_by: str | None,
    descending: bool,
) -> dict[str, Any]:
    """Compute full-result metadata before bounded filtering/sorting/pagination."""
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
    return dict(
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
