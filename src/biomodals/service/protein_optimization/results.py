"""Disposable disk-backed candidate queries; the remote CSV remains authoritative."""

# SQL identifiers are generated integer column offsets, never user strings;
# all filter values and IDs use bound parameters.
# ruff: noqa: S608

from __future__ import annotations

import re
import sqlite3
from collections.abc import Generator
from pathlib import Path

import orjson
import polars as pl

from biomodals.app.design.mutation_ridge.execution import MAX_RESULT_BYTES
from biomodals.helper.artifacts import sha256_file
from biomodals.service.protein_optimization.contracts import (
    OptimizationCandidatePage,
    OptimizationResultSummary,
)
from biomodals.service.selection import SelectionColumn
from biomodals.workflow.protein_optimization.publication import OptimizationManifest

SORT_COLUMNS = ("id", "mutations", "predicted_label", "n_mutations", "n_new_mutations")
_STRING_PREFIX = r"^[\p{White_Space}\p{Cc}]*[=+\-@]"


def _schema(manifest: OptimizationManifest) -> dict[str, pl.DataType]:
    return {
        "id": pl.String(),
        "mutations": pl.String(),
        "predicted_label": pl.Float64(),
        "n_mutations": pl.Int64(),
        "n_new_mutations": pl.Int64(),
        "warnings": pl.String(),
        **{column: pl.String() for column in manifest.chain_columns.values()},
    }


def csv_bytes(table: pl.DataFrame, *, include_header: bool) -> bytes:
    """Escape spreadsheet formulas in text only; preserve exact numeric predictions."""
    text = [name for name, dtype in table.schema.items() if dtype == pl.String]
    safe = table.with_columns(
        pl
        .when(pl.col(name).str.contains(_STRING_PREFIX))
        .then(pl.lit("'") + pl.col(name))
        .otherwise(pl.col(name))
        .alias(name)
        for name in text
    )
    # Current generated headers start with safe fixed prefixes. The data remain
    # raw in SQLite, so presentation quoting never changes candidate identity.
    return safe.write_csv(include_header=include_header).encode()


def build_projection(
    source: Path,
    destination: Path,
    database: Path,
    manifest: OptimizationManifest,
) -> tuple[int, str]:
    """Parse once in bounded Polars batches, index scalar keys and stream safe CSV."""
    if (source.stat().st_size, sha256_file(source)) != (
        manifest.csv_bytes,
        manifest.csv_sha256,
    ) or manifest.csv_bytes > MAX_RESULT_BYTES:
        raise ValueError("Candidate CSV does not match its publication")
    schema = _schema(manifest)
    names = list(schema)
    if pl.read_csv(source, n_rows=0).columns != names:
        raise ValueError("Candidate columns do not match their chain mapping")
    columns = [
        SelectionColumn(
            name=name,
            type="number"
            if dtype == pl.Float64
            else "integer"
            if dtype == pl.Int64
            else "string",
        )
        for name, dtype in schema.items()
    ]
    summary = OptimizationResultSummary(
        mode=manifest.mode,
        direction=manifest.direction,
        candidate_count=manifest.candidate_count,
        chain_columns=manifest.chain_columns,
        validation=manifest.validation,
    )
    with sqlite3.connect(database) as connection, destination.open("wb") as output:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        definitions = [
            f"c{index} {'REAL' if dtype == pl.Float64 else 'INTEGER' if dtype == pl.Int64 else 'TEXT'}"
            for index, dtype in enumerate(schema.values())
        ]
        connection.execute(
            f"CREATE TABLE candidates (ordinal INTEGER PRIMARY KEY, {', '.join(definitions)})"
        )
        connection.execute("CREATE TABLE metadata (content BLOB NOT NULL)")
        statement = f"INSERT INTO candidates VALUES ({','.join('?' for _ in range(len(names) + 1))})"
        reader = pl.scan_csv(source, schema_overrides=schema).collect_batches(
            chunk_size=1024
        )
        count = 0
        output.write(csv_bytes(pl.DataFrame(schema=schema), include_header=True))
        for table in reader:
            if (
                table.select(pl.col("predicted_label").is_finite().all()).item()
                is not True
                or table.select(
                    pl.any_horizontal(
                        pl.col(
                            "id",
                            "mutations",
                            "predicted_label",
                            "n_mutations",
                            "n_new_mutations",
                        ).is_null()
                    ).any()
                ).item()
                or table.filter(
                    (pl.col("n_mutations") < 1) | (pl.col("n_new_mutations") < 0)
                ).height
            ):
                raise ValueError("Candidate rows contain invalid scalar values")
            connection.executemany(
                statement,
                ((count + index, *row) for index, row in enumerate(table.iter_rows())),
            )
            count += table.height
            if count > manifest.candidate_count:
                raise ValueError("Candidate count exceeds its publication")
            output.write(csv_bytes(table, include_header=False))
        if count != manifest.candidate_count:
            raise ValueError("Candidate count differs from its publication")
        for name in SORT_COLUMNS:
            index = names.index(name)
            unique = "UNIQUE " if name == "id" else ""
            connection.execute(
                f"CREATE {unique}INDEX sort_{index} ON candidates(c{index})"
            )
        output.flush()
        size, digest = destination.stat().st_size, sha256_file(destination)
        connection.execute(
            "INSERT INTO metadata VALUES (?)",
            (
                orjson.dumps({
                    "download_sha256": digest,
                    "summary": summary.model_dump(mode="json"),
                    "columns": [column.model_dump() for column in columns],
                }),
            ),
        )
    return size, digest


def open_projection(path: Path, csv_sha256: str) -> tuple[sqlite3.Connection, dict]:
    """A caller must hold the owning verified Result lease throughout the read."""
    if path.is_symlink():
        raise ValueError("Invalid candidate projection")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        metadata = orjson.loads(
            connection.execute("SELECT content FROM metadata").fetchone()[0]
        )
        if metadata["download_sha256"] != csv_sha256:
            raise ValueError("Candidate projection belongs to another result")
        return connection, metadata
    except BaseException:
        connection.close()
        raise


def query_candidates(
    path: Path,
    csv_sha256: str,
    *,
    offset: int = 0,
    limit: int = 50,
    sort_by: str | None = None,
    descending: bool = False,
    mutations: str | None = None,
    n_mutations: int | None = None,
    n_new_mutations: int | None = None,
) -> OptimizationCandidatePage:
    """Only index-backed scalar sorts; original ordinal resets scientific order."""
    connection, metadata = open_projection(path, csv_sha256)
    try:
        names = [column["name"] for column in metadata["columns"]]
        predicates, values = [], []
        for name, value in (
            ("n_mutations", n_mutations),
            ("n_new_mutations", n_new_mutations),
        ):
            if value is not None:
                predicates.append(f"c{names.index(name)} = ?")
                values.append(value)
        if mutations:
            predicates.append("instr(c1, ?) > 0")
            values.append(mutations)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        if sort_by is not None and sort_by not in SORT_COLUMNS:
            raise ValueError("Unknown or unsupported candidate sort column")
        order = (
            "ordinal"
            if sort_by is None
            else f"c{names.index(sort_by)} {'DESC' if descending else 'ASC'}, ordinal"
        )
        total = connection.execute(
            "SELECT count(*) FROM candidates" + where, values
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT {','.join(f'c{i}' for i in range(len(names)))} FROM candidates{where} ORDER BY {order} LIMIT ? OFFSET ?",
            (*values, limit, offset),
        ).fetchall()
        return OptimizationCandidatePage(
            summary=metadata["summary"],
            columns=metadata["columns"],
            rows=[dict(zip(names, row, strict=True)) for row in rows],
            total_rows=total,
            offset=offset,
            limit=limit,
        )
    finally:
        connection.close()


def selected_csv(path: Path, csv_sha256: str, ids: list[str]) -> Generator[bytes]:
    """Verify every selected ID before the first byte; stream in scientific order."""
    if any(re.fullmatch(r"candidate_[0-9]{9}", value) is None for value in ids):
        raise ValueError("Invalid candidate ID")
    connection, metadata = open_projection(path, csv_sha256)
    try:
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TEMP TABLE selected (id TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT OR IGNORE INTO selected VALUES (?)", ((value,) for value in ids)
        )
        missing = connection.execute(
            "SELECT count(*) FROM selected LEFT JOIN candidates ON selected.id=candidates.c0 WHERE candidates.c0 IS NULL"
        ).fetchone()[0]
        if missing:
            raise ValueError("Selected candidates do not belong to this result")
        types = {"string": pl.String, "number": pl.Float64, "integer": pl.Int64}
        schema = {
            column["name"]: types[column["type"]] for column in metadata["columns"]
        }
        cursor = connection.execute(
            f"SELECT {','.join(f'c{i}' for i in range(len(schema)))} FROM candidates JOIN selected ON candidates.c0=selected.id ORDER BY ordinal"
        )
        yield csv_bytes(pl.DataFrame(schema=schema), include_header=True)
        while rows := cursor.fetchmany(1024):
            yield csv_bytes(
                pl.DataFrame(rows, schema=schema, orient="row"), include_header=False
            )
    finally:
        connection.close()
