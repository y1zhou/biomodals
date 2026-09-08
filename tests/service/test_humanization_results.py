"""Selection queries preserve scientific types and archives preserve evidence."""

import hashlib
import io
import zipfile
from pathlib import Path

import orjson
import polars as pl
import pytest

from biomodals.service.humanization.results import (
    SELECTION_SCHEMA,
    build_humanization_archive,
    query_selection,
)


def _selection(tmp_path: Path) -> Path:
    rows = [
        {"parent_id": "002", "candidate_id": "02", "panel_order": None},
        {"parent_id": "001", "candidate_id": "03", "panel_order": 2},
        {"parent_id": "001", "candidate_id": "01", "panel_order": 10},
        {"parent_id": "001", "candidate_id": "02", "panel_order": 2},
    ]
    path = tmp_path / "selection.csv"
    pl.DataFrame(rows, schema=SELECTION_SCHEMA).write_csv(path)
    return path


def test_selection_paging_types_and_parent_filter(tmp_path: Path) -> None:
    """Default pages preserve ordering, identifiers, nulls and scalar types."""
    path = _selection(tmp_path)
    page = query_selection(path, offset=1, limit=2)
    assert [row["candidate_id"] for row in page.rows] == ["03", "01"]
    assert page.total_rows == 4
    assert page.parent_ids == ["001", "002"]
    types = {column.name: column.type for column in page.columns}
    assert types["parent_id"] == "string"
    assert types["panel_order"] == "integer"
    assert types["sapiens_vh_mean_probability"] == "number"
    assert types["is_parent"] == "boolean"
    assert page.rows[0]["sapiens_vh_mean_probability"] is None
    filtered = query_selection(io.BytesIO(path.read_bytes()), parent_id="002")
    assert filtered.total_rows == 1
    assert filtered.rows[0]["parent_id"] == "002"
    assert filtered.parent_ids == ["001", "002"]
    assert query_selection(path, offset=99).rows == []
    assert query_selection(path, parent_id="missing").total_rows == 0


def test_visibility_defaults_use_all_rows_before_filter_and_page(
    tmp_path: Path,
) -> None:
    """An error outside the visible parent/page must remain discoverable."""
    path = _selection(tmp_path)
    table = pl.read_csv(path, schema_overrides=SELECTION_SCHEMA).with_columns(
        pl
        .when(pl.col("parent_id") == "002")
        .then(pl.lit("failed"))
        .otherwise(None)
        .alias("sapiens_error"),
        (pl.col("parent_id") != "002").alias("evaluation_complete"),
    )
    table.write_csv(path)
    expected = [
        name
        for name in table.columns
        if name.endswith("_error") and name != "sapiens_error"
    ]
    for kwargs in (
        {},
        {"offset": 1, "limit": 1},
        {"parent_id": "001"},
        {"parent_id": "missing"},
        {"sort_by": "panel_order", "descending": True},
    ):
        assert query_selection(path, **kwargs).default_hidden_columns == expected
    table.with_columns(
        pl.lit(None).alias("sapiens_error"), pl.lit(True).alias("evaluation_complete")
    ).write_csv(path)
    assert query_selection(path).default_hidden_columns == [
        *[name for name in table.columns if name.endswith("_error")],
        "evaluation_complete",
    ]


@pytest.mark.parametrize(
    "descending,expected",
    [(False, ["02", "03", "01", "02"]), (True, ["01", "02", "03", "02"])],
)
def test_selection_numeric_sort_nulls_last_and_stable_ties(
    tmp_path: Path,
    descending: bool,
    expected: list[str],
) -> None:
    """Both numeric sort directions keep nulls last and ties deterministic."""
    page = query_selection(
        _selection(tmp_path), sort_by="panel_order", descending=descending
    )
    assert [row["candidate_id"] for row in page.rows] == expected
    assert page.rows[-1]["panel_order"] is None


@pytest.mark.parametrize(
    "kwargs", [{"offset": -1}, {"limit": 0}, {"limit": 201}, {"sort_by": "missing"}]
)
def test_selection_rejects_invalid_queries(tmp_path: Path, kwargs: dict) -> None:
    """Invalid page bounds and unknown sort columns cannot reach Polars."""
    with pytest.raises(ValueError):
        query_selection(_selection(tmp_path), **kwargs)


def _result_directory(tmp_path: Path) -> Path:
    root = tmp_path / "result"
    root.mkdir()
    _selection(root)
    pl.DataFrame({"position": [1]}).write_parquet(root / "imgt_mutations.parquet")
    (root / "native").mkdir()
    (root / "native" / "generation.tar.zst").write_bytes(b"native evidence")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (root / "manifest.json").write_bytes(
        orjson.dumps({
            "schema_version": 2,
            "files": files,
            "parameters": {},
            "execution_run_id": "00000000-0000-0000-0000-000000000001",
            "scientific_versions": {},
        })
    )
    return root


def test_archive_rebuild_is_identical_and_preserves_all_files(tmp_path: Path) -> None:
    """Restoration ignores filesystem timestamps without losing file contents."""
    root = _result_directory(tmp_path)
    first, second = io.BytesIO(), io.BytesIO()
    metadata = build_humanization_archive(root, first)
    for path in root.rglob("*"):
        if path.is_file():
            path.touch()
    assert build_humanization_archive(root, second) == metadata
    assert first.getvalue() == second.getvalue()
    assert metadata.size_bytes == len(first.getvalue())
    assert metadata.sha256 == hashlib.sha256(first.getvalue()).hexdigest()
    with zipfile.ZipFile(first) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }


@pytest.mark.parametrize(
    "damage", ["digest", "extra", "missing", "symlink", "traversal"]
)
def test_archive_rejects_incomplete_or_unsafe_evidence(
    tmp_path: Path, damage: str
) -> None:
    """Incomplete, changed and path-unsafe scientific bundles are not published."""
    root = _result_directory(tmp_path)
    if damage == "digest":
        (root / "selection.csv").write_bytes(b"changed")
    elif damage == "extra":
        (root / "unexpected").write_bytes(b"extra")
    elif damage == "missing":
        (root / "selection.csv").unlink()
    elif damage == "symlink":
        (root / "linked").symlink_to(tmp_path)
    else:
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        manifest["files"][0]["path"] = "../outside"
        (root / "manifest.json").write_bytes(orjson.dumps(manifest))
    with pytest.raises(ValueError):
        build_humanization_archive(root, io.BytesIO())
