"""Humanization cross-evaluation archive contract used by its evaluator apps."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import orjson
import polars as pl

from biomodals.helper.shell import package_outputs
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


def scoring_result(
    method: str,
    inputs: pl.DataFrame,
    summary: pl.DataFrame,
    details: dict[str, pl.DataFrame],
    scientific_identity: dict[str, Any],
) -> AppRunResult:
    """Publish inputs, scalar summaries, native details, and exact file digests."""
    if method not in {"sapiens", "humatch", "pabnativ2"}:
        raise ValueError("Unknown humanization evaluator")
    if any(not name.isidentifier() for name in details):
        raise ValueError("Detail table names must be identifiers")
    with TemporaryDirectory(prefix=f"{method}_scores_") as temporary:
        root = Path(temporary) / f"{method}_scores"
        root.mkdir()
        inputs.write_csv(root / "input.csv")
        summary.write_csv(root / "summary.csv")
        for name, frame in details.items():
            frame.write_parquet(root / f"{name}.parquet", compression="zstd")
        manifest = {
            "schema_version": 1,
            "operation": f"{method}_score",
            "scientific_identity": scientific_identity,
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(root.iterdir())
            },
        }
        (root / "manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS)
        )
        archive = package_outputs(root, num_threads=2)
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name=f"{method}_scores",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=archive,
                    filename=f"{method}_scores.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
                metadata={"pair_count": inputs.height, "operation": f"{method}_score"},
            )
        ],
    )
