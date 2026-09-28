"""Disk-backed presentation preserves raw science and bounds response memory."""

import asyncio
from io import BytesIO
from uuid import uuid4

import polars as pl
import pytest

from biomodals.app.design.mutation_ridge.regression import (
    ValidationPoint,
    ValidationSummary,
)
from biomodals.helper.artifacts import sha256_file
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.protein_optimization.downloads import (
    load_selection,
    prepare_selection,
)
from biomodals.service.protein_optimization.results import (
    build_projection,
    query_candidates,
    selected_csv,
)
from biomodals.workflow.protein_optimization.publication import OptimizationManifest


def _result(tmp_path, count=251):
    table = pl.DataFrame(
        {
            "id": [f"candidate_{index + 1:09d}" for index in range(count)],
            "mutations": [f"A:A{index + 1}G" for index in range(count)],
            "predicted_label": [-0.12345678901234567 + index for index in range(count)],
            "n_mutations": [1] * count,
            "n_new_mutations": [0] * count,
            "warnings": [
                None if index % 2 else "\t =formula" for index in range(count)
            ],
            "sequence_A": ["AG" for _ in range(count)],
        },
        schema_overrides={"warnings": pl.String},
    )
    source = tmp_path / "source.csv"
    table.write_csv(source)
    manifest = OptimizationManifest(
        execution_run_id=uuid4(),
        design_digest="a" * 64,
        scientific_versions={"result": "1"},
        mode="combination",
        direction="minimize",
        candidate_count=count,
        chain_columns={"A": "sequence_A"},
        validation=ValidationSummary(
            regime="held_out_combinations",
            training_variants=4,
            evaluated_variants=0,
            folds=0,
            warnings=("Singles-only measurements",),
        ),
        csv_bytes=source.stat().st_size,
        csv_sha256=sha256_file(source),
    )
    return table, source, manifest


def test_projection_paging_raw_values_and_selected_order(tmp_path):
    """Filter/sort never changes identity, numeric precision or selected export order."""
    table, source, manifest = _result(tmp_path)
    csv, database = tmp_path / "download.csv", tmp_path / "index.sqlite"
    _, digest = build_projection(source, csv, database, manifest)
    # The source is not needed to serve any later query.
    source.unlink()
    page = query_candidates(database, digest, offset=200, limit=50)
    assert len(page.rows) == 50
    assert page.rows[0]["id"] == "candidate_000000201"
    assert page.total_rows == 251
    assert page.summary.validation.warnings == ("Singles-only measurements",)
    reverse = query_candidates(
        database, digest, sort_by="predicted_label", descending=True, limit=1
    )
    assert reverse.rows[0]["id"] == "candidate_000000251"
    filtered = query_candidates(
        database, digest, mutations="A:A2", n_mutations=1, n_new_mutations=0
    )
    assert filtered.total_rows == 63
    assert query_candidates(database, digest, mutations="%' OR 1=1 --").total_rows == 0
    first = query_candidates(database, digest, limit=1).rows[0]
    assert first["warnings"] == "\t =formula"
    assert first["predicted_label"] == table["predicted_label"][0]
    selected = pl.read_csv(
        BytesIO(
            b"".join(
                selected_csv(
                    database,
                    digest,
                    [
                        "candidate_000000251",
                        "candidate_000000001",
                        "candidate_000000001",
                    ],
                )
            )
        )
    )
    assert selected["id"].to_list() == ["candidate_000000001", "candidate_000000251"]
    assert selected["warnings"][0] == "'\t =formula"
    assert selected["predicted_label"][0] == table["predicted_label"][0]
    assert pl.read_csv(csv).rows()[:1] == selected.rows()[:1]
    with pytest.raises(ValueError, match="belong"):
        next(selected_csv(database, digest, ["candidate_000000999"]))
    with pytest.raises(ValueError, match="sort"):
        query_candidates(database, digest, sort_by="c0; DROP TABLE candidates")


@pytest.mark.parametrize("legacy", [False, True])
def test_heldout_evidence_survives_projection_and_candidate_filters(tmp_path, legacy):
    """The plot describes held-out measurements, never the visible candidate page."""
    _, source, manifest = _result(tmp_path)
    summary = ValidationSummary(
        regime="supported_combinations",
        training_variants=4,
        evaluated_variants=1,
        folds=1,
        mae=0.5,
        rmse=0.5,
        points=(
            ValidationPoint(
                mutations="A:A1G,A:A2G",
                measured_label=-1.25,
                predicted_label=-0.75,
                prediction_count=1,
            ),
        ),
    )
    if legacy:
        summary = ValidationSummary.model_validate_json(
            summary.model_dump_json(exclude={"points"})
        )
    manifest = manifest.model_copy(update={"validation": summary})
    database = tmp_path / "index.sqlite"
    _, digest = build_projection(source, tmp_path / "download.csv", database, manifest)
    for kwargs in ({}, {"offset": 200}, {"mutations": "unmatched"}):
        page = query_candidates(database, digest, **kwargs)
        assert page.summary.validation == summary
        assert page.summary.validation.evaluated_variants == 1
        assert len(page.summary.validation.points) == (0 if legacy else 1)


def test_projection_verifies_source_and_empty_result(tmp_path):
    """A header-only scientific result remains usable; edited source bytes do not."""
    _, source, manifest = _result(tmp_path, 0)
    csv, database = tmp_path / "download.csv", tmp_path / "index.sqlite"
    _, digest = build_projection(source, csv, database, manifest)
    assert query_candidates(database, digest).rows == []
    assert pl.read_csv(csv).columns[-1] == "sequence_A"
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="publication"):
        build_projection(source, csv, tmp_path / "other.sqlite", manifest)
    with pytest.raises(ValueError, match="another result"):
        query_candidates(database, "f" * 64)


def test_cache_projection_lease_accounting_and_deletion(tmp_path):
    """The disposable index consumes cache bytes and shares deletion protection."""

    async def scenario():
        cache = ArtifactCache(tmp_path / "cache")
        job_id = str(uuid4())
        _, source, manifest = _result(tmp_path)
        csv, database = cache.staging_path(job_id), cache.staging_path(job_id)
        size, digest = await cache.run_bounded(
            build_projection, source, csv, database, manifest
        )
        await cache.run_bounded(cache.publish_derived, job_id, database)
        await cache.publish_staged(job_id, csv, size_bytes=size, sha256=digest)
        expected = size + cache.derived_path(job_id).stat().st_size
        assert cache.usage().cached_bytes == expected
        lease = await cache.acquire_async(job_id, size_bytes=size, sha256=digest)
        assert lease is not None
        assert cache.clear().entries == 0
        assert cache.remove_job_files(job_id) is False
        lease.close()
        cleared = cache.clear()
        assert (cleared.entries, cleared.bytes) == (1, expected)
        assert cache.usage().cached_bytes == 0
        assert cache.remove_job_files(job_id) is True
        await cache.shutdown()

    asyncio.run(scenario())


def test_selection_ticket_expiry_and_identity(tmp_path):
    """Selection URLs neither grant foreign access nor outlive their short intent."""
    from urllib.parse import parse_qs, urlsplit
    from uuid import UUID

    _, source, manifest = _result(tmp_path, 2)
    csv, database = tmp_path / "download.csv", tmp_path / "index.sqlite"
    _, digest = build_projection(source, csv, database, manifest)
    owner, job_id = uuid4(), uuid4()
    path = tmp_path / "selection.json"
    ticket = prepare_selection(
        path,
        database,
        digest,
        ["candidate_000000001"],
        owner_user_id=owner,
        job_id=job_id,
        now=100,
    )
    token = UUID(parse_qs(urlsplit(ticket.download_url).query)["ticket"][0])
    args = dict(
        token=token, owner_user_id=owner, job_id=job_id, csv_sha256=digest, now=399
    )
    assert load_selection(path, **args) == ["candidate_000000001"]
    for change in (
        {"owner_user_id": uuid4()},
        {"job_id": uuid4()},
        {"csv_sha256": "f" * 64},
        {"now": 400},
    ):
        with pytest.raises(ValueError, match="expired or was replaced"):
            load_selection(path, **(args | change))
