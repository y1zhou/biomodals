"""Stateless local antibody API, partial inputs and build-once reference tests."""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from antibody_fixture import VH, VL, reference_csv
from test_api_contract import ORIGIN, _app, _humanization_session, _request

from biomodals.helper.antibody import assign_germlines
from biomodals.service.antibody_sequence_analysis.analysis import (
    AnalysisService,
    parse_fasta,
)
from biomodals.service.antibody_sequence_analysis.contracts import (
    AnalysisRequest,
    FastaGroup,
)
from biomodals.service.antibody_sequence_analysis.reference import (
    TherapeuticReference,
    build_reference,
    present_germlines,
)


def test_fasta_pair_forms_errors_and_original_order():
    """Malformed records do not discard other entries or guess missing partners."""
    entries, issues = parse_fasta(
        FastaGroup(
            id="a",
            fasta=f">paired description\n{VH}:{VL}\n>split_vl\n{VL}\n>split_vh\n{VH}\n>single\n{VH}\n>missing_vh\n{VH}\n>bad\nACX\n>duplicate\nACD\n>duplicate\nACD",
        )
    )
    assert not issues
    assert [entry.id for entry in entries] == [
        "paired",
        "split",
        "single",
        "missing",
        "bad",
        "duplicate",
    ]
    assert entries[0].sequences == entries[1].sequences == {"vh": VH, "vl": VL}
    assert entries[2].sequences == {"sequence": VH}
    assert [entry.issues[0].code for entry in entries[3:]] == [
        "missing_partner",
        "invalid_sequence",
        "duplicate_id",
    ]


@pytest.mark.parametrize(
    "fasta,code",
    [
        ("ACD", "invalid_fasta"),
        (">\nACD", "invalid_header"),
        ("", "empty_group"),
        ("\n".join(f">p{i}\nACD" for i in range(1001)), "too_many_entries"),
    ],
)
def test_group_errors_before_native_work(fasta, code):
    """Reject unparseable or excessive groups before any numbering work."""
    entries, issues = parse_fasta(FastaGroup(id="a", fasta=fasta))
    assert not entries
    assert issues[0].code == code


def test_reference_exact_cohort_dedup_species_ties_and_cache(tmp_path, monkeypatch):
    """Fractional credit is per species/gene, not per duplicated allele."""
    assignment = assign_germlines(VH)
    ref = assignment["v"][0]
    assignment["v"] = [ref, {**ref, "allele": "99"}, {**ref, "species": "Mus musculus"}]
    monkeypatch.setattr(
        "biomodals.service.antibody_sequence_analysis.reference.assign_germlines",
        lambda _: assignment,
    )
    raw = reference_csv()
    calls = []
    reference = TherapeuticReference(
        tmp_path / "reference.json", download=lambda: calls.append(1) or raw
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(lambda _: reference.get(), range(4)))
    assert calls == [1]
    snapshot = snapshots[0]
    assert snapshot.info.heavy_sequences == snapshot.info.light_sequences == 2
    assert snapshot.info.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert snapshot.info.downloaded_at
    usage = present_germlines(assignment, "vh", snapshot)
    assert [row.frequency for row in usage.v_usage] == pytest.approx([0.5, 0.5])
    assert len(usage.assignment["v"]) == 3
    loaded = TherapeuticReference(
        reference.path, download=lambda: pytest.fail("cache must not refresh")
    ).get()
    assert loaded == snapshot
    reference.path.write_text("broken")
    with pytest.raises(ValueError):
        TherapeuticReference(
            reference.path, download=lambda: pytest.fail("do not replace corrupt cache")
        ).get()
    assert reference.path.read_text() == "broken"


def test_analysis_deduplicates_native_work_and_preserves_partial_groups(
    tmp_path, monkeypatch
):
    """Repeated chains cost one analysis; independent errors retain valid metrics."""
    from biomodals.service.antibody_sequence_analysis import analysis

    calls = []
    original = analysis.analyze_chain

    def counted(sequence):
        calls.append(sequence)
        return original(sequence)

    monkeypatch.setattr(analysis, "analyze_chain", counted)
    service = AnalysisService(
        TherapeuticReference(tmp_path / "reference.json", download=reference_csv)
    )

    async def run():
        try:
            return await service.analyze(
                AnalysisRequest(
                    groups=[
                        FastaGroup(
                            id="a",
                            fasta=f">p\n{VH}:{VL}\n>q\n{VH}:{VL}\n>bad\nAX\n>swapped\n{VL}:{VH}\n>short\nACD",
                        ),
                        FastaGroup(id="b", fasta="not a fasta"),
                    ]
                )
            )
        finally:
            await service.shutdown()

    response = asyncio.run(run())
    assert sorted(calls) == sorted([VH, VL, "ACD"])
    first, repeated, invalid, swapped, short = response.groups[0].entries
    assert first.vh_vl_pi == repeated.vh_vl_pi
    assert invalid.issues[0].code == "invalid_sequence"
    assert swapped.issues[0].code == "wrong_chain_role"
    assert swapped.vh_vl_pi is None and swapped.vh.sequence == VL
    assert short.unassigned.metrics["pi"] > 0
    assert response.groups[1].issues[0].code == "invalid_fasta"


def test_private_api_options_details_limits_and_no_jobs(tmp_path):
    """Use shared authentication, no-store and body limits; never submit work."""
    app = _app(tmp_path)
    service = app.state.antibody_analysis
    service.reference = TherapeuticReference(
        tmp_path / "reference.json", download=reference_csv
    )
    root = "/api/v1/antibody-sequence-analysis"
    assert _request(app, "GET", root + "/options").status_code == 401
    assert (
        _request(
            app,
            "POST",
            root + "/analyze",
            headers={"Origin": ORIGIN},
            json={"groups": [{"id": "a", "fasta": ">x\nACD"}]},
        ).status_code
        == 401
    )
    _humanization_session(app)
    options = _request(app, "GET", root + "/options")
    assert options.json()["max_entries_per_group"] == 1000
    assert options.json()["max_chain_length"] == 512
    response = _request(
        app,
        "POST",
        root + "/analyze",
        json={"groups": [{"id": "a", "fasta": f">p\n{VH}:{VL}"}]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert (
        response.json()["groups"][0]["entries"][0]["vh"]["germlines"]["assignment"][
            "v_gene"
        ]
        == "IGHV1-2"
    )
    detail = _request(
        app, "POST", root + "/sequence", json={"sequence": VH, "scheme": "kabat"}
    )
    assert detail.status_code == 200 and detail.json()["residues"]
    oversized = _request(
        app, "POST", root + "/analyze", content=b" " * (4 * 1024 * 1024 + 1)
    )
    assert oversized.status_code == 413
    assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    asyncio.run(service.shutdown())


def test_missing_reference_leaves_metrics_available_and_corrupt_file_intact(tmp_path):
    """Usage failure is separate from the local scientific calculation."""
    path = tmp_path / "reference.json"
    path.write_text("corrupt")
    service = AnalysisService(
        TherapeuticReference(path, download=lambda: pytest.fail("no download"))
    )

    async def run():
        try:
            return await service.analyze(
                AnalysisRequest(groups=[FastaGroup(id="a", fasta=f">p\n{VH}:{VL}")])
            )
        finally:
            await service.shutdown()

    response = asyncio.run(run())
    assert response.reference.status == "unavailable"
    chain = response.groups[0].entries[0].vh
    assert chain.metrics["pi"] > 0 and chain.germlines.v_usage[0].frequency is None
    assert path.read_text() == "corrupt"


def test_real_fixture_reference_can_be_built_without_modal():
    """Small deterministic public reference uses the installed native package."""
    snapshot = build_reference(reference_csv())
    assert snapshot.info.heavy_sequences == snapshot.info.light_sequences == 2
    assert snapshot.usage


@pytest.mark.parametrize("tamper", [False, True])
def test_humanization_page_bounded_frozen_assignments_and_usage(tmp_path, tamper):
    """Serve only selected evidence and reject assignments for different chains."""
    from uuid import UUID, uuid4

    import polars as pl
    from antibody_fixture import annotated_archive

    from biomodals.service.humanization.results import SELECTION_SCHEMA
    from biomodals.service.store import JobState
    from biomodals.workflow.humanization.settings import HumanizationSettings

    app = _app(tmp_path)
    _humanization_session(app)
    app.state.antibody_analysis.reference = TherapeuticReference(
        tmp_path / "reference.json", download=reference_csv
    )
    submitted = _request(
        app,
        "POST",
        "/api/v1/humanization/jobs",
        json={"pairs": [{"id": "a", "vh": VH, "vl": VL}]},
        headers={"Origin": ORIGIN, "Idempotency-Key": str(uuid4())},
    )
    job_id = UUID(submitted.json()["job_id"])
    table = pl.DataFrame(
        [
            {"parent_id": "a", "candidate_id": "c1", "vh": VH, "vl": VL},
            {"parent_id": "b", "candidate_id": "c2", "vh": VH, "vl": VL},
        ],
        schema=SELECTION_SCHEMA,
    )
    content = annotated_archive(
        table, job_id, HumanizationSettings(), {}, tamper=tamper
    )
    cache = app.state.cache
    staging = cache.staging_path(str(job_id))
    staging.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    asyncio.run(
        cache.publish_staged(
            str(job_id), staging, size_bytes=len(content), sha256=digest
        )
    )
    app.state.store.complete_job(
        job_id,
        result_state=JobState.SUCCEEDED,
        result_filename="humanization.zip",
        result_media_type="application/zip",
        result_size_bytes=len(content),
        result_sha256=digest,
        result_archive_schema="humanization/1",
        now=110,
    )
    response = _request(
        app,
        "GET",
        f"/api/v1/humanization/jobs/{job_id}/selection?limit=1&parent_id=a&sort_by=vh_v_gene",
    )
    if tamper:
        assert response.status_code == 409
        assert response.json()["code"] == "result_invalid"
    else:
        assert response.status_code == 200, response.text
        result = response.json()
        assert set(result["germlines"]) == {"c1"}
        assert result["rows"][0]["vh_v_gene"] == "IGHV1-2"
        assert result["germlines"]["c1"]["vh"]["assignment"]["v_gene"] == "IGHV1-2"
        assert result["reference"]["status"] == "available"
        csv = _request(app, "GET", f"/api/v1/humanization/jobs/{job_id}/selection.csv")
        assert csv.status_code == 200
        assert "vh_v_gene" in csv.text.splitlines()[0]
    asyncio.run(app.state.antibody_analysis.shutdown())
