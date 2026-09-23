"""Local analysis admission, fair batch windows and cancellation ownership."""

import asyncio
from threading import Event, Lock
from uuid import uuid4

import httpx
import pytest
from nanobody_fixture import VHH
from test_api_contract import ORIGIN, _app, _humanization_session, _request

from biomodals.service.antibody_sequence_analysis import analysis
from biomodals.service.antibody_sequence_analysis.contracts import AnalysisRequest
from biomodals.service.antibody_sequence_analysis.reference import TherapeuticReference
from biomodals.service.http_contract import CodedAPIError


def test_cancelled_waiter_retains_capacity_until_work_finishes(tmp_path):
    """Disconnecting cannot free a slot still owned by a native computation."""
    service = analysis.AnalysisService(
        TherapeuticReference(tmp_path / "unused.json"), max_requests=1
    )
    started, release = Event(), Event()

    def blocked():
        started.set()
        assert release.wait(5)
        return 42

    async def check():
        waiter = asyncio.create_task(service.run(blocked))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            with pytest.raises(CodedAPIError) as error:
                await service.run(lambda: 0)
            assert error.value.status_code == 503
            assert error.value.code == "local_analysis_busy"
            release.set()
        finally:
            release.set()
            await service.shutdown()

    asyncio.run(check())


def test_large_batch_leaves_room_for_interactive_work(tmp_path, monkeypatch):
    """A large accepted batch does not enqueue every chain ahead of inspection."""
    service = analysis.AnalysisService(TherapeuticReference(tmp_path / "unused.json"))
    started, release, lock = Event(), Event(), Lock()
    calls = []

    def chain(sequence):
        with lock:
            calls.append(sequence)
            if len(calls) == 4:
                started.set()
        assert release.wait(5)
        return sequence

    async def reference():
        return None, None

    monkeypatch.setattr(analysis, "analyze_chain", chain)
    monkeypatch.setattr(service, "reference_snapshot", reference)
    monkeypatch.setattr(service, "_assemble", lambda *args: args[2])
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = ["AAA" + a + b for a in alphabet for b in alphabet]
    request = AnalysisRequest(
        groups=[
            {
                "id": "one",
                "fasta": "\n".join(f">s{i}\n{s}" for i, s in enumerate(sequences)),
            }
        ]
    )

    async def check():
        batch = asyncio.create_task(service.analyze(request))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert (
                await asyncio.wait_for(service.run(lambda: "inspection"), 1)
                == "inspection"
            )
            assert len(calls) == 4
            release.set()
            result = await batch
            assert result == {sequence: sequence for sequence in sequences}
        finally:
            release.set()
            await service.shutdown()

    asyncio.run(check())


def test_busy_api_rejects_before_admission_but_allows_exact_replay(tmp_path):
    """All four native CPU surfaces share one gate; saved replay needs no slot."""
    app = _app(tmp_path)
    _humanization_session(app)
    root = "/api/v1/nanobody-humanization"
    parents = [{"id": "one", "vhh": VHH}]
    preview = _request(app, "POST", root + "/prepare", json={"parents": parents}).json()
    payload = {"parents": parents, "preparation_digest": preview["preparation_digest"]}
    key = str(uuid4())
    admitted = _request(
        app,
        "POST",
        root + "/jobs",
        json=payload,
        headers={"Origin": ORIGIN, "Idempotency-Key": key},
    ).json()
    service = app.state.antibody_analysis
    started, release, lock = Event(), Event(), Lock()
    running = 0

    def block():
        nonlocal running
        with lock:
            running += 1
            if running == 8:
                started.set()
        assert release.wait(10)

    async def check():
        holders = [asyncio.create_task(service.run(block)) for _ in range(8)]
        try:
            assert await asyncio.to_thread(started.wait, 2)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url=ORIGIN,
                headers={"Origin": ORIGIN},
            ) as client:
                for path, body in (
                    (
                        "/api/v1/antibody-sequence-analysis/analyze",
                        {
                            "groups": [{"id": "one", "fasta": f">one\n{VHH}"}],
                        },
                    ),
                    ("/api/v1/antibody-sequence-analysis/sequence", {"sequence": VHH}),
                    (root + "/prepare", {"parents": parents}),
                    (root + "/jobs", payload),
                ):
                    response = await client.post(
                        path, json=body, headers={"Idempotency-Key": str(uuid4())}
                    )
                    assert response.status_code == 503, response.text
                    assert response.json()["code"] == "local_analysis_busy"
                    assert response.headers["retry-after"] == "1"
                replay = await client.post(
                    root + "/jobs", json=payload, headers={"Idempotency-Key": key}
                )
                assert replay.status_code == 202
                assert replay.json()["job_id"] == admitted["job_id"]
                assert len((await client.get("/api/v1/jobs")).json()["jobs"]) == 1
                assert app.state.remote_execution.preflights == 1
        finally:
            release.set()
            await asyncio.gather(*holders)
        # Completion releases admission, so an explicit retry can do local work.
        assert await service.run(lambda: 42) == 42
        await service.shutdown()

    asyncio.run(check())
