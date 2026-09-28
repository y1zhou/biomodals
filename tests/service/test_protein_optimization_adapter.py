"""The actual adapter restores content-bound CSVs without repeating model work."""

import asyncio
import shutil
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pytest
from protein_optimization_fixture import result_publication

from biomodals.service.artifacts import ArtifactCache, ArtifactIntegrityError
from biomodals.service.pending import PendingRequestStore
from biomodals.service.protein_optimization import modal as adapter_module
from biomodals.service.protein_optimization.results import query_candidates
from biomodals.workflow.protein_optimization.design import OptimizationDesign
from biomodals.workflow.protein_optimization.execution import (
    OptimizationExecutionRequest,
)


@pytest.mark.parametrize("tamper", [None, "digest", "chain", "count"])
def test_actual_adapter_verifies_and_restores_csv(tmp_path, monkeypatch, tamper):
    """Use real ridge fixture and actual adapter; substitute only remote file transport."""

    async def scenario():
        request = OptimizationExecutionRequest(
            run_name="fixture",
            design=OptimizationDesign(
                measurements_csv="mutations,label\n,0\nA:A1V,1\nB:G1A,2\n",
                parental_fasta=">A\nAA\n>B\nGG\n",
            ),
        )
        job_id = uuid4()
        pending = PendingRequestStore(tmp_path)
        pending.initialize()
        pending.put(job_id, request.to_bytes())
        source, manifest = result_publication(tmp_path / "remote", job_id, request)
        if tamper == "digest":
            manifest = manifest.model_copy(update={"csv_sha256": "f" * 64})
        elif tamper == "chain":
            manifest = manifest.model_copy(
                update={"chain_columns": {"C": "sequence_A"}}
            )
        elif tamper == "count":
            manifest = manifest.model_copy(update={"candidate_count": 2})

        async def read(_volume, path, *, max_bytes):
            assert (
                path
                == f"workflow-runs/{job_id}/nodes/publish/result/protein_optimization/manifest.json"
            )
            return manifest.model_dump_json().encode()

        def download(_volume, pairs, *, concurrency):
            for remote, destination in pairs:
                assert remote.endswith("/candidates.csv")
                shutil.copyfile(source, destination)

        adapter = adapter_module.ProteinOptimizationAdapter(pending)
        monkeypatch.setattr(adapter, "_volume", lambda _job: object())
        monkeypatch.setattr(adapter_module, "read_modal_volume_file", read)
        monkeypatch.setattr(adapter_module, "download_modal_volume_files", download)
        cache = ArtifactCache(tmp_path / "cache")
        job = SimpleNamespace(job_id=job_id)
        try:
            if tamper:
                with pytest.raises(ArtifactIntegrityError):
                    await adapter.prepare_result(job, cache, completed_at=1)
                return
            first = await adapter.prepare_result(job, cache, completed_at=1)
            assert first.media_type == "text/csv"
            assert first.archive_schema == "protein_optimization/1"
            page = query_candidates(cache.derived_path(str(job_id)), first.sha256)
            assert page.summary.candidate_count == 1
            assert page.summary.chain_columns == {"A": "sequence_A", "B": "sequence_B"}
            assert page.rows[0]["sequence_A"] == "VA"
            assert page.rows[0]["sequence_B"] == "AG"
            assert (
                page.rows[0]["predicted_label"]
                == pl.read_csv(source)["predicted_label"][0]
            )
            await cache.discard_async(str(job_id))
            restored = await adapter.prepare_result(job, cache, completed_at=20)
            assert restored == first
        finally:
            await cache.shutdown()

    asyncio.run(scenario())
