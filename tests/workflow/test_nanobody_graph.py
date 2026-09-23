"""Exercise the real scheduler with deterministic native-boundary substitutes."""

# ruff: noqa: D103

import hashlib
from uuid import UUID

import orjson
import polars as pl
import pytest

from biomodals.app.design.abnativ2_vhh.models import RUNTIME_IDENTITY, SOURCE_COMMIT
from biomodals.app.design.abnativ2_vhh.sampling import sample_combinations
from biomodals.app.design.abnativ2_vhh.worker import _tables
from biomodals.execution import DeploymentIdentity, GraphExecutionRunStore
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow.nanobody_humanization.annotation import (
    annotate_nanobody_candidates,
)
from biomodals.workflow.nanobody_humanization.artifacts import json_output
from biomodals.workflow.nanobody_humanization.preparation import VHInput, prepare_vh
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings
from biomodals.workflow.nanobody_humanization.workflow import build_nanobody_graph

SEQUENCE = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSV"
    "KGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"
)


@pytest.mark.parametrize(
    "failure",
    [None, "abnativ2_vhh", "both", "score", "score_chunk", "scalar", "annotation"],
)
@pytest.mark.parametrize("exploration_count", [0, 600])
def test_native_union_scoring_publication_and_partial_outcomes(
    tmp_path, failure, exploration_count
):
    parent = prepare_vh(VHInput(id="a", vhh=SEQUENCE))
    index = next(
        i for i in range(len(parent.sequence)) if i not in parent.protected_indices
    )
    residue = "A" if parent.sequence[index] != "A" else "E"
    candidate = parent.sequence[:index] + residue + parent.sequence[index + 1 :]
    mutable = [
        i for i in range(len(parent.sequence)) if i not in parent.protected_indices
    ][:10]
    alternatives = sample_combinations(
        parent.sequence,
        [aa + "AE" if i in mutable else aa for i, aa in enumerate(parent.sequence)],
        budget=600,
        seed=0,
    ).sequences
    if exploration_count:
        candidate = alternatives[0]
    settings = NanobodySettings(
        hudiff_nb_candidate_count=2,
        abnativ2_explore=bool(exploration_count),
        abnativ2_candidate_budget=exploration_count or 1000,
    )
    graph = build_nanobody_graph((parent,), settings)

    class Driver:
        def __init__(self):
            self.calls = []

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.calls.append((operation, kwargs))
            return f"call-{len(self.calls)}"

        def observe(self, provider_call_handle_id):
            operation, kwargs = self.calls[
                int(provider_call_handle_id.split("-")[1]) - 1
            ]
            if (
                (
                    operation.endswith(("_humanize", "_generate"))
                    and (failure == "both" or operation.startswith(str(failure)))
                )
                or (failure == "score" and operation == "abnativ2_vhh_score")
                or (
                    failure == "score_chunk"
                    and operation == "abnativ2_vhh_score"
                    and kwargs["model_type"] == "VH2"
                    and any(
                        row["sequence"] == parent.sequence
                        for row in kwargs["sequences"]
                    )
                )
                or (
                    failure == "annotation"
                    and operation == "annotate_nanobody_candidates"
                )
            ):
                return ProviderCallObservation(
                    ProviderCallObservationKind.FAILED,
                    message="Unavailable native operation",
                )
            if operation.endswith(("_humanize", "_generate")):
                method = operation.removesuffix("_humanize").removesuffix("_generate")
                count = kwargs.get("candidate_count", exploration_count or 1)
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        json_output(
                            "generation",
                            {
                                "schema_version": 2 if method == "abnativ2_vhh" else 1,
                                "settings": kwargs.get("settings"),
                                "source_commit": SOURCE_COMMIT,
                                "runtime_identity": RUNTIME_IDENTITY,
                                "search": {
                                    "possible_candidates": str(exploration_count),
                                    "evaluated_candidates": exploration_count,
                                    "accepted_candidates": exploration_count,
                                    "coverage": "complete",
                                }
                                if method == "abnativ2_vhh" and exploration_count
                                else None,
                                "method": method,
                                "input_sequence": parent.sequence,
                                "seed": kwargs.get("seed"),
                                "attempts": [
                                    {
                                        "attempt_index": attempt,
                                        "sequence": candidate
                                        if method == "hudiff_nb"
                                        else alternatives[attempt - 1]
                                        if exploration_count
                                        else parent.sequence,
                                        "error": None,
                                    }
                                    for attempt in range(1, count + 1)
                                ],
                            },
                        )
                    ],
                )
            elif operation == "abnativ2_vhh_score":
                rows = [
                    {
                        "candidate_id": seq["id"],
                        "sequence": seq["sequence"],
                        "score": None
                        if failure == "scalar" and seq["sequence"] == candidate
                        else 0.8,
                        "error": "Missing native score"
                        if failure == "scalar" and seq["sequence"] == candidate
                        else None,
                    }
                    for seq in kwargs["sequences"]
                ]
                frame = pl.DataFrame(
                    rows,
                    schema={
                        "candidate_id": pl.String,
                        "sequence": pl.String,
                        "score": pl.Float64,
                        "error": pl.String,
                    },
                )
                native = frame.select(pl.col("candidate_id").alias("seq_id"), "score")
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        _tables("scores", {"summary": frame, "native_scores": native})
                    ],
                )
            else:
                result = annotate_nanobody_candidates(**kwargs)
            return ProviderCallObservation(
                ProviderCallObservationKind.SUCCEEDED, result=result
            )

        def cancel(self, provider_call_handle_id):
            pass

    class Volume:
        def reload(self):
            pass

        def commit(self):
            pass

    run_id = UUID(int=1)
    store = GraphExecutionRunStore(tmp_path, run_id)
    driver = Driver()
    runtime = ExecutionGraphRuntime(
        graph=graph,
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "NanobodyHumanizationWorkflow", 1),
        volume_root=tmp_path,
        artifact_volume_name="workflow",
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=Volume(), store=store),
        store=store,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        poll_interval_seconds=0,
    )
    try:
        result = runtime.run(workload_run_key="native-boundary-test")
        if failure == "both":
            assert result.status == AppRunStatus.FAILED
            assert {call[0] for call in driver.calls} == {
                "abnativ2_vhh_generate",
                "hudiff_nb_humanize",
            }
            return
        assert result.status == (
            AppRunStatus.PARTIAL if failure else AppRunStatus.SUCCEEDED
        ), result
        bundle = next(
            output for output in result.outputs if output.name == "nanobody_results"
        )
        root = tmp_path / bundle.storage.path
        table = pl.read_csv(root / "selection.csv")
        designs = (
            exploration_count if exploration_count and failure != "abnativ2_vhh" else 1
        )
        assert table.height == 1 + designs
        assert table["is_parent"].to_list() == [True, *([False] * designs)]
        shared = table.filter(pl.col("vh") == candidate)
        assert shared["generating_methods"].item() == (
            "abnativ2_vhh;hudiff_nb"
            if exploration_count and failure != "abnativ2_vhh"
            else "hudiff_nb"
        )
        assert table["panel_order"].count() == (
            0
            if failure == "score"
            else designs - min(designs, 127)
            if failure == "score_chunk"
            else designs - int(failure == "scalar")
        )
        if failure == "score_chunk":
            assert table["abnativ2_vh_nativeness_error"].is_not_null().sum() == min(
                128, table.height
            )
            assert table["abnativ2_vhh_nativeness"].count() == table.height
        generation = pl.read_parquet(root / "generation.parquet")
        assert generation.height == 2 + (
            exploration_count if exploration_count and failure != "abnativ2_vhh" else 1
        )
        assert (
            generation.filter(pl.col("method") == "hudiff_nb")[
                "candidate_id"
            ].n_unique()
            == 1
        )
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        assert manifest["status"] == result.status.value
        assert manifest["parameters"] == settings.model_dump()
        assert manifest["candidate_count"] == table.height
        if exploration_count and failure != "abnativ2_vhh":
            assert (
                manifest["abnativ2_searches"][0]["accepted_candidates"]
                == exploration_count
            )
            assert (
                manifest["abnativ2_searches"][0]["sampling_seed"]
                == settings.abnativ_parameters(
                    parent.id, parent.sequence, parent.protected_indices
                ).sampling_seed
            )
        for file in manifest["files"]:
            content = (root / file["path"]).read_bytes()
            assert len(content) == file["size_bytes"]
            assert hashlib.sha256(content).hexdigest() == file["content_sha256"]
        score_calls = [
            kwargs for name, kwargs in driver.calls if name == "abnativ2_vhh_score"
        ]
        assert {call["model_type"] for call in score_calls} == {"VH2", "VHH2"}
        assert all(len(call["sequences"]) <= 128 for call in score_calls)
        for model in ("VH2", "VHH2"):
            records = [
                row
                for call in score_calls
                if call["model_type"] == model
                for row in call["sequences"]
            ]
            assert len(records) == table.height
            assert {row["sequence"] for row in records} == set(table["vh"])
    finally:
        store.close()
