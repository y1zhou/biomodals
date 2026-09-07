"""Real execution-kernel checks with fake provider calls, never live inference."""

import hashlib
import tarfile
from io import BytesIO
from uuid import UUID

import orjson
import polars as pl
import pytest
import zstandard

from biomodals.execution import DeploymentIdentity, GraphExecutionRunStore
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.workflow.humanization.artifacts import json_output
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.execution import (
    HumanizationExecutionCoordinator,
    HumanizationExecutionRequest,
    persist_execution_request,
)
from biomodals.workflow.humanization.scoring import scoring_result
from biomodals.workflow.humanization.settings import HumanizationSettings
from biomodals.workflow.humanization.tables import SCORE_COLUMNS
from biomodals.workflow.humanization.workflow import build_humanization_workflow


def test_all_generator_failures_still_publish_parental_union(tmp_path):
    """A local baseline makes all-generator failure a collectable partial result."""

    class Driver:
        def __init__(self):
            self.calls = []

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.calls.append(operation)
            return f"call-{len(self.calls)}"

        def observe(self, provider_call_handle_id):
            return ProviderCallObservation(
                ProviderCallObservationKind.FAILED, message="model unavailable"
            )

        def cancel(self, provider_call_handle_id):
            pass

    class Volume:
        def commit(self):
            pass

        def reload(self):
            pass

    run_id = UUID(int=1)
    store = GraphExecutionRunStore(tmp_path, run_id)
    driver = Driver()
    runtime = ExecutionGraphRuntime(
        graph=build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n"),
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "HumanizationWorkflow", 1),
        volume_root=tmp_path,
        artifact_volume_name="workflow",
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=Volume(), store=store),
        store=store,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        poll_interval_seconds=0,
    )
    try:
        result = runtime.run(workload_run_key="test")
        assert result.status == AppRunStatus.PARTIAL
        assert set(driver.calls) == {
            "sapiens_humanize",
            "humatch_humanize",
            "pabnativ2_humanize_pair",
            "hudiff_ab_humanize_pair",
            "sapiens_score",
            "humatch_score",
            "pabnativ2_score",
            "annotate_humanization_candidate",
        }
        publication = store.artifacts.load_node_result("union")
        assert publication is not None
        candidates = orjson.loads(
            (tmp_path / publication.outputs[0].storage.path).read_bytes()
        )
        assert len(candidates) == 1
        assert candidates[0]["is_parent"] is True
        assert candidates[0]["vh"] == "ACD"
    finally:
        runtime.close()


@pytest.mark.parametrize("fail_first", [False, True])
def test_full_graph_joins_successful_native_results_into_sortable_table(
    tmp_path, monkeypatch, fail_first
):
    """Exercise actual artifact materialization and candidate/evaluator joins."""

    def generation_archive(frame):
        stream = BytesIO()
        content = frame.write_csv().encode()
        with tarfile.open(fileobj=stream, mode="w") as bundle:
            member = tarfile.TarInfo("result/humanized.csv")
            member.size = len(content)
            bundle.addfile(member, BytesIO(content))
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="native",
                    kind=ArtifactKind.ARCHIVE,
                    storage=InlineBytes(
                        data=zstandard.ZstdCompressor().compress(stream.getvalue()),
                        filename="native.tar.zst",
                        media_type="application/zstd",
                    ),
                )
            ],
        )

    class Driver:
        def __init__(self):
            self.results = {}
            self.operations = []
            self.fail = fail_first

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.operations.append(operation)
            if operation in {"sapiens_humanize", "humatch_humanize"}:
                result = generation_archive(
                    pl.read_csv(BytesIO(kwargs["csv_bytes"])).with_columns(
                        pl.lit("ACH").alias("vh")
                    ),
                )
            elif operation == "pabnativ2_humanize_pair":
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        json_output(
                            "native",
                            {
                                "schema_version": 1,
                                "pair_result": {
                                    "humanized": kwargs["pair"],
                                    "pair_seed": 0,
                                },
                            },
                        )
                    ],
                )
            elif operation == "hudiff_ab_humanize_pair":
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        json_output(
                            "native",
                            {
                                "schema_version": 1,
                                "pair_result": {"candidates": [], "pair_seed": 42},
                            },
                        )
                    ],
                )
            elif operation == "annotate_humanization_candidate":
                candidate = kwargs["candidate"]
                changed = candidate["vh"] != kwargs["parent"]["vh"]
                result = AppRunResult(
                    status=AppRunStatus.SUCCEEDED,
                    outputs=[
                        json_output(
                            "annotation",
                            {
                                "parent_id": candidate["parent_id"],
                                "candidate_id": candidate["candidate_id"],
                                "cdr_preservation": "preserved",
                                "cdr_mutations": 0,
                                "vh_mutations": int(changed),
                                "vl_mutations": 0,
                            },
                        ),
                        json_output(
                            "imgt_mutations",
                            [
                                {
                                    "parent_id": candidate["parent_id"],
                                    "candidate_id": candidate["candidate_id"],
                                    "chain": "vh",
                                    "position": 3,
                                    "insertion_code": "",
                                    "parent_residue": "D",
                                    "candidate_residue": "H",
                                    "region": "framework",
                                    "change_type": "substitution",
                                }
                            ]
                            if changed
                            else [],
                        ),
                    ],
                )
            else:
                method = operation.removesuffix("_score")
                frame = pl.read_csv(BytesIO(kwargs["csv_bytes"]), infer_schema=False)
                row = {
                    "id": frame["id"][0],
                    **dict.fromkeys(SCORE_COLUMNS[method], 0.8),
                }
                if method == "humatch":
                    row.update(
                        vh_target_family="hv1",
                        vl_target_family="kv1",
                        vh_best_family="hv1",
                        vl_best_family="kv1",
                    )
                summary = pl.DataFrame([row])
                result = scoring_result(
                    method,
                    frame,
                    summary,
                    {"detail_scores": summary},
                    {"runtime": "test", "candidate_id": row["id"]},
                )
            call_id = f"call-{len(self.results)}"
            self.results[call_id] = result
            return call_id

        def observe(self, provider_call_handle_id):
            if (
                self.fail
                and self.operations[int(provider_call_handle_id.split("-")[1])]
                == "sapiens_score"
            ):
                return ProviderCallObservation(
                    ProviderCallObservationKind.FAILED,
                    message="temporary scorer failure",
                )
            return ProviderCallObservation(
                ProviderCallObservationKind.SUCCEEDED,
                result=self.results[provider_call_handle_id],
            )

        def cancel(self, provider_call_handle_id):
            pass

    class Volume:
        def commit(self):
            pass

        def reload(self):
            pass

    run_id = UUID(int=2)
    store = GraphExecutionRunStore(tmp_path, run_id)
    driver = Driver()
    runtime = ExecutionGraphRuntime(
        graph=build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n"),
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "HumanizationWorkflow", 1),
        volume_root=tmp_path,
        artifact_volume_name="workflow",
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=Volume(), store=store),
        store=store,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        poll_interval_seconds=0,
    )
    try:
        result = runtime.run(workload_run_key="test")
        assert result.status == (
            AppRunStatus.PARTIAL if fail_first else AppRunStatus.SUCCEEDED
        ), result
        publication = store.artifacts.load_node_result("evaluate")
        directory = next(
            output
            for output in publication.outputs
            if output.name == "humanization_results"
        )
        root = tmp_path / directory.storage.path
        table = pl.read_csv(root / "selection.csv")
        assert table.height == 2
        assert table["panel_order"].to_list() == [None, None if fail_first else 1]
        assert table["quality_tier"].to_list() == [None, None if fail_first else 1]
        assert table["evaluation_complete"].to_list() == [not fail_first] * 2
        assert table["generating_methods"].to_list() == [None, "humatch;sapiens"]
        assert (
            table["sapiens_vh_mean_probability_delta"].to_list()
            == [None if fail_first else 0.0] * 2
        )
        assert len(driver.results) == 12
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        assert manifest["ranking_policy"]["version"] == "1"
        assert manifest["protocols"]["pabnativ2"]["rasa_structure_count"] == 4
        assert manifest["protocols"]["pabnativ2"]["pssm_frequency_cutoff"] == 0.01
        assert manifest["protocols"]["pabnativ2"]["nativeness_weight"] == 10.0
        assert manifest["protocols"]["pabnativ2"]["pairing_weight"] == 1.0
        assert not list((root / "native").glob("*.parquet"))
        assert len(list((root / "native").iterdir())) == 4
        assert len(manifest["scoring_publications"]) == (4 if fail_first else 6)
        assert all(
            entry["manifest"]["scientific_identity"]["runtime"] == "test"
            for entry in manifest["scoring_publications"]
        )
        assert manifest["schema_version"] == 2
        assert len(manifest["candidate_provenance"]) == 2
        baseline = next(
            item
            for item in manifest["candidate_provenance"]
            if item["candidate_id"] == table["candidate_id"][0]
        )
        assert baseline["origins"][0]["method"] == "pabnativ2"
        assert manifest["candidate_count"] == 2
        for entry in manifest["files"]:
            content = (root / entry["path"]).read_bytes()
            assert len(content) == entry["size_bytes"]
            assert hashlib.sha256(content).hexdigest() == entry["content_sha256"]
        assert (
            pl.read_parquet(root / "scores/humatch_detail_scores.parquet").height == 2
        )
        # Original scorer input/summary and detail values survive consolidation.
        from polars.testing import assert_frame_equal

        from biomodals.workflow.humanization.artifacts import archive_members

        for call_id, original in driver.results.items():
            operation = driver.operations[int(call_id.split("-")[1])]
            if not operation.endswith("_score") or (
                fail_first and operation == "sapiens_score"
            ):
                continue
            method = operation.removesuffix("_score")
            members = archive_members(original.outputs[0].storage.data)
            inputs = pl.read_csv(BytesIO(members["input.csv"]), infer_schema=False)
            candidate_id = inputs["id"][0]
            selected = table.filter(pl.col("candidate_id") == candidate_id)
            assert_frame_equal(
                inputs, selected.select(pl.col("candidate_id").alias("id"), "vh", "vl")
            )
            summary = pl.read_csv(BytesIO(members["summary.csv"]))
            assert_frame_equal(
                summary,
                selected.select(
                    pl.col("candidate_id").alias("id"),
                    *(
                        pl.col(f"{method}_{name}").alias(name)
                        for name in summary.columns
                        if name != "id"
                    ),
                ),
            )
            detail = pl.read_parquet(BytesIO(members["detail_scores.parquet"]))
            assert_frame_equal(
                detail,
                pl
                .read_parquet(root / f"scores/{method}_detail_scores.parquet")
                .filter(pl.col("candidate_id") == candidate_id)
                .drop("parent_id", "candidate_id"),
            )
            recorded = next(
                entry
                for entry in manifest["scoring_publications"]
                if entry["task_key"] == f"{method}-{candidate_id}"
            )
            assert recorded["manifest"] == orjson.loads(members["manifest.json"])
        request = HumanizationExecutionRequest(
            run_name="test",
            pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),),
            max_active_provider_calls=2,
            max_active_gpu_provider_calls=1,
        )
        persist_execution_request(tmp_path, run_id, request)
    finally:
        runtime.close()

    successor = HumanizationExecutionCoordinator(
        execution_run_id=UUID(int=3),
        deployment=DeploymentIdentity("main", "HumanizationWorkflow", 1),
        volume_root=tmp_path,
        output_volume=Volume(),
        output_volume_name="workflow",
        provider_driver=driver,
        poll_interval_seconds=0,
    )
    driver.fail = False
    try:
        with pytest.raises(ValueError):
            successor.prepare_restart(
                predecessor_execution_run_id=run_id,
                predecessor_deployment=None,
                candidate_request=HumanizationExecutionRequest(
                    run_name="test",
                    pairs=request.pairs,
                    settings=HumanizationSettings(sapiens_iterations=2),
                ),
            )
        successor.prepare_restart(
            predecessor_execution_run_id=run_id,
            predecessor_deployment=None,
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=1,
        )
        result = successor.drive_prepared()
        assert result.run.status.value == "succeeded"
        assert driver.operations[12:] == (["sapiens_score"] * 2 if fail_first else [])
        publication = successor.result()
        directory = next(
            output
            for output in publication.outputs
            if output.name == "humanization_results"
        )
        assert (tmp_path / directory.storage.path / "selection.csv").is_file()
    finally:
        successor.close()
