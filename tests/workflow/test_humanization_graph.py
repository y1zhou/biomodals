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


def test_generation_methods_are_independent_nodes_with_a_union_barrier():
    """All methods can start together, while collection waits for every method."""
    request = HumanizationExecutionRequest(
        run_name="test",
        pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),),
    )
    plan = request.execution_plan
    generation = plan.nodes[:4]
    assert {node.node_key for node in generation} == {
        "generate_sapiens",
        "generate_humatch",
        "generate_pabnativ2",
        "generate_hudiff_ab",
    }
    assert all(node.dependencies == () for node in generation)
    union = next(node for node in plan.nodes if node.node_key == "union")
    assert {edge.node_key for edge in union.dependencies} == {
        node.node_key for node in generation
    }
    assert all(edge.accept_partial for edge in union.dependencies)


@pytest.mark.parametrize("pab_success", [False, True])
def test_generation_failures_preserve_baselines_and_successful_replicas(
    tmp_path, pab_success
):
    """A failed replica cannot discard another replica's successful design."""
    settings = HumanizationSettings(pabnativ2_num_seeds=3 if pab_success else 1)

    class Driver:
        def __init__(self):
            self.calls = []
            self.kwargs = []

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            self.calls.append(operation)
            self.kwargs.append(kwargs)
            return f"call-{len(self.calls)}"

        def observe(self, provider_call_handle_id):
            index = int(provider_call_handle_id.split("-")[1]) - 1
            kwargs = self.kwargs[index]
            if (
                pab_success
                and self.calls[index] == "pabnativ2_humanize_pair"
                and kwargs["seed"] == settings.pabnativ2_seeds[0]
            ):
                return ProviderCallObservation(
                    ProviderCallObservationKind.SUCCEEDED,
                    result=AppRunResult(
                        status=AppRunStatus.SUCCEEDED,
                        outputs=[
                            json_output(
                                "native",
                                {
                                    "schema_version": 1,
                                    "pair_result": {
                                        "humanized": {**kwargs["pair"], "vh": "ACH"},
                                        "pair_seed": kwargs["seed"],
                                    },
                                },
                            )
                        ],
                    ),
                )
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
        graph=build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n", settings),
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
        assert len(candidates) == 1 + int(pab_success)
        assert candidates[0]["is_parent"] is True
        assert candidates[0]["vh"] == "ACD"
        if pab_success:
            assert candidates[1]["vh"] == "ACH"
            assert candidates[1]["origins"][0]["seed"] == settings.pabnativ2_seeds[0]
        generation_errors = next(
            output
            for output in publication.outputs
            if output.name == "generation_errors"
        )
        assert set(
            orjson.loads((tmp_path / generation_errors.storage.path).read_bytes())
        ) == {
            "sapiens-0000",
            "humatch-0000",
            *(
                {"pabnativ2-0000-seed-01", "pabnativ2-0000-seed-02"}
                if pab_success
                else {"pabnativ2-0000"}
            ),
            "hudiff_ab-0000",
        }
        final = store.artifacts.load_node_result("evaluate")
        root = tmp_path / next(
            o.storage.path for o in final.outputs if o.name == "humanization_results"
        )
        ledger = pl.read_parquet(root / "generation.parquet")
        failures = ledger.filter(pl.col("outcome") == "failed")
        assert failures.height == (5 if pab_success else 4)
        assert failures["reason"].str.contains("model unavailable").all()
        assert failures["candidate_id"].null_count() == failures.height
        assert failures["seed"].null_count() == failures.height
        assert sorted(
            failures.filter(pl.col("method") == "pabnativ2")["root_seed"].to_list()
        ) == sorted(
            settings.pabnativ2_seeds[1:] if pab_success else settings.pabnativ2_seeds
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("num_seeds", [1, 3])
def test_full_graph_joins_successful_native_results_into_sortable_table(
    tmp_path, monkeypatch, fail_first, num_seeds
):
    """Exercise actual artifact materialization and candidate/evaluator joins."""

    def generation_archive(frame):
        stream = BytesIO()
        content = frame.write_csv().encode()
        with tarfile.open(fileobj=stream, mode="w") as bundle:
            member = tarfile.TarInfo("result/humanized.csv")
            member.size = len(content)
            bundle.addfile(member, BytesIO(content))
            designs = (
                pl
                .concat([
                    frame.with_columns(pl.lit(i).alias("iteration")) for i in (1, 2)
                ])
                .write_csv()
                .encode()
            )
            member = tarfile.TarInfo("result/iteration_designs.csv")
            member.size = len(designs)
            bundle.addfile(member, BytesIO(designs))
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
                                    "pair_seed": kwargs["seed"],
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
                                "pair_result": {
                                    "candidates": [],
                                    "attempts": [],
                                    "pair_seed": 42,
                                },
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
    settings = HumanizationSettings(sapiens_iterations=2, pabnativ2_num_seeds=num_seeds)
    expected_calls = 12 + num_seeds - 1
    runtime = ExecutionGraphRuntime(
        graph=build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n", settings),
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
        assert len(driver.results) == expected_calls
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        assert manifest["generation_seeds"]["pabnativ2"] == list(
            settings.pabnativ2_seeds
        )
        assert manifest["ranking_policy"]["version"] == "2"
        assert manifest["protocols"]["pabnativ2"]["rasa_structure_count"] == 4
        assert manifest["protocols"]["pabnativ2"]["pssm_frequency_cutoff"] == 0.01
        assert manifest["protocols"]["pabnativ2"]["nativeness_weight"] == 10.0
        assert manifest["protocols"]["pabnativ2"]["pairing_weight"] == 1.0
        assert manifest["schema_version"] == 3
        generation = pl.read_parquet(root / "generation.parquet")
        baseline = generation.filter(pl.col("candidate_id") == table["candidate_id"][0])
        assert baseline["method"].to_list() == ["pabnativ2"] * num_seeds
        assert baseline["outcome"].to_list() == ["no_op"] * num_seeds
        assert sorted(baseline["seed"].to_list()) == sorted(settings.pabnativ2_seeds)
        sapiens = generation.filter(pl.col("method") == "sapiens")
        assert sapiens["source_id"].to_list() == ["a__iteration_1", "a__iteration_2"]
        assert sapiens["iteration"].to_list() == [1, 2]
        assert sapiens["candidate_id"].to_list() == [table["candidate_id"][1]] * 2
        assert generation.filter(pl.col("method") == "hudiff_ab")[
            "outcome"
        ].to_list() == ["no_candidates"]
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
        request = HumanizationExecutionRequest(
            run_name="test",
            pairs=(AntibodyPair(id="a", vh="ACD", vl="EFG"),),
            settings=settings,
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
                    settings=HumanizationSettings(sapiens_iterations=3),
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
        assert driver.operations[expected_calls:] == (
            ["sapiens_score"] * 2 if fail_first else []
        )
        publication = successor.result()
        directory = next(
            output
            for output in publication.outputs
            if output.name == "humanization_results"
        )
        assert (tmp_path / directory.storage.path / "selection.csv").is_file()
    finally:
        successor.close()
