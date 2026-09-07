"""Real execution-kernel checks with fake provider calls, never live inference."""

import hashlib
import pickle
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
    orchestrator,
)
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.workflow.humanization.artifacts import json_output
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

    def archive(name, frame):
        stream = BytesIO()
        content = frame.write_csv().encode()
        with tarfile.open(fileobj=stream, mode="w") as bundle:
            member = tarfile.TarInfo(f"result/{name}")
            member.size = len(content)
            bundle.addfile(member, BytesIO(content))
            if name == "summary.csv":
                detail = BytesIO()
                frame.write_parquet(detail)
                member = tarfile.TarInfo("result/detail_scores.parquet")
                member.size = len(detail.getvalue())
                bundle.addfile(member, BytesIO(detail.getvalue()))
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
                result = archive(
                    "humanized.csv",
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
                                    "numbering_scheme": "imgt",
                                    "cdr_definition": "imgt",
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
                result = archive("summary.csv", pl.DataFrame([row]))
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
        selection = next(
            output for output in publication.outputs if output.name == "selection"
        )
        table = pl.read_csv(tmp_path / selection.storage.path)
        assert table.height == 2
        assert table["panel_order"].to_list() == [None, None if fail_first else 1]
        assert table["quality_tier"].to_list() == [None, None if fail_first else 1]
        assert table["evaluation_complete"].to_list() == [not fail_first] * 2
        assert table["generating_methods"].to_list() == ["pabnativ2", "humatch;sapiens"]
        assert (
            table["sapiens_vh_mean_probability_delta"].to_list()
            == [None if fail_first else 0.0] * 2
        )
        assert len(driver.results) == 12
        directory = next(
            output
            for output in publication.outputs
            if output.name == "humanization_results"
        )
        root = tmp_path / directory.storage.path
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        assert manifest["ranking_policy"]["version"] == "1"
        assert (root / "selection.csv").read_bytes() == (
            tmp_path / selection.storage.path
        ).read_bytes()
        assert (
            pl.read_parquet(root / "selection.parquet")["panel_order"].to_list()
            == table["panel_order"].to_list()
        )
        assert manifest["candidate_count"] == 2
        for entry in manifest["files"]:
            content = (root / entry["path"]).read_bytes()
            assert len(content) == entry["size_bytes"]
            assert hashlib.sha256(content).hexdigest() == entry["content_sha256"]
        assert len((root / "candidates.fasta").read_text().splitlines()) == 8
        assert pl.read_parquet(root / "humatch_detail_scores.parquet").height == 2
        graph = build_humanization_workflow(b"id,vh,vl\na,ACD,EFG\n")
        store.write_coordinator_plan(
            pickle.dumps(
                orchestrator.ExecutionCoordinatorPlan(
                    graph=graph,
                    workload_run_key="test",
                    max_active_provider_calls=2,
                    max_active_gpu_provider_calls=1,
                )
            )
        )
    finally:
        runtime.close()

    monkeypatch.setattr(orchestrator, "OUT_VOLUME", Volume())
    monkeypatch.setattr(orchestrator, "OUT_VOLUME_NAME", "workflow")
    monkeypatch.setattr(orchestrator.CONF, "output_volume_mountpoint", str(tmp_path))
    raw_cls = orchestrator.ExecutionCoordinator._get_user_cls()
    successor = raw_cls()
    successor.execution_run_id = str(UUID(int=3))
    successor.deployment_environment = "main"
    successor.deployment_name = "HumanizationWorkflow"
    successor.deployment_version = 1
    successor.development = False
    raw_cls.enter._get_raw_f()(successor)
    successor._modal_driver = lambda: driver
    driver.fail = False
    try:
        with pytest.raises(ValueError):
            raw_cls.prepare_restart_from._get_raw_f()(
                successor,
                predecessor_execution_run_id=str(run_id),
                workload_run_key="test",
                graph=build_humanization_workflow(
                    b"id,vh,vl\na,ACD,EFG\n", HumanizationSettings(sapiens_iterations=2)
                ),
            )
        raw_cls.prepare_restart_from._get_raw_f()(
            successor,
            predecessor_execution_run_id=str(run_id),
            workload_run_key="test",
            graph=graph,
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=1,
        )
        result = raw_cls.drive_prepared._get_raw_f()(successor)
        assert result.status == AppRunStatus.SUCCEEDED
        assert driver.operations[12:] == (["sapiens_score"] * 2 if fail_first else [])
    finally:
        raw_cls.exit._get_raw_f()(successor)
