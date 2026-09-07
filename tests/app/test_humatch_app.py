"""Contracts for the Humatch humanization app."""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import numpy as np
import polars as pl
import pytest

from biomodals.app.design.humatch import app as humatch_app
from biomodals.app.design.humatch.execution import (
    _RESULT_FILE,
    COLLECT_NODE,
    HUMANIZE_NODE,
    HUMATCH_BATCH_SIZE,
    HumatchExecutionRequest,
    _HumatchHumanizeNode,
    humatch_execution_graph,
    result_from_overview,
)
from biomodals.app.design.humatch.models import ASSETS
from biomodals.execution import DeploymentIdentity, GraphExecutionRunStore, RunStatus
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.execution.nodes import NodeRunContext, TaskDefinition
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

VALID_CSV = (
    b"id,vh,vl\n"
    b"pair-1,QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMG,"
    b"DIQMTQSPSSLSASVGDRVTITCRASQSI\n"
)


def _request() -> HumatchExecutionRequest:
    return HumatchExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        vh_target_family="auto",
        vl_target_family="auto",
        germline_likeness_target=0.4,
        vh_classifier_target=0.95,
        vl_classifier_target=0.95,
        pair_classifier_target=0.95,
        max_edits=60,
        mutate_cdrs=False,
        fixed_vh_positions="",
        fixed_vl_positions="",
        app_version="humatch-source",
        asset_record="asset-record",
        runtime_identity="runtime-identity",
    )


def test_modal_group_tag_does_not_depend_on_source_path() -> None:
    assert humatch_app.CONF.tags == {"group": "design"}


def test_parse_humatch_csv_accepts_complete_unique_pairs() -> None:
    frame = humatch_app.parse_humatch_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.to_dicts()[0]["id"] == "pair-1"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "non-canonical or lowercase"),
    ),
)
def test_parse_humatch_csv_rejects_invalid_batches(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        humatch_app.parse_humatch_csv(content)


def test_fixed_positions_are_normalized_without_exposing_upstream_spacing() -> None:
    assert humatch_app._normalize_fixed_positions("27, 33a,120", "fixed") == (
        "27 ",
        "33A",
        "120 ",
    )
    with pytest.raises(ValueError, match="duplicate"):
        humatch_app._normalize_fixed_positions("27,27", "fixed")


def test_execution_request_roundtrips_and_plans_fixed_cpu_batches() -> None:
    request = _request()

    assert HumatchExecutionRequest.from_bytes(request.to_bytes()) == request
    assert [node.node_key for node in request.execution_plan.nodes] == [
        HUMANIZE_NODE,
        COLLECT_NODE,
    ]
    node = _HumatchHumanizeNode(request)
    tasks = node.discover_remote_tasks(cast(NodeRunContext, None))
    call = node.prepare_remote_task_batch(cast(NodeRunContext, None), tasks)
    assert call.function_name == "humatch_humanize_batch"
    assert call.uses_gpu is False
    assert call.max_tasks_per_call == HUMATCH_BATCH_SIZE
    assert call.kwargs["pairs"] == [
        {
            "id": "pair-1",
            "vh": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGMNWVRQAPGQGLEWMG",
            "vl": "DIQMTQSPSSLSASVGDRVTITCRASQSI",
        }
    ]
    assert call.kwargs["max_edits"] == 60


def test_execution_request_allows_outer_cpu_call_limit() -> None:
    request = replace(_request(), max_active_provider_calls=17)

    assert request.max_active_provider_calls == 17
    with pytest.raises(ValueError, match="positive CPU"):
        replace(request, max_active_provider_calls=0)


def test_fixed_batch_rejects_more_than_six_tasks() -> None:
    task = TaskDefinition(
        task_key="pair",
        scientific_payload={},
        execution_payload={"id": "pair", "vh": "AAAA", "vl": "CCCC"},
    )
    with pytest.raises(ValueError, match="one to six"):
        _HumatchHumanizeNode(_request()).prepare_remote_task_batch(
            cast(NodeRunContext, None),
            tuple(replace(task, task_key=f"pair-{index}") for index in range(7)),
        )


def test_execution_request_rejects_boolean_edit_limit() -> None:
    with pytest.raises(ValueError, match="max_edits"):
        replace(_request(), max_edits=True)


@pytest.mark.parametrize(
    "run_name", ["../example", "/absolute/example", "a/b", "-example"]
)
def test_execution_request_rejects_unsafe_run_names(run_name: str) -> None:
    with pytest.raises(ValueError, match="safe filename component"):
        replace(_request(), run_name=run_name)


def test_worker_calls_one_pair_directly_and_multiple_pairs_in_threads(
    monkeypatch,
) -> None:
    executor_sizes: list[int] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            executor_sizes.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def map(self, function, records):
            return [function(record) for record in records]

    def fake_pair(pair, **_kwargs):
        return {
            "humanized": {"id": pair.identifier},
            "humanization_seconds": 1.0,
        }

    monkeypatch.setattr(humatch_app, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        humatch_app,
        "_load_upstream",
        lambda: {"canonical_numbering": ()},
    )
    monkeypatch.setattr(
        humatch_app,
        "_align_chain",
        lambda *, sequence, **_kwargs: sequence,
    )
    monkeypatch.setattr(humatch_app, "_load_models", lambda _upstream: ((1, 2, 3), 0.5))
    monkeypatch.setattr(humatch_app, "_humanize_pair", fake_pair)

    one = humatch_app._run_humatch_worker_batch(
        pairs=[{"id": "one", "vh": "AAAA", "vl": "CCCC"}]
    )
    many = humatch_app._run_humatch_worker_batch(
        pairs=[
            {"id": f"pair-{index}", "vh": "AAAA", "vl": "CCCC"} for index in range(6)
        ]
    )

    assert [result["humanized"]["id"] for result in one["pair_results"]] == ["one"]
    assert [result["humanized"]["id"] for result in many["pair_results"]] == [
        f"pair-{index}" for index in range(6)
    ]
    assert executor_sizes == [6]


def test_worker_aligns_all_pairs_before_loading_models(monkeypatch) -> None:
    events: list[str] = []

    def fake_align(*, identifier, chain_label, sequence, **_kwargs):
        events.append(f"align:{identifier}:{chain_label}")
        if identifier == "bad":
            raise ValueError("bad chain")
        return sequence

    monkeypatch.setattr(
        humatch_app,
        "_load_upstream",
        lambda: {"canonical_numbering": ()},
    )
    monkeypatch.setattr(humatch_app, "_align_chain", fake_align)
    monkeypatch.setattr(
        humatch_app,
        "_load_models",
        lambda _upstream: events.append("load_models"),
    )

    with pytest.raises(ValueError, match="bad chain"):
        humatch_app._run_humatch_worker_batch(
            pairs=[
                {"id": "good", "vh": "AAAA", "vl": "CCCC"},
                {"id": "bad", "vh": "AAAA", "vl": "CCCC"},
            ]
        )

    assert events == ["align:good:vh", "align:good:vl", "align:bad:vh"]


def test_execution_graph_batches_seven_pairs_as_six_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    csv_bytes = (
        "id,vh,vl\n" + "".join(f"pair-{index},AAAA,CCCC\n" for index in range(7))
    ).encode()
    request = replace(
        _request(),
        csv_bytes=csv_bytes,
        max_active_provider_calls=2,
    )

    class Driver:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []
            self.results: dict[str, object] = {}

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            del operation, args
            identifiers = [pair["id"] for pair in kwargs["pairs"]]
            self.batches.append(identifiers)
            call_id = f"call-{len(self.batches)}"
            self.results[call_id] = {
                "schema_version": 1,
                "pair_results": [
                    {"humanized": {"id": identifier}} for identifier in identifiers
                ],
                "metrics": {
                    "model_load_seconds": 1.0,
                    "humanization_seconds": 2.0,
                    "elapsed_seconds": 3.0,
                },
            }
            return call_id

        def observe(self, provider_call_handle_id):
            return ProviderCallObservation(
                ProviderCallObservationKind.SUCCEEDED,
                result=self.results[provider_call_handle_id],
            )

        def cancel(self, provider_call_handle_id):
            del provider_call_handle_id

    class Volume:
        def commit(self) -> None:
            pass

        def reload(self) -> None:
            pass

    monkeypatch.setattr(
        humatch_app,
        "_aggregate_humatch_results",
        lambda **_kwargs: (b"archive", {"pair_count": 7}),
    )
    run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    driver = Driver()
    store = GraphExecutionRunStore(tmp_path, run_id)
    runtime = ExecutionGraphRuntime(
        graph=humatch_execution_graph(request),
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "Humatch", 1),
        volume_root=tmp_path,
        artifact_volume_name="Humatch-outputs",
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=Volume(), store=store),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=0,
        store=store,
        now=iter(range(100, 1000)).__next__,
        poll_interval_seconds=0,
    )

    result = runtime.run(workload_run_key="example")

    assert driver.batches == [
        [f"pair-{index}" for index in range(6)],
        ["pair-6"],
    ]
    assert result.status == AppRunStatus.SUCCEEDED
    publication = store.artifacts.load_node_result(COLLECT_NODE)
    assert publication is not None
    assert publication.metrics["pair_count"] == 7
    runtime.close()


def test_result_loader_returns_content_verified_inline_archive() -> None:
    run_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    archive = b"archive"
    archive_path = f"workflow-runs/{run_id}/humatch/demo_humatch.tar.zst"
    publication = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="humatch_humanization",
                kind=ArtifactKind.ARCHIVE,
                storage=VolumePath(
                    volume_name="Humatch-outputs",
                    path=archive_path,
                    media_type=ZSTD_MEDIA_TYPE,
                ),
                metadata={
                    "files": [
                        {
                            "path": "demo_humatch.tar.zst",
                            "size_bytes": len(archive),
                            "content_sha256": sha256(archive).hexdigest(),
                        }
                    ]
                },
            )
        ],
    )
    files = {
        _RESULT_FILE.path(run_id).as_posix(): publication.model_dump_json().encode(),
        archive_path: archive,
    }

    class Volume:
        def read_file(self, path: str):
            yield files[path]

    overview = SimpleNamespace(
        run=SimpleNamespace(status=RunStatus.SUCCEEDED, execution_run_id=run_id)
    )

    result = result_from_overview(cast(Any, overview), Volume())

    assert result.outputs[0].storage == InlineBytes(
        data=archive,
        filename="demo_humatch.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )


def test_align_chain_requires_type_coverage_and_conserved_cysteines() -> None:
    sequence = "ACCEF"
    numbering = [
        ((1, " "), "A"),
        ((23, " "), "C"),
        ((104, " "), "C"),
        ((127, " "), "E"),
        ((128, " "), "F"),
    ]
    upstream = {
        "anarci": SimpleNamespace(number=lambda *_args, **_kwargs: (numbering, "H")),
        "canonical_numbering": ("1 ", "23 ", "104 ", "127 ", "128 "),
    }

    aligned = humatch_app._align_chain(
        identifier="pair",
        chain_label="vh",
        sequence=sequence,
        upstream=upstream,
    )

    assert aligned == "ACCEF"
    upstream["anarci"] = SimpleNamespace(
        number=lambda *_args, **_kwargs: (numbering, "L")
    )
    with pytest.raises(ValueError, match="heavy chain"):
        humatch_app._align_chain(
            identifier="pair",
            chain_label="vh",
            sequence=sequence,
            upstream=upstream,
        )


def test_humanize_pair_preserves_upstream_result_and_reports_endpoints() -> None:
    canonical = (
        "1 ",
        "23 ",
        "27 ",
        "38 ",
        "56 ",
        "65 ",
        "104 ",
        "105 ",
        "117 ",
        "127 ",
        "128 ",
    )
    parental_vh = "ACCCCCCCCCC"
    parental_vl = "ACCCCCCCCCC"
    calls: dict[str, Any] = {}

    def predict(sequences, model, *, num_cpus):
        assert num_cpus == 2
        if model == "h":
            return np.asarray([[0.01, 0.90, *([0.015] * 6)]])
        if model == "l":
            return np.asarray([[0.01, 0.80, *([0.01] * 16)]])
        return np.asarray([[0.10, 0.90]])

    def max_class(probabilities, classifier_type):
        classes = (
            humatch_app.HEAVY_CLASSES
            if classifier_type == "heavy"
            else humatch_app.LIGHT_CLASSES
        )
        index = int(np.argmax(probabilities[0, 1:])) + 1
        return [(classes[index], probabilities[0, index])]

    def humanise(vh, vl, _h, _l, _p, config):
        calls["config"] = config
        return {
            "Humatch_H": "V" + vh[1:],
            "Humatch_L": vl,
            "Edit": 1,
            "HV": "hv1",
            "LV": "lv1",
            "CNN_H": np.float32(0.96),
            "CNN_L": np.float32(0.97),
            "CNN_P": np.float32(0.98),
        }

    upstream = {
        "canonical_numbering": canonical,
        "predict": predict,
        "get_max_class": max_class,
        "gl_dir": "/models/gl",
        "gl_mutate": lambda sequence, *_args, **_kwargs: sequence,
        "gl_score": lambda *_args, **_kwargs: 0.4,
        "humanise": humanise,
    }
    result = humatch_app._humanize_pair(
        pair=humatch_app._AlignedPair("pair", parental_vh, parental_vl),
        models=("h", "l", "p"),
        upstream=upstream,
        vh_target_family="auto",
        vl_target_family="auto",
        germline_likeness_target=0.4,
        vh_classifier_target=0.95,
        vl_classifier_target=0.95,
        pair_classifier_target=0.95,
        max_edits=60,
        mutate_cdrs=False,
        fixed_vh_positions=(),
        fixed_vl_positions=(),
    )

    assert result["humanized"]["vh"].startswith("V")
    assert result["summary"]["humanization_success"] is True
    assert result["summary"]["edit_count"] == 1
    assert len(result["classifier_scores"]) == 2
    assert result["mutations"] == [
        {
            "id": "pair",
            "chain": "vh",
            "imgt_position": "1",
            "region": "FR1",
            "from_aa": "A",
            "to_aa": "V",
        }
    ]
    assert calls["config"]["target_gene_H"] == "hv1"
    assert calls["config"]["target_gene_L"] == "lv1"


def test_result_bundle_has_the_stable_workflow_files(monkeypatch) -> None:
    captured: dict[str, bytes] = {}

    def fake_package_outputs(result_dir: Any, *, num_threads: int) -> bytes:
        assert num_threads == 2
        assert pl.read_parquet(Path(result_dir) / "mutations.parquet").schema == {
            "id": pl.String,
            "chain": pl.String,
            "imgt_position": pl.String,
            "region": pl.String,
            "from_aa": pl.String,
            "to_aa": pl.String,
        }
        captured.update(
            (path.name, path.read_bytes()) for path in Path(result_dir).iterdir()
        )
        return b"archive"

    monkeypatch.setattr(humatch_app, "package_outputs", fake_package_outputs)
    pair_result = {
        "humanized": {"id": "pair 1", "vh": "AAAA", "vl": "CCCC"},
        "summary": {
            "id": "pair 1",
            "edit_count": 0,
            "humanization_success": True,
        },
        "classifier_scores": [{"id": "pair 1", "endpoint": "input", "h_neg": 0.0}],
        "alignment": [
            {
                "id": "pair 1",
                "chain": "vh",
                "imgt_position": "1",
                "region": "FR1",
                "input_aa": "A",
                "final_aa": "A",
                "mutated": False,
                "protected": False,
            }
        ],
        "mutations": [],
    }
    frame = pl.DataFrame([{"id": "pair 1", "vh": "AAAA", "vl": "CCCC"}])

    archive = humatch_app._write_result_bundle(
        run_name="demo",
        input_frame=frame,
        pair_results=[pair_result],
        parameters={},
    )

    assert captured["humanized.fasta"].decode().splitlines()[::2] == [
        ">pair_1_VH",
        ">pair_1_VL",
    ]
    assert archive == b"archive"
    assert set(captured) == {
        "input.csv",
        "humanized.csv",
        "humanized.fasta",
        "summary.csv",
        "classifier_scores.csv",
        "alignment.csv",
        "mutations.parquet",
        "manifest.json",
    }
    assert b'"schema_version": 2' in captured["manifest.json"]


def test_workflow_function_returns_inline_archive(monkeypatch) -> None:
    metrics = {
        "pair_count": 3,
        "mutation_count": 7,
        "success_count": 2,
        "model_load_seconds": 1.0,
        "humanization_seconds": 2.0,
        "elapsed_seconds": 3.0,
    }
    monkeypatch.setattr(
        humatch_app,
        "_run_humatch_humanization",
        lambda **kwargs: (b"archive", metrics),
    )

    result = humatch_app.humatch_humanize.get_raw_f()(
        run_name="../example",
        csv_bytes=VALID_CSV,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert result.metrics == metrics
    assert result.outputs[0].storage == InlineBytes(
        data=b"archive",
        filename="example_humatch.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )


def test_image_asset_manifest_contains_only_inference_assets() -> None:
    assert len(ASSETS) == 27
    assert {asset.subdirectory for asset in ASSETS} == {
        "trained_models",
        "germline_likeness_lookup_arrays",
    }
    assert all(not asset.filename.endswith(".zip") for asset in ASSETS)
