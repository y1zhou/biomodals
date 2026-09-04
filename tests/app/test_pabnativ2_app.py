"""Contracts for the p-AbNatiV2 humanization app."""

# ruff: noqa: D101,D102,D103

from __future__ import annotations

import inspect
import os
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import orjson
import polars as pl
import pytest

from biomodals.app.design.pabnativ2 import app as pabnativ2_app
from biomodals.app.design.pabnativ2 import execution as pabnativ2_execution
from biomodals.app.design.pabnativ2 import patches as pabnativ2_patches
from biomodals.app.design.pabnativ2.execution import (
    _RESULT_FILE,
    COLLECT_NODE,
    HUMANIZE_NODE,
    PAbNatiV2ExecutionRequest,
    _PAbNatiV2HumanizeNode,
    pabnativ2_execution_graph,
    result_from_overview,
)
from biomodals.app.design.pabnativ2.models import (
    IDENTITY,
    PAIRED_MODEL,
    STRUCTURE_MODEL_ARCHIVE,
)
from biomodals.app.design.pabnativ2.patches import _replace_once
from biomodals.execution import DeploymentIdentity, GraphExecutionRunStore, RunStatus
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.execution.nodes import NodeRunContext
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


def _write_test_pair_result(
    payload: tuple[dict[str, str], dict[str, Any]], output_path: str
) -> None:
    pair, _parameters = payload
    if pair["id"] == "hard-exit":
        os._exit(17)
    Path(output_path).write_text(
        AppRunResult(status=AppRunStatus.SUCCEEDED).model_dump_json(),
        encoding="utf-8",
    )


def _request() -> PAbNatiV2ExecutionRequest:
    return PAbNatiV2ExecutionRequest(
        run_name="example",
        csv_bytes=VALID_CSV,
        mutate_cdrs=False,
        fixed_vh_positions="27,28",
        fixed_vl_positions="",
        residue_score_threshold=0.98,
        rasa_threshold=0.15,
        max_relative_pairing_score_decrease=0.1,
        forbidden_residues="C,M",
        seed=7,
        source_commit=IDENTITY.abnativ_commit,
        paired_model_md5=PAIRED_MODEL.md5_hex,
        structure_archive_md5=STRUCTURE_MODEL_ARCHIVE.md5_hex,
        runtime_identity="runtime",
    )


def test_modal_group_tag_does_not_depend_on_source_path() -> None:
    assert pabnativ2_app.CONF.tags == {"group": "design"}


def test_parse_pabnativ2_csv_accepts_complete_unique_pairs() -> None:
    frame = pabnativ2_app.parse_pabnativ2_csv(VALID_CSV)

    assert frame.schema == {"id": pl.String, "vh": pl.String, "vl": pl.String}
    assert frame.item(0, "id") == "pair-1"


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"id,vh\na,AAAA\n", "exactly these columns"),
        (b"id,vh,vl\na,AAAA,CCCC\na,DDDD,EEEE\n", "duplicate id"),
        (b"id,vh,vl\na,AAAA,\n", "vl must be non-empty"),
        (b"id,vh,vl\na,AAAa,CCCC\n", "non-canonical or lowercase"),
    ),
)
def test_parse_pabnativ2_csv_rejects_invalid_batches(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pabnativ2_app.parse_pabnativ2_csv(content)


def test_controls_normalize_positions_and_forbidden_residues() -> None:
    parameters = pabnativ2_app._validate_parameters(
        mutate_cdrs=True,
        fixed_vh_positions="27, 149",
        fixed_vl_positions="",
        residue_score_threshold=0.98,
        rasa_threshold=0.15,
        max_relative_pairing_score_decrease=0.1,
        forbidden_residues="C,M",
        seed=0,
    )

    assert parameters.fixed_vh_positions == (27, 149)
    assert parameters.fixed_vl_positions == ()
    assert parameters.forbidden_residues == ("C", "M")
    assert 27 not in pabnativ2_app._allowed_positions(True, (27,))
    assert 27 not in pabnativ2_app._allowed_positions(False, ())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"fixed_vh_positions": "27,27"}, "duplicate"),
        ({"fixed_vl_positions": "150"}, "between 1 and 149"),
        ({"forbidden_residues": "C,C"}, "duplicates"),
        ({"forbidden_residues": "c"}, "uppercase canonical"),
        ({"residue_score_threshold": float("nan")}, "finite"),
        ({"seed": True}, "seed"),
    ),
)
def test_controls_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "mutate_cdrs": False,
        "fixed_vh_positions": "",
        "fixed_vl_positions": "",
        "residue_score_threshold": 0.98,
        "rasa_threshold": 0.15,
        "max_relative_pairing_score_decrease": 0.1,
        "forbidden_residues": "C,M",
        "seed": 0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        pabnativ2_app._validate_parameters(**values)  # type: ignore[arg-type]


def test_pair_seed_is_order_independent_and_input_specific() -> None:
    first = pabnativ2_app._pair_seed(4, "id", "AAAA", "CCCC")

    assert first == pabnativ2_app._pair_seed(4, "id", "AAAA", "CCCC")
    assert first != pabnativ2_app._pair_seed(4, "other", "AAAA", "CCCC")


def test_execution_request_roundtrips_and_plans_per_pair_gpu_tasks(monkeypatch) -> None:
    request = _request()
    records = tuple(pabnativ2_app.parse_pabnativ2_csv(VALID_CSV).to_dicts())
    monkeypatch.setattr(pabnativ2_execution, "_pair_records", lambda _content: records)

    assert PAbNatiV2ExecutionRequest.from_bytes(request.to_bytes()) == request
    assert [node.node_key for node in request.execution_plan.nodes] == [
        HUMANIZE_NODE,
        COLLECT_NODE,
    ]
    node = _PAbNatiV2HumanizeNode(request)
    task = node.discover_remote_tasks(cast(NodeRunContext, None))[0]
    call = node.prepare_remote_task(cast(NodeRunContext, None), task)
    assert call.function_name == "pabnativ2_humanize_pair"
    assert call.uses_gpu is True
    assert call.runtime_image_key == "pabnativ2-a10g"
    assert call.kwargs["pair"]["id"] == "pair-1"
    assert call.kwargs["seed"] == 7


def test_multi_pair_tasks_prepare_one_fixed_batch(monkeypatch) -> None:
    records = (
        {"id": "pair-1", "vh": "AAAA", "vl": "CCCC"},
        {"id": "pair-2", "vh": "DDDD", "vl": "EEEE"},
    )
    monkeypatch.setattr(pabnativ2_execution, "_pair_records", lambda _content: records)
    node = _PAbNatiV2HumanizeNode(_request())
    tasks = node.discover_remote_tasks(cast(NodeRunContext, None))

    descriptor = node.prepare_remote_task(cast(NodeRunContext, None), tasks[0])
    call = node.prepare_remote_task_batch(cast(NodeRunContext, None), tasks)

    assert descriptor == node.prepare_remote_task_batch(
        cast(NodeRunContext, None), (tasks[0],)
    )
    assert descriptor.function_name == "pabnativ2_humanize_batch"
    assert descriptor.max_tasks_per_call == 4
    assert call.function_name == "pabnativ2_humanize_batch"
    assert [pair["id"] for pair in call.kwargs["pairs"]] == ["pair-1", "pair-2"]


def test_execution_request_rejects_boolean_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        replace(_request(), seed=True)


def test_gpu_worker_returns_common_app_result(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = AppRunResult(status=AppRunStatus.SUCCEEDED)
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> AppRunResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(pabnativ2_app, "_run_pabnativ2_pair", fake_run)

    pair = {"id": "pair", "vh": "AAAA", "vl": "CCCC"}
    result = pabnativ2_app.pabnativ2_humanize_pair.get_raw_f()(pair)

    assert result is expected
    assert captured["pair"] == pair


def test_gpu_batch_worker_returns_pair_result_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"pair": AppRunResult(status=AppRunStatus.SUCCEEDED)}
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, AppRunResult]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(pabnativ2_app, "_run_pabnativ2_batch", fake_run)
    pairs = [{"id": "pair", "vh": "AAAA", "vl": "CCCC"}]

    result = pabnativ2_app.pabnativ2_humanize_batch.get_raw_f()(pairs)

    assert result == {
        pair_id: pair_result.model_dump(mode="json")
        for pair_id, pair_result in expected.items()
    }
    assert captured["pairs"] == pairs


@pytest.mark.parametrize("pair_count", (1, 2))
def test_batch_runner_spawns_one_process_per_pair_including_singletons(
    pair_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [
        {"id": f"pair-{index}", "vh": "AAAA", "vl": "CCCC"}
        for index in range(1, pair_count + 1)
    ]
    monkeypatch.setattr(pabnativ2_app, "_validate_antibody_chains", lambda _frame: None)
    monkeypatch.setattr(
        pabnativ2_app, "_write_pabnativ2_pair_result", _write_test_pair_result
    )

    results = pabnativ2_app._run_pabnativ2_batch(pairs=pairs)

    assert list(results) == [pair["id"] for pair in pairs]
    assert {result.status for result in results.values()} == {AppRunStatus.SUCCEEDED}


def test_hard_child_exit_preserves_successful_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [
        {"id": "successful", "vh": "AAAA", "vl": "CCCC"},
        {"id": "hard-exit", "vh": "DDDD", "vl": "EEEE"},
    ]
    monkeypatch.setattr(pabnativ2_app, "_validate_antibody_chains", lambda _frame: None)
    monkeypatch.setattr(
        pabnativ2_app, "_write_pabnativ2_pair_result", _write_test_pair_result
    )

    results = pabnativ2_app._run_pabnativ2_batch(pairs=pairs)

    assert results["successful"].status == AppRunStatus.SUCCEEDED
    assert results["hard-exit"].status == AppRunStatus.FAILED
    assert results["hard-exit"].warnings == [
        "hard-exit: ProcessExit: child exited with code 17"
    ]


def test_subprocess_failure_returns_a_failed_pair_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair = {"id": "pair-1", "vh": "AAAA", "vl": "CCCC"}

    def fail_pair(**_kwargs: object) -> AppRunResult:
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(pabnativ2_app, "_run_pabnativ2_pair", fail_pair)

    encoded = pabnativ2_app._run_pabnativ2_pair_in_subprocess((pair, {}))
    result = AppRunResult.model_validate_json(encoded)

    assert result.status == AppRunStatus.FAILED
    assert result.warnings == ["pair-1: RuntimeError: upstream failed"]
    assert "RuntimeError: upstream failed" in capsys.readouterr().err


def test_execution_request_requires_gpu_capacity_and_safe_run_name() -> None:
    request = replace(
        _request(), max_active_provider_calls=4, max_active_gpu_provider_calls=3
    )

    assert request.max_active_gpu_provider_calls == 3
    with pytest.raises(ValueError, match="positive GPU"):
        replace(request, max_active_gpu_provider_calls=0)
    with pytest.raises(ValueError, match="safe short filename"):
        replace(request, run_name="x" * 201)


def test_entrypoint_accepts_cli_provider_limits() -> None:
    entrypoint = pabnativ2_app.submit_pabnativ2_task.info.raw_f
    assert entrypoint is not None
    parameters = inspect.signature(entrypoint).parameters

    assert "max_containers" in parameters
    assert "max_gpu_containers" in parameters


@pytest.mark.parametrize("failed_pair_id", (None, "pair-2"))
def test_execution_graph_batches_four_pairs_and_preserves_pair_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_pair_id: str | None,
) -> None:
    records = tuple(
        {"id": f"pair-{index}", "vh": "AAAA", "vl": "CCCC"} for index in range(1, 6)
    )
    monkeypatch.setattr(pabnativ2_execution, "_pair_records", lambda _content: records)
    request = replace(
        _request(),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
    )

    class Driver:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []
            self.results: dict[str, object] = {}

        def resolve(self, binding):
            return binding.function_name

        def spawn(self, operation, *, args, kwargs):
            del operation, args
            pairs = kwargs["pairs"]
            pair_ids = [pair["id"] for pair in pairs]
            self.batches.append(pair_ids)
            call_id = f"call-{pair_ids[0]}"
            self.results[call_id] = {
                pair_id: AppRunResult(
                    status=(
                        AppRunStatus.FAILED
                        if pair_id == failed_pair_id
                        else AppRunStatus.SUCCEEDED
                    ),
                    outputs=(
                        []
                        if pair_id == failed_pair_id
                        else [
                            AppOutput(
                                name="pabnativ2_pair_result",
                                kind=ArtifactKind.REPORT,
                                storage=InlineBytes(
                                    data=orjson.dumps({
                                        "schema_version": 1,
                                        "pair_result": {"humanized": {"id": pair_id}},
                                    }),
                                    filename="pabnativ2-pair.json",
                                    media_type="application/json",
                                ),
                            ),
                        ]
                    ),
                    warnings=(["pair failed"] if pair_id == failed_pair_id else []),
                    metrics={"pair_count": 1},
                ).model_dump(mode="json")
                for pair_id in pair_ids
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
        pabnativ2_app,
        "_aggregate_pabnativ2_results",
        lambda **_kwargs: (b"archive", {"pair_count": 5}),
    )
    run_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    driver = Driver()
    store = GraphExecutionRunStore(tmp_path, run_id)
    runtime = ExecutionGraphRuntime(
        graph=pabnativ2_execution_graph(request),
        execution_run_id=run_id,
        deployment=DeploymentIdentity("main", "p-AbNatiV2", 1),
        volume_root=tmp_path,
        artifact_volume_name="p-AbNatiV2-outputs",
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=Volume(), store=store),
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
        store=store,
        now=iter(range(100, 1000)).__next__,
        poll_interval_seconds=0,
    )

    result = runtime.run(workload_run_key="example")

    assert driver.batches == [
        ["pair-1", "pair-2", "pair-3", "pair-4"],
        ["pair-5"],
    ]
    assert result.status == (
        AppRunStatus.FAILED if failed_pair_id is not None else AppRunStatus.SUCCEEDED
    )
    assert [
        task.status.value for task in store.execution.list_tasks(run_id, HUMANIZE_NODE)
    ] == [
        "failed" if pair_id == failed_pair_id else "succeeded"
        for pair_id in ("pair-1", "pair-2", "pair-3", "pair-4", "pair-5")
    ]
    publication = store.artifacts.load_node_result(COLLECT_NODE)
    if failed_pair_id is None:
        assert publication is not None
        assert publication.metrics["pair_count"] == 5
    else:
        assert publication is None
    runtime.close()


def test_result_loader_returns_content_verified_inline_archive() -> None:
    run_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    archive = b"archive"
    archive_path = f"workflow-runs/{run_id}/pabnativ2/demo_pabnativ2.tar.zst"
    publication = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="pabnativ2_humanization",
                kind=ArtifactKind.ARCHIVE,
                storage=VolumePath(
                    volume_name="p-AbNatiV2-outputs",
                    path=archive_path,
                    media_type=ZSTD_MEDIA_TYPE,
                ),
                metadata={
                    "files": [
                        {
                            "path": "demo_pabnativ2.tar.zst",
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
        filename="demo_pabnativ2.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )


def test_result_manifest_records_protocol_caveat_and_device(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_package(root: Path, *, num_threads: int) -> bytes:
        assert num_threads == 2
        captured.update(orjson.loads((root / "manifest.json").read_bytes()))
        assert (root / "structures/0001_pair-1/input.pdb").read_bytes() == b"input"
        assert pl.read_parquet(root / "mutations.parquet").schema == {
            "id": pl.String,
            "chain": pl.String,
            "aho_position": pl.Int64,
            "region": pl.String,
            "input_aa": pl.String,
            "final_aa": pl.String,
            "input_residue_score": pl.Float64,
            "final_residue_score": pl.Float64,
        }
        return b"archive"

    monkeypatch.setattr(pabnativ2_app, "package_outputs", fake_package)
    parameters = pabnativ2_app._validate_parameters(
        mutate_cdrs=False,
        fixed_vh_positions="",
        fixed_vl_positions="",
        residue_score_threshold=0.98,
        rasa_threshold=0.15,
        max_relative_pairing_score_decrease=0.1,
        forbidden_residues="C,M",
        seed=0,
    )
    result = {
        "humanized": {"id": "pair-1", "vh": "AAAA", "vl": "CCCC"},
        "sequence_scores": [
            {"id": "pair-1", "endpoint": endpoint} for endpoint in ("input", "final")
        ],
        "residue_scores": [
            {
                "id": "pair-1",
                "endpoint": "input",
                "chain": "vh",
                "aho_position": 1,
                "observed_aa": "A",
                "observed_residue_score": 0.9,
            }
        ],
        "mutations": [],
        "input_pdb": "aW5wdXQ=",
        "final_pdb": "ZmluYWw=",
        "pair_seed": 1,
        "device": "NVIDIA A10G",
    }

    archive = pabnativ2_app._write_bundle(
        run_name="demo",
        input_frame=pl.DataFrame([{"id": "pair-1", "vh": "AAAA", "vl": "CCCC"}]),
        pair_results=[result],
        parameters=parameters,
        asset_manifest={"schema_version": 1},
    )

    assert archive == b"archive"
    assert captured["schema_version"] == 2
    assert captured["protocol"]["equivalence_target"] == "AbNatiV 2.0.8 source"
    assert captured["protocol"]["paper_rasa_structure_count"] == 10
    assert captured["protocol"]["pairing_score_units"] == "fraction"
    assert captured["telemetry"]["accelerators"] == ["NVIDIA A10G"]
    assert captured["warnings"]
    assert "wrapper-protocol=2" in captured["scientific_identity"]["runtime"]


def test_compatibility_patches_are_guarded_and_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.py"
    path.write_text("return data[ranges]\n", encoding="utf-8")

    _replace_once(path, "return data[ranges]", "return data[tuple(ranges)]")
    _replace_once(path, "return data[ranges]", "return data[tuple(ranges)]")

    assert path.read_text(encoding="utf-8") == "return data[tuple(ranges)]\n"


def test_missing_pssm_is_hash_verified_and_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"pinned pssm"
    monkeypatch.setattr(
        pabnativ2_patches,
        "PSSM_SHA256",
        {"VH2_pssm.npy": sha256(content).hexdigest()},
    )
    monkeypatch.setattr(pabnativ2_patches, "_package_root", lambda _package: tmp_path)
    monkeypatch.setattr(
        pabnativ2_patches,
        "urlopen",
        lambda _url, timeout: BytesIO(content),  # noqa: ARG005
    )

    pabnativ2_patches.install_missing_abnativ_pssms()

    assert (tmp_path / "humanisation/pssms/VH2_pssm.npy").read_bytes() == content
