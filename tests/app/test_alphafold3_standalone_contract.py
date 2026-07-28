"""Tests for standalone AlphaFold3 app behavior."""

# ruff: noqa: D103

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3 import modal_adapters, upstream_inference
from biomodals.app.fold.alphafold3.generation_claims import GenerationClaim
from biomodals.app.fold.alphafold3.inference_inputs import (
    LoadedInferenceInput,
    PreparedInferenceRun,
    prepare_inference_run,
)
from biomodals.app.fold.alphafold3.modal_adapters import (
    ModalInferenceExecutor,
    ModalSearchExecutor,
    execute_profile_setup,
    stage_inference_run,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
)
from biomodals.app.fold.alphafold3.profiles import DATABASE_PROFILE_SPECS
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    SeedClaimPlan,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask


def test_app_public_functions_are_modal_endpoints() -> None:
    tree = ast.parse(inspect.getsource(alphafold3_app))
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        decorator_names = {
            decorator.func.attr
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "app"
        }
        if not decorator_names.intersection({"function", "local_entrypoint"}):
            violations.append(node.name)
    assert violations == []


def test_summary_claim_lifetime_matches_its_shorter_function_timeout() -> None:
    runtime = alphafold3_app._INFERENCE_RUNTIME

    assert runtime.summary_maximum_age_seconds == (
        alphafold3_app._SUMMARY_TIMEOUT_SECONDS + 900
    )
    assert runtime.summary_maximum_age_seconds < runtime.maximum_age_seconds


def test_profile_setup_adapter_fans_out_missing_profiles() -> None:
    spec = DATABASE_PROFILE_SPECS[0]
    inventory: dict[str, object] = {
        "invalid_profiles": {},
        "missing_database_ids": [spec.database_id],
    }
    build_result = {"database_id": spec.database_id, "status": "published"}
    final_inventory = {"missing_database_ids": []}
    inspect_remote = Mock(return_value=inventory)
    build_starmap = Mock(return_value=[build_result])
    finalize_remote = Mock(return_value=final_inventory)

    result = execute_profile_setup(
        SimpleNamespace(remote=inspect_remote),
        SimpleNamespace(starmap=build_starmap),
        SimpleNamespace(remote=finalize_remote),
        seqkit_threads=8,
        source_policy="keep",
    )

    assert result == {
        "status": "complete",
        "initial_inventory": inventory,
        "builder_results": [build_result],
        "final_inventory": final_inventory,
    }
    build_starmap.assert_called_once_with(
        ((spec.database_id, 8, "keep"),),
        return_exceptions=True,
    )


def test_modal_search_executor_marshals_remote_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production adapter should preserve task payloads and worker caps."""
    budgets: list[int] = []

    def fake_bounded_map(items, worker, *, max_parallel):
        budgets.append(max_parallel)
        return [worker(item) for item in items]

    inspect_raw = Mock(return_value=[{"status": "missing"}, {"status": "missing"}])
    run_raw = Mock(side_effect=({"status": "published"}, RuntimeError("search failed")))
    run_assembly = Mock(return_value={"status": "published"})
    inspect_templates = Mock(return_value=[{"status": "missing"}])
    run_template = Mock(return_value={"status": "published"})
    monkeypatch.setattr(modal_adapters, "bounded_map", fake_bounded_map)
    executor = ModalSearchExecutor(
        inspect_raw_function=SimpleNamespace(remote=inspect_raw),
        raw_search_function=SimpleNamespace(remote=run_raw),
        msa_assembly_function=SimpleNamespace(remote=run_assembly),
        inspect_templates_function=SimpleNamespace(remote=inspect_templates),
        template_search_function=SimpleNamespace(remote=run_template),
    )
    raw_tasks = (
        RawSearchTask(database_id="small_bfd", sequence="ACDE"),
        RawSearchTask(database_id="uniref90", sequence="FGHI"),
    )
    assert executor.inspect_raw(raw_tasks) == (
        {"status": "missing"},
        {"status": "missing"},
    )
    raw_outcomes = executor.run_raw(raw_tasks, max_parallel=2)
    assert raw_outcomes[0] == {"status": "published"}
    assert isinstance(raw_outcomes[1], RuntimeError)

    assembly_tasks = (
        MsaAssemblyTask(
            polymer="protein",
            sequence="ACDE",
            include_unpaired=True,
            include_paired=False,
        ),
    )
    assert executor.run_assemblies(assembly_tasks, max_parallel=3) == (
        {"status": "published"},
    )

    template_tasks = (
        TemplateTask(
            sequence="ACDE",
            unpaired_msa=">query\nACDE\n",
            publish_canonical=True,
            max_template_date="2021-09-30",
        ),
    )
    assert executor.inspect_templates(template_tasks) == ({"status": "missing"},)
    assert executor.run_templates(template_tasks, max_parallel=4) == (
        {"status": "published"},
    )

    assert budgets == [2, 3, 4]
    inspect_raw.assert_called_once_with([
        ("small_bfd", "ACDE"),
        ("uniref90", "FGHI"),
    ])
    assert run_raw.call_args_list == [
        call("small_bfd", "ACDE"),
        call("uniref90", "FGHI"),
    ]
    run_assembly.assert_called_once_with("protein", "ACDE", True, False)
    inspect_templates.assert_called_once_with([
        (
            "ACDE",
            template_tasks[0].unpaired_msa_sha256,
            "2021-09-30",
        )
    ])
    run_template.assert_called_once_with(
        "ACDE",
        ">query\nACDE\n",
        True,
        "2021-09-30",
    )


def test_modal_inference_executor_routes_spawn_poll_and_finalizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production adapter should marshal claims, workers, and publication."""
    prepared = prepare_inference_run(
        AF3Config(
            name="composition",
            modelSeeds=[1, 2],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(
                        id="A",
                        sequence="ACDE",
                        unpairedMsa="",
                        pairedMsa="",
                        templates=[],
                    )
                )
            ],
        ),
        (),
        output_mount_root=Path("/outputs"),
        recycle=3,
        sample=2,
    )
    claimed = tuple(
        ClaimedSeed(
            seed=seed,
            claim=GenerationClaim(
                scope_key=f"seed:{prepared.run_id}:{seed}",
                generation_id=f"generation-{seed}",
                owner={},
            ),
        )
        for seed in (1, 2)
    )
    spawned_seeds: list[int] = []
    poll_timeouts: list[int] = []

    class FakeFunctionCall:
        def __init__(self, result: dict[str, object], *, timeout_once: bool) -> None:
            self.result = result
            self.timeout_once = timeout_once

        def get(self, *, timeout: int) -> dict[str, object]:
            poll_timeouts.append(timeout)
            if self.timeout_once:
                self.timeout_once = False
                raise TimeoutError
            return self.result

    def spawn_worker(
        run_id: str,
        request_id: str,
        staged_input_record: dict[str, object],
        claim_records: list[dict[str, object]],
    ) -> FakeFunctionCall:
        assert (run_id, request_id) == (prepared.run_id, prepared.request_id)
        assert staged_input_record == prepared.staged_input.to_record()
        seed = claim_records[0].get("seed")
        assert isinstance(seed, int)
        spawned_seeds.append(seed)
        return FakeFunctionCall(
            {
                "run_id": run_id,
                "published_seeds": [seed] if seed == 1 else [],
                "reused_seeds": [seed] if seed == 2 else [],
            },
            timeout_once=seed == 1,
        )

    claim_remote = Mock(
        return_value=SeedClaimPlan(
            reused_seeds=(2,),
            owned=(claimed[0],),
            active=(),
        ).to_dict()
    )
    inspect_remote = Mock(
        side_effect=lambda _run_id, seeds, _sample: [
            {"status": "reused", "seed": seed} for seed in seeds
        ]
    )
    summary_remote = Mock(return_value={"status": "complete"})
    request_remote = Mock(return_value={"status": "complete"})
    spawn_remote = Mock(side_effect=spawn_worker)
    executor = ModalInferenceExecutor(
        claim_function=SimpleNamespace(remote=claim_remote),
        inspect_function=SimpleNamespace(remote=inspect_remote),
        worker_function=SimpleNamespace(spawn=spawn_remote),
        summary_function=SimpleNamespace(remote=summary_remote),
        request_function=SimpleNamespace(remote=request_remote),
    )
    assert executor.claim_seeds(
        prepared.run_id,
        (1, 2),
        sample_count=2,
    ) == SeedClaimPlan(reused_seeds=(2,), owned=(claimed[0],), active=())
    assert executor.inspect_seeds(
        prepared.run_id,
        (1, 2),
        sample_count=2,
    ) == (
        {"status": "reused", "seed": 1},
        {"status": "reused", "seed": 2},
    )

    outcome = executor.run_claimed(
        prepared,
        claimed,
        recycle=3,
        sample_count=2,
        max_workers=2,
        poll_timeout_seconds=7,
    )
    assert outcome.published_seeds == frozenset({1})
    assert outcome.reused_seeds == frozenset({2})
    assert outcome.failures == ()
    assert spawned_seeds == [1, 2]
    assert poll_timeouts == [7, 7, 7]

    assert executor.finalize_summary(prepared, sample_count=2) == {"status": "complete"}
    assert executor.finalize_request(
        prepared,
        sample_count=2,
    ) == {"status": "complete"}
    claim_remote.assert_called_once_with(prepared.run_id, [1, 2], 2)
    inspect_remote.assert_called_once_with(prepared.run_id, [1, 2], 2)
    summary_remote.assert_called_once_with(
        prepared.run_id,
        prepared.request_id,
        prepared.staged_input.to_record(),
    )
    request_remote.assert_called_once_with(
        prepared.run_id,
        prepared.request_id,
        [1, 2],
        [1, 2],
        2,
        "composition",
    )


def test_inference_staging_is_marker_last_and_reusable() -> None:
    prepared = prepare_inference_run(
        AF3Config(
            name="staging",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
            ],
        ),
        (),
        output_mount_root=Path("/outputs"),
        recycle=1,
        sample=1,
    )

    class FakeBatchUpload:
        def __init__(self, volume) -> None:
            self.volume = volume
            self.pending: dict[str, bytes] = {}

        def __enter__(self):
            return self

        def put_file(self, source, remote_path: str) -> None:
            self.pending[remote_path.lstrip("/")] = source.read()

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            self.volume.files.update(self.pending)
            self.volume.publications.append(tuple(self.pending))

    class FakeVolume:
        def __init__(self) -> None:
            self.files: dict[str, bytes] = {}
            self.publications: list[tuple[str, ...]] = []

        def read_file(self, path: str):
            if path not in self.files:
                raise FileNotFoundError(path)
            yield self.files[path]

        def batch_upload(self, *, force: bool):
            assert force is True
            return FakeBatchUpload(self)

    volume = FakeVolume()
    stage_inference_run(volume, prepared)

    assert len(volume.publications) == 2
    assert set(volume.publications[0]) == {
        upload.relative_path.as_posix() for upload in prepared.payload_uploads
    }
    assert volume.publications[1] == (prepared.staged_input.relative_path.as_posix(),)

    stage_inference_run(volume, prepared)
    assert len(volume.publications) == 2

    volume.files[prepared.staged_input.relative_path.as_posix()] = b"changed"
    with pytest.raises(RuntimeError, match="conflicts"):
        stage_inference_run(volume, prepared)


def test_submit_alphafold3_task_applies_run_name_to_prediction_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_json = tmp_path / "input.json"
    conf = AF3Config(
        name="original",
        modelSeeds=[11, 12],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )
    input_json.write_text(conf.model_dump_json(), encoding="utf-8")
    captured = {}

    def fake_predict_structures(
        prepared: PreparedInferenceRun,
        recycle: int,
        sample: int,
        num_containers: int,
    ) -> dict[str, object]:
        captured["prepared"] = prepared
        captured["recycle"] = recycle
        captured["sample"] = sample
        captured["num_containers"] = num_containers
        return {"request": {"status": "complete"}}

    monkeypatch.setattr(
        alphafold3_app,
        "stage_inference_run",
        lambda output_volume, prepared: None,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "_predict_structures",
        fake_predict_structures,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "create_request_archive",
        lambda reader, manifest, *, output_dir, display_name: (
            Path(output_dir) / f"{display_name}.tar.zst"
        ),
    )

    submit_task_info = alphafold3_app.submit_alphafold3_task.info
    assert submit_task_info is not None
    submit_task_raw_f = submit_task_info.raw_f
    assert submit_task_raw_f is not None
    submit_task_raw_f(
        input_json=str(input_json),
        out_dir=str(tmp_path),
        run_name="renamed",
        search_msa=False,
        max_num_gpus=4,
        recycle=3,
        sample=2,
    )

    prepared = captured.pop("prepared")
    assert isinstance(prepared, PreparedInferenceRun)
    assert prepared.display_name == "renamed"
    assert prepared.submitted_seeds == (11, 12)
    assert prepared.normalized_seeds == (11, 12)
    assert prepared.recycle == 3
    assert prepared.sample_count == 2
    input_upload = next(
        upload
        for upload in prepared.payload_uploads
        if upload.relative_path.name == "input.json"
    )
    assert AF3Config.model_validate_json(input_upload.content).modelSeeds == [11, 12]
    assert captured == {"recycle": 3, "sample": 2, "num_containers": 2}


def test_submit_alphafold3_task_rejects_input_json_symlink(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(
        AF3Config(
            name="symlink",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    input_path.symlink_to(target_path)
    entrypoint_info = alphafold3_app.submit_alphafold3_task.info
    assert entrypoint_info is not None
    raw_entrypoint = entrypoint_info.raw_f
    assert raw_entrypoint is not None

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        raw_entrypoint(input_json=str(input_path), search_msa=False)


def test_inference_pipeline_marks_bare_sequences_as_single_sequence_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    conf = AF3Config(
        name="single-seq",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )
    captured = {}
    reloads: list[None] = []

    def fake_run_command(cmd, *, output_mode, log_file):
        del output_mode, log_file
        captured["cmd"] = tuple(str(arg) for arg in cmd)
        json_path = Path(
            next(
                arg.removeprefix("--json_path=")
                for arg in cmd
                if str(arg).startswith("--json_path=")
            )
        )
        captured["input"] = orjson.loads(json_path.read_bytes())

    def fake_run_seed_prediction_worker(runtime, task, execute):
        del runtime
        captured["task"] = task
        worker_root = tmp_path / "worker"
        worker_root.mkdir()
        execute(worker_root, "af3-test", (1,))
        return {"status": "published"}

    monkeypatch.setattr(upstream_inference, "run_command", fake_run_command)
    monkeypatch.setattr(
        upstream_inference,
        "run_seed_prediction_worker",
        fake_run_seed_prediction_worker,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_staged_inference_input",
        lambda output_root, **identity: LoadedInferenceInput(
            config=conf,
            recycle=1,
            sample_count=1,
        ),
    )
    monkeypatch.setattr(
        alphafold3_app.CONF.output_volume,
        "reload",
        lambda: reloads.append(None),
    )

    result = alphafold3_app.run_inference_pipeline.get_raw_f()(
        run_id="a" * 64,
        request_id="b" * 64,
        staged_input_record={
            "path": "staged-input.json",
            "size_bytes": 1,
            "sha256": "c" * 64,
        },
        claimed_seed_records=[
            {
                "seed": 1,
                "claim": {
                    "scope_key": f"seed:{'a' * 64}:1",
                    "generation_id": "generation",
                    "owner": {},
                },
            }
        ],
    )

    assert result == {"status": "published"}
    assert reloads == [None]
    assert captured["task"].run_id == "a" * 64
    assert tuple(item.seed for item in captured["task"].claimed_seeds) == (1,)
    protein = captured["input"]["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []
    assert (
        f"--model_dir={alphafold3_app.CONF.model_volume_mountpoint}" in captured["cmd"]
    )
    jax_cache_arg = next(
        arg for arg in captured["cmd"] if arg.startswith("--jax_compilation_cache_dir=")
    )
    jax_cache_dir = Path(jax_cache_arg.partition("=")[2])
    assert jax_cache_dir.name == alphafold3_app.ALPHAFOLD3_COMMIT
    assert not jax_cache_dir.is_relative_to(alphafold3_app.CONF.model_volume_mountpoint)


def test_inference_worker_revalidates_loaded_numeric_limits(monkeypatch) -> None:
    conf = AF3Config(
        name="invalid-worker-counts",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )

    monkeypatch.setattr(
        alphafold3_app,
        "load_staged_inference_input",
        lambda output_root, **identity: LoadedInferenceInput(
            config=conf,
            recycle=101,
            sample_count=1,
        ),
    )
    monkeypatch.setattr(
        alphafold3_app.CONF.output_volume,
        "reload",
        lambda: None,
    )

    with pytest.raises(ValueError, match="between 0 and"):
        alphafold3_app.run_inference_pipeline.get_raw_f()(
            run_id="a" * 64,
            request_id="b" * 64,
            staged_input_record={
                "path": "staged-input.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
            claimed_seed_records=[],
        )


def test_inference_worker_rejects_seed_outside_staged_request(monkeypatch) -> None:
    conf = AF3Config(
        name="request-bound-worker",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_staged_inference_input",
        lambda output_root, **identity: LoadedInferenceInput(
            config=conf,
            recycle=1,
            sample_count=1,
        ),
    )
    monkeypatch.setattr(
        alphafold3_app.CONF.output_volume,
        "reload",
        lambda: None,
    )
    worker = Mock(side_effect=AssertionError("worker must not run"))
    monkeypatch.setattr(upstream_inference, "run_seed_prediction_worker", worker)

    with pytest.raises(ValueError, match="staged request"):
        alphafold3_app.run_inference_pipeline.get_raw_f()(
            run_id="a" * 64,
            request_id="b" * 64,
            staged_input_record={
                "path": "staged-input.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
            claimed_seed_records=[
                {
                    "seed": 2,
                    "claim": {
                        "scope_key": f"seed:{'a' * 64}:2",
                        "generation_id": "generation",
                        "owner": {},
                    },
                }
            ],
        )

    worker.assert_not_called()
