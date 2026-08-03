"""Tests for standalone AlphaFold3 app behavior."""

# ruff: noqa: D103

import ast
import inspect
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3 import upstream_inference
from biomodals.app.fold.alphafold3.generation_claims import (
    GenerationClaim,
    finish_generation_claim,
    generation_status,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    LoadedInferenceInput,
    VolumeUpload,
    prepare_inference_run,
)
from biomodals.app.fold.alphafold3.modal_adapters import (
    InProcessInferenceExecutor,
    execute_profile_setup,
    publish_invocation_receipt,
    stage_inference_run,
)
from biomodals.app.fold.alphafold3.profiles import DATABASE_PROFILE_SPECS
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    InferenceRuntime,
    SeedClaimPlan,
    guard_seed_prediction_claims,
)


class _ClaimStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(
        self,
        key: str,
        value: object,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)


class _Volume:
    def reload(self) -> None:
        pass

    def commit(self) -> None:
        pass


def _claimed_seed(run_id: str, seed: int) -> ClaimedSeed:
    scope_key = f"seed:{run_id}:{seed}"
    owner = {
        "scope_key": scope_key,
        "generation_id": "generation",
        "identity": {
            "schema_version": 1,
            "run_id": run_id,
            "seed": seed,
        },
        "container_id": "claim-container",
        "started_at": "2026-07-28T00:00:00Z",
        "started_at_epoch_seconds": 1.0,
        "maximum_age_seconds": 100.0,
    }
    return ClaimedSeed(
        seed=seed,
        claim=GenerationClaim(
            scope_key=scope_key,
            generation_id="generation",
            owner=owner,
        ),
    )


def _install_claim_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    claimed_seed: ClaimedSeed,
) -> _ClaimStore:
    claims = _ClaimStore()
    claims.put(
        f"claim:{claimed_seed.claim.scope_key}:root",
        claimed_seed.claim.owner,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "_INFERENCE_RUNTIME",
        InferenceRuntime(
            output_root=tmp_path,
            volume=_Volume(),
            claims=claims,
            container_id="worker-container",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
    )
    return claims


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


def test_in_process_inference_executor_uses_direct_function_bodies() -> None:
    prepared = prepare_inference_run(
        AF3Config(
            name="direct",
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
    worker_batches: list[list[int]] = []

    def worker(
        run_id: str,
        request_id: str,
        staged_input_record: dict[str, object],
        claim_records: list[dict[str, object]],
    ) -> dict[str, object]:
        assert (run_id, request_id) == (prepared.run_id, prepared.request_id)
        assert staged_input_record == prepared.staged_input.to_record()
        seeds = [int(record["seed"]) for record in claim_records]
        worker_batches.append(seeds)
        return {
            "run_id": run_id,
            "published_seeds": seeds,
            "reused_seeds": [],
        }

    executor = InProcessInferenceExecutor(
        claim_function=lambda run_id, seeds, sample_count: SeedClaimPlan(
            reused_seeds=(),
            owned=claimed,
            active=(),
        ).to_dict(),
        inspect_function=lambda run_id, seeds, sample_count: [
            {"run_id": run_id, "seed": seed, "status": "reused"} for seed in seeds
        ],
        worker_function=worker,
        summary_function=lambda run_id, request_id, staged: {
            "status": "complete",
        },
        request_function=(
            lambda run_id, request_id, submitted, normalized, sample_count, name: {
                "status": "complete",
            }
        ),
    )

    outcome = executor.run_claimed(
        prepared,
        claimed,
        max_workers=1,
        poll_timeout_seconds=7,
    )

    assert worker_batches == [[1, 2]]
    assert outcome.published_seeds == frozenset({1, 2})
    assert outcome.reused_seeds == frozenset()
    assert outcome.failures == ()


def test_inference_staging_is_marker_last_and_reusable() -> None:
    prepared = prepare_inference_run(
        AF3Config(
            name="staging",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
            ],
        ),
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

    missing_payload = prepared.payload_uploads[0]
    del volume.files[missing_payload.relative_path.as_posix()]
    stage_inference_run(volume, prepared)
    assert volume.files[missing_payload.relative_path.as_posix()] == (
        missing_payload.content
    )
    assert volume.publications[-2:] == [
        (missing_payload.relative_path.as_posix(),),
        (prepared.staged_input.relative_path.as_posix(),),
    ]

    corrupt_payload = prepared.payload_uploads[-1]
    volume.files[corrupt_payload.relative_path.as_posix()] = b"changed"
    stage_inference_run(volume, prepared)
    assert volume.files[corrupt_payload.relative_path.as_posix()] == (
        corrupt_payload.content
    )

    volume.files[prepared.staged_input.relative_path.as_posix()] = b"changed"
    with pytest.raises(RuntimeError, match="conflicts"):
        stage_inference_run(volume, prepared)

    class StreamingConflictVolume(FakeVolume):
        def read_file(self, path: str):
            if path == prepared.staged_input.relative_path.as_posix():
                yield prepared.staged_input.content + b"x"
                pytest.fail("staged marker read continued after a conflict")
            yield from super().read_file(path)

    with pytest.raises(RuntimeError, match="conflicts"):
        stage_inference_run(StreamingConflictVolume(), prepared)

    receipt_volume = FakeVolume()
    receipt = VolumeUpload(
        PurePosixPath("invocations/aa") / f"{'a' * 64}.json",
        b'{"status":"complete"}',
    )
    publish_invocation_receipt(receipt_volume, receipt)
    publish_invocation_receipt(receipt_volume, receipt)
    assert receipt_volume.publications == [(receipt.relative_path.as_posix(),)]
    receipt_volume.files[receipt.relative_path.as_posix()] = b"changed"
    with pytest.raises(RuntimeError, match="receipt conflicts"):
        publish_invocation_receipt(receipt_volume, receipt)


def test_submit_alphafold3_task_applies_run_name_to_prediction_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thin client stages a normalized request for the remote coordinator."""
    input_json = tmp_path / "input.json"
    conf = AF3Config(
        name="original",
        modelSeeds=[11, 12],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )
    input_json.write_text(conf.model_dump_json(), encoding="utf-8")
    captured: dict[str, object] = {}

    class CoordinatorMethod:
        def spawn(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(
                object_id="fc-coordinator",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=alphafold3_app.RunStatus.SUCCEEDED,
                        status_reason=None,
                        status_message=None,
                    )
                ),
            )

    class Coordinator:
        run = CoordinatorMethod()

    def stage(output_volume, execution_run_id, request):
        del output_volume
        captured["execution_run_id"] = execution_run_id
        captured["request"] = request

    def coordinator_handle(**kwargs):
        captured["coordinator"] = kwargs
        return Coordinator()

    manifest: dict[str, object] = {"status": "complete"}
    monkeypatch.setattr(alphafold3_app, "stage_execution_request", stage)
    monkeypatch.setattr(
        alphafold3_app,
        "stage_execution_launch",
        lambda _volume, run_id, predecessor: captured.update(
            launch=(run_id, predecessor)
        ),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "_execution_coordinator_handle",
        coordinator_handle,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_invocation_manifest",
        lambda output_volume, invocation: manifest,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "request_publication_from_manifest",
        lambda selected: SimpleNamespace(run_id="a" * 64),
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
        use_deployed_coordinator=True,
        deployment_environment="production",
        deployment_name="AlphaFold3Prod",
        deployment_version=7,
    )

    request = captured["request"]
    assert request.config.name == "renamed"
    assert request.config.modelSeeds == [11, 12]
    assert request.max_num_gpus == 4
    assert request.recycle == 3
    assert request.sample == 2
    assert captured["launch"] == (captured["execution_run_id"], None)
    assert captured["run_kwargs"] == {"development": False}
    deployment = captured["coordinator"]["deployment"]
    assert deployment.environment == "production"
    assert deployment.deployment_name == "AlphaFold3Prod"
    assert deployment.deployment_version == 7


def test_submit_alphafold3_task_routes_a_cache_hit_through_a_new_root_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated launch still records a root Run before reusing publications."""
    input_json = tmp_path / "input.json"
    input_json.write_text(
        AF3Config(
            name="cached",
            modelSeeds=[11],
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
        ).model_dump_json(),
        encoding="utf-8",
    )
    cached_manifest: dict[str, object] = {"status": "complete"}
    captured: dict[str, object] = {}

    class CoordinatorMethod:
        def spawn(self, **kwargs):
            captured["spawned"] = kwargs
            return SimpleNamespace(
                object_id="fc-cached",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=alphafold3_app.RunStatus.SUCCEEDED,
                        status_reason=None,
                        status_message=None,
                    )
                ),
            )

    def create_archive(reader, manifest, *, output_dir, display_name):
        del reader, output_dir
        captured["manifest"] = manifest
        captured["display_name"] = display_name
        return tmp_path / "cached.tar.zst"

    monkeypatch.setattr(
        alphafold3_app,
        "stage_execution_request",
        lambda output, run_id, request: captured.update({
            "run_id": run_id,
            "request": request,
        }),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "stage_execution_launch",
        lambda _volume, run_id, predecessor: captured.update(
            launch=(run_id, predecessor)
        ),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "_execution_coordinator_handle",
        lambda **kwargs: SimpleNamespace(run=CoordinatorMethod()),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_invocation_manifest",
        lambda output_volume, invocation: cached_manifest,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "request_publication_from_manifest",
        lambda selected: SimpleNamespace(run_id="a" * 64),
    )
    monkeypatch.setattr(alphafold3_app, "create_request_archive", create_archive)

    entrypoint = alphafold3_app.submit_alphafold3_task.info
    assert entrypoint is not None and entrypoint.raw_f is not None
    entrypoint.raw_f(
        input_json=str(input_json),
        out_dir=str(tmp_path),
        search_msa=False,
        recycle=1,
        sample=1,
    )

    assert captured["request"].config.name == "cached"
    assert captured["launch"] == (captured["run_id"], None)
    assert captured["spawned"] == {"development": True}
    assert captured["manifest"] is cached_manifest
    assert captured["display_name"] == "cached"


def test_submit_alphafold3_task_restart_creates_a_successor_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch-time restart delegates to the coordinator's successor operation."""
    input_json = tmp_path / "input.json"
    input_json.write_text(
        AF3Config(
            name="exact",
            modelSeeds=[11],
            sequences=[AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))],
        ).model_dump_json(),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {"status": "complete"}
    captured: dict[str, object] = {}
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    class ForbiddenRun:
        def spawn(self, **kwargs):
            pytest.fail(f"restart submitted a root Run: {kwargs}")

    class Restart:
        def spawn(self, **kwargs):
            captured["restart"] = kwargs
            return SimpleNamespace(
                object_id="fc-successor",
                get=lambda: SimpleNamespace(
                    run=SimpleNamespace(
                        status=alphafold3_app.RunStatus.SUCCEEDED,
                        status_reason=None,
                        status_message=None,
                    )
                ),
            )

    monkeypatch.setattr(
        alphafold3_app,
        "stage_execution_request",
        lambda _output, run_id, request: captured.update(staged=(run_id, request)),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "stage_execution_launch",
        lambda _volume, run_id, selected_predecessor: captured.update(
            launch=(run_id, selected_predecessor)
        ),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "_execution_coordinator_handle",
        lambda **kwargs: SimpleNamespace(
            run=ForbiddenRun(),
            restart_from=Restart(),
        ),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_invocation_manifest",
        lambda output, invocation: manifest,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "request_publication_from_manifest",
        lambda selected: SimpleNamespace(run_id="a" * 64),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "create_request_archive",
        lambda *args, **kwargs: tmp_path / "exact.tar.zst",
    )

    entrypoint = alphafold3_app.submit_alphafold3_task.info
    assert entrypoint is not None and entrypoint.raw_f is not None
    entrypoint.raw_f(
        input_json=str(input_json),
        out_dir=str(tmp_path),
        recycle=1,
        sample=1,
        restart_from=predecessor,
    )

    restart = cast(dict[str, object], captured["restart"])
    assert captured["launch"] == (captured["staged"][0], UUID(predecessor))
    assert restart["predecessor_execution_run_id"] == predecessor
    candidate_bytes = restart["candidate_request_bytes"]
    assert isinstance(candidate_bytes, bytes)
    candidate = alphafold3_app.AlphaFold3ExecutionRequest.from_bytes(candidate_bytes)
    assert candidate.config.name == "exact"
    assert candidate.execution_plan.workload_plan_fingerprint


def test_coordinator_launch_restart_forwards_candidate_bytes() -> None:
    """The launch convenience validates candidate bytes before staging state."""
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    successor = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    candidate = alphafold3_app.AlphaFold3ExecutionRequest.prepare(
        AF3Config(
            name="successor",
            modelSeeds=[11],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(id="A", sequence="ACDE"),
                )
            ],
        ),
        search_msa=False,
        search_protein_templates=False,
        max_parallel_search_workers=2,
        max_num_gpus=1,
        recycle=1,
        sample=1,
    )
    captured: dict[str, object] = {}

    class Adapter:
        def prepare_restart(self, **kwargs):
            captured.update(kwargs)

        def drive_prepared(self):
            return "snapshot"

    raw_cls = alphafold3_app.ExecutionCoordinator._get_user_cls()
    instance = raw_cls()
    instance.execution_run_id = successor
    instance.deployment_environment = "main"
    instance.deployment_name = "AlphaFold3"
    instance.deployment_version = 8
    alphafold3_app.initialize_execution_coordinator_host(instance)
    instance._coordinator_adapter = Adapter()
    instance._development = False

    result = raw_cls.restart_from._get_raw_f()(
        instance,
        predecessor,
        candidate.to_bytes(),
    )

    assert result == "snapshot"
    assert captured == {
        "predecessor_execution_run_id": alphafold3_app.UUID(predecessor),
        "predecessor_deployment": None,
        "candidate_request": candidate,
    }


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
        claimed_seed_records=[_claimed_seed("a" * 64, 1).to_dict()],
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


def test_inference_worker_revalidates_loaded_numeric_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    claimed_seed = _claimed_seed("a" * 64, 1)
    _install_claim_runtime(monkeypatch, tmp_path, claimed_seed)
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
            claimed_seed_records=[claimed_seed.to_dict()],
        )


def test_inference_worker_fails_claim_when_staged_input_loading_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "a" * 64
    seed = 42
    claimed_seed = _claimed_seed(run_id, seed)
    claims = _install_claim_runtime(monkeypatch, tmp_path, claimed_seed)
    monkeypatch.setattr(
        alphafold3_app.CONF.output_volume,
        "reload",
        lambda: None,
    )
    monkeypatch.setattr(
        alphafold3_app,
        "load_staged_inference_input",
        Mock(side_effect=ValueError("staged input is invalid")),
    )

    with pytest.raises(ValueError, match="staged input is invalid"):
        alphafold3_app.run_inference_pipeline.get_raw_f()(
            run_id=run_id,
            request_id="b" * 64,
            staged_input_record={
                "path": "staged-input.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
            claimed_seed_records=[claimed_seed.to_dict()],
        )

    status = generation_status(
        claims,
        claimed_seed.claim.scope_key,
        claimed_seed.claim.generation_id,
    )
    assert status is not None
    assert status["status"] == "failed"
    assert isinstance(status["finished_at"], str)
    assert status["error_type"] == "ValueError"
    assert status["message"] == "staged input is invalid"
    assert status["phase"] == "inference-worker"


def test_inference_worker_preserves_claim_already_completed_by_inner_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_id = "a" * 64
    claimed_seed = _claimed_seed(run_id, 42)
    claims = _install_claim_runtime(monkeypatch, tmp_path, claimed_seed)

    with pytest.raises(RuntimeError, match="after publication"):
        with guard_seed_prediction_claims(
            alphafold3_app._INFERENCE_RUNTIME,
            run_id,
            [claimed_seed.to_dict()],
        ):
            finish_generation_claim(
                claims,
                claimed_seed.claim,
                status="complete",
                detail={"publication": "published"},
            )
            raise RuntimeError("after publication")

    status = generation_status(
        claims,
        claimed_seed.claim.scope_key,
        claimed_seed.claim.generation_id,
    )
    assert status is not None
    assert status["status"] == "complete"
    assert status["publication"] == "published"


def test_inference_worker_rejects_seed_outside_staged_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    claimed_seed = _claimed_seed("a" * 64, 2)
    _install_claim_runtime(monkeypatch, tmp_path, claimed_seed)
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
            claimed_seed_records=[claimed_seed.to_dict()],
        )

    worker.assert_not_called()
