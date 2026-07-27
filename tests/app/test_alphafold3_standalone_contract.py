"""Tests for standalone AlphaFold3 app behavior."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3.generation_claims import GenerationClaim
from biomodals.app.fold.alphafold3.inference_inputs import (
    PreparedInferenceRun,
    prepare_inference_run,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MsaAssemblyTask,
    RawSearchTask,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    ClaimedSeed,
    SeedClaimPlan,
)
from biomodals.app.fold.alphafold3.template_search import TemplateTask


def test_modal_search_executor_marshals_remote_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production adapter should preserve task payloads and worker caps."""
    calls: list[tuple[object, ...]] = []
    budgets: list[int] = []

    def fake_bounded_map(items, worker, *, max_parallel):
        budgets.append(max_parallel)
        return [worker(item) for item in items]

    def inspect_raw(inputs):
        calls.append(("inspect-raw", inputs))
        return [{"status": "missing"} for _ in inputs]

    def run_raw(database_id, sequence):
        calls.append(("run-raw", database_id, sequence))
        if database_id == "uniref90":
            raise RuntimeError("search failed")
        return {"status": "published"}

    def run_assembly(polymer, sequence, include_unpaired, include_paired):
        calls.append((
            "assemble",
            polymer,
            sequence,
            include_unpaired,
            include_paired,
        ))
        return {"status": "published"}

    def inspect_templates(inputs):
        calls.append(("inspect-templates", inputs))
        return [{"status": "missing"} for _ in inputs]

    def run_template(sequence, unpaired_msa, publish_canonical, max_template_date):
        calls.append((
            "search-template",
            sequence,
            unpaired_msa,
            publish_canonical,
            max_template_date,
        ))
        return {"status": "published"}

    monkeypatch.setattr(alphafold3_app, "bounded_map", fake_bounded_map)
    monkeypatch.setattr(
        alphafold3_app,
        "inspect_msa_search_cache",
        SimpleNamespace(remote=inspect_raw),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "search_database_msa",
        SimpleNamespace(remote=run_raw),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "assemble_sequence_msas",
        SimpleNamespace(remote=run_assembly),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "inspect_protein_template_cache",
        SimpleNamespace(remote=inspect_templates),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "search_protein_templates",
        SimpleNamespace(remote=run_template),
    )

    executor = alphafold3_app._ModalSearchExecutor()
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
    assert calls == [
        ("inspect-raw", [("small_bfd", "ACDE"), ("uniref90", "FGHI")]),
        ("run-raw", "small_bfd", "ACDE"),
        ("run-raw", "uniref90", "FGHI"),
        ("assemble", "protein", "ACDE", True, False),
        (
            "inspect-templates",
            [
                (
                    "ACDE",
                    template_tasks[0].unpaired_msa_sha256,
                    "2021-09-30",
                )
            ],
        ),
        (
            "search-template",
            "ACDE",
            ">query\nACDE\n",
            True,
            "2021-09-30",
        ),
    ]


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
    remote_calls: list[tuple[object, ...]] = []
    spawn_calls: list[tuple[object, ...]] = []
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

    def spawn_worker(json_bytes, run_id, recycle, sample, claim_records):
        spawn_calls.append((json_bytes, run_id, recycle, sample, claim_records))
        seed = claim_records[0]["seed"]
        spawned_seeds.append(seed)
        return FakeFunctionCall(
            {
                "run_id": run_id,
                "published_seeds": [seed] if seed == 1 else [],
                "reused_seeds": [seed] if seed == 2 else [],
            },
            timeout_once=seed == 1,
        )

    def claim_remote(run_id, seeds, sample_count):
        remote_calls.append(("claim", run_id, seeds, sample_count))
        return SeedClaimPlan(
            reused_seeds=(2,),
            owned=(claimed[0],),
            active=(),
        ).to_dict()

    def inspect_remote(run_id, seeds, sample_count):
        remote_calls.append(("inspect", run_id, seeds, sample_count))
        return [{"status": "reused", "seed": seed} for seed in seeds]

    def summary_remote(json_bytes, run_id, sample_count):
        remote_calls.append(("summary", json_bytes, run_id, sample_count))
        return {"status": "complete"}

    def request_remote(
        run_id,
        request_id,
        submitted_seeds,
        normalized_seeds,
        sample_count,
        display_name,
        reused_seeds,
        published_seeds,
    ):
        remote_calls.append((
            "request",
            run_id,
            request_id,
            submitted_seeds,
            normalized_seeds,
            sample_count,
            display_name,
            reused_seeds,
            published_seeds,
        ))
        return {"status": "complete"}

    monkeypatch.setattr(
        alphafold3_app,
        "run_inference_pipeline",
        SimpleNamespace(spawn=spawn_worker),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "claim_seed_prediction_work",
        SimpleNamespace(remote=claim_remote),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "inspect_seed_prediction_cache",
        SimpleNamespace(remote=inspect_remote),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "finalize_inference_summary",
        SimpleNamespace(remote=summary_remote),
    )
    monkeypatch.setattr(
        alphafold3_app,
        "finalize_inference_request",
        SimpleNamespace(remote=request_remote),
    )

    executor = alphafold3_app._ModalInferenceExecutor()
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
    assert len(spawn_calls) == 2
    assert spawned_seeds == [1, 2]
    assert all(call[1:4] == (prepared.run_id, 3, 2) for call in spawn_calls)
    assert poll_timeouts == [7, 7, 7]

    assert executor.finalize_summary(prepared, sample_count=2) == {"status": "complete"}
    assert executor.finalize_request(
        prepared,
        sample_count=2,
        reused_seeds=(2,),
        published_seeds=(1,),
    ) == {"status": "complete"}
    assert remote_calls == [
        ("claim", prepared.run_id, [1, 2], 2),
        ("inspect", prepared.run_id, [1, 2], 2),
        (
            "summary",
            alphafold3_app.serialize_af3_input(prepared.worker_config),
            prepared.run_id,
            2,
        ),
        (
            "request",
            prepared.run_id,
            prepared.request_id,
            [1, 2],
            [1, 2],
            2,
            "composition",
            [2],
            [1],
        ),
    ]


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

    monkeypatch.setattr(alphafold3_app, "_stage_inference_run", lambda prepared: None)
    monkeypatch.setattr(alphafold3_app, "predict_structures", fake_predict_structures)
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
    assert prepared.worker_config.modelSeeds == [11, 12]
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

    monkeypatch.setattr(alphafold3_app, "run_command", fake_run_command)
    monkeypatch.setattr(
        alphafold3_app,
        "run_seed_prediction_worker",
        fake_run_seed_prediction_worker,
    )

    result = alphafold3_app.run_inference_pipeline.get_raw_f()(
        conf.model_dump_json().encode("utf-8"),
        run_id="a" * 64,
        recycle=1,
        sample=1,
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
    assert captured["task"].run_id == "a" * 64
    assert tuple(item.seed for item in captured["task"].claimed_seeds) == (1,)
    protein = captured["input"]["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []
    assert (
        f"--model_dir={alphafold3_app.CONF.model_volume_mountpoint}" in captured["cmd"]
    )
    assert (
        f"--jax_compilation_cache_dir={alphafold3_app.JAX_CACHE_DIR}" in captured["cmd"]
    )
    assert not alphafold3_app.JAX_CACHE_DIR.is_relative_to(
        alphafold3_app.CONF.model_volume_mountpoint
    )


def test_inference_worker_revalidates_numeric_limits() -> None:
    conf = AF3Config(
        name="invalid-worker-counts",
        modelSeeds=[1],
        sequences=[
            AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE")),
        ],
    )

    with pytest.raises(ValueError, match="between 0 and"):
        alphafold3_app.run_inference_pipeline.get_raw_f()(
            conf.model_dump_json().encode(),
            run_id="a" * 64,
            recycle=101,
            sample=1,
            claimed_seed_records=[],
        )
