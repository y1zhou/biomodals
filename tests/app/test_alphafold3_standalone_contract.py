"""Tests for standalone AlphaFold3 app behavior."""

# ruff: noqa: D103

from pathlib import Path

import orjson
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app
from biomodals.app.fold.alphafold3.inference_inputs import PreparedInferenceRun


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
