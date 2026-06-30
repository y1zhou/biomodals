"""Tests for standalone AlphaFold3 app behavior."""

# ruff: noqa: D103

from pathlib import Path

import orjson
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold import alphafold3_app


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
        prediction_conf,
        local_out_dir: Path,
        recycle: int,
        sample: int,
        num_containers: int,
    ) -> Path:
        captured["name"] = prediction_conf.name
        captured["model_seeds"] = list(prediction_conf.modelSeeds)
        captured["local_out_dir"] = local_out_dir
        captured["recycle"] = recycle
        captured["sample"] = sample
        captured["num_containers"] = num_containers
        return local_out_dir / f"{prediction_conf.name}.tar.zst"

    monkeypatch.setattr(alphafold3_app, "predict_structures", fake_predict_structures)

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

    assert captured == {
        "name": "renamed",
        "model_seeds": [11, 12],
        "local_out_dir": tmp_path,
        "recycle": 3,
        "sample": 2,
        "num_containers": 2,
    }


def test_inference_pipeline_marks_bare_sequences_as_single_sequence_inputs(
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
        json_path = Path(
            next(
                arg.removeprefix("--json_path=")
                for arg in cmd
                if str(arg).startswith("--json_path=")
            )
        )
        output_dir = Path(
            next(
                arg.removeprefix("--output_dir=")
                for arg in cmd
                if str(arg).startswith("--output_dir=")
            )
        )
        captured["input"] = orjson.loads(json_path.read_bytes())
        (output_dir / conf.name).mkdir(parents=True)
        (output_dir / conf.name / "done.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(alphafold3_app, "run_command", fake_run_command)

    alphafold3_app.run_inference_pipeline.get_raw_f()(
        conf.model_dump_json().encode("utf-8"),
        recycle=1,
        sample=1,
        model_seeds=[1],
    )

    protein = captured["input"]["sequences"][0]["protein"]
    assert protein["unpairedMsa"] == ""
    assert protein["pairedMsa"] == ""
    assert protein["templates"] == []
