"""Tests for the Caliby Modal app."""

# ruff: noqa: D103

import inspect
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from biomodals.app.design import caliby_app
from biomodals.cli import app as cli_app


def test_app_contract() -> None:
    assert caliby_app.CONF.name == "Caliby"
    assert caliby_app.CONF.repo_url == "https://github.com/ProteinDesignLab/caliby"
    assert caliby_app.CONF.repo_commit_hash == (
        "8136f57d912dadeb64f9f60cb1021c06791b271d"
    )
    assert caliby_app.CONF.gpu
    assert caliby_app.CONF.timeout == 86400
    assert caliby_app.submit_caliby_task.info is not None

    source = Path(caliby_app.__file__).read_text()
    assert "git clone {CONF.repo_url} {CONF.git_clone_dir}" in source
    assert "git checkout {CONF.repo_commit_hash}" in source
    assert ".uv_pip_install(str(CONF.git_clone_dir))" in source

    signature = inspect.signature(caliby_app.submit_caliby_task.info.raw_f)
    assert signature.parameters["input_yaml"].default is inspect.Parameter.empty
    assert signature.parameters["task"].default == "ensemble_design"
    assert signature.parameters["out_dir"].default == "."
    assert "download_models" not in signature.parameters
    assert "force_redownload" not in signature.parameters
    assert "batch_size" not in signature.parameters
    assert "num_workers" not in signature.parameters
    assert "temperature" not in signature.parameters
    assert "omit_aas" not in signature.parameters
    assert "sampling_overrides_json" not in signature.parameters
    assert "score_baseline" not in signature.parameters
    assert "pack_sidechains" not in signature.parameters


def test_app_info_groups_caliby_metadata() -> None:
    assert caliby_app.APP_INFO.structure_suffixes == (".pdb", ".cif")
    assert caliby_app.APP_INFO.valid_tasks == frozenset({"design", "ensemble_design"})
    assert caliby_app.APP_INFO.design_model_by_task == {
        "design": "soluble_caliby_v1",
        "ensemble_design": "caliby",
    }
    assert caliby_app.APP_INFO.default_num_workers == 2
    assert caliby_app.APP_INFO.default_batch_size == 4
    assert caliby_app.APP_INFO.default_ensemble_batch_size == 8
    assert caliby_app.APP_INFO.proteinmpnn_model_source == (
        caliby_app.APP_INFO.full_model_volume_mount / "LigandMPNN"
    )
    assert caliby_app.APP_INFO.protpardelle_model_source == (
        caliby_app.APP_INFO.full_model_volume_mount / "Caliby" / "protpardelle-1c"
    )

    source = Path(caliby_app.__file__).read_text()
    assert "VALID_TASKS =" not in source
    assert "DESIGN_MODEL_BY_TASK =" not in source
    assert "DEFAULT_BATCH_SIZE =" not in source


def test_reuses_shared_helpers_and_keeps_local_helpers_specific() -> None:
    from biomodals.helper import io as helper_io

    assert caliby_app.build_local_output_path is helper_io.build_local_output_path
    assert caliby_app.write_local_tarball is helper_io.write_local_tarball

    source = Path(caliby_app.__file__).read_text()
    assert "def validate_run_name(" not in source
    assert "def build_seq_des_command(" not in source
    assert "def build_generate_ensembles_command(" not in source
    assert "def build_seq_des_ensemble_command(" not in source
    assert "def build_score_command(" not in source
    assert "def build_score_ensemble_command(" not in source
    assert "def build_sidechain_pack_command(" not in source
    assert "def run_caliby_sidechain_pack(" not in source
    assert "caliby.eval.sampling.sidechain_pack" not in source
    assert "def _optional_override_args(" not in source
    assert "def load_caliby_yaml_config(" not in source
    assert "def stage_local_conformer_dir(" not in source
    assert "def package_caliby_outputs(" not in source
    assert '"python3",' not in source


def test_output_volume_path_helper_returns_relative_volume_path(tmp_path: Path) -> None:
    original_mountpoint = caliby_app.CONF.output_volume_mountpoint
    object.__setattr__(caliby_app.CONF, "output_volume_mountpoint", tmp_path)
    try:
        volume_path = caliby_app._output_volume_path(tmp_path / "demo" / "outputs")
    finally:
        object.__setattr__(
            caliby_app.CONF, "output_volume_mountpoint", original_mountpoint
        )

    assert volume_path == "demo/outputs"


def test_caliby_design_config_validates_yaml_fields() -> None:
    conf = caliby_app.CalibyDesignConfig.model_validate({
        "input_path": "inputs",
        "run_name": "demo",
        "num_seqs_per_pdb": 4,
        "pos_constraint_csv": "constraints.csv",
        "pdb_name_list": "names.txt",
        "score_baseline": False,
        "sampling_cfg_overrides": {
            "omit_aas": ["C"],
            "potts_sampling_cfg": {"potts_temperature": 0.1},
        },
    })

    assert conf.input_path == "inputs"
    assert conf.run_name == "demo"
    assert conf.num_seqs_per_pdb == 4
    assert conf.pos_constraint_csv == "constraints.csv"
    assert conf.pdb_name_list == "names.txt"
    assert conf.score_baseline is False
    assert conf.sampling_cfg_overrides == {
        "omit_aas": ["C"],
        "potts_sampling_cfg": {"potts_temperature": 0.1},
    }
    assert "pack_sidechains" not in caliby_app.CalibyDesignConfig.model_fields
    with pytest.raises(ValueError, match="pack_sidechains"):
        caliby_app.CalibyDesignConfig.model_validate({
            "input_path": "inputs",
            "run_name": "demo",
            "pack_sidechains": True,
        })


def test_caliby_ensemble_config_requires_input_path() -> None:
    with pytest.raises(ValueError, match="input_path"):
        caliby_app.CalibyEnsembleDesignConfig.model_validate({
            "run_name": "demo",
        })

    with pytest.raises(ValueError, match="conformer_dir"):
        caliby_app.CalibyEnsembleDesignConfig.model_validate({
            "conformer_dir": "conformers",
            "run_name": "demo",
        })

    conf = caliby_app.CalibyEnsembleDesignConfig.model_validate({
        "input_path": "inputs",
        "run_name": "demo",
        "max_num_conformers": 8,
        "include_primary_conformer": False,
        "sampling_yaml_path": "custom.yaml",
        "seed": 123,
    })

    assert conf.input_path == "inputs"
    assert conf.num_samples_per_pdb == caliby_app.APP_INFO.default_num_samples_per_pdb
    assert conf.max_num_conformers == 8
    assert conf.include_primary_conformer is False
    assert "conformer_dir" not in caliby_app.CalibyEnsembleDesignConfig.model_fields
    assert "pack_sidechains" not in caliby_app.CalibyEnsembleDesignConfig.model_fields
    assert conf.sampling_yaml_path == "custom.yaml"
    assert conf.seed == 123
    with pytest.raises(ValueError, match="pack_sidechains"):
        caliby_app.CalibyEnsembleDesignConfig.model_validate({
            "input_path": "inputs",
            "run_name": "demo",
            "pack_sidechains": True,
        })


def test_discover_structure_files_accepts_file_and_directory(tmp_path: Path) -> None:
    pdb = tmp_path / "a.pdb"
    cif = tmp_path / "b.cif"
    mmcif = tmp_path / "c.mmcif"
    pdb_gz = tmp_path / "d.pdb.gz"
    ignored = tmp_path / "notes.txt"
    pdb.write_text("ATOM\n")
    cif.write_text("data_b\n")
    mmcif.write_text("data_c\n")
    pdb_gz.write_text("not really gzip\n")
    ignored.write_text("ignore\n")

    assert caliby_app.discover_structure_files(pdb) == [pdb.resolve()]
    assert caliby_app.discover_structure_files(tmp_path) == [
        pdb.resolve(),
        cif.resolve(),
    ]
    with pytest.raises(ValueError, match="Unsupported structure file suffix"):
        caliby_app.discover_structure_files(mmcif)
    with pytest.raises(ValueError, match="Unsupported structure file suffix"):
        caliby_app.discover_structure_files(pdb_gz)


def test_discover_structure_files_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No PDB/CIF"):
        caliby_app.discover_structure_files(tmp_path)


def test_build_run_paths_uses_expected_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(caliby_app.CONF, "output_volume_mountpoint", tmp_path)
    paths = caliby_app.build_run_paths("demo/001")

    assert paths["run_root"] == tmp_path / "demo_001"
    assert paths["inputs_dir"] == tmp_path / "demo_001" / "inputs"
    assert paths["design_dir"] == tmp_path / "demo_001" / "outputs" / "design"
    assert paths["baseline_score_dir"] == (
        tmp_path / "demo_001" / "outputs" / "baseline_score"
    )
    assert paths["ensembles_dir"] == tmp_path / "demo_001" / "outputs" / "ensembles"


class _FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1

    def batch_upload(self, *, force=False):
        return _FakeBatchUpload(self)


class _FakeBatchUpload:
    def __init__(self, volume: _FakeOutputVolume) -> None:
        self.volume = volume

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def put_file(self, source: str | Path, destination: str) -> None:
        self.volume.uploaded_files.append((Path(source), destination))

    def put_directory(self, source: str | Path, destination: str) -> None:
        self.volume.uploaded_dirs.append((Path(source), destination))


def test_stage_local_structures_uploads_inputs_to_output_volume(tmp_path: Path) -> None:
    fake_volume = _FakeOutputVolume()
    original_mountpoint = caliby_app.CONF.output_volume_mountpoint
    original_volume = caliby_app.CONF.output_volume
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")

    object.__setattr__(caliby_app.CONF, "output_volume_mountpoint", "/mnt/Caliby")
    object.__setattr__(caliby_app.CONF, "output_volume", fake_volume)

    try:
        staged = caliby_app.stage_local_structures(structure, "demo")
    finally:
        object.__setattr__(
            caliby_app.CONF, "output_volume_mountpoint", original_mountpoint
        )
        object.__setattr__(caliby_app.CONF, "output_volume", original_volume)

    assert staged == ["/mnt/Caliby/demo/inputs/structures/input.pdb"]
    assert fake_volume.uploaded_files == [
        (structure.resolve(), "/demo/inputs/structures/input.pdb")
    ]


def test_stage_optional_local_file_uploads_to_output_volume(tmp_path: Path) -> None:
    fake_volume = _FakeOutputVolume()
    original_mountpoint = caliby_app.CONF.output_volume_mountpoint
    original_volume = caliby_app.CONF.output_volume
    constraints = tmp_path / "constraints.csv"
    constraints.write_text("pdb,pos,aa\n")

    object.__setattr__(caliby_app.CONF, "output_volume_mountpoint", "/mnt/Caliby")
    object.__setattr__(caliby_app.CONF, "output_volume", fake_volume)

    try:
        remote_path = caliby_app.stage_optional_local_file(
            constraints,
            "demo",
            "constraints",
        )
    finally:
        object.__setattr__(
            caliby_app.CONF, "output_volume_mountpoint", original_mountpoint
        )
        object.__setattr__(caliby_app.CONF, "output_volume", original_volume)

    assert remote_path == "/mnt/Caliby/demo/inputs/constraints/constraints.csv"
    assert fake_volume.uploaded_files == [
        (constraints.resolve(), "/demo/inputs/constraints/constraints.csv")
    ]


def test_stage_optional_local_file_fails_fast_for_missing_local_path() -> None:
    with pytest.raises(FileNotFoundError, match="remote:"):
        caliby_app.stage_optional_local_file(
            "missing_constraints.csv",
            "demo",
            "constraints",
        )


def test_stage_optional_local_file_accepts_explicit_remote_path() -> None:
    assert (
        caliby_app.stage_optional_local_file(
            "remote:/mnt/Caliby/demo/constraints.csv",
            "demo",
            "constraints",
        )
        == "/mnt/Caliby/demo/constraints.csv"
    )


def test_run_caliby_seq_des_writes_expected_command(
    monkeypatch, tmp_path: Path
) -> None:
    fake_volume = _FakeOutputVolume()
    commands: list[list[str]] = []
    original_mountpoint = caliby_app.CONF.output_volume_mountpoint
    original_volume = caliby_app.CONF.output_volume

    object.__setattr__(caliby_app.CONF, "output_volume_mountpoint", tmp_path)
    object.__setattr__(caliby_app.CONF, "output_volume", fake_volume)

    def fake_run_command_with_log(cmd, *, log_file, verbose=True, cwd=None):
        commands.append(list(cmd))
        out_dir = tmp_path / "demo" / "outputs" / "design"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "seq_des_outputs.csv").write_text(
            "example_id,out_pdb,seq,U,input_seq\n"
        )

    monkeypatch.setattr(
        caliby_app, "run_command_with_log", fake_run_command_with_log, raising=False
    )

    try:
        result = caliby_app.run_caliby_seq_des.get_raw_f()(
            run_name="demo",
            ckpt_name="soluble_caliby_v1",
            num_seqs_per_pdb=16,
            batch_size=4,
            num_workers=2,
            pos_constraint_csv=None,
            pdb_name_list="/mnt/Caliby/demo/inputs/pdb_name_lists/list.txt",
            sampling_cfg_overrides={
                "omit_aas": ["C"],
                "potts_sampling_cfg": {"potts_temperature": 0.1},
            },
        )
    finally:
        object.__setattr__(
            caliby_app.CONF, "output_volume_mountpoint", original_mountpoint
        )
        object.__setattr__(caliby_app.CONF, "output_volume", original_volume)

    assert commands
    assert commands[0][:3] == [sys.executable, "-m", "caliby.eval.sampling.seq_des"]
    assert (
        "input_cfg.pdb_name_list=/mnt/Caliby/demo/inputs/pdb_name_lists/list.txt"
        in commands[0]
    )
    assert '++sampling_cfg_overrides.omit_aas=["C"]' in commands[0]
    assert (
        "++sampling_cfg_overrides.potts_sampling_cfg.potts_temperature=0.1"
        in commands[0]
    )
    assert result["design_outputs_dir"].endswith("/demo/outputs/design")
    assert fake_volume.reload_count == 1
    assert fake_volume.commit_count >= 1


def test_run_caliby_generate_ensembles_adapts_model_layout_with_symlinks(
    monkeypatch, tmp_path: Path
) -> None:
    fake_volume = _FakeOutputVolume()
    commands: list[list[str]] = []
    links: list[tuple[Path, Path]] = []
    output_mountpoint = tmp_path / "outputs"
    model_params_mount = tmp_path / "caliby-model-params"
    model_volume_mount = tmp_path / "models"
    input_dir = output_mountpoint / "demo" / "inputs" / "structures"
    input_dir.mkdir(parents=True)
    (input_dir / "native.cif").write_text("data_native\n")
    proteinmpnn_source = model_volume_mount / "LigandMPNN"
    protpardelle_source = model_volume_mount / "Caliby" / "protpardelle-1c"
    proteinmpnn_source.mkdir(parents=True)
    protpardelle_source.mkdir(parents=True)
    original_output_mountpoint = caliby_app.CONF.output_volume_mountpoint
    original_volume = caliby_app.CONF.output_volume
    original_model_params_mount = caliby_app.APP_INFO.caliby_model_params_mount
    original_full_model_volume_mount = caliby_app.APP_INFO.full_model_volume_mount
    original_proteinmpnn_source = caliby_app.APP_INFO.proteinmpnn_model_source
    original_protpardelle_source = caliby_app.APP_INFO.protpardelle_model_source

    object.__setattr__(caliby_app.CONF, "output_volume_mountpoint", output_mountpoint)
    object.__setattr__(caliby_app.CONF, "output_volume", fake_volume)
    object.__setattr__(
        caliby_app.APP_INFO, "caliby_model_params_mount", model_params_mount
    )
    object.__setattr__(
        caliby_app.APP_INFO, "full_model_volume_mount", model_volume_mount
    )
    object.__setattr__(
        caliby_app.APP_INFO, "proteinmpnn_model_source", proteinmpnn_source
    )
    object.__setattr__(
        caliby_app.APP_INFO, "protpardelle_model_source", protpardelle_source
    )

    def fake_softlink_dir(src, dst):
        links.append((Path(src), Path(dst)))
        Path(dst).mkdir(parents=True, exist_ok=True)

    def fake_run_command_with_log(cmd, *, log_file, verbose=True, cwd=None):
        commands.append(list(cmd))
        ensemble_dir = (
            output_mountpoint
            / "demo"
            / "outputs"
            / "ensembles"
            / "cc95-epoch3490"
            / "7urp"
        )
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        (ensemble_dir / "alt.cif").write_text("data_alt\n")

    monkeypatch.setattr(caliby_app, "softlink_dir", fake_softlink_dir)
    monkeypatch.setattr(
        caliby_app, "run_command_with_log", fake_run_command_with_log, raising=False
    )

    try:
        result = caliby_app.run_caliby_generate_ensembles.get_raw_f()(
            run_name="demo",
            pdb_dir=str(input_dir),
            num_samples_per_pdb=1,
            batch_size=8,
            sampling_yaml_path=None,
            seed=0,
            pdb_name_list=None,
        )
    finally:
        object.__setattr__(
            caliby_app.CONF, "output_volume_mountpoint", original_output_mountpoint
        )
        object.__setattr__(caliby_app.CONF, "output_volume", original_volume)
        object.__setattr__(
            caliby_app.APP_INFO,
            "caliby_model_params_mount",
            original_model_params_mount,
        )
        object.__setattr__(
            caliby_app.APP_INFO,
            "full_model_volume_mount",
            original_full_model_volume_mount,
        )
        object.__setattr__(
            caliby_app.APP_INFO, "proteinmpnn_model_source", original_proteinmpnn_source
        )
        object.__setattr__(
            caliby_app.APP_INFO,
            "protpardelle_model_source",
            original_protpardelle_source,
        )

    assert links == [
        (proteinmpnn_source, model_params_mount / "proteinmpnn"),
        (protpardelle_source, model_params_mount / "protpardelle-1c"),
    ]
    assert commands
    assert f"model_params_path={model_params_mount}" in commands[0]
    assert result["conformer_dir"].endswith("/demo/outputs/ensembles/cc95-epoch3490")
    assert fake_volume.reload_count == 1
    assert fake_volume.commit_count >= 1


class _FakeRemote:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls
        self.args: list[tuple] = []
        self.kwargs: list[dict] = []

    def remote(self, *args, **kwargs):
        self.calls.append(self.name)
        self.args.append(args)
        self.kwargs.append(kwargs)
        if self.name == "package":
            return b"archive"
        if self.name == "clean":
            return {"cleaned_structures_dir": "/remote/cleaned_structures"}
        if self.name == "generate":
            return {"conformer_dir": "/remote/ensembles/generated"}
        return {f"{self.name}_result": "ok"}


def test_submit_design_yaml_composes_default_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    input_yaml = tmp_path / "caliby_design.yaml"
    input_yaml.write_text(
        f"input_path: {structure}\nrun_name: demo\nnum_seqs_per_pdb: 4\n",
    )
    calls: list[str] = []

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(caliby_app, "run_caliby_seq_des", _FakeRemote("design", calls))
    monkeypatch.setattr(caliby_app, "run_caliby_score", _FakeRemote("score", calls))
    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="design",
        out_dir=None,
    )

    assert calls == ["design", "score"]


def test_submit_design_yaml_honors_postprocessing_switches(
    monkeypatch, tmp_path: Path
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    input_yaml = tmp_path / "caliby_design.yaml"
    input_yaml.write_text(
        f"input_path: {structure}\nrun_name: demo\nscore_baseline: false\n"
    )
    calls: list[str] = []

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(caliby_app, "run_caliby_seq_des", _FakeRemote("design", calls))
    monkeypatch.setattr(caliby_app, "run_caliby_score", _FakeRemote("score", calls))

    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="design",
        out_dir=None,
    )

    assert calls == ["design"]


def test_submit_design_yaml_uploads_and_passes_pdb_name_list(
    monkeypatch, tmp_path: Path
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    pdb_name_list = tmp_path / "pdb_names.txt"
    pdb_name_list.write_text("input.pdb\n")
    input_yaml = tmp_path / "caliby_design.yaml"
    input_yaml.write_text(
        f"input_path: {structure}\nrun_name: demo\npdb_name_list: {pdb_name_list}\n"
    )
    calls: list[str] = []
    design_remote = _FakeRemote("design", calls)
    score_remote = _FakeRemote("score", calls)

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        caliby_app,
        "stage_optional_local_file",
        lambda path, *_args, **_kwargs: f"/remote/{Path(path).name}" if path else None,
    )
    monkeypatch.setattr(caliby_app, "run_caliby_seq_des", design_remote)
    monkeypatch.setattr(caliby_app, "run_caliby_score", score_remote)

    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="design",
        out_dir=None,
    )

    assert calls == ["design", "score"]
    assert design_remote.kwargs[0]["pdb_name_list"] == "/remote/pdb_names.txt"
    assert score_remote.kwargs[0]["pdb_name_list"] == "/remote/pdb_names.txt"


def test_submit_ensemble_design_cleans_inputs_before_generating_ensembles(
    monkeypatch, tmp_path: Path
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    input_yaml = tmp_path / "caliby_ensemble.yaml"
    input_yaml.write_text(
        f"input_path: {structure}\nrun_name: demo\nnum_samples_per_pdb: 8\n",
    )
    calls: list[str] = []

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        caliby_app, "run_caliby_clean_pdbs", _FakeRemote("clean", calls)
    )
    monkeypatch.setattr(
        caliby_app, "run_caliby_generate_ensembles", _FakeRemote("generate", calls)
    )
    monkeypatch.setattr(
        caliby_app, "run_caliby_seq_des_ensemble", _FakeRemote("design", calls)
    )
    monkeypatch.setattr(
        caliby_app, "run_caliby_score_ensemble", _FakeRemote("score", calls)
    )

    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="ensemble_design",
        out_dir=None,
    )

    assert calls == ["clean", "generate", "design", "score"]
    assert caliby_app.run_caliby_generate_ensembles.kwargs[0]["pdb_dir"] == (
        "/remote/cleaned_structures"
    )


def test_submit_design_with_ensemble_yaml_reports_task_mismatch(tmp_path: Path) -> None:
    input_yaml = tmp_path / "caliby_ensemble.yaml"
    input_yaml.write_text(
        "input_path: examples/data/caliby/native_pdbs\n"
        "run_name: demo\n"
        "num_samples_per_pdb: 8\n",
    )

    with pytest.raises(ValidationError) as exc_info:
        caliby_app.submit_caliby_task.info.raw_f(
            input_yaml=str(input_yaml),
            task="design",
            out_dir=None,
        )

    assert "Input YAML does not match task='design'" in str(exc_info.value)
    assert "num_samples_per_pdb" in str(exc_info.value)
    assert "--task ensemble_design" in str(exc_info.value)


def test_submit_design_prints_modal_volume_path_without_local_download(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    input_yaml = tmp_path / "caliby_design.yaml"
    input_yaml.write_text(f"input_path: {structure}\nrun_name: demo\n")
    calls: list[str] = []

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(caliby_app, "run_caliby_seq_des", _FakeRemote("design", calls))
    monkeypatch.setattr(caliby_app, "run_caliby_score", _FakeRemote("score", calls))

    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="design",
        out_dir=None,
    )

    assert "Caliby results available in Modal volume" in capsys.readouterr().out


def test_submit_design_download_uses_registered_package_helper(
    monkeypatch, tmp_path: Path
) -> None:
    structure = tmp_path / "input.pdb"
    structure.write_text("ATOM\n")
    input_yaml = tmp_path / "caliby_design.yaml"
    input_yaml.write_text(f"input_path: {structure}\nrun_name: demo\n")
    calls: list[str] = []
    package_remote = _FakeRemote("package", calls)

    monkeypatch.setattr(
        caliby_app, "stage_local_structures", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(caliby_app, "run_caliby_seq_des", _FakeRemote("design", calls))
    monkeypatch.setattr(caliby_app, "run_caliby_score", _FakeRemote("score", calls))
    monkeypatch.setattr(caliby_app, "package_outputs_helper", package_remote)

    caliby_app.submit_caliby_task.info.raw_f(
        input_yaml=str(input_yaml),
        task="design",
        out_dir=str(tmp_path),
    )

    assert calls == ["design", "score", "package"]
    assert package_remote.args[0][0].endswith("/demo")
    assert package_remote.kwargs[0] == {}


def test_app_help_accepts_caliby() -> None:
    result = CliRunner().invoke(cli_app, ["app", "help", "caliby"])

    assert result.exit_code == 0
    assert "Help for app 'caliby'" in result.output
    assert "--input-yaml" in result.output
    assert "seq_des_outputs.csv" in result.output
    assert "score_outputs.csv" in result.output
    assert "run_caliby_generate_ensembles" in result.output
    assert "run_caliby_seq_des_ensemble" in result.output
    assert "sidechain" not in result.output.lower()
    assert "--pdb-name-list" not in result.output
    assert "--clean-inputs" not in result.output
    assert "--batch-size" not in result.output
    assert "--num-workers" not in result.output
    assert "--download-models" not in result.output


def test_caliby_example_assets_are_present() -> None:
    data_root = Path("examples/data/caliby")

    assert (data_root / "native_pdbs" / "7urp.cif").is_file()
    assert (data_root / "native_pdbs" / "7xhz.cif").is_file()
    assert (data_root / "native_pdbs" / "7xz3.cif").is_file()
    assert (data_root / "native_pdbs" / "8huz.cif").is_file()
    assert (data_root / "native_pdbs" / "8sot.cif").is_file()
    assert (data_root / "pdb_name_lists" / "2_native_pdbs.txt").is_file()
    assert (data_root / "caliby_design.yaml").is_file()
    assert (data_root / "caliby_ensemble_design.yaml").is_file()
    assert (data_root / "pos_constraint_csvs" / "native_pdb_constraints.csv").is_file()


def test_caliby_example_constraints_match_selected_pdbs() -> None:
    data_root = Path("examples/data/caliby")
    for yaml_path in [
        data_root / "caliby_design.yaml",
        data_root / "caliby_ensemble_design.yaml",
    ]:
        payload = yaml.safe_load(yaml_path.read_text()) or {}
        pdb_name_list = payload.get("pdb_name_list")
        pos_constraint_csv = payload.get("pos_constraint_csv")
        if pdb_name_list is None or pos_constraint_csv is None:
            continue

        selected_keys = {
            Path(line.strip()).stem
            for line in (Path.cwd() / pdb_name_list).read_text().splitlines()
            if line.strip()
        }
        constrained_keys = set(
            pl
            .read_csv(Path.cwd() / pos_constraint_csv)
            .get_column("pdb_key")
            .cast(str)
            .to_list()
        )
        assert selected_keys & constrained_keys, yaml_path


def test_caliby_example_script_uses_caliby_data_and_run_command() -> None:
    source = Path("examples/app/caliby.sh").read_text()

    assert "../data/caliby/caliby_design.yaml" in source
    assert "../data/caliby/caliby_ensemble_design.yaml" in source
    assert "--input-yaml" in source
    assert 'ENTRY_BIN=$(realpath "${BIOMODALS_ROOT}/biomodals")' in source
    assert "app r caliby" in source
    assert "uv run biomodals" not in source
    assert "download_models" not in source
    assert "existing_conformers" not in source
    assert "sidechain" not in source


def test_app_file_uses_reference_section_order() -> None:
    source = Path(caliby_app.__file__).read_text()
    sections = [
        "# Modal configs",
        "# Image and app definitions",
        "# Helper functions",
        "# Inference functions",
        "# Entrypoint for persistent usage",
    ]

    positions = [source.index(section) for section in sections]
    assert positions == sorted(positions)
