"""Caliby source repo: <https://github.com/ProteinDesignLab/caliby>.

Caliby designs protein sequences conditioned on static structures or structural
ensembles.

The local entrypoint takes a YAML specification and validates it with Pydantic.
Use `task=design` for direct sequence design with `seq_des.py`, or
`task=ensemble_design` for the core Caliby ensemble pipeline using
`generate_ensembles.py` and `seq_des_ensemble.py`.

## Outputs

Results are saved in the `Caliby-outputs` Modal volume under `<run-name>/`.
If `--out-dir` is provided, a packaged `.tar.zst` copy is downloaded locally.

* `outputs/design/seq_des_outputs.csv` contains designed sequences, designed
  structure paths, and the Caliby Potts energy `U` for each design. Lower `U`
  values are more favorable under Caliby's learned energy model, but they are
  not experimental validation or independent structure-prediction confidence.
* `outputs/baseline_score/score_outputs.csv` contains native/input baseline
  scores when `score_baseline` is enabled in the YAML.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import modal
import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import build_local_output_path, write_local_tarball
from biomodals.helper.shell import (
    package_outputs,
    run_command_with_log,
    sanitize_filename,
    softlink_dir,
)
from biomodals.helper.volume_run import (
    build_volume_run_paths,
    volume_path_from_mount_path,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Caliby",
    repo_url="https://github.com/ProteinDesignLab/caliby",
    repo_commit_hash="8136f57d912dadeb64f9f60cb1021c06791b271d",
    package_name="caliby",
    python_version="3.12",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "86400")),
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Container for Caliby-specific information and defaults."""

    structure_suffixes: tuple[str, ...] = (".pdb", ".cif")
    valid_tasks: frozenset[str] = frozenset({"design", "ensemble_design"})
    design_model_by_task: Mapping[str, str] = field(
        default_factory=lambda: {
            "design": "soluble_caliby_v1",
            "ensemble_design": "caliby",
        }
    )
    default_batch_size: int = 4
    default_num_workers: int = 2
    default_ensemble_batch_size: int = 8
    default_num_samples_per_pdb: int = 32
    default_max_num_conformers: int = 32
    default_include_primary_conformer: bool = True
    default_score_baseline: bool = True
    default_sampling_yaml_path: str | None = None
    default_seed: int = 0
    remote_path_prefix: str = "remote:"
    caliby_model_params_mount: Path = Path("/mnt/caliby-model-params")
    full_model_volume_mount: Path = Path("/mnt/biomodals-store")
    proteinmpnn_model_source: Path = Path("/mnt/biomodals-store/LigandMPNN")
    protpardelle_model_source: Path = Path(
        "/mnt/biomodals-store/Caliby/protpardelle-1c"
    )
    protpardelle_sampling_yaml_path: str = (
        "caliby/configs/protpardelle-1c/multichain_backbone_partial_diffusion.yaml"
    )


APP_INFO = AppInfo()

##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "zstd")
    .env(CONF.default_env | {"MODEL_PARAMS_DIR": str(CONF.model_volume_mountpoint)})
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(str(CONF.git_clone_dir))
    .pipe(patch_image_for_helper)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
class CalibyCommonConfig(BaseModel):
    """Common Caliby YAML fields."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    num_seqs_per_pdb: int = Field(default=16, gt=0)
    pos_constraint_csv: str | None = None
    pdb_name_list: str | None = None
    score_baseline: bool = APP_INFO.default_score_baseline
    sampling_cfg_overrides: dict[str, object] = Field(default_factory=dict)


class CalibyDesignConfig(CalibyCommonConfig):
    """YAML schema for direct Caliby sequence design."""

    input_path: str


class CalibyEnsembleDesignConfig(CalibyCommonConfig):
    """YAML schema for Caliby ensemble-conditioned sequence design."""

    input_path: str
    num_samples_per_pdb: int = Field(default=APP_INFO.default_num_samples_per_pdb, gt=0)
    max_num_conformers: int = Field(default=APP_INFO.default_max_num_conformers, gt=0)
    include_primary_conformer: bool = APP_INFO.default_include_primary_conformer
    sampling_yaml_path: str | None = APP_INFO.default_sampling_yaml_path
    seed: int = APP_INFO.default_seed


def discover_structure_files(input_path: str | Path) -> list[Path]:
    """Return sorted local PDB/CIF-like structure files."""
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        if root.name.endswith(APP_INFO.structure_suffixes):
            return [root]
        msg = f"Unsupported structure file suffix: {root.name}"
        raise ValueError(msg)
    if not root.is_dir():
        msg = f"Input path does not exist: {input_path}"
        raise FileNotFoundError(msg)
    paths = [
        path.resolve()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name.endswith(APP_INFO.structure_suffixes)
    ]
    if not paths:
        msg = f"No PDB/CIF structure files found under {root}"
        raise FileNotFoundError(msg)
    return paths


def build_run_paths(run_name: str) -> dict[str, Path]:
    """Build persistent Caliby output paths for one run."""
    safe_run_name = sanitize_filename(run_name)
    run_paths = build_volume_run_paths(CONF.output_volume_mountpoint, safe_run_name)
    run_root = run_paths["run_root"]
    inputs_dir = run_paths["inputs_dir"]
    outputs_dir = run_paths["output_dir"]
    return {
        "run_root": run_root,
        "inputs_dir": inputs_dir,
        "structures_dir": inputs_dir / "structures",
        "outputs_dir": outputs_dir,
        "design_dir": outputs_dir / "design",
        "baseline_score_dir": outputs_dir / "baseline_score",
        "ensembles_dir": outputs_dir / "ensembles",
        "logs_dir": run_root / "logs",
    }


def _output_volume_path(path: Path) -> str:
    return volume_path_from_mount_path(
        str(path),
        str(CONF.output_volume_mountpoint),
        CONF.output_volume_name,
    ).path


def _ensure_stage_output(path: Path, expected_file: str, stage_name: str) -> None:
    if not (path / expected_file).is_file():
        msg = (
            f"Caliby {stage_name} did not produce expected file: {path / expected_file}"
        )
        raise RuntimeError(msg)


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    return orjson.dumps(value).decode()


def _sampling_override_args(
    overrides: Mapping[str, object],
    prefix: str = "sampling_cfg_overrides",
) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            args.extend(_sampling_override_args(value, f"{prefix}.{key}"))
        else:
            args.append(f"++{prefix}.{key}={_hydra_value(value)}")
    return args


def stage_local_structures(input_path: str | Path, run_name: str) -> list[str]:
    """Stage local structures into the Caliby output volume."""
    paths = build_run_paths(run_name)
    staged: list[str] = []
    with CONF.output_volume.batch_upload(force=True) as batch:
        for source in discover_structure_files(input_path):
            dest = paths["structures_dir"] / source.name
            volume_path = volume_path_from_mount_path(
                str(dest),
                str(CONF.output_volume_mountpoint),
                CONF.output_volume_name,
            )
            batch.put_file(source, f"/{volume_path.path}")
            staged.append(str(dest))
    return staged


def stage_optional_local_file(
    file_path: str | Path | None,
    run_name: str,
    input_subdir: str,
) -> str | None:
    """Upload a local optional config file, or return an explicit remote path."""
    if file_path is None:
        return None
    raw_path = str(file_path)
    if raw_path.startswith(APP_INFO.remote_path_prefix):
        remote_path = raw_path.removeprefix(APP_INFO.remote_path_prefix)
        if not remote_path:
            msg = f"Remote {input_subdir} path is empty."
            raise ValueError(msg)
        return remote_path
    path = Path(file_path).expanduser()
    if not path.exists():
        msg = (
            f"Local {input_subdir} file does not exist: {file_path}. "
            f"Use '{APP_INFO.remote_path_prefix}<path>' for an existing remote or "
            "container path."
        )
        raise FileNotFoundError(msg)
    source = path.resolve()
    if not source.is_file():
        msg = f"Expected a file for {input_subdir}: {source}"
        raise ValueError(msg)
    dest = build_run_paths(run_name)["inputs_dir"] / input_subdir / source.name
    with CONF.output_volume.batch_upload(force=True) as batch:
        volume_path = volume_path_from_mount_path(
            str(dest),
            str(CONF.output_volume_mountpoint),
            CONF.output_volume_name,
        )
        batch.put_file(source, f"/{volume_path.path}")
    return str(dest)


def validate_task_config(payload: dict, task: str):
    """Validate one Caliby YAML payload against the requested task schema."""
    schema_by_task = {
        "design": CalibyDesignConfig,
        "ensemble_design": CalibyEnsembleDesignConfig,
    }
    try:
        return schema_by_task[task].model_validate(payload)
    except ValidationError as exc:
        invalid_fields = sorted({
            str(error["loc"][0])
            for error in exc.errors()
            if error.get("type") == "extra_forbidden" and error.get("loc")
        })
        if invalid_fields:
            other_task = next(
                candidate for candidate in APP_INFO.valid_tasks if candidate != task
            )
            field_list = ", ".join(invalid_fields)
            msg = (
                f"Input YAML does not match task='{task}'. This YAML contains "
                f"fields that are not valid for this task: {field_list}. "
                f"Use --task {other_task} if this is a {other_task} YAML, or "
                f"pass a YAML file for task='{task}'."
            )
            raise ValidationError.from_exception_data(
                title=exc.title,
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("input_yaml",),
                        "msg": msg,
                        "input": payload,
                        "ctx": {"error": ValueError(msg)},
                    }
                ],
            ) from exc
        raise


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_caliby_seq_des(
    run_name: str,
    ckpt_name: str,
    num_seqs_per_pdb: int,
    batch_size: int,
    num_workers: int,
    pos_constraint_csv: str | None,
    pdb_name_list: str | None,
    sampling_cfg_overrides: dict[str, object],
) -> dict[str, str]:
    """Run upstream Caliby single-structure sequence design."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    paths["design_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    constraint_path = Path(pos_constraint_csv) if pos_constraint_csv else None
    cmd = [
        sys.executable,
        "-m",
        "caliby.eval.sampling.seq_des",
        f"ckpt_name_or_path={ckpt_name}",
        f"input_cfg.pdb_dir={paths['structures_dir']}",
        f"sampling_cfg_overrides.num_seqs_per_pdb={num_seqs_per_pdb}",
        f"sampling_cfg_overrides.batch_size={batch_size}",
        f"num_workers={num_workers}",
        f"out_dir={paths['design_dir']}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    if constraint_path is not None:
        cmd.append(f"pos_constraint_csv={constraint_path}")
    cmd.extend(_sampling_override_args(sampling_cfg_overrides))
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "seq_des.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        _ensure_stage_output(paths["design_dir"], "seq_des_outputs.csv", "seq_des")
    finally:
        CONF.output_volume.commit()
    return {
        "design_outputs_dir": str(paths["design_dir"]),
        "design_outputs_volume_path": _output_volume_path(paths["design_dir"]),
    }


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_caliby_seq_des_ensemble(
    run_name: str,
    conformer_dir: str,
    ckpt_name: str,
    num_seqs_per_pdb: int,
    batch_size: int,
    num_workers: int,
    max_num_conformers: int,
    include_primary_conformer: bool,
    pos_constraint_csv: str | None,
    pdb_name_list: str | None,
    sampling_cfg_overrides: dict[str, object],
) -> dict[str, str]:
    """Run upstream Caliby ensemble-conditioned sequence design."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    paths["design_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    constraint_path = Path(pos_constraint_csv) if pos_constraint_csv else None
    cmd = [
        sys.executable,
        "-m",
        "caliby.eval.sampling.seq_des_ensemble",
        f"ckpt_name_or_path={ckpt_name}",
        f"input_cfg.conformer_dir={Path(conformer_dir)}",
        f"sampling_cfg_overrides.num_seqs_per_pdb={num_seqs_per_pdb}",
        f"sampling_cfg_overrides.batch_size={batch_size}",
        f"num_workers={num_workers}",
        f"max_num_conformers={max_num_conformers}",
        f"include_primary_conformer={str(include_primary_conformer).lower()}",
        f"out_dir={paths['design_dir']}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    if constraint_path is not None:
        cmd.append(f"pos_constraint_csv={constraint_path}")
    cmd.extend(_sampling_override_args(sampling_cfg_overrides))
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "seq_des_ensemble.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        _ensure_stage_output(
            paths["design_dir"], "seq_des_outputs.csv", "seq_des_ensemble"
        )
    finally:
        CONF.output_volume.commit()
    return {
        "design_outputs_dir": str(paths["design_dir"]),
        "design_outputs_volume_path": _output_volume_path(paths["design_dir"]),
    }


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_caliby_score(
    run_name: str,
    ckpt_name: str,
    num_workers: int,
    pdb_name_list: str | None,
) -> dict[str, str]:
    """Run upstream Caliby single-structure baseline scoring."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    paths["baseline_score_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "caliby.eval.sampling.score",
        f"ckpt_name_or_path={ckpt_name}",
        f"input_cfg.pdb_dir={paths['structures_dir']}",
        f"num_workers={num_workers}",
        f"out_dir={paths['baseline_score_dir']}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "score.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        _ensure_stage_output(paths["baseline_score_dir"], "score_outputs.csv", "score")
    finally:
        CONF.output_volume.commit()
    return {
        "baseline_score_outputs_dir": str(paths["baseline_score_dir"]),
        "baseline_score_outputs_volume_path": _output_volume_path(
            paths["baseline_score_dir"]
        ),
    }


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_caliby_score_ensemble(
    run_name: str,
    conformer_dir: str,
    ckpt_name: str,
    num_workers: int,
    max_num_conformers: int,
    include_primary_conformer: bool,
    pdb_name_list: str | None,
) -> dict[str, str]:
    """Run upstream Caliby ensemble baseline scoring."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    paths["baseline_score_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "caliby.eval.sampling.score_ensemble",
        f"ckpt_name_or_path={ckpt_name}",
        f"input_cfg.conformer_dir={Path(conformer_dir)}",
        f"num_workers={num_workers}",
        f"max_num_conformers={max_num_conformers}",
        f"include_primary_conformer={str(include_primary_conformer).lower()}",
        f"out_dir={paths['baseline_score_dir']}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "score_ensemble.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        _ensure_stage_output(
            paths["baseline_score_dir"], "score_outputs.csv", "score_ensemble"
        )
    finally:
        CONF.output_volume.commit()
    return {
        "baseline_score_outputs_dir": str(paths["baseline_score_dir"]),
        "baseline_score_outputs_volume_path": _output_volume_path(
            paths["baseline_score_dir"]
        ),
    }


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True)
    | {
        str(APP_INFO.full_model_volume_mount): MODEL_VOLUME.with_mount_options(
            read_only=True
        )
    },
)
def run_caliby_generate_ensembles(
    run_name: str,
    pdb_dir: str,
    num_samples_per_pdb: int,
    batch_size: int,
    sampling_yaml_path: str | None,
    seed: int,
    pdb_name_list: str | None,
) -> dict[str, str]:
    """Run upstream Caliby Protpardelle-1c ensemble generation."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    paths["ensembles_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    input_dir = Path(pdb_dir)
    if not input_dir.is_dir():
        msg = f"Caliby ensemble-generation input directory is missing: {input_dir}"
        raise FileNotFoundError(msg)
    model_links = {
        APP_INFO.proteinmpnn_model_source: APP_INFO.caliby_model_params_mount
        / "proteinmpnn",
        APP_INFO.protpardelle_model_source: APP_INFO.caliby_model_params_mount
        / "protpardelle-1c",
    }
    for source_dir, link_path in model_links.items():
        if not source_dir.is_dir():
            msg = f"Caliby ensemble model directory is missing: {source_dir}"
            raise FileNotFoundError(msg)
        # Modal rejects mounting the same volume twice in one function, so the
        # shared model volume is mounted once and adapted to Caliby's expected
        # Protpardelle model_params_path layout with local symlinks.
        softlink_dir(source_dir, link_path)
    yaml_path = sampling_yaml_path or APP_INFO.protpardelle_sampling_yaml_path
    cmd = [
        sys.executable,
        "-m",
        "caliby.eval.sampling.generate_ensembles",
        f"model_params_path={APP_INFO.caliby_model_params_mount}",
        f"sampling_yaml_path={yaml_path}",
        f"input_cfg.pdb_dir={input_dir}",
        f"num_samples_per_pdb={num_samples_per_pdb}",
        f"batch_size={batch_size}",
        f"seed={seed}",
        f"out_dir={paths['ensembles_dir']}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "generate_ensembles.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        generated_roots = [
            path
            for path in sorted(paths["ensembles_dir"].iterdir())
            if path.is_dir() and path.name != "protpardelle_outputs_temp"
        ]
        if len(generated_roots) != 1:
            msg = (
                "Caliby generate_ensembles should produce exactly one conformer "
                f"directory under {paths['ensembles_dir']}; found {generated_roots}."
            )
            raise RuntimeError(msg)
        conformer_dir = generated_roots[0]
    finally:
        CONF.output_volume.commit()
    return {
        "conformer_dir": str(conformer_dir),
        "conformer_dir_volume_path": _output_volume_path(conformer_dir),
    }


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def run_caliby_clean_pdbs(
    run_name: str,
    num_workers: int,
    pdb_name_list: str | None,
) -> dict[str, str]:
    """Run upstream Caliby structure cleaning before ensemble generation."""
    CONF.output_volume.reload()
    paths = build_run_paths(run_name)
    cleaned_dir = paths["structures_dir"] / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "caliby.data.preprocessing.atomworks.clean_pdbs",
        f"input_cfg.pdb_dir={paths['structures_dir']}",
        f"num_workers={num_workers}",
        f"out_dir={cleaned_dir}",
    ]
    if pdb_name_list is not None:
        cmd.append(f"input_cfg.pdb_name_list={Path(pdb_name_list)}")
    try:
        run_command_with_log(
            cmd,
            log_file=paths["logs_dir"] / "clean_pdbs.log",
            verbose=True,
            cwd=CONF.git_clone_dir,
        )
        cleaned_files = sorted(
            path
            for path in cleaned_dir.iterdir()
            if path.is_file() and path.suffix in {".cif", ".mmcif"}
        )
        if not cleaned_files:
            msg = f"Caliby clean_pdbs produced no CIF/mmCIF files: {cleaned_dir}"
            raise RuntimeError(msg)
    finally:
        CONF.output_volume.commit()
    return {
        "cleaned_structures_dir": str(cleaned_dir),
        "cleaned_structures_volume_path": _output_volume_path(cleaned_dir),
    }


@app.function(
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def package_outputs_helper(root: str | Path) -> bytes:
    """Package a mounted output directory into a tar.zst archive."""
    return package_outputs(root)


##########################################
# Entrypoint for persistent usage
##########################################
@app.local_entrypoint()
def submit_caliby_task(
    input_yaml: str,
    task: str = "ensemble_design",
    out_dir: str | None = ".",
) -> None:
    """Run Caliby design from a YAML specification.

    Args:
        input_yaml: Path to a YAML file validated by `CalibyDesignConfig` when
            `task` is `design`, or `CalibyEnsembleDesignConfig` when `task` is
            `ensemble_design`. Local optional files referenced by the YAML fail
            fast when missing; use `remote:<path>` for existing remote/container
            files.
        task: One of `design` or `ensemble_design`.
        out_dir: Optional local directory for packaged results. Defaults to the
            current directory. Set to an empty value from Python to leave results
            only in the Modal volume.
    """
    if task not in APP_INFO.valid_tasks:
        raise ValueError(f"task must be one of {sorted(APP_INFO.valid_tasks)}.")

    import yaml

    with Path(input_yaml).expanduser().resolve().open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        msg = f"Caliby input YAML must contain a mapping: {input_yaml}"
        raise ValueError(msg)
    config = validate_task_config(payload, task)
    run_name = sanitize_filename(config.run_name)
    remote_pos_constraint_csv = stage_optional_local_file(
        config.pos_constraint_csv,
        run_name,
        "constraints",
    )
    remote_pdb_name_list = stage_optional_local_file(
        config.pdb_name_list,
        run_name,
        "pdb_name_lists",
    )

    if task == "design":
        if not isinstance(config, CalibyDesignConfig):
            raise TypeError("design task requires CalibyDesignConfig.")
        stage_local_structures(config.input_path, run_name)
        run_caliby_seq_des.remote(
            run_name=run_name,
            ckpt_name=APP_INFO.design_model_by_task["design"],
            num_seqs_per_pdb=config.num_seqs_per_pdb,
            batch_size=APP_INFO.default_batch_size,
            num_workers=APP_INFO.default_num_workers,
            pos_constraint_csv=remote_pos_constraint_csv,
            pdb_name_list=remote_pdb_name_list,
            sampling_cfg_overrides=config.sampling_cfg_overrides,
        )
        if config.score_baseline:
            run_caliby_score.remote(
                run_name=run_name,
                ckpt_name=APP_INFO.design_model_by_task["design"],
                num_workers=APP_INFO.default_num_workers,
                pdb_name_list=remote_pdb_name_list,
            )

    if task == "ensemble_design":
        if not isinstance(config, CalibyEnsembleDesignConfig):
            raise TypeError("ensemble_design task requires CalibyEnsembleDesignConfig.")
        stage_local_structures(config.input_path, run_name)
        clean_result = run_caliby_clean_pdbs.remote(
            run_name=run_name,
            num_workers=APP_INFO.default_num_workers,
            pdb_name_list=remote_pdb_name_list,
        )
        generate_result = run_caliby_generate_ensembles.remote(
            run_name=run_name,
            pdb_dir=clean_result["cleaned_structures_dir"],
            num_samples_per_pdb=config.num_samples_per_pdb,
            batch_size=APP_INFO.default_ensemble_batch_size,
            sampling_yaml_path=config.sampling_yaml_path,
            seed=config.seed,
            pdb_name_list=remote_pdb_name_list,
        )
        conformer_dir = generate_result["conformer_dir"]
        run_caliby_seq_des_ensemble.remote(
            run_name=run_name,
            conformer_dir=conformer_dir,
            ckpt_name=APP_INFO.design_model_by_task["ensemble_design"],
            num_seqs_per_pdb=config.num_seqs_per_pdb,
            batch_size=APP_INFO.default_batch_size,
            num_workers=APP_INFO.default_num_workers,
            max_num_conformers=config.max_num_conformers,
            include_primary_conformer=config.include_primary_conformer,
            pos_constraint_csv=remote_pos_constraint_csv,
            pdb_name_list=remote_pdb_name_list,
            sampling_cfg_overrides=config.sampling_cfg_overrides,
        )
        if config.score_baseline:
            run_caliby_score_ensemble.remote(
                run_name=run_name,
                conformer_dir=conformer_dir,
                ckpt_name=APP_INFO.design_model_by_task["ensemble_design"],
                num_workers=APP_INFO.default_num_workers,
                max_num_conformers=config.max_num_conformers,
                include_primary_conformer=config.include_primary_conformer,
                pdb_name_list=remote_pdb_name_list,
            )

    if out_dir is not None:
        archive = package_outputs_helper.remote(
            str(build_run_paths(run_name)["run_root"])
        )
        local_archive = build_local_output_path(out_dir, run_name=run_name)
        write_local_tarball(local_archive, archive)
        print(f"🧬 Caliby results written to {local_archive}")
    run_root_volume_path = volume_path_from_mount_path(
        str(build_run_paths(run_name)["run_root"]),
        str(CONF.output_volume_mountpoint),
        CONF.output_volume_name,
    )
    print(
        "🧬 Caliby results available in Modal volume "
        f"{run_root_volume_path.volume_name}:{run_root_volume_path.path}"
    )
