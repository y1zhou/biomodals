"""PPIFlow source repo: <https://github.com/Mingchenchen/PPIFlow/tree/main/tool/PPIFlow>."""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from functools import cached_property
from pathlib import Path

import modal
from pydantic import BaseModel, computed_field, model_validator

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT, MODEL_VOLUME, MODEL_VOLUME_NAME
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import (
    package_outputs,
    run_command_with_log,
    sanitize_filename,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="PPIFlow",
    repo_url="https://github.com/y1zhou/PPIFlow",
    repo_commit_hash="625b3cf9cf1f52fa24d99cf9bcec3aa8a891d7a7",
    package_name="ppiflow",
    python_version="3.11",
    cuda_version="cu126",
    gpu=os.environ.get("GPU", "L40S"),
)

# Volumes to be mounted
OUTPUTS_VOLUME = CONF.get_out_volume()
OUTPUTS_VOLUME_NAME = OUTPUTS_VOLUME.name or f"{CONF.name}-outputs"
SCRIPTS_DIR = CONF.git_clone_dir / "tool" / "PPIFlow"

##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    modal.Image.micromamba(python_version=CONF.python_version)
    .apt_install("git", "build-essential")
    .env(CONF.default_env)
    .run_commands(
        " && ".join(
            (
                f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
                f"cd {CONF.git_clone_dir}",
                f"git checkout {CONF.repo_commit_hash}",
                # f"micromamba env create -f {CONF.git_clone_dir / 'ppiflow_af3_merged.yaml'}",
            )
        )
    )
    .micromamba_install(
        [
            "cuda-nvcc=12.6",
            "gcc_linux-64=13",
            "gxx_linux-64=13",
            "ninja",
            "lmdb=0.9.31",
            "openmm",
            "pdbfixer",
        ],
        channels=["conda-forge", "bioconda", "nvidia"],
    )
    .uv_pip_install(
        "pyrosetta==2026.3+releasequarterly.5e498f1409",
        find_links="https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization",
    )
    .uv_pip_install(
        "absl-py==2.1.0",
        "biopython==1.87",
        "biotite==1.0.1",
        "biotraj==1.2.2",
        "chex==0.1.87",
        "deepspeed==0.18.8",
        "dm-haiku==0.0.13",
        "dm-tree==0.1.8",
        "e3nn==0.5.6",
        "easydict==1.13",
        "einops==0.8.1",
        "fair-esm==2.0.0",
        "freesasa==2.2.1",
        "gemmi==0.6.5",
        "gpustat==1.1.1",
        "gputil==1.4.0",
        "h5py==3.16.0",
        "hjson==3.1.0",
        "hydra-core==1.3.2",
        "jax==0.4.34",
        "jax-cuda12-pjrt==0.4.34",
        "jax-cuda12-plugin==0.4.34",
        "jax-triton==0.2.0",
        "jaxlib==0.4.34",
        "jmp==0.0.4",
        "lightning==2.6.1",
        "matplotlib==3.9.2",
        "mdtraj==1.10.3",
        "ml-collections==1.0.0",
        "modelcif==0.7",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "optree==0.14.1",
        "pandas==2.2.3",
        "pytorch-lightning==2.6.1",
        "pyyaml==6.0.2",
        "scipy==1.15.2",
        "seaborn==0.13.2",
        "scikit-learn==1.8.0",
        "tensorboard==2.19.0",
        "tmtools==0.2.0",
        "tqdm==4.67.1",
        "wandb==0.19.8",
        f"torch==2.10.0+{CONF.cuda_version}",
        f"pyg-lib==0.6.0+pt210{CONF.cuda_version}",
        f"torch-cluster==1.6.3+pt210{CONF.cuda_version}",
        "torch-geometric==2.7.0",
        f"torch-scatter==2.1.2+pt210{CONF.cuda_version}",
        f"torch-sparse==0.6.18+pt210{CONF.cuda_version}",
        find_links=f"https://data.pyg.org/whl/torch-2.10.0+{CONF.cuda_version}.html",
    )
    .workdir(str(SCRIPTS_DIR))
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
class CommonConfig(BaseModel):
    """Common input args for PPIFlow scripts."""

    name: str  # Test target name
    specified_hotspots: str  # Comma-separated, <chain><1-based resi>, e.g. "A123,B45"
    samples_per_target: int = 100  # Number of samples to generate


class SampleAntibodyNanobodyConfig(CommonConfig):
    """Input args for sample_antibody_nanobody.py.

    Note the `framework_pdb` file should only contain coordinates of the
    framework region and **not** the CDRs, otherwise the model would raise errors.
    """

    antigen_pdb: str | Path  # Input antigen protein PDB file path
    antigen_chain: str  # Chain ID of the antigen
    framework_pdb: str | Path
    heavy_chain: str
    light_chain: str | None = None  # Leave empty for nanobody design
    cdr_length: str = "CDRH1,5-12,CDRH2,4-17,CDRH3,5-26,CDRL1,5-12,CDRL2,3-10,CDRL3,4-13"  # CDR lengths to sample
    config: str | Path = (
        SCRIPTS_DIR / "configs" / "inference_nanobody.yaml"
    )  # Path to config in the PPIFlow repo


class SampleBinderConfig(CommonConfig):
    """Input args for sample_binder.py.

    Note that PPIFlow also supports `input_csv` to replace `input_pdb` and
    `target_chain`, but for simplicity we only support the PDB input mode here.
    """

    input_pdb: str | Path
    target_chain: str = "R"
    binder_chain: str
    samples_min_length: int = 50  # min(number of residues) per sample
    samples_max_length: int = 100  # max(number of residues) per sample
    sample_hotspot_rate_min: float = 0.2  # minimum hotspot sampling rate
    sample_hotspot_rate_max: float = 0.5  # maximum hotspot sampling rate
    config: str | Path = (
        SCRIPTS_DIR / "configs" / "inference_binder.yaml"
    )  # Path to config in the PPIFlow repo


class SampleAntibodyNanobodyPartialConfig(CommonConfig):
    """Input args for sample_antibody_nanobody_partial.py."""

    complex_pdb: str | Path
    fixed_positions: str  # Key residues to fix in complex_pdb. Format: 'H26,H27,H28,L50-63' (chain ID + residue number, '-' for ranges)."
    cdr_position: str  # Specify CDR residues, e.g. 'H26-32,H45-56,H97-113'
    antigen_chain: str  # Chain ID of the antigen
    heavy_chain: str
    light_chain: str | None = None  # Leave empty for nanobody design
    start_t: float  # starting t value for sampling
    retry_Limit: int = 10  # Maximum retry attempts if sampling fails
    config: str | Path = (
        SCRIPTS_DIR / "configs" / "inference_nanobody.yaml"
    )  # Path to config in the PPIFlow repo

    @model_validator(mode="after")
    def validate_start_t(self):
        """Ensure start_t is between 0 and 1."""
        if not (0 <= self.start_t <= 1):
            raise ValueError("start_t must be between 0 and 1.")
        return self


class SampleBinderPartialConfig(CommonConfig):
    """Input args for sample_binder_partial.py."""

    input_pdb: str | Path
    target_chain: str = "R"
    binder_chain: str = "L"
    fixed_positions: str  # Key residues to fix in input_pdb. e.g. 'L19-27,L31'
    interface_dist: float = 6.0  # interface distance between target and binder
    start_t: float = 0.15  # starting t value for sampling
    sample_hotspot_rate_min: float = 0.2  # minimum hotspot sampling rate
    sample_hotspot_rate_max: float = 0.5  # maximum hotspot sampling rate
    config: str | Path = (
        SCRIPTS_DIR / "configs" / "inference_binder_partial.yaml"
    )  # Path to config in the PPIFlow repo

    @model_validator(mode="after")
    def validate_start_t(self):
        """Ensure start_t is between 0 and 1."""
        if not (0 <= self.start_t <= 1):
            raise ValueError("start_t must be between 0 and 1.")
        return self


class PPIFlowArgs(BaseModel):
    """Input args for ppiflow_run."""

    args: (
        SampleAntibodyNanobodyConfig
        | SampleBinderConfig
        | SampleAntibodyNanobodyPartialConfig
        | SampleBinderPartialConfig
    )

    @computed_field
    @cached_property
    def script_name(self) -> str:
        """Determine which PPIFlow script to run based on the config."""
        if isinstance(self.args, SampleAntibodyNanobodyConfig):
            return "sample_antibody_nanobody.py"
        elif isinstance(self.args, SampleBinderConfig):
            return "sample_binder.py"
        elif isinstance(self.args, SampleAntibodyNanobodyPartialConfig):
            return "sample_antibody_nanobody_partial.py"
        elif isinstance(self.args, SampleBinderPartialConfig):
            return "sample_binder_partial.py"

    @computed_field
    @cached_property
    def model_weights_name(self) -> str:
        """Determine which PPIFlow model weights to use based on the config."""
        if isinstance(
            self.args,
            (SampleAntibodyNanobodyConfig, SampleAntibodyNanobodyPartialConfig),
        ):
            if self.args.light_chain is None:
                return "nanobody.ckpt"
            else:
                return "antibody.ckpt"
        elif isinstance(self.args, (SampleBinderConfig, SampleBinderPartialConfig)):
            return "binder.ckpt"

        else:
            raise ValueError(f"Unsupported config type: {type(self.args)}")


##########################################
# Fetch model weights
##########################################
@app.function(volumes={CONF.model_volume_mountpoint: MODEL_VOLUME}, timeout=MAX_TIMEOUT)
def fetch_model_weights(force: bool = False) -> None:
    """Download PPIFlow models into the mounted volume."""
    model_dir = CONF.model_dir
    base_url = "https://drive.google.com/uc?export=download&confirm=t&id="
    tasks = {
        f"{base_url}1WBSjCTEtia9S1hJ54mYH1PZdDqpLVsgw": model_dir / "antibody.ckpt",
        f"{base_url}1PbpoC7VdkCpoNlxduDhnQ3RuLyWwAuOT": model_dir / "binder.ckpt",
        f"{base_url}1Oo9nbSH3MwT8KIriij5clmnTFrhDJEn5": model_dir / "monomer.ckpt",
        f"{base_url}1aEwzmdlSN9tiIOl5TgM_muHjfFPLue8a": model_dir / "nanobody.ckpt",
    }
    raise RuntimeError(
        "This doesn't work because Google Drive requires confirmation for "
        "large file downloads. Please manually download the model weights and "
        f"place them in Volume {MODEL_VOLUME.name or MODEL_VOLUME_NAME}:\n"
        + "\n".join(
            f"  - {url} -> {path.relative_to(CONF.model_volume_mountpoint)}"
            for url, path in tasks.items()
        )
    )
    # download_files(tasks, force=force, progress_bar_desc="Downloading models...")
    # MODEL_VOLUME.commit()
    # print(f"💊 {CONF.name} model download complete")


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.output_volume_mountpoint: OUTPUTS_VOLUME,
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
    },
)
def ppiflow_run(args: PPIFlowArgs, run_name: str) -> str:
    """Actual remote runner of PPIFlow."""
    import sys

    workdir = Path(CONF.output_volume_mountpoint) / run_name
    out_dir = workdir / "outputs"
    if out_dir.exists():
        print(f"💊 Output path {out_dir} already exists.")
        return str(workdir)

    # Build command
    model_weights_path = CONF.model_dir / args.model_weights_name
    arg_fields = args.args.model_dump(exclude_none=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / args.script_name),
        *(f"--{k}={v}" for k, v in arg_fields.items()),
        f"--output_dir={out_dir}",
        f"--model_weights={model_weights_path}",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / f"{CONF.name}-run.log"
    print(f"💊 Running {CONF.name}, saving logs to {log_path}")
    run_command_with_log(cmd, log_file=log_path)

    OUTPUTS_VOLUME.commit()
    return str(workdir)


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME.read_only()},
)
def ppiflow_run_bytes(
    config_dict: dict[str, object],
    design_mode: str,
    input_files: list[tuple[str, bytes]],
    run_name: str,
) -> bytes:
    """Run PPIFlow from in-memory inputs and return packaged outputs.

    This workflow-oriented helper keeps the existing volume-backed
    ``ppiflow_run`` behavior intact while allowing orchestrators in other Modal
    apps to call deployed PPIFlow without reading the PPIFlow output volume.
    """
    from tempfile import TemporaryDirectory

    safe_run_name = sanitize_filename(run_name)
    with TemporaryDirectory(prefix=f"ppiflow_{safe_run_name}_") as tmpdir:
        workdir = Path(tmpdir) / safe_run_name
        inputs_dir = workdir / "inputs"
        out_dir = workdir / "outputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        for file_name, content in input_files:
            (inputs_dir / Path(file_name).name).write_bytes(content)

        conf_data = dict(config_dict)
        match design_mode:
            case "binder":
                conf_data["input_pdb"] = (
                    inputs_dir / Path(str(conf_data["input_pdb"])).name
                )
                conf = SampleBinderConfig.model_validate(conf_data)
            case "antibody" | "nanobody" | "antibody_nanobody":
                conf_data["antigen_pdb"] = (
                    inputs_dir / Path(str(conf_data["antigen_pdb"])).name
                )
                conf_data["framework_pdb"] = (
                    inputs_dir / Path(str(conf_data["framework_pdb"])).name
                )
                conf = SampleAntibodyNanobodyConfig.model_validate(conf_data)
            case "binder_partial" | "binder_partial_flow":
                conf_data["input_pdb"] = (
                    inputs_dir / Path(str(conf_data["input_pdb"])).name
                )
                conf = SampleBinderPartialConfig.model_validate(conf_data)
            case "antibody_nanobody_partial" | "ab_partial_flow" | "nb_partial_flow":
                conf_data["complex_pdb"] = (
                    inputs_dir / Path(str(conf_data["complex_pdb"])).name
                )
                conf = SampleAntibodyNanobodyPartialConfig.model_validate(conf_data)
            case _:
                raise ValueError(f"Unsupported design_mode: {design_mode}")

        args = PPIFlowArgs(args=conf)
        model_weights_path = CONF.model_dir / args.model_weights_name
        if not model_weights_path.exists():
            raise FileNotFoundError(
                f"PPIFlow model checkpoint is missing: {model_weights_path}"
            )

        arg_fields = args.args.model_dump(exclude_none=True)
        cmd = [
            "python",
            str(SCRIPTS_DIR / args.script_name),
            *(f"--{k}={v}" for k, v in arg_fields.items()),
            f"--output_dir={out_dir}",
            f"--model_weights={model_weights_path}",
        ]
        log_path = workdir / f"{CONF.name}-run.log"
        print(f"💊 Running {CONF.name}, saving logs to {log_path}")
        run_command_with_log(cmd, log_file=log_path)
        return package_outputs(workdir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_ppiflow_task(
    input_yaml: str,
    design_mode: str = "antibody_nanobody",
    out_dir: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
) -> None:
    """Run PPIFlow with results saved as a tarball to `out_dir`.

    Args:
        input_yaml: Path to YAML design specification file. See the `Sample*Config`
            classes in this script for details.
        design_mode: Available scripts are 'antibody_nanobody', 'binder',
            'antibody_nanobody_partial', and 'binder_partial'. The official
            implementation also supports 'monomer' design, but we have not yet
            supported it here.
            Different modes expect different `input_yaml` schemas.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in a Modal volume only.
        download_models: Whether to download model weights and skip running.
        force_redownload: Whether to force re-download of model weights even if they exist.
    """
    if download_models:
        fetch_model_weights.remote(force=force_redownload)
        return

    with open(input_yaml) as f:
        import yaml

        yaml_dict = yaml.safe_load(f)

    if "name" not in yaml_dict:
        raise ValueError(
            "Input YAML must contain a 'name' field for the design target."
        )
    run_name = sanitize_filename(yaml_dict["name"])
    remote_workdir = Path(CONF.output_volume_mountpoint) / run_name

    match design_mode:
        case "antibody_nanobody":
            conf = SampleAntibodyNanobodyConfig.model_validate(yaml_dict)
            files_to_upload = [conf.antigen_pdb, conf.framework_pdb]
            conf.antigen_pdb = remote_workdir / Path(conf.antigen_pdb).name
            conf.framework_pdb = remote_workdir / Path(conf.framework_pdb).name
        case "binder":
            conf = SampleBinderConfig.model_validate(yaml_dict)
            files_to_upload = [conf.input_pdb]
            conf.input_pdb = remote_workdir / Path(conf.input_pdb).name
        case "antibody_nanobody_partial":
            conf = SampleAntibodyNanobodyPartialConfig.model_validate(yaml_dict)
            files_to_upload = [conf.complex_pdb]
            conf.complex_pdb = remote_workdir / Path(conf.complex_pdb).name
        case "binder_partial":
            conf = SampleBinderPartialConfig.model_validate(yaml_dict)
            files_to_upload = [conf.input_pdb]
            conf.input_pdb = remote_workdir / Path(conf.input_pdb).name
        case _:
            raise ValueError(f"Unsupported design_mode: {design_mode}")

    # NOTE: make sure names are unique for different inputs

    with OUTPUTS_VOLUME.batch_upload() as batch:
        for file in files_to_upload:
            print(f"🧬 Uploading '{file}' to volume {OUTPUTS_VOLUME_NAME}...")
            batch.put_file(file, f"{run_name}/{Path(file).name}")

    print(f"🧬 Submitting PPIFlow task with run name: {run_name}")
    res = ppiflow_run.remote(PPIFlowArgs(args=conf), run_name)

    if out_dir is None:
        return

    print(f"🧬 Results saved to: {res}")
