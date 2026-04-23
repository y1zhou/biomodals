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
from biomodals.app.helper.shell import run_command_with_log, sanitize_filename

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="PPIFlow",
    repo_url="https://github.com/Mingchenchen/PPIFlow",
    repo_commit_hash="cc4f10dc52b5eadb8cfe2ab339aa8fb8fb5bc6d2",
    package_name="ppiflow",
    cuda_version="cu121",
    gpu=os.environ.get("GPU", "L40S"),
)

# Volumes to be mounted
OUTPUTS_VOLUME = CONF.get_out_volume()
OUTPUTS_VOLUME_NAME = OUTPUTS_VOLUME.name or f"{CONF.name}-outputs"

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
                f"micromamba env create -f {CONF.git_clone_dir / 'ppiflow_af3_merged.yaml'}",
            )
        )
    )
    .workdir(str(CONF.git_clone_dir / "tool" / "PPIFlow"))
    # .micromamba_install(spec_file="environment.yml")
    .env({"PATH": "/root/micromamba/envs/ppiflow_af3/bin:$PATH"}),
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
class CommonConfig(BaseModel):
    """Common input args for PPIFlow scripts."""

    name: str  # Test target name
    output_dir: str | Path  # **Remote** output directory to save results
    specified_hotspots: str  # Comma-separated, <chain><resi>, e.g. "A123,B45"
    model_weights: str | Path  # .ckpt file path for the relevant PPIFlow model
    samples_per_target: int = 100  # Number of samples to generate


class SampleAntibodyNanobodyConfig(CommonConfig):
    """Input args for sample_antibody_nanobody.py."""

    antigen_pdb: str | Path  # Input antigen protein PDB file path
    antigen_chain: str  # Chain ID of the antigen
    framework_pdb: str | Path
    heavy_chain: str
    light_chain: str | None = None  # Leave empty for nanobody design
    cdr_length: str = "CDRH1,5-12,CDRH2,4-17,CDRH3,5-26,CDRL1,5-12,CDRL2,3-10,CDRL3,4-13"  # CDR lengths to sample
    config: str | Path = "./configs/test_antibody.yaml"


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
    config: str | Path = "./configs/test_ppi_pairformer.yaml"


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
    config: str | Path = "./configs/test_antibody.yaml"

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
    config: str | Path = "./configs/test_ppi_complex_motifv1_partial.yaml"

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
def ppiflow_run(args: PPIFlowArgs) -> str:
    """Actual remote runner of PPIFlow."""
    import sys

    arg_fields = args.args.model_dump(exclude_none=True)

    # Build command
    cmd = [
        sys.executable,
        args.script_name,
        *(f"--{k}={v}" for k, v in arg_fields.items()),
    ]
    out_path = Path(args.args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / f"{CONF.name}-run.log"
    print(f"💊 Running {CONF.name}, saving logs to {log_path}")
    run_command_with_log(cmd, log_file=log_path, cwd=out_path)

    OUTPUTS_VOLUME.commit()
    return str(out_path)


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
    match design_mode:
        case "antibody_nanobody":
            conf = SampleAntibodyNanobodyConfig.model_validate(yaml_dict)
        case "binder":
            conf = SampleBinderConfig.model_validate(yaml_dict)
        case "antibody_nanobody_partial":
            conf = SampleAntibodyNanobodyPartialConfig.model_validate(yaml_dict)
        case "binder_partial":
            conf = SampleBinderPartialConfig.model_validate(yaml_dict)
        case _:
            raise ValueError(f"Unsupported design_mode: {design_mode}")

    # NOTE: make sure names are unique for different inputs
    run_name = sanitize_filename(conf.name)
    print(f"💊 Submitting PPIFlow task with run name: {run_name}")
    res = ppiflow_run.remote(PPIFlowArgs(args=conf))

    if out_dir is None:
        return

    print(f"🧬 Results saved to: {res}")
