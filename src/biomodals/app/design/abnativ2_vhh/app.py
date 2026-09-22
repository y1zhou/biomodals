"""Enhanced single-domain humanization with AbNatiV2 VH2/VHH2 and NbForge.

Upstream: https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ

Stage verified models with ``stage_abnativ2_vhh_models`` before inference.
The nanobody_humanization workflow owns production admission and recovery.
The one-call standalone entrypoint is development-only and consumes an already
prepared domain and its protected zero-based sequence indices. Native structures
are operational evidence, not affinity or immunogenicity validation.
"""

from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.design.abnativ2_vhh.models import (
    MODEL_ROOT,
    NBFORGE_COMMIT,
    SOURCE_COMMIT,
    stage_assets,
)
from biomodals.app.design.abnativ2_vhh.patches import (
    apply_colormap_patch,
    verify_runtime,
)
from biomodals.app.design.abnativ2_vhh.worker import (
    EnhancedSettings,
    humanize_vhh,
    score_vhh,
)
from biomodals.app.design.vhh import VHHInput
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.schema import InlineBytes

CONF = AppConfig(
    name="AbNatiV2-VHH",
    package_name="abnativ2_vhh",
    repo_url="https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ",
    repo_commit_hash=SOURCE_COMMIT,
    version="2.0.9",
    python_version="3.12",
    cuda_version="cu124",
    gpu="A10G",
    timeout=24 * 60 * 60,
    tags={"group": "design"},
)
runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install("build-essential", "git")
    .env(
        CONF.default_env
        | {
            "ABNATIV_MODELS_DIR": str(MODEL_ROOT),
            "NBFORGE_CKPT": str(MODEL_ROOT / "nbforge.ckpt"),
            "MPLBACKEND": "Agg",
        }
    )
    .micromamba_install(
        [
            "anarci==2020.04.23",
            "hmmer==3.4",
            "pdbfixer==1.12.0",
            "openmm==8.3.1",
            "freesasa==2.2.1",
            "biopython==1.86",
            "numpy==2.2.6",
            "pandas==2.2.3",
            "scipy==1.15.3",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .uv_pip_install(
        "torch==2.6.0+cu124", index_url="https://download.pytorch.org/whl/cu124"
    )
    .uv_pip_install(
        "lightning==2.5.6",
        "pytorch-lightning==2.5.6",
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "brewer2mpl==1.4.1",
        "dm-tree==0.1.9",
        "einops==0.8.1",
        "loguru==0.7.3",
        "ml-collections==1.1.0",
        "paretoset==1.2.5",
        "protein-topmodel==1.0.1",
        "scikit-learn==1.7.2",
        "typer==0.20.0",
    )
    # Native packaging includes the verified NbForge weights and V2 reference data.
    # Install only the audited inference closure, not training-only dependencies.
    .uv_pip_install(
        f"git+https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ.git@{SOURCE_COMMIT}",
        f"git+https://gitlab.doc.ic.ac.uk/sormanni-lab/nbforge.git@{NBFORGE_COMMIT}",
        extra_options="--no-deps",
    )
    .add_local_python_source("biomodals.app.design.abnativ2_vhh.patches", copy=True)
    .run_function(apply_colormap_patch)
    # Copy shared imports because the following build check must run after all
    # dependency resolution; deferred app sources are added only afterward.
    .pipe(patch_image_for_helper, copy_patch_files=True)
    .run_function(verify_runtime)
    .add_local_python_source(
        "biomodals.app.design.abnativ2_vhh", "biomodals.app.design.vhh"
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
abnativ2_vhh_humanize = app.function(
    name="abnativ2_vhh_humanize",
    gpu=CONF.gpu,
    cpu=4,
    memory=16384,
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False),
)(humanize_vhh)
abnativ2_vhh_score = app.function(
    name="abnativ2_vhh_score",
    gpu=CONF.gpu,
    cpu=2,
    memory=16384,
    timeout=3600,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False),
)(score_vhh)


@app.function(
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(model_volume=True, model_ro=False, model_mount_subdir=False),
)
def stage_abnativ2_vhh_models() -> dict[str, object]:
    """Prepare all assets on CPU and commit before inference can be admitted."""
    from importlib.resources import files  # noqa: PLC0415

    MODEL_VOLUME.reload()
    bundled = Path(str(files("NbForge") / "weights/nbforge.ckpt"))
    manifest = stage_assets(MODEL_ROOT, bundled)
    MODEL_VOLUME.commit()
    return manifest


@app.local_entrypoint()
def submit_abnativ2_vhh_task(
    input_json: str,
    output_dir: str,
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_vhh_score_decrease: float = 0.05,
) -> None:
    """Run one already prepared domain in explicit development mode.

    Args:
        input_json: JSON containing sequence and protected_indices; maximum 8 KiB.
        output_dir: New local directory for the generation report and native evidence.
        residue_score_threshold: Native VH2 residue-liability threshold.
        rasa_threshold: Minimum solvent exposure for allowed substitutions.
        max_relative_vhh_score_decrease: Native per-step VHH2 loss tolerance.
    """
    source, destination = Path(input_json), Path(output_dir)
    if not source.is_file() or source.stat().st_size > 8192:
        raise ValueError("Prepared input must be a regular JSON file at most 8 KiB")
    parent = VHHInput.model_validate_json(source.read_bytes())
    settings = EnhancedSettings(
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_vhh_score_decrease=max_relative_vhh_score_decrease,
    )
    destination.mkdir(parents=True, exist_ok=False)
    result = abnativ2_vhh_humanize.remote(
        parent=parent.model_dump(), settings=settings.model_dump()
    )
    for output in result.outputs:
        if not isinstance(output.storage, InlineBytes):
            raise TypeError("Expected bounded inline native evidence")
        (destination / Path(output.storage.filename).name).write_bytes(
            output.storage.data
        )
    print(f"🧬 AbNatiV2-VHH results saved to: {destination.resolve()}")
