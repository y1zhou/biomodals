"""Fixed-budget HuDiff-Nb sampling on a prepared single VH domain.

Upstream: https://github.com/TencentAI4S/HuDiff

Reuse the pinned HuDiff runtime and stage_hudiff_models checkpoint publication.
The nanobody_humanization workflow owns production admission and recovery.
This one-call development entrypoint accepts the complete prepared sequence and
protected zero-based sequence indices. Attempts can duplicate the parent or each
other; invalid attempts are recorded, never silently replaced.
"""

from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.design.hudiff_ab.app import (
    runtime_image as paired_runtime_image,
)
from biomodals.app.design.hudiff_ab.models import SOURCE_COMMIT
from biomodals.app.design.hudiff_nb.worker import humanize_vhh
from biomodals.app.design.hudiff_nb.worker import (
    stage_hudiff_nb_models as _stage_models,
)
from biomodals.app.design.vhh import VHHInput
from biomodals.helper import patch_image_for_helper
from biomodals.schema import InlineBytes

CONF = AppConfig(
    name="HuDiff-Nb",
    package_name="hudiff_nb",
    version="1.0.0",
    repo_url="https://github.com/TencentAI4S/HuDiff",
    repo_commit_hash=SOURCE_COMMIT,
    python_version="3.10",
    cuda_version="cu116",
    gpu="A10G",
    timeout=24 * 60 * 60,
    tags={"group": "design"},
)
runtime_image = paired_runtime_image.add_local_python_source(
    "biomodals.app.design.hudiff_nb", "biomodals.app.design.vhh"
)
staging_image = (
    modal.Image
    .debian_slim(python_version="3.12")
    .pipe(patch_image_for_helper)
    .add_local_python_source(
        "biomodals.app.design.hudiff_ab.models",
        "biomodals.app.design.hudiff_nb.worker",
        "biomodals.app.design.hudiff_nb.patches",
        "biomodals.app.design.vhh",
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
hudiff_nb_humanize = app.function(
    name="hudiff_nb_humanize",
    gpu=CONF.gpu,
    cpu=4,
    memory=16384,
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False),
)(humanize_vhh)


stage_hudiff_nb_models = app.function(
    image=staging_image,
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(model_volume=True, model_ro=False, model_mount_subdir=False),
)(_stage_models)


@app.local_entrypoint()
def submit_hudiff_nb_task(
    input_json: str, output_file: str, candidate_count: int = 10, seed: int = 0
) -> None:
    """Run exact-budget attempts in explicit development mode.

    Args:
        input_json: JSON containing sequence and protected_indices; maximum 8 KiB.
        output_file: New local generation JSON file.
        candidate_count: Total native attempts, from 1 to 25, without refill.
        seed: Explicit unsigned 32-bit root seed.
    """
    source, destination = Path(input_json), Path(output_file)
    if not source.is_file() or source.stat().st_size > 8192:
        raise ValueError("Prepared input must be a regular JSON file at most 8 KiB")
    parent = VHHInput.model_validate_json(source.read_bytes())
    if not 1 <= candidate_count <= 25 or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Expected 1–25 attempts and an unsigned 32-bit seed")
    if destination.exists():
        raise FileExistsError(destination)
    result = hudiff_nb_humanize.remote(
        parent=parent.model_dump(), candidate_count=candidate_count, seed=seed
    )
    output = result.outputs[0]
    if not isinstance(output.storage, InlineBytes):
        raise TypeError("Expected inline generation JSON")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(output.storage.data)
    print(f"🧬 HuDiff-Nb results saved to: {destination.resolve()}")
