"""Remove unused training imports; retain the released nanobody sampler helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def patch_identity() -> str:
    """Bind guarded compatibility edits and intentional sampling policy changes."""
    from biomodals.app.design.hudiff_nb import runtime

    return hashlib.sha256(
        Path(__file__).read_bytes() + Path(runtime.__file__).read_bytes()
    ).hexdigest()


def apply_nanobody_patches(root: Path) -> None:
    """Avoid importing training datasets/plotting into the inference-only image."""
    edits = {
        "nanobody_scripts/nanosample.py": (
            (
                "from dataset.oas_pair_dataset_new import light_pad_cdr, HEAVY_REGION_INDEX, LIGHT_REGION_INDEX",
                "from biomodals.app.design.hudiff_ab.upstream_runtime import HEAVY_REGION_INDEX",
            ),
            (
                "from utils.train_utils import model_selected",
                "# Biomodals: unused CLI training selector omitted on import.",
            ),
        ),
        "model/nanoencoder/model.py": (
            (
                "from .abnativ_scoring import get_abnativ_nativeness_scores",
                "# Biomodals: training-only score imports omitted for inference.",
            ),
        ),
    }
    for name, replacements in edits.items():
        path = root / name
        source = path.read_text()
        for before, after in replacements:
            if after in source and before not in source:
                continue
            if source.count(before) != 1 or after in source:
                raise ValueError(f"Pinned HuDiff-Nb patch preimage changed: {name}")
            source = source.replace(before, after)
        path.write_text(source)
