"""One prepared VH per provider call, with a bounded immutable attempt report."""

from __future__ import annotations

import subprocess
import sys
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import orjson

from biomodals.app.design.hudiff_ab.models import (
    CHECKPOINTS,
    RUNTIME_IDENTITY,
    SOURCE_COMMIT,
    assert_hudiff_assets,
)
from biomodals.app.design.hudiff_nb.patches import (
    apply_nanobody_patches,
    patch_identity,
)
from biomodals.app.design.vhh import VHHInput
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)

MODEL_ROOT = Path("/biomodals-store/hudiff")
SOURCE_ROOT = Path("/opt/HuDiff")
CHECKPOINT = next(
    item for item in CHECKPOINTS if item.path == "checkpoints/nanobody/hudiffnb.pt"
)
MAX_RESULT_BYTES = 1024 * 1024


@cache
def prepare_runtime() -> None:
    """Verify reused model assets and install guarded inference-only imports once."""
    assert_hudiff_assets(MODEL_ROOT)
    apply_nanobody_patches(SOURCE_ROOT)


def humanize_vhh(
    *, parent: dict[str, Any], candidate_count: int = 10, seed: int = 0
) -> AppRunResult:
    """Generate exact-budget attempts; duplicates/no-ops remain successful evidence."""
    prepared = VHHInput.model_validate(parent)
    if type(candidate_count) is not int or not 1 <= candidate_count <= 25:
        raise ValueError("candidate_count must be between 1 and 25")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")
    prepare_runtime()
    request = {
        "parent": prepared.model_dump(),
        "candidate_count": candidate_count,
        "seed": seed,
    }
    with TemporaryDirectory(prefix="hudiff-nb-") as directory:
        root = Path(directory)
        source, destination = root / "request.json", root / "result.json"
        source.write_bytes(orjson.dumps(request))
        subprocess.run(  # noqa: S603 - fixed module and controlled temporary paths.
            [
                sys.executable,
                "-m",
                "biomodals.app.design.hudiff_nb.runtime",
                "--input",
                str(source),
                "--output",
                str(destination),
                "--checkpoint",
                str(MODEL_ROOT / CHECKPOINT.path),
            ],
            cwd=SOURCE_ROOT,
            check=True,
            timeout=23 * 60 * 60,
        )
        if not destination.is_file() or destination.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError("HuDiff-Nb report is missing or oversized")
        report = orjson.loads(destination.read_bytes())
    if report["input_sequence"] != prepared.sequence or report["seed"] != seed:
        raise ValueError("HuDiff-Nb report does not match its prepared input")
    if [row["attempt_index"] for row in report["attempts"]] != list(
        range(1, candidate_count + 1)
    ):
        raise ValueError("HuDiff-Nb returned the wrong attempt identities")
    for row in report["attempts"]:
        if row["error"] is None:
            prepared.validate_candidate(row["sequence"])
    report.update({
        "schema_version": 1,
        "method": "hudiff_nb",
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT.sha256,
        "runtime_identity": RUNTIME_IDENTITY,
        "patch_identity": patch_identity(),
    })
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="generation",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    filename="generation.json",
                    media_type="application/json",
                    data=orjson.dumps(report),
                ),
            )
        ],
        metrics={
            "attempts": candidate_count,
            "valid_attempts": sum(row["error"] is None for row in report["attempts"]),
        },
    )
