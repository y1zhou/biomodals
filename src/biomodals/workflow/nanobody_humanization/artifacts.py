"""Native report validation and execution-owned table publications."""

from collections.abc import Mapping
from io import BytesIO
from typing import Any

import orjson
import polars as pl

from biomodals.app.design.abnativ2_vhh.contracts import SearchSummary
from biomodals.app.design.abnativ2_vhh.models import RUNTIME_IDENTITY, SOURCE_COMMIT
from biomodals.app.design.vhh import VHHInput
from biomodals.execution.nodes import NodeRunContext
from biomodals.helper.archives import MAX_ARCHIVE_BYTES, flat_archive_members
from biomodals.helper.artifacts import read_bounded_file_bytes
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.nanobody_humanization.preparation import PreparedVH
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings
from biomodals.workflow.nanobody_humanization.tables import GENERATION_SCHEMA


def json_output(name: str, value: Any) -> AppOutput:
    """Publish small primitive control evidence through the existing kernel."""
    return AppOutput(
        name=name,
        kind=ArtifactKind.REPORT,
        storage=InlineBytes(
            filename=f"{name}.json",
            media_type="application/json",
            data=orjson.dumps(value),
        ),
    )


def table_output(context: NodeRunContext, name: str, frame: pl.DataFrame) -> AppOutput:
    """Write Parquet directly to this Node's isolated execution-volume directory."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Table publication requires the execution volume")
    path = context.work_dir / f"{name}.parquet"
    relative = path.resolve().relative_to(context.volume_root.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    return AppOutput(
        name=name,
        kind=ArtifactKind.TABLE,
        storage=VolumePath(
            volume_name=context.artifact_volume_name,
            path=str(relative),
            media_type="application/vnd.apache.parquet",
        ),
    )


def output_bytes(context: NodeRunContext, output: AppOutput) -> bytes:
    """Read a recorded worker result only from the owning execution Volume."""
    if isinstance(output.storage, InlineBytes):
        return output.storage.data
    if (
        context.volume_root is None
        or output.storage.volume_name != context.artifact_volume_name
    ):
        raise ValueError("Worker result is outside the execution volume")
    path = (context.volume_root / output.storage.path).resolve()
    path.relative_to(context.volume_root.resolve())
    return read_bounded_file_bytes(
        path, field_name="Worker result", max_bytes=MAX_ARCHIVE_BYTES
    )


def generation_frame(
    parent: PreparedVH,
    method: str,
    settings: NanobodySettings,
    report: Mapping[str, Any],
) -> pl.DataFrame:
    """Bind native attempts to their exact parent, method, budget and fixed policy."""
    if (
        report.get("schema_version") != (2 if method == "abnativ2_vhh" else 1)
        or report.get("method") != method
        or report.get("input_sequence") != parent.sequence
    ):
        raise ValueError("Generation report changed its request identity")
    expected = settings.hudiff_nb_candidate_count if method == "hudiff_nb" else 1
    attempts = report["attempts"]
    if method == "abnativ2_vhh":
        parameters = settings.abnativ_parameters(
            parent.id, parent.sequence, parent.protected_indices
        )
        if (
            report.get("settings") != parameters.model_dump()
            or report.get("source_commit") != SOURCE_COMMIT
            or report.get("runtime_identity") != RUNTIME_IDENTITY
        ):
            raise ValueError(
                "AbNatiV generation changed its settings or scientific identity"
            )
        if parameters.explore:
            search = SearchSummary.model_validate(report.get("search"))
            if search.evaluated_candidates != min(
                parameters.candidate_budget, int(search.possible_candidates)
            ):
                raise ValueError("Exploration report changed its evaluation budget")
            expected = max(1, search.accepted_candidates)
            if search.accepted_candidates == 0 and attempts != [
                {"attempt_index": 1, "sequence": parent.sequence, "error": None}
            ]:
                raise ValueError(
                    "Empty exploration must retain the successful parental no-op"
                )
        elif report.get("search") is not None:
            raise ValueError("Enhanced report contains exploration evidence")
    if [row["attempt_index"] for row in attempts] != list(range(1, expected + 1)):
        raise ValueError("Generation report changed its attempt budget")
    if method == "hudiff_nb" and report.get("seed") != settings.root_seed:
        raise ValueError("Generation report changed its requested seed")
    policy = VHHInput(
        sequence=parent.sequence, protected_indices=parent.protected_indices
    )
    for row in attempts:
        if row["error"] is None:
            policy.validate_candidate(row["sequence"])
    return pl.DataFrame(
        [
            {
                "parent_id": parent.id,
                "method": method,
                "attempt_index": row["attempt_index"],
                "seed": settings.root_seed if method == "hudiff_nb" else None,
                "vh": row["sequence"],
                "error": row["error"],
            }
            for row in attempts
        ],
        schema=GENERATION_SCHEMA,
    )


def score_tables(
    context: NodeRunContext, result: AppRunResult
) -> dict[str, pl.DataFrame]:
    """Decode a worker's bounded table archive without intermediate disk copies."""
    if len(result.outputs) != 1:
        raise ValueError("Expected one scientific table archive")
    members = flat_archive_members(output_bytes(context, result.outputs[0]))
    return {
        name.removesuffix(".parquet"): pl.read_parquet(BytesIO(value))
        for name, value in members.items()
    }
