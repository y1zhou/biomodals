"""Bounded decoding of app-owned results into the shared candidate contract."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import orjson
import polars as pl

from biomodals.helper.archives import MAX_ARCHIVE_BYTES
from biomodals.helper.archives import flat_archive_members as archive_members
from biomodals.schema import AppOutput, AppRunResult, ArtifactKind, InlineBytes
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateOrigin,
    HumanizationMethod,
)


def json_output(name: str, value: Any) -> AppOutput:
    """Keep structured adapters as small UTF-8 execution artifacts."""
    return AppOutput(
        name=name,
        kind=ArtifactKind.REPORT,
        storage=InlineBytes(
            data=orjson.dumps(value),
            filename=f"{name}.json",
            media_type="application/json",
        ),
    )


def generated_pairs(
    method: HumanizationMethod,
    parent: AntibodyPair,
    result: AppRunResult,
    *,
    sapiens_iterations: int = 1,
) -> list[tuple[str, str, str, CandidateOrigin]]:
    """Decode one successful app call without mixing its chains or parental IDs."""
    if len(result.outputs) != 1 or not isinstance(
        result.outputs[0].storage, InlineBytes
    ):
        raise ValueError("Generator must return one bounded inline app publication")
    content = result.outputs[0].storage.data
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Generator result exceeds byte limit")
    if method in {"sapiens", "humatch"}:
        members = archive_members(content)
        filename = "iteration_designs.csv" if method == "sapiens" else "humanized.csv"
        frame = pl.read_csv(BytesIO(members[filename]), infer_schema=False)
        count = sapiens_iterations if method == "sapiens" else 1
        if frame.height != count or not frame["id"].eq(parent.id).all():
            raise ValueError("Generator output does not match its parent")
        if method == "sapiens" and frame["iteration"].to_list() != [
            str(i) for i in range(1, count + 1)
        ]:
            raise ValueError("Sapiens output does not match requested iterations")
        rows = frame.to_dicts()
        seed = None
    else:
        publication = orjson.loads(content)
        if publication.get("schema_version") != 1:
            raise ValueError("Unsupported generator publication")
        value = publication["pair_result"]
        rows = value["candidates"] if method == "hudiff_ab" else [value["humanized"]]
        seed = value.get("pair_seed")
    generated = []
    for row in rows:
        if row["id"] != parent.id:
            raise ValueError("Generator returned a foreign parent ID")
        pair = AntibodyPair(id=parent.id, vh=row["vh"], vl=row["vl"])
        origin = CandidateOrigin(
            method=method,
            source_id=f"{parent.id}__iteration_{row['iteration']}"
            if method == "sapiens"
            else row.get("candidate_id", parent.id),
            attempt_index=row.get("attempt_index"),
            seed=seed,
        )
        generated.append((parent.id, pair.vh, pair.vl, origin))
    return generated
