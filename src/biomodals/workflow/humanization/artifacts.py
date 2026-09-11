"""Bounded decoding of app-owned results into the shared candidate contract."""

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

import orjson
import polars as pl
import zstandard

from biomodals.schema import AppOutput, AppRunResult, ArtifactKind, InlineBytes
from biomodals.workflow.humanization.contracts import (
    AntibodyPair,
    CandidateOrigin,
    HumanizationMethod,
)

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def archive_members(content: bytes) -> dict[str, bytes]:
    """Read a small app archive without filesystem extraction or unbounded expansion."""
    if not content or len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError("Invalid compressed archive size")
    with zstandard.ZstdDecompressor().stream_reader(BytesIO(content)) as reader:
        expanded = reader.read(MAX_ARCHIVE_BYTES + 1)
    if len(expanded) > MAX_ARCHIVE_BYTES:
        raise ValueError("Expanded archive exceeds byte limit")
    files = {}
    with tarfile.open(fileobj=BytesIO(expanded), mode="r:") as archive:
        for index, member in enumerate(archive):
            path = PurePosixPath(member.name)
            if index > 1000 or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or len(path.parts) > 2 or path.name in files:
                raise ValueError("Unexpected archive member or duplicate basename")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive member")
            files[path.name] = stream.read()
    return files


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
