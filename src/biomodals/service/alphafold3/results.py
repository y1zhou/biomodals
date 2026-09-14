"""Private, bounded previews of one immutable AlphaFold3 request archive."""

from __future__ import annotations

import hashlib
import math
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import IO, Literal, cast

import numpy as np
import orjson
import zstandard
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.fold.alphafold3.inference_inputs import sanitize_af3_name
from biomodals.app.fold.alphafold3.request_results import (
    request_publication_from_manifest,
)
from biomodals.service.artifacts import ArtifactLease

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CIF_BYTES = 16 * 1024 * 1024
MAX_CONFIDENCE_BYTES = 32 * 1024 * 1024
MAX_SUMMARY_BYTES = 1024 * 1024
MAX_PAE_TOKENS = 2048
MAX_PAE_GRID = 512
MAX_ARCHIVE_SCAN_BYTES = 4 * 1024 * 1024 * 1024
_SUFFIXES = {
    "request_best_model_cif": ("model.cif", MAX_CIF_BYTES),
    "request_best_summary_confidences": ("summary_confidences.json", MAX_SUMMARY_BYTES),
    "request_best_confidences": ("confidences.json", MAX_CONFIDENCE_BYTES),
}


class PredictionSummary(BaseModel):
    """Identity and small scientific metadata for the Job's best prediction."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    seed: int
    sample_index: int
    ranking_score: float
    prediction_count: int
    ptm: float | None
    iptm: float | None
    has_clash: bool | None
    summary_error: Literal["summary_invalid"] | None
    pae_error: Literal["pae_too_large", "pae_invalid"] | None
    token_chain_ids: list[str]
    token_res_ids: list[int]
    max_pae_grid_size: int = MAX_PAE_GRID


class PaeWindow(BaseModel):
    """Row-major PAE: Y is aligned/frame token, X is scored token."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    x_edges: list[int]
    y_edges: list[int]
    values: list[list[float | None]]
    valid_counts: list[list[int]]
    aggregation: Literal["exact", "mean"]


class _NativeSummary(BaseModel):
    model_config = ConfigDict(strict=True)

    ptm: float | None = Field(ge=0, le=1)
    iptm: float | None = Field(ge=0, le=1)
    # AF3 natively writes this boolean as the number 0.0 or 1.0.
    has_clash: Literal[0, 1] | None


class PreviewTooLargeError(ValueError):
    """The artifact exceeds the service preview's resource budget."""


@dataclass(frozen=True)
class PredictionData:
    """Bounded native CIF and PAE, never the unrelated contact matrix."""

    summary: PredictionSummary
    cif: bytes
    pae: NDArray[np.float32] | None


class PredictionReader:
    """Two-entry memory cache, used only under the shared artifact worker.

    Every read still requires a fresh archive lease. Evicting the authoritative
    cache therefore immediately prevents serving a retained preview. No extra
    disk cache, worker pool, or provider download path is introduced.
    """

    def __init__(self) -> None:
        """Bound retained previews independently of the number of Service Jobs."""
        self._entries: OrderedDict[tuple[str, str], PredictionData] = OrderedDict()

    def read(
        self, lease: ArtifactLease, *, job_id: str, digest: str, display_name: str
    ) -> PredictionData:
        """Reuse immutable data only after ownership and archive verification."""
        key = (job_id, digest)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        # Evict before preparation so peak memory includes at most one old entry.
        while len(self._entries) >= 2:
            self._entries.popitem(last=False)
        try:
            data = read_prediction(lease, digest=digest, display_name=display_name)
        except (tarfile.TarError, zstandard.ZstdError, EOFError) as error:
            raise ValueError("Prediction archive is invalid") from error
        self._entries[key] = data
        return data


def _read_members(
    lease: ArtifactLease, presentation: str
) -> tuple[dict[str, bytes], set[str]]:
    """Capture only fixed preview aliases in one bounded sequential pass."""
    wanted = {f"{presentation}/request_manifest.json": MAX_MANIFEST_BYTES}
    wanted.update({
        f"{presentation}/{presentation}_{suffix}": limit
        for suffix, limit in _SUFFIXES.values()
    })
    found: dict[str, bytes] = {}
    oversized: set[str] = set()
    lease.seek(0)
    with zstandard.ZstdDecompressor().stream_reader(
        cast(IO[bytes], lease), closefd=False
    ) as source:
        with tarfile.open(fileobj=source, mode="r|") as archive:
            for member in archive:
                if member.offset_data + member.size > MAX_ARCHIVE_SCAN_BYTES:
                    raise PreviewTooLargeError("Archive exceeds preview scan limit")
                if member.name not in wanted:
                    continue
                if member.name in found or member.name in oversized:
                    raise ValueError("Duplicate prediction member")
                if not member.isfile():
                    raise ValueError("Prediction member is not a regular file")
                limit = wanted[member.name]
                if member.size > limit:
                    oversized.add(member.name)
                else:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("Prediction member is unreadable")
                    with handle:
                        content = handle.read(limit + 1)
                    if len(content) != member.size:
                        raise ValueError("Prediction member is truncated")
                    found[member.name] = content
                if len(found) + len(oversized) == len(wanted):
                    break
    return found, oversized


def _read_pae(content: bytes) -> tuple[NDArray[np.float32], list[str], list[int]]:
    """Parse once, discard contact data, validate and compact the token matrix."""
    document = orjson.loads(content)
    if not isinstance(document, dict):
        raise ValueError("Confidence document must be an object")
    rows = document.get("pae")
    chains = document.get("token_chain_ids")
    residues = document.get("token_res_ids")
    del document
    if not isinstance(chains, list) or not isinstance(residues, list):
        raise ValueError("Confidence token metadata is invalid")
    count = len(chains)
    if count > MAX_PAE_TOKENS:
        raise PreviewTooLargeError("PAE token count exceeds preview limit")
    if (
        not count
        or len(residues) != count
        or any(
            not isinstance(chain, str) or not 1 <= len(chain) <= 200 for chain in chains
        )
        or any(type(residue) is not int or residue < 1 for residue in residues)
        or not isinstance(rows, list)
        or len(rows) != count
        or any(not isinstance(row, list) or len(row) != count for row in rows)
    ):
        raise ValueError("Confidence matrix does not match token metadata")
    if any(
        value is not None and type(value) not in (float, int)
        for row in rows
        for value in row
    ):
        raise ValueError("PAE cells must be numbers or null")
    matrix = np.asarray(rows, dtype=np.float32)
    if np.any((matrix < 0) | (matrix > 32)):
        raise ValueError("PAE is outside the native model's range")
    matrix.flags.writeable = False
    return matrix, chains, residues


def read_prediction(
    lease: ArtifactLease, *, digest: str, display_name: str
) -> PredictionData:
    """Resolve and verify the archive's exact request-scoped best artifacts."""
    presentation = sanitize_af3_name(display_name)
    found, oversized = _read_members(lease, presentation)
    manifest_name = f"{presentation}/request_manifest.json"
    if manifest_name in oversized:
        raise PreviewTooLargeError("Request manifest exceeds preview limit")
    manifest = orjson.loads(found[manifest_name])
    if not isinstance(manifest, dict):
        raise ValueError("Request manifest must be an object")
    publication = request_publication_from_manifest(manifest)
    if publication.display_name != display_name:
        raise ValueError("Archive display_name does not match the request view")
    best = cast(dict[str, int | float], manifest["best"])

    def member(role: str) -> bytes:
        suffix, _ = _SUFFIXES[role]
        matches = [item for item in manifest["artifacts"] if item["role"] == role]
        if len(matches) != 1:
            raise ValueError(f"Request manifest requires exactly one {role} artifact")
        record = matches[0]
        # The embedded manifest already contains presentation-local filenames.
        alias = f"{presentation}_{suffix}"
        if record["archive_path"] != alias:
            raise ValueError("Best artifact does not match its preview alias")
        name = f"{presentation}/{alias}"
        expected_source = (
            f"{publication.run_id[:2]}/{publication.run_id}/outputs/"
            f"seed-{best['seed']}_sample-{best['sample_index']}/"
            f"{manifest['canonical_name']}_seed-{best['seed']}_sample-{best['sample_index']}_{suffix}"
        )
        if record["volume_path"] != expected_source:
            raise ValueError("Best artifact does not match the selected seed/sample")
        if name in oversized:
            raise PreviewTooLargeError("Prediction member exceeds preview limit")
        content = found.pop(name)
        if (
            len(content) != record["size_bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError("Prediction artifact does not match its manifest")
        return content

    cif = member("request_best_model_cif")
    summary_error = None
    try:
        scores = _NativeSummary.model_validate_json(
            member("request_best_summary_confidences")
        )
    except (KeyError, ValueError):
        scores = _NativeSummary(ptm=None, iptm=None, has_clash=None)
        summary_error = "summary_invalid"
    matrix = None
    chains: list[str] = []
    residues: list[int] = []
    pae_error = None
    try:
        matrix, chains, residues = _read_pae(member("request_best_confidences"))
    except PreviewTooLargeError:
        pae_error = "pae_too_large"
    except (KeyError, ValueError, TypeError):
        pae_error = "pae_invalid"
    return PredictionData(
        summary=PredictionSummary(
            prediction_id=digest,
            seed=best["seed"],
            sample_index=best["sample_index"],
            ranking_score=best["ranking_score"],
            prediction_count=len(manifest["ranking"]),
            ptm=scores.ptm,
            iptm=scores.iptm,
            has_clash=None if scores.has_clash is None else bool(scores.has_clash),
            summary_error=summary_error,
            pae_error=pae_error,
            token_chain_ids=chains,
            token_res_ids=residues,
        ),
        cif=cif,
        pae=matrix,
    )


def pae_window(
    data: PredictionData,
    *,
    x_start: int = 0,
    x_end: int | None = None,
    y_start: int = 0,
    y_end: int | None = None,
    max_size: int = MAX_PAE_GRID,
) -> PaeWindow:
    """Reduce a rectangle without transposing, symmetrizing or filling nulls."""
    if data.pae is None:
        raise ValueError("PAE preview is unavailable")
    count = len(data.pae)
    x_end = count if x_end is None else x_end
    y_end = count if y_end is None else y_end
    if not (
        0 <= x_start < x_end <= count
        and 0 <= y_start < y_end <= count
        and 1 <= max_size <= MAX_PAE_GRID
    ):
        raise ValueError("PAE window is outside the token matrix")
    x_step = math.ceil((x_end - x_start) / max_size)
    y_step = math.ceil((y_end - y_start) / max_size)
    x_edges = [*range(x_start, x_end, x_step), x_end]
    y_edges = [*range(y_start, y_end, y_step), y_end]
    window = data.pae[y_start:y_end, x_start:x_end]
    valid = np.isfinite(window)
    xs = np.asarray(x_edges[:-1]) - x_start
    ys = np.asarray(y_edges[:-1]) - y_start
    totals = np.add.reduceat(
        np.add.reduceat(np.where(valid, window, 0), ys, axis=0, dtype=np.float64),
        xs,
        axis=1,
    )
    counts = np.add.reduceat(
        np.add.reduceat(valid, ys, axis=0, dtype=np.int64), xs, axis=1
    )
    means = np.zeros_like(totals)
    np.divide(totals, counts, out=means, where=counts != 0)
    values = np.round(means, decimals=3).astype(object)
    values[counts == 0] = None
    return PaeWindow(
        prediction_id=data.summary.prediction_id,
        x_edges=x_edges,
        y_edges=y_edges,
        values=values.tolist(),
        valid_counts=counts.tolist(),
        aggregation="exact" if x_step == y_step == 1 else "mean",
    )
