"""Small synthetic AF3 publication used by API and real-browser tests."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import orjson
import zstandard

from biomodals.app.fold.alphafold3.inference_inputs import (
    hash_sequences,
    sanitize_af3_name,
)
from biomodals.app.fold.alphafold3.request_results import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    request_view_id,
)
from biomodals.app.fold.alphafold3.seed_predictions import canonical_output_name

CIF = (Path(__file__).parents[1] / "fixtures/alphafold3-preview.cif").read_bytes()
CHAINS = ["A", "A", "A", "B", "B", "C", "C"]
RESIDUES = [1, 2, 3, 1, 2, 1, 1]
PAE: list[list[float | None]] = [
    [round((i * 7 + j) / 10, 1) for j in range(7)] for i in range(7)
]
PAE[0][1] = None


def preview_archive(
    *,
    display_name: str = "Preview fixture",
    confidence: bytes | None = None,
    summary: bytes | None = None,
    manifest_last: bool = False,
    corrupt_best: bool = False,
    corrupt_role: str | None = None,
) -> bytes:
    """Build native-shaped aliases, immutable identities and two ranked seeds."""
    run_id = "d" * 64
    root = f"{run_id[:2]}/{run_id}"
    canonical = canonical_output_name(run_id)
    presentation = sanitize_af3_name(display_name)
    request_id = hash_sequences(run_id, [1, 2])
    view_id = request_view_id(request_id, (2, 1), display_name)
    contents = {
        "input": ("data.json", b"{}"),
        "request_best_model_cif": ("model.cif", CIF),
        "request_best_summary_confidences": (
            "summary_confidences.json",
            summary
            if summary is not None
            else orjson.dumps({"ptm": 0.81, "iptm": None, "has_clash": 0.0}),
        ),
        "request_best_confidences": (
            "confidences.json",
            confidence
            if confidence is not None
            else orjson.dumps({
                "pae": PAE,
                "token_chain_ids": CHAINS,
                "token_res_ids": RESIDUES,
                "contact_probs": [[0.5] * 7 for _ in range(7)],
                "atom_plddts": [91.0] * 7,
                "atom_chain_ids": CHAINS,
            }),
        ),
    }
    artifacts = []
    members = {}
    for role, (suffix, content) in contents.items():
        archived = f"{presentation}_{suffix}"
        volume_path = (
            f"{root}/inputs/{canonical}_data.json"
            if role == "input"
            else f"{root}/outputs/seed-2_sample-0/{canonical}_seed-2_sample-0_{suffix}"
        )
        digest = hashlib.sha256(content).hexdigest()
        artifacts.append({
            "role": role,
            "volume_path": volume_path,
            "archive_path": archived,
            "size_bytes": len(content),
            "sha256": digest,
            "archive_size_bytes": len(content),
            "archive_sha256": digest,
        })
        members[f"{presentation}/{archived}"] = (
            b"x" * len(content) if role == corrupt_role else content
        )
    ranking = [
        {"seed": 2, "sample_index": 0, "ranking_score": 0.91234},
        {"seed": 1, "sample_index": 0, "ranking_score": 0.91231},
    ]
    manifest = {
        "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "request_id": request_id,
        "view_id": view_id,
        "canonical_name": canonical,
        "sample_count": 1,
        "submitted_display_name": display_name,
        "presentation_name": presentation,
        "name_mapping": {"canonical": canonical, "presentation": presentation},
        "submitted_seeds": [2, 1],
        "normalized_seeds": [1, 2],
        "duplicates_removed": [],
        "ranking": ranking,
        "best": ranking[1] if corrupt_best else ranking[0],
        "artifacts": artifacts,
        "manifest_volume_path": f"{root}/requests/{request_id}/views/{view_id}/manifest.json",
    }
    marker = (f"{presentation}/request_manifest.json", orjson.dumps(manifest))
    ordered = (
        [*members.items(), marker] if manifest_last else [marker, *members.items()]
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in ordered:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return zstandard.ZstdCompressor().compress(buffer.getvalue())
