"""Exact AlphaFold 3 invocation identity and durable manifest receipts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import cast

import orjson
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeReader,
    json_bytes,
    read_volume_bytes,
    sha256_bytes,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    ALPHAFOLD3_APP_VERSION,
    DECLARED_MODEL_IDENTITY,
    RUN_IDENTITY_SCHEMA,
    STAGED_INPUT_SCHEMA_VERSION,
    PreparedInferenceRun,
    VolumeUpload,
    build_inference_identity_view,
    hash_sequences,
    validate_inference_parameters,
    validate_inference_workload,
    validate_submitted_af3_input,
)
from biomodals.app.fold.alphafold3.msa_search import (
    COMBINED_RESULT_SCHEMA_VERSION,
    RAW_RESULT_SCHEMA_VERSION,
    SEARCH_ADAPTER_VERSION,
    SEARCH_IDENTITY_SCHEMA_VERSION,
    scientific_search_parameters,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    DATABASE_PROFILE_SPECS,
    HMMER_VERSION,
    JACKHMMER_PATCH_SHA256,
)
from biomodals.app.fold.alphafold3.request_results import (
    REQUEST_MANIFEST_SCHEMA_VERSION,
    REQUEST_VIEW_IDENTITY_SCHEMA,
    RequestPublication,
    request_manifest_path,
    request_publication_from_manifest,
    request_view_id,
)
from biomodals.app.fold.alphafold3.template_search import (
    DEFAULT_MAX_TEMPLATE_DATE,
    TEMPLATE_ADAPTER_VERSION,
    TEMPLATE_IDENTITY_SCHEMA_VERSION,
    TEMPLATE_RESULT_SCHEMA_VERSION,
    template_search_parameters,
)

INVOCATION_IDENTITY_SCHEMA = "biomodals-alphafold3-invocation-v1"
INVOCATION_RECEIPT_SCHEMA_VERSION = 1
MAX_INVOCATION_RECEIPT_BYTES = 64 * 1024
MAX_REQUEST_MANIFEST_BYTES = 64 * 1024 * 1024

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PreparedInvocation:
    """Locally computable identity for one exact submitted request."""

    invocation_id: str
    identity: dict[str, object]

    @property
    def receipt_path(self) -> PurePosixPath:
        """Return the stable output-Volume receipt path."""
        return (
            PurePosixPath("invocations")
            / self.invocation_id[:2]
            / f"{self.invocation_id}.json"
        )


def _validate_invocation(invocation: PreparedInvocation) -> None:
    if (
        _DIGEST_PATTERN.fullmatch(invocation.invocation_id) is None
        or invocation.invocation_id != hash_sequences(invocation.identity)
        or invocation.identity.get("schema") != INVOCATION_IDENTITY_SCHEMA
    ):
        raise ValueError("Prepared invocation identity is invalid")


def _search_contract() -> dict[str, object]:
    return {
        "msa": {
            "identity_schema_version": SEARCH_IDENTITY_SCHEMA_VERSION,
            "raw_result_schema_version": RAW_RESULT_SCHEMA_VERSION,
            "combined_result_schema_version": COMBINED_RESULT_SCHEMA_VERSION,
            "adapter_version": SEARCH_ADAPTER_VERSION,
            "profiles": [
                {
                    "profile_id": spec.profile_id,
                    "parameters": scientific_search_parameters(spec),
                }
                for spec in DATABASE_PROFILE_SPECS
            ],
            "hmmer_version": HMMER_VERSION,
            "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        },
        "protein_templates": {
            "identity_schema_version": TEMPLATE_IDENTITY_SCHEMA_VERSION,
            "result_schema_version": TEMPLATE_RESULT_SCHEMA_VERSION,
            "adapter_version": TEMPLATE_ADAPTER_VERSION,
            "parameters": template_search_parameters(DEFAULT_MAX_TEMPLATE_DATE),
            "hmmer_version": HMMER_VERSION,
        },
    }


def prepare_invocation(
    config: AF3Config,
    *,
    search_msa: bool,
    search_protein_templates: bool,
    recycle: int,
    sample: int,
) -> PreparedInvocation:
    """Build the exact pre-enrichment invocation identity."""
    if not isinstance(search_msa, bool):
        raise TypeError("search_msa must be a boolean")
    if not isinstance(search_protein_templates, bool):
        raise TypeError("search_protein_templates must be a boolean")
    validate_inference_parameters(recycle, sample)
    validated = validate_submitted_af3_input(config)
    validate_inference_workload(validated.modelSeeds, sample)
    identity: dict[str, object] = {
        "schema": INVOCATION_IDENTITY_SCHEMA,
        "submitted_input": build_inference_identity_view(validated),
        "presentation": {
            "display_name": validated.name,
            "submitted_seeds": list(validated.modelSeeds),
        },
        "search": {
            "search_msa": search_msa,
            "search_protein_templates": search_protein_templates,
            "contract": _search_contract(),
        },
        "inference": {
            "num_recycles": recycle,
            "num_diffusion_samples": sample,
            "run_identity_schema": RUN_IDENTITY_SCHEMA,
            "staged_input_schema_version": STAGED_INPUT_SCHEMA_VERSION,
        },
        "publication": {
            "manifest_schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
            "view_identity_schema": REQUEST_VIEW_IDENTITY_SCHEMA,
        },
        "app": {
            "app_name": "AlphaFold3",
            "app_version": ALPHAFOLD3_APP_VERSION,
            "alphafold_commit": ALPHAFOLD3_COMMIT,
        },
        "declared_model_identity": DECLARED_MODEL_IDENTITY,
    }
    return PreparedInvocation(
        invocation_id=hash_sequences(identity),
        identity=identity,
    )


def _manifest_record(manifest: dict[str, object]) -> dict[str, object]:
    publication = request_publication_from_manifest(manifest)
    path = request_manifest_path(publication)
    content = json_bytes(manifest)
    if not 0 < len(content) <= MAX_REQUEST_MANIFEST_BYTES:
        raise ValueError("Request manifest exceeds its byte limit")
    return {
        "path": path.as_posix(),
        "size_bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def build_invocation_receipt(
    invocation: PreparedInvocation,
    prepared: PreparedInferenceRun,
    manifest: dict[str, object],
) -> VolumeUpload:
    """Bind one exact invocation to its completed request manifest."""
    _validate_invocation(invocation)
    publication = request_publication_from_manifest(manifest)
    if publication != RequestPublication.from_prepared(prepared):
        raise ValueError("Request manifest does not match the prepared run")
    view_id = request_view_id(
        publication.request_id,
        publication.submitted_seeds,
        publication.display_name,
    )
    presentation = invocation.identity.get("presentation")
    inference = invocation.identity.get("inference")
    if (
        not isinstance(presentation, dict)
        or presentation.get("display_name") != publication.display_name
        or presentation.get("submitted_seeds") != list(publication.submitted_seeds)
        or not isinstance(inference, dict)
        or inference.get("num_recycles") != prepared.recycle
        or inference.get("num_diffusion_samples") != publication.sample_count
    ):
        raise ValueError("Request manifest does not match the invocation")
    receipt = VolumeUpload(
        relative_path=invocation.receipt_path,
        content=json_bytes({
            "schema_version": INVOCATION_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "invocation_id": invocation.invocation_id,
            "run_id": publication.run_id,
            "request_id": publication.request_id,
            "view_id": view_id,
            "manifest": _manifest_record(manifest),
        }),
    )
    if len(receipt.content) > MAX_INVOCATION_RECEIPT_BYTES:
        raise RuntimeError("Invocation receipt exceeds its byte limit")
    return receipt


def _artifact_record(value: object, *, field_name: str) -> tuple[str, int, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} is invalid")
    path = value.get("path")
    size_bytes = value.get("size_bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or not 0 < size_bytes <= MAX_REQUEST_MANIFEST_BYTES
        or not isinstance(digest, str)
        or _DIGEST_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError(f"{field_name} is invalid")
    return path, size_bytes, digest


def load_invocation_manifest(
    reader: VolumeReader,
    invocation: PreparedInvocation,
) -> dict[str, object] | None:
    """Resolve one exact invocation to a validated completed manifest."""
    _validate_invocation(invocation)
    receipt_bytes = read_volume_bytes(
        reader,
        invocation.receipt_path.as_posix(),
        max_bytes=MAX_INVOCATION_RECEIPT_BYTES,
    )
    if receipt_bytes is None:
        return None
    try:
        receipt = orjson.loads(receipt_bytes)
    except orjson.JSONDecodeError as exc:
        raise RuntimeError("Invocation receipt is unreadable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != INVOCATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("invocation_id") != invocation.invocation_id
    ):
        raise RuntimeError("Invocation receipt identity is invalid")
    manifest_path, manifest_size, manifest_sha256 = _artifact_record(
        receipt.get("manifest"),
        field_name="Invocation manifest record",
    )
    manifest_bytes = read_volume_bytes(
        reader,
        manifest_path,
        max_bytes=manifest_size,
    )
    if manifest_bytes is None:
        raise RuntimeError("Invocation receipt references a missing manifest")
    if (
        len(manifest_bytes) != manifest_size
        or sha256_bytes(manifest_bytes) != manifest_sha256
    ):
        raise RuntimeError("Invocation receipt manifest digest is invalid")
    try:
        manifest = orjson.loads(manifest_bytes)
    except orjson.JSONDecodeError as exc:
        raise RuntimeError("Invocation receipt manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Invocation receipt manifest is invalid")
    selected = cast(dict[str, object], manifest)
    publication = request_publication_from_manifest(selected)
    expected_view_id = request_view_id(
        publication.request_id,
        publication.submitted_seeds,
        publication.display_name,
    )
    if (
        receipt.get("run_id") != publication.run_id
        or receipt.get("request_id") != publication.request_id
        or receipt.get("view_id") != expected_view_id
        or manifest_path != request_manifest_path(publication).as_posix()
    ):
        raise RuntimeError("Invocation receipt manifest identity is invalid")
    return selected
