"""Retained AlphaFold3 request validation outside service SQLite."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import orjson
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    MAX_SEED_SAMPLE_PAIRS,
    MAX_STAGED_INPUT_BYTES,
    build_inference_identity_view,
    normalize_model_seeds,
    serialize_af3_input,
)
from biomodals.helper.artifacts import replace_bytes_atomic

MAX_VALIDATION_BYTES = MAX_STAGED_INPUT_BYTES
MAX_JOB_NAME_LENGTH = 120
VALIDATION_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    """Local retention limits for validated AlphaFold3 inputs."""

    max_user_count: int = 8
    max_user_bytes: int = 1024**3
    max_total_count: int = 64
    max_total_bytes: int = 8 * 1024**3
    min_free_bytes: int = 1024**3


class ValidationLimitExceededError(RuntimeError):
    """Retained validation count or byte capacity is exhausted."""


class ValidationStorageLowError(RuntimeError):
    """The state filesystem cannot safely retain another validation."""


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    """Operational AlphaFold3 settings retained beside a native document."""

    search_msa: bool = True
    search_protein_templates: bool = True
    recycle: int = 10
    sample: int = 5


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    """One owner-private retained native AlphaFold3 document."""

    validation_id: UUID
    owner_user_id: UUID
    digest: str
    created_at: int
    expires_at: int
    settings: ValidationSettings
    preview: dict[str, object]
    directory: Path

    @property
    def document_path(self) -> Path:
        """Return the retained normalized native JSON document."""
        return self.directory / "document.json"

    @property
    def publication_scope_digest(self) -> str:
        """Identify requests that may share incomplete remote publications."""
        config = AF3Config.model_validate_json(self.document_path.read_bytes())
        return sha256(
            orjson.dumps(
                {
                    "input": build_inference_identity_view(config),
                    "search_msa": self.settings.search_msa,
                    "search_protein_templates": (
                        self.settings.search_protein_templates
                    ),
                    "recycle": self.settings.recycle,
                    "sample": self.settings.sample,
                },
                option=orjson.OPT_SORT_KEYS,
            )
        ).hexdigest()

    def request(
        self,
        *,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
        repair_execution_run_ids: tuple[UUID, ...] = (),
    ) -> AlphaFold3ExecutionRequest:
        """Revalidate retained bytes into the app's trusted request contract."""
        config = AF3Config.model_validate_json(self.document_path.read_bytes())
        return AlphaFold3ExecutionRequest.prepare(
            config,
            search_msa=self.settings.search_msa,
            search_protein_templates=self.settings.search_protein_templates,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            allow_large_inference=bool(self.preview["requires_confirmation"]),
            recycle=self.settings.recycle,
            sample=self.settings.sample,
            repair_execution_run_ids=repair_execution_run_ids,
        )


class ValidatedInputStore:
    """Atomically retain successful validations in the configured state dir."""

    def __init__(
        self,
        state_directory: Path,
        *,
        limits: ValidationLimits | None = None,
    ) -> None:
        """Place validation resources below the service state directory."""
        self.directory = state_directory / "validated-inputs"
        self.limits = limits if limits is not None else ValidationLimits()

    def initialize(self) -> None:
        """Create private validation storage."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)

    def validate_and_publish(
        self,
        source: Path,
        *,
        owner_user_id: UUID,
        digest: str,
        settings: ValidationSettings,
        now: int | None = None,
        validation_id: UUID | None = None,
    ) -> ValidatedInput:
        """Apply the existing app validator, then atomically publish a resource."""
        size = source.stat().st_size
        if not 0 < size <= MAX_VALIDATION_BYTES:
            raise ValueError("AlphaFold3 document has an invalid size")
        self.require_capacity(owner_user_id, additional_bytes=size)
        content = source.read_bytes()
        if sha256(content).hexdigest() != digest:
            raise ValueError("AlphaFold3 document digest changed during validation")
        document = orjson.loads(content)
        _reject_path_fields(document)
        config = AF3Config.model_validate(document)
        if len(config.name) > MAX_JOB_NAME_LENGTH:
            raise ValueError(
                f"AlphaFold3 Job name exceeds {MAX_JOB_NAME_LENGTH} characters"
            )
        normalized = orjson.loads(serialize_af3_input(config))
        prediction_count = (
            len(normalize_model_seeds(config.modelSeeds)) * settings.sample
        )
        request = AlphaFold3ExecutionRequest.prepare(
            config,
            search_msa=settings.search_msa,
            search_protein_templates=settings.search_protein_templates,
            allow_large_inference=prediction_count > MAX_SEED_SAMPLE_PAIRS,
            recycle=settings.recycle,
            sample=settings.sample,
        )
        del request
        created_at = int(time.time()) if now is None else now
        selected_id = validation_id or uuid4()
        target = self.directory / str(selected_id)
        staging = self.directory / f".{selected_id}.tmp"
        if target.exists() or staging.exists():
            raise FileExistsError(f"Validation already exists: {selected_id}")
        document_content = orjson.dumps(normalized)
        preview = _preview(normalized, settings, prediction_count)
        metadata_content = orjson.dumps(
            {
                "validation_id": str(selected_id),
                "owner_user_id": str(owner_user_id),
                "digest": digest,
                "created_at": created_at,
                "expires_at": created_at + VALIDATION_TTL_SECONDS,
                "settings": {
                    "search_msa": settings.search_msa,
                    "search_protein_templates": settings.search_protein_templates,
                    "recycle": settings.recycle,
                    "sample": settings.sample,
                },
                "preview": preview,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        self.require_capacity(
            owner_user_id,
            additional_bytes=len(document_content) + len(metadata_content),
        )
        staging.mkdir(mode=0o700)
        try:
            replace_bytes_atomic(staging / "document.json", document_content)
            replace_bytes_atomic(staging / "metadata.json", metadata_content)
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return self._load(target)

    def get(
        self,
        validation_id: UUID,
        *,
        owner_user_id: UUID,
        now: int | None = None,
    ) -> ValidatedInput | None:
        """Load an unexpired owner-private validation."""
        target = self.directory / str(validation_id)
        try:
            resource = self._load(target)
        except FileNotFoundError:
            return None
        observed_at = int(time.time()) if now is None else now
        if (
            resource.owner_user_id != owner_user_id
            or observed_at >= resource.expires_at
        ):
            return None
        return resource

    def delete(self, validation_id: UUID, *, owner_user_id: UUID) -> bool:
        """Delete one owner-private unsubmitted validation."""
        target = self.directory / str(validation_id)
        try:
            resource = self._load(target)
        except FileNotFoundError:
            return False
        if resource.owner_user_id != owner_user_id:
            return False
        shutil.rmtree(target)
        return True

    def get_claimed(
        self,
        validation_id: UUID,
        *,
        owner_user_id: UUID,
    ) -> ValidatedInput | None:
        """Load an admitted validation after ordinary expiry no longer applies."""
        try:
            resource = self._load(self.directory / str(validation_id))
        except FileNotFoundError:
            return None
        return resource if resource.owner_user_id == owner_user_id else None

    def delete_claimed(self, validation_id: UUID) -> None:
        """Delete one resource after the service verified remote staging."""
        shutil.rmtree(self.directory / str(validation_id), ignore_errors=True)

    def cleanup_expired(self, *, claimed: set[UUID], now: int) -> int:
        """Remove expired unclaimed resources and abandoned staging dirs."""
        removed = 0
        for path in self.directory.iterdir():
            if path.name.startswith("."):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
                continue
            try:
                validation_id = UUID(path.name)
                resource = self._load(path)
            except (ValueError, FileNotFoundError):
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
                continue
            if validation_id not in claimed and now >= resource.expires_at:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def usage(self) -> tuple[int, int]:
        """Return retained validation count and bytes for Admin storage views."""
        return self._usage()

    def require_free_space(self, additional_bytes: int = 0) -> None:
        """Reject writes that would consume the state filesystem reserve."""
        if (
            shutil.disk_usage(self.directory).free - additional_bytes
            < self.limits.min_free_bytes
        ):
            raise ValidationStorageLowError(
                "AlphaFold3 validation storage has insufficient free space"
            )

    def require_capacity(
        self,
        owner_user_id: UUID,
        *,
        additional_bytes: int,
    ) -> None:
        """Reject a validation before parsing when retention is full."""
        total_count, total_bytes = self._usage()
        user_count, user_bytes = self._usage(owner_user_id=owner_user_id)
        if (
            total_count >= self.limits.max_total_count
            or total_bytes + additional_bytes > self.limits.max_total_bytes
            or user_count >= self.limits.max_user_count
            or user_bytes + additional_bytes > self.limits.max_user_bytes
        ):
            raise ValidationLimitExceededError(
                "AlphaFold3 validation retention limit reached; delete an existing "
                "validation or retry after it expires"
            )
        self.require_free_space(additional_bytes)

    def _usage(self, *, owner_user_id: UUID | None = None) -> tuple[int, int]:
        count = 0
        size = 0
        for directory in self.directory.iterdir():
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                resource = self._load(directory)
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if owner_user_id is not None and resource.owner_user_id != owner_user_id:
                continue
            count += 1
            size += sum(
                path.stat().st_size for path in directory.iterdir() if path.is_file()
            )
        return count, size

    @staticmethod
    def _load(directory: Path) -> ValidatedInput:
        metadata = orjson.loads((directory / "metadata.json").read_bytes())
        settings = metadata["settings"]
        return ValidatedInput(
            validation_id=UUID(metadata["validation_id"]),
            owner_user_id=UUID(metadata["owner_user_id"]),
            digest=str(metadata["digest"]),
            created_at=int(metadata["created_at"]),
            expires_at=int(metadata["expires_at"]),
            settings=ValidationSettings(**settings),
            preview=dict(metadata["preview"]),
            directory=directory,
        )


def _reject_path_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower().endswith("path"):
                raise ValueError(
                    f"Path-valued field is not supported by the API: {key}"
                )
            _reject_path_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_path_fields(item)


def _preview(
    document: dict[str, Any],
    settings: ValidationSettings,
    prediction_count: int,
) -> dict[str, object]:
    entities: list[dict[str, object]] = []
    modifications = templates = custom_msas = 0
    for entry in document.get("sequences", []):
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        entity_type, raw = next(iter(entry.items()))
        if not isinstance(raw, dict):
            continue
        identifiers = raw.get("id", [])
        ids = [identifiers] if isinstance(identifiers, str) else list(identifiers)
        sequence = raw.get("sequence", "")
        modifications += len(raw.get("modifications", []))
        templates += len(raw.get("templates", []))
        custom_msas += sum(
            bool(raw.get(field)) for field in ("unpairedMsa", "pairedMsa")
        )
        entities.append({
            "type": entity_type,
            "ids": ids,
            "copies": len(ids),
            "length": len(sequence) if isinstance(sequence, str) else None,
        })
    seeds = list(document.get("modelSeeds", []))
    return {
        "name": document.get("name"),
        "entities": entities,
        "seeds": seeds,
        "seed_count": len(normalize_model_seeds(seeds)),
        "samples": settings.sample,
        "prediction_count": prediction_count,
        "requires_confirmation": prediction_count > MAX_SEED_SAMPLE_PAIRS,
        "search_msa": settings.search_msa,
        "search_protein_templates": settings.search_protein_templates,
        "recycle": settings.recycle,
        "advanced_counts": {
            "modifications": modifications,
            "bonds": len(document.get("bondedAtomPairs", [])),
            "custom_msas": custom_msas,
            "custom_templates": templates,
            "custom_ccd": int(bool(document.get("userCCD"))),
        },
        "warnings": (
            [f"This request creates {prediction_count:,} predictions."]
            if prediction_count > MAX_SEED_SAMPLE_PAIRS
            else []
        ),
    }
