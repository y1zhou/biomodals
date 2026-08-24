"""Retained AlphaFold3 API validation contracts."""

# ruff: noqa: D103

import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.inference_inputs import serialize_af3_input
from biomodals.service.alphafold3.validation import (
    MAX_VALIDATION_BYTES,
    ValidatedInputStore,
    ValidationLimitExceededError,
    ValidationLimits,
    ValidationSettings,
    ValidationStorageLowError,
)

OWNER = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _document() -> bytes:
    return serialize_af3_input(
        AF3Config(
            name="example",
            modelSeeds=[1, 2],
            sequences=[AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))],
        )
    )


def test_api_and_standalone_limits_are_256_mib() -> None:
    from biomodals.app.fold.alphafold3.inference_inputs import MAX_STAGED_INPUT_BYTES
    from biomodals.app.fold.alphafold3.template_search import (
        MAX_TEMPLATE_INSPECTION_BYTES,
    )

    assert MAX_VALIDATION_BYTES == 256 * 1024 * 1024
    assert MAX_STAGED_INPUT_BYTES == 256 * 1024 * 1024
    assert MAX_TEMPLATE_INSPECTION_BYTES == 1024 * 1024 * 1024


def test_validation_retains_native_document_and_bounded_preview(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)

    validated = store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=hashlib.sha256(content).hexdigest(),
        settings=ValidationSettings(sample=5),
        now=10,
    )

    assert validated.preview["name"] == "example"
    assert validated.preview["prediction_count"] == 10
    assert validated.preview["requires_confirmation"] is False
    assert orjson.loads(validated.document_path.read_bytes())["name"] == "example"
    assert store.get(validated.validation_id, owner_user_id=OWNER, now=11) is not None


def test_validation_allows_zero_recycles_and_bounds_job_name(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)

    validated = store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=hashlib.sha256(content).hexdigest(),
        settings=ValidationSettings(recycle=0),
    )
    assert validated.settings.recycle == 0

    document = orjson.loads(content)
    document["name"] = "x" * 121
    oversized = orjson.dumps(document)
    source.write_bytes(oversized)
    with pytest.raises(ValueError, match="name exceeds 120"):
        store.validate_and_publish(
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(oversized).hexdigest(),
            settings=ValidationSettings(),
        )


def test_validation_preview_separates_custom_inputs(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    document = orjson.loads(_document())
    document["sequences"][0]["protein"]["unpairedMsa"] = ">query\nACDE\n"
    document["userCCD"] = "data_custom"
    content = orjson.dumps(document)
    source.write_bytes(content)

    validated = store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=hashlib.sha256(content).hexdigest(),
        settings=ValidationSettings(sample=5),
    )

    assert validated.preview["advanced_counts"] == {
        "modifications": 0,
        "bonds": 0,
        "custom_msas": 1,
        "custom_templates": 0,
        "custom_ccd": 1,
    }


def test_expert_documents_reject_browser_local_paths(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    document = orjson.loads(_document())
    document["sequences"][0]["protein"]["unpairedMsaPath"] = "/remote/msa.a3m"
    content = orjson.dumps(document)
    source.write_bytes(content)

    with pytest.raises(ValueError, match="Path-valued"):
        store.validate_and_publish(
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(content).hexdigest(),
            settings=ValidationSettings(),
        )


def test_claimed_validation_survives_ordinary_expiry(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)
    validated = store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=hashlib.sha256(content).hexdigest(),
        settings=ValidationSettings(),
        now=10,
    )

    assert store.get(validated.validation_id, owner_user_id=OWNER, now=10**9) is None
    assert store.get_claimed(validated.validation_id, owner_user_id=OWNER) == validated


def test_validation_enforces_per_user_retention_limit(tmp_path: Path) -> None:
    store = ValidatedInputStore(
        tmp_path,
        limits=ValidationLimits(
            max_user_count=1,
            max_user_bytes=1024**2,
            max_total_count=2,
            max_total_bytes=2 * 1024**2,
            min_free_bytes=0,
        ),
    )
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=digest,
        settings=ValidationSettings(),
    )
    with pytest.raises(ValidationLimitExceededError):
        store.validate_and_publish(
            source,
            owner_user_id=OWNER,
            digest=digest,
            settings=ValidationSettings(),
        )


def test_validation_enforces_global_retention_limit(tmp_path: Path) -> None:
    store = ValidatedInputStore(
        tmp_path,
        limits=ValidationLimits(
            max_user_count=2,
            max_user_bytes=1024**2,
            max_total_count=1,
            max_total_bytes=2 * 1024**2,
            min_free_bytes=0,
        ),
    )
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    store.validate_and_publish(
        source,
        owner_user_id=OWNER,
        digest=digest,
        settings=ValidationSettings(),
    )
    with pytest.raises(ValidationLimitExceededError):
        store.validate_and_publish(
            source,
            owner_user_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            digest=digest,
            settings=ValidationSettings(),
        )


def test_validation_enforces_retained_byte_limit(tmp_path: Path) -> None:
    content = _document()
    store = ValidatedInputStore(
        tmp_path,
        limits=ValidationLimits(
            max_user_count=2,
            max_user_bytes=len(content) - 1,
            max_total_count=2,
            max_total_bytes=1024**2,
            min_free_bytes=0,
        ),
    )
    store.initialize()
    source = tmp_path / "input.json"
    source.write_bytes(content)

    with pytest.raises(ValidationLimitExceededError):
        store.validate_and_publish(
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(content).hexdigest(),
            settings=ValidationSettings(),
        )


def test_validation_reserves_minimum_free_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    monkeypatch.setattr(
        "biomodals.service.alphafold3.validation.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(ValidationStorageLowError):
        store.require_free_space()
