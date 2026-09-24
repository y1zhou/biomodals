"""Retained AlphaFold3 API validation contracts."""

# ruff: noqa: D103

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.chemistry import ChemistryReceipt
from biomodals.app.fold.alphafold3.inference_inputs import serialize_af3_input
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.execution import DeploymentIdentity
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


def _publish(store, source, *, owner_user_id, digest, settings, now=None):
    prepared = store.prepare(
        source, owner_user_id=owner_user_id, digest=digest, settings=settings
    )
    if any(prepared.preview["chemistry"].values()):
        prepared = replace(
            prepared,
            chemistry_receipt=ChemistryReceipt(
                input_sha256=hashlib.sha256(prepared.content).hexdigest(),
                ccd_sha256="a" * 64,
                upstream_commit=ALPHAFOLD3_COMMIT,
            ),
            chemistry_deployment=DeploymentIdentity("test", "af3", 1),
        )
    return store.publish(prepared, owner_user_id=owner_user_id, now=now)


def test_api_and_standalone_upload_limits_are_256_mib() -> None:
    from biomodals.app.fold.alphafold3.inference_inputs import MAX_INPUT_JSON_BYTES
    from biomodals.app.fold.alphafold3.template_search import (
        MAX_TEMPLATE_INSPECTION_BYTES,
    )

    assert MAX_VALIDATION_BYTES == 256 * 1024 * 1024
    assert MAX_INPUT_JSON_BYTES == 256 * 1024 * 1024
    assert MAX_TEMPLATE_INSPECTION_BYTES == 1024 * 1024 * 1024


def test_validation_retains_native_document_and_bounded_preview(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)

    validated = _publish(
        store,
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


def test_publication_scope_ignores_name_and_seed_selection(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    document = orjson.loads(_document())

    def validate(value: dict[str, object]):
        content = orjson.dumps(value)
        source.write_bytes(content)
        return _publish(
            store,
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(content).hexdigest(),
            settings=ValidationSettings(sample=5),
        )

    first = validate(document)
    document["name"] = "renamed"
    document["modelSeeds"] = [2, 3, 4]
    renamed = validate(document)
    document["sequences"][0]["protein"]["sequence"] = "ACDF"
    changed = validate(document)

    assert first.publication_scope_digest == renamed.publication_scope_digest
    assert changed.publication_scope_digest != first.publication_scope_digest


def test_validation_allows_zero_recycles_and_bounds_job_name(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)

    validated = _publish(
        store,
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
        _publish(
            store,
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(oversized).hexdigest(),
            settings=ValidationSettings(),
        )


def test_template_search_requires_msa_search(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    content = _document()
    source.write_bytes(content)

    with pytest.raises(ValueError, match="template search requires MSA search"):
        _publish(
            store,
            source,
            owner_user_id=OWNER,
            digest=hashlib.sha256(content).hexdigest(),
            settings=ValidationSettings(
                search_msa=False,
                search_protein_templates=True,
            ),
        )


def test_validation_preview_separates_custom_inputs(tmp_path: Path) -> None:
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    source = tmp_path / "input.json"
    document = orjson.loads(_document())
    document["sequences"][0]["protein"]["unpairedMsa"] = ">query\nACDE\n"
    document["sequences"][0]["protein"]["pairedMsa"] = ""
    document["userCCD"] = "data_custom"
    content = orjson.dumps(document)
    source.write_bytes(content)

    validated = _publish(
        store,
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
        _publish(
            store,
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
    validated = _publish(
        store,
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
    _publish(
        store,
        source,
        owner_user_id=OWNER,
        digest=digest,
        settings=ValidationSettings(),
    )
    with pytest.raises(ValidationLimitExceededError):
        _publish(
            store,
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

    _publish(
        store,
        source,
        owner_user_id=OWNER,
        digest=digest,
        settings=ValidationSettings(),
    )
    with pytest.raises(ValidationLimitExceededError):
        _publish(
            store,
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
        _publish(
            store,
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


def test_chemistry_publication_requires_evidence_for_exact_normalized_bytes(tmp_path):
    store = ValidatedInputStore(tmp_path)
    store.initialize()
    document = orjson.loads(_document())
    document["sequences"][0]["protein"]["modifications"] = [
        {"ptmType": "ALY", "ptmPosition": 1}
    ]
    content = orjson.dumps(document)
    source = tmp_path / "input.json"
    source.write_bytes(content)
    prepared = store.prepare(
        source,
        owner_user_id=OWNER,
        digest=hashlib.sha256(content).hexdigest(),
        settings=ValidationSettings(),
    )
    with pytest.raises(ValueError, match="does not match"):
        store.publish(prepared, owner_user_id=OWNER)
    wrong = replace(
        prepared,
        chemistry_deployment=DeploymentIdentity("main", "AF3", 1),
        chemistry_receipt=ChemistryReceipt(
            input_sha256="f" * 64,
            ccd_sha256="a" * 64,
            upstream_commit=ALPHAFOLD3_COMMIT,
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        store.publish(wrong, owner_user_id=OWNER)
    assert store.usage() == (0, 0)
