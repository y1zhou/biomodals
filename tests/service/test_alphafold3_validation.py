"""Retained AlphaFold3 API validation contracts."""

# ruff: noqa: D103

import hashlib
from pathlib import Path
from uuid import UUID

import orjson
import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

from biomodals.app.fold.alphafold3.inference_inputs import serialize_af3_input
from biomodals.service.alphafold3.validation import (
    MAX_VALIDATION_BYTES,
    ValidatedInputStore,
    ValidationSettings,
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


def test_api_limit_is_256_mib_and_cli_limit_is_unchanged() -> None:
    from biomodals.app.fold.alphafold3.inference_inputs import MAX_STAGED_INPUT_BYTES

    assert MAX_VALIDATION_BYTES == 256 * 1024 * 1024
    assert MAX_STAGED_INPUT_BYTES == 1024 * 1024 * 1024


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
