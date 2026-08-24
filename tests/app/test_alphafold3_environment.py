"""Tests for automatic AlphaFold3 environment preparation."""

# ruff: noqa: D101,D102,D103,D107

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from uniaf3.schema.alphafold3 import (
    AF3RNA,
    AF3Config,
    AF3Protein,
    AF3SequenceEntry,
)

import biomodals.app.fold.alphafold3.environment as environment_module
from biomodals.app.fold.alphafold3.environment import (
    EnvironmentAsset,
    EnvironmentRuntime,
    acquire_asset_claim,
    asset_ready,
    prepare_environment_asset,
    required_environment_assets,
)
from biomodals.app.fold.alphafold3.generation_claims import generation_status


class FakeVolume:
    def __init__(self) -> None:
        self.reloads = 0
        self.commits = 0

    def reload(self) -> None:
        self.reloads += 1

    def commit(self) -> None:
        self.commits += 1


class FakeClaims:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)


def _config(entry: AF3SequenceEntry) -> AF3Config:
    return AF3Config(name="example", modelSeeds=[1], sequences=[entry])


def _runtime(tmp_path: Path, claims: FakeClaims) -> EnvironmentRuntime:
    volume = cast(Any, FakeVolume())
    return EnvironmentRuntime(
        model_volume=volume,
        source_volume=volume,
        sharded_volume=volume,
        claims=cast(Any, claims),
        container_id="coordinator",
        model_root=tmp_path / "models",
        source_root=tmp_path / "source",
        sharded_root=tmp_path / "sharded",
    )


def test_protein_request_prepares_only_protein_search_assets() -> None:
    assets = required_environment_assets(
        _config(AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))),
        search_msa=True,
        search_protein_templates=True,
    )

    assert [asset.key for asset in assets] == [
        "model",
        "profile:small-bfd-64-v2",
        "profile:mgnify-512-v1",
        "profile:uniprot-384-v1",
        "profile:uniref90-256-v1",
        "template:pdb-seqres",
        "template:mmcif",
    ]


def test_rna_request_does_not_prepare_protein_or_template_assets() -> None:
    assets = required_environment_assets(
        _config(AF3SequenceEntry(rna=AF3RNA(id="A", sequence="ACGU"))),
        search_msa=True,
        search_protein_templates=True,
    )

    assert [asset.key for asset in assets] == [
        "model",
        "profile:nt-rna-256-v1",
        "profile:rfam-16-v1",
        "profile:rnacentral-64-v1",
    ]


def test_custom_msa_request_can_skip_all_search_assets() -> None:
    config = _config(
        AF3SequenceEntry(
            protein=AF3Protein(
                id="A",
                sequence="ACDE",
                unpairedMsa=">query\nACDE\n",
                pairedMsa=">query\nACDE\n",
                templates=[],
            )
        )
    )

    assets = required_environment_assets(
        config,
        search_msa=True,
        search_protein_templates=False,
    )

    assert assets == (EnvironmentAsset("model", "model"),)


def test_readiness_checks_only_the_expected_final_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeClaims())
    model = runtime.model_root / "AlphaFold3" / "af3.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")

    assert asset_ready(runtime, EnvironmentAsset("model", "model")) is True


def test_only_one_generation_claims_a_missing_asset(tmp_path: Path) -> None:
    claims = FakeClaims()
    first_runtime = _runtime(tmp_path, claims)
    second_runtime = replace(first_runtime, container_id="other-coordinator")
    asset = EnvironmentAsset("model", "model")

    assert acquire_asset_claim(first_runtime, asset, "first") is not None
    assert acquire_asset_claim(second_runtime, asset, "second") is None


def test_worker_completes_the_coordinator_claim_for_a_reused_asset(
    tmp_path: Path,
) -> None:
    claims = FakeClaims()
    coordinator = _runtime(tmp_path, claims)
    asset = EnvironmentAsset("model", "model")
    model = coordinator.model_root / "AlphaFold3" / "af3.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    assert acquire_asset_claim(coordinator, asset, "generation") is not None

    result = prepare_environment_asset(
        replace(coordinator, container_id="worker"),
        asset,
        "generation",
        build_profile=lambda *_arguments: {},
    )

    assert result == {"status": "reused", "asset_key": "model"}
    status = generation_status(claims, "model", "generation")
    assert status is not None
    assert status["status"] == "complete"
    assert status["asset_key"] == "model"
    assert status["result_status"] == "reused"


def test_compressed_file_is_published_only_after_decompression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = FakeVolume()

    def download(urls, **kwargs):
        del kwargs
        [path] = urls.values()
        Path(path).write_bytes(b"compressed")

    def decompress(command, *, check):
        assert check is True
        Path(command[-1]).write_bytes(b"model")

    monkeypatch.setattr(environment_module, "download_files", download)
    monkeypatch.setattr(environment_module, "require_executable", lambda name: name)
    monkeypatch.setattr(environment_module.subprocess, "run", decompress)

    environment_module._prepare_compressed_file(
        tmp_path,
        Path("AlphaFold3/af3.bin"),
        "https://example.test/af3.bin.zst",
        "generation",
        volume,
    )

    assert (tmp_path / "AlphaFold3" / "af3.bin").read_bytes() == b"model"
    assert not (tmp_path / ".setup" / "af3.bin.zst.part").exists()
    assert volume.commits == 1

    environment_module._prepare_compressed_file(
        tmp_path,
        Path("profile.fasta"),
        "https://example.test/profile.fasta.zst",
        "profile-generation",
        volume,
        commit=False,
    )

    assert (tmp_path / "profile.fasta").read_bytes() == b"model"
    assert volume.commits == 1
