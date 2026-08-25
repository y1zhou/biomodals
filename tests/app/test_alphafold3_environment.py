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
from biomodals.app.fold.alphafold3.generation_claims import (
    finish_generation_claim,
    generation_status,
)
from biomodals.app.fold.alphafold3.profiles import resolve_database_profile

GENERATION_ID = "a" * 64
OTHER_GENERATION_ID = "b" * 64


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
        scratch_root=tmp_path / "scratch",
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

    assert assets == (EnvironmentAsset("model"),)


def test_readiness_checks_only_the_expected_final_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeClaims())
    model = runtime.model_root / "AlphaFold3" / "af3.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")

    assert asset_ready(runtime, EnvironmentAsset("model")) is True


def test_only_one_generation_claims_a_missing_asset(tmp_path: Path) -> None:
    claims = FakeClaims()
    first_runtime = _runtime(tmp_path, claims)
    second_runtime = replace(first_runtime, container_id="other-coordinator")
    asset = EnvironmentAsset("model")

    assert acquire_asset_claim(first_runtime, asset, GENERATION_ID) is not None
    assert acquire_asset_claim(second_runtime, asset, OTHER_GENERATION_ID) is None


def test_worker_completes_the_coordinator_claim_for_a_reused_asset(
    tmp_path: Path,
) -> None:
    claims = FakeClaims()
    coordinator = _runtime(tmp_path, claims)
    asset = EnvironmentAsset("model")
    model = coordinator.model_root / "AlphaFold3" / "af3.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    assert acquire_asset_claim(coordinator, asset, GENERATION_ID) is not None

    result = prepare_environment_asset(
        replace(coordinator, container_id="worker"),
        asset,
        GENERATION_ID,
        build_profile=lambda *_arguments: {},
    )

    assert result == {"status": "reused", "asset_key": "model"}
    status = generation_status(claims, "model", GENERATION_ID)
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
        GENERATION_ID,
        volume,
    )

    assert (tmp_path / "AlphaFold3" / "af3.bin").read_bytes() == b"model"
    assert not (tmp_path / ".setup" / "af3.bin.zst.part").exists()
    assert volume.commits == 1


def test_profile_source_is_local_until_profile_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, FakeClaims())
    spec = resolve_database_profile("small_bfd")

    def download(urls, **kwargs):
        del kwargs
        [path] = urls.values()
        Path(path).write_bytes(b"compressed")

    def decompress(command, *, check):
        assert check is True
        Path(command[-1]).write_bytes(b">sequence\nACDE\n")

    monkeypatch.setattr(environment_module, "download_files", download)
    monkeypatch.setattr(environment_module, "require_executable", lambda name: name)
    monkeypatch.setattr(environment_module.subprocess, "run", decompress)

    source_path, partial_path = environment_module._prepare_profile_source(
        runtime,
        spec,
        GENERATION_ID,
    )

    assert source_path == runtime.scratch_root / GENERATION_ID / spec.source_filename
    assert source_path.read_text() == ">sequence\nACDE\n"
    assert partial_path.is_file()
    assert not (runtime.source_root / spec.source_filename).exists()


def test_successful_profile_preparation_cleans_temporary_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, FakeClaims())
    spec = resolve_database_profile("small_bfd")

    def download(urls, **kwargs):
        del kwargs
        [path] = urls.values()
        Path(path).write_bytes(b"compressed")

    def decompress(command, *, check):
        assert check is True
        Path(command[-1]).write_bytes(b">sequence\nACDE\n")

    observed_source: Path | None = None

    def build_profile(database_id: str, generation_id: str, source: Path):
        nonlocal observed_source
        assert database_id == spec.database_id
        assert generation_id == GENERATION_ID
        assert source.is_file()
        observed_source = source
        return {"status": "published"}

    monkeypatch.setattr(environment_module, "download_files", download)
    monkeypatch.setattr(environment_module, "require_executable", lambda name: name)
    monkeypatch.setattr(environment_module.subprocess, "run", decompress)

    result = prepare_environment_asset(
        runtime,
        EnvironmentAsset("profile", spec.database_id),
        GENERATION_ID,
        build_profile=build_profile,
    )

    assert result["status"] == "published"
    assert observed_source is not None
    assert not observed_source.exists()
    assert not (
        runtime.source_root / ".setup" / f"{spec.source_filename}.zst.part"
    ).exists()


def test_failed_profile_preparation_retains_only_resumable_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, FakeClaims())
    spec = resolve_database_profile("small_bfd")

    def download(urls, **kwargs):
        del kwargs
        [path] = urls.values()
        Path(path).write_bytes(b"compressed")

    def decompress(command, *, check):
        assert check is True
        Path(command[-1]).write_bytes(b">sequence\nACDE\n")

    monkeypatch.setattr(environment_module, "download_files", download)
    monkeypatch.setattr(environment_module, "require_executable", lambda name: name)
    monkeypatch.setattr(environment_module.subprocess, "run", decompress)

    def fail_build(*_arguments):
        raise RuntimeError("profile build failed")

    with pytest.raises(RuntimeError, match="profile build failed"):
        prepare_environment_asset(
            runtime,
            EnvironmentAsset("profile", spec.database_id),
            GENERATION_ID,
            build_profile=fail_build,
        )

    assert not (runtime.scratch_root / GENERATION_ID).exists()
    assert (
        runtime.source_root / ".setup" / f"{spec.source_filename}.zst.part"
    ).is_file()
    status = generation_status(
        claims=runtime.claims, scope_key=spec.profile_id, generation_id=GENERATION_ID
    )
    assert status is not None and status["status"] == "failed"


def test_preempted_environment_worker_leaves_claim_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedPreemption(BaseException):
        pass

    claims = FakeClaims()
    runtime = _runtime(tmp_path, claims)
    spec = resolve_database_profile("small_bfd")
    monkeypatch.setattr(
        environment_module,
        "_prepare_profile_source",
        lambda *args: (_ for _ in ()).throw(SimulatedPreemption()),
    )

    with pytest.raises(SimulatedPreemption):
        prepare_environment_asset(
            runtime,
            EnvironmentAsset("profile", spec.database_id),
            GENERATION_ID,
            build_profile=lambda *args: {"status": "published"},
        )

    assert generation_status(claims, spec.profile_id, GENERATION_ID) is None
    assert (
        acquire_asset_claim(
            replace(runtime, container_id="replacement"),
            EnvironmentAsset("profile", spec.database_id),
            GENERATION_ID,
        )
        is not None
    )


def test_profile_cleanup_is_scoped_and_status_aware(tmp_path: Path) -> None:
    claims = FakeClaims()
    runtime = _runtime(tmp_path, claims)
    spec = resolve_database_profile("small_bfd")
    asset = EnvironmentAsset("profile", spec.database_id)
    terminal_generation = "b" * 64
    unknown_generation = "c" * 64
    current_generation = "d" * 64
    abandoned_generation = "e" * 64

    terminal_claim = acquire_asset_claim(runtime, asset, terminal_generation)
    assert terminal_claim is not None
    finish_generation_claim(
        claims,
        terminal_claim,
        status="failed",
        detail={},
    )
    assert acquire_asset_claim(runtime, asset, current_generation) is not None

    staging = runtime.sharded_root / ".staging"
    removable = (
        f"{spec.profile_id}-{terminal_generation}",
        f"{spec.profile_id}-{current_generation}",
    )
    for name in removable:
        (staging / name).mkdir(parents=True)
    retained = staging / f"{spec.profile_id}-{unknown_generation}"
    retained.mkdir(parents=True)
    abandoned = staging / f"{spec.profile_id}-{abandoned_generation}"
    abandoned.mkdir(parents=True)
    claims.values[f"status:{spec.profile_id}:{abandoned_generation}"] = {
        "status": "abandoned"
    }
    unrelated = staging / f"other-profile-{terminal_generation}"
    unrelated.mkdir(parents=True)
    orphan = runtime.sharded_root / ".orphaned" / f"{spec.profile_id}-legacy"
    orphan.mkdir(parents=True)

    environment_module._cleanup_profile_workspaces(
        runtime,
        spec,
        current_generation,
    )

    assert all(not (staging / name).exists() for name in removable)
    assert retained.is_dir()
    assert abandoned.is_dir()
    assert unrelated.is_dir()
    assert not orphan.exists()


def test_environment_generation_rejects_unsafe_path_component(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FakeClaims())

    with pytest.raises(ValueError, match="64 lowercase hex"):
        acquire_asset_claim(runtime, EnvironmentAsset("model"), "../generation")
