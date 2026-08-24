"""Automatic preparation of AlphaFold3 environment-scoped assets."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast

from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    LostGenerationError,
    acquire_generation_claim,
    finish_generation_claim,
)
from biomodals.app.fold.alphafold3.input_enrichment import chain_msa_states
from biomodals.app.fold.alphafold3.msa_search import plan_msa_resolution
from biomodals.app.fold.alphafold3.profiles import (
    BUILD_TIMEOUT_SECONDS,
    DATABASE_PROFILE_SPECS,
    SHARDED_DB_VOLUME_NAME,
    SOURCE_DB_VOLUME_NAME,
    DatabaseProfileSpec,
    profile_root,
    resolve_database_profile,
)
from biomodals.app.fold.alphafold3.sharding import require_executable
from biomodals.helper.constant import MODEL_VOLUME_NAME
from biomodals.helper.web import download_files

MODEL_URL = "https://storage.googleapis.com/alphafold3/af3.bin.zst"
DATABASE_BASE_URL = "https://storage.googleapis.com/alphafold-databases/v3.0"
MODEL_RELPATH = Path("AlphaFold3/af3.bin")
PDB_SEQRES_FILENAME = "pdb_seqres_2022_09_28.fasta"
MMCIF_DIRNAME = "mmcif_files"
MMCIF_ARCHIVE_FILENAME = "pdb_2022_09_28_mmcif_files.tar.zst"
ENVIRONMENT_SETUP_CLAIM_DICT_NAME = "AlphaFold3-environment-setup-claims"
ENVIRONMENT_SETUP_TIMEOUT_SECONDS = BUILD_TIMEOUT_SECONDS
ENVIRONMENT_SETUP_STALE_SECONDS = ENVIRONMENT_SETUP_TIMEOUT_SECONDS + 900

AssetKind = Literal["model", "profile", "template-seqres", "template-mmcif"]


class VolumeHandle(Protocol):
    """Mounted Modal Volume operations used by setup helpers."""

    def reload(self) -> None:
        """Refresh a mounted Volume's view."""
        ...

    def commit(self) -> None:
        """Publish mounted filesystem changes."""
        ...


@dataclass(frozen=True, slots=True)
class EnvironmentAsset:
    """One immutable shared asset required by an AlphaFold3 request."""

    key: str
    kind: AssetKind
    database_id: str | None = None

    def to_record(self) -> dict[str, object]:
        """Serialize the small provider-call payload."""
        return asdict(self)

    @classmethod
    def from_record(cls, value: object) -> EnvironmentAsset:
        """Validate a provider-call payload."""
        if not isinstance(value, Mapping):
            raise TypeError("AlphaFold3 environment asset must be a mapping")
        return cls(
            key=cast(str, value.get("key")),
            kind=cast(AssetKind, value.get("kind")),
            database_id=cast(str | None, value.get("database_id")),
        ).validated()

    def validated(self) -> EnvironmentAsset:
        """Reject inconsistent asset descriptors."""
        if not self.key or self.kind not in {
            "model",
            "profile",
            "template-seqres",
            "template-mmcif",
        }:
            raise ValueError("Invalid AlphaFold3 environment asset")
        if self.kind == "profile":
            spec = resolve_database_profile(cast(str, self.database_id))
            if self.key != f"profile:{spec.profile_id}":
                raise ValueError("Profile asset key does not match its database")
        elif self.database_id is not None:
            raise ValueError("Only profile assets may select a database")
        return self


@dataclass(frozen=True, slots=True)
class EnvironmentRuntime:
    """Mounted roots and narrow persistence handles for setup work."""

    MODEL_MOUNT: ClassVar[str] = f"/{MODEL_VOLUME_NAME}"
    SOURCE_MOUNT: ClassVar[str] = f"/{SOURCE_DB_VOLUME_NAME}"
    SHARDED_MOUNT: ClassVar[str] = f"/{SHARDED_DB_VOLUME_NAME}"

    model_volume: VolumeHandle
    source_volume: VolumeHandle
    sharded_volume: VolumeHandle
    claims: ClaimStore
    container_id: str
    model_root: Path = Path(MODEL_MOUNT)
    source_root: Path = Path(SOURCE_MOUNT)
    sharded_root: Path = Path(SHARDED_MOUNT)

    def volume_for(self, asset: EnvironmentAsset) -> VolumeHandle:
        """Return the Volume that publishes an asset's readiness path."""
        if asset.kind == "model":
            return self.model_volume
        if asset.kind == "profile":
            return self.sharded_volume
        return self.source_volume


def required_environment_assets(
    config: AF3Config,
    *,
    search_msa: bool,
    search_protein_templates: bool,
) -> tuple[EnvironmentAsset, ...]:
    """Plan only the shared assets required by this scientific request."""
    assets = [EnvironmentAsset("model", "model")]
    states = chain_msa_states(config)
    if search_msa:
        required_databases = {
            task.database_id for task in plan_msa_resolution(states).raw_searches
        }
        assets.extend(
            EnvironmentAsset(
                f"profile:{spec.profile_id}",
                "profile",
                spec.database_id,
            )
            for spec in DATABASE_PROFILE_SPECS
            if spec.database_id in required_databases
        )
    needs_templates = (
        search_msa
        and search_protein_templates
        and any(
            entry.protein is not None and not entry.protein.templates
            for entry in config.sequences
        )
    )
    if needs_templates:
        assets.extend((
            EnvironmentAsset("template:pdb-seqres", "template-seqres"),
            EnvironmentAsset("template:mmcif", "template-mmcif"),
        ))
    return tuple(assets)


def asset_path(runtime: EnvironmentRuntime, asset: EnvironmentAsset) -> Path:
    """Resolve one trusted descriptor to its final readiness path."""
    selected = asset.validated()
    if selected.kind == "model":
        return runtime.model_root / MODEL_RELPATH
    if selected.kind == "profile":
        spec = resolve_database_profile(cast(str, selected.database_id))
        return profile_root(runtime.sharded_root, spec) / "manifest.json"
    if selected.kind == "template-seqres":
        return runtime.source_root / PDB_SEQRES_FILENAME
    return runtime.source_root / MMCIF_DIRNAME


def asset_ready(runtime: EnvironmentRuntime, asset: EnvironmentAsset) -> bool:
    """Perform the intentionally cheap recurring readiness check."""
    path = asset_path(runtime, asset)
    return path.is_dir() if asset.kind == "template-mmcif" else path.is_file()


def claim_identity(asset: EnvironmentAsset) -> dict[str, object]:
    """Return the stable identity shared by coordinator and worker claims."""
    selected = asset.validated()
    if selected.kind == "profile":
        spec = resolve_database_profile(cast(str, selected.database_id))
        return {"profile_id": spec.profile_id, "database_id": spec.database_id}
    return {"asset_key": selected.key, "asset_kind": selected.kind}


def claim_scope(asset: EnvironmentAsset) -> str:
    """Return the append-only claim namespace used by an asset writer."""
    if asset.kind == "profile":
        return resolve_database_profile(cast(str, asset.database_id)).profile_id
    return asset.key


def acquire_asset_claim(
    runtime: EnvironmentRuntime,
    asset: EnvironmentAsset,
    generation_id: str,
) -> GenerationClaim | None:
    """Elect this Run, returning ``None`` while another writer is active."""
    try:
        return acquire_generation_claim(
            runtime.claims,
            scope_key=claim_scope(asset),
            generation_id=generation_id,
            identity=claim_identity(asset),
            container_id=runtime.container_id,
            maximum_age_seconds=ENVIRONMENT_SETUP_STALE_SECONDS,
        )
    except ActiveGenerationError:
        return None


def prepare_environment_asset(
    runtime: EnvironmentRuntime,
    asset: EnvironmentAsset,
    generation_id: str,
    *,
    build_profile: Callable[[str, str], dict[str, object]],
) -> dict[str, object]:
    """Acquire, materialize, and publish one requested environment asset."""
    selected = asset.validated()
    runtime.volume_for(selected).reload()
    if asset_ready(runtime, selected):
        try:
            claim = acquire_asset_claim(runtime, selected, generation_id)
        except LostGenerationError:
            claim = None
        if claim is not None:
            finish_generation_claim(
                runtime.claims,
                claim,
                status="complete",
                detail={"asset_key": selected.key, "result_status": "reused"},
            )
        return {"status": "reused", "asset_key": selected.key}
    claim = acquire_asset_claim(runtime, selected, generation_id)
    if claim is None:
        raise RuntimeError(f"Environment asset {selected.key!r} has an active writer")
    status = "failed"
    detail: dict[str, object] = {"asset_key": selected.key}
    result: dict[str, object]
    try:
        runtime.volume_for(selected).reload()
        if asset_ready(runtime, selected):
            result = {"status": "reused", "asset_key": selected.key}
        elif selected.kind == "model":
            _prepare_compressed_file(
                runtime.model_root,
                MODEL_RELPATH,
                MODEL_URL,
                generation_id,
                runtime.model_volume,
            )
            result = {"status": "published", "asset_key": selected.key}
        elif selected.kind == "profile":
            spec = resolve_database_profile(cast(str, selected.database_id))
            _prepare_profile_source(runtime, spec, generation_id)
            profile_result = build_profile(spec.database_id, generation_id)
            result = {
                "status": (
                    "reused"
                    if profile_result.get("status") == "reused"
                    else "published"
                ),
                "asset_key": selected.key,
                "profile": profile_result,
            }
        elif selected.kind == "template-seqres":
            _prepare_compressed_file(
                runtime.source_root,
                Path(PDB_SEQRES_FILENAME),
                f"{DATABASE_BASE_URL}/{PDB_SEQRES_FILENAME}.zst",
                generation_id,
                runtime.source_volume,
            )
            result = {"status": "published", "asset_key": selected.key}
        else:
            _prepare_mmcif(runtime, generation_id)
            result = {"status": "published", "asset_key": selected.key}
        status = "complete"
        detail = {"asset_key": selected.key, "result_status": result["status"]}
        return result
    except Exception as exc:
        detail = {
            "asset_key": selected.key,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        finish_generation_claim(
            runtime.claims,
            claim,
            status=status,
            detail=detail,
        )


def _prepare_profile_source(
    runtime: EnvironmentRuntime,
    spec: DatabaseProfileSpec,
    generation_id: str,
) -> None:
    runtime.source_volume.reload()
    _prepare_compressed_file(
        runtime.source_root,
        Path(spec.source_filename),
        f"{DATABASE_BASE_URL}/{spec.source_filename}.zst",
        generation_id,
        runtime.source_volume,
        commit=False,
    )


def _prepare_compressed_file(
    root: Path,
    relative_path: Path,
    url: str,
    generation_id: str,
    volume: VolumeHandle,
    *,
    commit: bool = True,
) -> None:
    final_path = root / relative_path
    if final_path.is_file():
        return
    setup_root = root / ".setup"
    partial = setup_root / f"{relative_path.name}.zst.part"
    staging = setup_root / generation_id / relative_path.name
    partial.parent.mkdir(parents=True, exist_ok=True)
    download_files({url: partial}, resume=True, progress_bar_desc=relative_path.name)
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # noqa: S603 - fixed executable and trusted paths
            [
                require_executable("zstd"),
                "--quiet",
                "--decompress",
                "--force",
                str(partial),
                "-o",
                str(staging),
            ],
            check=True,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
        raise
    if final_path.exists():
        raise RuntimeError(
            f"Refusing to replace existing AlphaFold3 asset: {final_path}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final_path)
    partial.unlink(missing_ok=True)
    if commit:
        volume.commit()


def _prepare_mmcif(runtime: EnvironmentRuntime, generation_id: str) -> None:
    final_path = runtime.source_root / MMCIF_DIRNAME
    if final_path.is_dir():
        return
    setup_root = runtime.source_root / ".setup"
    partial = setup_root / f"{MMCIF_ARCHIVE_FILENAME}.part"
    staging_parent = setup_root / generation_id
    extracted = staging_parent / MMCIF_DIRNAME
    partial.parent.mkdir(parents=True, exist_ok=True)
    download_files(
        {f"{DATABASE_BASE_URL}/{MMCIF_ARCHIVE_FILENAME}": partial},
        resume=True,
        progress_bar_desc=MMCIF_ARCHIVE_FILENAME,
    )
    if staging_parent.exists():
        shutil.rmtree(staging_parent)
    staging_parent.mkdir(parents=True)
    try:
        subprocess.run(  # noqa: S603 - fixed executable and trusted paths
            [
                require_executable("tar"),
                "--no-same-owner",
                "--no-same-permissions",
                "--use-compress-program=zstd",
                "-xf",
                str(partial),
                "--directory",
                str(staging_parent),
            ],
            check=True,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    if not extracted.is_dir():
        raise RuntimeError("AlphaFold3 mmCIF archive did not contain mmcif_files")
    if final_path.exists():
        raise RuntimeError(
            f"Refusing to replace existing AlphaFold3 asset: {final_path}"
        )
    extracted.replace(final_path)
    partial.unlink(missing_ok=True)
    shutil.rmtree(staging_parent, ignore_errors=True)
    runtime.source_volume.commit()
