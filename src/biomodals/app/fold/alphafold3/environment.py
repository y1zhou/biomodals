"""Automatic preparation of AlphaFold3 environment-scoped assets."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.generation_claims import (
    ActiveGenerationError,
    ClaimStore,
    GenerationClaim,
    LostGenerationError,
    acquire_generation_claim,
    finish_generation_claim,
    generation_status,
    latest_generation_owner,
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
from biomodals.app.fold.alphafold3.template_search import (
    MMCIF_DIRECTORY_NAME,
    PDB_SEQRES_FILENAME,
)
from biomodals.helper.artifacts import VolumeHandle
from biomodals.helper.constant import MODEL_VOLUME_NAME
from biomodals.helper.web import download_files

MODEL_URL = "https://storage.googleapis.com/alphafold3/af3.bin.zst"
DATABASE_BASE_URL = "https://storage.googleapis.com/alphafold-databases/v3.0"
MODEL_RELPATH = Path("AlphaFold3/af3.bin")
MMCIF_ARCHIVE_FILENAME = "pdb_2022_09_28_mmcif_files.tar.zst"
ENVIRONMENT_SETUP_CLAIM_DICT_NAME = "AlphaFold3-environment-setup-claims"
ENVIRONMENT_SETUP_TIMEOUT_SECONDS = BUILD_TIMEOUT_SECONDS
ENVIRONMENT_SETUP_STALE_SECONDS = ENVIRONMENT_SETUP_TIMEOUT_SECONDS + 900

AssetKind = Literal["model", "profile", "template-seqres", "template-mmcif"]


@dataclass(frozen=True, slots=True)
class EnvironmentAsset:
    """One immutable shared asset required by an AlphaFold3 request."""

    kind: AssetKind
    database_id: str | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent asset descriptors at construction."""
        if self.kind not in {
            "model",
            "profile",
            "template-seqres",
            "template-mmcif",
        }:
            raise ValueError("Invalid AlphaFold3 environment asset")
        if self.kind == "profile":
            resolve_database_profile(cast(str, self.database_id))
        elif self.database_id is not None:
            raise ValueError("Only profile assets may select a database")

    @property
    def key(self) -> str:
        """Return the stable scheduler and claim key for this asset."""
        if self.kind == "profile":
            spec = resolve_database_profile(cast(str, self.database_id))
            return f"profile:{spec.profile_id}"
        if self.kind == "template-seqres":
            return "template:pdb-seqres"
        if self.kind == "template-mmcif":
            return "template:mmcif"
        return "model"

    def to_record(self) -> dict[str, object]:
        """Serialize the small provider-call payload."""
        return asdict(self)

    @classmethod
    def from_record(cls, value: object) -> EnvironmentAsset:
        """Validate a provider-call payload."""
        if not isinstance(value, Mapping):
            raise TypeError("AlphaFold3 environment asset must be a mapping")
        return cls(
            kind=cast(AssetKind, value.get("kind")),
            database_id=cast(str | None, value.get("database_id")),
        )


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
    scratch_root: Path = Path("/tmp/biomodals-alphafold3")  # noqa: S108

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
    assets = [EnvironmentAsset("model")]
    states = chain_msa_states(config)
    if search_msa:
        required_databases = {
            task.database_id for task in plan_msa_resolution(states).raw_searches
        }
        assets.extend(
            EnvironmentAsset(
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
            EnvironmentAsset("template-seqres"),
            EnvironmentAsset("template-mmcif"),
        ))
    return tuple(assets)


def asset_path(runtime: EnvironmentRuntime, asset: EnvironmentAsset) -> Path:
    """Resolve one trusted descriptor to its final readiness path."""
    selected = asset
    if selected.kind == "model":
        return runtime.model_root / MODEL_RELPATH
    if selected.kind == "profile":
        spec = resolve_database_profile(cast(str, selected.database_id))
        return profile_root(runtime.sharded_root, spec) / "manifest.json"
    if selected.kind == "template-seqres":
        return runtime.source_root / PDB_SEQRES_FILENAME
    return runtime.source_root / MMCIF_DIRECTORY_NAME


def asset_ready(runtime: EnvironmentRuntime, asset: EnvironmentAsset) -> bool:
    """Perform the intentionally cheap recurring readiness check."""
    path = asset_path(runtime, asset)
    return path.is_dir() if asset.kind == "template-mmcif" else path.is_file()


def claim_identity(asset: EnvironmentAsset) -> dict[str, object]:
    """Return the stable identity shared by coordinator and worker claims."""
    selected = asset
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
    _validate_environment_generation_id(generation_id)
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


def fail_asset_claim_if_current(
    runtime: EnvironmentRuntime,
    asset: EnvironmentAsset,
    generation_id: str,
    *,
    detail: dict[str, object],
) -> bool:
    """Fail this generation when it still owns an unfinished setup claim."""
    _validate_environment_generation_id(generation_id)
    scope = claim_scope(asset)
    owner = latest_generation_owner(runtime.claims, scope)
    if owner is None or owner.get("generation_id") != generation_id:
        return False
    if generation_status(runtime.claims, scope, generation_id) is not None:
        return False
    finish_generation_claim(
        runtime.claims,
        GenerationClaim(scope, generation_id, owner),
        status="failed",
        detail=detail,
    )
    return True


def prepare_environment_asset(
    runtime: EnvironmentRuntime,
    asset: EnvironmentAsset,
    generation_id: str,
    *,
    build_profile: Callable[[str, str, Path], dict[str, object]],
) -> dict[str, object]:
    """Acquire, materialize, and publish one requested environment asset."""
    _validate_environment_generation_id(generation_id)
    selected = asset
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
            _cleanup_profile_workspaces(runtime, spec, generation_id)
            source_path, partial_path = _prepare_profile_source(
                runtime,
                spec,
                generation_id,
            )
            try:
                profile_result = build_profile(
                    spec.database_id,
                    generation_id,
                    source_path,
                )
                partial_path.unlink(missing_ok=True)
            finally:
                shutil.rmtree(source_path.parent, ignore_errors=True)
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
) -> tuple[Path, Path]:
    """Download a durable archive and decompress into container-local scratch."""
    _validate_environment_generation_id(generation_id)
    runtime.source_volume.reload()
    partial = runtime.source_root / ".setup" / f"{spec.source_filename}.zst.part"
    source_path = runtime.scratch_root / generation_id / spec.source_filename
    _download_and_decompress(
        f"{DATABASE_BASE_URL}/{spec.source_filename}.zst",
        partial,
        source_path,
        progress_name=spec.source_filename,
    )
    return source_path, partial


def _download_and_decompress(
    url: str,
    partial: Path,
    destination: Path,
    *,
    progress_name: str,
) -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    download_files({url: partial}, resume=True, progress_bar_desc=progress_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # noqa: S603 - fixed executable and trusted paths
            [
                require_executable("zstd"),
                "--quiet",
                "--decompress",
                "--force",
                str(partial),
                "-o",
                str(destination),
            ],
            check=True,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _prepare_compressed_file(
    root: Path,
    relative_path: Path,
    url: str,
    generation_id: str,
    volume: VolumeHandle,
) -> None:
    _validate_environment_generation_id(generation_id)
    final_path = root / relative_path
    if final_path.is_file():
        return
    setup_root = root / ".setup"
    partial = setup_root / f"{relative_path.name}.zst.part"
    staging = setup_root / generation_id / relative_path.name
    _download_and_decompress(
        url,
        partial,
        staging,
        progress_name=relative_path.name,
    )
    if final_path.exists():
        raise RuntimeError(
            f"Refusing to replace existing AlphaFold3 asset: {final_path}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final_path)
    partial.unlink(missing_ok=True)
    volume.commit()


def _prepare_mmcif(runtime: EnvironmentRuntime, generation_id: str) -> None:
    _validate_environment_generation_id(generation_id)
    final_path = runtime.source_root / MMCIF_DIRECTORY_NAME
    if final_path.is_dir():
        return
    setup_root = runtime.source_root / ".setup"
    partial = setup_root / f"{MMCIF_ARCHIVE_FILENAME}.part"
    staging_parent = setup_root / generation_id
    extracted = staging_parent / MMCIF_DIRECTORY_NAME
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


def _cleanup_profile_workspaces(
    runtime: EnvironmentRuntime,
    spec: DatabaseProfileSpec,
    generation_id: str,
) -> None:
    """Remove only abandoned workspaces for the currently claimed profile."""
    staging_root = runtime.sharded_root / ".staging"
    prefix = f"{spec.profile_id}-"
    if staging_root.is_dir():
        for candidate in staging_root.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            candidate_generation = candidate.name.removeprefix(prefix)
            status = generation_status(
                runtime.claims,
                spec.profile_id,
                candidate_generation,
            )
            if candidate_generation == generation_id or (
                status is not None
                and status.get("status") in {"complete", "failed", "abandoned"}
            ):
                _remove_workspace(candidate)

    orphan_root = runtime.sharded_root / ".orphaned"
    if orphan_root.is_dir():
        for candidate in orphan_root.iterdir():
            if candidate.name.startswith(prefix):
                _remove_workspace(candidate)


def _remove_workspace(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Refusing to remove invalid setup workspace: {path}")
    shutil.rmtree(path)


def _validate_environment_generation_id(generation_id: str) -> str:
    """Validate the hash-shaped identifier used in setup filesystem paths."""
    if (
        not isinstance(generation_id, str)
        or len(generation_id) != 64
        or any(character not in "0123456789abcdef" for character in generation_id)
    ):
        raise ValueError("environment generation_id must be 64 lowercase hex digits")
    return generation_id
