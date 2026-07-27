"""Local AlphaFold 3 input materialization and inference identity.

This module owns the seed-independent run seam. It turns caller-local paths
into validated content, constructs the normalized inference identity, and
returns every byte that the app must stage in its output Volume.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import orjson
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.artifacts import json_bytes, sha256_bytes
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
)

ALPHAFOLD3_APP_VERSION = "3.0.2"
DECLARED_MODEL_IDENTITY = "AlphaFold3/af3.bin:v1"
RUN_IDENTITY_SCHEMA = "biomodals-alphafold3-inference-run-v1"


@dataclass(frozen=True, slots=True)
class LocalTemplateFile:
    """Caller template bytes captured before any remote work."""

    source_path: Path
    content: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class MaterializedLocalInput:
    """Validated input with local MSA/CCD paths replaced by inline content."""

    config: AF3Config
    custom_templates: tuple[LocalTemplateFile, ...]


@dataclass(frozen=True, slots=True)
class VolumeUpload:
    """One output-Volume-relative file prepared for batch upload."""

    relative_path: PurePosixPath
    content: bytes


@dataclass(frozen=True, slots=True)
class PreparedInferenceRun:
    """Seed-independent run identity and its complete staging payload."""

    run_id: str
    request_id: str
    run_root: PurePosixPath
    display_name: str
    submitted_seeds: tuple[int, ...]
    normalized_seeds: tuple[int, ...]
    worker_config: AF3Config
    uploads: tuple[VolumeUpload, ...]


def validate_af3_config(config: AF3Config) -> AF3Config:
    """Return a fully validated copy with explicit default field semantics."""
    return AF3Config.model_validate(
        config.model_dump(mode="python", exclude_unset=False)
    )


def serialize_af3_input(config: AF3Config) -> bytes:
    """Serialize one config in the strict upstream AlphaFold 3 JSON shape."""
    return validate_af3_config(config).to_json(exclude_unset=False).encode()


def hash_sequences(*fragments: object) -> str:
    """Hash ordered canonical-JSON fragments with unambiguous framing."""
    if not fragments:
        raise ValueError("hash_sequences requires at least one fragment")
    digest = hashlib.sha256()
    for fragment in fragments:
        encoded = orjson.dumps(fragment, option=orjson.OPT_SORT_KEYS)
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _resolve_regular_file(
    input_root: Path,
    value: str,
    *,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = input_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} is not a regular file: {path}")
    return path


def _materialize_text_pair(
    owner: object,
    *,
    inline_field: str,
    path_field: str,
    input_root: Path,
    field_name: str,
) -> None:
    inline_value = getattr(owner, inline_field)
    path_value = getattr(owner, path_field)
    if inline_value is not None and path_value is not None:
        raise ValueError(
            f"{field_name} cannot set both {inline_field} and {path_field}"
        )
    if path_value is None:
        return
    path = _resolve_regular_file(
        input_root,
        path_value,
        field_name=f"{field_name}.{path_field}",
    )
    setattr(owner, inline_field, path.read_text())
    setattr(owner, path_field, None)


def materialize_local_input(config_path: str | Path) -> MaterializedLocalInput:
    """Resolve every caller-local path needed before remote work."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON is not a regular file: {path}")
    input_root = path.parent
    conf = AF3Config.from_file(path)

    custom_templates: list[LocalTemplateFile] = []
    for chain_index, entry in enumerate(conf.sequences):
        if (protein := entry.protein) is not None:
            _materialize_text_pair(
                protein,
                inline_field="unpairedMsa",
                path_field="unpairedMsaPath",
                input_root=input_root,
                field_name=f"sequences[{chain_index}].protein",
            )
            _materialize_text_pair(
                protein,
                inline_field="pairedMsa",
                path_field="pairedMsaPath",
                input_root=input_root,
                field_name=f"sequences[{chain_index}].protein",
            )
            for template_index, template in enumerate(protein.templates):
                if template.mmcifPath is None:
                    continue
                template_path = _resolve_regular_file(
                    input_root,
                    template.mmcifPath,
                    field_name=(
                        f"sequences[{chain_index}].protein."
                        f"templates[{template_index}].mmcifPath"
                    ),
                )
                content = template_path.read_bytes()
                template.mmcifPath = str(template_path)
                custom_templates.append(
                    LocalTemplateFile(
                        source_path=template_path,
                        content=content,
                        sha256=sha256_bytes(content),
                    )
                )
        elif (rna := entry.rna) is not None:
            _materialize_text_pair(
                rna,
                inline_field="unpairedMsa",
                path_field="unpairedMsaPath",
                input_root=input_root,
                field_name=f"sequences[{chain_index}].rna",
            )

    if conf.userCCD is not None and conf.userCCDPath is not None:
        raise ValueError("userCCD and userCCDPath cannot both be set")
    if conf.userCCDPath is not None:
        ccd_path = _resolve_regular_file(
            input_root,
            conf.userCCDPath,
            field_name="userCCDPath",
        )
        conf.userCCD = ccd_path.read_text()
        conf.userCCDPath = None

    return MaterializedLocalInput(
        config=validate_af3_config(conf),
        custom_templates=tuple(custom_templates),
    )


def _template_files_by_source(
    custom_templates: tuple[LocalTemplateFile, ...],
) -> dict[str, LocalTemplateFile]:
    by_source: dict[str, LocalTemplateFile] = {}
    for artifact in custom_templates:
        expected_digest = sha256_bytes(artifact.content)
        if artifact.sha256 != expected_digest:
            raise ValueError(f"Custom template digest mismatch: {artifact.source_path}")
        source = str(artifact.source_path.resolve())
        existing = by_source.get(source)
        if existing is not None and existing.content != artifact.content:
            raise RuntimeError(f"Custom template source changed: {source}")
        by_source[source] = artifact
    return by_source


def _template_content(
    template: dict[str, object],
    template_files: dict[str, LocalTemplateFile],
) -> bytes:
    inline_value = template.get("mmcif")
    path_value = template.get("mmcifPath")
    if (inline_value is None) == (path_value is None):
        raise ValueError("Exactly one template mmCIF form must be populated")
    if inline_value is not None:
        if not isinstance(inline_value, str):
            raise TypeError("Inline template mmCIF must be a string")
        return inline_value.encode()
    if not isinstance(path_value, str):
        raise TypeError("Template mmcifPath must be a string")
    artifact = template_files.get(str(Path(path_value).resolve()))
    if artifact is None:
        raise ValueError(f"Template path was not materialized locally: {path_value}")
    return artifact.content


def build_inference_identity_view(
    conf: AF3Config,
    custom_templates: tuple[LocalTemplateFile, ...],
) -> dict[str, object]:
    """Return the explicit-default, seed/name-neutral biological identity."""
    validated = validate_af3_config(conf)
    raw_view = validated.model_dump(
        mode="json",
        exclude_unset=False,
        exclude_defaults=False,
        exclude_none=False,
    )
    view = cast(dict[str, object], raw_view)
    view.pop("name")
    view.pop("modelSeeds")

    template_files = _template_files_by_source(custom_templates)
    raw_sequences = view.get("sequences")
    if not isinstance(raw_sequences, list):
        raise RuntimeError("Validated AlphaFold input has no sequence list")
    for raw_entry in raw_sequences:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("Validated AlphaFold sequence entry is invalid")
        entry_view = cast(dict[str, object], raw_entry)
        raw_protein = entry_view.get("protein")
        if raw_protein is None:
            continue
        if not isinstance(raw_protein, dict):
            raise RuntimeError("Validated AlphaFold protein entry is invalid")
        protein_view = cast(dict[str, object], raw_protein)
        raw_templates = protein_view.get("templates")
        if not isinstance(raw_templates, list):
            raise RuntimeError("Validated AlphaFold template list is invalid")
        identity_templates: list[dict[str, object]] = []
        for raw_template in raw_templates:
            if not isinstance(raw_template, dict):
                raise RuntimeError("Validated AlphaFold template is invalid")
            template_view = cast(dict[str, object], raw_template)
            identity_templates.append({
                "mmcifSha256": sha256_bytes(
                    _template_content(template_view, template_files)
                ),
                "queryIndices": template_view.get("queryIndices"),
                "templateIndices": template_view.get("templateIndices"),
            })
        protein_view["templates"] = identity_templates
    return view


def normalize_model_seeds(seeds: list[int]) -> tuple[int, ...]:
    """Return the required non-empty sorted unique model-seed set."""
    if not seeds:
        raise ValueError("modelSeeds must contain at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("Every model seed must be an integer")
    return tuple(sorted(set(seeds)))


def validate_inference_parameters(recycle: int, sample: int) -> None:
    """Validate inference counts before any cost-incurring remote work."""
    if isinstance(recycle, bool) or not isinstance(recycle, int) or recycle < 0:
        raise ValueError("recycle must be a non-negative integer")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 1:
        raise ValueError("sample must be a positive integer")


def prepare_inference_run(
    enriched_config: AF3Config,
    custom_templates: tuple[LocalTemplateFile, ...],
    *,
    output_mount_root: Path,
    recycle: int,
    sample: int,
) -> PreparedInferenceRun:
    """Build run/request identities and every required Volume upload."""
    validate_inference_parameters(recycle, sample)
    mount_root = Path(output_mount_root)
    if not mount_root.is_absolute():
        raise ValueError("output_mount_root must be absolute")
    conf = validate_af3_config(enriched_config)
    submitted_seeds = tuple(conf.modelSeeds)
    normalized_seeds = normalize_model_seeds(conf.modelSeeds)
    display_name = conf.name
    identity_view = build_inference_identity_view(conf, custom_templates)

    inference_parameters = {
        "num_recycles": recycle,
        "num_diffusion_samples": sample,
    }
    app_identity = {
        "app_name": "AlphaFold3",
        "app_version": ALPHAFOLD3_APP_VERSION,
        "alphafold_repository": ALPHAFOLD3_REPOSITORY,
        "alphafold_commit": ALPHAFOLD3_COMMIT,
    }
    run_id = hash_sequences(
        identity_view,
        inference_parameters,
        app_identity,
        DECLARED_MODEL_IDENTITY,
        RUN_IDENTITY_SCHEMA,
    )
    request_id = hash_sequences(run_id, list(normalized_seeds))
    run_root = PurePosixPath(run_id[:2]) / run_id

    identity_document = {
        "schema": RUN_IDENTITY_SCHEMA,
        "run_id": run_id,
        "input": identity_view,
        "inference": inference_parameters,
        "app": app_identity,
        "declared_model_identity": DECLARED_MODEL_IDENTITY,
    }

    template_files = _template_files_by_source(custom_templates)
    staged_conf = conf.model_copy(deep=True)
    staged_conf.name = f"af3-{run_id[:16]}"
    staged_conf.modelSeeds = list(normalized_seeds)
    uploads: dict[PurePosixPath, bytes] = {}
    for entry in staged_conf.sequences:
        if (protein := entry.protein) is None:
            continue
        for template in protein.templates:
            template_view = cast(
                dict[str, object],
                template.model_dump(mode="python", exclude_unset=False),
            )
            content = _template_content(template_view, template_files)
            digest = sha256_bytes(content)
            relative_path = run_root / "custom-templates" / f"{digest}.cif"
            existing = uploads.get(relative_path)
            if existing is not None and existing != content:
                raise RuntimeError(f"Custom template digest collision: {digest}")
            uploads[relative_path] = content
            template.mmcif = None
            template.mmcifPath = str(mount_root / Path(relative_path.as_posix()))

    staged_conf = validate_af3_config(staged_conf)
    uploads[run_root / "inputs" / "identity.json"] = json_bytes(identity_document)
    uploads[run_root / "requests" / request_id / "input.json"] = serialize_af3_input(
        staged_conf
    )

    return PreparedInferenceRun(
        run_id=run_id,
        request_id=request_id,
        run_root=run_root,
        display_name=display_name,
        submitted_seeds=submitted_seeds,
        normalized_seeds=normalized_seeds,
        worker_config=staged_conf,
        uploads=tuple(
            VolumeUpload(relative_path=relative_path, content=content)
            for relative_path, content in sorted(
                uploads.items(),
                key=lambda item: item[0].as_posix(),
            )
        ),
    )
