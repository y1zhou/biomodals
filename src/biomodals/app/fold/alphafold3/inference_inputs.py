"""Local AlphaFold 3 input materialization and inference identity.

This module owns the seed-independent run seam. It turns caller-local paths
into validated content, constructs the normalized inference identity, and
returns every byte that the app must stage in its output Volume.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import orjson
from uniaf3.schema.alphafold3 import (
    AF3DNA,
    AF3RNA,
    AF3Config,
    AF3Ligand,
    AF3Protein,
    AF3SequenceEntry,
)

from biomodals.app.fold.alphafold3.artifacts import json_bytes, sha256_bytes
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
)

ALPHAFOLD3_APP_VERSION = "3.0.2"
DECLARED_MODEL_IDENTITY = "AlphaFold3/af3.bin:v1"
RUN_IDENTITY_SCHEMA = "biomodals-alphafold3-inference-run-v1"
MAX_INPUT_JSON_BYTES = 64 * 1024 * 1024
MAX_LOCAL_MSA_BYTES = 512 * 1024 * 1024
MAX_CUSTOM_TEMPLATE_BYTES = 64 * 1024 * 1024
MAX_USER_CCD_BYTES = 64 * 1024 * 1024
MAX_MODEL_SEEDS = 1000
MAX_NUM_RECYCLES = 100
MAX_DIFFUSION_SAMPLES = 100
MAX_INFERENCE_WORKERS = 100
MAX_PROTEIN_TEMPLATES = 20
_TEXT_SIZE_CHUNK_CHARS = 1024 * 1024

type _AF3Entity = AF3Protein | AF3RNA | AF3DNA | AF3Ligand


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


def sanitize_af3_name(name: str) -> str:
    """Mirror upstream ``Input.sanitised_name`` and require a safe component."""
    if not isinstance(name, str):
        raise TypeError("AlphaFold input name must be a string")
    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.")
    sanitized = "".join(char for char in spaceless_name if char in allowed_chars)
    if not name.strip() or not sanitized:
        raise ValueError(
            "AlphaFold input name must contain a letter, number, dot, dash, "
            "or underscore"
        )
    if sanitized in {".", ".."}:
        raise ValueError(
            "AlphaFold input name cannot resolve to a relative path component"
        )
    return sanitized


def _field_bytes(value: str, *, field_name: str, max_bytes: int) -> None:
    size_bytes = 0
    for start in range(0, len(value), _TEXT_SIZE_CHUNK_CHARS):
        size_bytes += len(value[start : start + _TEXT_SIZE_CHUNK_CHARS].encode())
        if size_bytes > max_bytes:
            raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")


def _entry_entity(entry: AF3SequenceEntry) -> tuple[str, _AF3Entity]:
    if entry.protein is not None:
        return "Protein", entry.protein
    if entry.rna is not None:
        return "RNA", entry.rna
    if entry.dna is not None:
        return "DNA", entry.dna
    if entry.ligand is not None:
        return "Ligand", entry.ligand
    raise RuntimeError("Validated AlphaFold sequence entry has no entity")


def _validate_entity_ids(
    entity: _AF3Entity,
    *,
    seen_ids: set[str],
) -> None:
    entity_ids = [entity.id] if isinstance(entity.id, str) else entity.id
    if not entity_ids or any(not entity_id for entity_id in entity_ids):
        raise ValueError("Input JSON contains sequences with unset IDs")
    invalid_ids = [
        entity_id
        for entity_id in entity_ids
        if not entity_id.isalpha() or entity_id.islower()
    ]
    if invalid_ids:
        raise ValueError(f"IDs must be upper case letters, got: {invalid_ids}")
    duplicate_ids: set[str] = set()
    entry_ids: set[str] = set()
    for entity_id in entity_ids:
        if entity_id in seen_ids or entity_id in entry_ids:
            duplicate_ids.add(entity_id)
        entry_ids.add(entity_id)
    if duplicate_ids:
        raise ValueError(
            f"Input JSON contains sequences with duplicate IDs: {sorted(duplicate_ids)}"
        )
    seen_ids.update(entry_ids)


def _validate_polymer(entity_name: str, entity: AF3Protein | AF3RNA | AF3DNA) -> None:
    if not entity.sequence or not all(residue.isalpha() for residue in entity.sequence):
        raise ValueError(
            f"{entity_name} must contain only letters, got {entity.sequence!r}"
        )
    if isinstance(entity, AF3Protein):
        modifications = entity.modifications or []
        modification_codes = [modification.ptmType for modification in modifications]
    else:
        modifications = entity.modifications or []
        modification_codes = [
            modification.modificationType for modification in modifications
        ]
    if any(code.startswith("CCD_") for code in modification_codes):
        raise ValueError(
            f"{entity_name} modifications must not contain the 'CCD_' prefix"
        )


def _validate_inline_inputs(config: AF3Config) -> None:
    for chain_index, entry in enumerate(config.sequences):
        if (protein := entry.protein) is not None:
            for field_name in ("unpairedMsa", "pairedMsa"):
                value = getattr(protein, field_name)
                if value is not None:
                    _field_bytes(
                        value,
                        field_name=f"sequences[{chain_index}].protein.{field_name}",
                        max_bytes=MAX_LOCAL_MSA_BYTES,
                    )
            if len(protein.templates) > MAX_PROTEIN_TEMPLATES:
                raise ValueError(
                    f"sequences[{chain_index}].protein.templates exceeds "
                    f"AlphaFold 3's {MAX_PROTEIN_TEMPLATES}-template limit"
                )
            for template_index, template in enumerate(protein.templates):
                if template.mmcif is not None:
                    _field_bytes(
                        template.mmcif,
                        field_name=(
                            f"sequences[{chain_index}].protein."
                            f"templates[{template_index}].mmcif"
                        ),
                        max_bytes=MAX_CUSTOM_TEMPLATE_BYTES,
                    )
        elif (rna := entry.rna) is not None and rna.unpairedMsa is not None:
            _field_bytes(
                rna.unpairedMsa,
                field_name=f"sequences[{chain_index}].rna.unpairedMsa",
                max_bytes=MAX_LOCAL_MSA_BYTES,
            )
    if config.userCCD is not None:
        _field_bytes(
            config.userCCD,
            field_name="userCCD",
            max_bytes=MAX_USER_CCD_BYTES,
        )


def _validated_model_seeds(seeds: list[int]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("modelSeeds must contain at least one seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("Every model seed must be an integer")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError(f"Model seeds must be 32-bit unsigned integers, got {seeds}")
    return tuple(seeds)


def _validated_submitted_model_seeds(seeds: list[int]) -> tuple[int, ...]:
    validated = _validated_model_seeds(seeds)
    if len(validated) > MAX_MODEL_SEEDS:
        raise ValueError(
            f"modelSeeds must contain no more than {MAX_MODEL_SEEDS} seeds"
        )
    return validated


def validate_upstream_af3_input(config: AF3Config) -> AF3Config:
    """Mirror cheap upstream input checks before any remote work."""
    validated = AF3Config.model_validate(
        config.model_dump(mode="python", exclude_unset=False)
    )
    sanitize_af3_name(validated.name)
    _validated_model_seeds(validated.modelSeeds)
    if not validated.sequences:
        raise ValueError("AlphaFold input must contain at least one sequence")

    seen_ids: set[str] = set()
    for entry in validated.sequences:
        entity_name, entity = _entry_entity(entry)
        _validate_entity_ids(entity, seen_ids=seen_ids)
        if isinstance(entity, (AF3Protein, AF3RNA, AF3DNA)):
            _validate_polymer(entity_name, entity)
    return validated


def validate_submitted_af3_input(config: AF3Config) -> AF3Config:
    """Validate one caller request, including its per-request seed ceiling."""
    validated = validate_upstream_af3_input(config)
    _validated_submitted_model_seeds(validated.modelSeeds)
    return validated


def serialize_af3_input(config: AF3Config) -> bytes:
    """Serialize one config in the strict upstream AlphaFold 3 JSON shape."""
    return validate_upstream_af3_input(config).to_json(exclude_unset=False).encode()


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
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{field_name} is not a regular file: {path}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{field_name} is not a regular file: {resolved}")
    return resolved


def _read_bounded_bytes(
    path: Path,
    *,
    field_name: str,
    max_bytes: int,
) -> bytes:
    stat_before = path.stat()
    if stat_before.st_size > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
    with path.open("rb") as handle:
        value = handle.read(max_bytes + 1)
    if len(value) > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
    stat_after = path.stat()
    before_identity = (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    )
    after_identity = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    if before_identity != after_identity or len(value) != stat_after.st_size:
        raise RuntimeError(f"{field_name} changed while it was being read: {path}")
    return value


def _read_bounded_text(
    path: Path,
    *,
    field_name: str,
    max_bytes: int,
) -> str:
    value = _read_bounded_bytes(
        path,
        field_name=field_name,
        max_bytes=max_bytes,
    )
    try:
        return value.decode()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be UTF-8 text: {path}") from exc


def _materialize_text_pair(
    owner: object,
    *,
    inline_field: str,
    path_field: str,
    input_root: Path,
    field_name: str,
    max_bytes: int,
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
    setattr(
        owner,
        inline_field,
        _read_bounded_text(
            path,
            field_name=f"{field_name}.{path_field}",
            max_bytes=max_bytes,
        ),
    )
    setattr(owner, path_field, None)


def materialize_local_input(config_path: str | Path) -> MaterializedLocalInput:
    """Resolve every caller-local path needed before remote work."""
    path = _resolve_regular_file(
        Path.cwd(),
        str(config_path),
        field_name="Input JSON",
    )
    input_root = path.parent
    conf = AF3Config.model_validate_json(
        _read_bounded_bytes(
            path,
            field_name="Input JSON",
            max_bytes=MAX_INPUT_JSON_BYTES,
        )
    )

    custom_templates: list[LocalTemplateFile] = []
    for chain_index, entry in enumerate(conf.sequences):
        if (protein := entry.protein) is not None:
            _materialize_text_pair(
                protein,
                inline_field="unpairedMsa",
                path_field="unpairedMsaPath",
                input_root=input_root,
                field_name=f"sequences[{chain_index}].protein",
                max_bytes=MAX_LOCAL_MSA_BYTES,
            )
            _materialize_text_pair(
                protein,
                inline_field="pairedMsa",
                path_field="pairedMsaPath",
                input_root=input_root,
                field_name=f"sequences[{chain_index}].protein",
                max_bytes=MAX_LOCAL_MSA_BYTES,
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
                content = _read_bounded_bytes(
                    template_path,
                    field_name=(
                        f"sequences[{chain_index}].protein."
                        f"templates[{template_index}].mmcifPath"
                    ),
                    max_bytes=MAX_CUSTOM_TEMPLATE_BYTES,
                )
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
                max_bytes=MAX_LOCAL_MSA_BYTES,
            )

    if conf.userCCD is not None and conf.userCCDPath is not None:
        raise ValueError("userCCD and userCCDPath cannot both be set")
    if conf.userCCDPath is not None:
        ccd_path = _resolve_regular_file(
            input_root,
            conf.userCCDPath,
            field_name="userCCDPath",
        )
        conf.userCCD = _read_bounded_text(
            ccd_path,
            field_name="userCCDPath",
            max_bytes=MAX_USER_CCD_BYTES,
        )
        conf.userCCDPath = None

    validated = validate_submitted_af3_input(conf)
    _validate_inline_inputs(validated)
    return MaterializedLocalInput(
        config=validated,
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
    validated = validate_upstream_af3_input(conf)
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
    return tuple(sorted(set(_validated_submitted_model_seeds(seeds))))


def validate_inference_parameters(recycle: int, sample: int) -> None:
    """Validate inference counts before any cost-incurring remote work."""
    if (
        isinstance(recycle, bool)
        or not isinstance(recycle, int)
        or not 0 <= recycle <= MAX_NUM_RECYCLES
    ):
        raise ValueError(f"recycle must be an integer between 0 and {MAX_NUM_RECYCLES}")
    if (
        isinstance(sample, bool)
        or not isinstance(sample, int)
        or not 1 <= sample <= MAX_DIFFUSION_SAMPLES
    ):
        raise ValueError(
            f"sample must be an integer between 1 and {MAX_DIFFUSION_SAMPLES}"
        )


def validate_inference_worker_budget(max_num_gpus: int) -> int:
    """Validate the GPU-worker cap before any cost-incurring remote work."""
    if (
        isinstance(max_num_gpus, bool)
        or not isinstance(max_num_gpus, int)
        or not 1 <= max_num_gpus <= MAX_INFERENCE_WORKERS
    ):
        raise ValueError(
            f"max_num_gpus must be an integer between 1 and {MAX_INFERENCE_WORKERS}"
        )
    return max_num_gpus


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
    conf = validate_submitted_af3_input(enriched_config)
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

    staged_conf = validate_upstream_af3_input(staged_conf)
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
