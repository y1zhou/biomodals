"""Local AlphaFold 3 input materialization and inference identity.

This module owns the seed-independent run seam. It turns caller-local paths
into validated content, constructs the normalized inference identity, and
returns every byte that the app must stage in its output Volume.
"""

from __future__ import annotations

import hashlib
import re
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

from biomodals.app.fold.alphafold3.artifacts import (
    MAX_MSA_FIELD_BYTES,
    json_bytes,
    load_artifact_bytes,
    sha256_bytes,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
)

ALPHAFOLD3_APP_VERSION = "3.0.2"
DECLARED_MODEL_IDENTITY = "AlphaFold3/af3.bin:v1"
RUN_IDENTITY_SCHEMA = "biomodals-alphafold3-inference-run-v1"
STAGED_INPUT_SCHEMA_VERSION = 1
MAX_INPUT_JSON_BYTES = 64 * 1024 * 1024
MAX_LOCAL_MSA_BYTES = MAX_MSA_FIELD_BYTES
MAX_STAGED_INPUT_BYTES = 1024 * 1024 * 1024
MAX_CUSTOM_TEMPLATE_BYTES = 64 * 1024 * 1024
MAX_CUSTOM_TEMPLATE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_USER_CCD_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_ENTITIES = 5_120
MAX_TOTAL_POLYMER_RESIDUES = 5_120
MAX_MODEL_SEEDS = 1000
MAX_NUM_RECYCLES = 100
MAX_DIFFUSION_SAMPLES = 100
MAX_SEED_SAMPLE_PAIRS = 1000
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

    def to_record(self) -> dict[str, object]:
        """Describe the immutable upload for staging and remote loading."""
        if not self.content:
            raise ValueError(f"Volume upload must be nonempty: {self.relative_path}")
        return {
            "path": self.relative_path.as_posix(),
            "size_bytes": len(self.content),
            "sha256": sha256_bytes(self.content),
        }


@dataclass(frozen=True, slots=True)
class PreparedInferenceRun:
    """Seed-independent run identity and its complete staging payload."""

    run_id: str
    request_id: str
    run_root: PurePosixPath
    display_name: str
    submitted_seeds: tuple[int, ...]
    normalized_seeds: tuple[int, ...]
    recycle: int
    sample_count: int
    payload_uploads: tuple[VolumeUpload, ...]
    staged_input: VolumeUpload


@dataclass(frozen=True, slots=True)
class LoadedInferenceInput:
    """A staged request whose complete run identity has been re-derived."""

    config: AF3Config
    recycle: int
    sample_count: int


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
) -> int:
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
    return len(entity_ids)


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


def validate_model_seed(seed: int) -> int:
    """Return one model seed after enforcing the upstream uint32 contract."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Every model seed must be an integer")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Model seeds must be 32-bit unsigned integers, got {seed}")
    return seed


def _validated_model_seeds(seeds: list[int]) -> tuple[int, ...]:
    if not seeds:
        raise ValueError("modelSeeds must contain at least one seed")
    return tuple(validate_model_seed(seed) for seed in seeds)


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
    expanded_entities = 0
    total_polymer_residues = 0
    for entry in validated.sequences:
        entity_name, entity = _entry_entity(entry)
        copies = _validate_entity_ids(entity, seen_ids=seen_ids)
        expanded_entities += copies
        if expanded_entities > MAX_EXPANDED_ENTITIES:
            raise ValueError(
                "Input must contain no more than "
                f"{MAX_EXPANDED_ENTITIES:,} expanded entities"
            )
        if isinstance(entity, (AF3Protein, AF3RNA, AF3DNA)):
            _validate_polymer(entity_name, entity)
            total_polymer_residues += copies * len(entity.sequence)
            if total_polymer_residues > MAX_TOTAL_POLYMER_RESIDUES:
                raise ValueError(
                    "Input must contain no more than "
                    f"{MAX_TOTAL_POLYMER_RESIDUES:,} total polymer residues"
                )
    return validated


def validate_submitted_af3_input(config: AF3Config) -> AF3Config:
    """Validate one caller request, including its per-request seed ceiling."""
    validated = validate_upstream_af3_input(config)
    _validated_submitted_model_seeds(validated.modelSeeds)
    return validated


def serialize_af3_input(config: AF3Config) -> bytes:
    """Serialize one config in the strict upstream AlphaFold 3 JSON shape."""
    content = validate_upstream_af3_input(config).to_json(exclude_unset=False).encode()
    if len(content) > MAX_STAGED_INPUT_BYTES:
        raise ValueError(
            f"staged input exceeds the {MAX_STAGED_INPUT_BYTES}-byte limit"
        )
    return content


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


def _validate_custom_template_total(size_bytes: int) -> None:
    if size_bytes > MAX_CUSTOM_TEMPLATE_TOTAL_BYTES:
        raise ValueError(
            f"custom templates exceed the {MAX_CUSTOM_TEMPLATE_TOTAL_BYTES}-byte limit"
        )


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

    custom_templates: dict[Path, LocalTemplateFile] = {}
    custom_template_content: dict[str, bytes] = {}
    custom_template_bytes = 0
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
                if template_path not in custom_templates:
                    content = _read_bounded_bytes(
                        template_path,
                        field_name=(
                            f"sequences[{chain_index}].protein."
                            f"templates[{template_index}].mmcifPath"
                        ),
                        max_bytes=MAX_CUSTOM_TEMPLATE_BYTES,
                    )
                    digest = sha256_bytes(content)
                    canonical_content = custom_template_content.get(digest)
                    if canonical_content is None:
                        custom_template_bytes += len(content)
                        _validate_custom_template_total(custom_template_bytes)
                        custom_template_content[digest] = content
                    elif canonical_content != content:
                        raise RuntimeError(
                            f"Custom template digest collision: {digest}"
                        )
                    custom_templates[template_path] = LocalTemplateFile(
                        source_path=template_path,
                        content=custom_template_content[digest],
                        sha256=digest,
                    )
                template.mmcifPath = str(template_path)
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
        custom_templates=tuple(custom_templates.values()),
    )


def _template_files_by_source(
    custom_templates: tuple[LocalTemplateFile, ...],
) -> dict[str, LocalTemplateFile]:
    by_source: dict[str, LocalTemplateFile] = {}
    for artifact in custom_templates:
        if not artifact.content:
            raise ValueError(
                f"Custom template must be nonempty: {artifact.source_path}"
            )
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


def validate_inference_workload(seeds: list[int], sample_count: int) -> int:
    """Bound the number of durable seed/sample prediction directories."""
    validate_inference_parameters(0, sample_count)
    prediction_count = len(normalize_model_seeds(seeds)) * sample_count
    if prediction_count > MAX_SEED_SAMPLE_PAIRS:
        raise ValueError(
            "modelSeeds × sample must not exceed "
            f"{MAX_SEED_SAMPLE_PAIRS}, got {prediction_count}"
        )
    return prediction_count


def _run_identity(
    identity_view: dict[str, object],
    *,
    recycle: int,
    sample_count: int,
) -> tuple[str, dict[str, object]]:
    inference_parameters = {
        "num_recycles": recycle,
        "num_diffusion_samples": sample_count,
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
    return run_id, {
        "schema": RUN_IDENTITY_SCHEMA,
        "run_id": run_id,
        "input": identity_view,
        "inference": inference_parameters,
        "app": app_identity,
        "declared_model_identity": DECLARED_MODEL_IDENTITY,
    }


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
    validate_inference_workload(conf.modelSeeds, sample)
    submitted_seeds = tuple(conf.modelSeeds)
    normalized_seeds = normalize_model_seeds(conf.modelSeeds)
    display_name = conf.name
    identity_view = build_inference_identity_view(conf, custom_templates)

    run_id, identity_document = _run_identity(
        identity_view,
        recycle=recycle,
        sample_count=sample,
    )
    request_id = hash_sequences(run_id, list(normalized_seeds))
    run_root = PurePosixPath(run_id[:2]) / run_id

    template_files = _template_files_by_source(custom_templates)
    staged_conf = conf.model_copy(deep=True)
    staged_conf.name = f"af3-{run_id[:16]}"
    staged_conf.modelSeeds = list(normalized_seeds)
    uploads: dict[PurePosixPath, bytes] = {}
    template_paths: set[PurePosixPath] = set()
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
            template_paths.add(relative_path)
            template.mmcif = None
            template.mmcifPath = str(mount_root / Path(relative_path.as_posix()))

    _validate_custom_template_total(sum(len(uploads[path]) for path in template_paths))
    staged_conf = validate_upstream_af3_input(staged_conf)
    identity_path = run_root / "inputs" / "identity.json"
    input_path = run_root / "requests" / request_id / "input.json"
    input_bytes = serialize_af3_input(staged_conf)
    identity_bytes = json_bytes(identity_document)
    if len(identity_bytes) > MAX_STAGED_INPUT_BYTES:
        raise ValueError(
            f"run identity exceeds the {MAX_STAGED_INPUT_BYTES}-byte limit"
        )
    uploads[identity_path] = identity_bytes
    uploads[input_path] = input_bytes
    payload_uploads = tuple(
        VolumeUpload(relative_path=relative_path, content=content)
        for relative_path, content in sorted(
            uploads.items(),
            key=lambda item: item[0].as_posix(),
        )
    )
    uploads_by_path = {upload.relative_path: upload for upload in payload_uploads}
    staged_input = VolumeUpload(
        relative_path=run_root / "requests" / request_id / "staged-input.json",
        content=json_bytes({
            "schema_version": STAGED_INPUT_SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id,
            "request_id": request_id,
            "identity": uploads_by_path[identity_path].to_record(),
            "input": uploads_by_path[input_path].to_record(),
            "custom_templates": [
                uploads_by_path[path].to_record() for path in sorted(template_paths)
            ],
        }),
    )

    return PreparedInferenceRun(
        run_id=run_id,
        request_id=request_id,
        run_root=run_root,
        display_name=display_name,
        submitted_seeds=submitted_seeds,
        normalized_seeds=normalized_seeds,
        recycle=recycle,
        sample_count=sample,
        payload_uploads=payload_uploads,
        staged_input=staged_input,
    )


def _validate_digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _load_staged_artifact(
    output_root: Path,
    record: object,
    expected_path: PurePosixPath,
    *,
    max_bytes: int | None = None,
) -> bytes:
    if max_bytes is not None:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid staged artifact record: {expected_path}")
        size_bytes = record.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes > max_bytes
        ):
            raise ValueError(f"Staged artifact is too large: {expected_path}")
    value = load_artifact_bytes(output_root, record, expected_path.as_posix())
    if value is None:
        raise RuntimeError(f"Staged artifact failed validation: {expected_path}")
    return value


def _json_object(value: bytes, *, field_name: str) -> dict[str, object]:
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    return cast(dict[str, object], parsed)


def _staged_template_files(
    config: AF3Config,
    raw_records: object,
    *,
    output_root: Path,
    run_root: PurePosixPath,
) -> tuple[LocalTemplateFile, ...]:
    if not isinstance(raw_records, list):
        raise ValueError("Staged custom_templates must be a list")
    template_root = run_root / "custom-templates"
    files_by_path: dict[PurePosixPath, LocalTemplateFile] = {}
    custom_template_bytes = 0
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("Invalid staged custom-template record")
        raw_path = raw_record.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("Invalid staged custom-template path")
        relative_path = PurePosixPath(raw_path)
        if (
            relative_path.parent != template_root
            or re.fullmatch(r"[0-9a-f]{64}\.cif", relative_path.name) is None
        ):
            raise ValueError(
                f"Staged custom template escapes its run directory: {raw_path}"
            )
        digest = relative_path.stem
        if raw_record.get("sha256") != digest or relative_path in files_by_path:
            raise ValueError(f"Invalid staged custom-template identity: {raw_path}")
        content = _load_staged_artifact(
            output_root,
            raw_record,
            relative_path,
            max_bytes=MAX_CUSTOM_TEMPLATE_BYTES,
        )
        custom_template_bytes += len(content)
        _validate_custom_template_total(custom_template_bytes)
        files_by_path[relative_path] = LocalTemplateFile(
            source_path=output_root / Path(relative_path.as_posix()),
            content=content,
            sha256=digest,
        )

    referenced_paths: set[PurePosixPath] = set()
    for entry in config.sequences:
        if (protein := entry.protein) is None:
            continue
        for template in protein.templates:
            if template.mmcifPath is None:
                continue
            if template.mmcif is not None:
                raise ValueError("Staged template sets both mmcif and mmcifPath")
            path = Path(template.mmcifPath)
            if not path.is_absolute():
                raise ValueError("Staged template path must be absolute")
            try:
                relative_path = PurePosixPath(path.relative_to(output_root).as_posix())
            except ValueError as exc:
                raise ValueError(
                    f"Staged template path escapes the output Volume: {path}"
                ) from exc
            expected_path = output_root / Path(relative_path.as_posix())
            if path != expected_path or relative_path not in files_by_path:
                raise ValueError(f"Staged template path is not marker-bound: {path}")
            referenced_paths.add(relative_path)
    if referenced_paths != set(files_by_path):
        raise ValueError("Staged custom-template records do not match the input")
    return tuple(files_by_path[path] for path in sorted(files_by_path))


def load_staged_inference_input(
    output_mount_root: Path,
    *,
    run_id: str,
    request_id: str,
    staged_input_record: object,
) -> LoadedInferenceInput:
    """Load and re-derive one marker-bound inference request from a Volume."""
    validated_run_id = _validate_digest(run_id, field_name="run_id")
    validated_request_id = _validate_digest(request_id, field_name="request_id")
    output_root = Path(output_mount_root)
    if not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("output_mount_root must be an absolute directory")

    run_root = PurePosixPath(validated_run_id[:2]) / validated_run_id
    marker_path = run_root / "requests" / validated_request_id / "staged-input.json"
    marker = _json_object(
        _load_staged_artifact(
            output_root,
            staged_input_record,
            marker_path,
            max_bytes=MAX_INPUT_JSON_BYTES,
        ),
        field_name="Staged input marker",
    )
    if (
        marker.get("schema_version") != STAGED_INPUT_SCHEMA_VERSION
        or marker.get("status") != "complete"
        or marker.get("run_id") != validated_run_id
        or marker.get("request_id") != validated_request_id
    ):
        raise ValueError("Staged input marker identity is invalid")

    identity_path = run_root / "inputs" / "identity.json"
    input_path = run_root / "requests" / validated_request_id / "input.json"
    identity_document = _json_object(
        _load_staged_artifact(
            output_root,
            marker.get("identity"),
            identity_path,
            max_bytes=MAX_STAGED_INPUT_BYTES,
        ),
        field_name="Run identity document",
    )
    input_bytes = _load_staged_artifact(
        output_root,
        marker.get("input"),
        input_path,
        max_bytes=MAX_STAGED_INPUT_BYTES,
    )
    config = validate_upstream_af3_input(AF3Config.model_validate_json(input_bytes))
    if config.name != f"af3-{validated_run_id[:16]}":
        raise ValueError("Staged input canonical name does not match run_id")
    normalized_seeds = normalize_model_seeds(config.modelSeeds)
    if tuple(config.modelSeeds) != normalized_seeds:
        raise ValueError("Staged input modelSeeds must be sorted and unique")
    if hash_sequences(validated_run_id, list(normalized_seeds)) != validated_request_id:
        raise ValueError("Staged input request_id does not match its modelSeeds")

    template_files = _staged_template_files(
        config,
        marker.get("custom_templates"),
        output_root=output_root,
        run_root=run_root,
    )
    raw_inference = identity_document.get("inference")
    if not isinstance(raw_inference, dict):
        raise ValueError("Run identity inference parameters are invalid")
    recycle = raw_inference.get("num_recycles")
    sample_count = raw_inference.get("num_diffusion_samples")
    if (
        isinstance(recycle, bool)
        or not isinstance(recycle, int)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
    ):
        raise ValueError("Run identity inference parameters are invalid")
    validate_inference_parameters(recycle, sample_count)

    identity_view = build_inference_identity_view(config, template_files)
    expected_run_id, expected_document = _run_identity(
        identity_view,
        recycle=recycle,
        sample_count=sample_count,
    )
    if expected_run_id != validated_run_id or identity_document != expected_document:
        raise ValueError("Staged input does not match its run identity")
    return LoadedInferenceInput(
        config=config,
        recycle=recycle,
        sample_count=sample_count,
    )
