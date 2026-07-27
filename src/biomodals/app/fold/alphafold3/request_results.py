"""Request-scoped AlphaFold 3 result views and local archives.

Durable prediction files remain canonical and seed-addressed on the output
Volume. This module publishes a small request view over exactly the requested
seeds, then downloads only that manifest-declared view and restores the
caller's presentation name in the local archive.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess as sp
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import IO, cast

import orjson

from biomodals.app.fold.alphafold3.artifacts import (
    VolumeReader,
    json_bytes,
    require_regular_file,
    sha256_file,
    utc_now,
    write_bytes_atomic,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    hash_sequences,
    sanitize_af3_name,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    CORE_OUTPUT_SUFFIXES,
    InferenceRuntime,
    canonical_output_name,
    copy_best_outputs,
    inference_run_root,
    load_seed_marker,
    load_summary_entry,
    ranked_rows,
    validate_run_id,
    write_ranking_table,
)
from biomodals.helper.shell import run_command

REQUEST_MANIFEST_SCHEMA_VERSION = 2

_CUSTOM_TEMPLATE_PATTERN = re.compile(r"(?P<digest>[0-9a-f]{64})\.cif")
_ARCHIVE_MANIFEST_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RequestPublication:
    """One successful request's immutable seed selection and observed outcome."""

    run_id: str
    request_id: str
    submitted_seeds: tuple[int, ...]
    normalized_seeds: tuple[int, ...]
    sample_count: int
    display_name: str
    reused_seeds: tuple[int, ...]
    published_seeds: tuple[int, ...]


def request_manifest_from_result(value: object) -> dict[str, object]:
    """Extract a request manifest from the inference coordinator result."""
    if not isinstance(value, dict) or not isinstance(
        request := value.get("request"),
        dict,
    ):
        raise RuntimeError(
            f"AlphaFold3 request publication returned invalid metadata: {value!r}"
        )
    return cast(dict[str, object], request)


def _validate_seed_tuple(
    value: tuple[int, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        raise TypeError(f"{field_name} must contain only integers")
    return value


def _validate_publication(spec: RequestPublication) -> RequestPublication:
    run_id = validate_run_id(spec.run_id)
    if (
        not isinstance(spec.request_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec.request_id) is None
    ):
        raise ValueError("request_id must be a lowercase SHA-256 digest")
    submitted = _validate_seed_tuple(
        spec.submitted_seeds,
        field_name="submitted_seeds",
        allow_empty=False,
    )
    normalized = _validate_seed_tuple(
        spec.normalized_seeds,
        field_name="normalized_seeds",
        allow_empty=False,
    )
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("normalized_seeds must be sorted and unique")
    if tuple(sorted(set(submitted))) != normalized:
        raise ValueError("submitted_seeds do not normalize to normalized_seeds")
    if spec.request_id != hash_sequences(run_id, list(normalized)):
        raise ValueError("request_id does not match run_id and normalized_seeds")
    if (
        isinstance(spec.sample_count, bool)
        or not isinstance(spec.sample_count, int)
        or spec.sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer")
    sanitize_af3_name(spec.display_name)
    reused = _validate_seed_tuple(
        spec.reused_seeds,
        field_name="reused_seeds",
        allow_empty=True,
    )
    published = _validate_seed_tuple(
        spec.published_seeds,
        field_name="published_seeds",
        allow_empty=True,
    )
    if len(set(reused)) != len(reused) or len(set(published)) != len(published):
        raise ValueError("Request outcome seed sets must be unique")
    if set(reused).intersection(published):
        raise ValueError("A seed cannot be both reused and newly published")
    if set(reused).union(published) != set(normalized):
        raise ValueError("Request outcome does not cover every normalized seed")
    return spec


def _duplicates_removed(submitted: tuple[int, ...]) -> list[int]:
    seen: set[int] = set()
    removed: list[int] = []
    for seed in submitted:
        if seed in seen:
            removed.append(seed)
        else:
            seen.add(seed)
    return removed


def _safe_archive_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe request archive path: {value!s}")
    return path


def _volume_relative_path(output_root: Path, path: Path) -> PurePosixPath:
    try:
        relative = path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Request artifact is outside the output Volume: {path}"
        ) from exc
    cursor = output_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Request artifact traverses a symlink: {path}")
    return _safe_archive_path(PurePosixPath(relative.as_posix()))


def _artifact_record(
    *,
    source: Path,
    output_root: Path,
    volume_path: Path,
    archive_path: str | PurePosixPath,
    role: str,
    worker_path: str | None = None,
) -> dict[str, object]:
    require_regular_file(source)
    record: dict[str, object] = {
        "role": role,
        "volume_path": _volume_relative_path(
            output_root,
            volume_path,
        ).as_posix(),
        "archive_path": _safe_archive_path(archive_path).as_posix(),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if worker_path is not None:
        record["worker_path"] = worker_path
    return record


def _request_input_path(run_root: Path, request_id: str) -> Path:
    return run_root / "requests" / request_id / "input.json"


def _custom_template_sources(
    *,
    input_path: Path,
    output_root: Path,
    run_root: Path,
) -> dict[str, Path]:
    require_regular_file(input_path)
    try:
        document = orjson.loads(input_path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid enriched request input: {input_path}") from exc
    if not isinstance(document, dict):
        raise TypeError("Enriched request input must be a JSON object")
    raw_sequences = document.get("sequences")
    if not isinstance(raw_sequences, list):
        raise TypeError("Enriched request input must contain a sequence list")

    selected: dict[str, Path] = {}
    expected_root = run_root / "custom-templates"
    for raw_entry in raw_sequences:
        if not isinstance(raw_entry, dict):
            raise TypeError("Invalid enriched request sequence entry")
        raw_protein = raw_entry.get("protein")
        if raw_protein is None:
            continue
        if not isinstance(raw_protein, dict):
            raise TypeError("Invalid enriched request protein entry")
        raw_templates = raw_protein.get("templates")
        if not isinstance(raw_templates, list):
            raise TypeError("Invalid enriched request template list")
        for raw_template in raw_templates:
            if not isinstance(raw_template, dict):
                raise TypeError("Invalid enriched request template entry")
            worker_path = raw_template.get("mmcifPath")
            if worker_path is None:
                continue
            if not isinstance(worker_path, str) or not worker_path:
                raise TypeError("Custom template mmcifPath must be a non-empty string")
            source = Path(worker_path)
            if not source.is_absolute():
                raise ValueError("Staged custom template path must be absolute")
            try:
                source.relative_to(expected_root)
            except ValueError as exc:
                raise ValueError(
                    f"Custom template is outside this run: {worker_path}"
                ) from exc
            match = _CUSTOM_TEMPLATE_PATTERN.fullmatch(source.name)
            if match is None or source.parent != expected_root:
                raise ValueError(f"Invalid staged custom template path: {worker_path}")
            require_regular_file(source)
            if sha256_file(source) != match.group("digest"):
                raise ValueError(f"Custom template digest mismatch: {worker_path}")
            _volume_relative_path(output_root, source)
            selected[worker_path] = source
    return selected


def _seed_artifacts(
    *,
    run_root: Path,
    output_root: Path,
    canonical_name: str,
    seeds: tuple[int, ...],
    sample_count: int,
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    outputs_root = run_root / "outputs"
    for seed in seeds:
        for sample_index in range(sample_count):
            directory_name = f"seed-{seed}_sample-{sample_index}"
            sample_root = outputs_root / directory_name
            if sample_root.is_symlink() or not sample_root.is_dir():
                raise FileNotFoundError(
                    f"Expected request sample directory: {sample_root}"
                )
            prefix = f"{canonical_name}_seed-{seed}_sample-{sample_index}"
            for suffix in CORE_OUTPUT_SUFFIXES:
                source = sample_root / f"{prefix}_{suffix}"
                artifacts.append(
                    _artifact_record(
                        source=source,
                        output_root=output_root,
                        volume_path=source,
                        archive_path=PurePosixPath(directory_name) / source.name,
                        role=f"seed_{suffix.removesuffix('.json').replace('.', '_')}",
                    )
                )
        for optional_role in ("embeddings", "distogram"):
            directory_name = f"seed-{seed}_{optional_role}"
            optional_root = outputs_root / directory_name
            if not optional_root.exists():
                continue
            if optional_root.is_symlink() or not optional_root.is_dir():
                raise ValueError(f"Invalid optional seed output: {optional_root}")
            source = optional_root / f"{canonical_name}_seed-{seed}_{optional_role}.npz"
            artifacts.append(
                _artifact_record(
                    source=source,
                    output_root=output_root,
                    volume_path=source,
                    archive_path=PurePosixPath(directory_name) / source.name,
                    role=f"seed_{optional_role}",
                )
            )
    return artifacts


def _replace_staged_file(source: Path, destination: Path) -> None:
    require_regular_file(source)
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise ValueError(f"Invalid request-view destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _record_publication_failure(
    runtime: InferenceRuntime,
    *,
    request_root: Path,
    run_id: str,
    request_id: str,
    generation_id: str,
    error: Exception,
) -> None:
    """Commit compact request failure evidence before staging cleanup."""
    failure_path = request_root / "failures" / f"{generation_id}.json"
    message = str(error)
    try:
        write_json_atomic(
            failure_path,
            {
                "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
                "status": "failed",
                "failed_at": utc_now(),
                "run_id": run_id,
                "request_id": request_id,
                "generation_id": generation_id,
                "error_type": type(error).__name__,
                "message": message[:4096],
                "message_truncated": len(message) > 4096,
            },
        )
        runtime.volume.commit()
    except Exception as evidence_error:
        error.add_note(
            "Failed to persist request publication evidence: "
            f"{type(evidence_error).__name__}: {evidence_error}"
        )


def publish_request_results(
    runtime: InferenceRuntime,
    publication: RequestPublication,
) -> dict[str, object]:
    """Publish a manifest-last request view over exactly the requested seeds."""
    spec = _validate_publication(publication)
    runtime.volume.reload()
    run_root = inference_run_root(runtime.output_root, spec.run_id)
    request_root = run_root / "requests" / spec.request_id
    input_path = _request_input_path(run_root, spec.request_id)
    require_regular_file(input_path)

    markers = []
    for seed in spec.normalized_seeds:
        marker = load_seed_marker(
            run_root,
            spec.run_id,
            seed,
            sample_count=spec.sample_count,
        )
        if marker is None:
            raise RuntimeError(f"Requested seed has no completion marker: {seed}")
        markers.append(marker)
    rows = ranked_rows(tuple(markers))
    if not rows:
        raise RuntimeError("Requested seed markers contain no rankings")
    best = rows[0]

    summary = load_summary_entry(run_root, spec.run_id)
    if summary is None:
        raise RuntimeError("The accumulated inference summary is not complete")
    if not set(spec.normalized_seeds).issubset(summary.included_seeds):
        raise RuntimeError("The accumulated summary does not cover this request")
    summary_marker_path = run_root / ".markers" / "summary.json"
    require_regular_file(summary_marker_path)

    canonical_name = canonical_output_name(spec.run_id)
    outputs_root = run_root / "outputs"
    generation_id = uuid.uuid4().hex
    staging_root = request_root / ".workers" / generation_id
    staging_root.mkdir(parents=True, exist_ok=False)

    generated_names = (
        f"{canonical_name}_ranking_scores.csv",
        *(f"{canonical_name}_{suffix}" for suffix in CORE_OUTPUT_SUFFIXES),
        "TERMS_OF_USE.md",
        f"global-summary-{summary.marker_sha256[:16]}.json",
    )
    try:
        observed_summary = (
            staging_root / f"global-summary-{summary.marker_sha256[:16]}.json"
        )
        shutil.copy2(summary_marker_path, observed_summary)
        if sha256_file(observed_summary) != summary.marker_sha256:
            raise RuntimeError("The observed summary marker changed during publication")
        write_ranking_table(
            staging_root / f"{canonical_name}_ranking_scores.csv",
            rows,
        )
        copy_best_outputs(
            staging_root,
            outputs_root,
            canonical_name,
            best,
        )
        require_regular_file(outputs_root / "TERMS_OF_USE.md")
        shutil.copy2(
            outputs_root / "TERMS_OF_USE.md",
            staging_root / "TERMS_OF_USE.md",
        )
        for name in generated_names:
            _replace_staged_file(staging_root / name, request_root / name)
        runtime.volume.commit()
    except Exception as exc:
        _record_publication_failure(
            runtime,
            request_root=request_root,
            run_id=spec.run_id,
            request_id=spec.request_id,
            generation_id=generation_id,
            error=exc,
        )
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    artifacts = [
        _artifact_record(
            source=input_path,
            output_root=runtime.output_root,
            volume_path=input_path,
            archive_path=f"{canonical_name}_data.json",
            role="input",
        ),
        _artifact_record(
            source=request_root / f"{canonical_name}_ranking_scores.csv",
            output_root=runtime.output_root,
            volume_path=request_root / f"{canonical_name}_ranking_scores.csv",
            archive_path=f"{canonical_name}_ranking_scores.csv",
            role="request_ranking",
        ),
    ]
    for suffix in CORE_OUTPUT_SUFFIXES:
        source = request_root / f"{canonical_name}_{suffix}"
        artifacts.append(
            _artifact_record(
                source=source,
                output_root=runtime.output_root,
                volume_path=source,
                archive_path=source.name,
                role=f"request_best_{suffix.removesuffix('.json').replace('.', '_')}",
            )
        )
    terms_path = request_root / "TERMS_OF_USE.md"
    artifacts.append(
        _artifact_record(
            source=terms_path,
            output_root=runtime.output_root,
            volume_path=terms_path,
            archive_path="TERMS_OF_USE.md",
            role="terms",
        )
    )
    observed_summary_path = (
        request_root / f"global-summary-{summary.marker_sha256[:16]}.json"
    )
    artifacts.append(
        _artifact_record(
            source=observed_summary_path,
            output_root=runtime.output_root,
            volume_path=observed_summary_path,
            archive_path="global_summary.json",
            role="observed_global_summary",
        )
    )
    artifacts.extend(
        _seed_artifacts(
            run_root=run_root,
            output_root=runtime.output_root,
            canonical_name=canonical_name,
            seeds=spec.normalized_seeds,
            sample_count=spec.sample_count,
        )
    )
    for worker_path, source in sorted(
        _custom_template_sources(
            input_path=input_path,
            output_root=runtime.output_root,
            run_root=run_root,
        ).items()
    ):
        artifacts.append(
            _artifact_record(
                source=source,
                output_root=runtime.output_root,
                volume_path=source,
                archive_path=PurePosixPath("custom-templates") / source.name,
                role="custom_template",
                worker_path=worker_path,
            )
        )
    artifacts.sort(
        key=lambda artifact: (
            cast(str, artifact["archive_path"]),
            cast(str, artifact["volume_path"]),
        )
    )
    archive_paths = [cast(str, artifact["archive_path"]) for artifact in artifacts]
    if len(set(archive_paths)) != len(archive_paths):
        raise RuntimeError("Request artifacts contain duplicate archive paths")

    manifest: dict[str, object] = {
        "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "published_at": utc_now(),
        "run_id": spec.run_id,
        "request_id": spec.request_id,
        "canonical_name": canonical_name,
        "submitted_display_name": spec.display_name,
        "presentation_name": sanitize_af3_name(spec.display_name),
        "name_mapping": {
            "canonical": canonical_name,
            "presentation": sanitize_af3_name(spec.display_name),
        },
        "submitted_seeds": list(spec.submitted_seeds),
        "normalized_seeds": list(spec.normalized_seeds),
        "duplicates_removed": _duplicates_removed(spec.submitted_seeds),
        "reused_seeds": list(spec.reused_seeds),
        "published_seeds": list(spec.published_seeds),
        "ranking": [row.to_dict() for row in rows],
        "best": best.to_dict(),
        "global_summary": {
            "marker_sha256": summary.marker_sha256,
            "included_seeds": list(summary.included_seeds),
            "best": summary.best.to_dict(),
        },
        "artifacts": artifacts,
        "manifest_volume_path": _volume_relative_path(
            runtime.output_root,
            request_root / "manifest.json",
        ).as_posix(),
    }
    write_json_atomic(request_root / "manifest.json", manifest)
    runtime.volume.commit()
    return manifest


def _validated_manifest_artifacts(
    manifest: dict[str, object],
) -> tuple[str, str, list[dict[str, object]]]:
    if (
        manifest.get("schema_version") != REQUEST_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Request manifest is not a supported complete publication")
    run_id = manifest.get("run_id")
    request_id = manifest.get("request_id")
    canonical_name = manifest.get("canonical_name")
    if (
        not isinstance(run_id, str)
        or not isinstance(request_id, str)
        or not isinstance(canonical_name, str)
        or canonical_name != canonical_output_name(run_id)
    ):
        raise ValueError("Request manifest identity is invalid")
    raw_normalized_seeds = manifest.get("normalized_seeds")
    if (
        not isinstance(raw_normalized_seeds, list)
        or not raw_normalized_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in raw_normalized_seeds
        )
        or raw_normalized_seeds != sorted(set(raw_normalized_seeds))
        or request_id != hash_sequences(run_id, raw_normalized_seeds)
    ):
        raise ValueError("Request manifest seed identity is invalid")
    expected_run_root = PurePosixPath(run_id[:2]) / run_id
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("Request manifest contains no artifacts")
    artifacts: list[dict[str, object]] = []
    archive_paths: set[str] = set()
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise TypeError("Request manifest artifact must be a dictionary")
        role = raw_artifact.get("role")
        volume_path = raw_artifact.get("volume_path")
        archive_path = raw_artifact.get("archive_path")
        size_bytes = raw_artifact.get("size_bytes")
        digest = raw_artifact.get("sha256")
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(volume_path, str)
            or not isinstance(archive_path, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"Invalid request artifact: {raw_artifact!r}")
        safe_volume_path = _safe_archive_path(volume_path)
        if not safe_volume_path.is_relative_to(expected_run_root):
            raise ValueError(f"Request artifact is outside its run root: {volume_path}")
        safe_archive_path = _safe_archive_path(archive_path).as_posix()
        if safe_archive_path in archive_paths:
            raise ValueError(f"Duplicate request archive path: {safe_archive_path}")
        archive_paths.add(safe_archive_path)
        artifacts.append(cast(dict[str, object], raw_artifact))
    return request_id, canonical_name, artifacts


def _presentation_archive_path(
    path: str,
    *,
    canonical_name: str,
    presentation_name: str,
) -> PurePosixPath:
    selected = _safe_archive_path(path)
    name = selected.name
    if name.startswith(canonical_name):
        name = presentation_name + name.removeprefix(canonical_name)
    return selected.parent / name


def _download_artifact(
    reader: VolumeReader,
    artifact: dict[str, object],
    destination: Path,
) -> None:
    volume_path = cast(str, artifact["volume_path"])
    expected_size = cast(int, artifact["size_bytes"])
    expected_sha256 = cast(str, artifact["sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("xb") as handle:
        for chunk in reader.read_file(volume_path):
            if not isinstance(chunk, bytes):
                raise TypeError(f"Volume reader returned non-bytes for {volume_path}")
            handle.write(chunk)
            written += len(chunk)
    if written != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {volume_path}: {written} != {expected_size}"
        )
    observed_sha256 = sha256_file(destination)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Downloaded SHA-256 mismatch for "
            f"{volume_path}: {observed_sha256} != {expected_sha256}"
        )


def _rewrite_downloaded_input(
    input_path: Path,
    *,
    display_name: str,
    custom_template_paths: dict[str, str],
) -> None:
    try:
        document = orjson.loads(input_path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Downloaded input is invalid JSON: {input_path}") from exc
    if not isinstance(document, dict):
        raise TypeError("Downloaded AlphaFold input must be a JSON object")
    document["name"] = display_name
    raw_sequences = document.get("sequences")
    if not isinstance(raw_sequences, list):
        raise TypeError("Downloaded AlphaFold input has no sequence list")
    for raw_entry in raw_sequences:
        if not isinstance(raw_entry, dict):
            raise TypeError("Downloaded AlphaFold sequence entry is invalid")
        raw_protein = raw_entry.get("protein")
        if raw_protein is None:
            continue
        if not isinstance(raw_protein, dict):
            raise TypeError("Downloaded AlphaFold protein entry is invalid")
        raw_templates = raw_protein.get("templates")
        if not isinstance(raw_templates, list):
            raise TypeError("Downloaded AlphaFold template list is invalid")
        for raw_template in raw_templates:
            if not isinstance(raw_template, dict):
                raise TypeError("Downloaded AlphaFold template entry is invalid")
            worker_path = raw_template.get("mmcifPath")
            if worker_path is None:
                continue
            if not isinstance(worker_path, str):
                raise TypeError("Downloaded custom template path is invalid")
            archive_path = custom_template_paths.get(worker_path)
            if archive_path is None:
                raise ValueError(
                    f"Downloaded input references undeclared template: {worker_path}"
                )
            raw_template["mmcifPath"] = archive_path
    write_bytes_atomic(input_path, json_bytes(document))


def _local_request_manifest(
    manifest: dict[str, object],
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]],
    *,
    display_name: str,
    canonical_name: str,
    presentation_name: str,
) -> dict[str, object]:
    """Build the presentation-local manifest embedded in the archive."""
    local_manifest = cast(
        dict[str, object],
        orjson.loads(orjson.dumps(manifest)),
    )
    local_manifest["submitted_display_name"] = display_name
    local_manifest["presentation_name"] = presentation_name
    local_manifest["name_mapping"] = {
        "canonical": canonical_name,
        "presentation": presentation_name,
    }
    local_artifacts = cast(list[dict[str, object]], local_manifest["artifacts"])
    for local_artifact, (_, transformed) in zip(
        local_artifacts,
        transformed_artifacts,
        strict=True,
    ):
        local_artifact["archive_path"] = transformed.as_posix()
        local_artifact.pop("worker_path", None)
    return local_manifest


def _record_archive_artifacts(
    local_manifest: dict[str, object],
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]],
    archive_root: Path,
) -> None:
    """Bind the presentation-local bytes, including the rewritten input."""
    local_artifacts = cast(list[dict[str, object]], local_manifest["artifacts"])
    for local_artifact, (_, transformed) in zip(
        local_artifacts,
        transformed_artifacts,
        strict=True,
    ):
        archived_path = archive_root / Path(transformed.as_posix())
        require_regular_file(archived_path)
        local_artifact["archive_size_bytes"] = archived_path.stat().st_size
        local_artifact["archive_sha256"] = sha256_file(archived_path)


def _custom_template_archive_paths(
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]],
) -> dict[str, str]:
    custom_template_paths: dict[str, str] = {}
    for artifact, transformed in transformed_artifacts:
        if artifact["role"] != "custom_template":
            continue
        worker_path = artifact.get("worker_path")
        if not isinstance(worker_path, str) or not worker_path:
            raise ValueError("Custom template artifact has no staged worker path")
        custom_template_paths[worker_path] = transformed.as_posix()
    return custom_template_paths


def _bind_expected_archive_artifacts(
    reader: VolumeReader,
    local_manifest: dict[str, object],
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]],
    *,
    display_name: str,
) -> None:
    """Derive presentation-local digests before reusing an existing archive."""
    local_artifacts = cast(list[dict[str, object]], local_manifest["artifacts"])
    input_pairs: list[tuple[dict[str, object], dict[str, object], PurePosixPath]] = []
    for local_artifact, (artifact, transformed) in zip(
        local_artifacts,
        transformed_artifacts,
        strict=True,
    ):
        if artifact["role"] == "input":
            input_pairs.append((local_artifact, artifact, transformed))
            continue
        local_artifact["archive_size_bytes"] = artifact["size_bytes"]
        local_artifact["archive_sha256"] = artifact["sha256"]
    if len(input_pairs) != 1:
        raise RuntimeError("Request archive requires exactly one input artifact")

    local_input, input_artifact, transformed_input = input_pairs[0]
    with TemporaryDirectory(prefix="alphafold3_archive_identity_") as directory:
        input_path = Path(directory) / transformed_input.name
        _download_artifact(reader, input_artifact, input_path)
        _rewrite_downloaded_input(
            input_path,
            display_name=display_name,
            custom_template_paths=_custom_template_archive_paths(transformed_artifacts),
        )
        local_input["archive_size_bytes"] = input_path.stat().st_size
        local_input["archive_sha256"] = sha256_file(input_path)


def _expected_archive_members(
    presentation_name: str,
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]],
) -> set[str]:
    """Return the exact files and parent directories generated by GNU tar."""
    relative_files = {
        PurePosixPath("request_manifest.json"),
        *(transformed for _, transformed in transformed_artifacts),
    }
    members = {f"{presentation_name}/"}
    for relative in relative_files:
        members.add(f"{presentation_name}/{relative.as_posix()}")
        for parent in relative.parents:
            if parent != PurePosixPath("."):
                members.add(f"{presentation_name}/{parent.as_posix()}/")
    return members


@dataclass(frozen=True, slots=True)
class _ArchiveInspection:
    members: frozenset[str]
    manifest: dict[str, object]
    files: dict[str, tuple[int, str]]


def _read_archive_member(
    handle: IO[bytes],
    *,
    capture: bool,
) -> tuple[int, str, bytes | None]:
    digest = hashlib.sha256()
    size_bytes = 0
    payload = bytearray() if capture else None
    while chunk := handle.read(1024 * 1024):
        size_bytes += len(chunk)
        digest.update(chunk)
        if payload is not None:
            if size_bytes > _ARCHIVE_MANIFEST_MAX_BYTES:
                raise ValueError("Archived request manifest is too large")
            payload.extend(chunk)
    return (
        size_bytes,
        digest.hexdigest(),
        bytes(payload) if payload is not None else None,
    )


def _inspect_request_archive(
    path: Path,
    presentation_name: str,
) -> _ArchiveInspection | None:
    """Stream and hash a local archive without extracting untrusted members."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        return None
    manifest_member = f"{presentation_name}/request_manifest.json"
    zstd = shutil.which("zstd")
    if zstd is None:
        return None
    process = sp.Popen(  # noqa: S603
        [zstd, "-dc", "--", str(path)],
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
    )
    members: set[str] = set()
    files: dict[str, tuple[int, str]] = {}
    manifest_bytes: bytes | None = None
    try:
        if process.stdout is None:
            raise RuntimeError("zstd archive reader has no stdout")
        with process.stdout:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    name = member.name
                    normalized_name = f"{name.rstrip('/')}/" if member.isdir() else name
                    if not normalized_name or normalized_name in members:
                        raise ValueError("Archive contains an invalid duplicate member")
                    members.add(normalized_name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("Archive contains a non-regular member")
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("Archive member is unreadable")
                    with handle:
                        size_bytes, digest, payload = _read_archive_member(
                            handle,
                            capture=name == manifest_member,
                        )
                    files[name] = (size_bytes, digest)
                    if payload is not None:
                        manifest_bytes = payload
        if process.wait() != 0 or manifest_bytes is None:
            return None
        manifest = orjson.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            return None
        return _ArchiveInspection(
            members=frozenset(members),
            manifest=cast(dict[str, object], manifest),
            files=files,
        )
    except (OSError, ValueError, tarfile.TarError, orjson.JSONDecodeError):
        return None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _archive_matches_request(
    path: Path,
    *,
    presentation_name: str,
    expected_members: set[str],
    expected_manifest: dict[str, object],
) -> bool:
    inspection = _inspect_request_archive(path, presentation_name)
    if (
        inspection is None
        or inspection.members != expected_members
        or inspection.manifest != expected_manifest
    ):
        return False
    artifacts = inspection.manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        archive_path = artifact.get("archive_path")
        size_bytes = artifact.get("archive_size_bytes")
        digest = artifact.get("archive_sha256")
        if (
            not isinstance(archive_path, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return False
        try:
            safe_archive_path = _safe_archive_path(archive_path)
        except ValueError:
            return False
        member = f"{presentation_name}/{safe_archive_path.as_posix()}"
        if inspection.files.get(member) != (size_bytes, digest):
            return False
    return True


def create_request_archive(
    reader: VolumeReader,
    manifest: dict[str, object],
    *,
    output_dir: str | Path,
    display_name: str,
) -> Path:
    """Download one request view and create a validated local ``.tar.zst``."""
    request_id, canonical_name, artifacts = _validated_manifest_artifacts(manifest)
    presentation_name = sanitize_af3_name(display_name)
    transformed_artifacts: list[tuple[dict[str, object], PurePosixPath]] = []
    transformed_paths: set[PurePosixPath] = set()
    for artifact in artifacts:
        transformed = _presentation_archive_path(
            cast(str, artifact["archive_path"]),
            canonical_name=canonical_name,
            presentation_name=presentation_name,
        )
        if transformed in transformed_paths:
            raise ValueError(f"Presentation path collision: {transformed}")
        transformed_paths.add(transformed)
        transformed_artifacts.append((artifact, transformed))
    local_manifest = _local_request_manifest(
        manifest,
        transformed_artifacts,
        display_name=display_name,
        canonical_name=canonical_name,
        presentation_name=presentation_name,
    )
    expected_members = _expected_archive_members(
        presentation_name,
        transformed_artifacts,
    )

    local_output_dir = Path(output_dir).expanduser().resolve()
    local_output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = (
        local_output_dir / f"{presentation_name}_{request_id[:12]}_AlphaFold3.tar.zst"
    )
    if archive_path.exists():
        _bind_expected_archive_artifacts(
            reader,
            local_manifest,
            transformed_artifacts,
            display_name=display_name,
        )
        if _archive_matches_request(
            archive_path,
            presentation_name=presentation_name,
            expected_members=expected_members,
            expected_manifest=local_manifest,
        ):
            return archive_path
        raise RuntimeError(
            "Existing AlphaFold3 archive does not match the current request: "
            f"{archive_path}"
        )

    temporary_archive = (
        local_output_dir / f".{archive_path.name}.{uuid.uuid4().hex}.partial"
    )
    try:
        with TemporaryDirectory(prefix="alphafold3_request_") as directory:
            work_root = Path(directory)
            archive_root = work_root / presentation_name
            archive_root.mkdir()
            input_paths: list[Path] = []
            for artifact, transformed in transformed_artifacts:
                destination = archive_root / Path(transformed.as_posix())
                _download_artifact(reader, artifact, destination)
                if artifact["role"] == "input":
                    input_paths.append(destination)
            if len(input_paths) != 1:
                raise RuntimeError(
                    "Request archive requires exactly one input artifact"
                )
            _rewrite_downloaded_input(
                input_paths[0],
                display_name=display_name,
                custom_template_paths=_custom_template_archive_paths(
                    transformed_artifacts
                ),
            )
            _record_archive_artifacts(
                local_manifest,
                transformed_artifacts,
                archive_root,
            )

            write_bytes_atomic(
                archive_root / "request_manifest.json",
                json_bytes(local_manifest),
            )

            run_command(
                [
                    "tar",
                    "-I",
                    "zstd -T0",
                    "-cf",
                    str(temporary_archive),
                    "--",
                    presentation_name,
                ],
                output_mode="capture",
                cwd=work_root,
            )
        if not _archive_matches_request(
            temporary_archive,
            presentation_name=presentation_name,
            expected_members=expected_members,
            expected_manifest=local_manifest,
        ):
            raise RuntimeError(
                "Generated AlphaFold3 archive failed exact member/manifest validation"
            )
        os.replace(temporary_archive, archive_path)
        return archive_path
    finally:
        temporary_archive.unlink(missing_ok=True)
