"""Humatch source repo: <https://github.com/oxpig/Humatch>.

Humanize complete VH-VL pairs with Humatch's joint heavy, light, and paired
classifiers. Input is a UTF-8 CSV with exactly ``id,vh,vl`` columns. The
result is one inline ``.tar.zst`` bundle containing final sequences, sequence-
level classifier distributions, IMGT alignment, and mutation provenance.
"""

# ruff: noqa: PLC0415

import hashlib
import re
import resource
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.app.design.humatch.execution import (
    MAX_RESULT_BYTES,
    HumatchExecutionCoordinator,
    HumatchExecutionRequest,
    load_execution_request,
    result_from_overview,
    stage_execution_request,
)
from biomodals.app.design.humatch.models import (
    ASSETS,
    IDENTITY,
    RUNTIME_IDENTITY,
    assert_humatch_assets,
    download_humatch_assets,
)
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    DeploymentIdentity,
    ExecutionOverview,
    RunStatus,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
    submit_staged_execution_run,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.io import build_local_output_path
from biomodals.helper.shell import package_outputs, sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
CSV_COLUMNS = ("id", "vh", "vl")
VH_FAMILIES = tuple(f"hv{i}" for i in range(1, 8))
VL_FAMILIES = tuple([
    *(f"lv{i}" for i in range(1, 11)),
    *(f"kv{i}" for i in range(1, 8)),
])
HEAVY_CLASSES = ("neg", *VH_FAMILIES)
LIGHT_CLASSES = ("neg", *VL_FAMILIES)
PAIRED_CLASSES = ("fake", "true")
MAX_PAIRS = 1000
MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_ID_LENGTH = 200
MAX_SEQUENCE_LENGTH = 200
NUM_CPUS = 16
CPU_LIMIT = 16.125
PAIR_PAD = "----------"
_COORDINATOR_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_POSITION_PATTERN = re.compile(r"([1-9]|[1-9][0-9]|1[01][0-9]|12[0-8])([A-L]?)")

CONF = AppConfig(
    tags={"group": "design"},
    name="Humatch",
    repo_url="https://github.com/oxpig/Humatch",
    repo_commit_hash=IDENTITY.source_commit,
    package_name="Humatch",
    version=IDENTITY.package_version,
    python_version="3.12",
    timeout=_COORDINATOR_TIMEOUT_SECONDS,
)

runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install("git")
    .micromamba_install(
        [
            f"anarci=={IDENTITY.anarci_version}",
            f"hmmer=={IDENTITY.hmmer_version}",
            f"biopython=={IDENTITY.biopython_version}",
            f"numpy=={IDENTITY.numpy_version}",
            f"pandas=={IDENTITY.pandas_version}",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .uv_pip_install(
        f"tensorflow=={IDENTITY.tensorflow_version}",
        f"keras=={IDENTITY.keras_version}",
        f"pyyaml=={IDENTITY.pyyaml_version}",
        f"matplotlib=={IDENTITY.matplotlib_version}",
        f"seaborn=={IDENTITY.seaborn_version}",
        f"requests=={IDENTITY.requests_version}",
    )
    .uv_pip_install(
        f"git+{CONF.repo_url}@{CONF.repo_commit_hash}",
        extra_options="--no-deps",
    )
    .add_local_python_source("biomodals.app.design.humatch.models", copy=True)
    .run_function(download_humatch_assets)
    .env({"TF_CPP_MIN_LOG_LEVEL": "3"})
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.humatch.execution")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)

EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_humatch_task"})


@dataclass(frozen=True, slots=True)
class _AlignedPair:
    identifier: str
    vh: str
    vl: str


def _cgroup_cpu_seconds() -> float | None:
    """Read CPU time for all processes in this Modal container."""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            name, value = line.split()
            if name == "usage_usec":
                return int(value) / 1_000_000
    except (OSError, ValueError):
        pass
    try:
        nanoseconds = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage").read_text()
        return int(nanoseconds.strip()) / 1_000_000_000
    except (OSError, ValueError):
        return None


def _cgroup_peak_memory_mib() -> float | None:
    """Read peak aggregate memory for the Modal container."""
    for path in (
        Path("/sys/fs/cgroup/memory.peak"),
        Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
    ):
        try:
            return int(path.read_text().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            pass
    return None


def _cgroup_current_memory_mib() -> float | None:
    """Read current aggregate memory for the Modal container."""
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            pass
    return None


def _process_peak_memory_mib() -> float:
    """Read the worker process high-water RSS on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _validate_parameters(
    *,
    vh_target_family: str,
    vl_target_family: str,
    germline_likeness_target: float,
    vh_classifier_target: float,
    vl_classifier_target: float,
    pair_classifier_target: float,
    max_edits: int,
    mutate_cdrs: bool,
    fixed_vh_positions: str,
    fixed_vl_positions: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if vh_target_family not in {"auto", *VH_FAMILIES}:
        raise ValueError("vh_target_family must be auto or hv1 through hv7")
    if vl_target_family not in {"auto", *VL_FAMILIES}:
        raise ValueError("vl_target_family must be auto, lv1-lv10, or kv1-kv7")
    for name, value in (
        ("germline_likeness_target", germline_likeness_target),
        ("vh_classifier_target", vh_classifier_target),
        ("vl_classifier_target", vl_classifier_target),
        ("pair_classifier_target", pair_classifier_target),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if type(max_edits) is not int or not 0 <= max_edits <= 400:
        raise ValueError("max_edits must be between 0 and 400")
    if type(mutate_cdrs) is not bool:
        raise ValueError("mutate_cdrs must be a boolean")
    return (
        _normalize_fixed_positions(fixed_vh_positions, "fixed_vh_positions"),
        _normalize_fixed_positions(fixed_vl_positions, "fixed_vl_positions"),
    )


def _normalize_fixed_positions(value: str, field_name: str) -> tuple[str, ...]:
    """Convert human-readable IMGT labels to Humatch's space-padded labels."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be comma-separated text")
    if not value.strip():
        return ()
    normalized: list[str] = []
    for raw_position in value.split(","):
        position = raw_position.strip().upper()
        match = _POSITION_PATTERN.fullmatch(position)
        if match is None:
            raise ValueError(
                f"{field_name} contains invalid IMGT position {position!r}"
            )
        upstream_position = position if match.group(2) else f"{position} "
        if upstream_position in normalized:
            raise ValueError(f"{field_name} contains duplicate IMGT positions")
        normalized.append(upstream_position)
    return tuple(normalized)


def parse_humatch_csv(content: bytes) -> pl.DataFrame:
    """Parse and validate a complete paired-chain input table."""
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV must be between 1 and {MAX_INPUT_BYTES} bytes")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Input CSV must be valid UTF-8") from exc
    try:
        frame = pl.read_csv(
            BytesIO(content),
            schema_overrides={column: pl.String for column in CSV_COLUMNS},
            infer_schema=False,
        )
    except Exception as exc:
        raise ValueError(f"Input CSV could not be parsed: {exc}") from exc
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ValueError("Input CSV must have exactly these columns: id,vh,vl")
    if not 1 <= frame.height <= MAX_PAIRS:
        raise ValueError(f"Input CSV must contain between 1 and {MAX_PAIRS} pairs")

    seen: set[str] = set()
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Row {row_number}: id must be non-empty")
        if identifier != identifier.strip():
            raise ValueError(f"Row {row_number}: id must not have outer whitespace")
        if len(identifier) > MAX_ID_LENGTH or any(
            ord(character) < 32 or character == ">" for character in identifier
        ):
            raise ValueError(f"Row {row_number}: id contains unsupported characters")
        if identifier in seen:
            raise ValueError(f"Row {row_number}: duplicate id {identifier!r}")
        seen.add(identifier)
        for column in ("vh", "vl"):
            sequence = row[column]
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Row {row_number}: {column} must be non-empty")
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"Row {row_number}: {column} exceeds {MAX_SEQUENCE_LENGTH} residues"
                )
            invalid = sorted(set(sequence) - AMINO_ACIDS)
            if invalid:
                raise ValueError(
                    f"Row {row_number}: {column} contains non-canonical or "
                    f"lowercase residues: {''.join(invalid)}"
                )
    return frame


def _load_upstream() -> dict[str, Any]:
    import anarci  # type: ignore[ty:unresolved-import]
    from Humatch.classify import (  # type: ignore[ty:unresolved-import]
        get_class_and_score_of_max_predictions_only,
        predict_from_list_of_seq_strs,
    )
    from Humatch.germline_likeness import (  # type: ignore[ty:unresolved-import]
        GL_DIR,
        get_normalised_germline_likeness_score,
        mutate_seq_to_match_germline_likeness,
    )
    from Humatch.humanise import humanise  # type: ignore[ty:unresolved-import]
    from Humatch.model import (  # type: ignore[ty:unresolved-import]
        HEAVY_WEIGHTS,
        LIGHT_WEIGHTS,
        PAIRED_WEIGHTS,
        load_cnn,
    )
    from Humatch.utils import CANONICAL_NUMBERING  # type: ignore[ty:unresolved-import]

    return {
        "anarci": anarci,
        "canonical_numbering": tuple(CANONICAL_NUMBERING),
        "get_max_class": get_class_and_score_of_max_predictions_only,
        "predict": predict_from_list_of_seq_strs,
        "gl_dir": GL_DIR,
        "gl_score": get_normalised_germline_likeness_score,
        "gl_mutate": mutate_seq_to_match_germline_likeness,
        "humanise": humanise,
        "load_cnn": load_cnn,
        "weights": (HEAVY_WEIGHTS, LIGHT_WEIGHTS, PAIRED_WEIGHTS),
    }


_MODEL_CACHE: tuple[Any, Any, Any] | None = None


def _load_models(upstream: dict[str, Any]) -> tuple[tuple[Any, Any, Any], float]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, 0.0
    assert_humatch_assets()
    started_at = time.perf_counter()
    _MODEL_CACHE = tuple(
        upstream["load_cnn"](path, model_type)
        for path, model_type in zip(
            upstream["weights"], ("heavy", "light", "paired"), strict=True
        )
    )
    return _MODEL_CACHE, time.perf_counter() - started_at


def _align_chain(
    *,
    identifier: str,
    chain_label: str,
    sequence: str,
    upstream: dict[str, Any],
) -> str:
    numbering, chain_type = upstream["anarci"].number(sequence, scheme="imgt")
    if numbering is False:
        raise ValueError(f"{identifier}: {chain_label} could not be numbered by ANARCI")
    expected_type = "H" if chain_label == "vh" else "L"
    if chain_type != expected_type:
        expected_name = "heavy" if chain_label == "vh" else "kappa/lambda light"
        raise ValueError(
            f"{identifier}: {chain_label} was not identified as a {expected_name} chain"
        )
    residues = {f"{position[0]}{position[1]}": aa for position, aa in numbering}
    aligned = "".join(
        residues.get(position, "-") for position in upstream["canonical_numbering"]
    )
    if aligned.replace("-", "") != sequence:
        raise ValueError(
            f"{identifier}: {chain_label} contains residues outside Humatch's "
            "canonical IMGT alignment"
        )
    required_end = "128 " if chain_label == "vh" else "127 "
    for required in ("1 ", "23 ", "104 ", required_end):
        if residues.get(required, "-") == "-":
            raise ValueError(
                f"{identifier}: {chain_label} lacks required IMGT position "
                f"{required.strip()}"
            )
    if residues["23 "] != "C" or residues["104 "] != "C":
        raise ValueError(
            f"{identifier}: {chain_label} lacks conserved cysteine at IMGT 23 or 104"
        )
    return aligned


def _select_target(
    probabilities: Any,
    classifier_type: str,
    requested: str,
    upstream: dict[str, Any],
) -> str:
    if requested != "auto":
        return requested
    selected, _ = upstream["get_max_class"](
        probabilities.reshape(1, -1).copy(), classifier_type
    )[0]
    return str(selected)


def _predict_distributions(
    vh: str,
    vl: str,
    models: tuple[Any, Any, Any],
    upstream: dict[str, Any],
) -> tuple[Any, Any, Any]:
    predict = upstream["predict"]
    vh_scores = predict([vh], models[0], num_cpus=NUM_CPUS)[0]
    vl_scores = predict([vl], models[1], num_cpus=NUM_CPUS)[0]
    pair_scores = predict([vh + PAIR_PAD + vl], models[2], num_cpus=NUM_CPUS)[0]
    if (
        len(vh_scores) != len(HEAVY_CLASSES)
        or len(vl_scores) != len(LIGHT_CLASSES)
        or len(pair_scores) != len(PAIRED_CLASSES)
    ):
        raise ValueError("Humatch returned an unexpected classifier shape")
    return vh_scores, vl_scores, pair_scores


def _classifier_row(
    *,
    identifier: str,
    endpoint: str,
    scores: tuple[Any, Any, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": identifier, "endpoint": endpoint}
    for prefix, classes, values in (
        ("h", HEAVY_CLASSES, scores[0]),
        ("l", LIGHT_CLASSES, scores[1]),
        ("p", PAIRED_CLASSES, scores[2]),
    ):
        row.update({
            f"{prefix}_{name}": float(value)
            for name, value in zip(classes, values, strict=True)
        })
    return row


def _region_for_index(index: int, canonical: tuple[str, ...]) -> str:
    cdr1 = (canonical.index("27 "), canonical.index("38 "))
    cdr2 = (canonical.index("56 "), canonical.index("65 "))
    cdr3 = (canonical.index("105 "), canonical.index("117 "))
    if index < cdr1[0]:
        return "FR1"
    if index <= cdr1[1]:
        return "CDR1"
    if index < cdr2[0]:
        return "FR2"
    if index <= cdr2[1]:
        return "CDR2"
    if index < cdr3[0]:
        return "FR3"
    if index <= cdr3[1]:
        return "CDR3"
    return "FR4"


def _alignment_and_mutations(
    *,
    identifier: str,
    chain: str,
    parental: str,
    final: str,
    canonical: tuple[str, ...],
    fixed_positions: tuple[str, ...],
    mutate_cdrs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alignment_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    fixed = set(fixed_positions)
    for index, (position, before, after) in enumerate(
        zip(canonical, parental, final, strict=True)
    ):
        if before == after == "-":
            continue
        region = _region_for_index(index, canonical)
        mutated = before != after
        position_label = position.strip()
        alignment_rows.append({
            "id": identifier,
            "chain": chain,
            "imgt_position": position_label,
            "region": region,
            "input_aa": before,
            "final_aa": after,
            "mutated": mutated,
            "protected": position in fixed
            or (not mutate_cdrs and region.startswith("CDR")),
        })
        if mutated:
            mutation_rows.append({
                "id": identifier,
                "chain": chain,
                "imgt_position": position_label,
                "region": region,
                "from_aa": before,
                "to_aa": after,
            })
    return alignment_rows, mutation_rows


def _humanize_pair(
    *,
    pair: _AlignedPair,
    models: tuple[Any, Any, Any],
    upstream: dict[str, Any],
    vh_target_family: str,
    vl_target_family: str,
    germline_likeness_target: float,
    vh_classifier_target: float,
    vl_classifier_target: float,
    pair_classifier_target: float,
    max_edits: int,
    mutate_cdrs: bool,
    fixed_vh_positions: tuple[str, ...],
    fixed_vl_positions: tuple[str, ...],
) -> dict[str, Any]:
    input_scores = _predict_distributions(pair.vh, pair.vl, models, upstream)
    selected_vh = _select_target(input_scores[0], "heavy", vh_target_family, upstream)
    selected_vl = _select_target(input_scores[1], "light", vl_target_family, upstream)
    gl_mutate = upstream["gl_mutate"]
    gl_dir = upstream["gl_dir"]
    post_gl_vh = gl_mutate(
        pair.vh,
        selected_vh,
        germline_likeness_target,
        allow_CDR_mutations=mutate_cdrs,
        fixed_imgt_positions=list(fixed_vh_positions),
        germline_likeness_lookup_arrays_dir=gl_dir,
    )
    post_gl_vl = gl_mutate(
        pair.vl,
        selected_vl,
        germline_likeness_target,
        allow_CDR_mutations=mutate_cdrs,
        fixed_imgt_positions=list(fixed_vl_positions),
        germline_likeness_lookup_arrays_dir=gl_dir,
    )
    config = {
        "max_edit": max_edits,
        "num_cpus": NUM_CPUS,
        "GL_target_score_H": germline_likeness_target,
        "GL_allow_CDR_mutations_H": mutate_cdrs,
        "GL_fixed_imgt_positions_H": list(fixed_vh_positions),
        "CNN_target_score_H": vh_classifier_target,
        "CNN_allow_CDR_mutations_H": mutate_cdrs,
        "CNN_fixed_imgt_positions_H": list(fixed_vh_positions),
        "target_gene_H": selected_vh,
        "GL_target_score_L": germline_likeness_target,
        "GL_allow_CDR_mutations_L": mutate_cdrs,
        "GL_fixed_imgt_positions_L": list(fixed_vl_positions),
        "CNN_target_score_L": vl_classifier_target,
        "CNN_allow_CDR_mutations_L": mutate_cdrs,
        "CNN_fixed_imgt_positions_L": list(fixed_vl_positions),
        "target_gene_L": selected_vl,
        "CNN_target_score_P": pair_classifier_target,
        "germline_likeness_lookup_arrays_dir": gl_dir,
    }
    started_at = time.perf_counter()
    cpu_started_at = _cgroup_cpu_seconds()
    result = upstream["humanise"](
        pair.vh,
        pair.vl,
        models[0],
        models[1],
        models[2],
        config,
    )
    humanization_seconds = time.perf_counter() - started_at
    cpu_finished_at = _cgroup_cpu_seconds()
    final_vh = str(result["Humatch_H"])
    final_vl = str(result["Humatch_L"])
    if len(final_vh) != len(pair.vh) or len(final_vl) != len(pair.vl):
        raise ValueError("Humatch returned an unexpected aligned sequence length")
    final_scores = _predict_distributions(final_vh, final_vl, models, upstream)
    gl_score = upstream["gl_score"]
    gl_values = {
        "vh_input_germline_likeness": float(gl_score(pair.vh, selected_vh, gl_dir)),
        "vh_post_germline_likeness": float(gl_score(post_gl_vh, selected_vh, gl_dir)),
        "vh_final_germline_likeness": float(gl_score(final_vh, selected_vh, gl_dir)),
        "vl_input_germline_likeness": float(gl_score(pair.vl, selected_vl, gl_dir)),
        "vl_post_germline_likeness": float(gl_score(post_gl_vl, selected_vl, gl_dir)),
        "vl_final_germline_likeness": float(gl_score(final_vl, selected_vl, gl_dir)),
    }
    returned_scores = (
        float(result["CNN_H"]),
        float(result["CNN_L"]),
        float(result["CNN_P"]),
    )
    success = (
        returned_scores[0] >= vh_classifier_target
        and returned_scores[1] >= vl_classifier_target
        and returned_scores[2] >= pair_classifier_target
    )
    canonical = upstream["canonical_numbering"]
    vh_alignment, vh_mutations = _alignment_and_mutations(
        identifier=pair.identifier,
        chain="vh",
        parental=pair.vh,
        final=final_vh,
        canonical=canonical,
        fixed_positions=fixed_vh_positions,
        mutate_cdrs=mutate_cdrs,
    )
    vl_alignment, vl_mutations = _alignment_and_mutations(
        identifier=pair.identifier,
        chain="vl",
        parental=pair.vl,
        final=final_vl,
        canonical=canonical,
        fixed_positions=fixed_vl_positions,
        mutate_cdrs=mutate_cdrs,
    )
    return {
        "humanized": {
            "id": pair.identifier,
            "vh": final_vh.replace("-", ""),
            "vl": final_vl.replace("-", ""),
        },
        "summary": {
            "id": pair.identifier,
            "vh_target_family": str(result["HV"]),
            "vl_target_family": str(result["LV"]),
            "edit_count": int(result["Edit"]),
            "humanization_success": success,
            "vh_input_target_probability": float(
                input_scores[0][HEAVY_CLASSES.index(selected_vh)]
            ),
            "vh_final_target_probability": returned_scores[0],
            "vl_input_target_probability": float(
                input_scores[1][LIGHT_CLASSES.index(selected_vl)]
            ),
            "vl_final_target_probability": returned_scores[1],
            "pair_input_true_probability": float(input_scores[2][1]),
            "pair_final_true_probability": returned_scores[2],
            **gl_values,
        },
        "classifier_scores": [
            _classifier_row(
                identifier=pair.identifier, endpoint="input", scores=input_scores
            ),
            _classifier_row(
                identifier=pair.identifier, endpoint="final", scores=final_scores
            ),
        ],
        "alignment": [*vh_alignment, *vl_alignment],
        "mutations": [*vh_mutations, *vl_mutations],
        "humanization_seconds": humanization_seconds,
        "humanization_cpu_seconds": (
            None
            if cpu_started_at is None or cpu_finished_at is None
            else max(0.0, cpu_finished_at - cpu_started_at)
        ),
    }


def _write_result_bundle(
    *,
    run_name: str,
    input_frame: pl.DataFrame,
    pair_results: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> bytes:
    with TemporaryDirectory(prefix="humatch_") as temporary:
        result_dir = Path(temporary) / run_name
        result_dir.mkdir()
        input_frame.write_csv(result_dir / "input.csv")
        humanized_rows = [result["humanized"] for result in pair_results]
        pl.DataFrame(
            humanized_rows,
            schema={column: pl.String for column in CSV_COLUMNS},
        ).write_csv(result_dir / "humanized.csv")
        fasta_lines = [
            line
            for row in humanized_rows
            for line in (
                f">{row['id']}_VH",
                row["vh"],
                f">{row['id']}_VL",
                row["vl"],
            )
        ]
        (result_dir / "humanized.fasta").write_text(
            "\n".join(fasta_lines) + "\n", encoding="utf-8"
        )
        pl.DataFrame([result["summary"] for result in pair_results]).write_csv(
            result_dir / "summary.csv"
        )
        pl.DataFrame([
            row for result in pair_results for row in result["classifier_scores"]
        ]).write_csv(result_dir / "classifier_scores.csv")
        pl.DataFrame([
            row for result in pair_results for row in result["alignment"]
        ]).write_csv(result_dir / "alignment.csv")
        mutation_schema = {
            "id": pl.String,
            "chain": pl.String,
            "imgt_position": pl.String,
            "region": pl.String,
            "from_aa": pl.String,
            "to_aa": pl.String,
        }
        pl.DataFrame(
            [row for result in pair_results for row in result["mutations"]],
            schema=mutation_schema,
        ).write_csv(result_dir / "mutations.csv")

        artifact_files = sorted(result_dir.iterdir())
        manifest = {
            "schema_version": 1,
            "run_name": run_name,
            "pair_count": input_frame.height,
            "parameters": parameters,
            "scientific_identity": {
                "humatch_package": IDENTITY.package_version,
                "humatch_commit": IDENTITY.source_commit,
                "asset_record": IDENTITY.asset_doi,
                "asset_md5": {asset.filename: asset.md5_hex for asset in ASSETS},
                "runtime": RUNTIME_IDENTITY,
            },
            "input_sha256": hashlib.sha256(
                input_frame.write_csv().encode("utf-8")
            ).hexdigest(),
            "files": {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in artifact_files
            },
        }
        (result_dir / "manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        archive = package_outputs(result_dir, num_threads=2)
        if len(archive) > MAX_RESULT_BYTES:
            raise ValueError("Humatch result archive exceeds the 128 MiB limit")
        return archive


def _run_humatch_humanization(
    *,
    run_name: str,
    csv_bytes: bytes,
    vh_target_family: str = "auto",
    vl_target_family: str = "auto",
    germline_likeness_target: float = 0.40,
    vh_classifier_target: float = 0.95,
    vl_classifier_target: float = 0.95,
    pair_classifier_target: float = 0.95,
    max_edits: int = 60,
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
) -> tuple[bytes, dict[str, Any]]:
    """Validate, humanize sequentially, and package one paired batch."""
    normalized_vh_positions, normalized_vl_positions = _validate_parameters(
        vh_target_family=vh_target_family,
        vl_target_family=vl_target_family,
        germline_likeness_target=germline_likeness_target,
        vh_classifier_target=vh_classifier_target,
        vl_classifier_target=vl_classifier_target,
        pair_classifier_target=pair_classifier_target,
        max_edits=max_edits,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
    )
    input_frame = parse_humatch_csv(csv_bytes)
    started_at = time.perf_counter()
    cpu_started_at = _cgroup_cpu_seconds()
    upstream = _load_upstream()
    canonical = set(upstream["canonical_numbering"])
    for field_name, positions in (
        ("fixed_vh_positions", normalized_vh_positions),
        ("fixed_vl_positions", normalized_vl_positions),
    ):
        invalid = [
            position.strip() for position in positions if position not in canonical
        ]
        if invalid:
            raise ValueError(
                f"{field_name} contains positions outside Humatch's canonical "
                f"numbering: {', '.join(invalid)}"
            )
    aligned_pairs = [
        _AlignedPair(
            identifier=row["id"],
            vh=_align_chain(
                identifier=row["id"],
                chain_label="vh",
                sequence=row["vh"],
                upstream=upstream,
            ),
            vl=_align_chain(
                identifier=row["id"],
                chain_label="vl",
                sequence=row["vl"],
                upstream=upstream,
            ),
        )
        for row in input_frame.iter_rows(named=True)
    ]
    models, model_load_seconds = _load_models(upstream)
    pair_results = [
        _humanize_pair(
            pair=pair,
            models=models,
            upstream=upstream,
            vh_target_family=vh_target_family,
            vl_target_family=vl_target_family,
            germline_likeness_target=germline_likeness_target,
            vh_classifier_target=vh_classifier_target,
            vl_classifier_target=vl_classifier_target,
            pair_classifier_target=pair_classifier_target,
            max_edits=max_edits,
            mutate_cdrs=mutate_cdrs,
            fixed_vh_positions=normalized_vh_positions,
            fixed_vl_positions=normalized_vl_positions,
        )
        for pair in aligned_pairs
    ]
    parameters = {
        "vh_target_family": vh_target_family,
        "vl_target_family": vl_target_family,
        "germline_likeness_target": germline_likeness_target,
        "vh_classifier_target": vh_classifier_target,
        "vl_classifier_target": vl_classifier_target,
        "pair_classifier_target": pair_classifier_target,
        "max_edits": max_edits,
        "mutate_cdrs": mutate_cdrs,
        "fixed_vh_positions": [
            position.strip() for position in normalized_vh_positions
        ],
        "fixed_vl_positions": [
            position.strip() for position in normalized_vl_positions
        ],
    }
    archive = _write_result_bundle(
        run_name=run_name,
        input_frame=input_frame,
        pair_results=pair_results,
        parameters=parameters,
    )
    elapsed_seconds = time.perf_counter() - started_at
    cpu_finished_at = _cgroup_cpu_seconds()
    humanization_seconds = sum(
        result["humanization_seconds"] for result in pair_results
    )
    humanization_cpu_seconds = sum(
        result["humanization_cpu_seconds"] or 0.0 for result in pair_results
    )
    metrics = {
        "pair_count": input_frame.height,
        "mutation_count": sum(len(result["mutations"]) for result in pair_results),
        "success_count": sum(
            bool(result["summary"]["humanization_success"]) for result in pair_results
        ),
        "model_load_seconds": model_load_seconds,
        "humanization_seconds": humanization_seconds,
        "elapsed_seconds": elapsed_seconds,
    }
    if cpu_started_at is not None and cpu_finished_at is not None:
        container_cpu_seconds = max(0.0, cpu_finished_at - cpu_started_at)
        metrics.update({
            "container_cpu_seconds": container_cpu_seconds,
            "average_container_cpu_fraction": (
                container_cpu_seconds / elapsed_seconds / CPU_LIMIT
            ),
        })
    if humanization_seconds > 0 and any(
        result["humanization_cpu_seconds"] is not None for result in pair_results
    ):
        metrics.update({
            "humanization_cpu_seconds": humanization_cpu_seconds,
            "average_humanization_cpu_fraction": (
                humanization_cpu_seconds / humanization_seconds / CPU_LIMIT
            ),
        })
    peak_memory_mib = _cgroup_peak_memory_mib()
    if peak_memory_mib is not None:
        metrics["peak_container_memory_mib"] = peak_memory_mib
    current_memory_mib = _cgroup_current_memory_mib()
    if current_memory_mib is not None:
        metrics["container_memory_mib_at_completion"] = current_memory_mib
    metrics["peak_worker_process_memory_mib"] = _process_peak_memory_mib()
    return archive, metrics


@app.function(
    cpu=(2.125, 16.125),
    memory=(256, 16384),
    timeout=CONF.timeout,
    max_containers=1,
)
def humatch_humanize(
    run_name: str,
    csv_bytes: bytes,
    vh_target_family: str = "auto",
    vl_target_family: str = "auto",
    germline_likeness_target: float = 0.40,
    vh_classifier_target: float = 0.95,
    vl_classifier_target: float = 0.95,
    pair_classifier_target: float = 0.95,
    max_edits: int = 60,
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
) -> AppRunResult:
    """Humanize a validated paired CSV batch and return an inline archive."""
    run_name = sanitize_filename(run_name)
    archive, metrics = _run_humatch_humanization(
        run_name=run_name,
        csv_bytes=csv_bytes,
        vh_target_family=vh_target_family,
        vl_target_family=vl_target_family,
        germline_likeness_target=germline_likeness_target,
        vh_classifier_target=vh_classifier_target,
        vl_classifier_target=vl_classifier_target,
        pair_classifier_target=pair_classifier_target,
        max_edits=max_edits,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
    )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="humatch_humanization",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=archive,
                    filename=f"{run_name}_humatch.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
                metadata={
                    "archive_format": "tar.zst",
                    "run_name": run_name,
                    "pair_count": metrics["pair_count"],
                    "classifier_endpoints": ["input", "final"],
                },
            )
        ],
        metrics=metrics,
    )


@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=_COORDINATOR_TIMEOUT_SECONDS,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with the Humatch worker."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh output state before accepting lifecycle methods."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive one staged root app run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read this run's durable kernel overview."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Request idempotent cancellation for this run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume this run without retrying failed work."""
        return self._adapter().resume()

    @modal.method()
    def prepare_restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> None:
        """Persist a validated successor request without driving it."""
        self._adapter().prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )

    @modal.method()
    def drive_prepared(self) -> ExecutionOverview:
        """Drive one previously prepared root or successor run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(self, predecessor_execution_run_id: str) -> ExecutionOverview:
        """Create a compatible successor while inferring predecessor identity."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                CONF.output_volume_mountpoint,
                UUID(self.execution_run_id),
            ),
        )
        return adapter.drive_prepared()

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return execution_coordinator_identity(self)

    def _adapter(
        self, *, development: bool | None = None
    ) -> HumatchExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: HumatchExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_volume_name=CONF.output_volume_name,
                provider_driver=_coordinator_modal_driver(development=selected_mode),
                app_version=CONF.repo_commit_hash or IDENTITY.source_commit,
                asset_record=IDENTITY.asset_doi,
                runtime_identity=RUNTIME_IDENTITY,
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {"humatch_humanize": humatch_humanize},
        workload_name=CONF.name,
    )


@app.local_entrypoint()
def submit_humatch_task(
    input_csv: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    vh_target_family: str = "auto",
    vl_target_family: str = "auto",
    germline_likeness_target: float = 0.40,
    vh_classifier_target: float = 0.95,
    vl_classifier_target: float = 0.95,
    pair_classifier_target: float = 0.95,
    max_edits: int = 60,
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Humanize paired variable regions and save the result archive locally."""
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if input_path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV exceeds {MAX_INPUT_BYTES} bytes")
    csv_bytes = input_path.read_bytes()
    parse_humatch_csv(csv_bytes)
    _validate_parameters(
        vh_target_family=vh_target_family,
        vl_target_family=vl_target_family,
        germline_likeness_target=germline_likeness_target,
        vh_classifier_target=vh_classifier_target,
        vl_classifier_target=vl_classifier_target,
        pair_classifier_target=pair_classifier_target,
        max_edits=max_edits,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
    )
    selected_run_name = sanitize_filename(run_name or input_path.stem)
    request = HumatchExecutionRequest(
        run_name=selected_run_name,
        csv_bytes=csv_bytes,
        vh_target_family=vh_target_family,
        vl_target_family=vl_target_family,
        germline_likeness_target=germline_likeness_target,
        vh_classifier_target=vh_classifier_target,
        vl_classifier_target=vl_classifier_target,
        pair_classifier_target=pair_classifier_target,
        max_edits=max_edits,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        app_version=CONF.repo_commit_hash or IDENTITY.source_commit,
        asset_record=IDENTITY.asset_doi,
        runtime_identity=RUNTIME_IDENTITY,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment, deployment_name, deployment_version
    )
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    overview = submit_staged_execution_run(
        CONF.output_volume,
        execution_run_id=execution_run_id,
        deployment=deployment,
        predecessor_execution_run_id=(
            None if restart_from is None else UUID(restart_from)
        ),
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
        workload_name=CONF.name,
        accepted_statuses=(RunStatus.SUCCEEDED,),
    )
    result = result_from_overview(overview, CONF.output_volume)
    elapsed_seconds = result.metrics.get("elapsed_seconds")
    if isinstance(elapsed_seconds, int | float):
        print(f"Humatch worker time: {elapsed_seconds:.3f} seconds")
    output = next(
        item for item in result.outputs if item.name == "humatch_humanization"
    )
    if not isinstance(output.storage, InlineBytes):
        raise TypeError("Humatch output must be inline .tar.zst bytes")
    local_output_dir = Path.cwd() if output_dir is None else Path(output_dir)
    output_path = build_local_output_path(
        local_output_dir,
        run_name=selected_run_name,
        suffix="humatch",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output.storage.data)
    print(f"Humatch results saved to: {output_path.resolve()}")
