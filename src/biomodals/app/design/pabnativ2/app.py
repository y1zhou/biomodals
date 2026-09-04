"""Paired antibody humanization with pinned p-AbNatiV2 source behavior.

Upstream: https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ

The checksum-verified model assets must first be staged with
``stage_pabnativ2_models``. Results are research-use design suggestions, not
evidence that binding, developability, or clinical immunogenicity is preserved.

Input is a UTF-8 CSV with exactly ``id,vh,vl`` columns. The completed standalone
run materializes one ``.tar.zst`` bundle with humanized pairs, endpoint sequence
and residue scores, endpoint mutations, and predicted structures.
"""

# ruff: noqa: PLC0415

import hashlib
import random
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
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
from biomodals.app.design.pabnativ2.execution import (
    MAX_PAIR_RESULT_BYTES,
    MAX_RESULT_BYTES,
    PAIRS_PER_GPU_CALL,
    PAbNatiV2ExecutionCoordinator,
    PAbNatiV2ExecutionRequest,
    load_execution_request,
    result_from_overview,
    stage_execution_request,
)
from biomodals.app.design.pabnativ2.models import (
    ABNATIV_WHEEL_URL,
    IDENTITY,
    PAIRED_MODEL,
    RUNTIME_IDENTITY,
    STRUCTURE_MODEL_ARCHIVE,
    assert_pabnativ2_assets,
    stage_pabnativ2_assets,
)
from biomodals.app.design.pabnativ2.patches import (
    PSSM_SHA256,
    apply_upstream_compatibility_patches,
    install_missing_abnativ_pssms,
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
    resolve_provider_call_limits,
    submit_staged_execution_run,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import build_local_output_path
from biomodals.helper.shell import package_outputs, sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
RECONSTRUCTION_SYMBOLS = (*AMINO_ACIDS, "-")
CSV_COLUMNS = ("id", "vh", "vl")
MAX_PAIRS = 1000
MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_ID_LENGTH = 200
MAX_SEQUENCE_LENGTH = 200
MAX_UINT32 = 2**32 - 1
MODEL_ROOT = Path("/biomodals-store")
_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_DEFAULT_MAX_GPU_CONTAINERS = 8
_WRAPPER_PROTOCOL_VERSION = "2"
_PATCH_PROTOCOL_VERSION = "1"
SCIENTIFIC_RUNTIME_IDENTITY = "|".join((
    RUNTIME_IDENTITY,
    f"wrapper-protocol={_WRAPPER_PROTOCOL_VERSION}",
    f"patch-protocol={_PATCH_PROTOCOL_VERSION}",
    "pssm-set-sha256="
    + hashlib.sha256(
        orjson.dumps(PSSM_SHA256, option=orjson.OPT_SORT_KEYS)
    ).hexdigest(),
))

CONF = AppConfig(
    tags={"group": "design"},
    name="p-AbNatiV2",
    repo_url="https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ",
    repo_commit_hash=IDENTITY.abnativ_commit,
    package_name="abnativ",
    version=IDENTITY.abnativ_version,
    python_version="3.12",
    cuda_version=IDENTITY.torch_cuda_version,
    timeout=_TIMEOUT_SECONDS,
)

runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install("build-essential", "git")
    .env(CONF.default_env | {"ABNATIV_MODELS_DIR": str(MODEL_ROOT)})
    .micromamba_install(
        [
            f"anarci=={IDENTITY.anarci_version}",
            f"hmmer=={IDENTITY.hmmer_version}",
            f"pdbfixer=={IDENTITY.pdbfixer_version}",
            f"freesasa=={IDENTITY.freesasa_version}",
            f"biopython=={IDENTITY.biopython_version}",
            f"numpy=={IDENTITY.numpy_version}",
            f"pandas=={IDENTITY.pandas_version}",
            f"scipy=={IDENTITY.scipy_version}",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .uv_pip_install(
        f"torch=={IDENTITY.torch_version}",
        f"pytorch-lightning=={IDENTITY.pytorch_lightning_version}",
        f"lightning=={IDENTITY.lightning_version}",
        f"matplotlib=={IDENTITY.matplotlib_version}",
        f"seaborn=={IDENTITY.seaborn_version}",
        "accelerate==1.9.0",
        "brewer2mpl==1.4.1",
        "dm-tree==0.1.9",
        "einops==0.8.1",
        "levenshtein==0.27.1",
        "loguru==0.7.3",
        "ml-collections==1.1.0",
        "paretoset==1.2.5",
        f"protein-topmodel=={IDENTITY.protein_topmodel_version}",
        "python-box==7.3.2",
        "pyyaml==6.0.2",
        "sentencepiece==0.2.1",
        "tensorboard==2.20.0",
        "tensorboardX==2.6.4",
        "transformers==4.53.3",
        "tqdm==4.67.1",
        ABNATIV_WHEEL_URL,
    )
    .uv_pip_install(
        f"git+https://github.com/Exscientia/abodybuilder3.git@"
        f"{IDENTITY.abodybuilder3_commit}",
        extra_options="--no-deps",
    )
    .add_local_python_source("biomodals.app.design.pabnativ2.patches", copy=True)
    .run_function(apply_upstream_compatibility_patches)
    .run_function(install_missing_abnativ_pssms)
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.pabnativ2")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_pabnativ2_task"})


@dataclass(frozen=True, slots=True)
class _Parameters:
    mutate_cdrs: bool
    fixed_vh_positions: tuple[int, ...]
    fixed_vl_positions: tuple[int, ...]
    residue_score_threshold: float
    rasa_threshold: float
    max_relative_pairing_score_decrease: float
    forbidden_residues: tuple[str, ...]
    seed: int


def parse_pabnativ2_csv(content: bytes) -> pl.DataFrame:
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

    canonical = set(AMINO_ACIDS)
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
            invalid = sorted(set(sequence) - canonical)
            if invalid:
                raise ValueError(
                    f"Row {row_number}: {column} contains non-canonical or "
                    f"lowercase residues: {''.join(invalid)}"
                )
    return frame


def _fixed_positions(value: str, name: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be comma-separated AHo integers")
    if not value.strip():
        return ()
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item.isascii() or not item.isdecimal():
            raise ValueError(f"{name} contains invalid AHo position {item!r}")
        position = int(item)
        if not 1 <= position <= 149:
            raise ValueError(f"{name} positions must be between 1 and 149")
        if position in result:
            raise ValueError(f"{name} contains duplicate AHo positions")
        result.append(position)
    return tuple(result)


def _validate_parameters(
    *,
    mutate_cdrs: bool,
    fixed_vh_positions: str,
    fixed_vl_positions: str,
    residue_score_threshold: float,
    rasa_threshold: float,
    max_relative_pairing_score_decrease: float,
    forbidden_residues: str,
    seed: int,
) -> _Parameters:
    if type(mutate_cdrs) is not bool:
        raise ValueError("mutate_cdrs must be a boolean")
    thresholds = (
        ("residue_score_threshold", residue_score_threshold),
        ("rasa_threshold", rasa_threshold),
        (
            "max_relative_pairing_score_decrease",
            max_relative_pairing_score_decrease,
        ),
    )
    for name, value in thresholds:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be numeric")
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if type(seed) is not int or not 0 <= seed <= MAX_UINT32:
        raise ValueError(f"seed must be an integer between 0 and {MAX_UINT32}")
    if not isinstance(forbidden_residues, str):
        raise ValueError("forbidden_residues must be comma-separated text")
    forbidden = (
        tuple(item.strip() for item in forbidden_residues.split(","))
        if forbidden_residues.strip()
        else ()
    )
    if any(len(item) != 1 or item not in AMINO_ACIDS for item in forbidden):
        raise ValueError("forbidden_residues must contain uppercase canonical residues")
    if len(set(forbidden)) != len(forbidden):
        raise ValueError("forbidden_residues must not contain duplicates")
    return _Parameters(
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=_fixed_positions(fixed_vh_positions, "fixed_vh_positions"),
        fixed_vl_positions=_fixed_positions(fixed_vl_positions, "fixed_vl_positions"),
        residue_score_threshold=float(residue_score_threshold),
        rasa_threshold=float(rasa_threshold),
        max_relative_pairing_score_decrease=float(max_relative_pairing_score_decrease),
        forbidden_residues=forbidden,
        seed=seed,
    )


def _validate_antibody_chains(frame: pl.DataFrame) -> None:
    """Require ANARCI to recognize every complete variable region in its role."""
    import anarci  # type: ignore[ty:unresolved-import]

    for row in frame.iter_rows(named=True):
        for column, allowed in (("vh", {"H"}), ("vl", {"K", "L"})):
            result = anarci.number(row[column], scheme="aho")
            if not result or result[0] is None or result[1] not in allowed:
                role = "heavy" if column == "vh" else "light"
                raise ValueError(
                    f"{row['id']}: {column} is not a complete {role} chain"
                )
            numbered = "".join(residue for _, residue in result[0] if residue != "-")
            if numbered != row[column]:
                raise ValueError(
                    f"{row['id']}: {column} contains residues outside the "
                    "AHo-numbered variable region"
                )


def _pair_seed(base_seed: int, identifier: str, vh: str, vl: str) -> int:
    payload = f"{base_seed}\0{identifier}\0{vh}\0{vl}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch  # type: ignore[ty:unresolved-import]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _allowed_positions(mutate_cdrs: bool, fixed: tuple[int, ...]) -> list[int]:
    if mutate_cdrs:
        candidates = range(1, 150)
    else:
        candidates = (*range(1, 27), *range(43, 57), *range(70, 108), *range(139, 150))
    blocked = set(fixed)
    return [position for position in candidates if position not in blocked]


def _region(position: int) -> str:
    if 27 <= position <= 42:
        return "CDR1"
    if 57 <= position <= 69:
        return "CDR2"
    if 108 <= position <= 138:
        return "CDR3"
    return "FR"


SEQUENCE_SCORE_COLUMNS = {
    "AbNatiV Heavy-Light Score": "joint_score",
    "AbNatiV Pairing Score (%)": "pairing_score",
    "AbNatiV Heavy Score": "vh_score",
    "AbNatiV Light Score": "vl_score",
    "AbNatiV CDR1-Heavy Score": "vh_cdr1_score",
    "AbNatiV CDR2-Heavy Score": "vh_cdr2_score",
    "AbNatiV CDR3-Heavy Score": "vh_cdr3_score",
    "AbNatiV FR-Heavy Score": "vh_fr_score",
    "AbNatiV CDR1-Light Score": "vl_cdr1_score",
    "AbNatiV CDR2-Light Score": "vl_cdr2_score",
    "AbNatiV CDR3-Light Score": "vl_cdr3_score",
    "AbNatiV FR-Light Score": "vl_fr_score",
    "AbNatiV Heavy-Light Percentile": "joint_percentile",
    "AbNatiV CDR1-Heavy Percentile": "vh_cdr1_percentile",
    "AbNatiV CDR2-Heavy Percentile": "vh_cdr2_percentile",
    "AbNatiV CDR3-Heavy Percentile": "vh_cdr3_percentile",
    "AbNatiV FR-Heavy Percentile": "vh_fr_percentile",
    "AbNatiV CDR1-Light Percentile": "vl_cdr1_percentile",
    "AbNatiV CDR2-Light Percentile": "vl_cdr2_percentile",
    "AbNatiV CDR3-Light Percentile": "vl_cdr3_percentile",
    "AbNatiV FR-Light Percentile": "vl_fr_percentile",
}

DISPLACEMENT_COLUMNS = {
    "RMSD (A)\nwith modelled WT\nof scaffold\nupon scaffold-superimposition": "structure_scaffold_rmsd_angstrom",
    "RMSD (A)\nwith modelled WT\nof H-CDR1\nupon scaffold-superimposition": "vh_cdr1_displacement_angstrom",
    "RMSD (A)\nwith modelled WT\nof H-CDR2\nupon scaffold-superimposition": "vh_cdr2_displacement_angstrom",
    "RMSD (A)\nwith modelled WT\nof H-CDR3\nupon scaffold-superimposition": "vh_cdr3_displacement_angstrom",
    "RMSD (A)\nwith modelled WT\nof L-CDR1\nupon scaffold-superimposition": "vl_cdr1_displacement_angstrom",
    "RMSD (A)\nwith modelled WT\nof L-CDR2\nupon scaffold-superimposition": "vl_cdr2_displacement_angstrom",
    "RMSD (A)\nwith modelled WT\nof L-CDR3\nupon scaffold-superimposition": "vl_cdr3_displacement_angstrom",
}


def _python_number(value: Any) -> int | float | None:
    import numpy as np

    if value is None or bool(np.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _normalize_scores(
    identifier: str,
    pair_seed: int,
    mean: Any,
    profile: Any,
    humanizer_mean: Any,
) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    endpoints = ("input", "final")
    if len(mean) != 2 or len(profile) != 2 * 298:
        raise RuntimeError("p-AbNatiV2 returned an unexpected endpoint score shape")
    sequence_rows: list[dict[str, Any]] = []
    for index, endpoint in enumerate(endpoints):
        source = mean.iloc[index]
        row: dict[str, Any] = {
            "id": identifier,
            "endpoint": endpoint,
            "vh": str(source["input_seq_vh"]),
            "vl": str(source["input_seq_vl"]),
            "aligned_vh": str(source["aligned_seq_vh"]),
            "aligned_vl": str(source["aligned_seq_vl"]),
            "pair_seed": pair_seed,
        }
        row.update({
            output: _python_number(source[input_name])
            for input_name, output in SEQUENCE_SCORE_COLUMNS.items()
        })
        displacement_source = humanizer_mean.iloc[index]
        row.update({
            output: _python_number(displacement_source.get(input_name))
            for input_name, output in DISPLACEMENT_COLUMNS.items()
        })
        sequence_rows.append(row)

    profile_rows: list[dict[str, Any]] = []
    for source in profile.to_dict(orient="records"):
        upstream_id = str(source["seq_id"])
        endpoint = next(
            (name for name in endpoints if upstream_id.endswith(f"__{name}")),
            None,
        )
        if endpoint is None:
            raise RuntimeError("p-AbNatiV2 returned an unexpected profile ID")
        chain, position_text = str(source["AHo position"]).split("-", maxsplit=1)
        row = {
            "id": identifier,
            "endpoint": endpoint,
            "chain": "vh" if chain == "H" else "vl",
            "aho_position": int(position_text),
            "observed_aa": str(source["aa"]),
            "observed_residue_score": _python_number(
                source["AbNatiV VPaired2 Residue Score"]
            ),
        }
        row.update({
            f"recon_{symbol if symbol != '-' else 'gap'}": _python_number(
                source[symbol]
            )
            for symbol in RECONSTRUCTION_SYMBOLS
        })
        profile_rows.append(row)
    return sequence_rows, pl.DataFrame(profile_rows)


def _endpoint_mutations(
    identifier: str,
    sequence_scores: list[dict[str, Any]],
    residue_scores: pl.DataFrame,
) -> list[dict[str, Any]]:
    input_row, final_row = sequence_scores
    scores = {
        (row["endpoint"], row["chain"], row["aho_position"]): row[
            "observed_residue_score"
        ]
        for row in residue_scores.iter_rows(named=True)
    }
    mutations: list[dict[str, Any]] = []
    for chain in ("vh", "vl"):
        input_aligned = input_row[f"aligned_{chain}"]
        final_aligned = final_row[f"aligned_{chain}"]
        if len(input_aligned) != 149 or len(final_aligned) != 149:
            raise RuntimeError("p-AbNatiV2 returned a non-AHo endpoint alignment")
        for position, (before, after) in enumerate(
            zip(input_aligned, final_aligned, strict=True), start=1
        ):
            if before == after:
                continue
            mutations.append({
                "id": identifier,
                "chain": chain,
                "aho_position": position,
                "region": _region(position),
                "input_aa": before,
                "final_aa": after,
                "input_residue_score": scores[("input", chain, position)],
                "final_residue_score": scores[("final", chain, position)],
            })
    return mutations


def _humanize_pair(
    *,
    row_number: int,
    row: dict[str, str],
    parameters: _Parameters,
    scratch_root: Path,
) -> dict[str, Any]:
    import pandas as pd
    from abnativ.humanisation.vh_vl_humanisation_functions import (  # type: ignore[ty:unresolved-import]
        abnativ_vh_vl_humanisation_paired,
    )
    from abnativ.model.scoring_functions import (  # type: ignore[ty:unresolved-import]
        abnativ_scoring_paired,
    )

    pair_name = f"{row_number:04d}_{sanitize_filename(row['id'])}"
    pair_seed = _pair_seed(parameters.seed, row["id"], row["vh"], row["vl"])
    _set_seed(pair_seed)
    humanizer_mean = abnativ_vh_vl_humanisation_paired(
        row["vh"],
        row["vl"],
        name_seq=pair_name,
        output_dir=str(scratch_root),
        allowed_user_positions_h=_allowed_positions(
            parameters.mutate_cdrs, parameters.fixed_vh_positions
        ),
        allowed_user_positions_l=_allowed_positions(
            parameters.mutate_cdrs, parameters.fixed_vl_positions
        ),
        threshold_abnativ_score=parameters.residue_score_threshold,
        threshold_rasa_score=parameters.rasa_threshold,
        percentage_pairing_decrease=parameters.max_relative_pairing_score_decrease,
        a=IDENTITY.nativeness_weight,
        b=IDENTITY.pairing_weight,
        forbidden_mut=list(parameters.forbidden_residues),
        verbose=False,
    )
    if humanizer_mean is None or len(humanizer_mean) != 2:
        raise RuntimeError("p-AbNatiV2 did not return two humanization endpoints")
    final = humanizer_mean.iloc[-1]
    final_vh = str(final["input_seq_vh"])
    final_vl = str(final["input_seq_vl"])
    mean, profile = abnativ_scoring_paired(
        pd.DataFrame({
            "ID": [f"{pair_name}__input", f"{pair_name}__final"],
            "vh_seq": [row["vh"], final_vh],
            "vl_seq": [row["vl"], final_vl],
        }),
        batch_size=2,
        mean_score_only=False,
        do_align=True,
        verbose=False,
    )

    sequence_scores, residue_scores = _normalize_scores(
        row["id"], pair_seed, mean, profile, humanizer_mean
    )
    mutations = _endpoint_mutations(row["id"], sequence_scores, residue_scores)
    structure_dir = scratch_root / pair_name / "structures"
    input_pdb = structure_dir / f"{pair_name}_abnativ_wt_abb3_aho.pdb"
    final_pdb = structure_dir / f"{pair_name}_abnativ_hum_abb3_aho.pdb"
    if not input_pdb.is_file() or not final_pdb.is_file():
        raise RuntimeError("p-AbNatiV2 did not produce both endpoint structures")
    import torch  # type: ignore[ty:unresolved-import]

    return {
        "humanized": {"id": row["id"], "vh": final_vh, "vl": final_vl},
        "sequence_scores": sequence_scores,
        "residue_scores": residue_scores.to_dicts(),
        "mutations": mutations,
        "input_pdb": b64encode(input_pdb.read_bytes()).decode("ascii"),
        "final_pdb": b64encode(final_pdb.read_bytes()).decode("ascii"),
        "pair_seed": pair_seed,
        "device": torch.cuda.get_device_name(0),
    }


_asset_manifest: dict[str, object] | None = None


def _assert_assets_once() -> dict[str, object]:
    global _asset_manifest
    if _asset_manifest is None:
        _asset_manifest = assert_pabnativ2_assets(MODEL_ROOT)
    return _asset_manifest


def _write_bundle(
    *,
    run_name: str,
    input_frame: pl.DataFrame,
    pair_results: Sequence[Mapping[str, Any]],
    parameters: _Parameters,
    asset_manifest: dict[str, object],
) -> bytes:
    with TemporaryDirectory(prefix="pabnativ2_bundle_") as temporary:
        root = Path(temporary) / run_name
        root.mkdir()
        input_frame.write_csv(root / "input.csv")
        (root / "input.fasta").write_text(
            "\n".join(
                line
                for row in input_frame.iter_rows(named=True)
                for line in (
                    f">{row['id']}_VH",
                    row["vh"],
                    f">{row['id']}_VL",
                    row["vl"],
                )
            )
            + "\n",
            encoding="utf-8",
        )
        humanized_rows = [result["humanized"] for result in pair_results]
        pl.DataFrame(
            humanized_rows,
            schema={column: pl.String for column in CSV_COLUMNS},
        ).write_csv(root / "humanized.csv")
        (root / "humanized.fasta").write_text(
            "\n".join(
                line
                for row in humanized_rows
                for line in (
                    f">{row['id']}_VH",
                    row["vh"],
                    f">{row['id']}_VL",
                    row["vl"],
                )
            )
            + "\n",
            encoding="utf-8",
        )
        pl.DataFrame([
            row for result in pair_results for row in result["sequence_scores"]
        ]).write_csv(root / "sequence_scores.csv")
        pl.concat(
            [pl.DataFrame(result["residue_scores"]) for result in pair_results],
            how="vertical",
            rechunk=True,
        ).write_parquet(root / "residue_scores.parquet", compression="zstd")
        mutation_schema = {
            "id": pl.String,
            "chain": pl.String,
            "aho_position": pl.Int64,
            "region": pl.String,
            "input_aa": pl.String,
            "final_aa": pl.String,
            "input_residue_score": pl.Float64,
            "final_residue_score": pl.Float64,
        }
        pl.DataFrame(
            [row for result in pair_results for row in result["mutations"]],
            schema=mutation_schema,
        ).write_parquet(root / "mutations.parquet", compression="zstd")
        for row_number, result in enumerate(pair_results, start=1):
            humanized = result["humanized"]
            identifier = sanitize_filename(humanized["id"])
            destination = root / "structures" / f"{row_number:04d}_{identifier}"
            destination.mkdir(parents=True)
            (destination / "input.pdb").write_bytes(
                b64decode(result["input_pdb"], validate=True)
            )
            (destination / "final.pdb").write_bytes(
                b64decode(result["final_pdb"], validate=True)
            )

        files = sorted(path for path in root.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 2,
            "run_name": run_name,
            "pair_count": input_frame.height,
            "input_sha256": hashlib.sha256(
                input_frame.write_csv().encode()
            ).hexdigest(),
            "parameters": {
                "mutate_cdrs": parameters.mutate_cdrs,
                "fixed_vh_positions": parameters.fixed_vh_positions,
                "fixed_vl_positions": parameters.fixed_vl_positions,
                "residue_score_threshold": parameters.residue_score_threshold,
                "rasa_threshold": parameters.rasa_threshold,
                "max_relative_pairing_score_decrease": (
                    parameters.max_relative_pairing_score_decrease
                ),
                "forbidden_residues": parameters.forbidden_residues,
                "seed": parameters.seed,
                "pair_seeds": {
                    result["humanized"]["id"]: result["pair_seed"]
                    for result in pair_results
                },
                "numbering_scheme": "aho",
                "rasa_structure_count": IDENTITY.rasa_structure_count,
                "pssm_frequency_cutoff": IDENTITY.pssm_frequency_cutoff,
                "nativeness_weight": IDENTITY.nativeness_weight,
                "pairing_weight": IDENTITY.pairing_weight,
            },
            "scientific_identity": {
                "runtime": SCIENTIFIC_RUNTIME_IDENTITY,
                "wrapper_protocol": _WRAPPER_PROTOCOL_VERSION,
                "patch_protocol": _PATCH_PROTOCOL_VERSION,
                "paired_model_md5": PAIRED_MODEL.md5_hex,
                "structure_archive_md5": STRUCTURE_MODEL_ARCHIVE.md5_hex,
                "pssm_sha256": PSSM_SHA256,
                "staged_assets": asset_manifest,
            },
            "protocol": {
                "equivalence_target": "AbNatiV 2.0.8 source",
                "rasa_structure_count": IDENTITY.rasa_structure_count,
                "paper_rasa_structure_count": 10,
                "pairing_score_units": "fraction",
                "pairing_score_interpretation": (
                    "model score against synthetic pairing negatives; not a "
                    "calibrated probability of physical assembly"
                ),
            },
            "telemetry": {
                "accelerators": sorted({
                    str(result["device"]) for result in pair_results
                })
            },
            "warnings": [
                "Research use only; paired humanization has not been prospectively "
                "validated to preserve binding or developability."
            ],
            "files": {
                path.relative_to(root).as_posix(): {
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in files
            },
        }
        (root / "manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        archive = package_outputs(root, num_threads=2)
        if len(archive) > MAX_RESULT_BYTES:
            raise ValueError("p-AbNatiV2 result archive exceeds the byte limit")
        return archive


def _run_pabnativ2_pair(
    *,
    pair: dict[str, str],
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_pairing_score_decrease: float = 0.10,
    forbidden_residues: str = "C,M",
    seed: int = 0,
) -> AppRunResult:
    if (
        not isinstance(pair, dict)
        or set(pair) != set(CSV_COLUMNS)
        or not all(isinstance(pair[column], str) for column in CSV_COLUMNS)
    ):
        raise ValueError("p-AbNatiV2 pair must contain exactly id, vh, and vl")
    parameters = _validate_parameters(
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_pairing_score_decrease=max_relative_pairing_score_decrease,
        forbidden_residues=forbidden_residues,
        seed=seed,
    )
    input_frame = parse_pabnativ2_csv(
        pl.DataFrame([pair]).select(CSV_COLUMNS).write_csv().encode()
    )
    asset_manifest = _assert_assets_once()
    _validate_antibody_chains(input_frame)
    with TemporaryDirectory(prefix="pabnativ2_run_") as scratch:
        result = _humanize_pair(
            row_number=1,
            row=input_frame.row(0, named=True),
            parameters=parameters,
            scratch_root=Path(scratch),
        )
    result["asset_manifest"] = asset_manifest
    content = orjson.dumps({"schema_version": 1, "pair_result": result})
    if len(content) > MAX_PAIR_RESULT_BYTES:
        raise ValueError("p-AbNatiV2 pair result exceeds the byte limit")
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="pabnativ2_pair_result",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=content,
                    filename="pabnativ2-pair.json",
                    media_type="application/json",
                ),
                metadata={"pair_id": pair["id"]},
            )
        ],
        metrics={"pair_count": 1, "mutation_count": len(result["mutations"])},
    )


def _pabnativ2_pair_failure(pair_id: str, reason: str) -> AppRunResult:
    return AppRunResult(
        status=AppRunStatus.FAILED,
        warnings=[f"{pair_id}: {reason}"],
        metrics={"pair_count": 1, "mutation_count": 0},
    )


def _run_pabnativ2_pair_in_subprocess(
    payload: tuple[dict[str, str], dict[str, Any]],
) -> str:
    """Run one isolated pair and return a process-safe result encoding."""
    pair, parameters = payload
    try:
        return _run_pabnativ2_pair(pair=pair, **parameters).model_dump_json()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return _pabnativ2_pair_failure(
            pair["id"], f"{type(exc).__name__}: {exc}"
        ).model_dump_json()


def _write_pabnativ2_pair_result(
    payload: tuple[dict[str, str], dict[str, Any]],
    output_path: str,
) -> None:
    """Write one pair outcome outside multiprocessing transport state."""
    Path(output_path).write_text(
        _run_pabnativ2_pair_in_subprocess(payload), encoding="utf-8"
    )


def _run_pabnativ2_batch(
    *,
    pairs: list[dict[str, str]],
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_pairing_score_decrease: float = 0.10,
    forbidden_residues: str = "C,M",
    seed: int = 0,
) -> dict[str, AppRunResult]:
    """Humanize at most four pairs in isolated processes on one A10G."""
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= PAIRS_PER_GPU_CALL:
        raise ValueError("p-AbNatiV2 GPU batch must contain one to four pairs")
    if any(
        not isinstance(pair, dict)
        or set(pair) != set(CSV_COLUMNS)
        or not all(isinstance(pair[column], str) for column in CSV_COLUMNS)
        for pair in pairs
    ):
        raise ValueError("Each p-AbNatiV2 pair must contain exactly id, vh, and vl")
    frame = parse_pabnativ2_csv(
        pl.DataFrame(pairs).select(CSV_COLUMNS).write_csv().encode()
    )
    _validate_antibody_chains(frame)
    normalized_pairs = frame.to_dicts()
    parameters = {
        "mutate_cdrs": mutate_cdrs,
        "fixed_vh_positions": fixed_vh_positions,
        "fixed_vl_positions": fixed_vl_positions,
        "residue_score_threshold": residue_score_threshold,
        "rasa_threshold": rasa_threshold,
        "max_relative_pairing_score_decrease": max_relative_pairing_score_decrease,
        "forbidden_residues": forbidden_residues,
        "seed": seed,
    }
    from multiprocessing import get_context

    payloads = [(pair, parameters) for pair in normalized_pairs]
    context = get_context("spawn")
    results: dict[str, AppRunResult] = {}
    with TemporaryDirectory(prefix="pabnativ2_batch_") as temporary:
        workers = []
        for index, (pair, payload) in enumerate(
            zip(normalized_pairs, payloads, strict=True)
        ):
            output_path = Path(temporary) / f"{index}.json"
            process = context.Process(
                target=_write_pabnativ2_pair_result,
                args=(payload, str(output_path)),
            )
            try:
                process.start()
            except Exception as exc:
                import traceback

                traceback.print_exc()
                results[pair["id"]] = _pabnativ2_pair_failure(
                    pair["id"], f"{type(exc).__name__}: {exc}"
                )
            else:
                workers.append((pair, output_path, process))

        for pair, output_path, process in workers:
            process.join()
            if process.exitcode != 0:
                results[pair["id"]] = _pabnativ2_pair_failure(
                    pair["id"],
                    f"ProcessExit: child exited with code {process.exitcode}",
                )
                continue
            try:
                if output_path.stat().st_size > 2 * MAX_PAIR_RESULT_BYTES:
                    raise ValueError("child result exceeds the byte limit")
                results[pair["id"]] = AppRunResult.model_validate_json(
                    output_path.read_bytes()
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                results[pair["id"]] = _pabnativ2_pair_failure(
                    pair["id"], f"{type(exc).__name__}: {exc}"
                )
    return {pair["id"]: results[pair["id"]] for pair in normalized_pairs}


def _aggregate_pabnativ2_results(
    *,
    run_name: str,
    csv_bytes: bytes,
    pair_results: Sequence[Mapping[str, Any]],
    parameters: dict[str, Any],
) -> tuple[bytes, dict[str, int]]:
    input_frame = parse_pabnativ2_csv(csv_bytes)
    if [result["humanized"]["id"] for result in pair_results] != input_frame[
        "id"
    ].to_list():
        raise ValueError("p-AbNatiV2 pair results do not match input order")
    normalized = _validate_parameters(**parameters)
    asset_manifests = {
        orjson.dumps(result["asset_manifest"], option=orjson.OPT_SORT_KEYS)
        for result in pair_results
    }
    if len(asset_manifests) != 1:
        raise ValueError("p-AbNatiV2 pair results used different model assets")
    archive = _write_bundle(
        run_name=run_name,
        input_frame=input_frame,
        pair_results=pair_results,
        parameters=normalized,
        asset_manifest=orjson.loads(asset_manifests.pop()),
    )
    return archive, {
        "pair_count": len(pair_results),
        "mutation_count": sum(len(result["mutations"]) for result in pair_results),
    }


def _worker_kwargs() -> dict[str, Any]:
    return {
        "memory": (1024, 32768),
        "timeout": CONF.timeout,
        "volumes": CONF.mounts(model_volume=True),
    }


def _batch_worker_kwargs() -> dict[str, Any]:
    return {
        "memory": (512, 65536),
        "timeout": CONF.timeout,
        "volumes": CONF.mounts(model_volume=True),
    }


@app.function(cpu=(0.125, 8.125), gpu="A10G", **_worker_kwargs())
def pabnativ2_humanize_pair(
    pair: dict[str, str],
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_pairing_score_decrease: float = 0.10,
    forbidden_residues: str = "C,M",
    seed: int = 0,
) -> AppRunResult:
    """Humanize one complete antibody pair on the selected A10G worker."""
    return _run_pabnativ2_pair(
        pair=pair,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_pairing_score_decrease=max_relative_pairing_score_decrease,
        forbidden_residues=forbidden_residues,
        seed=seed,
    )


@app.function(cpu=(0.125, 8.125), gpu="A10G", **_batch_worker_kwargs())
def pabnativ2_humanize_batch(
    pairs: list[dict[str, str]],
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_pairing_score_decrease: float = 0.10,
    forbidden_residues: str = "C,M",
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Humanize one fixed batch of at most four pairs on one A10G."""
    results = _run_pabnativ2_batch(
        pairs=pairs,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_pairing_score_decrease=max_relative_pairing_score_decrease,
        forbidden_residues=forbidden_residues,
        seed=seed,
    )
    return {
        pair_id: result.model_dump(mode="json") for pair_id, result in results.items()
    }


@app.function(
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(model_volume=True, model_ro=False),
)
def stage_pabnativ2_models() -> dict[str, object]:
    """Explicitly download, validate, and commit the pinned checkpoints."""
    manifest = stage_pabnativ2_assets(MODEL_ROOT)
    MODEL_VOLUME.commit()
    return manifest


@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=_TIMEOUT_SECONDS,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with the p-AbNatiV2 workers."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh durable state before accepting lifecycle methods."""
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
        self,
        *,
        development: bool | None = None,
    ) -> PAbNatiV2ExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: PAbNatiV2ExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_volume_name=CONF.output_volume_name,
                provider_driver=_coordinator_modal_driver(development=selected_mode),
                source_commit=IDENTITY.abnativ_commit,
                paired_model_md5=PAIRED_MODEL.md5_hex,
                structure_archive_md5=STRUCTURE_MODEL_ARCHIVE.md5_hex,
                runtime_identity=SCIENTIFIC_RUNTIME_IDENTITY,
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve the deployed worker or current-source development handle."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "pabnativ2_humanize_pair": pabnativ2_humanize_pair,
            "pabnativ2_humanize_batch": pabnativ2_humanize_batch,
        },
        workload_name=CONF.name,
    )


@app.local_entrypoint()
def submit_pabnativ2_task(
    input_csv: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    mutate_cdrs: bool = False,
    fixed_vh_positions: str = "",
    fixed_vl_positions: str = "",
    residue_score_threshold: float = 0.98,
    rasa_threshold: float = 0.15,
    max_relative_pairing_score_decrease: float = 0.10,
    forbidden_residues: str = "C,M",
    seed: int = 0,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Humanize paired VH-VL sequences and save the result archive locally.

    Args:
        input_csv: UTF-8 CSV with exactly ``id,vh,vl`` columns.
        output_dir: Local directory for the downloaded result archive.
        run_name: Optional safe display name for this run.
        mutate_cdrs: Permit CDR mutations in addition to framework mutations.
        fixed_vh_positions: Comma-separated protected heavy-chain AHo positions.
        fixed_vl_positions: Comma-separated protected light-chain AHo positions.
        residue_score_threshold: Residue score at or below which positions take
            the score-liability route; framework PSSM mismatches are an independent
            liability route in upstream p-AbNatiV2.
        rasa_threshold: Minimum relative solvent accessibility for mutation.
        max_relative_pairing_score_decrease: Maximum accepted pairing-score loss.
        forbidden_residues: Comma-separated amino acids forbidden as replacements.
        seed: Unsigned 32-bit base seed used to derive stable per-pair seeds.
        max_containers: Run-wide provider-call ceiling injected by the CLI.
        max_gpu_containers: Run-wide GPU-call ceiling injected by the CLI.
        use_deployed_coordinator: Use the pinned deployed coordinator.
        deployment_environment: Modal environment containing the deployment.
        deployment_name: Exact deployed app name.
        deployment_version: Exact deployed app version.
        restart_from: Optional predecessor Execution Run UUID.
    """
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if input_path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV exceeds {MAX_INPUT_BYTES} bytes")
    csv_bytes = input_path.read_bytes()
    input_frame = parse_pabnativ2_csv(csv_bytes)
    _validate_parameters(
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_pairing_score_decrease=max_relative_pairing_score_decrease,
        forbidden_residues=forbidden_residues,
        seed=seed,
    )
    selected_run_name = sanitize_filename(run_name or input_path.stem)
    provider_call_count = (
        input_frame.height + PAIRS_PER_GPU_CALL - 1
    ) // PAIRS_PER_GPU_CALL
    total_limit, gpu_limit = resolve_provider_call_limits(
        default_max_containers=min(provider_call_count, _DEFAULT_MAX_GPU_CONTAINERS),
        default_max_gpu_containers=min(
            provider_call_count, _DEFAULT_MAX_GPU_CONTAINERS
        ),
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
    )
    request = PAbNatiV2ExecutionRequest(
        run_name=selected_run_name,
        csv_bytes=csv_bytes,
        mutate_cdrs=mutate_cdrs,
        fixed_vh_positions=fixed_vh_positions,
        fixed_vl_positions=fixed_vl_positions,
        residue_score_threshold=residue_score_threshold,
        rasa_threshold=rasa_threshold,
        max_relative_pairing_score_decrease=max_relative_pairing_score_decrease,
        forbidden_residues=forbidden_residues,
        seed=seed,
        source_commit=IDENTITY.abnativ_commit,
        paired_model_md5=PAIRED_MODEL.md5_hex,
        structure_archive_md5=STRUCTURE_MODEL_ARCHIVE.md5_hex,
        runtime_identity=SCIENTIFIC_RUNTIME_IDENTITY,
        max_active_provider_calls=total_limit,
        max_active_gpu_provider_calls=gpu_limit,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
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
    output = next(
        item for item in result.outputs if item.name == "pabnativ2_humanization"
    )
    if not isinstance(output.storage, InlineBytes):
        raise TypeError("p-AbNatiV2 output must be inline .tar.zst bytes")
    local_output_dir = Path.cwd() if output_dir is None else Path(output_dir)
    output_path = build_local_output_path(
        local_output_dir,
        run_name=selected_run_name,
        suffix="pabnativ2",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output.storage.data)
    print(f"p-AbNatiV2 results saved to: {output_path.resolve()}")
