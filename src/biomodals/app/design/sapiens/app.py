"""Sapiens source repo: <https://github.com/Merck/Sapiens>.

Humanize complete VH-VL pairs with the BioPhi Sapiens argmax procedure. Input
is a UTF-8 CSV with exactly ``id,vh,vl`` columns. The result is one inline
``.tar.zst`` bundle containing paired sequences, mutation provenance, and the
20-amino-acid Sapiens score matrices for the input and final sequences.

Parental CDRs are preserved by default. Numbering and CDR definitions are
independent options so future numbering schemes can be added without changing
the result schema.
"""

# ruff: noqa: PLC0415

import hashlib
import time
from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.app.design.sapiens.execution import (
    SapiensExecutionCoordinator,
    SapiensExecutionRequest,
    load_execution_request,
    result_from_overview,
    stage_execution_request,
)
from biomodals.app.design.sapiens.models import (
    IDENTITY,
    MODEL_ROOT,
    RUNTIME_IDENTITY,
    download_sapiens_models,
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
from biomodals.helper.shell import package_outputs, sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
CSV_COLUMNS = ("id", "vh", "vl")
NUMBERING_SCHEMES = frozenset({"kabat", "chothia", "imgt"})
CDR_DEFINITIONS = frozenset({"kabat", "chothia", "imgt", "north"})
MAX_PAIRS = 1000
MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_ID_LENGTH = 200
MAX_VH_LENGTH = 142
MAX_VL_LENGTH = 126
_COORDINATOR_TIMEOUT_SECONDS = 24 * 60 * 60
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8

CONF = AppConfig(
    tags={"group": Path(__file__).parent.parent.name},
    name="Sapiens",
    repo_url="https://github.com/Merck/Sapiens",
    repo_commit_hash="3d676ecde0b6fc113d3f9c5bcb3721a7d70a85b7",
    package_name="sapiens",
    version=IDENTITY.sapiens_version,
    python_version="3.12",
    timeout=_COORDINATOR_TIMEOUT_SECONDS,
)


runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .micromamba_install(
        [
            f"abnumber=={IDENTITY.abnumber_version}",
            f"anarci=={IDENTITY.anarci_version}",
            f"hmmer=={IDENTITY.hmmer_version}",
            f"biopython=={IDENTITY.biopython_version}",
            f"numpy=={IDENTITY.numpy_version}",
            f"pandas=={IDENTITY.pandas_version}",
        ],
        channels=["bioconda", "conda-forge"],
    )
    .uv_pip_install(
        f"torch=={IDENTITY.torch_version}",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .uv_pip_install(
        f"{CONF.package_name}=={CONF.version}",
        f"transformers=={IDENTITY.transformers_version}",
        f"huggingface-hub=={IDENTITY.huggingface_hub_version}",
        f"tokenizers=={IDENTITY.tokenizers_version}",
        f"safetensors=={IDENTITY.safetensors_version}",
    )
    .add_local_python_source(
        "biomodals.app.design.sapiens.models",
        copy=True,
    )
    .run_function(download_sapiens_models)
    .env({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.sapiens.execution")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)

EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_sapiens_task"})


def _validate_parameters(
    *,
    iterations: int,
    numbering_scheme: str,
    cdr_definition: str,
    mutate_cdrs: bool,
) -> None:
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("iterations must be between 1 and 5")
    if not isinstance(numbering_scheme, str) or (
        numbering_scheme not in NUMBERING_SCHEMES
    ):
        choices = ", ".join(sorted(NUMBERING_SCHEMES))
        raise ValueError(f"numbering_scheme must be one of: {choices}")
    if not isinstance(cdr_definition, str) or cdr_definition not in CDR_DEFINITIONS:
        choices = ", ".join(sorted(CDR_DEFINITIONS))
        raise ValueError(f"cdr_definition must be one of: {choices}")
    if type(mutate_cdrs) is not bool:
        raise ValueError("mutate_cdrs must be a boolean")


def parse_sapiens_csv(content: bytes) -> pl.DataFrame:
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
    canonical = set(AMINO_ACIDS)
    for row_number, row in enumerate(frame.iter_rows(named=True), start=2):
        identifier = row["id"]
        vh = row["vh"]
        vl = row["vl"]
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
        for column, sequence, max_length in (
            ("vh", vh, MAX_VH_LENGTH),
            ("vl", vl, MAX_VL_LENGTH),
        ):
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Row {row_number}: {column} must be non-empty")
            if len(sequence) > max_length:
                raise ValueError(
                    f"Row {row_number}: {column} exceeds {max_length} residues"
                )
            invalid = sorted(set(sequence) - canonical)
            if invalid:
                raise ValueError(
                    f"Row {row_number}: {column} contains non-canonical or "
                    f"lowercase residues: {''.join(invalid)}"
                )
    return frame


def _score_frame(
    *,
    identifier: str,
    chain_label: str,
    endpoint: str,
    chain: Any,
    scores: Any,
) -> pl.DataFrame:
    values = scores.loc[:, list(AMINO_ACIDS)].to_numpy(dtype="float32")
    positions = list(chain.positions)
    if values.shape != (len(positions), len(AMINO_ACIDS)):
        raise ValueError("Sapiens returned an unexpected score matrix shape")
    metadata = pl.DataFrame({
        "id": [identifier] * len(positions),
        "chain": [chain_label] * len(positions),
        "endpoint": [endpoint] * len(positions),
        "sequence_index": list(range(1, len(positions) + 1)),
        "numbered_position": [position.format() for position in positions],
        "region": [position.get_region() for position in positions],
        "observed_aa": list(chain.seq),
        "argmax_aa": list(scores.idxmax(axis=1).values),
    })
    return metadata.hstack(pl.from_numpy(values, schema=list(AMINO_ACIDS)))


def _predict_scores(chain: Any, model_root: Path) -> Any:
    # Import the upstream top-level package, not this Biomodals package.
    sapiens_module = cast(Any, import_module("sapiens"))

    model_dir = model_root / ("vh" if chain.chain_type == "H" else "vl")
    return sapiens_module.predict_scores(
        seq=chain.seq,
        chain_type=chain.chain_type,
        checkpoint_path=str(model_dir),
        tokenizer_path=str(model_root / "tokenizer"),
    )


def _humanize_chain(
    *,
    identifier: str,
    chain_label: str,
    sequence: str,
    iterations: int,
    numbering_scheme: str,
    cdr_definition: str,
    mutate_cdrs: bool,
    model_root: Path,
) -> tuple[Any, list[pl.DataFrame], list[dict[str, Any]]]:
    from abnumber import Chain  # type: ignore[ty:unresolved-import]

    parental = Chain(
        sequence,
        name=identifier,
        scheme=numbering_scheme,
        cdr_definition=cdr_definition,
    )
    if parental.seq != sequence:
        raise ValueError(
            f"{identifier}: {chain_label} contains residues outside the numbered "
            "variable region"
        )
    if chain_label == "vh" and not parental.is_heavy_chain():
        raise ValueError(f"{identifier}: vh was not identified as a heavy chain")
    if chain_label == "vl" and not parental.is_light_chain():
        raise ValueError(f"{identifier}: vl was not identified as a light chain")

    current = parental.clone()
    input_scores = _predict_scores(current, model_root)
    score_frames = [
        _score_frame(
            identifier=identifier,
            chain_label=chain_label,
            endpoint="input",
            chain=parental,
            scores=input_scores,
        )
    ]
    mutation_rows: list[dict[str, Any]] = []
    for iteration in range(1, iterations + 1):
        scores = (
            input_scores if iteration == 1 else _predict_scores(current, model_root)
        )
        proposed = "".join(scores.idxmax(axis=1).values)
        next_chain = parental.clone(proposed)
        if not mutate_cdrs:
            next_chain = parental.graft_cdrs_onto(
                next_chain,
                backmutate_vernier=False,
            )
        for sequence_index, (position, before, after) in enumerate(
            zip(current.positions, current.seq, next_chain.seq, strict=True),
            start=1,
        ):
            if before != after:
                mutation_rows.append({
                    "id": identifier,
                    "chain": chain_label,
                    "iteration": iteration,
                    "sequence_index": sequence_index,
                    "numbered_position": position.format(),
                    "region": position.get_region(),
                    "from_aa": before,
                    "to_aa": after,
                })
        current = next_chain

    final_scores = _predict_scores(current, model_root)
    score_frames.append(
        _score_frame(
            identifier=identifier,
            chain_label=chain_label,
            endpoint="final",
            chain=current,
            scores=final_scores,
        )
    )
    return current, score_frames, mutation_rows


def _write_result_bundle(
    *,
    run_name: str,
    input_frame: pl.DataFrame,
    humanized_rows: list[dict[str, str]],
    score_frames: list[pl.DataFrame],
    mutation_rows: list[dict[str, Any]],
    iterations: int,
    numbering_scheme: str,
    cdr_definition: str,
    mutate_cdrs: bool,
) -> bytes:
    with TemporaryDirectory(prefix="sapiens_") as temporary:
        result_dir = Path(temporary) / run_name
        result_dir.mkdir()
        input_frame.write_csv(result_dir / "input.csv")
        humanized = pl.DataFrame(
            humanized_rows,
            schema={column: pl.String for column in CSV_COLUMNS},
        )
        humanized.write_csv(result_dir / "humanized.csv")
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
            "\n".join(fasta_lines) + "\n",
            encoding="utf-8",
        )
        mutation_schema = {
            "id": pl.String,
            "chain": pl.String,
            "iteration": pl.Int64,
            "sequence_index": pl.Int64,
            "numbered_position": pl.String,
            "region": pl.String,
            "from_aa": pl.String,
            "to_aa": pl.String,
        }
        pl.DataFrame(mutation_rows, schema=mutation_schema).write_csv(
            result_dir / "mutation_history.csv"
        )
        pl.concat(score_frames, how="vertical", rechunk=True).write_parquet(
            result_dir / "residue_scores.parquet",
            compression="zstd",
        )

        artifact_files = sorted(result_dir.iterdir())
        manifest = {
            "schema_version": 1,
            "run_name": run_name,
            "pair_count": input_frame.height,
            "parameters": {
                "iterations": iterations,
                "numbering_scheme": numbering_scheme,
                "cdr_definition": cdr_definition,
                "mutate_cdrs": mutate_cdrs,
            },
            "scientific_identity": {
                "sapiens_package": CONF.version,
                "sapiens_commit": CONF.repo_commit_hash,
                "abnumber": IDENTITY.abnumber_version,
                "anarci": IDENTITY.anarci_version,
                "torch": IDENTITY.torch_version,
                "transformers": IDENTITY.transformers_version,
                "huggingface_hub": IDENTITY.huggingface_hub_version,
                "tokenizers": IDENTITY.tokenizers_version,
                "safetensors": IDENTITY.safetensors_version,
                "numpy": IDENTITY.numpy_version,
                "pandas": IDENTITY.pandas_version,
                "biopython": IDENTITY.biopython_version,
                "hmmer": IDENTITY.hmmer_version,
                "models": {
                    "vh": {
                        "repo": IDENTITY.vh_repo,
                        "revision": IDENTITY.vh_revision,
                    },
                    "vl": {
                        "repo": IDENTITY.vl_repo,
                        "revision": IDENTITY.vl_revision,
                    },
                    "tokenizer": {
                        "repo": IDENTITY.tokenizer_repo,
                        "revision": IDENTITY.tokenizer_revision,
                    },
                },
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
        return package_outputs(result_dir, num_threads=2)


def _run_sapiens_humanization(
    *,
    run_name: str,
    csv_bytes: bytes,
    iterations: int,
    numbering_scheme: str,
    cdr_definition: str,
    mutate_cdrs: bool,
    model_root: Path = MODEL_ROOT,
) -> tuple[bytes, int, int, float]:
    """Validate, humanize sequentially, and package one paired batch."""
    _validate_parameters(
        iterations=iterations,
        numbering_scheme=numbering_scheme,
        cdr_definition=cdr_definition,
        mutate_cdrs=mutate_cdrs,
    )
    input_frame = parse_sapiens_csv(csv_bytes)
    import torch  # type: ignore[ty:unresolved-import]

    torch.set_num_threads(2)
    started_at = time.perf_counter()
    humanized_rows: list[dict[str, str]] = []
    score_frames: list[pl.DataFrame] = []
    mutation_rows: list[dict[str, Any]] = []
    for row in input_frame.iter_rows(named=True):
        humanized_row = {"id": row["id"]}
        for chain_label in ("vh", "vl"):
            chain, chain_scores, chain_mutations = _humanize_chain(
                identifier=row["id"],
                chain_label=chain_label,
                sequence=row[chain_label],
                iterations=iterations,
                numbering_scheme=numbering_scheme,
                cdr_definition=cdr_definition,
                mutate_cdrs=mutate_cdrs,
                model_root=model_root,
            )
            humanized_row[chain_label] = chain.seq
            score_frames.extend(chain_scores)
            mutation_rows.extend(chain_mutations)
        humanized_rows.append(humanized_row)
    archive = _write_result_bundle(
        run_name=run_name,
        input_frame=input_frame,
        humanized_rows=humanized_rows,
        score_frames=score_frames,
        mutation_rows=mutation_rows,
        iterations=iterations,
        numbering_scheme=numbering_scheme,
        cdr_definition=cdr_definition,
        mutate_cdrs=mutate_cdrs,
    )
    elapsed_seconds = time.perf_counter() - started_at
    return archive, input_frame.height, len(mutation_rows), elapsed_seconds


@app.function(cpu=2, memory=2048, timeout=CONF.timeout)
def sapiens_humanize(
    run_name: str,
    csv_bytes: bytes,
    iterations: int = 1,
    numbering_scheme: str = "kabat",
    cdr_definition: str = "kabat",
    mutate_cdrs: bool = False,
) -> AppRunResult:
    """Humanize a validated CSV batch and return an inline result archive.

    Args:
        run_name: Safe logical name used for the archive and its root directory.
        csv_bytes: UTF-8 CSV bytes with exactly ``id,vh,vl`` columns.
        iterations: Number of BioPhi-style argmax passes, from 1 through 5.
        numbering_scheme: Antibody numbering scheme: kabat, chothia, or imgt.
        cdr_definition: CDR boundary definition: kabat, chothia, imgt, or north.
        mutate_cdrs: Whether Sapiens may change residues inside parental CDRs.

    Returns:
        A successful app result containing one inline ``.tar.zst`` archive.
    """
    run_name = sanitize_filename(run_name)
    archive, pair_count, mutation_count, elapsed_seconds = _run_sapiens_humanization(
        run_name=run_name,
        csv_bytes=csv_bytes,
        iterations=iterations,
        numbering_scheme=numbering_scheme,
        cdr_definition=cdr_definition,
        mutate_cdrs=mutate_cdrs,
    )
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="sapiens_humanization",
                kind=ArtifactKind.ARCHIVE,
                storage=InlineBytes(
                    data=archive,
                    filename=f"{run_name}_sapiens.tar.zst",
                    media_type=ZSTD_MEDIA_TYPE,
                ),
                metadata={
                    "archive_format": "tar.zst",
                    "run_name": run_name,
                    "pair_count": pair_count,
                    "score_endpoints": ["input", "final"],
                },
            )
        ],
        metrics={
            "pair_count": pair_count,
            "mutation_count": mutation_count,
            "iterations": iterations,
            "elapsed_seconds": elapsed_seconds,
        },
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
    """Run-scoped single writer deployed with the Sapiens worker."""

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
        self,
        *,
        development: bool | None = None,
    ) -> SapiensExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: SapiensExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_volume_name=CONF.output_volume_name,
                provider_driver=_coordinator_modal_driver(development=selected_mode),
                app_version=CONF.repo_commit_hash or CONF.version or "unknown",
                model_vh_revision=IDENTITY.vh_revision,
                model_vl_revision=IDENTITY.vl_revision,
                tokenizer_revision=IDENTITY.tokenizer_revision,
                runtime_identity=RUNTIME_IDENTITY,
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve the deployed worker or current-source development handle."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {"sapiens_humanize": sapiens_humanize},
        workload_name=CONF.name,
    )


@app.local_entrypoint()
def submit_sapiens_task(
    input_csv: str,
    output_dir: str | None = None,
    run_name: str | None = None,
    iterations: int = 1,
    numbering_scheme: str = "kabat",
    cdr_definition: str = "kabat",
    mutate_cdrs: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Humanize paired variable regions and save the result archive locally.

    Args:
        input_csv: UTF-8 CSV path with exactly ``id,vh,vl`` columns.
        output_dir: Local directory for the result; defaults to the current directory.
        run_name: Logical run name; defaults to the input file stem.
        iterations: Number of BioPhi-style argmax passes, from 1 through 5.
        numbering_scheme: Antibody numbering scheme: kabat, chothia, or imgt.
        cdr_definition: CDR boundary definition: kabat, chothia, imgt, or north.
        mutate_cdrs: Allow mutations inside parental CDRs; disabled by default.
        use_deployed_coordinator: Target the exact deployed coordinator.
        deployment_environment: Modal environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor execution run ID for a successor run.
    """
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if input_path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input CSV exceeds {MAX_INPUT_BYTES} bytes")
    csv_bytes = input_path.read_bytes()
    parse_sapiens_csv(csv_bytes)
    _validate_parameters(
        iterations=iterations,
        numbering_scheme=numbering_scheme,
        cdr_definition=cdr_definition,
        mutate_cdrs=mutate_cdrs,
    )
    selected_run_name = sanitize_filename(run_name or input_path.stem)
    request = SapiensExecutionRequest(
        run_name=selected_run_name,
        csv_bytes=csv_bytes,
        iterations=iterations,
        numbering_scheme=numbering_scheme,
        cdr_definition=cdr_definition,
        mutate_cdrs=mutate_cdrs,
        app_version=CONF.repo_commit_hash or CONF.version or "unknown",
        model_vh_revision=IDENTITY.vh_revision,
        model_vl_revision=IDENTITY.vl_revision,
        tokenizer_revision=IDENTITY.tokenizer_revision,
        runtime_identity=RUNTIME_IDENTITY,
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
    elapsed_seconds = result.metrics.get("elapsed_seconds")
    if isinstance(elapsed_seconds, int | float):
        print(f"Sapiens worker time: {elapsed_seconds:.3f} seconds")
    output = next(
        item for item in result.outputs if item.name == "sapiens_humanization"
    )
    if not isinstance(output.storage, InlineBytes):
        raise TypeError("Sapiens output must be inline .tar.zst bytes")
    local_output_dir = (
        Path.cwd() if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    local_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = local_output_dir / output.storage.filename
    output_path.write_bytes(output.storage.data)
    print(f"Sapiens results saved to: {output_path}")
