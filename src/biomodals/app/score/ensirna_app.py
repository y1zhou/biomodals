"""ENsiRNA source repo: <https://github.com/tanwenchong/ENsiRNA>.

ENsiRNA designs siRNA candidates from an mRNA FASTA file. This wrapper uses the
upstream Linux runtime requirements, builds on the Biomodals Rosetta base image,
and runs the documented `design.sh` data-prep and inference steps as separate
CPU and GPU Modal functions.

## Additional notes

The upstream Linux instructions require Rosetta for PDB generation. The runtime
uses the same public Rosetta base image as `rosetta_app.py`; commercial use may
require a separate Rosetta license. Model weights and checkpoints are stored in
the standard Biomodals model volume.

FASTA record names are sanitized before reaching upstream shell and path logic.
For candidates whose RNAplex-derived positional arrays have the wrong length,
the wrapper pads or truncates those arrays so the candidate can be scored;
unpatched upstream ENsiRNA drops those candidates. Candidates with correctly
sized arrays retain their upstream features.

## Outputs

Results are saved locally as `<run-name>.xlsx`, containing the upstream
`mrna_result.xlsx` workbook.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import modal

from biomodals.app.config import AppConfig
from biomodals.app.score.ensirna_execution import (
    EnsirnaExecutionCoordinator,
    EnsirnaExecutionRequest,
    EnsirnaPdbChunkSpec,
    EnsirnaPreparationPlan,
    load_execution_request,
    stage_execution_request,
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
)
from biomodals.execution.modal import (
    execution_coordinator_handle as _execution_coordinator_handle,
)
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.app_execution import stage_execution_launch
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import build_local_output_path, resolve_local_output_dir
from biomodals.helper.shell import run_command, sanitize_filename
from biomodals.helper.web import download_files

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="ENsiRNA",
    repo_url="https://github.com/tanwenchong/ENsiRNA",
    repo_commit_hash="028824341635903f3c661f5d1cc737de106493d5",
    package_name="ensirna",
    version="3",
    python_version="3.10",
    cuda_version="cu118",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "7200")),
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Container for ENsiRNA-specific paths, dependencies, and patches."""

    conda_env_name: str = "base"
    mamba_root: str = "/root/micromamba"
    ensirna_dir: Path = CONF.git_clone_dir / "ENsiRNA"
    input_fasta_name: str = "mrna.fasta"
    input_stem: str = "mrna"
    prepared_marker_name: str = "prepared.json"
    candidate_csv_marker_name: str = "candidates.json"
    preprocess_shard_marker_name: str = "sealed.json"
    cache_schema_version: int = 3
    cache_app_name: str = CONF.name
    cache_app_version: str = CONF.version or ""
    cache_upstream_commit: str = CONF.repo_commit_hash or ""
    wrapper_patch_version: str = "safe-ids-resumable-rnafm-v2"
    result_marker_name: str = "inference.json"
    pdb_prep_dir_name: str = "pdb_chunks"
    inference_prep_dir_name: str = "inference"
    max_prepare_jobs: int = 64
    max_pdb_cores: int = 32
    max_total_pdb_cores: int = 64
    preprocess_shard_size: int = 1024
    rnafm_device_env: str = "ENSIRNA_RNAFM_DEVICE"
    rosetta_compat_root: Path = Path(
        "/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371"
    )
    rnafm_revision: str = "91d4a46d28d8054a7b429955e8fc0c253ba0afd6"
    get_pdb_source_sha256: str = (
        "8e509f253b552c6312f4bd655bc75a47f9f017b57925d7928ae63459fefe1fb8"
    )
    dataset_source_sha256: str = (
        "dc3dae6f9f2b950c6a6c2a31f85b37e95f302f2402324aa8969e1fe7de2bc1c8"
    )
    rnafm_cache_path: Path = (
        Path(CONF.default_env["TORCH_HOME"]) / "hub/checkpoints/RNA-FM_pretrained.pth"
    )
    checkpoint_filenames: tuple[str, ...] = tuple(
        f"checkpoint_{idx}.ckpt" for idx in range(1, 6)
    )
    candidate_csv_columns: tuple[str, ...] = (
        "siRNA",
        "anti seq",
        "sense seq",
        "mRNA_seq",
        "position",
        "efficacy",
    )
    checkpoint_dir: Path = Path(CONF.model_volume_mountpoint) / "pkl"
    conda_packages: tuple[str, ...] = (
        f"python={CONF.python_version}",
        "pip",
        "viennarna=2.6.4-0",
    )
    conda_channels: tuple[str, ...] = ("conda-forge", "bioconda")
    pip_packages: tuple[str, ...] = (
        "biopython==1.83",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.1",
        "tensorboard==2.17.1",
        "tqdm==4.66.5",
        "openpyxl==3.1.5",
        "rdkit==2023.9.6",
        "scikit-learn==1.5.1",
        "xgboost==2.1.1",
    )
    torch_packages: tuple[str, ...] = (
        "torch==2.2.1",
        "torchvision==0.17.1",
        "torchaudio==2.2.1",
    )
    torch_index_url: str = "https://download.pytorch.org/whl/cu118"
    extra_pip_packages: tuple[str, ...] = (
        "torch-geometric==2.5.3",
        "rna-fm==0.2.2",
    )

    @property
    def rnafm_pretrained_url(self) -> str:
        """Return the immutable RNA-FM checkpoint URL."""
        return (
            "https://huggingface.co/cuhkaih/rnafm/resolve/"
            f"{self.rnafm_revision}/RNA-FM_pretrained.pth"
        )

    @property
    def cache_namespace(self) -> str:
        """Return semantic runtime state that affects prepared artifacts."""
        return "\n".join((
            self.cache_app_name,
            self.cache_app_version,
            self.cache_upstream_commit,
            self.wrapper_patch_version,
            self.rnafm_revision,
            *self.conda_packages,
            *self.pip_packages,
            *self.torch_packages,
            *self.extra_pip_packages,
        ))

    @property
    def mamba_bin_path(self) -> str:
        """Return PATH with micromamba's base environment first."""
        return f"{self.mamba_root}/bin:/root/.local/bin:$PATH"

    @property
    def mamba_lib_path(self) -> str:
        """Return the micromamba library path required by ViennaRNA."""
        return f"{self.mamba_root}/lib"

    @property
    def rosetta_rna_denovo(self) -> Path:
        """Return the ENsiRNA-expected Rosetta rna_denovo path."""
        return self.rosetta_compat_root / (
            "main/source/bin/rna_denovo.static.linuxgccrelease"
        )

    @property
    def rosetta_extract(self) -> Path:
        """Return the ENsiRNA-expected Rosetta extract script path."""
        return self.rosetta_compat_root / (
            "main/tools/rna_tools/silent_util/extract_lowscore_decoys.py"
        )

    @property
    def model_downloads(self) -> dict[str, Path]:
        """Return ENsiRNA model URLs mapped to model-volume paths."""
        return {
            self.rnafm_pretrained_url: self.rnafm_cache_path,
            **{
                (
                    f"{CONF.repo_url}/raw/{CONF.repo_commit_hash}/ENsiRNA/pkl/"
                    f"{filename}"
                ): self.checkpoint_dir / filename
                for filename in self.checkpoint_filenames
            },
        }

    @property
    def rosetta_extract_shim(self) -> str:
        """Return the Rosetta extract shim written into the image."""
        return """#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

silent_file = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
if silent_file is None:
    raise SystemExit("usage: extract_lowscore_decoys.py <silent-file> ...")

before = {p.name for p in Path.cwd().glob("*.pdb")}
subprocess.run(["extract_pdbs", "-in:file:silent", silent_file], check=True)
created = [p for p in Path.cwd().glob("*.pdb") if p.name not in before]
if not created:
    raise SystemExit("extract_pdbs did not produce a PDB")

target = Path(f"{silent_file}.1.pdb")
if created[0] != target:
    created[0].replace(target)
"""

    @property
    def rosetta_extract_writer(self) -> str:
        """Return a Python one-liner that writes the Rosetta extract shim."""
        return (
            "from pathlib import Path; "
            f"Path({str(self.rosetta_extract)!r}).write_text("
            f"{self.rosetta_extract_shim!r})"
        )

    @property
    def patched_sources_compile_command(self) -> str:
        """Return the build-time syntax check for patched upstream modules."""
        sources = (
            self.ensirna_dir / "data/get_pdb.py",
            self.ensirna_dir / "data/dataset.py",
        )
        return "python -m py_compile " + " ".join(
            shlex.quote(str(source)) for source in sources
        )

    @property
    def get_pdb_runtime_patch(self) -> str:
        """Return the source patch for ENsiRNA's PDB generation helper."""
        return f"""import hashlib
from pathlib import Path

path = Path({str(self.ensirna_dir / "data/get_pdb.py")!r})
source = path.read_bytes()
actual_sha256 = hashlib.sha256(source).hexdigest()
expected_sha256 = {self.get_pdb_source_sha256!r}
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"ENsiRNA get_pdb.py source hash mismatch: "
        f"expected {{expected_sha256}}, got {{actual_sha256}}"
    )
text = source.decode("utf-8").replace("\\r\\n", "\\n")
init_old = '''    def __init__(self,excel_dir,pdb_dir):
'''
init_new = '''    def __init__(self,excel_dir,pdb_dir,num_cores=1):
'''
cores_old = '''        self.num_cores = multiprocessing.cpu_count()
'''
cores_new = '''        if num_cores < 1:
            raise ValueError('num_cores must be at least 1')
        self.num_cores = num_cores
'''
pool_old = '''        self.chunk_size = max(1, total_rows // self.num_cores)
        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(processes=len(chunks)) as pool:
'''
pool_new = '''        self.chunk_size = max(
            1,
            (total_rows + self.num_cores - 1) // self.num_cores,
        )
        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(
            processes=min(self.num_cores, len(chunks)),
        ) as pool:
'''
parser_old = '''    parser.add_argument('-p','--pdb_dir', type=str, default=None, help='Path to save processed data')
'''
parser_new = '''    parser.add_argument('-p','--pdb_dir', type=str, default=None, help='Path to save processed data')
    parser.add_argument('--num-cores', type=int, default=1, help='Local PDB worker processes')
'''
call_old = '''        Data_Prepare(filename,args.pdb_dir).process()
'''
call_new = '''        Data_Prepare(filename,args.pdb_dir,args.num_cores).process()
'''
old = '''        if len(sec_pos) != 61+len(seq2)+len(seq1)+1+1:
            print('!=',data['siRNA'],len(sec_pos),len(seq2))
            return None
        return sec_pos,chain
'''
new = '''        expected_len = 61 + len(seq2) + len(seq1) + 1 + 1
        if len(sec_pos) != expected_len:
            print('!=', data['siRNA'], len(sec_pos), len(seq2))
            while len(sec_pos) < expected_len:
                sec_pos.append(sec_pos[-1] - 1 if sec_pos else 0)
                chain.append(3)
            if len(sec_pos) > expected_len:
                sec_pos = sec_pos[:expected_len]
                chain = chain[:expected_len]
        return sec_pos, chain
'''
secstruct_old = '''        secondary_seq = secondary_seq1 + ' ' + secondary_seq2
'''
secstruct_new = '''        def _fit_secstruct(secstruct, size):
            if len(secstruct) < size:
                return secstruct + '.' * (size - len(secstruct))
            while len(secstruct) > size and secstruct.startswith('.'):
                secstruct = secstruct[1:]
            while len(secstruct) > size and secstruct.endswith('.'):
                secstruct = secstruct[:-1]
            return secstruct[:size]

        secondary_seq1 = _fit_secstruct(secondary_seq1, len(seq1))
        secondary_seq2 = _fit_secstruct(secondary_seq2, len(seq2))
        secondary_seq = secondary_seq1 + ' ' + secondary_seq2
'''
rosetta_cmd_old = '''        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna'])
'''
rosetta_cmd_new = '''        subprocess.run([FF,'-sequence',seq,'-secstruct',secondary_seq,'-minimize_rna','-out:file:silent','default.out'])
'''
if init_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb constructor not found")
if cores_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb CPU core block not found")
if pool_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb process-pool block not found")
if parser_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb argument parser not found")
if call_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb entrypoint call not found")
if old not in text:
    raise SystemExit("expected ENsiRNA get_pdb length check not found")
if secstruct_old not in text:
    raise SystemExit("expected ENsiRNA secondary structure block not found")
if rosetta_cmd_old not in text:
    raise SystemExit("expected ENsiRNA Rosetta command not found")
text = text.replace(init_old, init_new)
text = text.replace(cores_old, cores_new)
text = text.replace(pool_old, pool_new)
text = text.replace(parser_old, parser_new)
text = text.replace(call_old, call_new)
text = text.replace(old, new)
text = text.replace(secstruct_old, secstruct_new)
text = text.replace(rosetta_cmd_old, rosetta_cmd_new)
path.write_text(text, encoding="utf-8")
"""

    @property
    def get_pdb_runtime_patch_runner(self) -> str:
        """Return a Python one-liner that applies the PDB helper patch."""
        return f"exec({self.get_pdb_runtime_patch!r})"

    @property
    def dataset_runtime_patch(self) -> str:
        """Return the source patch for lazy RNA-FM preprocessing."""
        return f"""import hashlib
from pathlib import Path

path = Path({str(self.ensirna_dir / "data/dataset.py")!r})
source = path.read_bytes()
actual_sha256 = hashlib.sha256(source).hexdigest()
expected_sha256 = {self.dataset_source_sha256!r}
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"ENsiRNA dataset.py source hash mismatch: "
        f"expected {{expected_sha256}}, got {{actual_sha256}}"
    )
text = source.decode("utf-8").replace("\\r\\n", "\\n")
old = '''device = 'cuda'
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
model.to(device=device)
'''
new = '''device = os.environ.get({self.rnafm_device_env!r}, 'cuda')
model = None
batch_converter = None

def _ensure_rnafm_loaded():
    global model, batch_converter
    if model is not None and batch_converter is not None:
        return
    loaded_model, alphabet = fm.pretrained.rna_fm_t12()
    loaded_model.eval()  # disables dropout for deterministic results
    loaded_model.to(device=device)
    model = loaded_model
    batch_converter = alphabet.get_batch_converter()
'''
preprocess_old = '''        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\\\\n')
'''
preprocess_new = '''        _ensure_rnafm_loaded()
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\\\\n')
'''
if old not in text:
    raise SystemExit("expected ENsiRNA RNA-FM eager load block not found")
if preprocess_old not in text:
    raise SystemExit("expected ENsiRNA dataset preprocess block not found")
text = text.replace(old, new)
text = text.replace(preprocess_old, preprocess_new)
path.write_text(text, encoding="utf-8")
"""

    @property
    def dataset_runtime_patch_runner(self) -> str:
        """Return a Python one-liner that applies the dataset patch."""
        return f"exec({self.dataset_runtime_patch!r})"


def _sanitize_fasta_for_upstream(mrna_fasta_bytes: bytes) -> bytes:
    """Return canonical FASTA bytes with shell- and path-safe record names."""
    try:
        lines = mrna_fasta_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("ENsiRNA mRNA FASTA must be UTF-8 text") from exc

    records: list[tuple[str, list[str]]] = []
    raw_names: set[str] = set()
    safe_names: set[str] = set()
    current_sequence: list[str] | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            raw_name = line[1:].strip()
            if not raw_name:
                raise ValueError(f"FASTA record name is empty on line {line_number}")
            if raw_name in raw_names:
                raise ValueError(f"Duplicate FASTA record name: {raw_name}")
            raw_names.add(raw_name)
            try:
                safe_name = sanitize_filename(raw_name)
            except ValueError as exc:
                raise ValueError(f"Unsafe FASTA record name: {raw_name}") from exc
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name)
            safe_name = re.sub(r"_+", "_", safe_name).strip("._-")
            if not safe_name:
                raise ValueError(f"Unsafe FASTA record name: {raw_name}")
            if len(safe_name) > 180:
                safe_name = f"{safe_name[:160]}_{hash_string(raw_name)[:16]}"
            if safe_name in safe_names:
                raise ValueError(
                    "FASTA record name collision after sanitization: "
                    f"{raw_name} -> {safe_name}"
                )
            safe_names.add(safe_name)
            current_sequence = []
            records.append((safe_name, current_sequence))
            continue
        if current_sequence is None:
            raise ValueError(
                f"FASTA sequence appears before the first record on line {line_number}"
            )
        sequence_line = re.sub(r"\s+", "", line).upper()
        unsupported = sorted(set(sequence_line) - set("ACGUTN"))
        if unsupported:
            raise ValueError(
                "ENsiRNA FASTA contains unsupported bases on line "
                f"{line_number}: {''.join(unsupported)}"
            )
        current_sequence.append(sequence_line)

    if not records:
        raise ValueError("ENsiRNA mRNA FASTA must contain at least one record")
    canonical = []
    for name, sequence_lines in records:
        sequence = "".join(sequence_lines)
        if len(sequence) < 19:
            raise ValueError(
                f"ENsiRNA FASTA record {name!r} must contain at least 19 bases"
            )
        canonical.extend((f">{name}", sequence))
    return ("\n".join(canonical) + "\n").encode()


def _cache_key_for_fasta(
    mrna_fasta_bytes: bytes, *, force_generation: str | None = None
) -> str:
    """Return a deterministic cache key for one mRNA FASTA payload."""
    canonical_fasta = _sanitize_fasta_for_upstream(mrna_fasta_bytes)
    return hash_string(
        "\n".join((
            APP_INFO.cache_namespace,
            canonical_fasta.decode("utf-8"),
            f"force_generation:{force_generation or ''}",
        ))
    )


def _layout_for_cache_key(cache_key: str) -> AppRunLayout:
    """Return the shared volume layout for one prepared FASTA."""
    if re.fullmatch(r"[0-9a-f]{64}", cache_key) is None:
        raise ValueError("ENsiRNA cache key must be a SHA-256 hexadecimal digest")
    return AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / "cache" / cache_key[:2] / cache_key
    )


def _pdb_prep_dir(layout: AppRunLayout) -> Path:
    """Return the preparation directory reserved for PDB chunk artifacts."""
    return layout.prep_dir / APP_INFO.pdb_prep_dir_name


def _inference_prep_dir(layout: AppRunLayout) -> Path:
    """Return the preparation directory reserved for inference staging."""
    return layout.prep_dir / APP_INFO.inference_prep_dir_name


def _same_path(left: str | Path, right: str | Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return Path(left).resolve() == Path(right).resolve()


def _layout_from_prepared_dir(prepared_dir: str | Path) -> AppRunLayout:
    """Validate and return the content-addressed layout for a prepared path."""
    supplied_root = Path(prepared_dir)
    expected = _layout_for_cache_key(supplied_root.name)
    if not _same_path(supplied_root, expected.run_root):
        raise ValueError(
            f"ENsiRNA prepared directory is outside its cache key: {prepared_dir}"
        )
    return expected


def _validate_chunk_spec(chunk: EnsirnaPdbChunkSpec) -> AppRunLayout:
    """Validate that a PDB chunk can only access its content-addressed run."""
    if re.fullmatch(r"chunk_[0-9]{4,}", chunk.chunk_name) is None:
        raise ValueError(f"Invalid ENsiRNA PDB chunk name: {chunk.chunk_name!r}")
    csv_path = Path(chunk.csv_path)
    try:
        supplied_root = csv_path.parents[2]
    except IndexError as exc:
        raise ValueError(f"Invalid ENsiRNA PDB chunk path: {csv_path}") from exc
    layout = _layout_from_prepared_dir(supplied_root)
    expected_prep_dir = _pdb_prep_dir(layout)
    expected_paths = (
        (csv_path, expected_prep_dir / f"{chunk.chunk_name}.csv"),
        (Path(chunk.json_path), expected_prep_dir / f"{chunk.chunk_name}.json"),
        (
            Path(chunk.pdb_dir),
            layout.outputs_dir / f"{APP_INFO.input_stem}_pdb",
        ),
    )
    if any(not _same_path(actual, expected) for actual, expected in expected_paths):
        raise ValueError(f"ENsiRNA PDB chunk paths do not match {chunk.chunk_name}")
    return layout


def _validate_preparation_plan(plan: EnsirnaPreparationPlan) -> AppRunLayout:
    """Validate all paths and counts in a cross-function preparation plan."""
    layout = _layout_from_prepared_dir(plan.prepared_dir)
    if plan.cache_key != layout.run_root.name:
        raise ValueError("ENsiRNA preparation plan cache key does not match its path")
    expected_paths = (
        (plan.json_path, layout.outputs_dir / f"{APP_INFO.input_stem}.json"),
        (
            plan.processed_dir,
            layout.outputs_dir / f"{APP_INFO.input_stem}_processed",
        ),
    )
    if any(not _same_path(actual, expected) for actual, expected in expected_paths):
        raise ValueError("ENsiRNA preparation plan paths leave the cache layout")
    if plan.candidate_count <= 0 or plan.chunk_count < 0:
        raise ValueError("ENsiRNA preparation plan counts must be nonnegative")
    for chunk in plan.chunks:
        chunk_layout = _validate_chunk_spec(chunk)
        if not _same_path(chunk_layout.run_root, layout.run_root):
            raise ValueError("ENsiRNA PDB chunk belongs to a different cache entry")
    return layout


def _prepared_marker_path(layout: AppRunLayout) -> Path:
    """Return the prepared marker path for a cached ENsiRNA input."""
    return layout.markers_dir / APP_INFO.prepared_marker_name


def _candidate_csv_marker_path(layout: AppRunLayout) -> Path:
    """Return the completion marker for the generated candidate CSV."""
    return layout.markers_dir / APP_INFO.candidate_csv_marker_name


def _result_marker_path(layout: AppRunLayout) -> Path:
    """Return the inference-result completion marker path."""
    return layout.markers_dir / APP_INFO.result_marker_name


def _result_ready(layout: AppRunLayout, cache_key: str) -> bool:
    """Return whether the atomically published XLSX matches its marker."""
    import orjson

    result_path = layout.outputs_dir / f"{APP_INFO.input_stem}_result.xlsx"
    marker_path = _result_marker_path(layout)
    if not result_path.is_file() or result_path.stat().st_size == 0:
        return False
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("schema_version") == APP_INFO.cache_schema_version
        and marker.get("cache_key") == cache_key
        and marker.get("size") == result_path.stat().st_size
        and marker.get("sha256") == _file_sha256(result_path)
    )


def _required_prepared_paths(layout: AppRunLayout) -> tuple[Path, ...]:
    """Return paths required before GPU inference may start."""
    stem = APP_INFO.input_stem
    return (
        _prepared_marker_path(layout),
        layout.outputs_dir / f"{stem}.json",
        layout.outputs_dir / f"{stem}_processed" / "_metainfo",
    )


def _atomic_write(path: Path, content: bytes) -> None:
    """Publish bytes with a same-directory atomic replacement."""
    from uuid import uuid4

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    """Return the SHA-256 digest for an in-memory artifact."""
    from hashlib import sha256

    return sha256(content).hexdigest()


def _candidate_csv_facts(
    csv_path: Path, *, reject_unsafe_ids: bool = False
) -> dict[str, object] | None:
    """Return validated candidate CSV facts, or ``None`` when incomplete."""
    import polars as pl

    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return None
    try:
        frame = pl.read_csv(csv_path)
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        pl.exceptions.PolarsError,
    ):
        return None
    columns = tuple(frame.columns)
    if columns != APP_INFO.candidate_csv_columns or frame.height == 0:
        return None
    candidate_ids = [str(value) for value in frame.get_column("siRNA").to_list()]
    try:
        _validate_candidate_ids(candidate_ids)
    except ValueError:
        if reject_unsafe_ids:
            raise
        return None
    try:
        size = csv_path.stat().st_size
        sha256 = _file_sha256(csv_path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return {
        "columns": list(columns),
        "candidate_count": frame.height,
        "size": size,
        "sha256": sha256,
    }


def _candidate_csv_valid(
    *, layout: AppRunLayout, cache_key: str, input_sha256: str
) -> bool:
    """Return whether candidate CSV content matches its last-written marker."""
    import orjson

    csv_path = layout.outputs_dir / f"{APP_INFO.input_stem}.csv"
    facts = _candidate_csv_facts(csv_path)
    if facts is None:
        return False
    try:
        marker = orjson.loads(_candidate_csv_marker_path(layout).read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("schema_version") == APP_INFO.cache_schema_version
        and marker.get("cache_key") == cache_key
        and marker.get("input_sha256") == input_sha256
        and marker.get("csv_path") == str(csv_path)
        and marker.get("columns") == facts["columns"]
        and marker.get("candidate_count") == facts["candidate_count"]
        and marker.get("size") == facts["size"]
        and marker.get("sha256") == facts["sha256"]
    )


def _write_candidate_csv_marker(
    *,
    layout: AppRunLayout,
    cache_key: str,
    input_sha256: str,
    facts: dict[str, object],
) -> None:
    """Seal an already-atomically-published candidate CSV."""
    import orjson

    csv_path = layout.outputs_dir / f"{APP_INFO.input_stem}.csv"
    _atomic_write(
        _candidate_csv_marker_path(layout),
        orjson.dumps({
            "schema_version": APP_INFO.cache_schema_version,
            "cache_key": cache_key,
            "input_sha256": input_sha256,
            "csv_path": str(csv_path),
            **facts,
        }),
    )


def _json_records(json_path: Path) -> list[dict]:
    """Load and validate prepared ENsiRNA JSON records."""
    import orjson

    if not json_path.is_file() or json_path.stat().st_size == 0:
        return []
    records = []
    seen_ids = set()
    try:
        lines = json_path.read_bytes().splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = orjson.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("siRNA"), str):
                return []
            candidate_id = record["siRNA"]
            if candidate_id in seen_ids:
                return []
            seen_ids.add(candidate_id)
            records.append(record)
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return []
    return records


def _processed_manifest_facts(
    processed_dir: Path,
    expected_count: int,
    *,
    include_digests: bool = True,
) -> list[dict[str, int | str]] | None:
    """Return content facts for every complete processed shard, or ``None``."""
    import orjson

    marker = processed_dir / "_metainfo"
    if not marker.is_file():
        return None
    try:
        metadata = orjson.loads(marker.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return None
    if not isinstance(metadata, dict):
        return None
    file_names = metadata.get("file_names")
    file_num_entries = metadata.get("file_num_entries")
    if not isinstance(file_names, list) or not isinstance(file_num_entries, list):
        return None
    if not file_names or len(file_names) != len(file_num_entries):
        return None
    if not all(isinstance(value, int) and value > 0 for value in file_num_entries):
        return None
    if (
        metadata.get("num_entry") != sum(file_num_entries)
        or sum(file_num_entries) != expected_count
    ):
        return None
    processed_root = processed_dir.resolve()
    facts: list[dict[str, int | str]] = []
    seen_paths: set[str] = set()
    for raw_path, entry_count in zip(file_names, file_num_entries, strict=True):
        if not isinstance(raw_path, str) or not isinstance(entry_count, int):
            return None
        part_path = Path(raw_path).resolve()
        if not part_path.is_relative_to(processed_root):
            return None
        if not part_path.is_file() or part_path.stat().st_size == 0:
            return None
        relative_path = part_path.relative_to(processed_root).as_posix()
        if relative_path in seen_paths:
            return None
        seen_paths.add(relative_path)
        fact: dict[str, int | str] = {
            "path": relative_path,
            "entries": entry_count,
            "size": part_path.stat().st_size,
        }
        if include_digests:
            fact["sha256"] = _file_sha256(part_path)
        facts.append(fact)
    return facts


def _processed_manifest_valid(processed_dir: Path, expected_count: int) -> bool:
    """Return whether upstream metadata references complete processed shards."""
    return (
        _processed_manifest_facts(processed_dir, expected_count, include_digests=False)
        is not None
    )


def _processed_shard_marker_path(processed_dir: Path) -> Path:
    """Return the independent integrity marker for one RNA-FM shard."""
    return processed_dir / APP_INFO.preprocess_shard_marker_name


def _write_processed_shard_marker(
    *,
    processed_dir: Path,
    cache_key: str,
    shard_index: int,
    input_sha256: str,
    entry_count: int,
    processed_parts: list[dict[str, int | str]],
) -> None:
    """Seal one structurally validated RNA-FM shard before publication."""
    import orjson

    _atomic_write(
        _processed_shard_marker_path(processed_dir),
        orjson.dumps({
            "schema_version": APP_INFO.cache_schema_version,
            "cache_key": cache_key,
            "shard_index": shard_index,
            "input_sha256": input_sha256,
            "entry_count": entry_count,
            "processed_parts": processed_parts,
        }),
    )


def _processed_shard_valid(
    *,
    processed_dir: Path,
    cache_key: str,
    shard_index: int,
    input_sha256: str,
    entry_count: int,
) -> bool:
    """Return whether an RNA-FM shard exactly matches its independent seal."""
    import orjson

    processed_parts = _processed_manifest_facts(processed_dir, entry_count)
    if processed_parts is None:
        return False
    try:
        marker = orjson.loads(_processed_shard_marker_path(processed_dir).read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("schema_version") == APP_INFO.cache_schema_version
        and marker.get("cache_key") == cache_key
        and marker.get("shard_index") == shard_index
        and marker.get("input_sha256") == input_sha256
        and marker.get("entry_count") == entry_count
        and marker.get("processed_parts") == processed_parts
    )


def _prepared_metadata(layout: AppRunLayout) -> dict | None:
    """Return validated prepared-cache metadata, or ``None``."""
    import orjson

    if not all(path.is_file() for path in _required_prepared_paths(layout)):
        return None
    try:
        metadata = orjson.loads(_prepared_marker_path(layout).read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get("schema_version") != APP_INFO.cache_schema_version:
        return None
    cache_key = layout.run_root.name
    try:
        expected_layout = _layout_for_cache_key(cache_key)
    except ValueError:
        return None
    if not _same_path(layout.run_root, expected_layout.run_root):
        return None
    if (
        metadata.get("cache_key") != cache_key
        or metadata.get("cache_namespace_sha256")
        != hash_string(APP_INFO.cache_namespace)
        or metadata.get("json_path")
        != str(layout.outputs_dir / f"{APP_INFO.input_stem}.json")
        or metadata.get("processed_dir")
        != str(layout.outputs_dir / f"{APP_INFO.input_stem}_processed")
    ):
        return None
    candidate_count = metadata.get("candidate_count")
    json_records = metadata.get("json_records")
    if (
        not isinstance(candidate_count, int)
        or candidate_count <= 0
        or json_records != candidate_count
    ):
        return None
    json_path = layout.outputs_dir / f"{APP_INFO.input_stem}.json"
    records = _json_records(json_path)
    if len(records) != candidate_count:
        return None
    try:
        _validate_candidate_ids([str(record["siRNA"]) for record in records])
    except ValueError:
        return None
    if metadata.get("json_sha256") != _file_sha256(json_path):
        return None
    processed_dir = layout.outputs_dir / f"{APP_INFO.input_stem}_processed"
    processed_parts = _processed_manifest_facts(processed_dir, candidate_count)
    if processed_parts is None or metadata.get("processed_parts") != processed_parts:
        return None
    return metadata


def _is_prepared(layout: AppRunLayout) -> bool:
    """Return whether the prepared-input cache is complete and internally valid."""
    return _prepared_metadata(layout) is not None


def _plan_from_layout(
    *, cache_key: str, layout: AppRunLayout, candidate_count: int, chunk_count: int
) -> EnsirnaPreparationPlan:
    """Build a primitive-path preparation plan."""
    stem = APP_INFO.input_stem
    return EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / f"{stem}.json"),
        processed_dir=str(layout.outputs_dir / f"{stem}_processed"),
        candidate_count=candidate_count,
        chunk_count=chunk_count,
        chunks=[],
        cached=True,
    )


def _cached_preparation_plan(
    *, cache_key: str, layout: AppRunLayout
) -> EnsirnaPreparationPlan | None:
    """Return a cached preparation plan when the prepared marker is valid."""
    marker = _prepared_metadata(layout)
    if marker is None or marker.get("cache_key") != cache_key:
        return None
    return _plan_from_layout(
        cache_key=cache_key,
        layout=layout,
        candidate_count=int(marker.get("candidate_count", 0)),
        chunk_count=int(marker.get("chunk_count", 0)),
    )


def _write_prepared_marker(
    *, layout: AppRunLayout, plan: EnsirnaPreparationPlan, json_records: int
) -> None:
    """Write the prepared-input cache marker."""
    import orjson

    json_path = Path(plan.json_path)
    processed_dir = Path(plan.processed_dir)
    processed_parts = _processed_manifest_facts(processed_dir, plan.candidate_count)
    if processed_parts is None:
        raise RuntimeError("ENsiRNA processed dataset cannot be marked complete")
    _atomic_write(
        _prepared_marker_path(layout),
        orjson.dumps({
            "schema_version": APP_INFO.cache_schema_version,
            "cache_key": plan.cache_key,
            "cache_namespace_sha256": hash_string(APP_INFO.cache_namespace),
            "candidate_count": plan.candidate_count,
            "chunk_count": plan.chunk_count,
            "json_records": json_records,
            "json_path": plan.json_path,
            "processed_dir": plan.processed_dir,
            "json_sha256": _file_sha256(json_path),
            "processed_parts": processed_parts,
        }),
    )


def _load_prepared_json_records(layout: AppRunLayout) -> dict[str, dict]:
    """Return prepared JSON records keyed by siRNA ID, newest chunk last."""
    stem = APP_INFO.input_stem
    records: dict[str, dict] = {}
    json_paths = [
        layout.outputs_dir / f"{stem}.json",
        *sorted(_pdb_prep_dir(layout).glob("chunk_*.json")),
    ]
    for json_path in json_paths:
        parsed_records = _json_records(json_path)
        if not parsed_records:
            continue
        for record in parsed_records:
            candidate_id = str(record["siRNA"])
            records[candidate_id] = record
    return records


def _validate_candidate_ids(candidate_ids: list[str]) -> None:
    """Reject upstream candidate identifiers that are unsafe or ambiguous."""
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("ENsiRNA generated duplicate candidate IDs")
    invalid = [
        candidate_id
        for candidate_id in candidate_ids
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,239}", candidate_id) is None
    ]
    if invalid:
        raise ValueError(f"ENsiRNA generated an unsafe candidate ID: {invalid[0]!r}")


def _validate_pdb_records(records: list[dict], pdb_dir: Path) -> None:
    """Require one exact, nonempty PDB artifact for every candidate record."""
    pdb_root = pdb_dir.resolve()
    missing = []
    candidate_ids = [str(record.get("siRNA", "")) for record in records]
    _validate_candidate_ids(candidate_ids)
    for record in records:
        candidate_id = str(record["siRNA"])
        raw_path = record.get("pdb_data_path")
        if not isinstance(raw_path, str):
            missing.append(candidate_id)
            continue
        pdb_path = Path(raw_path).resolve()
        expected_path = (pdb_root / f"{candidate_id}.pdb").resolve()
        if (
            pdb_path != expected_path
            or not pdb_path.is_file()
            or pdb_path.stat().st_size == 0
        ):
            missing.append(candidate_id)
    if missing:
        raise FileNotFoundError(
            "ENsiRNA PDB artifacts are incomplete for "
            f"{len(missing)} candidates: {missing[:5]}"
        )


def _chunk_artifacts_valid(
    chunk: EnsirnaPdbChunkSpec,
    *,
    json_path: Path | None = None,
    pdb_dir: Path | None = None,
) -> bool:
    """Return whether a PDB chunk has exactly its expected JSON and PDB outputs."""
    import polars as pl

    try:
        candidate_ids = [
            str(value)
            for value in pl.read_csv(chunk.csv_path).get_column("siRNA").to_list()
        ]
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        pl.exceptions.PolarsError,
    ):
        return False
    records = _json_records(json_path or Path(chunk.json_path))
    if [str(record["siRNA"]) for record in records] != candidate_ids:
        return False
    try:
        _validate_pdb_records(records, pdb_dir or Path(chunk.pdb_dir))
    except FileNotFoundError:
        return False
    return bool(records)


def _next_pdb_chunk_index(prep_dir: Path) -> int:
    """Return the next collision-free numeric PDB chunk index."""
    next_index = 0
    for path in prep_dir.glob("chunk_*.*"):
        chunk_id = path.stem.removeprefix("chunk_")
        if chunk_id.isdecimal():
            next_index = max(next_index, int(chunk_id) + 1)
    return next_index


def _link_checkpoints() -> None:
    """Link model-volume checkpoints into ENsiRNA's expected pkl directory."""
    checkpoint_dir = APP_INFO.ensirna_dir / "pkl"
    checkpoint_dir.mkdir(exist_ok=True)
    for filename in APP_INFO.checkpoint_filenames:
        checkpoint = APP_INFO.checkpoint_dir / filename
        if not checkpoint.exists():
            raise FileNotFoundError(f"ENsiRNA checkpoint not found: {checkpoint}")
        link = checkpoint_dir / filename
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(checkpoint)


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    modal.Image
    .from_registry("rosettacommons/rosetta:serial-420", add_python=CONF.python_version)
    .apt_install("git", "curl", "ca-certificates", "build-essential", "zstd")
    .env(
        CONF.default_env
        | {
            "MAMBA_ROOT_PREFIX": APP_INFO.mamba_root,
            "PATH": APP_INFO.mamba_bin_path,
            "LD_LIBRARY_PATH": APP_INFO.mamba_lib_path,
        }
    )
    .run_commands("curl -L micro.mamba.pm/install.sh | bash")
    .micromamba_install(
        *APP_INFO.conda_packages, channels=list(APP_INFO.conda_channels)
    )
    .run_commands(f"micromamba run -n {APP_INFO.conda_env_name} python -c 'import RNA'")
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "find . -path '*/pkl/*.ckpt' -delete",
        ))
    )
    .run_commands(
        " && ".join((
            f"mkdir -p {APP_INFO.rosetta_rna_denovo.parent} "
            f"{APP_INFO.rosetta_extract.parent}",
            f"ln -sf /usr/local/bin/rna_denovo {APP_INFO.rosetta_rna_denovo}",
            f"python -c {shlex.quote(APP_INFO.rosetta_extract_writer)}",
            f"chmod +x {APP_INFO.rosetta_extract}",
        ))
    )
    .run_commands(f"python -c {shlex.quote(APP_INFO.get_pdb_runtime_patch_runner)}")
    .run_commands(f"python -c {shlex.quote(APP_INFO.dataset_runtime_patch_runner)}")
    .run_commands(APP_INFO.patched_sources_compile_command)
    .uv_pip_install(*APP_INFO.pip_packages)
    .uv_pip_install(*APP_INFO.torch_packages, index_url=APP_INFO.torch_index_url)
    .uv_pip_install(*APP_INFO.extra_pip_packages)
    .pipe(patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3"])
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
ENSIRNA_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_ensirna_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=CONF.timeout
)
def download_ensirna_models(force: bool = False) -> None:
    """Download ENsiRNA model files into the standard model volume."""
    download_files(
        APP_INFO.model_downloads,
        force=force,
        num_retries=3,
        progress_bar_desc="ENsiRNA model downloads",
    )
    MODEL_VOLUME.commit()


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def ensirna_prepare_inputs(
    mrna_fasta_bytes: bytes,
    max_prepare_jobs: int = 4,
    force_generation: str | None = None,
) -> EnsirnaPreparationPlan:
    """Create siRNA CSV and CPU PDB chunk work in the output volume."""
    import polars as pl

    if not 1 <= max_prepare_jobs <= APP_INFO.max_prepare_jobs:
        raise ValueError(
            f"max_prepare_jobs must be between 1 and {APP_INFO.max_prepare_jobs}"
        )
    canonical_fasta = _sanitize_fasta_for_upstream(mrna_fasta_bytes)
    cache_key = _cache_key_for_fasta(canonical_fasta, force_generation=force_generation)
    CONF.output_volume.reload()
    layout = _layout_for_cache_key(cache_key)
    if cached_plan := _cached_preparation_plan(cache_key=cache_key, layout=layout):
        return cached_plan

    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    layout.outputs_dir.mkdir(parents=True, exist_ok=True)
    pdb_prep_dir = _pdb_prep_dir(layout)
    pdb_prep_dir.mkdir(parents=True, exist_ok=True)

    mrna_fasta = layout.inputs_dir / APP_INFO.input_fasta_name
    _atomic_write(mrna_fasta, canonical_fasta)
    input_sha256 = _bytes_sha256(canonical_fasta)
    stem = APP_INFO.input_stem
    csv_path = layout.outputs_dir / f"{stem}.csv"
    if not _candidate_csv_valid(
        layout=layout,
        cache_key=cache_key,
        input_sha256=input_sha256,
    ):
        from uuid import uuid4

        staging_dir = layout.outputs_dir / f".candidates.{uuid4().hex}.tmp"
        staging_csv = staging_dir / csv_path.name
        staging_dir.mkdir(parents=True)
        try:
            run_command(
                [
                    "micromamba",
                    "run",
                    "-n",
                    APP_INFO.conda_env_name,
                    "python",
                    "get_siRNA.py",
                    "-i",
                    str(mrna_fasta),
                    "-o",
                    str(staging_dir),
                ],
                cwd=APP_INFO.ensirna_dir,
                output_mode="capture",
            )
            staged_facts = _candidate_csv_facts(staging_csv, reject_unsafe_ids=True)
            if staged_facts is None:
                raise RuntimeError("ENsiRNA generated an incomplete candidate CSV")
            _candidate_csv_marker_path(layout).unlink(missing_ok=True)
            staging_csv.replace(csv_path)
            published_facts = _candidate_csv_facts(csv_path, reject_unsafe_ids=True)
            if published_facts is None or published_facts != staged_facts:
                raise RuntimeError("ENsiRNA candidate CSV failed atomic publication")
            _write_candidate_csv_marker(
                layout=layout,
                cache_key=cache_key,
                input_sha256=input_sha256,
                facts=published_facts,
            )
            if not _candidate_csv_valid(
                layout=layout,
                cache_key=cache_key,
                input_sha256=input_sha256,
            ):
                raise RuntimeError("ENsiRNA candidate CSV seal is invalid")
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    frame = pl.read_csv(csv_path)
    candidate_count = frame.height
    if candidate_count == 0:
        raise RuntimeError("ENsiRNA did not generate any siRNA candidates.")

    pdb_dir = layout.outputs_dir / f"{stem}_pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    candidate_sirnas = [str(value) for value in frame.get_column("siRNA").to_list()]
    _validate_candidate_ids(candidate_sirnas)
    for sirna in candidate_sirnas:
        lock_dir = pdb_dir / sirna
        if lock_dir.is_dir():
            shutil.rmtree(lock_dir)
    existing_records = _load_prepared_json_records(layout)
    valid_existing_sirnas = set()
    for sirna in candidate_sirnas:
        record = existing_records.get(sirna)
        if record is None:
            continue
        try:
            _validate_pdb_records([record], pdb_dir)
        except (FileNotFoundError, ValueError):
            continue
        valid_existing_sirnas.add(sirna)
    remaining_sirnas = [
        sirna for sirna in candidate_sirnas if sirna not in valid_existing_sirnas
    ]
    for sirna in remaining_sirnas:
        (pdb_dir / f"{sirna}.pdb").unlink(missing_ok=True)
    remaining_frame = frame.filter(
        pl.col("siRNA").cast(pl.String).is_in(remaining_sirnas)
    )

    chunks = []
    remaining_count = remaining_frame.height
    chunk_count = min(max(1, max_prepare_jobs), remaining_count)
    chunk_size = (
        (remaining_count + chunk_count - 1) // chunk_count if remaining_count else 1
    )
    next_chunk_index = _next_pdb_chunk_index(pdb_prep_dir)
    for idx, offset in enumerate(range(0, remaining_count, chunk_size)):
        chunk_name = f"chunk_{idx:04d}"
        if next_chunk_index:
            chunk_name = f"chunk_{next_chunk_index + idx:04d}"
        chunk_csv = pdb_prep_dir / f"{chunk_name}.csv"
        remaining_frame.slice(offset, chunk_size).write_csv(chunk_csv)
        chunks.append(
            EnsirnaPdbChunkSpec(
                chunk_name=chunk_name,
                csv_path=str(chunk_csv),
                json_path=str(pdb_prep_dir / f"{chunk_name}.json"),
                pdb_dir=str(pdb_dir),
            )
        )

    CONF.output_volume.commit()
    return EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / f"{stem}.json"),
        processed_dir=str(layout.outputs_dir / f"{stem}_processed"),
        candidate_count=candidate_count,
        chunk_count=len(chunks),
        chunks=chunks,
        cached=False,
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def ensirna_prepare_pdb_chunk(
    chunk: EnsirnaPdbChunkSpec, pdb_cores: int = 1
) -> dict[str, int | str]:
    """Run one PDB chunk with a bounded local Rosetta process pool."""
    from uuid import uuid4

    import orjson

    if not 1 <= pdb_cores <= APP_INFO.max_pdb_cores:
        raise ValueError(f"pdb_cores must be between 1 and {APP_INFO.max_pdb_cores}")
    _validate_chunk_spec(chunk)
    CONF.output_volume.reload()
    json_path = Path(chunk.json_path)
    if not _chunk_artifacts_valid(chunk):
        csv_path = Path(chunk.csv_path)
        staging_dir = csv_path.parent / f".{chunk.chunk_name}.{uuid4().hex}.tmp"
        staging_csv = staging_dir / f"{chunk.chunk_name}.csv"
        staging_json = staging_csv.with_suffix(".json")
        staging_pdb_dir = staging_dir / "pdb"
        staging_pdb_dir.mkdir(parents=True)
        shutil.copy2(csv_path, staging_csv)
        try:
            run_command(
                [
                    "micromamba",
                    "run",
                    "-n",
                    APP_INFO.conda_env_name,
                    "python",
                    "-m",
                    "data.get_pdb",
                    "-f",
                    str(staging_csv),
                    "-p",
                    str(staging_pdb_dir),
                    "--num-cores",
                    str(pdb_cores),
                ],
                cwd=APP_INFO.ensirna_dir,
                output_mode="inherit",
            )
            if not _chunk_artifacts_valid(
                chunk,
                json_path=staging_json,
                pdb_dir=staging_pdb_dir,
            ):
                raise RuntimeError(
                    f"ENsiRNA PDB chunk is incomplete: {chunk.chunk_name}"
                )
            records = _json_records(staging_json)
            final_pdb_dir = Path(chunk.pdb_dir)
            final_pdb_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                candidate_id = str(record["siRNA"])
                staged_pdb = staging_pdb_dir / f"{candidate_id}.pdb"
                final_pdb = final_pdb_dir / staged_pdb.name
                temporary_pdb = final_pdb.with_name(
                    f".{final_pdb.name}.{uuid4().hex}.tmp"
                )
                try:
                    shutil.copy2(staged_pdb, temporary_pdb)
                    temporary_pdb.replace(final_pdb)
                finally:
                    temporary_pdb.unlink(missing_ok=True)
                record["pdb_data_path"] = str(final_pdb)
            _atomic_write(
                json_path,
                b"".join(orjson.dumps(record) + b"\n" for record in records),
            )
            if not _chunk_artifacts_valid(chunk):
                raise RuntimeError(
                    f"ENsiRNA PDB chunk failed publication: {chunk.chunk_name}"
                )
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        CONF.output_volume.commit()
    return {
        "chunk_name": chunk.chunk_name,
        "json_path": str(json_path),
        "cached": int(_chunk_artifacts_valid(chunk)),
    }


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def ensirna_finalize_prepared_inputs(
    plan: EnsirnaPreparationPlan,
) -> EnsirnaPreparationPlan:
    """Merge PDB chunk JSON files into the prepared dataset JSON."""
    import orjson
    import polars as pl

    layout = _validate_preparation_plan(plan)
    CONF.output_volume.reload()
    if cached_plan := _cached_preparation_plan(cache_key=plan.cache_key, layout=layout):
        return cached_plan

    for chunk in plan.chunks:
        chunk_json = Path(chunk.json_path)
        if not chunk_json.exists():
            raise FileNotFoundError(f"ENsiRNA PDB chunk JSON not found: {chunk_json}")

    records = _load_prepared_json_records(layout)
    stem = APP_INFO.input_stem
    csv_path = layout.outputs_dir / f"{stem}.csv"
    if csv_path.exists():
        candidate_sirnas = [
            str(value) for value in pl.read_csv(csv_path).get_column("siRNA").to_list()
        ]
        missing_sirnas = [sirna for sirna in candidate_sirnas if sirna not in records]
        if missing_sirnas:
            raise FileNotFoundError(
                "ENsiRNA PDB JSON records missing for "
                f"{len(missing_sirnas)} candidates: {missing_sirnas[:5]}"
            )
        ordered_records = [records[sirna] for sirna in candidate_sirnas]
    else:
        ordered_records = list(records.values())

    pdb_dir = layout.outputs_dir / f"{stem}_pdb"
    _validate_pdb_records(ordered_records, pdb_dir)

    json_path = Path(plan.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        json_path,
        b"".join(orjson.dumps(record) + b"\n" for record in ordered_records),
    )

    CONF.output_volume.commit()
    return EnsirnaPreparationPlan(
        cache_key=plan.cache_key,
        prepared_dir=plan.prepared_dir,
        json_path=plan.json_path,
        processed_dir=plan.processed_dir,
        candidate_count=plan.candidate_count,
        chunk_count=plan.chunk_count,
        chunks=[],
        cached=False,
    )


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def ensirna_preprocess_dataset(
    plan: EnsirnaPreparationPlan,
    preprocess_shard_size: int = APP_INFO.preprocess_shard_size,
) -> EnsirnaPreparationPlan:
    """Build resumable RNA-FM preprocessing shards on GPU."""
    import orjson

    layout = _validate_preparation_plan(plan)
    if preprocess_shard_size < 1:
        raise ValueError("preprocess_shard_size must be at least 1")
    CONF.output_volume.reload()
    if cached_plan := _cached_preparation_plan(cache_key=plan.cache_key, layout=layout):
        return cached_plan

    json_path = Path(plan.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"ENsiRNA merged JSON not found: {json_path}")

    if not APP_INFO.rnafm_cache_path.exists():
        raise FileNotFoundError(
            f"RNA-FM checkpoint not found: {APP_INFO.rnafm_cache_path}"
        )
    json_lines = [line for line in json_path.read_bytes().splitlines() if line.strip()]
    json_records = len(json_lines)
    if json_records != plan.candidate_count:
        raise RuntimeError(
            "ENsiRNA merged JSON count does not match the preparation plan: "
            f"{json_records} != {plan.candidate_count}"
        )

    processed_dir = Path(plan.processed_dir)
    shards_dir = processed_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[str] = []
    part_counts: list[int] = []
    for shard_index, offset in enumerate(range(0, json_records, preprocess_shard_size)):
        shard_lines = json_lines[offset : offset + preprocess_shard_size]
        shard_count = len(shard_lines)
        shard_content = b"\n".join(shard_lines) + b"\n"
        shard_input_sha256 = _bytes_sha256(shard_content)
        shard_dir = shards_dir / f"shard_{shard_index:04d}"
        shard_output = shard_dir / "processed"
        if not _processed_shard_valid(
            processed_dir=shard_output,
            cache_key=plan.cache_key,
            shard_index=shard_index,
            input_sha256=shard_input_sha256,
            entry_count=shard_count,
        ):
            from uuid import uuid4

            staging_dir = shards_dir / f".shard_{shard_index:04d}.{uuid4().hex}.tmp"
            staging_output = staging_dir / "processed"
            staging_input = staging_dir / "input.json"
            staging_dir.mkdir(parents=True)
            _atomic_write(staging_input, shard_content)
            try:
                run_command(
                    [
                        "micromamba",
                        "run",
                        "-n",
                        APP_INFO.conda_env_name,
                        "python",
                        "-m",
                        "data.dataset",
                        "--dataset",
                        str(staging_input),
                        "--save_dir",
                        str(staging_output),
                    ],
                    cwd=APP_INFO.ensirna_dir,
                    env={APP_INFO.rnafm_device_env: "cuda"},
                    output_mode="inherit",
                )
                staged_parts = _processed_manifest_facts(staging_output, shard_count)
                if staged_parts is None:
                    raise RuntimeError(
                        f"ENsiRNA RNA-FM shard {shard_index} is incomplete"
                    )
                shard_metadata = orjson.loads(
                    (staging_output / "_metainfo").read_bytes()
                )
                relative_parts = [
                    Path(raw_path).resolve().relative_to(staging_dir.resolve())
                    for raw_path in shard_metadata["file_names"]
                ]
                shard_metadata["file_names"] = [
                    str(shard_dir / relative_path) for relative_path in relative_parts
                ]
                _atomic_write(
                    staging_output / "_metainfo", orjson.dumps(shard_metadata)
                )
                _write_processed_shard_marker(
                    processed_dir=staging_output,
                    cache_key=plan.cache_key,
                    shard_index=shard_index,
                    input_sha256=shard_input_sha256,
                    entry_count=shard_count,
                    processed_parts=staged_parts,
                )
                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                staging_dir.replace(shard_dir)
                (shard_dir / "input.json").unlink(missing_ok=True)
                if not _processed_shard_valid(
                    processed_dir=shard_output,
                    cache_key=plan.cache_key,
                    shard_index=shard_index,
                    input_sha256=shard_input_sha256,
                    entry_count=shard_count,
                ):
                    raise RuntimeError(
                        f"ENsiRNA RNA-FM shard {shard_index} failed publication"
                    )
                CONF.output_volume.commit()
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)

        shard_metadata = orjson.loads((shard_output / "_metainfo").read_bytes())
        part_paths.extend(str(path) for path in shard_metadata["file_names"])
        part_counts.extend(int(count) for count in shard_metadata["file_num_entries"])

    _atomic_write(
        processed_dir / "_metainfo",
        orjson.dumps({
            "num_entry": sum(part_counts),
            "file_names": part_paths,
            "file_num_entries": part_counts,
        }),
    )
    if not _processed_manifest_valid(processed_dir, json_records):
        raise RuntimeError("ENsiRNA processed dataset manifest is incomplete")

    prepared_plan = EnsirnaPreparationPlan(
        cache_key=plan.cache_key,
        prepared_dir=plan.prepared_dir,
        json_path=plan.json_path,
        processed_dir=plan.processed_dir,
        candidate_count=plan.candidate_count,
        chunk_count=plan.chunk_count,
        chunks=[],
        cached=False,
    )
    _write_prepared_marker(
        layout=layout,
        plan=prepared_plan,
        json_records=json_records,
    )
    CONF.output_volume.commit()

    # Compact artifacts are now durable; transient Rosetta and chunk files are not
    # needed by upstream inference because the processed pickle shards are complete.
    for transient in (
        _pdb_prep_dir(layout),
        layout.outputs_dir / f"{APP_INFO.input_stem}_pdb",
    ):
        if transient.exists():
            shutil.rmtree(transient)
    (layout.outputs_dir / f"{APP_INFO.input_stem}.csv").unlink(missing_ok=True)
    _candidate_csv_marker_path(layout).unlink(missing_ok=True)
    CONF.output_volume.commit()
    return prepared_plan


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_ensirna_inference(prepared_dir: str, force: bool = False) -> bytes:
    """Run ENsiRNA model inference on prepared volume-backed artifacts."""
    from uuid import uuid4

    import orjson

    layout = _layout_from_prepared_dir(prepared_dir)
    CONF.output_volume.reload()
    prepared_metadata = _prepared_metadata(layout)
    if prepared_metadata is None:
        missing = [
            str(path) for path in _required_prepared_paths(layout) if not path.exists()
        ]
        raise FileNotFoundError(f"ENsiRNA prepared inputs are incomplete: {missing}")
    cache_key = str(prepared_metadata["cache_key"])

    result_xlsx = layout.outputs_dir / "mrna_result.xlsx"
    if not force and _result_ready(layout, cache_key):
        return result_xlsx.read_bytes()

    _link_checkpoints()
    checkpoint_args = [
        str(APP_INFO.ensirna_dir / "pkl" / filename)
        for filename in APP_INFO.checkpoint_filenames
    ]
    staging_dir = _inference_prep_dir(layout) / uuid4().hex
    staging_dir.mkdir(parents=True)
    staging_result = staging_dir / result_xlsx.name
    try:
        run_command(
            [
                "micromamba",
                "run",
                "-n",
                APP_INFO.conda_env_name,
                "python",
                "run.py",
                "--ckpt",
                *checkpoint_args,
                "--test_set",
                str(layout.outputs_dir / f"{APP_INFO.input_stem}.json"),
                "--save_dir",
                str(staging_dir),
                "--gpu",
                "0",
                "--id",
                APP_INFO.input_stem,
            ],
            cwd=APP_INFO.ensirna_dir,
            output_mode="inherit",
        )
        if not staging_result.is_file() or staging_result.stat().st_size == 0:
            raise FileNotFoundError(f"ENsiRNA result XLSX not found: {staging_result}")
        staging_result.replace(result_xlsx)
        _atomic_write(
            _result_marker_path(layout),
            orjson.dumps({
                "schema_version": APP_INFO.cache_schema_version,
                "cache_key": cache_key,
                "size": result_xlsx.stat().st_size,
                "sha256": _file_sha256(result_xlsx),
            }),
        )
        CONF.output_volume.commit()
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    if not _result_ready(layout, cache_key):
        raise FileNotFoundError(f"ENsiRNA result XLSX not found: {result_xlsx}")
    return result_xlsx.read_bytes()


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with ENsiRNA functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh output state before accepting lifecycle methods."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive one staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read this Run's durable kernel overview."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume this Run without retrying failed Tasks."""
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
        """Persist a validated Successor request without driving it."""
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
        """Drive one previously prepared root or Successor Run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
    ) -> ExecutionOverview:
        """Create a compatible Successor while inferring predecessor identity."""
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
    ) -> EnsirnaExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: EnsirnaExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_claims=ENSIRNA_OUTPUT_CLAIMS,
                modal_driver=_coordinator_modal_driver(development=selected_mode),
                app_version=CONF.repo_commit_hash or CONF.version or "unknown",
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "download_ensirna_models": download_ensirna_models,
            "ensirna_prepare_inputs": ensirna_prepare_inputs,
            "ensirna_prepare_pdb_chunk": ensirna_prepare_pdb_chunk,
            "ensirna_finalize_prepared_inputs": ensirna_finalize_prepared_inputs,
            "ensirna_preprocess_dataset": ensirna_preprocess_dataset,
            "run_ensirna_inference": run_ensirna_inference,
        },
        workload_name="ENsiRNA",
    )


##########################################
# Local entrypoint client
##########################################
@app.local_entrypoint()
def submit_ensirna_task(
    mrna_fasta: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    prepare_workers: int = 4,
    pdb_cores: int = 1,
    preprocess_shard_size: int = APP_INFO.preprocess_shard_size,
    force: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Run ENsiRNA siRNA candidate design.

    Args:
        mrna_fasta: Local mRNA FASTA file to design siRNA candidates for.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the mRNA FASTA
            filename stem.
        prepare_workers: Maximum concurrent Modal containers used for Rosetta PDB
            preparation chunks.
        pdb_cores: Local Rosetta worker processes per preparation container. The
            product of this value and prepare_workers cannot exceed 64.
        preprocess_shard_size: Candidate records checkpointed per RNA-FM shard;
            completed preparation caches remain reusable across values.
        force: Rebuild prepared artifacts and rerun inference instead of using
            matching cached Modal volume outputs.
        use_deployed_coordinator: Target the exact deployed coordinator. The
            Biomodals CLI supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    input_path = Path(mrna_fasta).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"mRNA FASTA not found: {input_path}")
    run_name = run_name or input_path.stem
    local_out_dir = resolve_local_output_dir(out_dir)
    out_file = build_local_output_path(
        local_out_dir,
        run_name=run_name,
        extension=".xlsx",
        overwrite=force,
    )

    if not 1 <= prepare_workers <= APP_INFO.max_prepare_jobs:
        raise ValueError(
            f"prepare_workers must be between 1 and {APP_INFO.max_prepare_jobs}"
        )
    if not 1 <= pdb_cores <= APP_INFO.max_pdb_cores:
        raise ValueError(f"pdb_cores must be between 1 and {APP_INFO.max_pdb_cores}")
    if prepare_workers * pdb_cores > APP_INFO.max_total_pdb_cores:
        raise ValueError(
            "prepare_workers * pdb_cores must not exceed "
            f"{APP_INFO.max_total_pdb_cores}"
        )
    if preprocess_shard_size < 1:
        raise ValueError("preprocess_shard_size must be at least 1")

    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    request = EnsirnaExecutionRequest(
        run_name=run_name,
        fasta_content=_sanitize_fasta_for_upstream(input_path.read_bytes()),
        prepare_workers=prepare_workers,
        pdb_cores=pdb_cores,
        preprocess_shard_size=preprocess_shard_size,
        force_generation=uuid4().hex if force else None,
        app_version=CONF.repo_commit_hash or CONF.version or "unknown",
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    stage_execution_launch(
        CONF.output_volume,
        execution_run_id,
        predecessor_execution_run_id,
    )
    coordinator = _execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(development=not use_deployed_coordinator)
    else:
        call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
        )
    print(f"Execution Run ID: {execution_run_id}")
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    overview = call.get()
    if overview.run.status != RunStatus.SUCCEEDED:
        diagnostic = overview.run.status_message or (
            overview.run.status_reason.value
            if overview.run.status_reason is not None
            else overview.run.status.value
        )
        raise RuntimeError(
            f"{CONF.name} Execution Run ended as "
            f"{overview.run.status.value}: {diagnostic}"
        )

    cache_key = _cache_key_for_fasta(
        request.fasta_content,
        force_generation=request.force_generation,
    )
    result_path = _layout_for_cache_key(cache_key).outputs_dir / "mrna_result.xlsx"
    relative_result = result_path.relative_to(CONF.output_volume_mountpoint)
    xlsx_bytes = b"".join(CONF.output_volume.read_file(relative_result.as_posix()))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_file.with_name(f".{out_file.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(xlsx_bytes)
        temporary.replace(out_file)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"🧬 ENsiRNA run complete! Results saved to {out_file}")
