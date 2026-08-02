"""OligoFormer source repo: <https://github.com/lulab/OligoFormer>.

OligoFormer predicts siRNA efficacy from an mRNA FASTA file. This wrapper builds
the runtime from the upstream README and supports OligoFormer's off-target and
toxicity options for standalone runs.

## Off-target prediction

When `--off-target` is set, provide both `--utr-file` and `--orf-file`, or set
`--all-human` to use human references converted from TargetScan 8.0 UTR and ORF
data.

For positive `--top-n` values, Biomodals ranks candidates by predicted efficacy
first, then sends the actual sequence from each selected ranked row to PITA and
TargetScan. Candidate names use the original zero-based row identity
(`RNA<index>`), so off-target evidence can be merged back into the ranked table
without confusing top-N rank with candidate identity. `--top-n -1` scores every
candidate.

Advanced tuning is available through CLI flags rather than image environment
variables. `--targetscan-ref-shard-size` controls how many UTR records are put
into each TargetScan reference-preparation shard; when omitted, Biomodals uses
`--targetscan-prepare-nodes` to derive the shard size. Candidate, context,
RNAplfold, and PITA shard sizes can be tuned independently. The
`--off-target-process-slots` run-wide budget bounds concurrent TargetScan and
PITA subprocesses (default and maximum: 64); each branch receives half.

## Outputs

Results are saved locally as `<run-name>_oligoformer.tar.zst`. The tarball
contains the final top-level `.txt`, `_ranked.txt`, and `_ranked_filtered.txt`
tables only; detailed off-target logs stay in the Modal output-volume run
directory under `<stem>/<cache-key>/`. Final-table variants for different filter
thresholds reuse the same efficacy and merged off-target evidence; reference or
`top_n` variants reuse efficacy while building distinct evidence. After final
tables are generated, bulky off-target shard inputs under
`prepare/off_target/<stem>/` are removed. The app preserves compact merged PITA
and TargetScan evidence, final tables, completion markers, logs, efficacy
outputs, model caches, and reusable reference caches.

For each input mRNA record, the tarball contains three TSV tables:

- `<stem>.txt`: candidates in original mRNA-window order.
- `<stem>_ranked.txt`: all candidates sorted by predicted `efficacy`
  descending.
- `<stem>_ranked_filtered.txt`: candidates with `filter == 0`, sorted by
  predicted `efficacy` descending.

Output columns:

- `pos`: 1-based candidate index/window position in the mRNA scan. If a custom
  siRNA FASTA was provided, treat this mostly as input row order.
- `sense`: sense/passenger strand, computed as the complement of `siRNA`.
- `siRNA`: antisense/guide siRNA candidate sequence.
- `efficacy`: OligoFormer predicted efficacy score. Higher is better; treat it
  as a ranking score, not a guaranteed percent knockdown.
- `func_filter`: upstream functionality-rule filter. `0` passes. Nonzero means
  the sequence failed one of the rule checks: bad GC content, homopolymer run,
  GC/C-rich run, or palindromic sequence.
- `pita_score`: PITA off-target score, present with `--off-target`. More
  negative is worse; the default failure threshold is `< -10`.
- `targetscan_score`: TargetScan off-target score, present with `--off-target`.
  Higher is worse; the default failure threshold is `> 1`.
- `off_target_filter`: combined off-target flag, present with `--off-target`.
  `0` passes, `1` means predicted off-target risk, and `-5` means the candidate
  was not off-target evaluated because only the top `--top-n` candidates were
  scored.
- `Seed`: 6-mer seed used for toxicity lookup, present with `--toxicity`.
- `cell_viability`: toxicity table cell-viability value for `Seed`, present
  with `--toxicity`. Lower is worse; the default failure threshold is `< 50`.
- `toxicity_filter`: toxicity flag, present with `--toxicity`. `0` passes and
  `1` fails.
- `filter`: count of enabled filter failures. `0` means the candidate passed
  all enabled filters. Nonzero means reject or manually review.

Candidate selection:

Start from `<stem>_ranked_filtered.txt`, take the highest-`efficacy` rows, and
choose several candidates that are not all clustered at the same `pos`. For a
conservative therapeutic-style screen, require `filter == 0`, `func_filter == 0`,
`off_target_filter == 0` when off-target prediction is enabled, and
`toxicity_filter == 0` when toxicity prediction is enabled. If off-target
prediction was run with the default `--top-n 20`, only the top 20 efficacy
candidates were off-target scored; do not treat `off_target_filter == -5` as
safe. Rerun with a larger `--top-n` or `--top-n -1` before selecting lower-ranked
candidates from `<stem>_ranked.txt`.

"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import hashlib
import os
import re
import shlex
import shutil
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar, cast
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.app.score.oligoformer_execution import (
    OligoformerExecutionCoordinator,
    OligoformerExecutionRequest,
    load_execution_request,
    stage_execution_request,
)
from biomodals.execution import DeploymentIdentity, ExecutionSnapshot, RunStatus
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
from biomodals.helper.constant import MAX_TIMEOUT, MODEL_VOLUME
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import (
    package_outputs,
    run_command,
    sanitize_filename,
    warmup_directory,
)
from biomodals.helper.task_budget import bounded_map
from biomodals.helper.web import download_files

TARGETSCAN_RNAPLFOLD_MAX_NODES = 32
TARGETSCAN_RNAPLFOLD_MAX_WORKERS = 32
TARGETSCAN_RNAPLFOLD_MANIFEST_VERSION = 1
TARGETSCAN_RNAPLFOLD_CACHE_MARKER_VERSION = 2

T = TypeVar("T")
TARGETSCAN_COLUMNS = ("refseq", "siRNA", "targetscan_score")
TARGETSCAN_HEADER = "\t".join(TARGETSCAN_COLUMNS)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="OligoFormer",
    repo_url="https://github.com/lulab/OligoFormer",
    repo_commit_hash="e2f53ad63387bbe166bf123949151e2bc9bf6ec3",
    package_name="oligoformer",
    version="1.0",
    python_version="3.10",
    cuda_version="cu118",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "7200")),
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    """OligoFormer runtime paths and pinned dependencies."""

    requirements: tuple[str, ...] = (
        "bio==1.6.2",
        "matplotlib==3.7.5",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "prefetch-generator==1.0.3",
        "ptflops==0.7.3",
        "pytorch-ignite==0.5.0.post2",
        "scikit-learn==1.3.1",
        "scipy==1.10.1",
        "torch==2.2.1",
        "torchvision==0.17.1",
        "tqdm==4.66.1",
        "yacs==0.1.8",
    )
    rnafm_archive_url: str = (
        "https://cloud.tsinghua.edu.cn/f/46d71884ee8848b3a958/?dl=1"
    )
    viennarna_archive_url: str = (
        "https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_7_x/"
        "ViennaRNA-2.7.2.tar.gz"
    )
    targetscan_data_page_url: str = (
        "https://www.targetscan.org/cgi-bin/targetscan/data_download.vert80.cgi"
    )
    targetscan_utr_url: str = (
        "https://www.targetscan.org/vert_80/vert_80_data_download/UTR_Sequences.txt.zip"
    )
    targetscan_orf_url: str = (
        "https://www.targetscan.org/vert_80/vert_80_data_download/ORF_Sequences.txt.zip"
    )
    targetscan_species_id: str = "9606"
    targetscan_version: str = "8.0"
    off_target_cache_salt: str = (
        "targetscan-8.0-viennarna-2.7.2-semantic-topn-targetscan-polars-"
        "pita-ref-mount-v2"
    )
    postprocess_cache_salt: str = "final-tables-v3"
    repo_rnafm_dir: Path = CONF.git_clone_dir / "RNA-FM"
    model_rnafm_dir: Path = Path(CONF.model_volume_mountpoint) / "RNA-FM"
    repo_ref_dir: Path = CONF.git_clone_dir / "off-target/ref"
    model_ref_dir: Path = Path(CONF.model_volume_mountpoint) / "off-target/ref"
    human_ref_filenames: tuple[str, ...] = ("human_UTR.txt", "human_ORF.txt")
    default_top_n: int = 20
    prepared_marker_name: str = "oligoformer.json"
    default_off_target_nodes: int = 32
    default_off_target_workers_per_node: int = 32
    default_off_target_process_slots: int = 64
    max_off_target_process_slots: int = 64
    default_off_target_prep_workers: int = 16
    default_pita_prepare_nodes: int = 32
    default_pita_prepare_workers: int = 32
    default_pita_prepare_utr_shard_size: int = 1000
    default_pita_row_shard_size: int = 1000
    default_pita_row_attempts: int = 3
    default_targetscan_rnaplfold_nodes: int = 32
    default_targetscan_rnaplfold_workers: int = 8
    default_targetscan_rnaplfold_shard_size: int = 500
    default_targetscan_prepare_nodes: int = 32
    default_targetscan_candidate_shard_size: int = 20
    default_targetscan_context_nodes: int = 100
    default_targetscan_context_workers: int = 32
    default_targetscan_context_shard_size: int = 500
    default_targetscan_context_attempts: int = 3
    default_targetscan_merge_nodes: int = 16
    cache_lock_dict_name: str = f"{CONF.package_name}-cache-locks"
    cache_lock_poll_seconds: float = 5.0
    cache_lock_stale_seconds: float = MAX_TIMEOUT + 600

    @property
    def model_rnafm_redevelop_dir(self) -> Path:
        """Return the RNA-FM redevelop directory expected by OligoFormer."""
        return self.model_rnafm_dir / "redevelop"

    @property
    def repo_rnafm_redevelop_dir(self) -> Path:
        """Return the runtime RNA-FM redevelop directory inside OligoFormer."""
        return self.repo_rnafm_dir / "redevelop"

    @property
    def human_ref_downloads(self) -> dict[str, Path]:
        """Return TargetScan 8.0 full-human off-target reference zip downloads."""
        return {
            self.targetscan_utr_url: self.model_ref_dir
            / "TargetScan_8_0_UTR_Sequences.txt.zip",
            self.targetscan_orf_url: self.model_ref_dir
            / "TargetScan_8_0_ORF_Sequences.txt.zip",
        }

    @property
    def model_human_ref_paths(self) -> tuple[Path, ...]:
        """Return converted full-human off-target reference paths."""
        return tuple(self.model_ref_dir / name for name in self.human_ref_filenames)

    @property
    def model_rnafm_identity_path(self) -> Path:
        """Return the model-volume identity for the extracted RNA-FM tree."""
        return self.model_rnafm_dir / "biomodals_rnafm_identity.json"

    @property
    def rnafm_identity_path(self) -> Path:
        """Return the output-volume identity used by efficacy cache keys."""
        return (
            Path(CONF.output_volume_mountpoint)
            / "reference-cache"
            / "rnafm_identity.json"
        )

    @property
    def rnafm_identity_metadata(self) -> dict[str, object]:
        """Return semantic metadata identifying the RNA-FM model source."""
        return {
            "source_url": self.rnafm_archive_url,
            "upstream_commit": CONF.repo_commit_hash,
        }

    @property
    def targetscan_ref_marker_path(self) -> Path:
        """Return the marker for converted TargetScan 8.0 human references."""
        return self.model_ref_dir / "targetscan_8_0_human_refs.json"

    @property
    def targetscan_ref_identity_path(self) -> Path:
        """Return the output-volume identity for converted human references."""
        return (
            Path(CONF.output_volume_mountpoint)
            / "reference-cache"
            / "targetscan_8_0_human_refs.json"
        )

    @property
    def targetscan_rnaplfold_cache_dir(self) -> Path:
        """Return cached TargetScan 8.0 RNAplfold outputs for human UTRs."""
        return (
            Path(CONF.output_volume_mountpoint)
            / "reference-cache"
            / "targetscan_8_0_RNAplfold_in_out"
        )

    @property
    def targetscan_rnaplfold_shard_dir(self) -> Path:
        """Return shard inputs for building the RNAplfold cache."""
        return (
            Path(CONF.output_volume_mountpoint)
            / "reference-cache"
            / "targetscan_8_0_RNAplfold_shards"
        )

    @property
    def targetscan_rnaplfold_marker_path(self) -> Path:
        """Return the marker for cached TargetScan 8.0 RNAplfold outputs."""
        return (
            Path(CONF.output_volume_mountpoint)
            / "reference-cache"
            / "targetscan_8_0_rnaplfold_cache.json"
        )

    @property
    def targetscan_ref_metadata(self) -> dict[str, object]:
        """Return metadata identifying the converted TargetScan human refs."""
        return {
            "targetscan_version": self.targetscan_version,
            "species_id": self.targetscan_species_id,
            "source_page_url": self.targetscan_data_page_url,
            "source_urls": sorted(self.human_ref_downloads),
            "sequence_transform": "strip alignment gaps and write FASTA pairs",
            "output_files": list(self.human_ref_filenames),
        }

    @property
    def targetscan_rnaplfold_metadata(self) -> dict[str, object]:
        """Return metadata identifying cached RNAplfold reference outputs."""
        return {
            "targetscan_version": self.targetscan_version,
            "species_id": self.targetscan_species_id,
            "reference_metadata": self.targetscan_ref_metadata,
            "viennarna_url": self.viennarna_archive_url,
            "command": "RNAplfold -L 40 -W 80 -u 20",
            "input_file": "human_UTR.txt",
        }

    @property
    def stage_patch(self) -> str:
        """Return the source patch for explicit Biomodals stage selection."""
        return f"""from pathlib import Path

main_path = Path({str(CONF.git_clone_dir / "scripts/main.py")!r})
main_text = main_path.read_text()
main_old = '''    parser.add_argument('-i2','--infer_siRNA_fasta', nargs='?', const=False, help='siRNA fasta file to infer')
'''
main_new = main_old + '''    parser.add_argument('--biomodals_stage', choices=['full', 'efficacy'], default='full', help='Biomodals stage selector')
'''
if main_old not in main_text:
    raise SystemExit("expected OligoFormer main.py siRNA argument not found")
if "--biomodals_stage" not in main_text:
    main_text = main_text.replace(main_old, main_new)
main_path.write_text(main_text)

infer_path = Path({str(CONF.git_clone_dir / "scripts/infer.py")!r})
infer_text = infer_path.read_text()
infer_old = '''\\tif Args.all_human:
\\t\\tArgs.utr = './off-target/ref/human_UTR.txt'
\\t\\tArgs.orf = './off-target/ref/human_ORF.txt'
'''
infer_new = infer_old + '''\\tif getattr(Args, 'biomodals_stage', 'full') == 'efficacy':
\\t\\tArgs.off_target = False
\\t\\tArgs.toxicity = False
'''
if infer_old not in infer_text:
    raise SystemExit("expected OligoFormer infer.py all-human block not found")
if "biomodals_stage" not in infer_text:
    infer_text = infer_text.replace(infer_old, infer_new)
infer_path.write_text(infer_text)
"""

    @property
    def stage_patch_runner(self) -> str:
        """Return a Python one-liner that applies the stage-selection patch."""
        return f"exec({self.stage_patch!r})"


APP_INFO = AppInfo()


@dataclass(frozen=True, slots=True)
class OligoformerRunConfig:
    """Semantic configuration shared by OligoFormer compute and final stages."""

    off_target: bool = False
    toxicity: bool = False
    all_human: bool = False
    top_n: int = 20
    functionality_filter: bool = True
    pita_threshold: float = -10.0
    targetscan_threshold: float = 1.0
    toxicity_threshold: float = 50.0


@dataclass(frozen=True, slots=True)
class OligoformerExecutionConfig:
    """Result-neutral per-run OligoFormer fanout, sharding, and retry controls."""

    off_target_nodes: int = APP_INFO.default_off_target_nodes
    off_target_workers: int = APP_INFO.default_off_target_workers_per_node
    off_target_process_slots: int = APP_INFO.default_off_target_process_slots
    off_target_prep_workers: int = APP_INFO.default_off_target_prep_workers
    pita_prepare_nodes: int = APP_INFO.default_pita_prepare_nodes
    pita_prepare_workers: int = APP_INFO.default_pita_prepare_workers
    pita_prepare_utr_shard_size: int = APP_INFO.default_pita_prepare_utr_shard_size
    pita_row_shard_size: int = APP_INFO.default_pita_row_shard_size
    pita_row_attempts: int = APP_INFO.default_pita_row_attempts
    targetscan_rnaplfold_nodes: int = APP_INFO.default_targetscan_rnaplfold_nodes
    targetscan_rnaplfold_workers: int = APP_INFO.default_targetscan_rnaplfold_workers
    targetscan_rnaplfold_shard_size: int = (
        APP_INFO.default_targetscan_rnaplfold_shard_size
    )
    targetscan_prepare_nodes: int = APP_INFO.default_targetscan_prepare_nodes
    targetscan_candidate_shard_size: int = (
        APP_INFO.default_targetscan_candidate_shard_size
    )
    targetscan_context_nodes: int = APP_INFO.default_targetscan_context_nodes
    targetscan_context_workers: int = APP_INFO.default_targetscan_context_workers
    targetscan_context_shard_size: int = APP_INFO.default_targetscan_context_shard_size
    targetscan_context_attempts: int = APP_INFO.default_targetscan_context_attempts
    targetscan_merge_nodes: int = APP_INFO.default_targetscan_merge_nodes

    def __post_init__(self) -> None:
        """Reject misleading or unsafe per-run resource settings."""
        for name in self.__slots__:
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.off_target_process_slots < 2:
            raise ValueError(
                "off_target_process_slots must be at least 2 so TargetScan and "
                "PITA each receive one process slot"
            )
        if self.off_target_process_slots > APP_INFO.max_off_target_process_slots:
            raise ValueError(
                "off_target_process_slots must not exceed "
                f"{APP_INFO.max_off_target_process_slots}"
            )
        worker_limits = {
            "off_target_workers": 32,
            "off_target_prep_workers": 32,
            "pita_prepare_workers": 32,
            "targetscan_rnaplfold_workers": TARGETSCAN_RNAPLFOLD_MAX_WORKERS,
            "targetscan_context_workers": 32,
        }
        for name, limit in worker_limits.items():
            if getattr(self, name) > limit:
                raise ValueError(f"{name} must not exceed {limit}")
        if self.targetscan_rnaplfold_nodes > TARGETSCAN_RNAPLFOLD_MAX_NODES:
            raise ValueError(
                "targetscan_rnaplfold_nodes must not exceed "
                f"{TARGETSCAN_RNAPLFOLD_MAX_NODES}"
            )


DEFAULT_EXECUTION_CONFIG = OligoformerExecutionConfig()


@dataclass(frozen=True, slots=True)
class OligoformerRunPlan:
    """Volume-backed OligoFormer run plan."""

    cache_key: str
    efficacy_key: str
    run_root: str
    efficacy_dir: str
    output_dir: str
    output_stems: tuple[str, ...]
    config: OligoformerRunConfig
    postprocess_key: str
    efficacy_ready: bool
    evidence_ready: bool
    final_ready: bool
    reference_identity: str | None = None
    model_identity: str | None = None


@dataclass(frozen=True, slots=True)
class OffTargetSirnaRecord:
    """One siRNA FASTA record for OligoFormer off-target tools."""

    name: str
    sequence: str


@dataclass(frozen=True, slots=True)
class OffTargetShardResult:
    """Per-siRNA off-target output paths inside a temporary shard workdir."""

    index: int
    pita_path: str


@dataclass(frozen=True, slots=True)
class OffTargetShardSpec:
    """One siRNA off-target task backed by the shared output-volume cache."""

    run_root: str
    output_dir: str
    stem: str
    index: int
    record_name: str
    record_sequence: str
    utr_path: str
    orf_path: str
    row_shard_size: int


@dataclass(frozen=True, slots=True)
class TargetscanBatchSpec:
    """One TargetScan batch for a stem's selected siRNAs."""

    run_root: str
    output_dir: str
    stem: str
    ref_shard_size: int
    shard_index: int
    sirna_path: str
    sirna_count: int
    utr_path: str
    orf_path: str
    rnaplfold_cache_dir: str
    candidate_shard_size: int = APP_INFO.default_targetscan_candidate_shard_size
    candidate_shard_index: int = 0
    context_shard_size: int = APP_INFO.default_targetscan_context_shard_size


@dataclass(frozen=True, slots=True)
class TargetscanReferenceShard:
    """One transcript-aligned TargetScan reference shard."""

    ref_shard_size: int
    shard_index: int
    utr_path: str
    orf_path: str
    rnaplfold_cache_dir: str


@dataclass(frozen=True, slots=True)
class PitaRowShardSpec:
    """One cached PITA potential-target row shard."""

    run_root: str
    stem: str
    sirna_index: int
    record_name: str
    shard_index: int
    start_row: int
    end_row: int
    potential_targets_path: str
    input_path: str
    ext_utr_path: str
    output_path: str
    log_path: str


@dataclass(frozen=True, slots=True)
class PitaPrepareUtrShardSpec:
    """One cached PITA UTR shard for potential-target discovery."""

    shard_index: int
    input_path: str
    mir_stab_path: str
    output_path: str
    log_path: str


@dataclass(frozen=True, slots=True)
class TargetscanRnaPlfoldShardSpec:
    """One shard of TargetScan UTRs for RNAplfold cache preparation."""

    shard_index: int
    shard_path: str
    output_dir: str
    log_path: str


@dataclass(frozen=True, slots=True)
class TargetscanContextShardSpec:
    """One TargetScan context-score target-table shard."""

    shard_index: int
    common_dir: str
    targets_path: str
    output_path: str
    log_path: str
    rnaplfold_cache_dir: str


@dataclass(frozen=True, slots=True)
class PreparedTargetscanBatch:
    """Prepared TargetScan context shards for a siRNA batch."""

    targetscan_path: str
    logs_dir: str
    context_shards: tuple[TargetscanContextShardSpec, ...]
    needs_merge: bool


@dataclass(frozen=True, slots=True)
class PitaPreparePlan:
    """Prepared PITA target-discovery inputs for one siRNA."""

    spec: OffTargetShardSpec
    utr_shards: tuple[PitaPrepareUtrShardSpec, ...]
    row_count: int | None
    ext_utr_path: str = ""


@dataclass(frozen=True, slots=True)
class PitaReferencePlan:
    """Reusable PITA reference-only stabilization data for one run stem."""

    utr_shard_paths: tuple[str, ...]
    ext_utr_path: str


@dataclass(frozen=True, slots=True)
class PreparedOffTargetShard:
    """Cached per-siRNA off-target inputs ready for row-shard scoring."""

    index: int
    record_name: str
    cache_dir: str
    logs_dir: str
    pita_path: str
    row_shards: tuple[PitaRowShardSpec, ...]


@dataclass(frozen=True, slots=True)
class OligoformerReferencePlan:
    """Finite RNAplfold reference-shard publication plan."""

    record_count: int
    shard_specs: tuple[TargetscanRnaPlfoldShardSpec, ...]


@dataclass(frozen=True, slots=True)
class OligoformerEvidenceStemPlan:
    """Deterministic PITA and TargetScan Tasks for one efficacy output."""

    stem: str
    pita_specs: tuple[OffTargetShardSpec, ...]
    targetscan_specs: tuple[TargetscanBatchSpec, ...]


@dataclass(frozen=True, slots=True)
class OligoformerEvidencePlan:
    """Finite off-target Task plan discovered after efficacy prediction."""

    stems: tuple[OligoformerEvidenceStemPlan, ...]


def _hash_bytes(data: bytes | None) -> str:
    """Return a stable hash for optional bytes."""
    if data is None:
        return ""
    return hashlib.sha256(data).hexdigest()


def _hash_path(path: Path) -> str:
    """Return a stable SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    """Return a stable digest of relative names and bytes in one directory."""
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _fasta_record_names(fasta_bytes: bytes) -> tuple[str, ...]:
    """Return safe upstream-normalized FASTA record names."""
    return _sanitize_fasta_record_names(fasta_bytes, label="mRNA")[1]


def _safe_fasta_record_name(raw_name: str, *, label: str) -> str:
    """Return one unique-path-safe upstream FASTA identifier."""
    if not raw_name:
        raise ValueError(f"OligoFormer {label} FASTA record names cannot be empty")
    flattened = sanitize_filename(raw_name.replace(" ", "_@_"))
    safe_name = re.sub(r"[^A-Za-z0-9._@-]+", "_", flattened).strip("._-")
    safe_name = re.sub(r"__+", "_", safe_name)
    if not safe_name:
        raise ValueError(
            f"OligoFormer {label} FASTA record names must contain safe characters"
        )
    if len(safe_name) > 180:
        safe_name = f"{safe_name[:160]}_{hash_string(raw_name)[:16]}"
    return safe_name


def _sanitize_fasta_record_names(
    fasta_bytes: bytes,
    *,
    label: str,
) -> tuple[bytes, tuple[str, ...]]:
    """Rewrite FASTA headers to unique shell- and path-safe components."""
    lines = fasta_bytes.decode("utf-8").splitlines()
    names: list[str] = []
    rewritten: list[str] = []
    for line in lines:
        if not line.startswith(">"):
            rewritten.append(line)
            continue
        raw_name = line[1:].strip()
        safe_name = _safe_fasta_record_name(raw_name, label=label)
        if safe_name in names:
            raise ValueError(
                f"OligoFormer {label} FASTA record names must be unique after "
                f"sanitization; duplicate output name: {safe_name}"
            )
        names.append(safe_name)
        rewritten.append(f">{safe_name}")
    if not names:
        raise ValueError(f"OligoFormer {label} FASTA must contain at least one record")
    suffix = "\n" if fasta_bytes.endswith((b"\n", b"\r")) else ""
    return ("\n".join(rewritten) + suffix).encode(), tuple(names)


def _sanitize_reference_fasta(
    fasta_bytes: bytes,
    *,
    label: str,
) -> tuple[bytes, dict[str, str]]:
    """Validate and canonicalize one custom off-target reference FASTA."""
    records: list[tuple[str, str]] = []
    original_names_by_safe_name: dict[str, str] = {}
    current_raw_name: str | None = None
    sequence_chunks: list[str] = []

    def _finish_record() -> None:
        if current_raw_name is None:
            return
        if not sequence_chunks:
            raise ValueError(
                f"OligoFormer {label} FASTA record {current_raw_name!r} is empty"
            )
        sequence = "".join(sequence_chunks).upper()
        if re.fullmatch(r"[ACGTURYSWKMBDHVN]+", sequence) is None:
            raise ValueError(
                f"OligoFormer {label} FASTA sequences must use IUPAC nucleotide "
                "characters"
            )
        safe_name = _safe_fasta_record_name(current_raw_name, label=label)
        if safe_name in original_names_by_safe_name:
            raise ValueError(
                f"OligoFormer {label} FASTA record names must be unique after "
                f"sanitization; duplicate output name: {safe_name}"
            )
        original_names_by_safe_name[safe_name] = current_raw_name
        records.append((safe_name, sequence))

    for raw_line in fasta_bytes.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if raw_line.startswith(">"):
            _finish_record()
            current_raw_name = line[1:].strip()
            sequence_chunks = []
            continue
        if current_raw_name is None:
            raise ValueError(
                f"OligoFormer {label} FASTA sequence appears before its header"
            )
        sequence_chunks.append(line)
    _finish_record()
    if not records:
        raise ValueError(f"OligoFormer {label} FASTA must contain at least one record")
    canonical = "".join(f">{name}\n{sequence}\n" for name, sequence in records)
    return canonical.encode(), original_names_by_safe_name


def _sanitize_paired_reference_fastas(
    utr_bytes: bytes,
    orf_bytes: bytes,
) -> tuple[bytes, bytes]:
    """Canonicalize paired custom UTR/ORF references without ambiguous joins."""
    safe_utr, utr_names = _sanitize_reference_fasta(utr_bytes, label="UTR reference")
    safe_orf, orf_names = _sanitize_reference_fasta(orf_bytes, label="ORF reference")
    for safe_name in utr_names.keys() & orf_names.keys():
        if utr_names[safe_name] != orf_names[safe_name]:
            raise ValueError(
                "OligoFormer paired UTR and ORF records that sanitize to the same "
                f"identifier must use the same original name: {safe_name}"
            )
    return safe_utr, safe_orf


def _run_layout_for_cache_key(
    cache_key: str,
    output_stems: tuple[str, ...],
) -> AppRunLayout:
    """Return the output-volume run layout for an OligoFormer cache key."""
    if not output_stems:
        raise ValueError("OligoFormer run layout requires at least one output stem")
    return AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / output_stems[0] / cache_key
    )


def _efficacy_layout_for_key(
    efficacy_key: str,
    output_stems: tuple[str, ...],
) -> AppRunLayout:
    """Return the shared output-volume layout for one efficacy input."""
    return AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint)
        / "efficacy-cache"
        / output_stems[0]
        / efficacy_key
    )


def _marker_path(layout: AppRunLayout, marker: str) -> Path:
    """Return an OligoFormer cache marker path."""
    return layout.markers_dir / marker


def _final_marker_name(postprocess_key: str) -> str:
    """Return the completion marker name for one final-table variant."""
    return f"final.{postprocess_key}.done"


def _output_paths(output_dir: Path, output_stems: tuple[str, ...]) -> tuple[Path, ...]:
    """Return upstream OligoFormer output paths for all FASTA records."""
    paths = []
    for stem in output_stems:
        paths.extend((
            output_dir / f"{stem}.txt",
            output_dir / f"{stem}_ranked.txt",
            output_dir / f"{stem}_ranked_filtered.txt",
        ))
    return tuple(paths)


def _output_bundle_paths(output_stems: tuple[str, ...]) -> tuple[Path, ...]:
    """Return final table paths to include in the local result tarball."""
    return tuple(
        Path(f"{stem}{suffix}.txt")
        for stem in output_stems
        for suffix in ("", "_ranked", "_ranked_filtered")
    )


def _package_output_tables(output_dir: Path, output_stems: tuple[str, ...]) -> bytes:
    """Package only final OligoFormer result tables, excluding diagnostic logs."""
    bundle_paths = _output_bundle_paths(output_stems)
    missing = [
        str(output_dir / path)
        for path in bundle_paths
        if not (output_dir / path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "OligoFormer final output tables are incomplete: " + ", ".join(missing)
        )
    return package_outputs(output_dir, paths_to_bundle=bundle_paths)


def _marker_matches(marker: Path, expected: dict[str, object]) -> bool:
    """Return whether a cache marker contains expected metadata."""
    import orjson

    if not marker.exists():
        return False
    try:
        metadata = orjson.loads(marker.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if not isinstance(metadata, dict):
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def _paths_ready(
    paths: tuple[Path, ...],
    marker: Path,
    *,
    expected_marker: dict[str, object] | None = None,
) -> bool:
    """Return whether a tabular output bundle matches its manifest."""
    import orjson

    if not paths or not marker.is_file() or not all(path.is_file() for path in paths):
        return False
    try:
        manifest = orjson.loads(marker.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or (
            expected_marker is not None
            and not all(
                manifest.get(key) == value for key, value in expected_marker.items()
            )
        )
    ):
        return False
    tables = manifest.get("tables")
    if not isinstance(tables, list):
        return False
    try:
        actual_tables = [
            _tabular_output_metadata(path, root=paths[0].parent) for path in paths
        ]
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        UnicodeDecodeError,
        ValueError,
    ):
        return False
    if tables != actual_tables:
        return False
    required_columns = {"pos", "sense", "siRNA", "efficacy", "func_filter", "filter"}
    for start in range(0, len(actual_tables), 3):
        table_group = actual_tables[start : start + 3]
        if len(table_group) != 3:
            return False
        schemas = [cast(list[str], table["columns"]) for table in table_group]
        if schemas[0] != schemas[1] or schemas[0] != schemas[2]:
            return False
        if not required_columns.issubset(schemas[0]):
            return False
        if table_group[0]["row_count"] != table_group[1]["row_count"]:
            return False
        if cast(int, table_group[2]["row_count"]) > cast(
            int, table_group[0]["row_count"]
        ):
            return False
    return True


def _tabular_output_metadata(path: Path, *, root: Path) -> dict[str, object]:
    """Return streaming identity, schema, and row facts for one TSV output."""
    digest = hashlib.sha256()
    columns: list[str] | None = None
    row_count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle):
            digest.update(line)
            if line_number == 0:
                columns = line.rstrip(b"\r\n").decode("utf-8").split("\t")
            else:
                row_count += 1
    if columns is None or not columns or columns == [""]:
        raise ValueError(f"OligoFormer output table is empty: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "columns": columns,
        "row_count": row_count,
    }


def _publish_output_bundle_marker(
    marker: Path,
    *,
    output_dir: Path,
    paths: tuple[Path, ...],
    identity: dict[str, object],
) -> None:
    """Atomically publish a deeply validated output-table bundle manifest."""
    import orjson

    tables = [_tabular_output_metadata(path, root=output_dir) for path in paths]
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(marker)
    try:
        tmp_path.write_bytes(orjson.dumps({"version": 1, **identity, "tables": tables}))
        tmp_path.replace(marker)
    finally:
        tmp_path.unlink(missing_ok=True)


def _discard_output_bundle(marker: Path, paths: tuple[Path, ...]) -> None:
    """Invalidate a corrupt published table bundle before rebuilding it."""
    marker.unlink(missing_ok=True)
    for path in paths:
        path.unlink(missing_ok=True)


def _build_plan(
    cache_key: str,
    efficacy_key: str,
    output_stems: tuple[str, ...],
    run_root: str | Path | None = None,
    *,
    config: OligoformerRunConfig,
    postprocess_key: str,
    reference_identity: str | None = None,
    model_identity: str | None = None,
) -> OligoformerRunPlan:
    """Build an OligoFormer run plan from current volume state."""
    layout = (
        AppRunLayout.from_run_root(run_root)
        if run_root is not None
        else _run_layout_for_cache_key(cache_key, output_stems)
    )
    efficacy_layout = _efficacy_layout_for_key(efficacy_key, output_stems)
    efficacy_dir = efficacy_layout.outputs_dir
    output_dir = layout.outputs_dir / postprocess_key
    return OligoformerRunPlan(
        cache_key=cache_key,
        efficacy_key=efficacy_key,
        run_root=str(layout.run_root),
        efficacy_dir=str(efficacy_dir),
        output_dir=str(output_dir),
        output_stems=output_stems,
        config=config,
        postprocess_key=postprocess_key,
        efficacy_ready=_paths_ready(
            _output_paths(efficacy_dir, output_stems),
            _marker_path(efficacy_layout, "efficacy.done"),
            expected_marker={
                "efficacy_key": efficacy_key,
                "output_stems": list(output_stems),
            },
        ),
        evidence_ready=(
            not config.off_target
            or all(
                _raw_off_target_ready(
                    layout.prep_dir / "off_target" / stem,
                    expected_identity=_off_target_evidence_identity(
                        layout.run_root,
                        stem,
                    ),
                )
                for stem in output_stems
            )
        ),
        final_ready=_paths_ready(
            _output_paths(output_dir, output_stems),
            _marker_path(layout, _final_marker_name(postprocess_key)),
            expected_marker={
                "cache_key": cache_key,
                "postprocess_key": postprocess_key,
                "output_stems": list(output_stems),
                "postprocess_cache_salt": APP_INFO.postprocess_cache_salt,
            },
        ),
        reference_identity=reference_identity,
        model_identity=model_identity,
    )


def _load_rnafm_identity(path: Path) -> dict[str, object]:
    """Load and validate one persisted RNA-FM content identity."""
    import orjson

    try:
        identity = orjson.loads(path.read_bytes())
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
        raise FileNotFoundError(
            "OligoFormer RNA-FM identity is missing. Run "
            "download_oligoformer_models first."
        ) from exc
    except orjson.JSONDecodeError as exc:
        raise FileNotFoundError(
            "OligoFormer RNA-FM identity is invalid. Run "
            "download_oligoformer_models first."
        ) from exc
    if (
        not isinstance(identity, dict)
        or not all(
            identity.get(key) == value
            for key, value in APP_INFO.rnafm_identity_metadata.items()
        )
        or not isinstance(identity.get("content_sha256"), str)
    ):
        raise FileNotFoundError(
            "OligoFormer RNA-FM identity is stale. Run "
            "download_oligoformer_models first."
        )
    return identity


def _rnafm_model_identity() -> dict[str, object]:
    """Return the output-volume content identity for RNA-FM weights."""
    return _load_rnafm_identity(APP_INFO.rnafm_identity_path)


def _rnafm_model_identity_digest() -> str:
    """Return the canonical RNA-FM identity digest for efficacy cache keys."""
    import orjson

    return _hash_bytes(
        orjson.dumps(_rnafm_model_identity(), option=orjson.OPT_SORT_KEYS)
    )


def _oligoformer_model_volume_identity_digest() -> str:
    """Return the exact RNA-FM identity currently on the model volume."""
    import orjson

    return _hash_bytes(
        orjson.dumps(
            _load_rnafm_identity(APP_INFO.model_rnafm_identity_path),
            option=orjson.OPT_SORT_KEYS,
        )
    )


def _rnafm_model_identity_matches_model() -> bool:
    """Return whether output and model-volume RNA-FM identities match."""
    import orjson

    try:
        model_identity = orjson.loads(APP_INFO.model_rnafm_identity_path.read_bytes())
        return (
            isinstance(model_identity, dict)
            and model_identity == _rnafm_model_identity()
        )
    except (FileNotFoundError, orjson.JSONDecodeError):
        return False


def _targetscan_human_refs_ready() -> bool:
    """Return whether converted TargetScan 8.0 human refs are available."""
    import orjson

    if not all(path.is_file() for path in APP_INFO.model_human_ref_paths):
        return False
    if not APP_INFO.targetscan_ref_marker_path.is_file():
        return False
    try:
        marker = orjson.loads(APP_INFO.targetscan_ref_marker_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if not isinstance(marker, dict):
        return False
    expected = APP_INFO.targetscan_ref_metadata
    if not all(marker.get(key) == value for key, value in expected.items()):
        return False
    content_sha256 = marker.get("content_sha256")
    if not isinstance(content_sha256, dict):
        return False
    return all(
        content_sha256.get(path.name) == _hash_path(path)
        for path in APP_INFO.model_human_ref_paths
    )


def _load_targetscan_ref_identity(path: Path) -> dict[str, object]:
    """Load and validate one persisted full-human reference identity."""
    import orjson

    try:
        identity = orjson.loads(path.read_bytes())
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
        raise FileNotFoundError(
            "OligoFormer full-human reference identity is missing. Run "
            "download_oligoformer_models first."
        ) from exc
    except orjson.JSONDecodeError as exc:
        raise FileNotFoundError(
            "OligoFormer full-human reference identity is invalid. Run "
            "download_oligoformer_models first."
        ) from exc
    if not isinstance(identity, dict):
        raise FileNotFoundError(
            "OligoFormer full-human reference identity is invalid. Run "
            "download_oligoformer_models first."
        )
    expected = APP_INFO.targetscan_ref_metadata
    content_sha256 = identity.get("content_sha256")
    if not all(identity.get(key) == value for key, value in expected.items()) or not (
        isinstance(content_sha256, dict)
        and all(
            isinstance(content_sha256.get(name), str)
            for name in APP_INFO.human_ref_filenames
        )
    ):
        raise FileNotFoundError(
            "OligoFormer full-human reference identity is stale. Run "
            "download_oligoformer_models first."
        )
    return identity


def _targetscan_ref_identity() -> dict[str, object]:
    """Return the output-volume content identity for full-human references."""
    return _load_targetscan_ref_identity(APP_INFO.targetscan_ref_identity_path)


def _targetscan_ref_identity_digest() -> str:
    """Return the canonical digest pinned by full-human evidence plans."""
    import orjson

    return _hash_bytes(
        orjson.dumps(_targetscan_ref_identity(), option=orjson.OPT_SORT_KEYS)
    )


def _oligoformer_reference_volume_identity_digest() -> str:
    """Return the exact TargetScan identity currently on the model volume."""
    import orjson

    return _hash_bytes(
        orjson.dumps(
            _load_targetscan_ref_identity(APP_INFO.targetscan_ref_marker_path),
            option=orjson.OPT_SORT_KEYS,
        )
    )


def _targetscan_ref_identity_matches_model() -> bool:
    """Return whether output identity matches committed model reference bytes."""
    import orjson

    if not _targetscan_human_refs_ready():
        return False
    try:
        model_identity = orjson.loads(APP_INFO.targetscan_ref_marker_path.read_bytes())
        output_identity = _targetscan_ref_identity()
    except (FileNotFoundError, orjson.JSONDecodeError):
        return False
    return isinstance(model_identity, dict) and output_identity == model_identity


def _targetscan_rnaplfold_expected_metadata() -> dict[str, object]:
    """Return RNAplfold cache metadata tied to the converted UTR bytes."""
    identity = _targetscan_ref_identity()
    content_sha256 = cast(dict[str, object], identity["content_sha256"])
    return APP_INFO.targetscan_rnaplfold_metadata | {
        "input_sha256": content_sha256["human_UTR.txt"]
    }


def _convert_targetscan_zip_to_fasta(
    zip_path: Path,
    output_path: Path,
    *,
    transcript_col: int,
    species_col: int,
    sequence_col: int,
) -> int:
    """Convert one TargetScan 8.0 table zip to an OligoFormer FASTA-like ref."""
    import zipfile

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    row_count = 0
    with zipfile.ZipFile(zip_path) as ref_zip:
        member = next(
            name
            for name in ref_zip.namelist()
            if not name.endswith("/") and name.endswith(".txt")
        )
        with (
            ref_zip.open(member) as source,
            tmp_path.open("w", encoding="utf-8") as out,
        ):
            next(source)
            for raw_line in source:
                fields = raw_line.decode("utf-8").rstrip("\r\n").split("\t")
                if len(fields) <= max(transcript_col, species_col, sequence_col):
                    continue
                if fields[species_col].strip() != APP_INFO.targetscan_species_id:
                    continue
                transcript_id = fields[transcript_col].strip()
                sequence = (
                    fields[sequence_col]
                    .strip()
                    .upper()
                    .replace("-", "")
                    .replace("T", "U")
                )
                if not transcript_id or not sequence:
                    continue
                out.write(f">{transcript_id}\n{sequence}\n")
                row_count += 1
    if row_count == 0:
        raise RuntimeError(
            f"TargetScan 8.0 conversion produced no human rows from {zip_path}"
        )
    tmp_path.replace(output_path)
    return row_count


def _convert_targetscan_human_refs() -> dict[str, int]:
    """Convert downloaded TargetScan 8.0 UTR and ORF zips to human refs."""
    downloads = APP_INFO.human_ref_downloads
    utr_count = _convert_targetscan_zip_to_fasta(
        downloads[APP_INFO.targetscan_utr_url],
        APP_INFO.model_ref_dir / "human_UTR.txt",
        transcript_col=0,
        species_col=3,
        sequence_col=4,
    )
    orf_count = _convert_targetscan_zip_to_fasta(
        downloads[APP_INFO.targetscan_orf_url],
        APP_INFO.model_ref_dir / "human_ORF.txt",
        transcript_col=0,
        species_col=1,
        sequence_col=2,
    )
    return {"human_UTR.txt": utr_count, "human_ORF.txt": orf_count}


def _read_fasta_pairs(path: Path) -> list[tuple[str, str]]:
    """Read simple two-line FASTA-like records."""
    records = []
    name = ""
    chunks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name:
                records.append((name, "".join(chunks)))
            name = line[1:]
            chunks = []
        else:
            chunks.append(line)
    if name:
        records.append((name, "".join(chunks)))
    return records


def _targetscan_rnaplfold_output_name(record_name: str) -> str:
    """Return the cache filename for one TargetScan transcript."""
    if not record_name or "/" in record_name or "\0" in record_name:
        raise ValueError(f"Unsafe TargetScan transcript identifier: {record_name!r}")
    return f"{record_name}.{APP_INFO.targetscan_species_id}_lunp"


def _targetscan_rnaplfold_shard_marker_path(
    spec: TargetscanRnaPlfoldShardSpec,
) -> Path:
    """Return the deterministic integrity marker for one RNAplfold shard."""
    return Path(spec.shard_path).with_suffix(".manifest.json")


def _targetscan_rnaplfold_shard_spec(
    shard_index: int,
) -> TargetscanRnaPlfoldShardSpec:
    """Return the deterministic on-volume paths for one RNAplfold shard."""
    shard_dir = APP_INFO.targetscan_rnaplfold_shard_dir
    return TargetscanRnaPlfoldShardSpec(
        shard_index=shard_index,
        shard_path=str(shard_dir / f"{shard_index:05d}.fa"),
        output_dir=str(APP_INFO.targetscan_rnaplfold_cache_dir),
        log_path=str(shard_dir / "logs" / f"{shard_index:05d}.log"),
    )


def _targetscan_rnaplfold_file_metadata(path: Path) -> dict[str, object]:
    """Return nonempty-file metadata for an RNAplfold cache artifact."""
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"OligoFormer RNAplfold artifact is empty: {path}")
    return {"size": size, "sha256": _hash_path(path)}


def _targetscan_rnaplfold_shard_state(
    spec: TargetscanRnaPlfoldShardSpec,
    *,
    verify_output_hashes: bool,
) -> tuple[bool, set[str], int]:
    """Return shard readiness, valid outputs, and expected output count."""
    import orjson

    shard_path = Path(spec.shard_path)
    marker_path = _targetscan_rnaplfold_shard_marker_path(spec)
    try:
        records = _read_fasta_pairs(shard_path)
        output_names = [_targetscan_rnaplfold_output_name(name) for name, _ in records]
        marker = orjson.loads(marker_path.read_bytes())
        expected_reference = _targetscan_rnaplfold_expected_metadata()
        shard_metadata = {
            "name": shard_path.name,
            **_targetscan_rnaplfold_file_metadata(shard_path),
        }
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        orjson.JSONDecodeError,
    ):
        return False, set(), 0
    if (
        not records
        or len(output_names) != len(set(output_names))
        or not isinstance(marker, dict)
        or marker.get("version") != TARGETSCAN_RNAPLFOLD_MANIFEST_VERSION
        or marker.get("shard_index") != spec.shard_index
        or marker.get("reference") != expected_reference
        or marker.get("shard_input") != shard_metadata
    ):
        return False, set(), len(output_names)
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(output_names):
        return False, set(), len(output_names)

    output_dir = Path(spec.output_dir)
    valid_outputs: set[str] = set()
    for output_name in output_names:
        metadata = outputs.get(output_name)
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("size"), int)
            or metadata["size"] <= 0
            or not isinstance(metadata.get("sha256"), str)
            or len(metadata["sha256"]) != 64
        ):
            continue
        output_path = output_dir / output_name
        try:
            if (
                not output_path.is_file()
                or output_path.stat().st_size != metadata["size"]
            ):
                continue
            if verify_output_hashes and _hash_path(output_path) != metadata["sha256"]:
                continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        valid_outputs.add(output_name)
    return len(valid_outputs) == len(output_names), valid_outputs, len(output_names)


def _publish_targetscan_rnaplfold_shard_manifest(
    spec: TargetscanRnaPlfoldShardSpec,
) -> None:
    """Atomically publish exact input and output integrity for one shard."""
    import orjson

    shard_path = Path(spec.shard_path)
    records = _read_fasta_pairs(shard_path)
    output_names = [_targetscan_rnaplfold_output_name(name) for name, _ in records]
    if not records or len(output_names) != len(set(output_names)):
        raise ValueError(
            f"OligoFormer RNAplfold shard {spec.shard_index} has invalid records"
        )
    output_dir = Path(spec.output_dir)
    outputs = {
        output_name: _targetscan_rnaplfold_file_metadata(output_dir / output_name)
        for output_name in output_names
    }
    marker_path = _targetscan_rnaplfold_shard_marker_path(spec)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(marker_path)
    try:
        tmp_path.write_bytes(
            orjson.dumps({
                "version": TARGETSCAN_RNAPLFOLD_MANIFEST_VERSION,
                "shard_index": spec.shard_index,
                "reference": _targetscan_rnaplfold_expected_metadata(),
                "shard_input": {
                    "name": shard_path.name,
                    **_targetscan_rnaplfold_file_metadata(shard_path),
                },
                "outputs": outputs,
            })
        )
        tmp_path.replace(marker_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _publish_targetscan_rnaplfold_cache_marker(
    shard_specs: list[TargetscanRnaPlfoldShardSpec],
    *,
    record_count: int,
) -> None:
    """Atomically publish the exact set of completed RNAplfold shards."""
    import orjson

    if record_count <= 0 or not shard_specs:
        raise ValueError("OligoFormer RNAplfold cache marker requires records")
    shard_markers = []
    validated_count = 0
    for expected_index, spec in enumerate(shard_specs):
        if spec.shard_index != expected_index:
            raise ValueError("OligoFormer RNAplfold shard indexes must be contiguous")
        ready, _, output_count = _targetscan_rnaplfold_shard_state(
            spec,
            verify_output_hashes=False,
        )
        if not ready:
            raise RuntimeError(
                f"OligoFormer RNAplfold shard {spec.shard_index} is incomplete"
            )
        validated_count += output_count
        marker_path = _targetscan_rnaplfold_shard_marker_path(spec)
        shard_markers.append({
            "index": spec.shard_index,
            "name": marker_path.name,
            **_targetscan_rnaplfold_file_metadata(marker_path),
        })
    if validated_count != record_count:
        raise RuntimeError(
            "OligoFormer RNAplfold shard manifests do not cover every record"
        )

    marker_path = APP_INFO.targetscan_rnaplfold_marker_path
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(marker_path)
    try:
        tmp_path.write_bytes(
            orjson.dumps(
                _targetscan_rnaplfold_expected_metadata()
                | {
                    "version": TARGETSCAN_RNAPLFOLD_CACHE_MARKER_VERSION,
                    "record_count": record_count,
                    "shard_count": len(shard_specs),
                    "shard_markers": shard_markers,
                }
            )
        )
        tmp_path.replace(marker_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _targetscan_rnaplfold_cache_ready() -> bool:
    """Return whether every manifest-bound RNAplfold output is ready."""
    import orjson

    marker_path = APP_INFO.targetscan_rnaplfold_marker_path
    cache_dir = APP_INFO.targetscan_rnaplfold_cache_dir
    if not marker_path.is_file() or not cache_dir.is_dir():
        return False
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    try:
        expected = _targetscan_rnaplfold_expected_metadata()
    except FileNotFoundError:
        return False
    if (
        not isinstance(marker, dict)
        or marker.get("version") != TARGETSCAN_RNAPLFOLD_CACHE_MARKER_VERSION
        or not all(marker.get(key) == value for key, value in expected.items())
    ):
        return False
    record_count = marker.get("record_count")
    shard_count = marker.get("shard_count")
    shard_markers = marker.get("shard_markers")
    if (
        not isinstance(record_count, int)
        or record_count <= 0
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or not isinstance(shard_markers, list)
        or len(shard_markers) != shard_count
    ):
        return False
    validated_count = 0
    for shard_index, recorded_marker in enumerate(shard_markers):
        spec = _targetscan_rnaplfold_shard_spec(shard_index)
        shard_marker_path = _targetscan_rnaplfold_shard_marker_path(spec)
        if not isinstance(recorded_marker, dict):
            return False
        try:
            actual_marker = {
                "index": shard_index,
                "name": shard_marker_path.name,
                **_targetscan_rnaplfold_file_metadata(shard_marker_path),
            }
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, ValueError):
            return False
        if recorded_marker != actual_marker:
            return False
        ready, _, output_count = _targetscan_rnaplfold_shard_state(
            spec,
            verify_output_hashes=False,
        )
        if not ready:
            return False
        validated_count += output_count
    return validated_count == record_count


def _is_model_human_ref_pair(utr_path: str, orf_path: str) -> bool:
    """Return whether the paths point at the converted all-human refs."""
    return (
        Path(utr_path) == APP_INFO.model_ref_dir / "human_UTR.txt"
        and Path(orf_path) == APP_INFO.model_ref_dir / "human_ORF.txt"
    )


##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "ca-certificates",
        "curl",
        "pkg-config",
        "unzip",
        "perl",
        "libstatistics-lite-perl",
        "libbio-perl-perl",
        "zstd",
    )
    .env(CONF.default_env)
    .run_commands(
        " && ".join((
            "tmpdir=$(mktemp -d)",
            f"curl -fsSL {shlex.quote(APP_INFO.viennarna_archive_url)} "
            '-o "$tmpdir/ViennaRNA-2.7.2.tar.gz"',
            'tar -xzf "$tmpdir/ViennaRNA-2.7.2.tar.gz" -C "$tmpdir"',
            'cd "$tmpdir/ViennaRNA-2.7.2"',
            "./configure --without-gsl --without-swig --without-perl "
            "--without-python --without-doc --without-svm --disable-openmp "
            "--disable-unittests --disable-check-executables",
            "make -j$(nproc)",
            "make install",
            "ldconfig",
            "RNAplfold --version",
            "cd /",
            'rm -rf "$tmpdir"',
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "grep -q \"Args.orf = './off-target/ref/human_UTR.txt'\" scripts/infer.py",
            "sed -i \"s|Args.orf = './off-target/ref/human_UTR.txt'|Args.orf = './off-target/ref/human_ORF.txt'|\" scripts/infer.py",
            'grep -q "for i in range(Args.top_n):" scripts/infer.py',
            'sed -i "s|for i in range(Args.top_n):|for i in range(min(Args.top_n, RESULT_ranked.shape[0])):|g" scripts/infer.py',
            f"python -c {shlex.quote(APP_INFO.stage_patch_runner)}",
            "rm -f off-target/ref/human_UTR.txt.zip "
            "off-target/ref/human_ORF.txt.zip "
            "off-target/ref/human_UTR.txt "
            "off-target/ref/human_ORF.txt",
            "cd off-target/pita",
            "make install",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(*APP_INFO.requirements)
    .pipe(
        patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3", "modal"]
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
OLIGOFORMER_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_oligoformer_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8


##########################################
# Fetch model weights
##########################################
def _download_oligoformer_models_locked(force: bool) -> None:
    """Download model assets while holding the global reference-state lock."""
    import shutil

    import orjson

    CONF.output_volume.reload()
    refs_ready = _targetscan_human_refs_ready()
    model_identity_ready = (
        APP_INFO.model_rnafm_redevelop_dir.is_dir()
        and _rnafm_model_identity_matches_model()
    )
    try:
        output_ref_identity = _targetscan_ref_identity()
        model_ref_identity = orjson.loads(
            APP_INFO.targetscan_ref_marker_path.read_bytes()
        )
        ref_identity_ready = refs_ready and output_ref_identity == model_ref_identity
    except (FileNotFoundError, orjson.JSONDecodeError):
        ref_identity_ready = False
    if (
        APP_INFO.model_rnafm_redevelop_dir.is_dir()
        and model_identity_ready
        and refs_ready
        and ref_identity_ready
        and not force
    ):
        print("💊 OligoFormer models and TargetScan 8.0 human refs already available")
        return

    if APP_INFO.model_rnafm_dir.exists() and force:
        shutil.rmtree(APP_INFO.model_rnafm_dir)

    if force or not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        with TemporaryDirectory(prefix="oligoformer_models_") as tmpdir:
            archive_path = Path(tmpdir) / "RNA-FM.tar.gz"
            download_files(
                {APP_INFO.rnafm_archive_url: archive_path},
                force=True,
                num_retries=3,
                progress_bar_desc="OligoFormer model downloads",
            )
            APP_INFO.model_rnafm_dir.parent.mkdir(parents=True, exist_ok=True)
            run_command([
                "tar",
                "-xzf",
                str(archive_path),
                "-C",
                str(APP_INFO.model_rnafm_dir.parent),
            ])

    model_identity_to_publish: dict[str, object] | None = None
    if force or not APP_INFO.model_rnafm_identity_path.is_file():
        model_identity_to_publish = APP_INFO.rnafm_identity_metadata | {
            "content_sha256": _hash_directory(APP_INFO.model_rnafm_redevelop_dir)
        }
        APP_INFO.model_rnafm_identity_path.write_bytes(
            orjson.dumps(model_identity_to_publish)
        )
    elif not model_identity_ready:
        loaded_model_identity = orjson.loads(
            APP_INFO.model_rnafm_identity_path.read_bytes()
        )
        if not isinstance(loaded_model_identity, dict):
            raise FileNotFoundError("OligoFormer RNA-FM model identity is invalid")
        model_identity_to_publish = loaded_model_identity

    identity_to_publish: dict[str, object] | None = None
    if force or not refs_ready:
        APP_INFO.model_ref_dir.mkdir(parents=True, exist_ok=True)
        download_files(
            APP_INFO.human_ref_downloads,
            force=force,
            num_retries=3,
            progress_bar_desc="OligoFormer human ref downloads",
        )
        row_counts = _convert_targetscan_human_refs()
        identity = APP_INFO.targetscan_ref_metadata | {
            "row_counts": row_counts,
            "content_sha256": {
                path.name: _hash_path(path) for path in APP_INFO.model_human_ref_paths
            },
        }
        APP_INFO.targetscan_ref_marker_path.write_bytes(orjson.dumps(identity))
        identity_to_publish = identity
        print(
            "💊 Converted TargetScan 8.0 human refs: "
            + ", ".join(f"{name}={count}" for name, count in row_counts.items())
        )

    if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        raise FileNotFoundError(
            "OligoFormer RNA-FM weights were not extracted to "
            f"{APP_INFO.model_rnafm_redevelop_dir}"
        )
    if not _targetscan_human_refs_ready():
        raise FileNotFoundError(
            "OligoFormer TargetScan 8.0 human refs were not converted under "
            f"{APP_INFO.model_ref_dir}"
        )
    missing_refs = [
        str(path) for path in APP_INFO.model_human_ref_paths if not path.is_file()
    ]
    if missing_refs:
        raise FileNotFoundError(
            "OligoFormer full-human refs were not extracted: " + ", ".join(missing_refs)
        )

    MODEL_VOLUME.commit()
    if model_identity_to_publish is not None:
        APP_INFO.rnafm_identity_path.parent.mkdir(parents=True, exist_ok=True)
        APP_INFO.rnafm_identity_path.write_bytes(
            orjson.dumps(model_identity_to_publish)
        )
    if identity_to_publish is not None or not ref_identity_ready:
        if identity_to_publish is not None:
            shutil.rmtree(APP_INFO.targetscan_rnaplfold_cache_dir, ignore_errors=True)
            shutil.rmtree(APP_INFO.targetscan_rnaplfold_shard_dir, ignore_errors=True)
            APP_INFO.targetscan_rnaplfold_marker_path.unlink(missing_ok=True)
            identity_bytes = orjson.dumps(identity_to_publish)
        else:
            identity_bytes = APP_INFO.targetscan_ref_marker_path.read_bytes()
        APP_INFO.targetscan_ref_identity_path.parent.mkdir(parents=True, exist_ok=True)
        APP_INFO.targetscan_ref_identity_path.write_bytes(identity_bytes)
    if (
        model_identity_to_publish is not None
        or identity_to_publish is not None
        or not ref_identity_ready
    ):
        CONF.output_volume.commit()
    print("💊 OligoFormer models and TargetScan 8.0 human refs committed")


@app.function(
    volumes=CONF.mounts(output_volume=True, model_volume=True, model_ro=False),
    timeout=MAX_TIMEOUT,
)
def download_oligoformer_models(force: bool = False) -> None:
    """Download RNA-FM weights and TargetScan 8.0 human refs into the model volume."""
    if not force:
        CONF.output_volume.reload()
        MODEL_VOLUME.reload()
        if (
            APP_INFO.model_rnafm_redevelop_dir.is_dir()
            and _rnafm_model_identity_matches_model()
            and _targetscan_ref_identity_matches_model()
        ):
            print("💊 OligoFormer models and TargetScan 8.0 human refs available")
            return
    with _cache_build_lock(
        "targetscan-reference-state",
        "global",
        rebuild=True,
    ) as owns_reference_state:
        if not owns_reference_state:
            raise RuntimeError(
                "OligoFormer reference-state publication was not serialized"
            )
        MODEL_VOLUME.reload()
        _download_oligoformer_models_locked(force)


def _oligoformer_models_ready() -> bool:
    """Return whether model and converted-reference identities are available."""
    return (
        APP_INFO.model_rnafm_redevelop_dir.is_dir()
        and _rnafm_model_identity_matches_model()
        and _targetscan_ref_identity_matches_model()
    )


def _run_rnaplfold_for_record(
    *,
    name: str,
    sequence: str,
    output_dir: Path,
    workdir: Path,
    log_path: Path,
) -> bool:
    """Run RNAplfold for one TargetScan UTR record if the output is missing."""
    import shutil
    import subprocess as sp

    output_path = output_dir / _targetscan_rnaplfold_output_name(name)
    species = APP_INFO.targetscan_species_id
    if output_path.exists() and output_path.stat().st_size > 0:
        return False

    fasta_bytes = f">{name}.{species}\n{sequence}\n".encode()
    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    tmp_output_path.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rnaplfold_bin = shutil.which("RNAplfold")
    if rnaplfold_bin is None:
        raise FileNotFoundError("RNAplfold is not available on PATH")
    with log_path.open("ab") as log:
        log.write(f"Running RNAplfold for {name}.{species}\n".encode())
        sp.run(  # noqa: S603 - RNAplfold is resolved from PATH and input is FASTA bytes.
            [rnaplfold_bin, "-L", "40", "-W", "80", "-u", "20"],
            input=fasta_bytes,
            cwd=workdir,
            stdout=log,
            stderr=log,
            check=True,
        )

    generated_lunp = workdir / f"{name}.{species}_lunp"
    generated_ps = workdir / f"{name}.{species}_dp.ps"
    if not generated_lunp.is_file():
        raise FileNotFoundError(f"RNAplfold did not produce {generated_lunp}")
    shutil.copy2(generated_lunp, tmp_output_path)
    tmp_output_path.replace(output_path)
    generated_lunp.unlink(missing_ok=True)
    generated_ps.unlink(missing_ok=True)
    return True


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_targetscan_rnaplfold_shard(
    spec: TargetscanRnaPlfoldShardSpec,
    local_workers: int = APP_INFO.default_targetscan_rnaplfold_workers,
) -> int:
    """Populate one shard of cached TargetScan RNAplfold outputs."""
    CONF.output_volume.reload()
    records = _read_fasta_pairs(Path(spec.shard_path))
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    _, valid_outputs, _ = _targetscan_rnaplfold_shard_state(
        spec,
        verify_output_hashes=False,
    )
    marker_path = _targetscan_rnaplfold_shard_marker_path(spec)
    marker_path.unlink(missing_ok=True)
    for name, _ in records:
        output_name = _targetscan_rnaplfold_output_name(name)
        if output_name not in valid_outputs:
            (output_dir / output_name).unlink(missing_ok=True)
    local_workers = _targetscan_rnaplfold_worker_count(local_workers)
    print(
        "💊 Running OligoFormer TargetScan RNAplfold shard "
        f"{spec.shard_index} with {len(records)} UTRs using "
        f"{local_workers} workers; log: {log_path}"
    )
    with TemporaryDirectory(
        prefix=f"oligoformer_rnaplfold_{spec.shard_index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        created_flags = bounded_map(
            records,
            lambda record: _run_rnaplfold_for_record(
                name=record[0],
                sequence=record[1],
                output_dir=output_dir,
                workdir=workdir,
                log_path=log_path,
            ),
            max_parallel=local_workers,
        )
    created = sum(created_flags)
    _publish_targetscan_rnaplfold_shard_manifest(spec)
    CONF.output_volume.commit()
    print(
        "💊 OligoFormer TargetScan RNAplfold shard "
        f"{spec.shard_index} created {created}/{len(records)} outputs"
    )
    return created


def _plan_targetscan_rnaplfold_cache(
    force: bool,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> OligoformerReferencePlan:
    """Persist deterministic RNAplfold shard inputs for kernel dispatch."""
    import shutil

    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    _ensure_human_refs()

    cache_dir = APP_INFO.targetscan_rnaplfold_cache_dir
    shard_dir = APP_INFO.targetscan_rnaplfold_shard_dir
    marker_path = APP_INFO.targetscan_rnaplfold_marker_path
    if force:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
        marker_path.unlink(missing_ok=True)

    records = _read_fasta_pairs(APP_INFO.model_ref_dir / "human_UTR.txt")
    if not records:
        raise RuntimeError("TargetScan human UTR refs contain no records")

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = execution.targetscan_rnaplfold_shard_size
    shard_specs: list[TargetscanRnaPlfoldShardSpec] = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[start : start + shard_size]
        spec = _targetscan_rnaplfold_shard_spec(shard_index)
        shard_path = Path(spec.shard_path)
        tmp_shard_path = _unique_tmp_path(shard_path)
        try:
            with tmp_shard_path.open("w", encoding="utf-8") as handle:
                for name, sequence in shard_records:
                    handle.write(f">{name}\n{sequence}\n")
            if not shard_path.is_file() or _hash_path(shard_path) != _hash_path(
                tmp_shard_path
            ):
                tmp_shard_path.replace(shard_path)
        finally:
            tmp_shard_path.unlink(missing_ok=True)
        shard_specs.append(spec)

    CONF.output_volume.commit()
    return OligoformerReferencePlan(
        record_count=len(records),
        shard_specs=tuple(shard_specs),
    )


def _publish_targetscan_rnaplfold_cache(
    plan: OligoformerReferencePlan,
) -> None:
    """Validate every RNAplfold shard before publishing the top marker."""
    missing = [
        spec
        for spec in plan.shard_specs
        if not _targetscan_rnaplfold_shard_state(
            spec,
            verify_output_hashes=False,
        )[0]
    ]
    if missing:
        raise FileNotFoundError(
            "OligoFormer RNAplfold reference cache is missing "
            f"{len(missing)} shards; first missing shard: {missing[0].shard_index}"
        )
    _publish_targetscan_rnaplfold_cache_marker(
        list(plan.shard_specs),
        record_count=plan.record_count,
    )
    CONF.output_volume.commit()


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def plan_oligoformer_targetscan_rnaplfold_cache(
    force: bool = False,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> OligoformerReferencePlan:
    """Publish finite RNAplfold shard inputs for the execution kernel."""
    return _plan_targetscan_rnaplfold_cache(force, execution)


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_oligoformer_targetscan_rnaplfold_cache(
    plan: OligoformerReferencePlan,
) -> None:
    """Publish the validated top-level RNAplfold cache marker."""
    CONF.output_volume.reload()
    _publish_targetscan_rnaplfold_cache(plan)


##########################################
# Inference functions
##########################################
def _ensure_rnafm_runtime() -> None:
    """Copy RNA-FM into the writable checkout expected by upstream."""
    import shutil

    if APP_INFO.repo_rnafm_dir.is_symlink():
        APP_INFO.repo_rnafm_dir.unlink()
    elif APP_INFO.repo_rnafm_dir.exists():
        shutil.rmtree(APP_INFO.repo_rnafm_dir)
    APP_INFO.repo_rnafm_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(APP_INFO.model_rnafm_dir, APP_INFO.repo_rnafm_dir)


def _ensure_human_refs() -> None:
    """Validate TargetScan 8.0 full-human refs in the model volume."""
    missing_refs = [
        str(path) for path in APP_INFO.model_human_ref_paths if not path.is_file()
    ]
    if missing_refs:
        raise FileNotFoundError(
            "OligoFormer full-human refs are missing. Run "
            "download_oligoformer_models first: " + ", ".join(missing_refs)
        )
    if not _targetscan_human_refs_ready():
        raise FileNotFoundError(
            "OligoFormer full-human refs were not converted from TargetScan 8.0. "
            "Run download_oligoformer_models first."
        )


def _efficacy_key_for_run(
    *,
    mrna_fasta_bytes: bytes,
    sirna_fasta_bytes: bytes | None,
    functionality_filter: bool,
    model_identity: str | None = None,
    force_generation: str | None = None,
) -> str:
    """Return a deterministic key for GPU efficacy outputs."""
    return hash_string(
        "\n".join((
            CONF.name,
            CONF.version or "",
            CONF.repo_commit_hash or "",
            f"mrna:{_hash_bytes(mrna_fasta_bytes)}",
            f"sirna:{_hash_bytes(sirna_fasta_bytes)}",
            f"functionality_filter:{int(functionality_filter)}",
            f"model_identity:{model_identity or ''}",
            f"force_generation:{force_generation or ''}",
        ))
    )


def _cache_key_for_run(
    *,
    efficacy_key: str,
    utr_bytes: bytes | None,
    orf_bytes: bytes | None,
    config: OligoformerRunConfig,
    reference_identity: str | None = None,
) -> str:
    """Return a deterministic key for reusable off-target evidence."""
    ref_parts = ["off-target=0"]
    if config.off_target:
        ref_parts = [f"all-human={int(config.all_human)}"]
        if config.all_human:
            if reference_identity is None:
                raise ValueError(
                    "Full-human off-target caching requires reference identity"
                )
            ref_parts.append(f"human-ref-identity:{reference_identity}")
        else:
            ref_parts.extend((
                f"utr:{_hash_bytes(utr_bytes)}",
                f"orf:{_hash_bytes(orf_bytes)}",
            ))

    return hash_string(
        "\n".join((
            efficacy_key,
            f"off_target:{int(config.off_target)}",
            "off_target_cache_salt:"
            f"{APP_INFO.off_target_cache_salt if config.off_target else ''}",
            f"top_n:{config.top_n if config.off_target else ''}",
            *ref_parts,
        ))
    )


def _postprocess_key_for_run(
    *,
    cache_key: str,
    config: OligoformerRunConfig,
) -> str:
    """Return a key for one final-table filtering configuration."""
    return hash_string(
        "\n".join((
            cache_key,
            APP_INFO.postprocess_cache_salt,
            f"off_target:{int(config.off_target)}",
            f"toxicity:{int(config.toxicity)}",
            f"functionality_filter:{int(config.functionality_filter)}",
            f"pita_threshold:{config.pita_threshold if config.off_target else ''}",
            "targetscan_threshold:"
            f"{config.targetscan_threshold if config.off_target else ''}",
            "toxicity_threshold:"
            f"{config.toxicity_threshold if config.toxicity else ''}",
        ))
    )


def _write_cache_marker(
    layout: AppRunLayout,
    marker: str,
    plan: OligoformerRunPlan,
    *,
    extra_metadata: dict[str, object] | None = None,
):
    """Publish a deeply validated final-table cache marker."""
    metadata = {
        "cache_key": plan.cache_key,
        "postprocess_key": plan.postprocess_key,
        "output_stems": list(plan.output_stems),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    output_dir = Path(plan.output_dir)
    _publish_output_bundle_marker(
        _marker_path(layout, marker),
        output_dir=output_dir,
        paths=_output_paths(output_dir, plan.output_stems),
        identity=metadata,
    )


def _copy_outputs(src_dir: Path, dst_dir: Path, output_stems: tuple[str, ...]) -> None:
    """Copy upstream OligoFormer output files between cache stages."""
    import shutil

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in _output_paths(src_dir, output_stems):
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def _read_efficacy_output(path: Path) -> pl.DataFrame:
    """Read efficacy output with legacy float parsing for stable output diffs."""
    import pandas as pd

    return pl.from_pandas(pd.read_csv(path, sep="\t"))


def _off_target_sirna_records(
    result: pl.DataFrame, top_n: int
) -> list[OffTargetSirnaRecord]:
    """Return siRNA FASTA records for off-target tools."""
    if top_n == -1:
        return [
            OffTargetSirnaRecord(name=f"RNA{int(pos) - 1}", sequence=str(sirna))
            for pos, sirna in result.select("pos", "siRNA").iter_rows()
        ]

    ranked_rows = (
        result
        .with_row_index("_biomodals_index")
        .sort("efficacy", descending=True)
        .head(top_n)
        .select("_biomodals_index", "siRNA")
    )
    return [
        OffTargetSirnaRecord(
            name=f"RNA{int(index)}",
            sequence=str(sirna),
        )
        for index, sirna in ranked_rows.iter_rows()
    ]


def _write_sirna_records(records: list[OffTargetSirnaRecord], sirna_file: Path) -> None:
    """Write siRNA FASTA records for off-target tools."""
    sirna_file.parent.mkdir(parents=True, exist_ok=True)
    with sirna_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.name}\n")
            handle.write(f"{record.sequence}\n")


def _targetscan_ref_shard_size(
    utr_count: int,
    configured_size: int | None = None,
    *,
    prepare_nodes: int = APP_INFO.default_targetscan_prepare_nodes,
) -> int:
    """Choose a TargetScan reference shard size from explicit tuning or fanout."""
    if utr_count < 1:
        raise ValueError("TargetScan requires at least one UTR record")
    if configured_size is not None:
        if configured_size < 1:
            raise ValueError("targetscan_ref_shard_size must be a positive integer")
        return configured_size

    if prepare_nodes < 1:
        raise ValueError("prepare_nodes must be a positive integer")
    return max(1, (utr_count + prepare_nodes - 1) // prepare_nodes)


def _bounded_node_count(task_count: int, configured_nodes: int) -> int:
    """Return a per-run Modal node count capped to the task count."""
    if configured_nodes < 1:
        raise ValueError("configured_nodes must be a positive integer")
    if task_count < 1:
        return 1
    return max(1, min(configured_nodes, task_count))


def _validated_worker_count(
    value: int,
    *,
    name: str = "local_workers",
    maximum: int = 32,
) -> int:
    """Return a worker count that fits the static Modal CPU envelope."""
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _targetscan_rnaplfold_worker_count(configured_workers: int) -> int:
    """Return validated RNAplfold workers per node."""
    return _validated_worker_count(
        configured_workers,
        name="targetscan_rnaplfold_workers",
        maximum=TARGETSCAN_RNAPLFOLD_MAX_WORKERS,
    )


def _targetscan_rnaplfold_node_count(task_count: int, configured_nodes: int) -> int:
    """Return validated RNAplfold fanout within the deployment limit."""
    if not 1 <= configured_nodes <= TARGETSCAN_RNAPLFOLD_MAX_NODES:
        raise ValueError(
            "targetscan_rnaplfold_nodes must be between 1 and "
            f"{TARGETSCAN_RNAPLFOLD_MAX_NODES}"
        )
    return _bounded_node_count(task_count, configured_nodes)


def _unique_tmp_path(path: Path) -> Path:
    """Return a collision-resistant temporary sibling path."""
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")


def _artifact_file_metadata(path: Path) -> dict[str, object]:
    """Return streaming integrity metadata for one reusable artifact."""
    line_count = 0
    digest = hashlib.sha256()
    last_byte = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        line_count += 1
    return {
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "line_count": line_count,
    }


def _artifact_marker_ready(
    marker_path: Path,
    *,
    kind: str,
    artifacts: dict[str, Path],
    identity: dict[str, object] | None = None,
) -> bool:
    """Return whether exact expected artifacts match an atomic marker."""
    import orjson

    if not marker_path.is_file() or not all(
        path.is_file() for path in artifacts.values()
    ):
        return False
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if (
        not isinstance(marker, dict)
        or marker.get("version") != 1
        or marker.get("kind") != kind
        or marker.get("identity") != (identity or {})
        or set(marker.get("artifacts", {})) != set(artifacts)
    ):
        return False
    try:
        actual = {
            name: _artifact_file_metadata(path) for name, path in artifacts.items()
        }
    except (FileNotFoundError, NotADirectoryError):
        return False
    return marker["artifacts"] == actual


def _publish_artifact_marker(
    marker_path: Path,
    *,
    kind: str,
    artifacts: dict[str, Path],
    identity: dict[str, object] | None = None,
) -> None:
    """Atomically publish exact reusable-artifact integrity metadata."""
    import orjson

    metadata = {name: _artifact_file_metadata(path) for name, path in artifacts.items()}
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(marker_path)
    try:
        tmp_path.write_bytes(
            orjson.dumps({
                "version": 1,
                "kind": kind,
                "identity": identity or {},
                "artifacts": metadata,
            })
        )
        tmp_path.replace(marker_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _warmup_file_parents(
    paths: Iterable[str | Path],
    *,
    file_pattern: str,
) -> None:
    """Warm unique parent directories before reading matching file contents."""
    for parent in sorted({Path(path).parent for path in paths}):
        warmup_directory(parent, file_pattern=file_pattern)


def _off_target_shard_cache_dir(spec: OffTargetShardSpec) -> Path:
    """Return the shared-cache directory for one siRNA off-target shard."""
    return (
        AppRunLayout.from_run_root(spec.run_root).prep_dir
        / "off_target"
        / spec.stem
        / f"{spec.index:05d}_{spec.record_name}"
    )


def _off_target_shard_logs_dir(spec: OffTargetShardSpec) -> Path:
    """Return the packaged log directory for one siRNA off-target shard."""
    return (
        Path(spec.output_dir)
        / "logs"
        / "off_target"
        / spec.stem
        / f"{spec.index:05d}_{spec.record_name}"
    )


def _off_target_shard_spec(
    *,
    run_root: str,
    output_dir: Path,
    stem: str,
    item: tuple[int, OffTargetSirnaRecord],
    utr_path: str,
    orf_path: str,
    row_shard_size: int,
) -> OffTargetShardSpec:
    """Build a serializable off-target task spec for one siRNA."""
    index, record = item
    return OffTargetShardSpec(
        run_root=run_root,
        output_dir=str(output_dir),
        stem=stem,
        index=index,
        record_name=record.name,
        record_sequence=record.sequence,
        utr_path=utr_path,
        orf_path=orf_path,
        row_shard_size=row_shard_size,
    )


def _write_fasta_pairs(records: list[tuple[str, str]], path: Path) -> None:
    """Write simple FASTA-like records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def _targetscan_reference_shards(
    *,
    run_root: str,
    stem: str,
    utr_path: str,
    orf_path: str,
    ref_shard_size: int | None = None,
    prepare_nodes: int = APP_INFO.default_targetscan_prepare_nodes,
) -> list[TargetscanReferenceShard]:
    """Persist transcript-aligned TargetScan reference shards once per run."""
    layout = AppRunLayout.from_run_root(run_root)
    off_target_root = layout.prep_dir / "off_target" / stem
    shard_root = off_target_root / "targetscan_ref_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    utr_records = _read_fasta_pairs(Path(utr_path))
    orf_records_by_name = dict(_read_fasta_pairs(Path(orf_path)))
    shard_size = _targetscan_ref_shard_size(
        len(utr_records), ref_shard_size, prepare_nodes=prepare_nodes
    )
    rnaplfold_cache_dir = ""
    if _is_model_human_ref_pair(utr_path, orf_path):
        if not _targetscan_rnaplfold_cache_ready():
            raise FileNotFoundError("OligoFormer TargetScan RNAplfold cache is missing")
        rnaplfold_cache_dir = str(APP_INFO.targetscan_rnaplfold_cache_dir)

    shards = []
    for shard_index, start in enumerate(range(0, len(utr_records), shard_size)):
        shard_utr_records = utr_records[start : start + shard_size]
        if not shard_utr_records:
            raise RuntimeError(f"TargetScan reference shard {shard_index} is empty")
        shard_orf_records = [
            (name, orf_records_by_name[name])
            for name, _sequence in shard_utr_records
            if name in orf_records_by_name
        ]
        shard_dir = shard_root / f"{shard_index:05d}"
        shard_utr_path = shard_dir / "UTR.fa"
        shard_orf_path = shard_dir / "ORF.fa"
        marker_path = shard_dir / "refs.done"
        utr_bytes = "".join(
            f">{name}\n{sequence}\n" for name, sequence in shard_utr_records
        ).encode()
        orf_bytes = "".join(
            f">{name}\n{sequence}\n" for name, sequence in shard_orf_records
        ).encode()
        marker_text = (
            f"ref_shard_size={shard_size}\n"
            f"start={start}\n"
            f"utr_records={len(shard_utr_records)}\n"
            f"orf_records={len(shard_orf_records)}\n"
            f"utr_sha256={_hash_bytes(utr_bytes)}\n"
            f"orf_sha256={_hash_bytes(orf_bytes)}\n"
        )
        existing_marker = (
            marker_path.read_text(encoding="utf-8") if marker_path.exists() else None
        )
        reference_files_ready = (
            existing_marker == marker_text
            and shard_utr_path.is_file()
            and shard_utr_path.stat().st_size == len(utr_bytes)
            and _hash_path(shard_utr_path) == _hash_bytes(utr_bytes)
            and shard_orf_path.is_file()
            and shard_orf_path.stat().st_size == len(orf_bytes)
            and _hash_path(shard_orf_path) == _hash_bytes(orf_bytes)
        )
        if not reference_files_ready:
            shard_dir.mkdir(parents=True, exist_ok=True)
            for path, contents in (
                (shard_utr_path, utr_bytes),
                (shard_orf_path, orf_bytes),
            ):
                tmp_path = _unique_tmp_path(path)
                try:
                    tmp_path.write_bytes(contents)
                    tmp_path.replace(path)
                finally:
                    tmp_path.unlink(missing_ok=True)
            tmp_marker_path = _unique_tmp_path(marker_path)
            try:
                tmp_marker_path.write_text(marker_text, encoding="utf-8")
                tmp_marker_path.replace(marker_path)
            finally:
                tmp_marker_path.unlink(missing_ok=True)
        shards.append(
            TargetscanReferenceShard(
                ref_shard_size=shard_size,
                shard_index=shard_index,
                utr_path=str(shard_utr_path),
                orf_path=str(shard_orf_path),
                rnaplfold_cache_dir=rnaplfold_cache_dir,
            )
        )
    return shards


def _targetscan_batch_spec_waves(
    *,
    run_root: str,
    output_dir: Path,
    stem: str,
    records: list[OffTargetSirnaRecord],
    utr_path: str,
    orf_path: str,
    max_tiles_per_wave: int,
    ref_shard_size: int | None = None,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> Iterable[list[TargetscanBatchSpec]]:
    """Yield bounded candidate/reference tiles without materializing the product."""
    if max_tiles_per_wave < 1:
        raise ValueError("max_tiles_per_wave must be a positive integer")
    if not records:
        return

    reference_shards = _targetscan_reference_shards(
        run_root=run_root,
        stem=stem,
        utr_path=utr_path,
        orf_path=orf_path,
        ref_shard_size=ref_shard_size,
        prepare_nodes=execution.targetscan_prepare_nodes,
    )
    candidate_shard_size = execution.targetscan_candidate_shard_size
    off_target_root = (
        AppRunLayout.from_run_root(run_root).prep_dir / "off_target" / stem
    )
    for candidate_shard_index, start in enumerate(
        range(0, len(records), candidate_shard_size)
    ):
        # Never mix candidate shards in one wave: each reference shard then has
        # exactly one writer, and the committed first candidate wave publishes
        # reference-only preparation before the next candidate wave can reuse it.
        wave: list[TargetscanBatchSpec] = []
        candidate_records = records[start : start + candidate_shard_size]
        sirna_path = (
            off_target_root
            / "targetscan_siRNA_shards"
            / f"{candidate_shard_index:05d}.fa"
        )
        _write_sirna_records(candidate_records, sirna_path)
        for reference in reference_shards:
            wave.append(
                TargetscanBatchSpec(
                    run_root=run_root,
                    output_dir=str(output_dir),
                    stem=stem,
                    ref_shard_size=reference.ref_shard_size,
                    shard_index=reference.shard_index,
                    sirna_path=str(sirna_path),
                    sirna_count=len(candidate_records),
                    utr_path=reference.utr_path,
                    orf_path=reference.orf_path,
                    rnaplfold_cache_dir=reference.rnaplfold_cache_dir,
                    candidate_shard_size=candidate_shard_size,
                    candidate_shard_index=candidate_shard_index,
                    context_shard_size=execution.targetscan_context_shard_size,
                )
            )
            if len(wave) == max_tiles_per_wave:
                yield wave
                wave = []
        if wave:
            yield wave


def _targetscan_batch_specs(
    *,
    run_root: str,
    output_dir: Path,
    stem: str,
    records: list[OffTargetSirnaRecord],
    utr_path: str,
    orf_path: str,
    ref_shard_size: int | None = None,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> list[TargetscanBatchSpec]:
    """Build all TargetScan tiles for tests and small direct callers."""
    return [
        spec
        for wave in _targetscan_batch_spec_waves(
            run_root=run_root,
            output_dir=output_dir,
            stem=stem,
            records=records,
            utr_path=utr_path,
            orf_path=orf_path,
            ref_shard_size=ref_shard_size,
            max_tiles_per_wave=max(1, len(records)),
            execution=execution,
        )
        for spec in wave
    ]


def _targetscan_batch_cache_dir(spec: TargetscanBatchSpec) -> Path:
    """Return the shared-cache directory for one TargetScan batch."""
    return (
        AppRunLayout.from_run_root(spec.run_root).prep_dir
        / "off_target"
        / spec.stem
        / "targetscan"
        / f"candidates_{spec.candidate_shard_size}"
        / f"{spec.candidate_shard_index:05d}"
        / f"size_{spec.ref_shard_size}"
        / f"{spec.shard_index:05d}"
    )


def _targetscan_batch_logs_dir(spec: TargetscanBatchSpec) -> Path:
    """Return the packaged log directory for one TargetScan batch."""
    return (
        Path(spec.output_dir)
        / "logs"
        / "off_target"
        / spec.stem
        / "targetscan"
        / f"candidates_{spec.candidate_shard_size}"
        / f"{spec.candidate_shard_index:05d}"
        / f"size_{spec.ref_shard_size}"
        / f"{spec.shard_index:05d}"
    )


def _batch_items_for_local_workers(  # noqa: UP047 - runtime image uses Python 3.10.
    items: list[T],
    *,
    max_nodes: int,
    local_workers: int,
) -> list[list[T]]:
    """Split work so each Modal node can keep local workers busy."""
    if not items:
        return []
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    if local_workers < 1:
        raise ValueError("local_workers must be at least 1")

    worker_sized_node_count = (len(items) + local_workers - 1) // local_workers
    if worker_sized_node_count <= max_nodes:
        batch_size = local_workers
    else:
        batch_size = (len(items) + max_nodes - 1) // max_nodes
    return [
        items[index : index + batch_size] for index in range(0, len(items), batch_size)
    ]


def _bounded_worker_topology(
    *,
    task_count: int,
    configured_nodes: int,
    configured_workers: int,
    max_process_slots: int | None,
) -> tuple[int, int]:
    """Cap Modal nodes times local workers to one branch-wide slot budget."""
    if task_count < 1:
        return 0, 1
    if configured_nodes < 1 or configured_workers < 1:
        raise ValueError("Configured nodes and workers must be positive")
    slot_count = (
        configured_nodes * configured_workers
        if max_process_slots is None
        else max_process_slots
    )
    if slot_count < 1:
        raise ValueError("max_process_slots must be a positive integer")
    local_workers = min(configured_workers, task_count, slot_count)
    node_count = min(
        configured_nodes,
        (task_count + local_workers - 1) // local_workers,
        max(1, slot_count // local_workers),
    )
    return node_count, local_workers


def _write_pita_stage0_script(
    script_path: Path,
    utr_stab_path: Path,
    mir_stab_path: Path,
) -> None:
    """Write a patched PITA runner that stops after STAB preparation."""
    source = (CONF.git_clone_dir / "off-target/pita/lib/pita_run.pl").read_text(
        encoding="utf-8"
    )
    before_search, separator, _ = source.partition(
        "## Step 1: Search potential targets"
    )
    if not separator:
        raise RuntimeError("Could not patch PITA runner before target discovery")
    _, helper_separator, helpers = source.partition("sub dsystem")
    if not helper_separator:
        raise RuntimeError("Could not find PITA dsystem helper")
    perl_utr_stab_path = str(utr_stab_path).replace("\\", "\\\\").replace("'", "\\'")
    perl_mir_stab_path = str(mir_stab_path).replace("\\", "\\\\").replace("'", "\\'")
    script_path.write_text(
        before_search
        + f"""
require File::Copy;
File::Copy::copy("tmp_utr_stab_$r", '{perl_utr_stab_path}') or die "copy tmp_utr_stab failed: $!";
File::Copy::copy("tmp_mir_stab_$r", '{perl_mir_stab_path}') or die "copy tmp_mir_stab failed: $!";
exit (0);
sub dsystem{helpers}
""",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def _pita_prepare_utr_shard_specs(
    *,
    utr_stab_path: Path,
    mir_stab_path: Path,
    shard_dir: Path,
    logs_dir: Path,
    shard_size: int,
) -> tuple[PitaPrepareUtrShardSpec, ...]:
    """Write local UTR STAB shard files and return discovery specs."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    start_row = 0
    with utr_stab_path.open(encoding="utf-8") as rows:
        for shard_index in count():
            shard_rows = list(islice(rows, shard_size))
            if not shard_rows:
                break
            end_row = start_row + len(shard_rows)
            shard_name = f"{shard_index:05d}_{start_row:012d}_{end_row:012d}"
            input_path = shard_dir / f"{shard_name}.utr.stab"
            output_path = shard_dir / f"{shard_name}.potential.tsv"
            _write_pita_row_input(input_path, shard_rows)
            specs.append(
                PitaPrepareUtrShardSpec(
                    shard_index=shard_index,
                    input_path=str(input_path),
                    mir_stab_path=str(mir_stab_path),
                    output_path=str(output_path),
                    log_path=str(
                        logs_dir / "pita_prepare_utr_shards" / f"{shard_name}.log"
                    ),
                )
            )
            start_row = end_row
    return tuple(specs)


def _pita_prepare_specs_from_reference(
    *,
    reference: PitaReferencePlan,
    mir_stab_path: Path,
    shard_dir: Path,
    logs_dir: Path,
) -> tuple[PitaPrepareUtrShardSpec, ...]:
    """Build candidate-specific discovery specs over shared UTR STAB shards."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    return tuple(
        PitaPrepareUtrShardSpec(
            shard_index=shard_index,
            input_path=input_path,
            mir_stab_path=str(mir_stab_path),
            output_path=str(shard_dir / f"{Path(input_path).stem}.potential.tsv"),
            log_path=str(
                logs_dir / "pita_prepare_utr_shards" / f"{shard_index:05d}.log"
            ),
        )
        for shard_index, input_path in enumerate(reference.utr_shard_paths)
    )


def _cached_pita_prepare_utr_shard_specs(
    *,
    shard_dir: Path,
    mir_stab_path: Path,
    logs_dir: Path,
    expected_count: int,
) -> tuple[PitaPrepareUtrShardSpec, ...] | None:
    """Rebuild a PITA stage0 plan from a complete persistent checkpoint."""
    input_paths = sorted(shard_dir.glob("*.utr.stab"))
    if len(input_paths) != expected_count or any(
        path.stat().st_size == 0 for path in input_paths
    ):
        return None
    return tuple(
        PitaPrepareUtrShardSpec(
            shard_index=shard_index,
            input_path=str(input_path),
            mir_stab_path=str(mir_stab_path),
            output_path=str(
                shard_dir / f"{input_path.name.removesuffix('.utr.stab')}.potential.tsv"
            ),
            log_path=str(
                logs_dir
                / "pita_prepare_utr_shards"
                / f"{input_path.name.removesuffix('.utr.stab')}.log"
            ),
        )
        for shard_index, input_path in enumerate(input_paths)
    )


def _run_pita_prepare_utr_shard(
    spec: PitaPrepareUtrShardSpec,
    attempts: int = APP_INFO.default_pita_row_attempts,
) -> str:
    """Run one local UTR STAB shard through PITA potential-target discovery."""
    import subprocess as sp
    from time import sleep

    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if _pita_prepare_utr_shard_ready(spec):
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
    if attempts < 1:
        raise ValueError("attempts must be a positive integer")
    for attempt in range(1, attempts + 1):
        tmp_output_path = _unique_tmp_path(output_path)
        cmd = (
            "set -euo pipefail; "
            f"perl {shlex.quote(str(pita_lib / 'find_potential_mirna_targets.pl'))} "
            f"{shlex.quote(spec.input_path)} -f {shlex.quote(spec.mir_stab_path)} "
            f"> {shlex.quote(str(tmp_output_path))}"
        )
        try:
            run_command(
                ["bash", "-lc", cmd],
                cwd=pita_lib.parent,
                output_mode="log",
                log_file=spec.log_path,
                show_command=False,
                warn_on_error=False,
            )
        except sp.CalledProcessError as exc:
            tmp_output_path.unlink(missing_ok=True)
            if exc.returncode in {-2, -15} and attempt < attempts:
                print(
                    "💊 Retrying OligoFormer PITA target-discovery shard "
                    f"{spec.shard_index} after signal {-exc.returncode}; "
                    f"log: {spec.log_path}"
                )
                sleep(min(30, 2 ** (attempt - 1)))
                continue
            raise RuntimeError(
                "OligoFormer PITA target-discovery shard "
                f"{spec.shard_index} failed with return code {exc.returncode}. "
                f"Check log file {spec.log_path}."
            ) from exc
        tmp_output_path.replace(output_path)
        break
    _publish_artifact_marker(
        marker_path,
        kind="pita-target-discovery",
        artifacts={"potential_targets": output_path},
    )
    return str(output_path)


def _pita_prepare_utr_shard_ready(spec: PitaPrepareUtrShardSpec) -> bool:
    """Return whether one PITA target-discovery shard is complete."""
    output_path = Path(spec.output_path)
    return _artifact_marker_ready(
        output_path.with_suffix(output_path.suffix + ".done"),
        kind="pita-target-discovery",
        artifacts={"potential_targets": output_path},
    )


def _pita_row_shard_specs(
    *,
    spec: OffTargetShardSpec,
    row_count: int,
    potential_targets_path: Path,
    ext_utr_path: Path,
    row_dir: Path,
) -> tuple[PitaRowShardSpec, ...]:
    """Build all PITA row-shard specs for one siRNA."""
    return tuple(
        _pita_row_shard_spec(
            spec=spec,
            shard_index=shard_index,
            start_row=start_row,
            end_row=min(start_row + spec.row_shard_size, row_count),
            potential_targets_path=potential_targets_path,
            ext_utr_path=ext_utr_path,
            row_dir=row_dir,
        )
        for shard_index, start_row in enumerate(
            range(0, row_count, spec.row_shard_size)
        )
    )


def _write_pita_row_input(path: Path, lines: list[str]) -> None:
    """Atomically write one PITA potential-target row input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(path)
    try:
        with tmp_path.open("w", encoding="utf-8") as out:
            out.writelines(lines)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _ensure_pita_row_inputs(
    potential_targets_path: Path,
    row_shards: tuple[PitaRowShardSpec, ...],
) -> None:
    """Write missing per-row-shard potential-target inputs."""
    missing = [row for row in row_shards if not Path(row.input_path).exists()]
    if not missing:
        return

    _warmup_file_parents(
        [potential_targets_path],
        file_pattern=f"^{re.escape(potential_targets_path.name)}$",
    )
    missing_indices = {row.shard_index for row in missing}
    with potential_targets_path.open("r", encoding="utf-8") as potential_rows:
        for row in row_shards:
            shard_rows = [
                line if line.endswith("\n") else f"{line}\n"
                for line in islice(
                    potential_rows,
                    row.end_row - row.start_row,
                )
            ]
            if row.shard_index not in missing_indices:
                continue
            _write_pita_row_input(Path(row.input_path), shard_rows)


def _pita_prepared_shard_from_plan(
    plan: PitaPreparePlan,
    *,
    row_count: int,
) -> PreparedOffTargetShard:
    """Build a prepared off-target shard from completed PITA target discovery."""
    cache_dir = _off_target_shard_cache_dir(plan.spec)
    logs_dir = _off_target_shard_logs_dir(plan.spec)
    potential_targets_path = cache_dir / "potential_targets.tsv"
    row_shards = _pita_row_shard_specs(
        spec=plan.spec,
        row_count=row_count,
        potential_targets_path=potential_targets_path,
        ext_utr_path=(
            Path(plan.ext_utr_path)
            if plan.ext_utr_path
            else cache_dir
            / f"{plan.spec.stem}_shard_{plan.spec.index:05d}_ext_utr.stab"
        ),
        row_dir=cache_dir / "pita_rows",
    )
    _ensure_pita_row_inputs(potential_targets_path, row_shards)
    return PreparedOffTargetShard(
        index=plan.spec.index,
        record_name=plan.spec.record_name,
        cache_dir=str(cache_dir),
        logs_dir=str(logs_dir),
        pita_path=str(cache_dir / "pita.tab"),
        row_shards=row_shards,
    )


def _prepare_pita_reference_plan(
    spec: OffTargetShardSpec,
    prepare_root: Path,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> PitaReferencePlan:
    """Prepare reference-only PITA STAB data once for all selected siRNAs."""
    import shutil

    reference_dir = (
        AppRunLayout.from_run_root(spec.run_root).prep_dir
        / "off_target"
        / spec.stem
        / "pita_reference"
    )
    shard_dir = reference_dir / "utr_shards"
    ext_utr_path = reference_dir / "reference_ext_utr.stab"
    marker_path = reference_dir / "reference.done"
    shard_size = execution.pita_prepare_utr_shard_size
    reference_identity: dict[str, object] = {
        "cache_salt": APP_INFO.off_target_cache_salt,
        "shard_size": shard_size,
        "utr_sha256": _hash_path(Path(spec.utr_path)),
        "orf_sha256": _hash_path(Path(spec.orf_path)),
    }
    cached_shards = tuple(sorted(shard_dir.glob("*.utr.stab")))
    reference_artifacts = {
        "ext_utr": ext_utr_path,
        **{f"utr_shard/{path.name}": path for path in cached_shards},
    }
    if cached_shards and _artifact_marker_ready(
        marker_path,
        kind="pita-reference-stage0",
        artifacts=reference_artifacts,
        identity=reference_identity,
    ):
        print(f"💊 Reusing OligoFormer PITA reference STAB cache for {spec.stem}")
        return PitaReferencePlan(
            utr_shard_paths=tuple(map(str, cached_shards)),
            ext_utr_path=str(ext_utr_path),
        )

    shutil.rmtree(reference_dir, ignore_errors=True)
    reference_dir.mkdir(parents=True)
    reference_root = prepare_root / f"{spec.stem}_pita_reference"
    pita_root = reference_root / "off-target" / "pita"
    pita_root.parent.mkdir(parents=True)
    shutil.copytree(CONF.git_clone_dir / "off-target/pita", pita_root)
    sirna_file = reference_root / "siRNA.fa"
    _write_sirna_records(
        [OffTargetSirnaRecord(spec.record_name, spec.record_sequence)],
        sirna_file,
    )
    utr_stab_path = reference_root / "reference_utr.stab"
    mir_stab_path = reference_root / "reference_mir.stab"
    script_path = pita_root / "prepare_pita_reference.pl"
    _write_pita_stage0_script(script_path, utr_stab_path, mir_stab_path)
    run_command(
        [
            "perl",
            str(script_path),
            "-utr",
            spec.utr_path,
            "-mir",
            str(sirna_file),
            "-prefix",
            "reference_",
            "-upstream",
            spec.orf_path,
            "-output",
            f"{reference_dir}/",
        ],
        cwd=pita_root,
        output_mode="log",
        log_file=Path(spec.output_dir)
        / "logs"
        / "off_target"
        / spec.stem
        / "pita_reference.log",
        show_command=False,
    )
    if not utr_stab_path.is_file() or not ext_utr_path.is_file():
        raise FileNotFoundError(
            "OligoFormer PITA reference preparation did not produce UTR STAB data"
        )
    reference_specs = _pita_prepare_utr_shard_specs(
        utr_stab_path=utr_stab_path,
        mir_stab_path=mir_stab_path,
        shard_dir=shard_dir,
        logs_dir=Path(spec.output_dir) / "logs" / "off_target" / spec.stem,
        shard_size=shard_size,
    )
    if not reference_specs:
        raise RuntimeError("OligoFormer PITA reference UTR STAB data is empty")
    reference_artifacts = {
        "ext_utr": ext_utr_path,
        **{
            f"utr_shard/{Path(item.input_path).name}": Path(item.input_path)
            for item in reference_specs
        },
    }
    _publish_artifact_marker(
        marker_path,
        kind="pita-reference-stage0",
        artifacts=reference_artifacts,
        identity=reference_identity,
    )
    return PitaReferencePlan(
        utr_shard_paths=tuple(item.input_path for item in reference_specs),
        ext_utr_path=str(ext_utr_path),
    )


def _prepare_pita_target_discovery_plan(
    spec: OffTargetShardSpec,
    shard_root: Path,
    reference: PitaReferencePlan | None = None,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> PitaPreparePlan:
    """Prepare local PITA stage0 files and return target-discovery shards."""
    invalid_refs = [
        str(path)
        for path in (Path(spec.utr_path), Path(spec.orf_path))
        if not path.is_file() or path.stat().st_size == 0
    ]
    if invalid_refs:
        raise FileNotFoundError(
            "OligoFormer PITA reference files are missing or empty: "
            + ", ".join(invalid_refs)
        )

    cache_dir = _off_target_shard_cache_dir(spec)
    logs_dir = _off_target_shard_logs_dir(spec)
    potential_targets_path = cache_dir / "potential_targets.tsv"
    ext_utr_path = (
        Path(reference.ext_utr_path)
        if reference is not None
        else cache_dir / f"{spec.stem}_shard_{spec.index:05d}_ext_utr.stab"
    )
    marker_path = cache_dir / "pita_prepare.done"
    stage0_marker_path = cache_dir / "pita_stage0.done"
    cached_mir_stab_path = cache_dir / f"{spec.stem}_shard_{spec.index:05d}_mir.stab"
    stage0_shard_dir = cache_dir / "pita_prepare_utr_shards"
    stage0_shard_size = execution.pita_prepare_utr_shard_size
    stage0_identity: dict[str, object] = {
        "cache_salt": APP_INFO.off_target_cache_salt,
        "record_name": spec.record_name,
        "record_sequence_sha256": _hash_bytes(spec.record_sequence.encode()),
        "shard_size": stage0_shard_size,
        "utr_sha256": _hash_path(Path(spec.utr_path)),
        "orf_sha256": _hash_path(Path(spec.orf_path)),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _artifact_marker_ready(
        marker_path,
        kind="pita-target-discovery-merge",
        artifacts={
            "potential_targets": potential_targets_path,
            "ext_utr": ext_utr_path,
        },
    ):
        return PitaPreparePlan(
            spec=spec,
            utr_shards=(),
            row_count=cast(
                int,
                _artifact_file_metadata(potential_targets_path)["line_count"],
            ),
            ext_utr_path=str(ext_utr_path),
        )

    pita_root = shard_root / "off-target/pita"
    sirna_file = (
        shard_root / "data/infer" / f"{spec.stem}_shard_{spec.index:05d}" / "siRNA.fa"
    )
    if not sirna_file.exists():
        _write_sirna_records(
            [OffTargetSirnaRecord(spec.record_name, spec.record_sequence)],
            sirna_file,
        )

    if reference is not None:
        utr_shards = _pita_prepare_specs_from_reference(
            reference=reference,
            mir_stab_path=cached_mir_stab_path,
            shard_dir=stage0_shard_dir,
            logs_dir=logs_dir,
        )
        stage0_artifacts = {
            "mir_stab": cached_mir_stab_path,
            "ext_utr": ext_utr_path,
            **{
                f"utr_shard/{Path(shard.input_path).name}": Path(shard.input_path)
                for shard in utr_shards
            },
        }
        if not _artifact_marker_ready(
            stage0_marker_path,
            kind="pita-candidate-stage0",
            artifacts=stage0_artifacts,
            identity=stage0_identity,
        ):
            tmp_mir_stab_path = _unique_tmp_path(cached_mir_stab_path)
            cmd = (
                f"perl {shlex.quote(str(pita_root / 'lib/fasta2stab.pl'))} "
                f"{shlex.quote(str(sirna_file))} "
                f"> {shlex.quote(str(tmp_mir_stab_path))}"
            )
            run_command(
                ["bash", "-lc", cmd],
                cwd=pita_root,
                output_mode="log",
                log_file=logs_dir / "pita_prepare_mir.log",
                show_command=False,
            )
            tmp_mir_stab_path.replace(cached_mir_stab_path)
            _publish_artifact_marker(
                stage0_marker_path,
                kind="pita-candidate-stage0",
                artifacts=stage0_artifacts,
                identity=stage0_identity,
            )
        return PitaPreparePlan(
            spec=spec,
            utr_shards=utr_shards,
            row_count=None,
            ext_utr_path=str(ext_utr_path),
        )

    cached_input_paths = tuple(sorted(stage0_shard_dir.glob("*.utr.stab")))
    cached_utr_shards = _cached_pita_prepare_utr_shard_specs(
        shard_dir=stage0_shard_dir,
        mir_stab_path=cached_mir_stab_path,
        logs_dir=logs_dir,
        expected_count=len(cached_input_paths),
    )
    if cached_utr_shards:
        stage0_artifacts = {
            "mir_stab": cached_mir_stab_path,
            "ext_utr": ext_utr_path,
            **{
                f"utr_shard/{Path(shard.input_path).name}": Path(shard.input_path)
                for shard in cached_utr_shards
            },
        }
        if _artifact_marker_ready(
            stage0_marker_path,
            kind="pita-candidate-stage0",
            artifacts=stage0_artifacts,
            identity=stage0_identity,
        ):
            print(
                f"💊 Reusing OligoFormer PITA stage0 checkpoint for {spec.record_name}"
            )
            return PitaPreparePlan(
                spec=spec,
                utr_shards=cached_utr_shards,
                row_count=None,
                ext_utr_path=str(ext_utr_path),
            )

    utr_stab_path = shard_root / f"{spec.stem}_shard_{spec.index:05d}_utr.stab"
    mir_stab_path = shard_root / f"{spec.stem}_shard_{spec.index:05d}_mir.stab"
    script_path = pita_root / "prepare_pita_stage0.pl"
    _write_pita_stage0_script(script_path, utr_stab_path, mir_stab_path)
    cmd = [
        "perl",
        str(script_path),
        "-utr",
        str(spec.utr_path),
        "-mir",
        str(sirna_file),
        "-prefix",
        f"{spec.stem}_shard_{spec.index:05d}_",
        "-upstream",
        str(spec.orf_path),
        "-output",
        f"{cache_dir}/",
    ]
    run_command(
        cmd,
        cwd=pita_root,
        output_mode="log",
        log_file=logs_dir / "pita_prepare_stage0.log",
        show_command=False,
    )
    missing_stage0_outputs = [
        str(path)
        for path in (utr_stab_path, mir_stab_path, ext_utr_path)
        if not path.exists()
    ]
    if missing_stage0_outputs:
        raise FileNotFoundError(
            "OligoFormer PITA stage 0 did not produce expected files: "
            + ", ".join(missing_stage0_outputs)
        )
    cached_mir_stab_path.write_bytes(mir_stab_path.read_bytes())
    utr_shards = _pita_prepare_utr_shard_specs(
        utr_stab_path=utr_stab_path,
        mir_stab_path=cached_mir_stab_path,
        shard_dir=stage0_shard_dir,
        logs_dir=logs_dir,
        shard_size=stage0_shard_size,
    )
    stage0_artifacts = {
        "mir_stab": cached_mir_stab_path,
        "ext_utr": ext_utr_path,
        **{
            f"utr_shard/{Path(shard.input_path).name}": Path(shard.input_path)
            for shard in utr_shards
        },
    }
    _publish_artifact_marker(
        stage0_marker_path,
        kind="pita-candidate-stage0",
        artifacts=stage0_artifacts,
        identity=stage0_identity,
    )
    return PitaPreparePlan(
        spec=spec,
        utr_shards=utr_shards,
        row_count=None,
        ext_utr_path=str(ext_utr_path),
    )


def _finalize_pita_target_discovery_plan(
    plan: PitaPreparePlan,
) -> PreparedOffTargetShard:
    """Merge PITA target-discovery shards and return row-score specs."""
    if plan.row_count is not None:
        return _pita_prepared_shard_from_plan(plan, row_count=plan.row_count)

    outputs = [
        shard.output_path
        for shard in sorted(plan.utr_shards, key=lambda item: item.shard_index)
    ]
    missing_outputs = [
        shard.output_path
        for shard in plan.utr_shards
        if not _pita_prepare_utr_shard_ready(shard)
    ]
    if missing_outputs:
        raise FileNotFoundError(
            "OligoFormer PITA consolidation is missing "
            f"{len(missing_outputs)} target-discovery outputs; first missing output: "
            f"{missing_outputs[0]}"
        )

    cache_dir = _off_target_shard_cache_dir(plan.spec)
    potential_targets_path = cache_dir / "potential_targets.tsv"
    ext_utr_path = (
        Path(plan.ext_utr_path)
        if plan.ext_utr_path
        else cache_dir / f"{plan.spec.stem}_shard_{plan.spec.index:05d}_ext_utr.stab"
    )
    row_dir = cache_dir / "pita_rows"
    potential_targets_path.parent.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)
    _warmup_file_parents(
        outputs,
        file_pattern=r"\.potential\.tsv$",
    )

    tmp_potential_targets_path = _unique_tmp_path(potential_targets_path)
    row_count = 0
    row_lines: list[str] = []
    row_shard_index = 0

    def _flush_row_lines() -> None:
        nonlocal row_shard_index
        if not row_lines:
            return
        row_spec = _pita_row_shard_spec(
            spec=plan.spec,
            shard_index=row_shard_index,
            start_row=row_count - len(row_lines),
            end_row=row_count,
            potential_targets_path=potential_targets_path,
            ext_utr_path=ext_utr_path,
            row_dir=row_dir,
        )
        _write_pita_row_input(Path(row_spec.input_path), row_lines)
        row_lines.clear()
        row_shard_index += 1

    try:
        with tmp_potential_targets_path.open("w", encoding="utf-8") as out:
            for output in outputs:
                with Path(output).open("r", encoding="utf-8") as shard_rows:
                    for line in shard_rows:
                        normalized_line = line if line.endswith("\n") else f"{line}\n"
                        out.write(normalized_line)
                        row_lines.append(normalized_line)
                        row_count += 1
                        if len(row_lines) < plan.spec.row_shard_size:
                            continue
                        _flush_row_lines()

            _flush_row_lines()
        tmp_potential_targets_path.replace(potential_targets_path)
    finally:
        tmp_potential_targets_path.unlink(missing_ok=True)

    marker_path = cache_dir / "pita_prepare.done"
    _publish_artifact_marker(
        marker_path,
        kind="pita-target-discovery-merge",
        artifacts={
            "potential_targets": potential_targets_path,
            "ext_utr": ext_utr_path,
        },
    )
    return _pita_prepared_shard_from_plan(plan, row_count=row_count)


def _pita_row_shard_spec(
    *,
    spec: OffTargetShardSpec,
    shard_index: int,
    start_row: int,
    end_row: int,
    potential_targets_path: Path,
    ext_utr_path: Path,
    row_dir: Path,
) -> PitaRowShardSpec:
    """Build a serializable PITA row-shard spec."""
    shard_name = f"{shard_index:05d}_{start_row:012d}_{end_row:012d}"
    input_path = row_dir / f"{shard_name}.potential.tsv"
    output_path = row_dir / f"{shard_name}.scored.tsv"
    return PitaRowShardSpec(
        run_root=spec.run_root,
        stem=spec.stem,
        sirna_index=spec.index,
        record_name=spec.record_name,
        shard_index=shard_index,
        start_row=start_row,
        end_row=end_row,
        potential_targets_path=str(potential_targets_path),
        input_path=str(input_path),
        ext_utr_path=str(ext_utr_path),
        output_path=str(output_path),
        log_path=str(
            _off_target_shard_logs_dir(spec) / "pita_rows" / f"{shard_name}.log"
        ),
    )


def _targetscan_context_shard_specs(
    *,
    context_dir: Path,
    logs_dir: Path,
    rnaplfold_cache_dir: str,
) -> tuple[TargetscanContextShardSpec, ...]:
    """Build context-score shard specs from prepared TargetScan target shards."""
    shard_paths = sorted((context_dir / "shards").glob("targets_*"))
    return tuple(
        TargetscanContextShardSpec(
            shard_index=shard_index,
            common_dir=str(context_dir / "common"),
            targets_path=str(shard_path),
            output_path=str(
                context_dir
                / "outputs"
                / f"Targets.BL_PCT.context_scores.{shard_index:05d}.txt"
            ),
            log_path=str(logs_dir / "targetscan_context" / f"{shard_index:05d}.log"),
            rnaplfold_cache_dir=rnaplfold_cache_dir,
        )
        for shard_index, shard_path in enumerate(shard_paths)
    )


def _run_targetscan_context_shard(
    spec: TargetscanContextShardSpec,
    attempts: int = APP_INFO.default_targetscan_context_attempts,
) -> str:
    """Run one TargetScan context-score shard on a CPU node."""
    import shutil
    import subprocess as sp
    from time import sleep

    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if _targetscan_context_shard_ready(spec):
        return str(output_path)

    common_dir = Path(spec.common_dir)
    with TemporaryDirectory(
        prefix=f"oligoformer_targetscan_context_{spec.shard_index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        shutil.copy2(
            CONF.git_clone_dir
            / "off-target/targetscan/targetscan_70_context_scores.pl",
            workdir / "targetscan_70_context_scores.pl",
        )
        for name in (
            "sirnas_for_context_scores.txt",
            "UTR.txt",
            "ORF.length.txt",
            "ORF_8mer_counts.txt",
            "TA_SPS_by_seed_region.txt",
            "Agarwal_2015_parameters.txt",
            "All_cell_lines.AIRs.txt",
        ):
            (workdir / name).symlink_to(common_dir / name)
        (workdir / "predicted_targets.txt").symlink_to(Path(spec.targets_path))
        if spec.rnaplfold_cache_dir:
            (workdir / "RNAplfold_in_out").symlink_to(Path(spec.rnaplfold_cache_dir))

        tmp_output_path = workdir / "Targets.BL_PCT.context_scores.txt"
        cmd = [
            "perl",
            "targetscan_70_context_scores.pl",
            "sirnas_for_context_scores.txt",
            "UTR.txt",
            "predicted_targets.txt",
            "ORF.length.txt",
            "ORF_8mer_counts.txt",
            str(tmp_output_path),
        ]
        if attempts < 1:
            raise ValueError("attempts must be a positive integer")
        transient_return_codes = {-2, -15}
        for attempt in range(1, attempts + 1):
            tmp_output_path.unlink(missing_ok=True)
            try:
                run_command(
                    cmd,
                    cwd=workdir,
                    output_mode="log",
                    log_file=spec.log_path,
                    show_command=False,
                    warn_on_error=False,
                )
            except sp.CalledProcessError as exc:
                tmp_output_path.unlink(missing_ok=True)
                can_retry = (
                    exc.returncode in transient_return_codes and attempt < attempts
                )
                if can_retry:
                    print(
                        "💊 Retrying OligoFormer TargetScan context shard "
                        f"{spec.shard_index} after signal {-exc.returncode}; "
                        f"log: {spec.log_path}"
                    )
                    sleep(min(30, 2 ** (attempt - 1)))
                    continue
                raise RuntimeError(
                    "OligoFormer TargetScan context shard "
                    f"{spec.shard_index} failed with return code {exc.returncode}. "
                    f"Check log file {spec.log_path}."
                ) from exc
            break
        output_path.parent.mkdir(parents=True, exist_ok=True)
        durable_tmp_path = _unique_tmp_path(output_path)
        try:
            shutil.copy2(tmp_output_path, durable_tmp_path)
            durable_tmp_path.replace(output_path)
        finally:
            durable_tmp_path.unlink(missing_ok=True)

    _publish_artifact_marker(
        marker_path,
        kind="targetscan-context-score",
        artifacts={"context_scores": output_path},
    )
    return str(output_path)


def _targetscan_context_shard_ready(spec: TargetscanContextShardSpec) -> bool:
    """Return whether one context-score shard output is ready."""
    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    return _artifact_marker_ready(
        marker_path,
        kind="targetscan-context-score",
        artifacts={"context_scores": output_path},
    )


def _targetscan_tile_ready(spec: TargetscanBatchSpec) -> bool:
    """Return whether one complete TargetScan tile publication is valid."""
    targetscan_path = _targetscan_batch_cache_dir(spec) / "targetscan.tab"
    return _artifact_marker_ready(
        targetscan_path.parent / "targetscan.done",
        kind="targetscan-candidate-reference-tile",
        artifacts={"targetscan": targetscan_path},
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_targetscan_context_shard_batch(
    specs: list[TargetscanContextShardSpec],
    local_workers: int,
    attempts: int = APP_INFO.default_targetscan_context_attempts,
) -> int:
    """Run a fixed batch of TargetScan context-score shards on one CPU node."""
    local_workers = _validated_worker_count(local_workers)
    if attempts < 1:
        raise ValueError("attempts must be a positive integer")
    CONF.output_volume.reload()
    outputs = bounded_map(
        specs,
        lambda spec: _run_targetscan_context_shard(spec, attempts),
        max_parallel=local_workers,
    )
    CONF.output_volume.commit()
    return len(outputs)


def _merge_targetscan_context_outputs(
    *,
    context_outputs: list[str],
    targetscan_path: Path,
    log_file: Path,
) -> None:
    """Merge TargetScan context-score shards into OligoFormer's target table."""
    targetscan_path.parent.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        output for output in context_outputs if Path(output).stat().st_size > 0
    ]
    if not existing_outputs:
        targetscan_path.write_text(TARGETSCAN_HEADER + "\n", encoding="utf-8")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "No TargetScan context outputs to merge.\n", encoding="utf-8"
        )
        return

    if len(existing_outputs) > 1:
        warmup_directory(Path(existing_outputs[0]).parent)
    context_scores = pl.scan_csv(
        existing_outputs,
        separator="\t",
        infer_schema_length=0,
    )
    column_names = context_scores.collect_schema().names()
    if len(column_names) < 28:
        raise ValueError(
            "TargetScan context output must contain at least 28 columns; "
            f"found {len(column_names)}"
        )

    target_col, sirna_col, site_type_col, score_col = (
        column_names[0],
        column_names[2],
        column_names[3],
        column_names[27],
    )
    merged = (
        context_scores
        .select(
            pl.col(target_col).alias("refseq"),
            pl.col(sirna_col).alias("siRNA"),
            pl.col(site_type_col).alias("site_type"),
            pl.col(score_col).cast(pl.Float64, strict=False).alias("score"),
        )
        .filter(pl.col("score").is_not_null())
        .filter(
            (pl.col("site_type") == "6mer")
            | ((pl.col("site_type") == "7mer-1a") & (pl.col("score") < -0.01))
            | ((pl.col("site_type") == "7mer-m8") & (pl.col("score") < -0.02))
            | ((pl.col("site_type") == "8mer-1a") & (pl.col("score") < -0.03))
        )
        .group_by("refseq", "siRNA")
        .agg((-pl.col("score").sum()).alias("targetscan_score"))
        .sort("refseq", "siRNA")
        .collect()
    )
    merged.write_csv(
        targetscan_path,
        separator="\t",
        include_header=merged.height == 0,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "Merged TargetScan context outputs with Polars.\n"
        f"context_outputs\t{len(existing_outputs)}\n"
        f"output_rows\t{merged.height}\n",
        encoding="utf-8",
    )


TARGETSCAN_REFERENCE_INPUT_NAMES = (
    "UTR.txt",
    "ORF.txt",
    "ORF.length.txt",
    "UTRs_median_BLs_bins.txt",
)
TARGETSCAN_CONTEXT_COMMON_NAMES = (
    "sirnas_for_context_scores.txt",
    "UTR.txt",
    "ORF.length.txt",
    "ORF_8mer_counts.txt",
    "TA_SPS_by_seed_region.txt",
    "Agarwal_2015_parameters.txt",
    "All_cell_lines.AIRs.txt",
)


def _targetscan_context_prep_artifacts(context_dir: Path) -> dict[str, Path]:
    """Return the exact common and target-shard artifacts for one prep stage."""
    artifacts: dict[str, Path] = {
        f"common/{name}": context_dir / "common" / name
        for name in TARGETSCAN_CONTEXT_COMMON_NAMES
    }
    artifacts.update({
        f"shards/{path.name}": path
        for path in sorted((context_dir / "shards").glob("targets_*"))
    })
    return artifacts


def _targetscan_reference_identity(spec: TargetscanBatchSpec) -> str:
    """Return the identity of reference-only TargetScan inputs for one shard."""
    return hash_string(
        "\n".join((
            APP_INFO.off_target_cache_salt,
            APP_INFO.targetscan_version,
            APP_INFO.targetscan_species_id,
            _hash_path(Path(spec.utr_path)),
            _hash_path(Path(spec.orf_path)),
        ))
    )


def _targetscan_reference_inputs_ready(
    reference_dir: Path,
    *,
    identity: str,
) -> bool:
    """Return whether reference-only TargetScan inputs match their manifest."""
    import orjson

    marker_path = reference_dir / "reference.done"
    if not marker_path.is_file():
        return False
    try:
        manifest = orjson.loads(marker_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("identity") != identity
    ):
        return False
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(
        TARGETSCAN_REFERENCE_INPUT_NAMES
    ):
        return False
    for name in TARGETSCAN_REFERENCE_INPUT_NAMES:
        metadata = files.get(name)
        path = reference_dir / name
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("size"), int)
            or metadata["size"] < 0
            or not isinstance(metadata.get("sha256"), str)
            or not path.is_file()
            or path.stat().st_size != metadata["size"]
            or _hash_path(path) != metadata["sha256"]
        ):
            return False
    return True


def _prepare_targetscan_reference_inputs(
    spec: TargetscanBatchSpec,
    *,
    targetscan_workdir: Path,
    log_file: Path,
) -> Path:
    """Prepare and atomically publish reference-only TargetScan inputs once."""
    import shutil

    import orjson

    reference_dir = Path(spec.utr_path).parent / "prepared_reference"
    identity = _targetscan_reference_identity(spec)
    if _targetscan_reference_inputs_ready(reference_dir, identity=identity):
        return reference_dir

    local_dir = targetscan_workdir / "biomodals_reference_inputs"
    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True)
    species = APP_INFO.targetscan_species_id
    utrs = _read_fasta_pairs(Path(spec.utr_path))
    orfs = _read_fasta_pairs(Path(spec.orf_path))
    local_dir.joinpath("UTR.txt").write_text(
        "".join(f"{name}\t{species}\t{sequence}\n" for name, sequence in utrs),
        encoding="utf-8",
    )
    local_dir.joinpath("ORF.txt").write_text(
        "".join(f"{name}\t{species}\t{sequence}\n" for name, sequence in orfs),
        encoding="utf-8",
    )
    local_dir.joinpath("ORF.length.txt").write_text(
        "".join(f"{name}\t{species}\t{len(sequence)}\n" for name, sequence in orfs),
        encoding="utf-8",
    )
    bins_lines = run_command(
        [
            "perl",
            "targetscan_70_BL_bins.pl",
            str(local_dir / "UTR.txt"),
        ],
        cwd=targetscan_workdir,
        output_mode="capture",
        log_file=log_file,
        show_command=False,
    )
    local_dir.joinpath("UTRs_median_BLs_bins.txt").write_text(
        "\n".join(bins_lines) + ("\n" if bins_lines else ""),
        encoding="utf-8",
    )

    reference_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for name in TARGETSCAN_REFERENCE_INPUT_NAMES:
        source = local_dir / name
        destination = reference_dir / name
        tmp_path = _unique_tmp_path(destination)
        try:
            shutil.copy2(source, tmp_path)
            tmp_path.replace(destination)
        finally:
            tmp_path.unlink(missing_ok=True)
        files[name] = {
            "size": destination.stat().st_size,
            "sha256": _hash_path(destination),
        }
    marker_path = reference_dir / "reference.done"
    tmp_marker_path = _unique_tmp_path(marker_path)
    try:
        tmp_marker_path.write_bytes(
            orjson.dumps({
                "version": 1,
                "identity": identity,
                "files": files,
            })
        )
        tmp_marker_path.replace(marker_path)
    finally:
        tmp_marker_path.unlink(missing_ok=True)
    return reference_dir


def _prepare_targetscan_batch_context_plan(
    spec: TargetscanBatchSpec,
    batch_root: Path,
) -> PreparedTargetscanBatch:
    """Prepare TargetScan context-score shard specs for a siRNA batch."""
    import shutil
    from time import monotonic

    if spec.sirna_count < 1:
        raise RuntimeError("TargetScan batch requires at least one siRNA record")

    cache_dir = _targetscan_batch_cache_dir(spec)
    targetscan_path = cache_dir / "targetscan.tab"
    marker_path = cache_dir / "targetscan.done"
    logs_dir = _targetscan_batch_logs_dir(spec)
    if _artifact_marker_ready(
        marker_path,
        kind="targetscan-candidate-reference-tile",
        artifacts={"targetscan": targetscan_path},
    ):
        return PreparedTargetscanBatch(
            targetscan_path=str(targetscan_path),
            logs_dir=str(logs_dir),
            context_shards=(),
            needs_merge=False,
        )

    rnaplfold_cache_dir = spec.rnaplfold_cache_dir
    context_shard_size = spec.context_shard_size
    context_identity: dict[str, object] = {
        "context_shard_size": context_shard_size,
    }
    context_dir = cache_dir / "targetscan_context"
    prep_marker_path = context_dir / "targetscan_prepare.done"
    print(
        "💊 Preparing OligoFormer TargetScan batch "
        f"{spec.stem}:{spec.shard_index} for {spec.sirna_count} siRNAs with "
        f"{context_shard_size} target rows per context shard; "
        f"log: {logs_dir / 'targetscan_prep.log'}"
    )
    if not _artifact_marker_ready(
        prep_marker_path,
        kind="targetscan-context-preparation",
        artifacts=_targetscan_context_prep_artifacts(context_dir),
        identity=context_identity,
    ):
        targetscan_workdir = batch_root / "off-target" / "tmp"
        shutil.copytree(
            batch_root / "off-target" / "targetscan",
            targetscan_workdir,
        )
        species = APP_INFO.targetscan_species_id
        sirnas = _read_fasta_pairs(Path(spec.sirna_path))
        reference_dir = _prepare_targetscan_reference_inputs(
            spec,
            targetscan_workdir=targetscan_workdir,
            log_file=logs_dir / "targetscan_reference.log",
        )
        for name in TARGETSCAN_REFERENCE_INPUT_NAMES:
            targetscan_workdir.joinpath(name).symlink_to(reference_dir / name)
        targetscan_workdir.joinpath("sirnas_for_context_scores.txt").write_text(
            "".join(
                f"{name}\t{species}\t{name}\t{sequence}\n" for name, sequence in sirnas
            ),
            encoding="utf-8",
        )
        targetscan_workdir.joinpath("sirnas.txt").write_text(
            "".join(
                f"{name}\t{sequence[1:8]}\t{species}\n" for name, sequence in sirnas
            ),
            encoding="utf-8",
        )
        checkpoint_dir = cache_dir / "targetscan_seed"
        checkpoint_path = checkpoint_dir / "targetscan_70_output.txt"
        checkpoint_marker = checkpoint_dir / "targetscan_70.done"
        local_seed_path = targetscan_workdir / "targetscan_70_output.txt"
        if _artifact_marker_ready(
            checkpoint_marker,
            kind="targetscan-seed-scan",
            artifacts={"seed_scan": checkpoint_path},
        ):
            print(
                "💊 Reusing OligoFormer TargetScan seed checkpoint for batch "
                f"{spec.stem}:{spec.shard_index}"
            )
            shutil.copy2(checkpoint_path, local_seed_path)
        else:
            print(
                "💊 Running OligoFormer TargetScan seed scan for batch "
                f"{spec.stem}:{spec.shard_index}"
            )
            started_at = monotonic()
            run_command(
                [
                    "perl",
                    "targetscan_70.pl",
                    "sirnas.txt",
                    "UTR.txt",
                    local_seed_path.name,
                ],
                cwd=targetscan_workdir,
                output_mode="log",
                log_file=logs_dir / "targetscan_prep.log",
                show_command=False,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            tmp_checkpoint = checkpoint_path.with_name(
                f".{checkpoint_path.name}.tmp.{os.getpid()}"
            )
            tmp_checkpoint.unlink(missing_ok=True)
            shutil.copy2(local_seed_path, tmp_checkpoint)
            tmp_checkpoint.replace(checkpoint_path)
            _publish_artifact_marker(
                checkpoint_marker,
                kind="targetscan-seed-scan",
                artifacts={"seed_scan": checkpoint_path},
            )
            CONF.output_volume.commit()
            print(
                "💊 Checkpointed OligoFormer TargetScan seed scan for batch "
                f"{spec.stem}:{spec.shard_index} in "
                f"{monotonic() - started_at:.1f}s"
            )

        targetscan_cmd = r"""
set -eu
context_dir=$1
context_shard_size=$2
perl targetscan_70_BL_PCT.pl sirnas.txt targetscan_70_output.txt UTRs_median_BLs_bins.txt > targetscan_70_output.BL_PCT.txt
perl targetscan_count_8mers.pl sirnas.txt ORF.txt > ORF_8mer_counts.txt
rm -rf "$context_dir"
mkdir -p "$context_dir/common" "$context_dir/shards" "$context_dir/outputs"
cp sirnas_for_context_scores.txt UTR.txt ORF.length.txt ORF_8mer_counts.txt "$context_dir/common/"
cp TA_SPS_by_seed_region.txt Agarwal_2015_parameters.txt All_cell_lines.AIRs.txt "$context_dir/common/"
if [[ -s targetscan_70_output.BL_PCT.txt ]]; then
  split -d -a 5 -l "$context_shard_size" targetscan_70_output.BL_PCT.txt "$context_dir/shards/targets_"
fi
"""
        print(
            "💊 Building OligoFormer TargetScan context inputs for batch "
            f"{spec.stem}:{spec.shard_index}"
        )
        run_command(
            [
                "bash",
                "-lc",
                targetscan_cmd,
                "prepare_targetscan_context",
                str(context_dir),
                str(context_shard_size),
            ],
            cwd=targetscan_workdir,
            output_mode="log",
            log_file=logs_dir / "targetscan_prep.log",
            show_command=False,
        )
        _publish_artifact_marker(
            prep_marker_path,
            kind="targetscan-context-preparation",
            artifacts=_targetscan_context_prep_artifacts(context_dir),
            identity=context_identity,
        )

    context_shards = _targetscan_context_shard_specs(
        context_dir=context_dir,
        logs_dir=logs_dir,
        rnaplfold_cache_dir=rnaplfold_cache_dir,
    )
    print(
        "💊 Prepared OligoFormer TargetScan batch "
        f"{spec.stem}:{spec.shard_index} with {len(context_shards)} context shards"
    )
    return PreparedTargetscanBatch(
        targetscan_path=str(targetscan_path),
        logs_dir=str(logs_dir),
        context_shards=context_shards,
        needs_merge=True,
    )


def _finalize_targetscan_batch_context_plan(
    plan: PreparedTargetscanBatch,
    context_outputs: list[str],
) -> str:
    """Merge TargetScan context-score outputs for one siRNA batch."""
    if plan.needs_merge:
        targetscan_path = Path(plan.targetscan_path)
        _merge_targetscan_context_outputs(
            context_outputs=context_outputs,
            targetscan_path=targetscan_path,
            log_file=Path(plan.logs_dir) / "targetscan_merge.log",
        )
        _publish_artifact_marker(
            targetscan_path.parent / "targetscan.done",
            kind="targetscan-candidate-reference-tile",
            artifacts={"targetscan": targetscan_path},
        )
    return plan.targetscan_path


def _merge_targetscan_batch_outputs(
    *,
    targetscan_paths: list[str],
    output_path: Path,
) -> None:
    """Merge TargetScan reference-shard outputs in upstream output order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scans = []
    for path in map(Path, targetscan_paths):
        if path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8") as handle:
            has_header = handle.readline().rstrip("\n") == TARGETSCAN_HEADER
        scans.append(
            pl.scan_csv(
                path,
                separator="\t",
                has_header=has_header,
                new_columns=None if has_header else list(TARGETSCAN_COLUMNS),
                schema_overrides={"targetscan_score": pl.String},
                infer_schema_length=0,
            ).select(TARGETSCAN_COLUMNS)
        )
    if not scans:
        output_path.write_text(TARGETSCAN_HEADER + "\n", encoding="utf-8")
        return

    tmp_path = _unique_tmp_path(output_path)
    pl.concat(scans).sort("refseq", "siRNA").sink_csv(
        tmp_path,
        separator="\t",
        include_header=False,
    )
    if tmp_path.stat().st_size == 0:
        tmp_path.write_text(TARGETSCAN_HEADER + "\n", encoding="utf-8")
    tmp_path.replace(output_path)


def _scan_targetscan_table(path: Path) -> pl.LazyFrame:
    """Lazily scan headerless or header-only TargetScan raw output."""
    schema = {
        "refseq": pl.String,
        "siRNA": pl.String,
        "targetscan_score": pl.Float64,
    }
    if path.stat().st_size == 0:
        return pl.LazyFrame(schema=schema)
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
    if first_line == TARGETSCAN_HEADER:
        return pl.scan_csv(path, separator="\t", schema_overrides=schema)
    return pl.scan_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(TARGETSCAN_COLUMNS),
        schema_overrides={"targetscan_score": pl.Float64},
    )


def _read_targetscan_table(path: Path) -> pl.DataFrame:
    """Read headerless or header-only TargetScan raw output."""
    return _scan_targetscan_table(path).collect(engine="streaming")


def _off_target_evidence_identity(run_root: str | Path, stem: str) -> str:
    """Return the semantic identity recorded by one evidence manifest."""
    return hash_string("\n".join((APP_INFO.off_target_cache_salt, str(run_root), stem)))


def _off_target_table_metadata(path: Path) -> dict[str, object]:
    """Validate one merged evidence table and return durable metadata."""
    if path.name == "pita.tab":
        expected_columns = ["RefSeq", "microRNA", "Sites", "Score"]
        scan = (
            pl.LazyFrame(schema={name: pl.String for name in expected_columns})
            if path.stat().st_size == 0
            else pl.scan_csv(path, separator="\t", infer_schema_length=0)
        )
    elif path.name == "targetscan.tab":
        expected_columns = list(TARGETSCAN_COLUMNS)
        if path.stat().st_size == 0:
            scan = pl.LazyFrame(schema={name: pl.String for name in expected_columns})
        else:
            with path.open(encoding="utf-8") as handle:
                has_header = handle.readline().rstrip("\n") == TARGETSCAN_HEADER
            scan = pl.scan_csv(
                path,
                separator="\t",
                has_header=has_header,
                new_columns=None if has_header else expected_columns,
                infer_schema_length=0,
            )
    else:
        raise ValueError(f"Unexpected off-target evidence table: {path.name}")
    columns = scan.collect_schema().names()
    if columns != expected_columns:
        raise ValueError(
            f"OligoFormer {path.name} schema is {columns}; expected {expected_columns}"
        )
    row_count = (
        scan.select(pl.len().alias("row_count")).collect(engine="streaming").item()
    )
    return {
        "columns": expected_columns,
        "row_count": row_count,
        "size": path.stat().st_size,
        "sha256": _hash_path(path),
    }


def _publish_off_target_manifest(
    raw_off_target_dir: Path,
    *,
    identity: str,
) -> None:
    """Atomically publish validated merged off-target evidence metadata."""
    import orjson

    tables = {
        name: _off_target_table_metadata(raw_off_target_dir / name)
        for name in ("pita.tab", "targetscan.tab")
    }
    marker_path = raw_off_target_dir / "off_target.done"
    tmp_path = _unique_tmp_path(marker_path)
    try:
        tmp_path.write_bytes(
            orjson.dumps({
                "version": 1,
                "identity": identity,
                "cache_salt": APP_INFO.off_target_cache_salt,
                "tables": tables,
            })
        )
        tmp_path.replace(marker_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _raw_off_target_ready(
    raw_off_target_dir: Path,
    *,
    expected_identity: str | None = None,
) -> bool:
    """Return whether merged evidence matches its deeply validated manifest."""
    import orjson

    marker_path = raw_off_target_dir / "off_target.done"
    if not marker_path.is_file():
        return False
    try:
        manifest = orjson.loads(marker_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("cache_salt") != APP_INFO.off_target_cache_salt
        or not isinstance(manifest.get("identity"), str)
        or (
            expected_identity is not None
            and manifest.get("identity") != expected_identity
        )
    ):
        return False
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or set(tables) != {"pita.tab", "targetscan.tab"}:
        return False
    for name, expected_columns in (
        ("pita.tab", ["RefSeq", "microRNA", "Sites", "Score"]),
        ("targetscan.tab", list(TARGETSCAN_COLUMNS)),
    ):
        metadata = tables.get(name)
        path = raw_off_target_dir / name
        if (
            not isinstance(metadata, dict)
            or metadata.get("columns") != expected_columns
            or not isinstance(metadata.get("row_count"), int)
            or metadata["row_count"] < 0
            or not isinstance(metadata.get("size"), int)
            or metadata["size"] < 0
            or not isinstance(metadata.get("sha256"), str)
            or not path.is_file()
            or path.stat().st_size != metadata["size"]
            or _hash_path(path) != metadata["sha256"]
        ):
            return False
    return True


def _copy_merged_off_target_evidence(
    raw_off_target_dir: Path,
    infer_dir: Path,
) -> None:
    """Stream compact evidence into upstream's working directory."""
    for name in ("pita.tab", "targetscan.tab"):
        shutil.copyfile(raw_off_target_dir / name, infer_dir / name)


@contextmanager
def _cache_build_lock(
    stage: str,
    identity: str,
    *,
    rebuild: bool = False,
    coalesce_rebuild: bool = True,
):
    """Elect one cache builder using append-only lock generations.

    Coalescing lets repair waiters share an already-active rebuild. Exclusive
    callers retain rebuild intent until they acquire their own next generation.
    """
    from time import sleep, time
    from uuid import uuid4

    locks = modal.Dict.from_name(
        APP_INFO.cache_lock_dict_name,
        create_if_missing=True,
    )
    lock_key = hash_string(f"{stage}\n{identity}")
    head_key = f"{lock_key}:head"
    owner = {"id": uuid4().hex, "acquired_at": time()}
    stored_head = locks.get(head_key, 0)
    generation = stored_head if isinstance(stored_head, int) else 0
    owns_generation = False
    rebuild_pending = rebuild
    while True:
        owner_key = f"{lock_key}:owner:{generation}"
        status_key = f"{lock_key}:status:{generation}"
        if locks.put(owner_key, owner, skip_if_exists=True):
            owns_generation = True
            locks.put(head_key, generation)
            break
        status = locks.get(status_key)
        if isinstance(status, dict) and status.get("state") == "complete":
            if locks.get(f"{lock_key}:owner:{generation + 1}") is not None:
                if coalesce_rebuild:
                    rebuild_pending = False
                generation += 1
                continue
            if rebuild_pending:
                rebuild_pending = False
                generation += 1
                continue
            break
        if isinstance(status, dict) and status.get("state") in {
            "abandoned",
            "failed",
        }:
            generation += 1
            continue
        current = locks.get(owner_key)
        if rebuild_pending and coalesce_rebuild and isinstance(current, dict):
            # An existing active generation is already rebuilding this identity.
            # Follow it instead of cascading another rebuild after it completes.
            rebuild_pending = False
        if (
            isinstance(current, dict)
            and isinstance(current.get("acquired_at"), (int, float))
            and time() - current["acquired_at"] > APP_INFO.cache_lock_stale_seconds
        ):
            locks.put(
                status_key,
                {"state": "abandoned", "recorded_at": time()},
                skip_if_exists=True,
            )
            continue
        sleep(APP_INFO.cache_lock_poll_seconds)
    try:
        yield owns_generation
    except BaseException:
        if owns_generation:
            locks.put(
                status_key,
                {"state": "failed", "recorded_at": time()},
                skip_if_exists=True,
            )
        raise
    else:
        if owns_generation:
            locks.put(
                status_key,
                {"state": "complete", "recorded_at": time()},
                skip_if_exists=True,
            )


def _cleanup_off_target_transients(raw_off_target_dir: Path) -> None:
    """Remove bulky shard inputs after merged off-target evidence exists."""
    import shutil

    if not _raw_off_target_ready(raw_off_target_dir):
        raise RuntimeError("Refusing to clean up without validated off-target evidence")
    keep_names = {"off_target.done", "pita.tab", "targetscan.tab"}
    for child in raw_off_target_dir.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _discard_invalid_off_target_evidence(
    raw_off_target_dir: Path,
    *,
    expected_identity: str,
) -> bool:
    """Discard one invalid published evidence set after acquiring rebuild ownership."""
    marker_path = raw_off_target_dir / "off_target.done"
    if _raw_off_target_ready(
        raw_off_target_dir,
        expected_identity=expected_identity,
    ):
        return False
    evidence_paths = (
        marker_path,
        raw_off_target_dir / "pita.tab",
        raw_off_target_dir / "targetscan.tab",
    )
    had_artifacts = any(path.exists() for path in evidence_paths)
    # Invalidate publication first; tables without the manifest are never reusable.
    marker_path.unlink(missing_ok=True)
    raw_off_target_dir.joinpath("pita.tab").unlink(missing_ok=True)
    raw_off_target_dir.joinpath("targetscan.tab").unlink(missing_ok=True)
    return had_artifacts


def _run_pita_row_shard(
    spec: PitaRowShardSpec,
    attempts: int = APP_INFO.default_pita_row_attempts,
) -> str:
    """Run or reuse one cached PITA row-shard score table."""
    import subprocess as sp
    from time import sleep

    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if _artifact_marker_ready(
        marker_path,
        kind="pita-row-score",
        artifacts={"scored_rows": output_path},
    ):
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f"oligoformer_pita_rows_{spec.sirna_index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        input_path = workdir / "potential_targets.tsv"
        shutil.copyfile(spec.input_path, input_path)
        pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
        if attempts < 1:
            raise ValueError("attempts must be a positive integer")
        transient_return_codes = {-2, -15}
        for attempt in range(1, attempts + 1):
            tmp_output_path = _unique_tmp_path(output_path)
            cmd = (
                "set -euo pipefail; "
                f"cat {shlex.quote(str(input_path))} "
                f"| {shlex.quote(str(pita_lib / 'join.pl'))} -1 2 - "
                f"{shlex.quote(spec.ext_utr_path)} "
                f"| {shlex.quote(str(pita_lib / 'cut.pl'))} -f 1-9,11- "
                f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 2,3 -a 200 "
                f"| {shlex.quote(str(pita_lib / 'RNAddG_compute.pl'))} "
                "-ddgarea 70 -upstream_rest 0 -downstream_rest 0 "
                f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 2,3 -s 200 "
                f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} "
                "-c 9,10,11,12,13,14 -m '\"-1\"' "
                f"| {shlex.quote(str(pita_lib / 'cut.pl'))} -f 1-15,14 "
                f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 15 -sc 14 "
                f"| {shlex.quote(str(pita_lib / 'cut.pl'))} -f 1-12,14-16,13 "
                f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} "
                "-c 9,10,11,12,13,14,15 -p 2 -m '\"-1\"' "
                f"> {shlex.quote(str(tmp_output_path))}"
            )
            try:
                run_command(
                    ["bash", "-lc", cmd],
                    cwd=workdir,
                    output_mode="log",
                    log_file=spec.log_path,
                    show_command=False,
                    warn_on_error=False,
                )
            except sp.CalledProcessError as exc:
                tmp_output_path.unlink(missing_ok=True)
                can_retry = (
                    exc.returncode in transient_return_codes and attempt < attempts
                )
                if can_retry:
                    print(
                        "💊 Retrying OligoFormer PITA row shard "
                        f"{spec.record_name}:{spec.shard_index} after signal "
                        f"{-exc.returncode}; log: {spec.log_path}"
                    )
                    sleep(min(30, 2 ** (attempt - 1)))
                    continue
                raise RuntimeError(
                    "OligoFormer PITA row shard "
                    f"{spec.record_name}:{spec.shard_index} failed with return "
                    f"code {exc.returncode}. Check log file {spec.log_path}."
                ) from exc
            tmp_output_path.replace(output_path)
            break
    _publish_artifact_marker(
        marker_path,
        kind="pita-row-score",
        artifacts={"scored_rows": output_path},
    )
    return str(output_path)


def _write_pita_targets_from_scored_rows(
    prepared: PreparedOffTargetShard, row_outputs: list[Path]
) -> None:
    """Reduce scored PITA rows to upstream-shaped per-siRNA target scores."""
    cache_dir = Path(prepared.cache_dir)
    raw_results_path = cache_dir / f"{prepared.record_name}_pita_results.tab"
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_results_path.open("w", encoding="utf-8") as out:
        out.write(
            "UTR\tmicroRNA\tStart\tEnd\tSeed\tMismatchs\tG:U\tLoop\tSite size\t"
            "dGduplex\tdG5\tdG3\tdG0\tdG1\tdGopen\tddG\n"
        )
        for row_output in row_outputs:
            if row_output.exists():
                with row_output.open("r", encoding="utf-8") as rows:
                    for line in rows:
                        out.write(line if line.endswith("\n") else f"{line}\n")

    with TemporaryDirectory(
        prefix=f"oligoformer_pita_finalize_{prepared.index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
        local_results = workdir / raw_results_path.name
        shutil.copyfile(raw_results_path, local_results)
        tmp_path = workdir / f"tmp_{os.getpid()}"
        final_results = workdir / local_results.name
        targets_path = workdir / f"{prepared.record_name}_pita_results_targets.tab"
        log_file = (
            Path(prepared.row_shards[0].log_path).parent.parent / "pita_finalize.log"
            if prepared.row_shards
            else Path(prepared.logs_dir) / "pita_finalize.log"
        )
        cmd = (
            "set -euo pipefail; "
            f"mv {shlex.quote(str(local_results))} {shlex.quote(str(tmp_path))}; "
            f"head -n 1 {shlex.quote(str(tmp_path))} | cut -f 1-5,8,10- "
            f"> {shlex.quote(str(final_results))}; "
            f"cat {shlex.quote(str(tmp_path))} "
            f"| {shlex.quote(str(pita_lib / 'body.pl'))} 2 -1 "
            "| tr -d '\\r' "
            "| cut -f 1-8,10- "
            f"| {shlex.quote(str(pita_lib / 'merge_columns.pl'))} -1 4 -2 5 -d ':' "
            f"| {shlex.quote(str(pita_lib / 'merge_columns.pl'))} -1 4 -2 5 -d ':' "
            "| sed 's/Seed:Mismatchs:G:U/Seed/g' "
            f"| sort -k 13n >> {shlex.quote(str(final_results))}; "
            f"cat {shlex.quote(str(final_results))} "
            f"| {shlex.quote(str(pita_lib / 'body.pl'))} 2 -1 "
            "| cut -f 1,2,13 "
            f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 2 -m '\"-1\"' "
            f"| {shlex.quote(str(pita_lib / 'average_rows.pl'))} -k 0,1 -losoe -n "
            f"| {shlex.quote(str(pita_lib / 'cut.pl'))} -f 2,3,1,4 "
            f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 3 -m '\"-1\"' "
            f"| {shlex.quote(str(pita_lib / 'modify_column.pl'))} -c 3 -p 2 "
            "| sort -k 4n "
            f"| {shlex.quote(str(pita_lib / 'cap.pl'))} "
            "'RefSeq,microRNA,Sites,Score' "
            f"> {shlex.quote(str(targets_path))}"
        )
        run_command(
            ["bash", "-lc", cmd],
            cwd=workdir,
            output_mode="log",
            log_file=log_file,
            show_command=False,
        )
        pita_path = Path(prepared.pita_path)
        pita_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_pita_path = _unique_tmp_path(pita_path)
        try:
            shutil.copy2(targets_path, tmp_pita_path)
            tmp_pita_path.replace(pita_path)
        finally:
            tmp_pita_path.unlink(missing_ok=True)
    _publish_artifact_marker(
        Path(prepared.cache_dir, "pita_finalize.done"),
        kind="pita-candidate-final",
        artifacts={"pita": Path(prepared.pita_path)},
    )


def _finalize_oligoformer_pita_shard(
    prepared: PreparedOffTargetShard,
) -> OffTargetShardResult:
    """Finalize one per-siRNA PITA table from cached row shards."""
    pita_path = Path(prepared.pita_path)
    marker_path = Path(prepared.cache_dir) / "pita_finalize.done"
    if not _artifact_marker_ready(
        marker_path,
        kind="pita-candidate-final",
        artifacts={"pita": pita_path},
    ):
        row_outputs = [
            Path(row.output_path)
            for row in sorted(prepared.row_shards, key=lambda item: item.shard_index)
        ]
        _write_pita_targets_from_scored_rows(prepared, row_outputs)
    return OffTargetShardResult(
        index=prepared.index,
        pita_path=prepared.pita_path,
    )


def _pita_candidate_ready(spec: OffTargetShardSpec) -> bool:
    """Return whether one complete PITA candidate publication is valid."""
    cache_dir = _off_target_shard_cache_dir(spec)
    return _artifact_marker_ready(
        cache_dir / "pita_finalize.done",
        kind="pita-candidate-final",
        artifacts={"pita": cache_dir / "pita.tab"},
    )


def _merge_pita_shards(shard_results: list[OffTargetShardResult], output_path: Path):
    """Merge per-siRNA PITA tables in upstream score-sorted order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(shard.pita_path)
        for shard in sorted(shard_results, key=lambda result: result.index)
        if Path(shard.pita_path).stat().st_size > 0
    ]
    if not paths:
        output_path.write_text(
            "RefSeq\tmicroRNA\tSites\tScore\n",
            encoding="utf-8",
        )
        return
    _warmup_file_parents(paths, file_pattern=r"^pita\.tab$")
    scan = pl.scan_csv(paths, separator="\t", infer_schema_length=0)
    columns = scan.collect_schema().names()
    if "Score" not in columns:
        raise ValueError("PITA shard output must contain a Score column")
    tmp_path = _unique_tmp_path(output_path)
    (
        scan
        .with_columns(
            pl.col("Score").cast(pl.Float64).alias("_biomodals_score"),
            pl.concat_str(columns, separator="\t").alias("_biomodals_row"),
        )
        .sort("_biomodals_score", "_biomodals_row")
        .select(columns)
        .sink_csv(tmp_path, separator="\t")
    )
    tmp_path.replace(output_path)


def _prepare_pita_target_discovery_plan_for_spec(
    *,
    spec: OffTargetShardSpec,
    prepare_root: Path,
    reference: PitaReferencePlan | None = None,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> PitaPreparePlan:
    """Prepare one siRNA's PITA target-discovery plan."""
    import shutil

    shard_root = prepare_root / f"{spec.stem}_shard_{spec.index:05d}"
    off_target_root = shard_root / "off-target"
    off_target_root.mkdir(parents=True)
    shutil.copytree(CONF.git_clone_dir / "off-target/pita", off_target_root / "pita")
    try:
        return _prepare_pita_target_discovery_plan(
            spec, shard_root, reference, execution
        )
    finally:
        shutil.rmtree(shard_root, ignore_errors=True)


def _off_target_branch_slots(
    execution: OligoformerExecutionConfig,
) -> tuple[int, int]:
    """Split the existing run-wide process budget between both CPU branches."""
    targetscan_slots = max(1, execution.off_target_process_slots // 2)
    return targetscan_slots, execution.off_target_process_slots - targetscan_slots


def _pita_local_workers(execution: OligoformerExecutionConfig) -> tuple[int, int]:
    """Derive per-container PITA worker counts from its process-slot share."""
    _, slots = _off_target_branch_slots(execution)
    nodes = min(execution.off_target_nodes, slots)
    workers = max(1, slots // nodes)
    return (
        min(
            workers,
            execution.off_target_prep_workers,
            execution.pita_prepare_workers,
        ),
        min(workers, execution.off_target_workers),
    )


def _targetscan_local_workers(execution: OligoformerExecutionConfig) -> int:
    """Derive per-container TargetScan workers from its process-slot share."""
    slots, _ = _off_target_branch_slots(execution)
    nodes = min(
        execution.targetscan_prepare_nodes,
        execution.targetscan_context_nodes,
        slots,
    )
    return min(
        execution.targetscan_context_workers,
        max(1, slots // nodes),
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def prepare_oligoformer_pita_reference(
    spec: OffTargetShardSpec,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> PitaReferencePlan:
    """Publish reusable PITA reference STAB shards for one evidence stem."""
    CONF.output_volume.reload()
    with TemporaryDirectory(prefix=f"oligoformer_{spec.stem}_pita_reference_") as tmp:
        plan = _prepare_pita_reference_plan(spec, Path(tmp), execution)
    CONF.output_volume.commit()
    return plan


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_pita_candidate(
    spec: OffTargetShardSpec,
    reference: PitaReferencePlan,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> OffTargetShardResult:
    """Run one complete deterministic PITA candidate Task."""
    CONF.output_volume.reload()
    prepare_workers, row_workers = _pita_local_workers(execution)
    with TemporaryDirectory(prefix=f"oligoformer_{spec.stem}_pita_candidate_") as tmp:
        plan = _prepare_pita_target_discovery_plan_for_spec(
            spec=spec,
            prepare_root=Path(tmp),
            reference=reference,
            execution=execution,
        )
    bounded_map(
        plan.utr_shards,
        lambda shard: _run_pita_prepare_utr_shard(
            shard,
            execution.pita_row_attempts,
        ),
        max_parallel=prepare_workers,
    )
    prepared = _finalize_pita_target_discovery_plan(plan)
    bounded_map(
        prepared.row_shards,
        lambda row: _run_pita_row_shard(row, execution.pita_row_attempts),
        max_parallel=row_workers,
    )
    result = _finalize_oligoformer_pita_shard(prepared)
    CONF.output_volume.commit()
    return result


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_oligoformer_targetscan_tile(
    spec: TargetscanBatchSpec,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> str:
    """Run one candidate/reference TargetScan tile without nested calls."""
    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    with TemporaryDirectory(
        prefix=f"oligoformer_targetscan_{spec.stem}_{spec.shard_index}_"
    ) as tmp:
        batch_root = Path(tmp)
        off_target_root = batch_root / "off-target"
        off_target_root.mkdir(parents=True)
        shutil.copytree(
            CONF.git_clone_dir / "off-target/targetscan",
            off_target_root / "targetscan",
        )
        plan = _prepare_targetscan_batch_context_plan(
            spec=spec,
            batch_root=batch_root,
        )
    context_outputs = bounded_map(
        plan.context_shards,
        lambda shard: _run_targetscan_context_shard(
            shard,
            execution.targetscan_context_attempts,
        ),
        max_parallel=_targetscan_local_workers(execution),
    )
    targetscan_path = _finalize_targetscan_batch_context_plan(
        plan,
        list(context_outputs),
    )
    CONF.output_volume.commit()
    return targetscan_path


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def plan_oligoformer_off_target_evidence(
    plan: OligoformerRunPlan,
    targetscan_ref_shard_size: int | None = None,
    execution: OligoformerExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> OligoformerEvidencePlan:
    """Discover finite PITA candidates and TargetScan tiles after efficacy."""
    if not plan.config.off_target:
        return OligoformerEvidencePlan(())
    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    refreshed = _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )
    if not refreshed.efficacy_ready:
        raise FileNotFoundError("OligoFormer efficacy outputs are incomplete")
    layout = AppRunLayout.from_run_root(refreshed.run_root)
    if refreshed.config.all_human:
        _ensure_human_refs()
        if not _targetscan_rnaplfold_cache_ready():
            raise FileNotFoundError("OligoFormer RNAplfold reference cache is missing")
        utr_path = str(APP_INFO.model_ref_dir / "human_UTR.txt")
        orf_path = str(APP_INFO.model_ref_dir / "human_ORF.txt")
    else:
        utr_path = str(layout.inputs_dir / "utr.txt")
        orf_path = str(layout.inputs_dir / "orf.txt")

    stem_plans = []
    for stem in refreshed.output_stems:
        records = _off_target_sirna_records(
            _read_efficacy_output(Path(refreshed.efficacy_dir) / f"{stem}.txt"),
            refreshed.config.top_n,
        )
        if not records:
            raise RuntimeError(
                f"No siRNA records are available for off-target prediction: {stem}"
            )
        pita_specs = tuple(
            _off_target_shard_spec(
                run_root=refreshed.run_root,
                output_dir=Path(refreshed.output_dir),
                stem=stem,
                item=item,
                utr_path=utr_path,
                orf_path=orf_path,
                row_shard_size=execution.pita_row_shard_size,
            )
            for item in enumerate(records)
        )
        targetscan_specs = tuple(
            _targetscan_batch_specs(
                run_root=refreshed.run_root,
                output_dir=Path(refreshed.output_dir),
                stem=stem,
                records=records,
                utr_path=utr_path,
                orf_path=orf_path,
                ref_shard_size=targetscan_ref_shard_size,
                execution=execution,
            )
        )
        stem_plans.append(
            OligoformerEvidenceStemPlan(
                stem=stem,
                pita_specs=pita_specs,
                targetscan_specs=targetscan_specs,
            )
        )
    CONF.output_volume.commit()
    return OligoformerEvidencePlan(tuple(stem_plans))


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def publish_oligoformer_off_target_evidence(
    run_root: str,
    stem_plan: OligoformerEvidenceStemPlan,
) -> None:
    """Merge complete scientific tiles and publish one evidence manifest."""
    CONF.output_volume.reload()
    evidence_dir = (
        AppRunLayout.from_run_root(run_root).prep_dir / "off_target" / stem_plan.stem
    )
    identity = _off_target_evidence_identity(run_root, stem_plan.stem)
    if _raw_off_target_ready(evidence_dir, expected_identity=identity):
        return
    pita_results = [
        OffTargetShardResult(
            index=spec.index,
            pita_path=str(_off_target_shard_cache_dir(spec) / "pita.tab"),
        )
        for spec in stem_plan.pita_specs
    ]
    targetscan_paths = [
        str(_targetscan_batch_cache_dir(spec) / "targetscan.tab")
        for spec in stem_plan.targetscan_specs
    ]
    _merge_pita_shards(pita_results, evidence_dir / "pita.tab")
    _merge_targetscan_batch_outputs(
        targetscan_paths=targetscan_paths,
        output_path=evidence_dir / "targetscan.tab",
    )
    _publish_off_target_manifest(evidence_dir, identity=identity)
    CONF.output_volume.commit()


def _apply_off_target_filters(
    *,
    result,
    run_root: str,
    stem: str,
    top_n: int,
    pita_threshold: float,
    targetscan_threshold: float,
):
    """Apply filters from already-published PITA and TargetScan evidence."""
    import shutil

    infer_dir = CONF.git_clone_dir / "data/infer" / stem
    if infer_dir.exists():
        shutil.rmtree(infer_dir)
    infer_dir.mkdir(parents=True)

    if top_n == -1:
        sirna_file = infer_dir / "siRNA.fa"
    else:
        sirna_file = infer_dir / "top_n_siRNA.fa"
    records = _off_target_sirna_records(result, top_n)
    _write_sirna_records(records, sirna_file)
    evidence_dir = AppRunLayout.from_run_root(run_root).prep_dir / "off_target" / stem
    if not _raw_off_target_ready(
        evidence_dir,
        expected_identity=_off_target_evidence_identity(run_root, stem),
    ):
        raise FileNotFoundError(
            f"OligoFormer off-target evidence is incomplete for {stem}"
        )
    _copy_merged_off_target_evidence(evidence_dir, infer_dir)

    original_columns = list(result.columns)
    result = result.with_columns(
        (pl.lit("RNA") + (pl.col("pos").cast(pl.Int64) - 1).cast(pl.String)).alias(
            "tmp"
        )
    )
    evaluated_sirnas = pl.DataFrame({
        "tmp": [record.name for record in records],
        "_off_target_evaluated": [True] * len(records),
    })
    result = result.join(evaluated_sirnas, on="tmp", how="left")
    pita = (
        pl
        .scan_csv(
            infer_dir / "pita.tab",
            separator="\t",
            schema_overrides={"Score": pl.String},
        )
        .with_columns(
            pl.col("Score").cast(pl.Float64, strict=False).alias("_pita_score")
        )
        .group_by("microRNA")
        .agg(
            pl
            .col("Score")
            .get(pl.col("_pita_score").arg_min().fill_null(0))
            .alias("pita_score"),
            pl.col("_pita_score").min().alias("_pita_score"),
        )
        .collect(engine="streaming")
    )
    result = result.join(pita, left_on="tmp", right_on="microRNA", how="left")
    result = result.with_columns(
        pl
        .when(pl.col("_pita_score") < pita_threshold)
        .then(1)
        .otherwise(0)
        .alias("pita_filter")
    )

    targetscan = (
        _scan_targetscan_table(infer_dir / "targetscan.tab")
        .group_by("siRNA")
        .agg(pl.col("targetscan_score").max().alias("targetscan_score"))
        .collect(engine="streaming")
    )
    result = result.join(targetscan, left_on="tmp", right_on="siRNA", how="left")
    result = result.with_columns(
        pl
        .when(
            pl.col("_pita_score").is_not_null() & pl.col("targetscan_score").is_null()
        )
        .then(0.0)
        .otherwise(pl.col("targetscan_score"))
        .alias("targetscan_score")
    )
    result = result.with_columns(
        pl
        .when(pl.col("targetscan_score") > targetscan_threshold)
        .then(1)
        .otherwise(0)
        .alias("targetscan_filter")
    )

    pita_hit = pl.col("pita_filter") == 1
    targetscan_hit = pl.col("targetscan_filter") == 1
    if top_n == -1:
        off_target_filter = pl.when(pita_hit | targetscan_hit).then(1).otherwise(0)
    else:
        was_evaluated = pl.col("_off_target_evaluated").fill_null(False)
        off_target_filter = (
            pl
            .when(~was_evaluated)
            .then(-5)
            .when(pita_hit | targetscan_hit)
            .then(1)
            .otherwise(0)
        )
    result = result.with_columns(off_target_filter.alias("off_target_filter"))
    return result.select(
        original_columns + ["pita_score", "targetscan_score", "off_target_filter"]
    )


def _apply_toxicity_filters(*, result, toxicity_threshold: float):
    """Apply upstream-equivalent toxicity post-processing."""
    toxicity_table = pl.read_csv(
        CONF.git_clone_dir / "toxicity/cell_viability.txt", separator="\t"
    )
    return (
        result
        .with_columns(pl.col("siRNA").str.slice(1, 6).alias("Seed"))
        .join(toxicity_table, on="Seed", how="left")
        .with_columns(
            pl
            .when(pl.col("cell_viability") < toxicity_threshold)
            .then(1)
            .otherwise(0)
            .alias("toxicity_filter")
        )
    )


def _apply_final_filter(
    *, result, off_target: bool, toxicity: bool, functionality_filter: bool
):
    """Apply upstream-equivalent final filter aggregation."""
    if "filter" in result.columns:
        result = result.drop("filter")

    filter_terms = []
    if functionality_filter:
        filter_terms.append(pl.col("func_filter") != 0)
    if off_target:
        filter_terms.append(pl.col("off_target_filter") != 0)
    if toxicity:
        filter_terms.append(pl.col("toxicity_filter") != 0)

    filter_expr = pl.lit(0)
    for term in filter_terms:
        filter_expr = filter_expr + term.cast(pl.Int64)
    return result.with_columns(filter_expr.alias("filter"))


def _write_final_outputs(result: pl.DataFrame, output_dir: Path, stem: str) -> None:
    """Write upstream-shaped OligoFormer final output tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = result.sort("efficacy", descending=True)
    ranked_filtered = result.filter(pl.col("filter") == 0).sort(
        "efficacy", descending=True
    )
    result.write_csv(output_dir / f"{stem}.txt", separator="\t")
    ranked.write_csv(output_dir / f"{stem}_ranked.txt", separator="\t")
    ranked_filtered.write_csv(
        output_dir / f"{stem}_ranked_filtered.txt", separator="\t"
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def prepare_oligoformer_run(
    mrna_fasta_bytes: bytes,
    sirna_fasta_bytes: bytes | None = None,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    utr_bytes: bytes | None = None,
    orf_bytes: bytes | None = None,
    top_n: int = APP_INFO.default_top_n,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
    force: bool = False,
    force_generation: str | None = None,
) -> OligoformerRunPlan:
    """Prepare or discover a volume-backed OligoFormer run cache."""
    if top_n != -1 and top_n < 1:
        raise ValueError("top_n must be -1 or a positive integer")
    if off_target and not all_human and (utr_bytes is None or orf_bytes is None):
        raise ValueError(
            "OligoFormer off-target mode requires both UTR and ORF references "
            "unless all_human is enabled."
        )

    mrna_fasta_bytes, output_stems = _sanitize_fasta_record_names(
        mrna_fasta_bytes,
        label="mRNA",
    )
    if sirna_fasta_bytes is not None:
        sirna_fasta_bytes, _ = _sanitize_fasta_record_names(
            sirna_fasta_bytes,
            label="siRNA",
        )
    if off_target and not all_human:
        if utr_bytes is None or orf_bytes is None:
            raise ValueError("Custom off-target references are incomplete")
        utr_bytes, orf_bytes = _sanitize_paired_reference_fastas(
            utr_bytes,
            orf_bytes,
        )
    if force and force_generation is None:
        from uuid import uuid4

        force_generation = uuid4().hex
    CONF.output_volume.reload()
    config = OligoformerRunConfig(
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
    )
    try:
        model_identity = _rnafm_model_identity_digest()
    except FileNotFoundError:
        model_identity = None
    efficacy_key = _efficacy_key_for_run(
        mrna_fasta_bytes=mrna_fasta_bytes,
        sirna_fasta_bytes=sirna_fasta_bytes,
        functionality_filter=functionality_filter,
        model_identity=model_identity,
        force_generation=force_generation if force else None,
    )
    reference_identity = (
        _targetscan_ref_identity_digest() if off_target and all_human else None
    )
    cache_key = _cache_key_for_run(
        efficacy_key=efficacy_key,
        utr_bytes=utr_bytes,
        orf_bytes=orf_bytes,
        config=config,
        reference_identity=reference_identity,
    )
    postprocess_key = _postprocess_key_for_run(
        cache_key=cache_key,
        config=config,
    )

    layout = _run_layout_for_cache_key(cache_key, output_stems)
    plan = _build_plan(
        cache_key,
        efficacy_key,
        output_stems,
        config=config,
        postprocess_key=postprocess_key,
        reference_identity=reference_identity,
        model_identity=model_identity,
    )
    if plan.final_ready:
        return plan

    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    (layout.inputs_dir / "mrna.fa").write_bytes(mrna_fasta_bytes)
    if sirna_fasta_bytes is not None:
        (layout.inputs_dir / "sirna.fa").write_bytes(sirna_fasta_bytes)
    if off_target and not all_human:
        (layout.inputs_dir / "utr.txt").write_bytes(utr_bytes or b"")
        (layout.inputs_dir / "orf.txt").write_bytes(orf_bytes or b"")

    CONF.output_volume.commit()
    return _build_plan(
        cache_key,
        efficacy_key,
        output_stems,
        config=config,
        postprocess_key=postprocess_key,
        reference_identity=reference_identity,
        model_identity=model_identity,
    )


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_oligoformer_efficacy(
    plan: OligoformerRunPlan,
    functionality_filter: bool = True,
) -> OligoformerRunPlan:
    """Run GPU efficacy prediction into the output-volume cache."""
    if functionality_filter != plan.config.functionality_filter:
        raise ValueError(
            "OligoFormer efficacy settings do not match the prepared run plan"
        )
    CONF.output_volume.reload()
    refreshed_plan = _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )
    if refreshed_plan.efficacy_ready:
        return refreshed_plan

    efficacy_layout = _efficacy_layout_for_key(
        plan.efficacy_key,
        plan.output_stems,
    )
    efficacy_marker = _marker_path(efficacy_layout, "efficacy.done")
    efficacy_paths = _output_paths(Path(plan.efficacy_dir), plan.output_stems)
    published_efficacy_artifacts = efficacy_marker.is_file() or any(
        path.exists() for path in efficacy_paths
    )
    with _cache_build_lock(
        "efficacy",
        plan.efficacy_key,
        rebuild=True,
    ) as owns_cache_build:
        CONF.output_volume.reload()
        refreshed_plan = _build_plan(
            plan.cache_key,
            plan.efficacy_key,
            plan.output_stems,
            plan.run_root,
            config=plan.config,
            postprocess_key=plan.postprocess_key,
            reference_identity=plan.reference_identity,
            model_identity=plan.model_identity,
        )
        if refreshed_plan.efficacy_ready:
            return refreshed_plan
        if not owns_cache_build:
            raise RuntimeError(
                "OligoFormer efficacy cache was marked complete without outputs"
            )
        if published_efficacy_artifacts:
            _discard_output_bundle(
                efficacy_marker,
                efficacy_paths,
            )
        MODEL_VOLUME.reload()
        if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
            raise FileNotFoundError(
                "OligoFormer RNA-FM weights are missing. Run "
                "download_oligoformer_models first."
            )
        if plan.model_identity is None:
            raise FileNotFoundError(
                "OligoFormer RNA-FM identity was not available during run "
                "preparation. Prepare and submit the run again."
            )
        if (
            not _rnafm_model_identity_matches_model()
            or _rnafm_model_identity_digest() != plan.model_identity
        ):
            raise FileNotFoundError(
                "OligoFormer RNA-FM weights changed after run preparation. "
                "Prepare and submit the run again."
            )

        _ensure_rnafm_runtime()
        input_layout = AppRunLayout.from_run_root(plan.run_root)
        efficacy_dir = Path(plan.efficacy_dir)
        efficacy_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "scripts/main.py",
            "-i",
            "1",
            "-i1",
            str(input_layout.inputs_dir / "mrna.fa"),
            "--output_dir",
            f"{efficacy_dir}/",
            "--biomodals_stage",
            "efficacy",
        ]

        sirna_fasta = input_layout.inputs_dir / "sirna.fa"
        if sirna_fasta.exists():
            cmd.extend(["-i2", str(sirna_fasta)])
        if not functionality_filter:
            cmd.append("--no_func")

        run_command(cmd, cwd=CONF.git_clone_dir)
        _publish_output_bundle_marker(
            _marker_path(efficacy_layout, "efficacy.done"),
            output_dir=efficacy_dir,
            paths=_output_paths(efficacy_dir, plan.output_stems),
            identity={
                "efficacy_key": plan.efficacy_key,
                "output_stems": list(plan.output_stems),
            },
        )
        CONF.output_volume.commit()
        return _build_plan(
            plan.cache_key,
            plan.efficacy_key,
            plan.output_stems,
            plan.run_root,
            config=plan.config,
            postprocess_key=plan.postprocess_key,
            reference_identity=plan.reference_identity,
            model_identity=plan.model_identity,
        )


def _run_oligoformer_postprocess_locked(
    plan: OligoformerRunPlan,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    top_n: int = APP_INFO.default_top_n,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
) -> bytes:
    """Build final tables while holding their cache-generation lock."""
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.run_root)
    refreshed_plan = _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )
    if refreshed_plan.final_ready:
        return _package_output_tables(
            Path(refreshed_plan.output_dir), refreshed_plan.output_stems
        )
    if not refreshed_plan.efficacy_ready:
        raise FileNotFoundError(
            "OligoFormer efficacy outputs are incomplete for "
            f"{refreshed_plan.cache_key}"
        )

    efficacy_dir = Path(refreshed_plan.efficacy_dir)
    output_dir = Path(refreshed_plan.output_dir)
    if not off_target and not toxicity:
        _copy_outputs(efficacy_dir, output_dir, refreshed_plan.output_stems)
    else:
        if off_target and all_human:
            _ensure_human_refs()
            if (
                refreshed_plan.reference_identity is None
                or _targetscan_ref_identity_digest()
                != refreshed_plan.reference_identity
                or not _targetscan_ref_identity_matches_model()
                or not _targetscan_rnaplfold_cache_ready()
            ):
                raise FileNotFoundError(
                    "OligoFormer human references changed after run preparation"
                )

        for stem in refreshed_plan.output_stems:
            result = _read_efficacy_output(efficacy_dir / f"{stem}.txt")
            if off_target:
                result = _apply_off_target_filters(
                    result=result,
                    run_root=refreshed_plan.run_root,
                    stem=stem,
                    top_n=top_n,
                    pita_threshold=pita_threshold,
                    targetscan_threshold=targetscan_threshold,
                )
            if toxicity:
                result = _apply_toxicity_filters(
                    result=result,
                    toxicity_threshold=toxicity_threshold,
                )
            result = _apply_final_filter(
                result=result,
                off_target=off_target,
                toxicity=toxicity,
                functionality_filter=functionality_filter,
            )
            _write_final_outputs(result, output_dir, stem)

    _write_cache_marker(
        layout,
        _final_marker_name(refreshed_plan.postprocess_key),
        refreshed_plan,
        extra_metadata={"postprocess_cache_salt": APP_INFO.postprocess_cache_salt},
    )
    CONF.output_volume.commit()
    if off_target:
        for stem in refreshed_plan.output_stems:
            _cleanup_off_target_transients(layout.prep_dir / "off_target" / stem)
        CONF.output_volume.commit()
    return _package_output_tables(output_dir, refreshed_plan.output_stems)


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_oligoformer_postprocess(
    plan: OligoformerRunPlan,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    top_n: int = APP_INFO.default_top_n,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
) -> bytes:
    """Run CPU post-processing and return packaged OligoFormer outputs."""
    requested_config = OligoformerRunConfig(
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
    )
    if requested_config != plan.config:
        raise ValueError(
            "OligoFormer post-processing settings do not match the prepared run plan"
        )

    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.run_root)
    final_marker = _marker_path(layout, _final_marker_name(plan.postprocess_key))
    final_paths = _output_paths(Path(plan.output_dir), plan.output_stems)
    published_final_artifacts = final_marker.is_file() or any(
        path.exists() for path in final_paths
    )
    final_ready = _paths_ready(
        final_paths,
        final_marker,
        expected_marker={
            "cache_key": plan.cache_key,
            "postprocess_key": plan.postprocess_key,
            "output_stems": list(plan.output_stems),
            "postprocess_cache_salt": APP_INFO.postprocess_cache_salt,
        },
    )
    with _cache_build_lock(
        "final-tables",
        f"{plan.run_root}\n{plan.postprocess_key}",
        rebuild=not final_ready,
    ) as owns_final_build:
        CONF.output_volume.reload()
        refreshed_plan = _build_plan(
            plan.cache_key,
            plan.efficacy_key,
            plan.output_stems,
            plan.run_root,
            config=plan.config,
            postprocess_key=plan.postprocess_key,
            reference_identity=plan.reference_identity,
            model_identity=plan.model_identity,
        )
        if refreshed_plan.final_ready:
            return _package_output_tables(
                Path(refreshed_plan.output_dir), refreshed_plan.output_stems
            )
        if not owns_final_build:
            raise RuntimeError(
                "OligoFormer final-table cache was marked complete without outputs"
            )
        if published_final_artifacts:
            _discard_output_bundle(final_marker, final_paths)
        return _run_oligoformer_postprocess_locked(
            plan=plan,
            off_target=off_target,
            toxicity=toxicity,
            all_human=all_human,
            top_n=top_n,
            functionality_filter=functionality_filter,
            pita_threshold=pita_threshold,
            targetscan_threshold=targetscan_threshold,
            toxicity_threshold=toxicity_threshold,
        )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def build_oligoformer_final_tables(
    plan: OligoformerRunPlan,
) -> OligoformerRunPlan:
    """Build final tables and return only their refreshed publication plan."""
    run_oligoformer_postprocess.get_raw_f()(
        plan=plan,
        off_target=plan.config.off_target,
        toxicity=plan.config.toxicity,
        all_human=plan.config.all_human,
        top_n=plan.config.top_n,
        functionality_filter=plan.config.functionality_filter,
        pita_threshold=plan.config.pita_threshold,
        targetscan_threshold=plan.config.targetscan_threshold,
        toxicity_threshold=plan.config.toxicity_threshold,
    )
    return _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def package_oligoformer_outputs(plan: OligoformerRunPlan) -> bytes:
    """Return cached OligoFormer outputs as standalone tarball bytes."""
    CONF.output_volume.reload()
    refreshed_plan = _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )
    if not refreshed_plan.final_ready:
        raise FileNotFoundError(
            f"OligoFormer final outputs are incomplete for {refreshed_plan.cache_key}"
        )
    return _package_output_tables(
        Path(refreshed_plan.output_dir), refreshed_plan.output_stems
    )


def _oligoformer_result_archive_path(plan: OligoformerRunPlan) -> Path:
    """Return the reconstructable result archive published for direct clients."""
    return Path(plan.output_dir) / "oligoformer.tar.zst"


def _oligoformer_result_record_path(
    output_root: str | Path,
    publication_key: str,
) -> Path:
    if len(publication_key) != 64 or any(
        character not in "0123456789abcdef" for character in publication_key
    ):
        raise ValueError("OligoFormer result publication key must be SHA-256")
    return (
        Path(output_root)
        / ".biomodals"
        / "oligoformer"
        / "results"
        / f"{publication_key}.json"
    )


def _parse_oligoformer_result_publication(
    content: bytes,
    publication_key: str,
) -> dict[str, object] | None:
    import orjson

    try:
        value = orjson.loads(content)
    except orjson.JSONDecodeError:
        return None
    if not (
        isinstance(value, dict)
        and value.get("version") == 2
        and value.get("publication_key") == publication_key
        and isinstance(value.get("result_path"), str)
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and value["size_bytes"] > 0
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and isinstance(value.get("model_identity"), str)
        and bool(value["model_identity"])
        and (
            value.get("reference_identity") is None
            or isinstance(value.get("reference_identity"), str)
        )
    ):
        return None
    relative = Path(cast(str, value["result_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return cast(dict[str, object], value)


def _publish_oligoformer_result_record(
    output_root: str | Path,
    publication_key: str,
    archive_path: Path,
    *,
    model_identity: str,
    reference_identity: str | None,
) -> None:
    """Atomically bind one reconstructable archive to its scientific plan."""
    import orjson

    root = Path(output_root).resolve()
    relative = archive_path.resolve().relative_to(root)
    size = archive_path.stat().st_size
    if archive_path.is_symlink() or size < 1:
        raise RuntimeError("OligoFormer result archive is not a regular artifact")
    marker = _oligoformer_result_record_path(root, publication_key)
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(marker)
    try:
        tmp_path.write_bytes(
            orjson.dumps({
                "version": 2,
                "publication_key": publication_key,
                "model_identity": model_identity,
                "reference_identity": reference_identity,
                "result_path": relative.as_posix(),
                "size_bytes": size,
                "sha256": _hash_path(archive_path),
            })
        )
        tmp_path.replace(marker)
    finally:
        tmp_path.unlink(missing_ok=True)


def _oligoformer_result_publication(
    output_root: str | Path,
    publication_key: str,
    *,
    expected_identities: tuple[str, str | None] | None = None,
) -> dict[str, object] | None:
    """Return one validated local archive publication, if complete."""
    marker = _oligoformer_result_record_path(output_root, publication_key)
    try:
        publication = _parse_oligoformer_result_publication(
            marker.read_bytes(),
            publication_key,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None
    if publication is None:
        return None
    if (
        expected_identities is not None
        and (
            publication["model_identity"],
            publication["reference_identity"],
        )
        != expected_identities
    ):
        return None
    archive = Path(output_root).joinpath(
        *Path(cast(str, publication["result_path"])).parts
    )
    try:
        if archive.is_symlink() or archive.stat().st_size != publication["size_bytes"]:
            return None
        if _hash_path(archive) != publication["sha256"]:
            return None
    except (FileNotFoundError, NotADirectoryError):
        return None
    return publication


def _oligoformer_result_publication_from_volume(
    volume,
    publication_key: str,
) -> dict[str, object] | None:
    """Read the bounded result record through the client-side Volume API."""
    relative = (
        Path(".biomodals") / "oligoformer" / "results" / f"{publication_key}.json"
    )
    content = bytearray()
    try:
        for chunk in volume.read_file(relative.as_posix()):
            if not isinstance(chunk, bytes):
                raise TypeError("OligoFormer result publication must be bytes")
            if len(content) + len(chunk) > 64 * 1024:
                raise ValueError("OligoFormer result publication is too large")
            content.extend(chunk)
    except FileNotFoundError:
        return None
    return _parse_oligoformer_result_publication(bytes(content), publication_key)


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def publish_oligoformer_outputs(
    plan: OligoformerRunPlan,
    publication_key: str,
) -> dict[str, object]:
    """Publish the final standalone archive for Volume API download."""
    CONF.output_volume.reload()
    refreshed = _build_plan(
        plan.cache_key,
        plan.efficacy_key,
        plan.output_stems,
        plan.run_root,
        config=plan.config,
        postprocess_key=plan.postprocess_key,
        reference_identity=plan.reference_identity,
        model_identity=plan.model_identity,
    )
    if not refreshed.final_ready:
        raise FileNotFoundError("OligoFormer final outputs are incomplete")
    if refreshed.model_identity is None:
        raise FileNotFoundError("OligoFormer model identity is unavailable")
    archive_path = _oligoformer_result_archive_path(refreshed)
    archive_bytes = _package_output_tables(
        Path(refreshed.output_dir),
        refreshed.output_stems,
    )
    tmp_path = _unique_tmp_path(archive_path)
    try:
        tmp_path.write_bytes(archive_bytes)
        tmp_path.replace(archive_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    _publish_oligoformer_result_record(
        CONF.output_volume_mountpoint,
        publication_key,
        archive_path,
        model_identity=refreshed.model_identity,
        reference_identity=refreshed.reference_identity,
    )
    CONF.output_volume.commit()
    return {"result_path": str(archive_path), "size_bytes": len(archive_bytes)}


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with OligoFormer functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh mounted state before accepting lifecycle calls."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()
        MODEL_VOLUME.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionSnapshot:
        """Drive one staged root App Run."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionSnapshot:
        """Read this Run's durable snapshot."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionSnapshot:
        """Resume without retrying failed Tasks."""
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
    def drive_prepared(self) -> ExecutionSnapshot:
        """Drive one previously prepared root or Successor Run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> ExecutionSnapshot:
        """Create a Successor Run from conclusive predecessor state."""
        return self._adapter().restart(
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
    def restart_from(
        self,
        predecessor_execution_run_id: str,
    ) -> ExecutionSnapshot:
        """Create a compatible Successor while inferring deployment identity."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                CONF.output_volume_mountpoint,
                UUID(self.execution_run_id),
            ),
        )

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
    ) -> OligoformerExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: OligoformerExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                model_volume=MODEL_VOLUME,
                output_claims=OLIGOFORMER_OUTPUT_CLAIMS,
                modal_driver=_coordinator_modal_driver(development=selected_mode),
                app_version=CONF.repo_commit_hash or CONF.version or "unknown",
                model_version=_hash_bytes(
                    orjson.dumps(
                        APP_INFO.rnafm_identity_metadata,
                        option=orjson.OPT_SORT_KEYS,
                    )
                ),
                reference_version=_hash_bytes(
                    orjson.dumps(
                        APP_INFO.targetscan_ref_metadata,
                        option=orjson.OPT_SORT_KEYS,
                    )
                ),
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "download_oligoformer_models": download_oligoformer_models,
            "prepare_oligoformer_run": prepare_oligoformer_run,
            "plan_oligoformer_targetscan_rnaplfold_cache": (
                plan_oligoformer_targetscan_rnaplfold_cache
            ),
            "run_oligoformer_targetscan_rnaplfold_shard": (
                run_oligoformer_targetscan_rnaplfold_shard
            ),
            "finalize_oligoformer_targetscan_rnaplfold_cache": (
                finalize_oligoformer_targetscan_rnaplfold_cache
            ),
            "run_oligoformer_efficacy": run_oligoformer_efficacy,
            "plan_oligoformer_off_target_evidence": (
                plan_oligoformer_off_target_evidence
            ),
            "prepare_oligoformer_pita_reference": prepare_oligoformer_pita_reference,
            "run_oligoformer_pita_candidate": run_oligoformer_pita_candidate,
            "run_oligoformer_targetscan_tile": run_oligoformer_targetscan_tile,
            "publish_oligoformer_off_target_evidence": (
                publish_oligoformer_off_target_evidence
            ),
            "build_oligoformer_final_tables": build_oligoformer_final_tables,
            "publish_oligoformer_outputs": publish_oligoformer_outputs,
        },
        workload_name="OligoFormer",
    )


def _optional_local_bytes(path: str | None, label: str) -> bytes | None:
    """Read one optional local CLI input."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved.read_bytes()


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_oligoformer_task(
    mrna_fasta: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    sirna_fasta: str | None = None,
    off_target: bool = False,
    toxicity: bool = False,
    all_human: bool = False,
    utr_file: str | None = None,
    orf_file: str | None = None,
    top_n: int = APP_INFO.default_top_n,
    functionality_filter: bool = True,
    pita_threshold: float = -10.0,
    targetscan_threshold: float = 1.0,
    toxicity_threshold: float = 50.0,
    off_target_nodes: int = APP_INFO.default_off_target_nodes,
    off_target_workers: int = APP_INFO.default_off_target_workers_per_node,
    off_target_process_slots: int = APP_INFO.default_off_target_process_slots,
    off_target_prep_workers: int = APP_INFO.default_off_target_prep_workers,
    pita_prepare_nodes: int = APP_INFO.default_pita_prepare_nodes,
    pita_prepare_workers: int = APP_INFO.default_pita_prepare_workers,
    pita_prepare_utr_shard_size: int = APP_INFO.default_pita_prepare_utr_shard_size,
    pita_row_shard_size: int = APP_INFO.default_pita_row_shard_size,
    pita_row_attempts: int = APP_INFO.default_pita_row_attempts,
    targetscan_rnaplfold_nodes: int = APP_INFO.default_targetscan_rnaplfold_nodes,
    targetscan_rnaplfold_workers: int = APP_INFO.default_targetscan_rnaplfold_workers,
    targetscan_rnaplfold_shard_size: int = (
        APP_INFO.default_targetscan_rnaplfold_shard_size
    ),
    targetscan_prepare_nodes: int = APP_INFO.default_targetscan_prepare_nodes,
    targetscan_ref_shard_size: int | None = None,
    targetscan_candidate_shard_size: int = (
        APP_INFO.default_targetscan_candidate_shard_size
    ),
    targetscan_context_nodes: int = APP_INFO.default_targetscan_context_nodes,
    targetscan_context_workers: int = APP_INFO.default_targetscan_context_workers,
    targetscan_context_shard_size: int = (
        APP_INFO.default_targetscan_context_shard_size
    ),
    targetscan_context_attempts: int = APP_INFO.default_targetscan_context_attempts,
    targetscan_merge_nodes: int = APP_INFO.default_targetscan_merge_nodes,
    force: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Run OligoFormer siRNA efficacy prediction.

    Args:
        mrna_fasta: Local mRNA FASTA file to scan for siRNA candidates.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the mRNA FASTA
            filename stem.
        sirna_fasta: Optional FASTA file of specific siRNAs to score instead of
            traversing the mRNA with OligoFormer's default 19 nt window.
        off_target: Enable OligoFormer off-target prediction.
        toxicity: Enable OligoFormer toxicity prediction.
        all_human: Use converted TargetScan 8.0 human ORF and UTR references
            for off-target prediction.
        utr_file: Local UTR reference file for off-target prediction.
        orf_file: Local ORF reference file for off-target prediction.
        top_n: Number of top siRNAs to use for off-target prediction. Defaults
            to 20; use -1 for all candidates.
        functionality_filter: Keep upstream functionality filtering enabled.
        pita_threshold: PITA threshold used by off-target prediction.
        targetscan_threshold: TargetScan threshold used by off-target prediction.
        toxicity_threshold: Toxicity filter threshold.
        off_target_nodes: Maximum PITA CPU containers per stage.
        off_target_workers: Maximum PITA worker processes per container.
        off_target_process_slots: Run-wide TargetScan and PITA process budget.
        off_target_prep_workers: Local workers used to prepare PITA candidates.
        pita_prepare_nodes: Maximum PITA target-discovery CPU containers.
        pita_prepare_workers: PITA target-discovery workers per container.
        pita_prepare_utr_shard_size: UTR STAB rows per PITA discovery shard.
        pita_row_shard_size: Potential-target rows per PITA scoring shard.
        pita_row_attempts: Attempts for interrupted PITA discovery and row shards.
        targetscan_rnaplfold_nodes: Maximum RNAplfold CPU containers.
        targetscan_rnaplfold_workers: RNAplfold workers per container.
        targetscan_rnaplfold_shard_size: UTR records per RNAplfold shard.
        targetscan_prepare_nodes: Maximum TargetScan preparation containers.
        targetscan_ref_shard_size: Advanced TargetScan UTR records per
            reference-preparation shard. When omitted, the UTR reference is
            distributed across targetscan_prepare_nodes shards.
        targetscan_candidate_shard_size: siRNAs per TargetScan candidate shard.
        targetscan_context_nodes: Maximum TargetScan context-score containers.
        targetscan_context_workers: Context-score workers per container.
        targetscan_context_shard_size: Target rows per context-score shard.
        targetscan_context_attempts: Attempts for TargetScan context scoring.
        targetscan_merge_nodes: Maximum TargetScan merge containers.
        force: Rebuild cached intermediates and outputs.
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
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    request = OligoformerExecutionRequest(
        run_name=run_name,
        mrna_fasta_bytes=input_path.read_bytes(),
        sirna_fasta_bytes=_optional_local_bytes(sirna_fasta, "siRNA FASTA"),
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        utr_bytes=_optional_local_bytes(utr_file, "UTR reference"),
        orf_bytes=_optional_local_bytes(orf_file, "ORF reference"),
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
        off_target_nodes=off_target_nodes,
        off_target_workers=off_target_workers,
        off_target_process_slots=off_target_process_slots,
        off_target_prep_workers=off_target_prep_workers,
        pita_prepare_nodes=pita_prepare_nodes,
        pita_prepare_workers=pita_prepare_workers,
        pita_prepare_utr_shard_size=pita_prepare_utr_shard_size,
        pita_row_shard_size=pita_row_shard_size,
        pita_row_attempts=pita_row_attempts,
        targetscan_rnaplfold_nodes=targetscan_rnaplfold_nodes,
        targetscan_rnaplfold_workers=targetscan_rnaplfold_workers,
        targetscan_rnaplfold_shard_size=targetscan_rnaplfold_shard_size,
        targetscan_prepare_nodes=targetscan_prepare_nodes,
        targetscan_ref_shard_size=targetscan_ref_shard_size,
        targetscan_candidate_shard_size=targetscan_candidate_shard_size,
        targetscan_context_nodes=targetscan_context_nodes,
        targetscan_context_workers=targetscan_context_workers,
        targetscan_context_shard_size=targetscan_context_shard_size,
        targetscan_context_attempts=targetscan_context_attempts,
        targetscan_merge_nodes=targetscan_merge_nodes,
        force=force,
        force_generation=uuid4().hex if force else None,
        app_version=CONF.repo_commit_hash or CONF.version or "unknown",
        model_version=_hash_bytes(
            orjson.dumps(
                APP_INFO.rnafm_identity_metadata,
                option=orjson.OPT_SORT_KEYS,
            )
        ),
        reference_version=(
            _hash_bytes(
                orjson.dumps(
                    APP_INFO.targetscan_ref_metadata,
                    option=orjson.OPT_SORT_KEYS,
                )
            )
            if off_target and all_human
            else None
        ),
    )

    out_file = build_local_output_path(
        resolve_local_output_dir(out_dir),
        run_name=request.run_name,
        suffix="oligoformer",
        overwrite=force,
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
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    snapshot = call.get()
    if snapshot.run.status != RunStatus.SUCCEEDED:
        diagnostic = snapshot.run.status_message or (
            snapshot.run.status_reason.value
            if snapshot.run.status_reason is not None
            else snapshot.run.status.value
        )
        raise RuntimeError(
            f"OligoFormer Execution Run ended as "
            f"{snapshot.run.status.value}: {diagnostic}"
        )
    publication = _oligoformer_result_publication_from_volume(
        CONF.output_volume,
        request.execution_plan.workload_plan_fingerprint,
    )
    if not isinstance(publication, dict) or not isinstance(
        publication.get("result_path"),
        str,
    ):
        raise RuntimeError("OligoFormer result publication is missing")
    result_path = cast(str, publication["result_path"])
    tarball_bytes = b"".join(CONF.output_volume.read_file(result_path))
    if (
        len(tarball_bytes) != publication["size_bytes"]
        or hashlib.sha256(tarball_bytes).hexdigest() != publication["sha256"]
    ):
        raise RuntimeError("OligoFormer result archive failed integrity validation")
    write_local_tarball(out_file, tarball_bytes, overwrite=force)
    print(f"🧬 OligoFormer run complete! Results saved to {out_file}")
