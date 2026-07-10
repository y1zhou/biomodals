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

Advanced tuning: `--targetscan-ref-shard-size` controls how many UTR records are
put into each TargetScan reference-preparation shard. Larger values reduce Modal
fanout and are useful for exact sharded-vs-unsharded comparisons; smaller values
increase reference-prep parallelism.

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

## Upstream equivalence

Use `scripts/verify_oligoformer_upstream_equivalence.py` when changing
off-target reducers, candidate identity handling, or cache semantics. The
verifier defaults to `--top-n -1` with `examples/data/sirna_target.fa` plus
small checked-in UTR/ORF fixtures, generates Biomodals app artifacts, then runs
direct upstream OligoFormer/PITA/TargetScan commands in a GPU Modal Sandbox and
compares canonicalized app and upstream artifacts. It is intentionally separate
from fast pytest coverage and reuses persisted Modal output-volume artifacts by
default; pass `--force` to recompute them.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import hashlib
import os
import queue as queue_lib
import shlex
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar, cast

import modal
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.constant import MAX_TIMEOUT, MODEL_VOLUME
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import package_outputs, run_command, warmup_directory
from biomodals.helper.task_budget import batches_for_total_concurrency, bounded_map
from biomodals.helper.web import download_files

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
        "targetscan-8.0-viennarna-2.7.2-semantic-topn-targetscan-polars-v1"
    )
    postprocess_cache_salt: str = "final-tables-v3"
    repo_rnafm_dir: Path = CONF.git_clone_dir / "RNA-FM"
    model_rnafm_dir: Path = Path(CONF.model_volume_mountpoint) / "RNA-FM"
    repo_ref_dir: Path = CONF.git_clone_dir / "off-target/ref"
    model_ref_dir: Path = Path(CONF.model_volume_mountpoint) / "off-target/ref"
    human_ref_filenames: tuple[str, ...] = ("human_UTR.txt", "human_ORF.txt")
    default_top_n: int = 20
    prepared_marker_name: str = "oligoformer.json"
    off_target_workers_env: str = "OLIGOFORMER_OFF_TARGET_WORKERS"
    off_target_nodes_env: str = "OLIGOFORMER_OFF_TARGET_NODES"
    off_target_prep_workers_env: str = "OLIGOFORMER_OFF_TARGET_PREP_WORKERS"
    off_target_pita_prepare_nodes_env: str = "OLIGOFORMER_PITA_PREPARE_NODES"
    off_target_pita_prepare_workers_env: str = "OLIGOFORMER_PITA_PREPARE_WORKERS"
    off_target_pita_prepare_utr_shard_size_env: str = (
        "OLIGOFORMER_PITA_PREPARE_UTR_SHARD_SIZE"
    )
    off_target_row_shard_size_env: str = "OLIGOFORMER_PITA_ROW_SHARD_SIZE"
    off_target_row_attempts_env: str = "OLIGOFORMER_PITA_ROW_ATTEMPTS"
    targetscan_rnaplfold_nodes_env: str = "OLIGOFORMER_RNAPLFOLD_NODES"
    targetscan_rnaplfold_workers_env: str = "OLIGOFORMER_RNAPLFOLD_WORKERS"
    targetscan_rnaplfold_shard_size_env: str = "OLIGOFORMER_RNAPLFOLD_SHARD_SIZE"
    targetscan_prepare_nodes_env: str = "OLIGOFORMER_TARGETSCAN_PREPARE_NODES"
    targetscan_prepare_ref_shard_size_env: str = (
        "OLIGOFORMER_TARGETSCAN_PREPARE_REF_SHARD_SIZE"
    )
    targetscan_context_nodes_env: str = "OLIGOFORMER_TARGETSCAN_CONTEXT_NODES"
    targetscan_context_workers_env: str = "OLIGOFORMER_TARGETSCAN_CONTEXT_WORKERS"
    targetscan_context_shard_size_env: str = "OLIGOFORMER_TARGETSCAN_CONTEXT_SHARD_SIZE"
    targetscan_context_attempts_env: str = "OLIGOFORMER_TARGETSCAN_CONTEXT_ATTEMPTS"
    targetscan_merge_nodes_env: str = "OLIGOFORMER_TARGETSCAN_MERGE_NODES"
    default_off_target_nodes: int = 32
    default_off_target_workers_per_node: int = 32
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
    default_targetscan_prepare_ref_shard_size: int = 1000
    default_targetscan_context_nodes: int = 100
    default_targetscan_context_workers: int = 32
    default_targetscan_context_shard_size: int = 500
    default_targetscan_context_attempts: int = 3
    targetscan_context_queue_put_batch_size: int = 256
    targetscan_context_queue_initial_batches: int = 4
    targetscan_context_queue_idle_timeout: float = 45.0
    targetscan_context_queue_sentinel: str = (
        "__biomodals_oligoformer_targetscan_context_stop__"
    )
    default_targetscan_merge_nodes: int = 16
    cache_lock_dict_name: str = f"{CONF.package_name}-cache-locks"
    cache_lock_poll_seconds: float = 5.0
    cache_lock_stale_seconds: float = MAX_TIMEOUT + 600

    @property
    def tuning_env_names(self) -> tuple[str, ...]:
        """Return local env vars that should be forwarded into Modal containers."""
        return (
            self.off_target_workers_env,
            self.off_target_nodes_env,
            self.off_target_prep_workers_env,
            self.off_target_pita_prepare_nodes_env,
            self.off_target_pita_prepare_workers_env,
            self.off_target_pita_prepare_utr_shard_size_env,
            self.off_target_row_shard_size_env,
            self.off_target_row_attempts_env,
            self.targetscan_rnaplfold_nodes_env,
            self.targetscan_rnaplfold_workers_env,
            self.targetscan_rnaplfold_shard_size_env,
            self.targetscan_prepare_nodes_env,
            self.targetscan_prepare_ref_shard_size_env,
            self.targetscan_context_nodes_env,
            self.targetscan_context_workers_env,
            self.targetscan_context_shard_size_env,
            self.targetscan_context_attempts_env,
            self.targetscan_merge_nodes_env,
        )

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
    records: tuple[OffTargetSirnaRecord, ...]
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


@dataclass(frozen=True, slots=True)
class PreparedOffTargetShard:
    """Cached per-siRNA off-target inputs ready for row-shard scoring."""

    index: int
    record_name: str
    cache_dir: str
    logs_dir: str
    pita_path: str
    row_shards: tuple[PitaRowShardSpec, ...]


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


def _fasta_record_names(fasta_bytes: bytes) -> tuple[str, ...]:
    """Return upstream-normalized FASTA record names."""
    names = []
    for raw_line in fasta_bytes.decode("utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            names.append(line[1:].replace(" ", "_@_"))
    if not names:
        raise ValueError("OligoFormer mRNA FASTA must contain at least one record")
    return tuple(names)


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
    """Return whether a marker and all paths exist."""
    if not all(path.exists() for path in paths):
        return False
    if expected_marker is None:
        return marker.exists()
    return _marker_matches(marker, expected_marker)


def _build_plan(
    cache_key: str,
    efficacy_key: str,
    output_stems: tuple[str, ...],
    run_root: str | Path | None = None,
    *,
    config: OligoformerRunConfig,
    postprocess_key: str,
    reference_identity: str | None = None,
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
                _raw_off_target_ready(layout.prep_dir / "off_target" / stem)
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
    )


APP_INFO = AppInfo()


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


def _targetscan_ref_identity() -> dict[str, object]:
    """Return the persisted content identity for full-human references."""
    import orjson

    path = APP_INFO.targetscan_ref_identity_path
    if not path.is_file():
        raise FileNotFoundError(
            "OligoFormer full-human reference identity is missing. Run "
            "download_oligoformer_models first."
        )
    try:
        identity = orjson.loads(path.read_bytes())
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


def _targetscan_ref_identity_digest() -> str:
    """Return the canonical digest pinned by full-human evidence plans."""
    import orjson

    return _hash_bytes(
        orjson.dumps(_targetscan_ref_identity(), option=orjson.OPT_SORT_KEYS)
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


def _targetscan_rnaplfold_cache_ready() -> bool:
    """Return whether the TargetScan RNAplfold cache is ready for human refs."""
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
    if not all(marker.get(key) == value for key, value in expected.items()):
        return False
    expected_count = marker.get("record_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        return False
    sample_records = marker.get("sample_records")
    if not isinstance(sample_records, list) or not sample_records:
        return False
    return all(
        (cache_dir / f"{name}.{APP_INFO.targetscan_species_id}_lunp").is_file()
        for name in sample_records
        if isinstance(name, str)
    )


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
    .env({
        name: value
        for name in APP_INFO.tuning_env_names
        if (value := os.environ.get(name))
    })
    .pipe(
        patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3", "modal"]
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
def _download_oligoformer_models_locked(force: bool) -> None:
    """Download model assets while holding the global reference-state lock."""
    import shutil

    import orjson

    CONF.output_volume.reload()
    refs_ready = _targetscan_human_refs_ready()
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

    if "/" in name or "\0" in name:
        raise ValueError(f"Unsafe TargetScan transcript identifier: {name!r}")
    species = APP_INFO.targetscan_species_id
    output_path = output_dir / f"{name}.{species}_lunp"
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
    cpu=APP_INFO.default_targetscan_rnaplfold_workers,
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_targetscan_rnaplfold_shard(
    spec: TargetscanRnaPlfoldShardSpec,
) -> int:
    """Populate one shard of cached TargetScan RNAplfold outputs."""
    CONF.output_volume.reload()
    records = _read_fasta_pairs(Path(spec.shard_path))
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(spec.log_path)
    local_workers = _positive_int_from_env(
        APP_INFO.targetscan_rnaplfold_workers_env,
        APP_INFO.default_targetscan_rnaplfold_workers,
    )
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
    CONF.output_volume.commit()
    print(
        "💊 OligoFormer TargetScan RNAplfold shard "
        f"{spec.shard_index} created {created}/{len(records)} outputs"
    )
    return created


def _build_targetscan_rnaplfold_cache(force: bool) -> None:
    """Build the TargetScan RNAplfold cache while holding its build lock."""
    import shutil

    import orjson

    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    _ensure_human_refs()
    if _targetscan_rnaplfold_cache_ready() and not force:
        print("💊 OligoFormer TargetScan RNAplfold cache already available")
        return

    cache_dir = APP_INFO.targetscan_rnaplfold_cache_dir
    shard_dir = APP_INFO.targetscan_rnaplfold_shard_dir
    marker_path = APP_INFO.targetscan_rnaplfold_marker_path
    if force or marker_path.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(shard_dir, ignore_errors=True)
        marker_path.unlink(missing_ok=True)

    records = _read_fasta_pairs(APP_INFO.model_ref_dir / "human_UTR.txt")
    if not records:
        raise RuntimeError("TargetScan human UTR refs contain no records")

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = _positive_int_from_env(
        APP_INFO.targetscan_rnaplfold_shard_size_env,
        APP_INFO.default_targetscan_rnaplfold_shard_size,
    )
    shard_specs = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[start : start + shard_size]
        shard_path = shard_dir / f"{shard_index:05d}.fa"
        with shard_path.open("w", encoding="utf-8") as handle:
            for name, sequence in shard_records:
                handle.write(f">{name}\n{sequence}\n")
        shard_specs.append(
            TargetscanRnaPlfoldShardSpec(
                shard_index=shard_index,
                shard_path=str(shard_path),
                output_dir=str(cache_dir),
                log_path=str(shard_dir / "logs" / f"{shard_index:05d}.log"),
            )
        )

    CONF.output_volume.commit()
    node_count = _bounded_node_count(
        len(shard_specs),
        env_name=APP_INFO.targetscan_rnaplfold_nodes_env,
        default=APP_INFO.default_targetscan_rnaplfold_nodes,
    )
    local_workers = _positive_int_from_env(
        APP_INFO.targetscan_rnaplfold_workers_env,
        APP_INFO.default_targetscan_rnaplfold_workers,
    )
    print(
        "💊 Preparing OligoFormer TargetScan RNAplfold cache for "
        f"{len(records)} UTRs across {len(shard_specs)} shards on up to "
        f"{node_count} CPU nodes with {local_workers} workers each"
    )
    bounded_map(
        shard_specs,
        lambda spec: run_oligoformer_targetscan_rnaplfold_shard.remote(spec),
        max_parallel=node_count,
    )

    CONF.output_volume.reload()
    missing = [
        name
        for name, _ in records
        if not (cache_dir / f"{name}.{APP_INFO.targetscan_species_id}_lunp").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "OligoFormer TargetScan RNAplfold cache is incomplete; missing "
            + ", ".join(missing[:10])
        )
    sample_indexes = sorted({0, len(records) // 2, len(records) - 1})
    marker_path.write_bytes(
        orjson.dumps(
            _targetscan_rnaplfold_expected_metadata()
            | {
                "record_count": len(records),
                "shard_count": len(shard_specs),
                "sample_records": [records[index][0] for index in sample_indexes],
            }
        )
    )
    CONF.output_volume.commit()
    print(f"💊 OligoFormer TargetScan RNAplfold cache committed: {len(records)} UTRs")


@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def prepare_oligoformer_targetscan_rnaplfold_cache(force: bool = False) -> None:
    """Build or reuse cached TargetScan RNAplfold outputs for all-human refs."""
    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    _ensure_human_refs()
    if not _targetscan_ref_identity_matches_model():
        raise FileNotFoundError(
            "OligoFormer human reference identity does not match the model volume. "
            "Run download_oligoformer_models first."
        )
    if _targetscan_rnaplfold_cache_ready() and not force:
        print("💊 OligoFormer TargetScan RNAplfold cache already available")
        return
    with _cache_build_lock(
        "targetscan-reference-state",
        "global",
        rebuild=True,
    ) as owns_cache_build:
        CONF.output_volume.reload()
        MODEL_VOLUME.reload()
        _ensure_human_refs()
        if not _targetscan_ref_identity_matches_model():
            raise FileNotFoundError(
                "OligoFormer human reference identity changed during RNAplfold "
                "setup. Run download_oligoformer_models first."
            )
        if _targetscan_rnaplfold_cache_ready() and (not force or not owns_cache_build):
            print("💊 OligoFormer TargetScan RNAplfold cache already available")
            return
        if not owns_cache_build:
            raise RuntimeError(
                "OligoFormer RNAplfold cache was marked complete without outputs"
            )
        _build_targetscan_rnaplfold_cache(force)


##########################################
# Inference functions
##########################################
def _ensure_rnafm_runtime() -> None:
    """Copy RNA-FM weights from the model volume into the writable repo path."""
    import shutil

    if APP_INFO.repo_rnafm_dir.is_symlink():
        APP_INFO.repo_rnafm_dir.unlink()
    elif (
        APP_INFO.repo_rnafm_dir.exists()
        and not APP_INFO.repo_rnafm_redevelop_dir.is_dir()
    ):
        shutil.rmtree(APP_INFO.repo_rnafm_dir)
    if not APP_INFO.repo_rnafm_redevelop_dir.is_dir():
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
    """Write a small cache marker after a stage completes."""
    import orjson

    layout.markers_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_key": plan.cache_key,
        "postprocess_key": plan.postprocess_key,
        "output_stems": list(plan.output_stems),
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    _marker_path(layout, marker).write_bytes(orjson.dumps(metadata))


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


def _positive_int_from_env(env_name: str, default: int) -> int:
    """Return a positive integer environment override."""
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{env_name} must be positive")
    return value


def _bounded_node_count(task_count: int, *, env_name: str, default: int) -> int:
    """Return env-configured Modal node count capped to the task count."""
    if task_count < 1:
        return 1
    nodes = _positive_int_from_env(env_name, default)
    return max(1, min(nodes, task_count))


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


def _targetscan_batch_specs(
    *,
    run_root: str,
    output_dir: Path,
    stem: str,
    records: list[OffTargetSirnaRecord],
    utr_path: str,
    orf_path: str,
    ref_shard_size: int | None = None,
) -> list[TargetscanBatchSpec]:
    """Build transcript-aligned TargetScan reference-shard specs."""
    if not records:
        return []

    layout = AppRunLayout.from_run_root(run_root)
    shard_root = layout.prep_dir / "off_target" / stem / "targetscan_ref_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    utr_records = _read_fasta_pairs(Path(utr_path))
    orf_records_by_name = dict(_read_fasta_pairs(Path(orf_path)))
    shard_size = ref_shard_size or _positive_int_from_env(
        APP_INFO.targetscan_prepare_ref_shard_size_env,
        APP_INFO.default_targetscan_prepare_ref_shard_size,
    )
    if shard_size < 1:
        raise ValueError("targetscan_ref_shard_size must be a positive integer")
    rnaplfold_cache_dir = ""
    if _is_model_human_ref_pair(utr_path, orf_path):
        if not _targetscan_rnaplfold_cache_ready():
            raise FileNotFoundError(
                "OligoFormer TargetScan RNAplfold cache is missing. Run "
                "prepare_oligoformer_targetscan_rnaplfold_cache first."
            )
        rnaplfold_cache_dir = str(APP_INFO.targetscan_rnaplfold_cache_dir)

    specs = []
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
        marker_text = (
            f"ref_shard_size={shard_size}\n"
            f"start={start}\n"
            f"utr_records={len(shard_utr_records)}\n"
            f"orf_records={len(shard_orf_records)}\n"
        )
        existing_marker = (
            marker_path.read_text(encoding="utf-8") if marker_path.exists() else None
        )
        if existing_marker != marker_text:
            _write_fasta_pairs(shard_utr_records, shard_utr_path)
            _write_fasta_pairs(shard_orf_records, shard_orf_path)
            marker_path.write_text(marker_text, encoding="utf-8")
        specs.append(
            TargetscanBatchSpec(
                run_root=run_root,
                output_dir=str(output_dir),
                stem=stem,
                ref_shard_size=shard_size,
                shard_index=shard_index,
                records=tuple(records),
                utr_path=str(shard_utr_path),
                orf_path=str(shard_orf_path),
                rnaplfold_cache_dir=rnaplfold_cache_dir,
            )
        )
    return specs


def _targetscan_batch_cache_dir(spec: TargetscanBatchSpec) -> Path:
    """Return the shared-cache directory for one TargetScan batch."""
    return (
        AppRunLayout.from_run_root(spec.run_root).prep_dir
        / "off_target"
        / spec.stem
        / "targetscan"
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
    rows = utr_stab_path.read_text(encoding="utf-8").splitlines()
    if not rows:
        return ()

    shard_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for shard_index, start_row in enumerate(range(0, len(rows), shard_size)):
        end_row = min(start_row + shard_size, len(rows))
        shard_name = f"{shard_index:05d}_{start_row:012d}_{end_row:012d}"
        input_path = shard_dir / f"{shard_name}.utr.stab"
        output_path = shard_dir / f"{shard_name}.potential.tsv"
        input_path.write_text(
            "\n".join(rows[start_row:end_row]) + "\n",
            encoding="utf-8",
        )
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
    return tuple(specs)


def _run_pita_prepare_utr_shard(spec: PitaPrepareUtrShardSpec) -> str:
    """Run one local UTR STAB shard through PITA potential-target discovery."""
    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if marker_path.exists() and output_path.exists():
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    tmp_output_path.unlink(missing_ok=True)
    pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
    cmd = (
        "set -euo pipefail; "
        f"perl {shlex.quote(str(pita_lib / 'find_potential_mirna_targets.pl'))} "
        f"{shlex.quote(spec.input_path)} -f {shlex.quote(spec.mir_stab_path)} "
        f"> {shlex.quote(str(tmp_output_path))}"
    )
    run_command(
        ["bash", "-lc", cmd],
        cwd=pita_lib.parent,
        output_mode="log",
        log_file=spec.log_path,
        show_command=False,
    )
    tmp_output_path.replace(output_path)
    marker_path.write_text("done", encoding="utf-8")
    return str(output_path)


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_pita_prepare_utr_shard_batch(
    specs: list[PitaPrepareUtrShardSpec],
    local_workers: int,
) -> list[str]:
    """Run cached PITA UTR target-discovery shards on one CPU node."""
    CONF.output_volume.reload()
    print(
        "💊 Running OligoFormer PITA target-discovery batch with "
        f"{len(specs)} shards using {local_workers} workers; logs under "
        f"{Path(specs[0].log_path).parent if specs else 'n/a'}"
    )
    outputs = bounded_map(
        specs,
        _run_pita_prepare_utr_shard,
        max_parallel=local_workers,
    )
    CONF.output_volume.commit()
    return outputs


def _write_pita_potential_targets_from_outputs(
    *,
    outputs: list[str],
    potential_targets_path: Path,
) -> int:
    """Concatenate PITA target-discovery shard outputs in UTR order."""
    potential_targets_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_potential_targets_path = potential_targets_path.with_name(
        f".{potential_targets_path.name}.tmp.{os.getpid()}"
    )
    tmp_potential_targets_path.unlink(missing_ok=True)
    row_count = 0
    with tmp_potential_targets_path.open("w", encoding="utf-8") as out:
        for output in outputs:
            data = Path(output).read_text(encoding="utf-8")
            if not data:
                continue
            out.write(data)
            if not data.endswith("\n"):
                out.write("\n")
            row_count += len(data.splitlines())
    tmp_potential_targets_path.replace(potential_targets_path)
    return row_count


def _run_pita_prepare_utr_shard_batches(
    specs: list[PitaPrepareUtrShardSpec],
) -> list[str]:
    """Run PITA target-discovery shard batches with bounded global fanout."""
    if not specs:
        return []

    local_workers = _positive_int_from_env(
        APP_INFO.off_target_pita_prepare_workers_env,
        APP_INFO.default_pita_prepare_workers,
    )
    batches = _batch_items_for_local_workers(
        specs,
        max_nodes=_bounded_node_count(
            len(specs),
            env_name=APP_INFO.off_target_pita_prepare_nodes_env,
            default=APP_INFO.default_pita_prepare_nodes,
        ),
        local_workers=local_workers,
    )
    print(
        "💊 Running OligoFormer PITA target discovery for "
        f"{len(specs)} UTR shards on {len(batches)} CPU nodes with "
        f"{local_workers} workers each"
    )
    output_batches = bounded_map(
        batches,
        lambda batch: run_oligoformer_pita_prepare_utr_shard_batch.remote(
            list(batch),
            local_workers=local_workers,
        ),
        max_parallel=len(batches),
    )
    return [output for batch in output_batches for output in batch]


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


def _ensure_pita_row_inputs(
    potential_targets_path: Path,
    row_shards: tuple[PitaRowShardSpec, ...],
) -> None:
    """Write missing per-row-shard potential-target inputs."""
    missing = [row for row in row_shards if not Path(row.input_path).exists()]
    if not missing:
        return

    potential_rows = potential_targets_path.read_text(encoding="utf-8").splitlines()
    for row in missing:
        input_path = Path(row.input_path)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        shard_rows = potential_rows[row.start_row : row.end_row]
        input_path.write_text("\n".join(shard_rows) + "\n", encoding="utf-8")


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
        ext_utr_path=cache_dir
        / f"{plan.spec.stem}_shard_{plan.spec.index:05d}_ext_utr.stab",
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


def _prepare_pita_target_discovery_plan(
    spec: OffTargetShardSpec,
    shard_root: Path,
) -> PitaPreparePlan:
    """Prepare local PITA stage0 files and return target-discovery shards."""
    cache_dir = _off_target_shard_cache_dir(spec)
    logs_dir = _off_target_shard_logs_dir(spec)
    potential_targets_path = cache_dir / "potential_targets.tsv"
    ext_utr_path = cache_dir / f"{spec.stem}_shard_{spec.index:05d}_ext_utr.stab"
    marker_path = cache_dir / "pita_prepare.done"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if (
        marker_path.exists()
        and potential_targets_path.exists()
        and ext_utr_path.exists()
    ):
        return PitaPreparePlan(
            spec=spec,
            utr_shards=(),
            row_count=int(marker_path.read_text(encoding="utf-8")),
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
    cached_mir_stab_path = cache_dir / f"{spec.stem}_shard_{spec.index:05d}_mir.stab"
    cached_mir_stab_path.write_bytes(mir_stab_path.read_bytes())
    utr_shards = _pita_prepare_utr_shard_specs(
        utr_stab_path=utr_stab_path,
        mir_stab_path=cached_mir_stab_path,
        shard_dir=cache_dir / "pita_prepare_utr_shards",
        logs_dir=logs_dir,
        shard_size=_positive_int_from_env(
            APP_INFO.off_target_pita_prepare_utr_shard_size_env,
            APP_INFO.default_pita_prepare_utr_shard_size,
        ),
    )
    return PitaPreparePlan(
        spec=spec,
        utr_shards=utr_shards,
        row_count=None,
    )


def _finalize_pita_target_discovery_plan(
    plan: PitaPreparePlan,
    outputs_by_path: dict[str, str],
) -> PreparedOffTargetShard:
    """Merge PITA target-discovery shards and return row-score specs."""
    if plan.row_count is not None:
        return _pita_prepared_shard_from_plan(plan, row_count=plan.row_count)

    outputs = [
        outputs_by_path[shard.output_path]
        for shard in sorted(plan.utr_shards, key=lambda item: item.shard_index)
    ]
    cache_dir = _off_target_shard_cache_dir(plan.spec)
    row_count = _write_pita_potential_targets_from_outputs(
        outputs=outputs,
        potential_targets_path=cache_dir / "potential_targets.tsv",
    )
    cache_dir.joinpath("pita_prepare.done").write_text(str(row_count), encoding="utf-8")
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


def _run_targetscan_context_shard(spec: TargetscanContextShardSpec) -> str:
    """Run one TargetScan context-score shard on a CPU node."""
    import shutil

    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if marker_path.exists() and output_path.exists():
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
        run_command(
            [
                "perl",
                "targetscan_70_context_scores.pl",
                "sirnas_for_context_scores.txt",
                "UTR.txt",
                "predicted_targets.txt",
                "ORF.length.txt",
                "ORF_8mer_counts.txt",
                str(tmp_output_path),
            ],
            cwd=workdir,
            output_mode="log",
            log_file=spec.log_path,
            show_command=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tmp_output_path.read_bytes())

    marker_path.write_text("done", encoding="utf-8")
    return str(output_path)


def _targetscan_context_shard_ready(spec: TargetscanContextShardSpec) -> bool:
    """Return whether one context-score shard output is ready."""
    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    return marker_path.exists() and output_path.exists()


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_targetscan_context_queue_worker(
    queue_name: str,
    local_workers: int,
) -> int:
    """Drain queued TargetScan context-score shards on one CPU node."""
    CONF.output_volume.reload()

    queue = modal.Queue.from_name(queue_name)
    idle_timeout = APP_INFO.targetscan_context_queue_idle_timeout

    def _worker(_worker_index: int) -> int:
        output_count = 0
        while True:
            try:
                spec = queue.get(timeout=idle_timeout)
            except queue_lib.Empty:
                return output_count
            if spec == APP_INFO.targetscan_context_queue_sentinel:
                return output_count
            _run_targetscan_context_shard(cast(TargetscanContextShardSpec, spec))
            output_count += 1

    try:
        return sum(
            bounded_map(range(local_workers), _worker, max_parallel=local_workers)
        )
    finally:
        CONF.output_volume.commit()


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


def _run_targetscan_context_batches(
    context_shards: list[TargetscanContextShardSpec],
) -> list[str]:
    """Run TargetScan context-score shards with queue-based work stealing."""
    if not context_shards:
        return []

    from uuid import uuid4

    context_workers = _positive_int_from_env(
        APP_INFO.targetscan_context_workers_env,
        APP_INFO.default_targetscan_context_workers,
    )
    expected_outputs = [shard.output_path for shard in context_shards]
    queue_key = hash_string("\n".join(expected_outputs))
    put_batch_size = APP_INFO.targetscan_context_queue_put_batch_size
    max_attempts = _positive_int_from_env(
        APP_INFO.targetscan_context_attempts_env,
        APP_INFO.default_targetscan_context_attempts,
    )
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        CONF.output_volume.reload()
        pending_shards = [
            shard
            for shard in context_shards
            if not _targetscan_context_shard_ready(shard)
        ]
        if not pending_shards:
            return expected_outputs

        local_workers = min(context_workers, len(pending_shards))
        queue_sized_node_count = (
            len(pending_shards) + local_workers - 1
        ) // local_workers
        context_node_count = _bounded_node_count(
            queue_sized_node_count,
            env_name=APP_INFO.targetscan_context_nodes_env,
            default=APP_INFO.default_targetscan_context_nodes,
        )
        queue_name = (
            f"{CONF.package_name}-targetscan-context-{queue_key[:16]}-"
            f"{attempt}-{uuid4().hex[:12]}"
        )
        queue = modal.Queue.from_name(queue_name, create_if_missing=True)
        print(
            "💊 Running OligoFormer TargetScan context scoring for "
            f"{len(pending_shards)} pending of {len(context_shards)} shards via "
            f"{context_node_count} queue workers with {local_workers} local workers "
            f"each (attempt {attempt}/{max_attempts})"
        )
        worker_calls = []
        print(
            "💊 Queueing OligoFormer TargetScan context scoring for "
            f"{len(pending_shards)} pending of {len(context_shards)} shards"
        )
        try:
            initial_count = min(
                len(pending_shards),
                put_batch_size * APP_INFO.targetscan_context_queue_initial_batches,
            )
            for start in range(0, initial_count, put_batch_size):
                queue.put_many(pending_shards[start : start + put_batch_size])
            worker_calls = bounded_map(
                range(context_node_count),
                lambda _index, q=queue_name, w=local_workers: (
                    run_oligoformer_targetscan_context_queue_worker.spawn(
                        q,
                        local_workers=w,
                    )
                ),
                max_parallel=context_node_count,
            )
            for start in range(initial_count, len(pending_shards), put_batch_size):
                queue.put_many(pending_shards[start : start + put_batch_size])
            stop_tokens = [APP_INFO.targetscan_context_queue_sentinel] * (
                context_node_count * local_workers
            )
            for start in range(0, len(stop_tokens), put_batch_size):
                queue.put_many(stop_tokens[start : start + put_batch_size])
            for call in worker_calls:
                call.get()
        except Exception as exc:
            last_error = exc
            for call in worker_calls:
                with suppress(Exception):
                    call.cancel(terminate_containers=True)
        except BaseException:
            for call in worker_calls:
                with suppress(Exception):
                    call.cancel(terminate_containers=True)
            raise
        finally:
            with suppress(Exception):
                modal.Queue.objects.delete(queue_name)

        CONF.output_volume.reload()
        missing_outputs = [
            shard.output_path
            for shard in context_shards
            if not _targetscan_context_shard_ready(shard)
        ]
        if not missing_outputs:
            return expected_outputs
        if attempt < max_attempts:
            reason = f" after {type(last_error).__name__}" if last_error else ""
            print(
                "💊 Retrying OligoFormer TargetScan context scoring for "
                f"{len(missing_outputs)} missing shards{reason}"
            )

    CONF.output_volume.reload()
    missing_outputs = [
        shard.output_path
        for shard in context_shards
        if not _targetscan_context_shard_ready(shard)
    ]
    if missing_outputs:
        preview = ", ".join(missing_outputs[:5])
        if len(missing_outputs) > 5:
            preview += f", ... ({len(missing_outputs)} total)"
        error = RuntimeError(
            "OligoFormer TargetScan context scoring did not produce all outputs: "
            f"{preview}"
        )
        if last_error is not None:
            raise error from last_error
        raise error
    return expected_outputs


def _prepare_targetscan_batch_context_plan(
    spec: TargetscanBatchSpec,
    batch_root: Path,
) -> PreparedTargetscanBatch:
    """Prepare TargetScan context-score shard specs for a siRNA batch."""
    if not spec.records:
        raise RuntimeError("TargetScan batch requires at least one siRNA record")

    cache_dir = _targetscan_batch_cache_dir(spec)
    targetscan_path = cache_dir / "targetscan.tab"
    marker_path = cache_dir / "targetscan.done"
    logs_dir = _targetscan_batch_logs_dir(spec)
    if marker_path.exists() and targetscan_path.exists():
        return PreparedTargetscanBatch(
            targetscan_path=str(targetscan_path),
            logs_dir=str(logs_dir),
            context_shards=(),
            needs_merge=False,
        )

    sirna_file = (
        batch_root
        / "data/infer"
        / f"{spec.stem}_targetscan_{spec.shard_index:05d}"
        / "siRNA.fa"
    )
    if not sirna_file.exists():
        _write_sirna_records(list(spec.records), sirna_file)
    rnaplfold_cache_dir = spec.rnaplfold_cache_dir

    targetscan_cmd = r"""
set -eu
shopt -s nullglob
mir=$1
utr=$2
orf=$3
stem=$4
context_dir=$5
context_shard_size=${6:-500}
rm -rf ./off-target/tmp
cp -r ./off-target/targetscan ./off-target/tmp
cp "$mir" "$utr" "$orf" ./off-target/tmp/
cd ./off-target/tmp
mir=$(basename "$mir")
utr=$(basename "$utr")
orf=$(basename "$orf")
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",x,$1;}' "$mir" | sed 's/>//g' > sirnas_for_context_scores.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,substr($1,2,7),"9606";}' "$mir" | sed 's/>//g' > sirnas.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",$1;}' "$utr" | sed 's/>//g' > UTR.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",$1;}' "$orf" | sed 's/>//g' > ORF.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",length($1);}' "$orf" | sed 's/>//g' > ORF.length.txt
perl targetscan_70.pl sirnas.txt UTR.txt targetscan_70_output.txt
perl targetscan_70_BL_bins.pl UTR.txt > UTRs_median_BLs_bins.txt
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
    context_shard_size = _positive_int_from_env(
        APP_INFO.targetscan_context_shard_size_env,
        APP_INFO.default_targetscan_context_shard_size,
    )
    context_dir = cache_dir / "targetscan_context"
    prep_marker_path = context_dir / "targetscan_prepare.done"
    print(
        "💊 Preparing OligoFormer TargetScan batch "
        f"{spec.stem}:{spec.shard_index} for {len(spec.records)} siRNAs with "
        f"{context_shard_size} target rows per context shard; "
        f"log: {logs_dir / 'targetscan_prep.log'}"
    )
    if not prep_marker_path.exists():
        run_command(
            [
                "bash",
                "-lc",
                targetscan_cmd,
                "run_targetscan_cached",
                str(sirna_file),
                spec.utr_path,
                spec.orf_path,
                f"{spec.stem}_targetscan_{spec.shard_index:05d}",
                str(context_dir),
                str(context_shard_size),
            ],
            cwd=batch_root,
            output_mode="log",
            log_file=logs_dir / "targetscan_prep.log",
            show_command=False,
        )
        prep_marker_path.write_text("done", encoding="utf-8")

    context_shards = _targetscan_context_shard_specs(
        context_dir=context_dir,
        logs_dir=logs_dir,
        rnaplfold_cache_dir=rnaplfold_cache_dir,
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
        targetscan_path.parent.joinpath("targetscan.done").write_text(
            "done",
            encoding="utf-8",
        )
    return plan.targetscan_path


@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_oligoformer_targetscan_batch_context_plan(
    plan: PreparedTargetscanBatch,
    context_outputs: list[str],
) -> str:
    """Finalize one TargetScan reference batch from context-score outputs."""
    CONF.output_volume.reload()
    targetscan_path = _finalize_targetscan_batch_context_plan(plan, context_outputs)
    CONF.output_volume.commit()
    return targetscan_path


@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def prepare_oligoformer_targetscan_batch_shard(
    spec: TargetscanBatchSpec,
) -> PreparedTargetscanBatch:
    """Prepare one TargetScan candidate-batch/reference-shard tile."""
    import shutil

    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    with TemporaryDirectory(
        prefix=f"oligoformer_targetscan_{spec.stem}_{spec.shard_index}_"
    ) as tmpdir:
        batch_root = Path(tmpdir)
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
    CONF.output_volume.commit()
    return plan


def _run_targetscan_prepare_batches(
    specs: list[TargetscanBatchSpec],
) -> list[PreparedTargetscanBatch]:
    """Prepare TargetScan reference shards with bounded remote fanout."""
    if not specs:
        return []

    node_count = _bounded_node_count(
        len(specs),
        env_name=APP_INFO.targetscan_prepare_nodes_env,
        default=APP_INFO.default_targetscan_prepare_nodes,
    )
    print(
        "💊 Preparing OligoFormer TargetScan for "
        f"{len(specs)} reference shards on up to {node_count} CPU nodes"
    )
    return bounded_map(
        specs,
        lambda spec: prepare_oligoformer_targetscan_batch_shard.remote(spec),
        max_parallel=node_count,
    )


def _merge_targetscan_batch_outputs(
    *,
    targetscan_paths: list[str],
    output_path: Path,
) -> None:
    """Merge TargetScan reference-shard outputs in upstream output order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for targetscan_path in targetscan_paths:
        rows.extend(
            line
            for line in Path(targetscan_path).read_text(encoding="utf-8").splitlines()
            if line and line != TARGETSCAN_HEADER
        )
    rows.sort(key=lambda row: tuple(row.split("\t")[:2]))
    with output_path.open("w", encoding="utf-8") as out:
        if rows:
            out.write("\n".join(rows))
            out.write("\n")
        else:
            out.write(TARGETSCAN_HEADER + "\n")


def _read_targetscan_table(path: Path) -> pl.DataFrame:
    """Read headerless or header-only TargetScan raw output."""
    schema = {
        "refseq": pl.String,
        "siRNA": pl.String,
        "targetscan_score": pl.Float64,
    }
    if path.stat().st_size == 0:
        return pl.DataFrame(schema=schema)
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline().rstrip("\n")
    if first_line == TARGETSCAN_HEADER:
        return pl.read_csv(path, separator="\t", schema_overrides=schema)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(TARGETSCAN_COLUMNS),
        schema_overrides={"targetscan_score": pl.Float64},
    )


def _raw_off_target_ready(raw_off_target_dir: Path) -> bool:
    """Return whether merged off-target evidence is ready for reuse."""
    return all(
        (raw_off_target_dir / name).exists()
        for name in ("off_target.done", "pita.tab", "targetscan.tab")
    )


@contextmanager
def _cache_build_lock(stage: str, identity: str, *, rebuild: bool = False):
    """Elect one cache builder using append-only lock generations."""
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

    keep_names = {"off_target.done", "pita.tab", "targetscan.tab"}
    for child in raw_off_target_dir.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _run_pita_row_shard(spec: PitaRowShardSpec) -> str:
    """Run or reuse one cached PITA row-shard score table."""
    import subprocess as sp
    from time import sleep

    output_path = Path(spec.output_path)
    marker_path = output_path.with_suffix(output_path.suffix + ".done")
    if marker_path.exists() and output_path.exists():
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f"oligoformer_pita_rows_{spec.sirna_index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        input_path = workdir / "potential_targets.tsv"
        input_path.write_bytes(Path(spec.input_path).read_bytes())
        pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
        attempts = _positive_int_from_env(
            APP_INFO.off_target_row_attempts_env,
            APP_INFO.default_pita_row_attempts,
        )
        transient_return_codes = {-2, -15}
        for attempt in range(1, attempts + 1):
            tmp_output_path = output_path.with_name(
                f".{output_path.name}.tmp.{os.getpid()}.{attempt}"
            )
            tmp_output_path.unlink(missing_ok=True)
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
    marker_path.write_text("done", encoding="utf-8")
    return str(output_path)


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_pita_row_shard_batch(
    row_shards: list[PitaRowShardSpec],
    local_workers: int,
) -> list[str]:
    """Run cached PITA row-shard scoring on one CPU node."""
    CONF.output_volume.reload()
    outputs = bounded_map(
        row_shards,
        _run_pita_row_shard,
        max_parallel=local_workers,
    )
    CONF.output_volume.commit()
    return outputs


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
                data = row_output.read_text(encoding="utf-8")
                out.write(data)
                if data and not data.endswith("\n"):
                    out.write("\n")

    with TemporaryDirectory(
        prefix=f"oligoformer_pita_finalize_{prepared.index}_"
    ) as tmpdir:
        workdir = Path(tmpdir)
        pita_lib = CONF.git_clone_dir / "off-target/pita/lib"
        local_results = workdir / raw_results_path.name
        local_results.write_bytes(raw_results_path.read_bytes())
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
        Path(prepared.pita_path).write_bytes(targets_path.read_bytes())
    Path(prepared.cache_dir, "pita_finalize.done").write_text("done", encoding="utf-8")


def _finalize_oligoformer_pita_shard(
    prepared: PreparedOffTargetShard,
) -> OffTargetShardResult:
    """Finalize one per-siRNA PITA table from cached row shards."""
    pita_path = Path(prepared.pita_path)
    marker_path = Path(prepared.cache_dir) / "pita_finalize.done"
    if not marker_path.exists() or not pita_path.exists():
        row_outputs = [
            Path(row.output_path)
            for row in sorted(prepared.row_shards, key=lambda item: item.shard_index)
        ]
        _write_pita_targets_from_scored_rows(prepared, row_outputs)
    return OffTargetShardResult(
        index=prepared.index,
        pita_path=prepared.pita_path,
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_oligoformer_pita_shard_batch(
    prepared_shards: list[PreparedOffTargetShard],
    local_workers: int,
) -> list[OffTargetShardResult]:
    """Finalize cached per-siRNA PITA tables on one CPU node."""
    CONF.output_volume.reload()
    print(
        "💊 Running OligoFormer PITA finalize batch with "
        f"{len(prepared_shards)} siRNAs using {local_workers} workers"
    )
    results = bounded_map(
        prepared_shards,
        _finalize_oligoformer_pita_shard,
        max_parallel=local_workers,
    )
    CONF.output_volume.commit()
    return results


def _merge_pita_shards(shard_results: list[OffTargetShardResult], output_path: Path):
    """Merge per-siRNA PITA tables in upstream score-sorted order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    rows = []
    with output_path.open("w", encoding="utf-8") as out:
        for shard in sorted(shard_results, key=lambda result: result.index):
            lines = Path(shard.pita_path).read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
            rows.extend(line for line in lines[1:] if line)
        if header is not None:
            out.write(header)
            out.write("\n")
        score_index = header.split("\t").index("Score") if header is not None else 3
        rows.sort(key=lambda row: (float(row.split("\t")[score_index]), row))
        if rows:
            out.write("\n".join(rows))
            out.write("\n")


def _prepare_pita_target_discovery_plan_for_spec(
    *,
    spec: OffTargetShardSpec,
    prepare_root: Path,
) -> PitaPreparePlan:
    """Prepare one siRNA's PITA target-discovery plan."""
    import shutil

    shard_root = prepare_root / f"{spec.stem}_shard_{spec.index:05d}"
    off_target_root = shard_root / "off-target"
    off_target_root.mkdir(parents=True)
    shutil.copytree(CONF.git_clone_dir / "off-target/pita", off_target_root / "pita")
    return _prepare_pita_target_discovery_plan(spec, shard_root)


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_targetscan_branch(
    targetscan_specs: list[TargetscanBatchSpec],
) -> list[str]:
    """Run TargetScan off-target scoring for all selected siRNAs."""
    if not targetscan_specs:
        return []

    CONF.output_volume.reload()
    targetscan_plans = _run_targetscan_prepare_batches(targetscan_specs)
    context_shards = [
        shard
        for plan in targetscan_plans
        for shard in sorted(plan.context_shards, key=lambda item: item.shard_index)
    ]
    context_outputs = _run_targetscan_context_batches(context_shards)
    CONF.output_volume.reload()
    context_outputs_by_path = {
        shard.output_path: output
        for shard, output in zip(context_shards, context_outputs, strict=True)
    }
    merge_inputs = [
        (
            plan,
            [
                context_outputs_by_path[shard.output_path]
                for shard in sorted(
                    plan.context_shards, key=lambda item: item.shard_index
                )
            ],
        )
        for plan in targetscan_plans
    ]
    merge_node_count = _bounded_node_count(
        len(merge_inputs),
        env_name=APP_INFO.targetscan_merge_nodes_env,
        default=APP_INFO.default_targetscan_merge_nodes,
    )
    print(
        "💊 Merging OligoFormer TargetScan context outputs for "
        f"{len(merge_inputs)} reference batches on up to {merge_node_count} CPU nodes"
    )
    targetscan_paths = bounded_map(
        merge_inputs,
        lambda item: finalize_oligoformer_targetscan_batch_context_plan.remote(
            item[0],
            item[1],
        ),
        max_parallel=merge_node_count,
    )
    CONF.output_volume.commit()
    return targetscan_paths


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def run_oligoformer_pita_branch(
    shard_specs: list[OffTargetShardSpec],
    node_count: int,
    local_workers: int,
) -> list[OffTargetShardResult]:
    """Run PITA off-target scoring for all selected siRNAs."""
    if not shard_specs:
        return []

    CONF.output_volume.reload()
    prep_workers = min(
        _positive_int_from_env(
            APP_INFO.off_target_prep_workers_env,
            APP_INFO.default_off_target_prep_workers,
        ),
        len(shard_specs),
    )
    print(
        "💊 Preparing OligoFormer PITA inputs for "
        f"{len(shard_specs)} siRNAs with {prep_workers} local workers"
    )
    with TemporaryDirectory(
        prefix=f"oligoformer_{shard_specs[0].stem}_off_target_prepare_"
    ) as tmpdir:
        prepare_root = Path(tmpdir)
        pita_plans = bounded_map(
            shard_specs,
            lambda spec: _prepare_pita_target_discovery_plan_for_spec(
                spec=spec,
                prepare_root=prepare_root,
            ),
            max_parallel=prep_workers,
        )

    CONF.output_volume.commit()
    pita_utr_shards = [
        shard
        for plan in pita_plans
        for shard in sorted(plan.utr_shards, key=lambda item: item.shard_index)
    ]
    pita_utr_outputs = _run_pita_prepare_utr_shard_batches(pita_utr_shards)
    CONF.output_volume.reload()
    pita_outputs_by_path = {
        shard.output_path: output
        for shard, output in zip(pita_utr_shards, pita_utr_outputs, strict=True)
    }
    prepared_shards = [
        _finalize_pita_target_discovery_plan(plan, pita_outputs_by_path)
        for plan in pita_plans
    ]
    CONF.output_volume.commit()

    row_shards = [row for prepared in prepared_shards for row in prepared.row_shards]
    row_batches = _batch_items_for_local_workers(
        row_shards,
        max_nodes=_bounded_node_count(
            len(row_shards),
            env_name=APP_INFO.off_target_nodes_env,
            default=APP_INFO.default_off_target_nodes,
        ),
        local_workers=local_workers,
    )
    if row_batches:
        print(
            "💊 Running OligoFormer PITA scoring for "
            f"{len(row_shards)} row shards on up to {len(row_batches)} CPU nodes "
            f"with {local_workers} workers each"
        )
        bounded_map(
            row_batches,
            lambda batch: run_oligoformer_pita_row_shard_batch.remote(
                list(batch),
                local_workers=local_workers,
            ),
            max_parallel=len(row_batches),
        )

    finalize_batches, finalize_workers = batches_for_total_concurrency(
        prepared_shards,
        max_batches=node_count,
        max_workers_per_batch=local_workers,
        total_concurrency=node_count,
    )
    print(
        "💊 Finalizing OligoFormer PITA tables for "
        f"{len(prepared_shards)} siRNAs in {len(finalize_batches)} batches with "
        f"{finalize_workers} workers each"
    )
    result_batches = bounded_map(
        finalize_batches,
        lambda batch: finalize_oligoformer_pita_shard_batch.remote(
            list(batch),
            local_workers=finalize_workers,
        ),
        max_parallel=len(finalize_batches),
    )
    return [result for batch in result_batches for result in batch]


def _run_off_target_shards(
    *,
    run_root: str,
    records: list[OffTargetSirnaRecord],
    stem: str,
    utr_path: str,
    orf_path: str,
    infer_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    targetscan_ref_shard_size: int | None = None,
) -> None:
    """Run per-siRNA off-target shards and merge their raw outputs."""
    if not records:
        raise RuntimeError("No siRNA records available for off-target prediction")

    raw_off_target_dir = (
        AppRunLayout.from_run_root(run_root).prep_dir / "off_target" / stem
    )
    merged_pita_path = raw_off_target_dir / "pita.tab"
    merged_targetscan_path = raw_off_target_dir / "targetscan.tab"
    if _raw_off_target_ready(raw_off_target_dir):
        infer_dir.joinpath("pita.tab").write_bytes(merged_pita_path.read_bytes())
        infer_dir.joinpath("targetscan.tab").write_bytes(
            merged_targetscan_path.read_bytes()
        )
        print(
            f"💊 Reusing cached OligoFormer off-target evidence: {raw_off_target_dir}"
        )
        return

    row_shard_size = _positive_int_from_env(
        APP_INFO.off_target_row_shard_size_env,
        APP_INFO.default_pita_row_shard_size,
    )
    node_count = _bounded_node_count(
        len(records),
        env_name=APP_INFO.off_target_nodes_env,
        default=APP_INFO.default_off_target_nodes,
    )
    local_workers = _positive_int_from_env(
        APP_INFO.off_target_workers_env,
        APP_INFO.default_off_target_workers_per_node,
    )
    shard_specs = [
        _off_target_shard_spec(
            run_root=run_root,
            output_dir=output_dir,
            stem=stem,
            item=item,
            utr_path=utr_path,
            orf_path=orf_path,
            row_shard_size=row_shard_size,
        )
        for item in enumerate(records)
    ]
    targetscan_specs = _targetscan_batch_specs(
        run_root=run_root,
        output_dir=output_dir,
        stem=stem,
        records=records,
        utr_path=utr_path,
        orf_path=orf_path,
        ref_shard_size=targetscan_ref_shard_size,
    )
    print(
        "💊 Preparing OligoFormer off-target inputs for "
        f"{len(records)} siRNAs in the postprocess node"
    )
    print(f"💊 Saving OligoFormer off-target logs under {logs_dir}")

    CONF.output_volume.commit()
    branch_calls = (
        run_oligoformer_targetscan_branch.spawn(targetscan_specs),
        run_oligoformer_pita_branch.spawn(
            shard_specs,
            node_count=node_count,
            local_workers=local_workers,
        ),
    )
    print("💊 Running OligoFormer TargetScan and PITA branches concurrently")
    try:
        targetscan_paths_raw, shard_results_raw = modal.FunctionCall.gather(
            *branch_calls
        )
    except BaseException:
        for call in branch_calls:
            with suppress(Exception):
                call.cancel()
        raise
    targetscan_paths = cast(list[str], targetscan_paths_raw)
    shard_results = cast(list[OffTargetShardResult], shard_results_raw)

    CONF.output_volume.reload()
    _merge_pita_shards(shard_results, merged_pita_path)
    _merge_targetscan_batch_outputs(
        targetscan_paths=targetscan_paths,
        output_path=merged_targetscan_path,
    )
    raw_off_target_dir.joinpath("off_target.done").write_text("done", encoding="utf-8")
    _cleanup_off_target_transients(raw_off_target_dir)
    infer_dir.joinpath("pita.tab").write_bytes(merged_pita_path.read_bytes())
    infer_dir.joinpath("targetscan.tab").write_bytes(
        merged_targetscan_path.read_bytes()
    )


def _apply_off_target_filters(
    *,
    result,
    run_root: str,
    stem: str,
    utr_path: str,
    orf_path: str,
    output_dir: Path,
    top_n: int,
    pita_threshold: float,
    targetscan_threshold: float,
    targetscan_ref_shard_size: int | None = None,
):
    """Apply upstream-equivalent PITA and TargetScan post-processing."""
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
    off_target_logs_dir = output_dir / "logs" / "off_target" / stem
    _run_off_target_shards(
        run_root=run_root,
        records=records,
        stem=stem,
        utr_path=utr_path,
        orf_path=orf_path,
        infer_dir=infer_dir,
        output_dir=output_dir,
        logs_dir=off_target_logs_dir,
        targetscan_ref_shard_size=targetscan_ref_shard_size,
    )

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
    pita_raw = pl.read_csv(
        infer_dir / "pita.tab",
        separator="\t",
        schema_overrides={"Score": pl.String},
    )
    pita = (
        pita_raw
        .with_columns(
            pl.col("Score").cast(pl.Float64, strict=False).alias("_pita_score")
        )
        .sort(["microRNA", "_pita_score"], nulls_last=True)
        .group_by("microRNA", maintain_order=True)
        .agg(
            pl.first("Score").alias("pita_score"),
            pl.first("_pita_score").alias("_pita_score"),
        )
    )
    result = result.join(pita, left_on="tmp", right_on="microRNA", how="left")
    result = result.with_columns(
        pl
        .when(pl.col("_pita_score") < pita_threshold)
        .then(1)
        .otherwise(0)
        .alias("pita_filter")
    )

    targetscan_raw = _read_targetscan_table(infer_dir / "targetscan.tab")
    targetscan = targetscan_raw.group_by("siRNA").agg(
        pl.col("targetscan_score").max().alias("targetscan_score")
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

    output_stems = _fasta_record_names(mrna_fasta_bytes)
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
    efficacy_key = _efficacy_key_for_run(
        mrna_fasta_bytes=mrna_fasta_bytes,
        sirna_fasta_bytes=sirna_fasta_bytes,
        functionality_filter=functionality_filter,
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
    )


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
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
    )
    if refreshed_plan.efficacy_ready:
        return refreshed_plan

    with _cache_build_lock(
        "efficacy",
        plan.efficacy_key,
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
        )
        if refreshed_plan.efficacy_ready:
            return refreshed_plan
        if not owns_cache_build:
            raise RuntimeError(
                "OligoFormer efficacy cache was marked complete without outputs"
            )
        if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
            raise FileNotFoundError(
                "OligoFormer RNA-FM weights are missing. Run "
                "download_oligoformer_models first."
            )

        _ensure_rnafm_runtime()
        import orjson

        input_layout = AppRunLayout.from_run_root(plan.run_root)
        efficacy_layout = _efficacy_layout_for_key(
            plan.efficacy_key,
            plan.output_stems,
        )
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
        efficacy_layout.markers_dir.mkdir(parents=True, exist_ok=True)
        _marker_path(efficacy_layout, "efficacy.done").write_bytes(
            orjson.dumps({
                "efficacy_key": plan.efficacy_key,
                "output_stems": list(plan.output_stems),
            })
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
    targetscan_ref_shard_size: int | None = None,
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
        needs_reference_guard = (
            off_target and all_human and not refreshed_plan.evidence_ready
        )
        if off_target and all_human:
            if needs_reference_guard:
                _ensure_human_refs()
                if not _targetscan_rnaplfold_cache_ready():
                    prepare_oligoformer_targetscan_rnaplfold_cache.remote(force=False)
                    CONF.output_volume.reload()
            utr_path = str(APP_INFO.model_ref_dir / "human_UTR.txt")
            orf_path = str(APP_INFO.model_ref_dir / "human_ORF.txt")
        else:
            utr_path = str(layout.inputs_dir / "utr.txt")
            orf_path = str(layout.inputs_dir / "orf.txt")

        reference_guard = (
            _cache_build_lock(
                "targetscan-reference-state",
                "global",
                rebuild=True,
            )
            if needs_reference_guard
            else nullcontext(False)
        )
        with reference_guard as owns_reference_state:
            if needs_reference_guard:
                if not owns_reference_state:
                    raise RuntimeError(
                        "OligoFormer reference-state access was not serialized"
                    )
                CONF.output_volume.reload()
                MODEL_VOLUME.reload()
                if (
                    refreshed_plan.reference_identity is None
                    or _targetscan_ref_identity_digest()
                    != refreshed_plan.reference_identity
                    or not _targetscan_ref_identity_matches_model()
                ):
                    raise FileNotFoundError(
                        "OligoFormer human references changed after run preparation. "
                        "Prepare and submit the run again."
                    )
                if not _targetscan_rnaplfold_cache_ready():
                    raise FileNotFoundError(
                        "OligoFormer RNAplfold references changed after preparation. "
                        "Prepare and submit the run again."
                    )

            for stem in refreshed_plan.output_stems:
                result = _read_efficacy_output(efficacy_dir / f"{stem}.txt")
                if off_target:
                    evidence_dir = layout.prep_dir / "off_target" / stem
                    needs_lock = not _raw_off_target_ready(evidence_dir)
                    lock = (
                        _cache_build_lock(
                            "off-target-evidence",
                            f"{refreshed_plan.run_root}\n{stem}",
                        )
                        if needs_lock
                        else nullcontext()
                    )
                    with lock as owns_cache_build:
                        if needs_lock:
                            CONF.output_volume.reload()
                            if not owns_cache_build and not _raw_off_target_ready(
                                evidence_dir
                            ):
                                raise RuntimeError(
                                    "OligoFormer off-target evidence cache was marked "
                                    "complete without outputs"
                                )
                        result = _apply_off_target_filters(
                            result=result,
                            run_root=refreshed_plan.run_root,
                            stem=stem,
                            utr_path=utr_path,
                            orf_path=orf_path,
                            output_dir=output_dir,
                            top_n=top_n,
                            pita_threshold=pita_threshold,
                            targetscan_threshold=targetscan_threshold,
                            targetscan_ref_shard_size=targetscan_ref_shard_size,
                        )
                        if owns_cache_build:
                            CONF.output_volume.commit()
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
    return _package_output_tables(output_dir, refreshed_plan.output_stems)


@app.function(
    cpu=(0.125, 16.125),
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
    targetscan_ref_shard_size: int | None = None,
) -> bytes:
    """Run CPU post-processing and return packaged OligoFormer outputs."""
    if targetscan_ref_shard_size is not None and targetscan_ref_shard_size < 1:
        raise ValueError("targetscan_ref_shard_size must be a positive integer")
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

    with _cache_build_lock(
        "final-tables",
        f"{plan.run_root}\n{plan.postprocess_key}",
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
        )
        if refreshed_plan.final_ready:
            return _package_output_tables(
                Path(refreshed_plan.output_dir), refreshed_plan.output_stems
            )
        if not owns_final_build:
            raise RuntimeError(
                "OligoFormer final-table cache was marked complete without outputs"
            )
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
            targetscan_ref_shard_size=targetscan_ref_shard_size,
        )


@app.function(
    cpu=(0.125, 16.125),
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
    )
    if not refreshed_plan.final_ready:
        raise FileNotFoundError(
            f"OligoFormer final outputs are incomplete for {refreshed_plan.cache_key}"
        )
    return _package_output_tables(
        Path(refreshed_plan.output_dir), refreshed_plan.output_stems
    )


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
    targetscan_ref_shard_size: int | None = None,
    force: bool = False,
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
        targetscan_ref_shard_size: Advanced TargetScan UTR records per
            reference-preparation shard. Defaults to the
            OLIGOFORMER_TARGETSCAN_PREPARE_REF_SHARD_SIZE environment variable
            when set, otherwise 1000.
        force: Rebuild cached intermediates and outputs.
    """
    input_path = Path(mrna_fasta).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"mRNA FASTA not found: {input_path}")
    run_name = run_name or input_path.stem
    out_file = build_local_output_path(
        resolve_local_output_dir(out_dir),
        run_name=run_name,
        suffix="oligoformer",
    )

    if off_target and not all_human and (utr_file is None or orf_file is None):
        raise ValueError(
            "Set --utr-file and --orf-file for off-target prediction, or pass "
            "--all-human."
        )
    if targetscan_ref_shard_size is not None and targetscan_ref_shard_size < 1:
        raise ValueError("targetscan_ref_shard_size must be a positive integer")
    resolved_targetscan_ref_shard_size = (
        targetscan_ref_shard_size
        if targetscan_ref_shard_size is not None
        else _positive_int_from_env(
            APP_INFO.targetscan_prepare_ref_shard_size_env,
            APP_INFO.default_targetscan_prepare_ref_shard_size,
        )
    )

    sirna_fasta_bytes = None
    if sirna_fasta is not None:
        sirna_path = Path(sirna_fasta).expanduser().resolve()
        if not sirna_path.exists():
            raise FileNotFoundError(f"siRNA FASTA not found: {sirna_path}")
        sirna_fasta_bytes = sirna_path.read_bytes()

    utr_bytes = None
    if utr_file is not None:
        utr_path = Path(utr_file).expanduser().resolve()
        if not utr_path.exists():
            raise FileNotFoundError(f"UTR reference not found: {utr_path}")
        utr_bytes = utr_path.read_bytes()

    orf_bytes = None
    if orf_file is not None:
        orf_path = Path(orf_file).expanduser().resolve()
        if not orf_path.exists():
            raise FileNotFoundError(f"ORF reference not found: {orf_path}")
        orf_bytes = orf_path.read_bytes()

    print(f"🧬 Submitting OligoFormer run '{run_name}'")
    force_generation = None
    if force:
        from uuid import uuid4

        force_generation = uuid4().hex
    prepare_kwargs = {
        "mrna_fasta_bytes": input_path.read_bytes(),
        "sirna_fasta_bytes": sirna_fasta_bytes,
        "off_target": off_target,
        "toxicity": toxicity,
        "all_human": all_human,
        "utr_bytes": utr_bytes,
        "orf_bytes": orf_bytes,
        "top_n": top_n,
        "functionality_filter": functionality_filter,
        "pita_threshold": pita_threshold,
        "targetscan_threshold": targetscan_threshold,
        "toxicity_threshold": toxicity_threshold,
        "force": force,
        "force_generation": force_generation,
    }
    model_assets_ready = False
    try:
        plan = prepare_oligoformer_run.remote(**prepare_kwargs)
    except FileNotFoundError:
        if not (off_target and all_human):
            raise
        download_oligoformer_models.remote(force=False)
        model_assets_ready = True
        plan = prepare_oligoformer_run.remote(**prepare_kwargs)

    needs_human_reference_cache = off_target and all_human and not plan.evidence_ready
    if (
        not plan.final_ready
        and (not plan.efficacy_ready or needs_human_reference_cache)
        and not model_assets_ready
    ):
        download_oligoformer_models.remote(force=False)
        model_assets_ready = True
        if off_target and all_human:
            plan = prepare_oligoformer_run.remote(**prepare_kwargs)

    if plan.final_ready:
        print(f"🧬 Reusing cached OligoFormer outputs: {plan.run_root}")
        tarball_bytes = package_oligoformer_outputs.remote(plan)
    else:
        needs_human_reference_cache = (
            off_target and all_human and not plan.evidence_ready
        )

        reference_call = None
        if needs_human_reference_cache:
            print("🧬 Preparing reusable TargetScan RNAplfold references")
            reference_call = prepare_oligoformer_targetscan_rnaplfold_cache.spawn(
                force=False
            )

        efficacy_call = None
        if plan.efficacy_ready:
            print(f"🧬 Reusing cached OligoFormer efficacy: {plan.run_root}")
            efficacy_plan = plan
        else:
            print("🧬 Running OligoFormer efficacy on GPU")
            efficacy_call = run_oligoformer_efficacy.spawn(
                plan=plan,
                functionality_filter=functionality_filter,
            )

        if efficacy_call is not None and reference_call is not None:
            try:
                efficacy_plan_raw, _ = modal.FunctionCall.gather(
                    efficacy_call,
                    reference_call,
                )
            except BaseException:
                for call in (efficacy_call, reference_call):
                    with suppress(Exception):
                        call.cancel()
                raise
            efficacy_plan = cast(OligoformerRunPlan, efficacy_plan_raw)
        elif efficacy_call is not None:
            efficacy_plan = efficacy_call.get()
        elif reference_call is not None:
            reference_call.get()

        print("🧬 Running OligoFormer CPU post-processing")
        tarball_bytes = run_oligoformer_postprocess.remote(
            plan=efficacy_plan,
            off_target=off_target,
            toxicity=toxicity,
            all_human=all_human,
            top_n=top_n,
            functionality_filter=functionality_filter,
            pita_threshold=pita_threshold,
            targetscan_threshold=targetscan_threshold,
            toxicity_threshold=toxicity_threshold,
            targetscan_ref_shard_size=resolved_targetscan_ref_shard_size,
        )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 OligoFormer run complete! Results saved to {out_file}")
