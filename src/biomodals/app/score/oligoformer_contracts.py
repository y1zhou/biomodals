"""Canonical OligoFormer configurations and worker plans, independent of Modal apps."""

from __future__ import annotations

from dataclasses import dataclass

EFFICACY_POLICY_VERSION = "eval-inference-mode-v1"
TARGETSCAN_RNAPLFOLD_MAX_NODES = 32
TARGETSCAN_RNAPLFOLD_MAX_WORKERS = 32


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

    off_target_nodes: int = 32
    off_target_workers: int = 32
    off_target_process_slots: int = 64
    off_target_prep_workers: int = 16
    pita_prepare_nodes: int = 32
    pita_prepare_workers: int = 32
    pita_prepare_utr_shard_size: int = 1000
    pita_row_shard_size: int = 1000
    pita_row_attempts: int = 3
    targetscan_rnaplfold_nodes: int = 32
    targetscan_rnaplfold_workers: int = 8
    targetscan_rnaplfold_shard_size: int = 500
    targetscan_prepare_nodes: int = 32
    targetscan_candidate_shard_size: int = 20
    targetscan_context_nodes: int = 100
    targetscan_context_workers: int = 32
    targetscan_context_shard_size: int = 500
    targetscan_context_attempts: int = 3
    targetscan_merge_nodes: int = 16

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
        if self.off_target_process_slots > 64:
            raise ValueError(f"off_target_process_slots must not exceed {64}")
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
    candidate_shard_size: int = 20
    candidate_shard_index: int = 0
    context_shard_size: int = 500


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
