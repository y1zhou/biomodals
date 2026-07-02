"""OligoFormer source repo: <https://github.com/lulab/OligoFormer>.

OligoFormer predicts siRNA efficacy from an mRNA FASTA file. This wrapper builds
the runtime from the upstream README and supports OligoFormer's off-target and
toxicity options for standalone runs.

## Off-target prediction

When `--off-target` is set, provide both `--utr-file` and `--orf-file`, or set
`--all-human` to use the upstream bundled human references.

## Outputs

Results are saved locally as `<run-name>_oligoformer.tar.zst`. The tarball
contains the final top-level `.txt`, `_ranked.txt`, and `_ranked_filtered.txt`
tables only; detailed off-target logs stay in the Modal output-volume cache.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import hashlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import modal
import polars as pl

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import package_outputs, run_command
from biomodals.helper.task_budget import bounded_map
from biomodals.helper.web import download_files

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
    repo_rnafm_dir: Path = CONF.git_clone_dir / "RNA-FM"
    model_rnafm_dir: Path = Path(CONF.model_volume_mountpoint) / "RNA-FM"
    repo_ref_dir: Path = CONF.git_clone_dir / "off-target/ref"
    model_ref_dir: Path = Path(CONF.model_volume_mountpoint) / "off-target/ref"
    human_ref_filenames: tuple[str, ...] = ("human_UTR.txt", "human_ORF.txt")
    default_top_n: int = 20
    prepared_marker_name: str = "oligoformer.json"
    off_target_workers_env: str = "OLIGOFORMER_OFF_TARGET_WORKERS"
    off_target_nodes_env: str = "OLIGOFORMER_OFF_TARGET_NODES"
    off_target_row_shard_size_env: str = "OLIGOFORMER_PITA_ROW_SHARD_SIZE"
    off_target_row_attempts_env: str = "OLIGOFORMER_PITA_ROW_ATTEMPTS"
    default_off_target_nodes: int = 32
    default_off_target_workers_per_node: int = 32
    default_pita_row_shard_size: int = 1000
    default_pita_row_attempts: int = 3

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
        """Return full-human off-target reference zip downloads."""
        return {
            (
                f"{CONF.repo_url}/raw/{CONF.repo_commit_hash}/off-target/ref/"
                f"{filename}.zip"
            ): self.model_ref_dir / f"{filename}.zip"
            for filename in self.human_ref_filenames
        }

    @property
    def model_human_ref_paths(self) -> tuple[Path, ...]:
        """Return extracted full-human off-target reference paths."""
        return tuple(self.model_ref_dir / name for name in self.human_ref_filenames)

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
class OligoformerRunPlan:
    """Volume-backed OligoFormer run plan."""

    cache_key: str
    run_root: str
    efficacy_dir: str
    output_dir: str
    output_stems: tuple[str, ...]
    efficacy_ready: bool
    final_ready: bool


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
    targetscan_path: str


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
class PreparedOffTargetShard:
    """Cached per-siRNA off-target inputs ready for row-shard scoring."""

    index: int
    record_name: str
    cache_dir: str
    logs_dir: str
    pita_path: str
    targetscan_path: str
    row_shards: tuple[PitaRowShardSpec, ...]


def _hash_bytes(data: bytes | None) -> str:
    """Return a stable hash for optional bytes."""
    if data is None:
        return ""
    return hashlib.sha256(data).hexdigest()


def _hash_path(path: Path) -> str:
    """Return a stable hash for a file path."""
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


def _run_layout_for_cache_key(cache_key: str) -> AppRunLayout:
    """Return the output-volume run layout for an OligoFormer cache key."""
    return AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / "cache" / cache_key[:2] / cache_key
    )


def _marker_path(layout: AppRunLayout, marker: str) -> Path:
    """Return an OligoFormer cache marker path."""
    return layout.markers_dir / marker


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


def _paths_ready(paths: tuple[Path, ...], marker: Path) -> bool:
    """Return whether a marker and all paths exist."""
    return marker.exists() and all(path.exists() for path in paths)


def _build_plan(
    cache_key: str,
    output_stems: tuple[str, ...],
    run_root: str | Path | None = None,
) -> OligoformerRunPlan:
    """Build an OligoFormer run plan from current volume state."""
    layout = (
        AppRunLayout.from_run_root(run_root)
        if run_root is not None
        else _run_layout_for_cache_key(cache_key)
    )
    efficacy_dir = layout.prep_dir / "efficacy"
    output_dir = layout.outputs_dir
    return OligoformerRunPlan(
        cache_key=cache_key,
        run_root=str(layout.run_root),
        efficacy_dir=str(efficacy_dir),
        output_dir=str(output_dir),
        output_stems=output_stems,
        efficacy_ready=_paths_ready(
            _output_paths(efficacy_dir, output_stems),
            _marker_path(layout, "efficacy.done"),
        ),
        final_ready=_paths_ready(
            _output_paths(output_dir, output_stems),
            _marker_path(layout, "final.done"),
        ),
    )


APP_INFO = AppInfo()


##########################################
# Image and app definitions
##########################################
# TODO(oligoformer-perf): Preserve current Biomodals TargetScan outputs for the
# first GPU/CPU split. Before adding upstream RNAplfold binaries to PATH and
# restoring site-accessibility scoring, add golden tests because that can change
# off-target scores and increase CPU runtime.
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "ca-certificates",
        "unzip",
        "perl",
        "libstatistics-lite-perl",
        "libbio-perl-perl",
        "zstd",
    )
    .env(CONF.default_env)
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "grep -q \"Args.orf = './off-target/ref/human_UTR.txt'\" scripts/infer.py",
            "sed -i \"s|Args.orf = './off-target/ref/human_UTR.txt'|Args.orf = './off-target/ref/human_ORF.txt'|\" scripts/infer.py",
            'grep -q "for i in range(Args.top_n):" scripts/infer.py',
            'sed -i "s|for i in range(Args.top_n):|for i in range(min(Args.top_n, RESULT_ranked.shape[0])):|g" scripts/infer.py',
            f"python -c {shlex.quote(APP_INFO.stage_patch_runner)}",
            "rm -f off-target/ref/human_UTR.txt.zip off-target/ref/human_ORF.txt.zip",
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


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=CONF.timeout
)
def download_oligoformer_models(force: bool = False) -> None:
    """Download RNA-FM weights and full-human refs into the model volume."""
    import shutil
    import zipfile

    refs_ready = all(path.is_file() for path in APP_INFO.model_human_ref_paths)
    if APP_INFO.model_rnafm_redevelop_dir.is_dir() and refs_ready and not force:
        print("🧬 OligoFormer models and human refs already available")
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

    if force or not refs_ready:
        APP_INFO.model_ref_dir.mkdir(parents=True, exist_ok=True)
        download_files(
            APP_INFO.human_ref_downloads,
            force=force,
            num_retries=3,
            progress_bar_desc="OligoFormer human ref downloads",
        )
        for ref_path in APP_INFO.model_human_ref_paths:
            if force or not ref_path.is_file():
                with zipfile.ZipFile(
                    ref_path.with_suffix(ref_path.suffix + ".zip")
                ) as ref_zip:
                    ref_path.write_bytes(ref_zip.read(ref_path.name))

    if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        raise FileNotFoundError(
            "OligoFormer RNA-FM weights were not extracted to "
            f"{APP_INFO.model_rnafm_redevelop_dir}"
        )
    missing_refs = [
        str(path) for path in APP_INFO.model_human_ref_paths if not path.is_file()
    ]
    if missing_refs:
        raise FileNotFoundError(
            "OligoFormer full-human refs were not extracted: " + ", ".join(missing_refs)
        )

    MODEL_VOLUME.commit()
    print("🧬 OligoFormer models and human refs downloaded and committed")


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
    """Validate full-human refs in the model volume."""
    missing_refs = [
        str(path) for path in APP_INFO.model_human_ref_paths if not path.is_file()
    ]
    if missing_refs:
        raise FileNotFoundError(
            "OligoFormer full-human refs are missing. Run "
            "download_oligoformer_models first: " + ", ".join(missing_refs)
        )


def _cache_key_for_run(
    *,
    mrna_fasta_bytes: bytes,
    sirna_fasta_bytes: bytes | None,
    off_target: bool,
    toxicity: bool,
    all_human: bool,
    utr_bytes: bytes | None,
    orf_bytes: bytes | None,
    top_n: int,
    functionality_filter: bool,
    pita_threshold: float,
    targetscan_threshold: float,
    toxicity_threshold: float,
) -> str:
    """Return a deterministic cache key for one OligoFormer run."""
    ref_parts = ["off-target=0"]
    if off_target:
        ref_parts = [f"all-human={int(all_human)}"]
        if all_human:
            _ensure_human_refs()
            ref_parts.extend(
                f"{path.name}:{_hash_path(path)}"
                for path in APP_INFO.model_human_ref_paths
            )
        else:
            ref_parts.extend((
                f"utr:{_hash_bytes(utr_bytes)}",
                f"orf:{_hash_bytes(orf_bytes)}",
            ))

    return hash_string(
        "\n".join((
            CONF.name,
            CONF.version or "",
            CONF.repo_commit_hash or "",
            f"mrna:{_hash_bytes(mrna_fasta_bytes)}",
            f"sirna:{_hash_bytes(sirna_fasta_bytes)}",
            f"off_target:{int(off_target)}",
            f"toxicity:{int(toxicity)}",
            f"top_n:{top_n if off_target else ''}",
            f"functionality_filter:{int(functionality_filter)}",
            f"pita_threshold:{pita_threshold if off_target else ''}",
            f"targetscan_threshold:{targetscan_threshold if off_target else ''}",
            f"toxicity_threshold:{toxicity_threshold if toxicity else ''}",
            *ref_parts,
        ))
    )


def _write_cache_marker(layout: AppRunLayout, marker: str, plan: OligoformerRunPlan):
    """Write a small cache marker after a stage completes."""
    import orjson

    layout.markers_dir.mkdir(parents=True, exist_ok=True)
    _marker_path(layout, marker).write_bytes(
        orjson.dumps({
            "cache_key": plan.cache_key,
            "output_stems": plan.output_stems,
        })
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
    """Return upstream-shaped siRNA FASTA records for off-target tools."""
    if top_n == -1:
        return [
            OffTargetSirnaRecord(name=f"RNA{int(pos) - 1}", sequence=str(sirna))
            for pos, sirna in result.select("pos", "siRNA").iter_rows()
        ]

    original_sirnas = result.get_column("siRNA").to_list()
    ranked_indices = (
        result
        .with_row_index("_biomodals_index")
        .sort("efficacy", descending=True)
        .get_column("_biomodals_index")
        .to_list()
    )
    return [
        # Preserve upstream's current top_n behavior, including its index use.
        OffTargetSirnaRecord(
            name=f"RNA{ranked_indices[idx]}", sequence=str(original_sirnas[idx])
        )
        for idx in range(min(top_n, len(ranked_indices)))
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


def _off_target_nodes(task_count: int) -> int:
    """Return the distributed off-target node fanout."""
    if task_count < 1:
        return 1
    nodes = _positive_int_from_env(
        APP_INFO.off_target_nodes_env, APP_INFO.default_off_target_nodes
    )
    return max(1, min(nodes, task_count))


def _off_target_workers_per_node() -> int:
    """Return the local worker budget for each off-target CPU node."""
    return _positive_int_from_env(
        APP_INFO.off_target_workers_env, APP_INFO.default_off_target_workers_per_node
    )


def _pita_row_shard_size() -> int:
    """Return the PITA row-shard size, preserving upstream's 1000-row batches."""
    return _positive_int_from_env(
        APP_INFO.off_target_row_shard_size_env, APP_INFO.default_pita_row_shard_size
    )


def _pita_row_attempts() -> int:
    """Return attempts for transient PITA row-shard subprocess interruption."""
    return _positive_int_from_env(
        APP_INFO.off_target_row_attempts_env, APP_INFO.default_pita_row_attempts
    )


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


def _off_target_shard_infer_dir(spec: OffTargetShardSpec, shard_root: Path) -> Path:
    """Return the isolated upstream infer directory for one siRNA shard."""
    return shard_root / "data/infer" / f"{spec.stem}_shard_{spec.index:05d}"


def _off_target_shard_sirna_file(spec: OffTargetShardSpec, shard_root: Path) -> Path:
    """Write and return the single-siRNA FASTA path for one shard."""
    sirna_file = _off_target_shard_infer_dir(spec, shard_root) / "siRNA.fa"
    if not sirna_file.exists():
        _write_sirna_records(
            [OffTargetSirnaRecord(spec.record_name, spec.record_sequence)],
            sirna_file,
        )
    return sirna_file


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


def _batch_items(
    items: list[PitaRowShardSpec],
    max_batches: int,
) -> list[list[PitaRowShardSpec]]:
    """Split row-shard specs into at most ``max_batches`` ordered batches."""
    if not items:
        return []
    batch_count = max(1, min(max_batches, len(items)))
    batch_size = (len(items) + batch_count - 1) // batch_count
    return [
        items[index : index + batch_size] for index in range(0, len(items), batch_size)
    ]


def _prepare_off_target_shard_root(shard_root: Path) -> None:
    """Create an isolated off-target script workspace for one siRNA shard."""
    import shutil

    off_target_root = shard_root / "off-target"
    off_target_root.mkdir(parents=True)
    shutil.copytree(CONF.git_clone_dir / "off-target/pita", off_target_root / "pita")
    shutil.copytree(
        CONF.git_clone_dir / "off-target/targetscan",
        off_target_root / "targetscan",
    )


def _write_pita_prepare_script(script_path: Path, potential_targets_path: Path) -> None:
    """Write a patched PITA runner that stops after potential target discovery."""
    source = (CONF.git_clone_dir / "off-target/pita/lib/pita_run.pl").read_text(
        encoding="utf-8"
    )
    before_scoring, separator, _ = source.partition("## Step 2: Compute site scores")
    if not separator:
        raise RuntimeError("Could not patch PITA runner before site scoring")
    _, helper_separator, helpers = source.partition("sub dsystem")
    if not helper_separator:
        raise RuntimeError("Could not find PITA dsystem helper")
    perl_potential_targets_path = (
        str(potential_targets_path).replace("\\", "\\\\").replace("'", "\\'")
    )
    script_path.write_text(
        before_scoring
        + f"""
require File::Copy;
File::Copy::copy("tmp_pt_$r", '{perl_potential_targets_path}') or die "copy tmp_pt failed: $!";
exit (0);
sub dsystem{helpers}
""",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


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


def _run_pita_prepare(
    spec: OffTargetShardSpec,
    shard_root: Path,
) -> PreparedOffTargetShard:
    """Prepare cached PITA potential-target rows for one siRNA."""
    cache_dir = _off_target_shard_cache_dir(spec)
    logs_dir = _off_target_shard_logs_dir(spec)
    row_dir = cache_dir / "pita_rows"
    potential_targets_path = cache_dir / "potential_targets.tsv"
    ext_utr_path = cache_dir / f"{spec.stem}_shard_{spec.index:05d}_ext_utr.stab"
    marker_path = cache_dir / "pita_prepare.done"
    cache_dir.mkdir(parents=True, exist_ok=True)
    row_dir.mkdir(parents=True, exist_ok=True)

    if (
        marker_path.exists()
        and potential_targets_path.exists()
        and ext_utr_path.exists()
    ):
        row_count = int(marker_path.read_text(encoding="utf-8"))
        row_shards = _pita_row_shard_specs(
            spec=spec,
            row_count=row_count,
            potential_targets_path=potential_targets_path,
            ext_utr_path=ext_utr_path,
            row_dir=row_dir,
        )
        _ensure_pita_row_inputs(potential_targets_path, row_shards)
        return PreparedOffTargetShard(
            index=spec.index,
            record_name=spec.record_name,
            cache_dir=str(cache_dir),
            logs_dir=str(logs_dir),
            pita_path=str(cache_dir / "pita.tab"),
            targetscan_path=str(cache_dir / "targetscan.tab"),
            row_shards=row_shards,
        )

    pita_root = shard_root / "off-target/pita"
    sirna_file = _off_target_shard_sirna_file(spec, shard_root)
    script_path = pita_root / "prepare_pita_rows.pl"
    _write_pita_prepare_script(script_path, potential_targets_path)
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
        log_file=logs_dir / "pita_prepare.log",
        show_command=False,
    )
    row_count = len(potential_targets_path.read_text(encoding="utf-8").splitlines())
    marker_path.write_text(str(row_count), encoding="utf-8")

    row_shards = _pita_row_shard_specs(
        spec=spec,
        row_count=row_count,
        potential_targets_path=potential_targets_path,
        ext_utr_path=ext_utr_path,
        row_dir=row_dir,
    )
    _ensure_pita_row_inputs(potential_targets_path, row_shards)
    return PreparedOffTargetShard(
        index=spec.index,
        record_name=spec.record_name,
        cache_dir=str(cache_dir),
        logs_dir=str(logs_dir),
        pita_path=str(cache_dir / "pita.tab"),
        targetscan_path=str(cache_dir / "targetscan.tab"),
        row_shards=row_shards,
    )


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


def _run_targetscan_cached(spec: OffTargetShardSpec, shard_root: Path) -> str:
    """Run or reuse cached TargetScan output for one siRNA."""
    cache_dir = _off_target_shard_cache_dir(spec)
    targetscan_path = cache_dir / "targetscan.tab"
    marker_path = cache_dir / "targetscan.done"
    if marker_path.exists() and targetscan_path.exists():
        return str(targetscan_path)

    infer_dir = _off_target_shard_infer_dir(spec, shard_root)
    sirna_file = _off_target_shard_sirna_file(spec, shard_root)
    logs_dir = _off_target_shard_logs_dir(spec)
    run_command(
        [
            "bash",
            str(CONF.git_clone_dir / "scripts/targetscan.sh"),
            str(sirna_file),
            spec.utr_path,
            spec.orf_path,
            f"{spec.stem}_shard_{spec.index:05d}",
        ],
        cwd=shard_root,
        output_mode="log",
        log_file=logs_dir / "targetscan.log",
        show_command=False,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    targetscan_path.write_bytes((infer_dir / "targetscan.tab").read_bytes())
    marker_path.write_text("done", encoding="utf-8")
    return str(targetscan_path)


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def prepare_oligoformer_off_target_shard(
    spec: OffTargetShardSpec,
) -> PreparedOffTargetShard:
    """Prepare one siRNA off-target shard in the shared output-volume cache."""
    CONF.output_volume.reload()
    with TemporaryDirectory(prefix=f"oligoformer_{spec.stem}_prepare_") as tmpdir:
        shard_root = Path(tmpdir) / f"{spec.stem}_shard_{spec.index:05d}"
        _prepare_off_target_shard_root(shard_root)

        _off_target_shard_sirna_file(spec, shard_root)
        targetscan_path, prepared = bounded_map(
            (
                lambda: _run_targetscan_cached(spec, shard_root),
                lambda: _run_pita_prepare(spec, shard_root),
            ),
            lambda task: task(),
            max_parallel=2,
        )
        prepared = PreparedOffTargetShard(
            index=prepared.index,
            record_name=prepared.record_name,
            cache_dir=prepared.cache_dir,
            logs_dir=prepared.logs_dir,
            pita_path=prepared.pita_path,
            targetscan_path=targetscan_path,
            row_shards=prepared.row_shards,
        )
    CONF.output_volume.commit()
    return prepared


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
        attempts = _pita_row_attempts()
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
    timeout=CONF.timeout,
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


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_oligoformer_pita_shard(
    prepared: PreparedOffTargetShard,
) -> OffTargetShardResult:
    """Finalize one per-siRNA PITA table from cached row shards."""
    CONF.output_volume.reload()
    pita_path = Path(prepared.pita_path)
    marker_path = Path(prepared.cache_dir) / "pita_finalize.done"
    if not marker_path.exists() or not pita_path.exists():
        row_outputs = [
            Path(row.output_path)
            for row in sorted(prepared.row_shards, key=lambda item: item.shard_index)
        ]
        _write_pita_targets_from_scored_rows(prepared, row_outputs)
        CONF.output_volume.commit()
    return OffTargetShardResult(
        index=prepared.index,
        pita_path=prepared.pita_path,
        targetscan_path=prepared.targetscan_path,
    )


def _merge_pita_shards(shard_results: list[OffTargetShardResult], output_path: Path):
    """Merge per-siRNA PITA tables, preserving one header row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with output_path.open("w", encoding="utf-8") as out:
        for shard in sorted(shard_results, key=lambda result: result.index):
            lines = Path(shard.pita_path).read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            if not wrote_header:
                out.write("\n".join(lines))
                out.write("\n")
                wrote_header = True
            elif len(lines) > 1:
                out.write("\n".join(lines[1:]))
                out.write("\n")


def _merge_targetscan_shards(
    shard_results: list[OffTargetShardResult], output_path: Path
):
    """Merge per-siRNA TargetScan tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for shard in sorted(shard_results, key=lambda result: result.index):
            data = Path(shard.targetscan_path).read_text(encoding="utf-8")
            if not data:
                continue
            out.write(data)
            if not data.endswith("\n"):
                out.write("\n")


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
) -> None:
    """Run per-siRNA off-target shards and merge their raw outputs."""
    if not records:
        raise RuntimeError("No siRNA records available for off-target prediction")

    row_shard_size = _pita_row_shard_size()
    node_count = _off_target_nodes(len(records))
    local_workers = _off_target_workers_per_node()
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
    print(
        "💊 Running OligoFormer off-target prep for "
        f"{len(records)} siRNAs on up to {node_count} CPU nodes"
    )
    print(f"💊 Saving OligoFormer off-target logs under {logs_dir}")
    prepared_shards = bounded_map(
        shard_specs,
        lambda spec: prepare_oligoformer_off_target_shard.remote(spec),
        max_parallel=node_count,
    )

    row_shards = [row for prepared in prepared_shards for row in prepared.row_shards]
    row_batches = _batch_items(row_shards, _off_target_nodes(len(row_shards)))
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

    CONF.output_volume.reload()
    shard_results = bounded_map(
        prepared_shards,
        lambda prepared: finalize_oligoformer_pita_shard.remote(prepared),
        max_parallel=node_count,
    )
    CONF.output_volume.reload()
    _merge_pita_shards(shard_results, infer_dir / "pita.tab")
    _merge_targetscan_shards(shard_results, infer_dir / "targetscan.tab")


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
    )

    original_columns = list(result.columns)
    result = result.with_columns(
        (pl.lit("RNA") + (pl.col("pos").cast(pl.Int64) - 1).cast(pl.String)).alias(
            "tmp"
        )
    )
    pita_raw = pl.read_csv(infer_dir / "pita.tab", separator="\t")
    pita = pita_raw.group_by("microRNA").agg(pl.col("Score").min().alias("pita_score"))
    result = result.join(pita, left_on="tmp", right_on="microRNA", how="left")
    result = result.with_columns(
        pl
        .when(pl.col("pita_score") < pita_threshold)
        .then(1)
        .otherwise(0)
        .alias("pita_filter")
    )
    if top_n != -1:
        result = result.with_columns(
            pl
            .when(pl.col("pita_score").is_null())
            .then(-1)
            .otherwise(pl.col("pita_filter"))
            .alias("pita_filter")
        )

    targetscan_raw = pl.read_csv(
        infer_dir / "targetscan.tab",
        separator="\t",
        has_header=False,
        new_columns=["refseq", "siRNA", "targetscan_score"],
    )
    targetscan = targetscan_raw.group_by("siRNA").agg(
        pl.col("targetscan_score").max().alias("targetscan_score")
    )
    pita_sirnas = pita.select(pl.col("microRNA").alias("siRNA"))
    missing_targetscan = pita_sirnas.join(
        targetscan.select("siRNA"), on="siRNA", how="anti"
    ).with_columns(pl.lit(0).alias("targetscan_score"))
    targetscan = pl.concat([targetscan, missing_targetscan], how="vertical_relaxed")
    result = result.join(targetscan, left_on="tmp", right_on="siRNA", how="left")
    result = result.with_columns(
        pl
        .when(pl.col("targetscan_score") > targetscan_threshold)
        .then(1)
        .otherwise(0)
        .alias("targetscan_filter")
    )
    if top_n != -1:
        result = result.with_columns(
            pl
            .when(pl.col("targetscan_score").is_null())
            .then(-1)
            .otherwise(pl.col("targetscan_filter"))
            .alias("targetscan_filter")
        )

    pita_hit = pl.col("pita_filter") == 1
    targetscan_hit = pl.col("targetscan_filter") == 1
    if top_n == -1:
        off_target_filter = pl.when(pita_hit | targetscan_hit).then(1).otherwise(0)
    else:
        pita_missing = pl.col("pita_filter") == -1
        targetscan_missing = pl.col("targetscan_filter") == -1
        off_target_filter = (
            pl
            .when(pita_missing | targetscan_missing)
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
    filter_terms = []
    if functionality_filter:
        filter_terms.append(pl.col("func_filter"))
    if off_target:
        filter_terms.append(pl.col("off_target_filter"))
    if toxicity:
        filter_terms.append(pl.col("toxicity_filter"))

    return result.with_columns(sum(filter_terms, start=pl.lit(0)).alias("filter"))


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
    volumes=CONF.mounts(output_volume=True, model_volume=True),
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
    CONF.output_volume.reload()
    cache_key = _cache_key_for_run(
        mrna_fasta_bytes=mrna_fasta_bytes,
        sirna_fasta_bytes=sirna_fasta_bytes,
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        utr_bytes=utr_bytes,
        orf_bytes=orf_bytes,
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
    )

    layout = _run_layout_for_cache_key(cache_key)
    if force and layout.run_root.exists():
        import shutil

        shutil.rmtree(layout.run_root)

    plan = _build_plan(cache_key, output_stems)
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
    return _build_plan(cache_key, output_stems)


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True, model_volume=True),
)
def run_oligoformer_efficacy(
    plan: OligoformerRunPlan,
    functionality_filter: bool = True,
) -> OligoformerRunPlan:
    """Run GPU efficacy prediction into the output-volume cache."""
    CONF.output_volume.reload()
    if plan.efficacy_ready:
        return plan
    if not APP_INFO.model_rnafm_redevelop_dir.is_dir():
        raise FileNotFoundError(
            "OligoFormer RNA-FM weights are missing. Run "
            "download_oligoformer_models first."
        )

    _ensure_rnafm_runtime()
    layout = AppRunLayout.from_run_root(plan.run_root)
    efficacy_dir = Path(plan.efficacy_dir)
    efficacy_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "scripts/main.py",
        "-i",
        "1",
        "-i1",
        str(layout.inputs_dir / "mrna.fa"),
        "--output_dir",
        f"{efficacy_dir}/",
        "--biomodals_stage",
        "efficacy",
    ]

    sirna_fasta = layout.inputs_dir / "sirna.fa"
    if sirna_fasta.exists():
        cmd.extend(["-i2", str(sirna_fasta)])
    if not functionality_filter:
        cmd.append("--no_func")

    run_command(cmd, cwd=CONF.git_clone_dir)
    _write_cache_marker(layout, "efficacy.done", plan)
    CONF.output_volume.commit()
    return _build_plan(plan.cache_key, plan.output_stems, plan.run_root)


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
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
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.run_root)
    refreshed_plan = _build_plan(plan.cache_key, plan.output_stems, plan.run_root)
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
            utr_path = str(APP_INFO.model_ref_dir / "human_UTR.txt")
            orf_path = str(APP_INFO.model_ref_dir / "human_ORF.txt")
        else:
            utr_path = str(layout.inputs_dir / "utr.txt")
            orf_path = str(layout.inputs_dir / "orf.txt")

        for stem in refreshed_plan.output_stems:
            result = _read_efficacy_output(efficacy_dir / f"{stem}.txt")
            if off_target:
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

    _write_cache_marker(layout, "final.done", refreshed_plan)
    CONF.output_volume.commit()
    return _package_output_tables(output_dir, refreshed_plan.output_stems)


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def package_oligoformer_outputs(plan: OligoformerRunPlan) -> bytes:
    """Return cached OligoFormer outputs as standalone tarball bytes."""
    CONF.output_volume.reload()
    refreshed_plan = _build_plan(plan.cache_key, plan.output_stems, plan.run_root)
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
        all_human: Use upstream bundled human ORF and UTR references for
            off-target prediction.
        utr_file: Local UTR reference file for off-target prediction.
        orf_file: Local ORF reference file for off-target prediction.
        top_n: Number of top siRNAs to use for off-target prediction. Defaults
            to 20; use -1 for all candidates.
        functionality_filter: Keep upstream functionality filtering enabled.
        pita_threshold: PITA threshold used by off-target prediction.
        targetscan_threshold: TargetScan threshold used by off-target prediction.
        toxicity_threshold: Toxicity filter threshold.
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
    download_oligoformer_models.remote(force=False)
    plan = prepare_oligoformer_run.remote(
        mrna_fasta_bytes=input_path.read_bytes(),
        sirna_fasta_bytes=sirna_fasta_bytes,
        off_target=off_target,
        toxicity=toxicity,
        all_human=all_human,
        utr_bytes=utr_bytes,
        orf_bytes=orf_bytes,
        top_n=top_n,
        functionality_filter=functionality_filter,
        pita_threshold=pita_threshold,
        targetscan_threshold=targetscan_threshold,
        toxicity_threshold=toxicity_threshold,
        force=force,
    )
    if plan.final_ready:
        print(f"🧬 Reusing cached OligoFormer outputs: {plan.run_root}")
        tarball_bytes = package_oligoformer_outputs.remote(plan)
    else:
        if plan.efficacy_ready:
            print(f"🧬 Reusing cached OligoFormer efficacy: {plan.run_root}")
            efficacy_plan = plan
        else:
            print("🧬 Running OligoFormer efficacy on GPU")
            efficacy_plan = run_oligoformer_efficacy.remote(
                plan=plan,
                functionality_filter=functionality_filter,
            )
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
        )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 OligoFormer run complete! Results saved to {out_file}")
