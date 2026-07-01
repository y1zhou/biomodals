"""OligoFormer source repo: <https://github.com/lulab/OligoFormer>.

OligoFormer predicts siRNA efficacy from an mRNA FASTA file. This wrapper builds
the runtime from the upstream README and supports OligoFormer's off-target and
toxicity options for standalone runs.

## Off-target prediction

When `--off-target` is set, provide both `--utr-file` and `--orf-file`, or set
`--all-human` to use the upstream bundled human references.

## Outputs

Results are saved locally as `<run-name>_oligoformer.tar.zst`.
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
    input_dir: str
    efficacy_dir: str
    output_dir: str
    output_stems: tuple[str, ...]
    efficacy_ready: bool
    final_ready: bool


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
        input_dir=str(layout.inputs_dir),
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


def _write_sirna_fasta(result, sirna_file: Path, top_n: int) -> None:
    """Write the upstream-shaped siRNA FASTA used by off-target tools."""
    sirna_file.parent.mkdir(parents=True, exist_ok=True)
    with sirna_file.open("w", encoding="utf-8") as handle:
        if top_n == -1:
            for _, row in result.iterrows():
                handle.write(f">RNA{int(row['pos']) - 1}\n")
                handle.write(f"{row['siRNA']}\n")
            return

        ranked = result.sort_values(by="efficacy", ascending=False)
        for idx in range(min(top_n, ranked.shape[0])):
            # Preserve upstream's current top_n behavior, including its index use.
            handle.write(f">RNA{ranked.index[idx]}\n")
            handle.write(f"{ranked['siRNA'].loc[idx]}\n")


def _apply_off_target_filters(
    *,
    result,
    stem: str,
    utr_path: str,
    orf_path: str,
    top_n: int,
    pita_threshold: float,
    targetscan_threshold: float,
):
    """Apply upstream-equivalent PITA and TargetScan post-processing."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    infer_dir = CONF.git_clone_dir / "data/infer" / stem
    if infer_dir.exists():
        shutil.rmtree(infer_dir)
    infer_dir.mkdir(parents=True)

    if top_n == -1:
        sirna_file = infer_dir / "siRNA.fa"
    else:
        sirna_file = infer_dir / "top_n_siRNA.fa"
    _write_sirna_fasta(result, sirna_file, top_n)

    # TODO(oligoformer-scale): Modal-node sharding can preserve Biomodals output
    # by splitting siRNAs into isolated workdirs, then reducing PITA min(Score)
    # and TargetScan max(score). The stock scripts are not safe to shard inside
    # one checkout: PITA uses tmp_seqfile1/2 and TargetScan uses off-target/tmp.
    commands = (
        ["bash", "scripts/pita.sh", utr_path, str(sirna_file), orf_path, stem],
        ["bash", "scripts/targetscan.sh", str(sirna_file), utr_path, orf_path, stem],
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_command, command, cwd=CONF.git_clone_dir)
            for command in commands
        ]
        for future in futures:
            future.result()

    try:
        import polars as pl
    except ImportError:
        return _apply_off_target_filters_pandas(
            result=result,
            infer_dir=infer_dir,
            top_n=top_n,
            pita_threshold=pita_threshold,
            targetscan_threshold=targetscan_threshold,
        )

    original_columns = list(result.columns)
    pl_when = pl.when
    result_pl = pl.DataFrame(result.to_dict(orient="list")).with_columns(
        (pl.lit("RNA") + (pl.col("pos").cast(pl.Int64) - 1).cast(pl.String)).alias(
            "tmp"
        )
    )
    pita_raw = pl.read_csv(infer_dir / "pita.tab", separator="\t")
    pita = pita_raw.group_by("microRNA").agg(pl.col("Score").min().alias("pita_score"))
    result_pl = result_pl.join(pita, left_on="tmp", right_on="microRNA", how="left")
    result_pl = result_pl.with_columns(
        pl_when(pl.col("pita_score") < pita_threshold)
        .then(1)
        .otherwise(0)
        .alias("pita_filter")
    )
    if top_n != -1:
        result_pl = result_pl.with_columns(
            pl_when(pl.col("pita_score").is_null())
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
    ).with_columns(pl.lit(0.0).alias("targetscan_score"))
    targetscan = pl.concat([targetscan, missing_targetscan], how="vertical_relaxed")
    result_pl = result_pl.join(targetscan, left_on="tmp", right_on="siRNA", how="left")
    result_pl = result_pl.with_columns(
        pl_when(pl.col("targetscan_score") > targetscan_threshold)
        .then(1)
        .otherwise(0)
        .alias("targetscan_filter")
    )
    if top_n != -1:
        result_pl = result_pl.with_columns(
            pl_when(pl.col("targetscan_score").is_null())
            .then(-1)
            .otherwise(pl.col("targetscan_filter"))
            .alias("targetscan_filter")
        )

    pita_hit = pl.col("pita_filter") == 1
    targetscan_hit = pl.col("targetscan_filter") == 1
    if top_n == -1:
        off_target_filter = pl_when(pita_hit | targetscan_hit).then(1).otherwise(0)
    else:
        pita_missing = pl.col("pita_filter") == -1
        targetscan_missing = pl.col("targetscan_filter") == -1
        off_target_filter = (
            pl_when(pita_missing | targetscan_missing)
            .then(-5)
            .when(pita_hit | targetscan_hit)
            .then(1)
            .otherwise(0)
        )
    result_pl = result_pl.with_columns(off_target_filter.alias("off_target_filter"))
    result_pl = result_pl.select(
        original_columns + ["pita_score", "targetscan_score", "off_target_filter"]
    )
    return result.__class__(result_pl.to_dict(as_series=False))


def _apply_off_target_filters_pandas(
    *,
    result,
    infer_dir: Path,
    top_n: int,
    pita_threshold: float,
    targetscan_threshold: float,
):
    """Apply off-target table merges with the legacy pandas implementation."""
    import pandas as pd

    pita = pd.read_csv(infer_dir / "pita.tab", sep="\t")
    pita = (
        pita
        .groupby("microRNA")
        .agg({"Score": "min"})
        .rename(columns={"Score": "pita_score"})
    )
    result["tmp"] = result["pos"].astype(str).apply(lambda x: "RNA" + str(int(x) - 1))
    result = pd.merge(result, pita, left_on="tmp", right_on="microRNA", how="left")
    result["pita_filter"] = [
        1 if value < pita_threshold else 0 for value in result["pita_score"]
    ]
    if top_n != -1:
        result.loc[result["pita_score"].isna(), "pita_filter"] = -1

    targetscan = pd.read_csv(
        infer_dir / "targetscan.tab",
        sep="\t",
        header=None,
        names=["refseq", "siRNA", "targetscan_score"],
    )
    targetscan = targetscan.groupby("siRNA").agg({"targetscan_score": "max"})
    for idx in list(set(pita.index) - set(targetscan.index)):
        targetscan.loc[idx] = 0
    result = pd.merge(result, targetscan, left_on="tmp", right_on="siRNA", how="left")
    result["targetscan_filter"] = [
        1 if value > targetscan_threshold else 0 for value in result["targetscan_score"]
    ]
    if top_n != -1:
        result.loc[result["targetscan_score"].isna(), "targetscan_filter"] = -1

    if top_n == -1:
        result["off_target_filter"] = [
            1 if pita_value == 1 or targetscan_value == 1 else 0
            for pita_value, targetscan_value in zip(
                result["pita_filter"], result["targetscan_filter"], strict=True
            )
        ]
    else:
        result["off_target_filter"] = [
            -5
            if pita_value == -1 or targetscan_value == -1
            else 1
            if pita_value == 1 or targetscan_value == 1
            else 0
            for pita_value, targetscan_value in zip(
                result["pita_filter"], result["targetscan_filter"], strict=True
            )
        ]
    return result.drop(columns=["tmp", "pita_filter", "targetscan_filter"])


def _apply_toxicity_filters(*, result, toxicity_threshold: float):
    """Apply upstream-equivalent toxicity post-processing."""
    import pandas as pd

    toxicity = pd.read_csv(CONF.git_clone_dir / "toxicity/cell_viability.txt", sep="\t")
    result["seed"] = result["siRNA"].str.slice(1, 7)
    result = pd.merge(result, toxicity, left_on="seed", right_on="Seed", how="left")
    result["toxicity_filter"] = [
        1 if value < toxicity_threshold else 0 for value in result["cell_viability"]
    ]
    return result.drop(columns=["seed"])


def _apply_final_filter(
    *, result, off_target: bool, toxicity: bool, functionality_filter: bool
):
    """Apply upstream-equivalent final filter aggregation."""
    if functionality_filter:
        if off_target:
            if toxicity:
                result["filter"] = (
                    result["func_filter"]
                    + result["off_target_filter"]
                    + result["toxicity_filter"]
                )
            else:
                result["filter"] = result["func_filter"] + result["off_target_filter"]
        elif toxicity:
            result["filter"] = result["func_filter"] + result["toxicity_filter"]
        else:
            result["filter"] = result["func_filter"]
    elif off_target:
        if toxicity:
            result["filter"] = result["off_target_filter"] + result["toxicity_filter"]
        else:
            result["filter"] = result["off_target_filter"]
    elif toxicity:
        result["filter"] = result["toxicity_filter"]
    else:
        result["filter"] = [0] * result.shape[0]
    return result


def _write_final_outputs(result, output_dir: Path, stem: str) -> None:
    """Write upstream-shaped OligoFormer final output tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = result.sort_values(by="efficacy", ascending=False)
    ranked_filtered = result[result["filter"] == 0].sort_values(
        by="efficacy", ascending=False
    )
    result.to_csv(output_dir / f"{stem}.txt", sep="\t", index=None, header=True)
    ranked.to_csv(output_dir / f"{stem}_ranked.txt", sep="\t", index=None, header=True)
    ranked_filtered.to_csv(
        output_dir / f"{stem}_ranked_filtered.txt", sep="\t", index=None, header=True
    )


@app.function(
    cpu=(0.125, 16.125),
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
    import pandas as pd

    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.run_root)
    refreshed_plan = _build_plan(plan.cache_key, plan.output_stems, plan.run_root)
    if refreshed_plan.final_ready:
        return package_outputs(Path(refreshed_plan.output_dir))
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
            result = pd.read_csv(efficacy_dir / f"{stem}.txt", sep="\t")
            if off_target:
                result = _apply_off_target_filters(
                    result=result,
                    stem=stem,
                    utr_path=utr_path,
                    orf_path=orf_path,
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
    return package_outputs(output_dir)


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
    return package_outputs(Path(refreshed_plan.output_dir))


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
