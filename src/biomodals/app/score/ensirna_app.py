"""ENsiRNA source repo: <https://github.com/tanwenchong/ENsiRNA>.

ENsiRNA designs siRNA candidates from an mRNA FASTA file. This wrapper uses the
upstream Linux runtime requirements, builds on the Biomodals Rosetta base image,
and preserves the documented `design.sh` data-prep and inference steps while
running them as separate CPU and GPU Modal functions.

## Additional notes

The upstream Linux instructions require Rosetta for PDB generation. The runtime
uses the same public Rosetta base image as `rosetta_app.py`; commercial use may
require a separate Rosetta license. Model weights and checkpoints are stored in
the standard Biomodals model volume.

## Outputs

Results are saved locally as `<run-name>_ensirna.tar.zst`. The archive contains
the upstream ENsiRNA output directory, including `mrna_result.xlsx` when the run
completes successfully.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

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
from biomodals.helper.task_budget import bounded_map
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
    version="2",
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
    cache_namespace: str = (
        f"{CONF.name}:{CONF.version or ''}:{CONF.repo_commit_hash or ''}"
    )
    rnafm_device_env: str = "ENSIRNA_RNAFM_DEVICE"
    pdb_cores_env: str = "ENSIRNA_PDB_CORES"
    rosetta_compat_root: Path = Path(
        "/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371"
    )
    rnafm_pretrained_url: str = (
        "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
    )
    rnafm_cache_path: Path = (
        Path(CONF.default_env["TORCH_HOME"]) / "hub/checkpoints/RNA-FM_pretrained.pth"
    )
    checkpoint_filenames: tuple[str, ...] = tuple(
        f"checkpoint_{idx}.ckpt" for idx in range(1, 6)
    )
    checkpoint_dir: Path = Path(CONF.model_volume_mountpoint) / "pkl"
    conda_packages: tuple[str, ...] = (
        f"python={CONF.python_version}",
        "pip",
        "viennarna=2.6.4-0",
    )
    conda_channels: tuple[str, ...] = ("conda-forge", "bioconda")
    pip_packages: tuple[str, ...] = (
        "biopython",
        "numpy",
        "pandas",
        "scipy",
        "tensorboard",
        "tqdm",
        "openpyxl",
        "rdkit",
        "scikit-learn",
        "xgboost",
    )
    torch_packages: tuple[str, ...] = ("torch", "torchvision", "torchaudio")
    torch_index_url: str = "https://download.pytorch.org/whl/cu118"
    extra_pip_packages: tuple[str, ...] = ("torch-geometric", "rna-fm")

    @property
    def mamba_bin_path(self) -> str:
        """Return PATH with micromamba's base environment first."""
        return f"{self.mamba_root}/bin:/root/.local/bin:$PATH"

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
    def get_pdb_runtime_patch(self) -> str:
        """Return the source patch for ENsiRNA's PDB generation helper."""
        return f"""from pathlib import Path

path = Path({str(self.ensirna_dir / "data/get_pdb.py")!r})
text = path.read_text()
cores_old = '''        self.num_cores = multiprocessing.cpu_count()
'''
cores_new = '''        self.num_cores = max(
            1,
            int(os.environ.get({self.pdb_cores_env!r}, multiprocessing.cpu_count())),
        )
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
if cores_old not in text:
    raise SystemExit("expected ENsiRNA get_pdb CPU core block not found")
if old not in text:
    raise SystemExit("expected ENsiRNA get_pdb length check not found")
if secstruct_old not in text:
    raise SystemExit("expected ENsiRNA secondary structure block not found")
if rosetta_cmd_old not in text:
    raise SystemExit("expected ENsiRNA Rosetta command not found")
text = text.replace(cores_old, cores_new)
text = text.replace(old, new)
text = text.replace(secstruct_old, secstruct_new)
text = text.replace(rosetta_cmd_old, rosetta_cmd_new)
path.write_text(text)
"""

    @property
    def get_pdb_runtime_patch_runner(self) -> str:
        """Return a Python one-liner that applies the PDB helper patch."""
        return f"exec({self.get_pdb_runtime_patch!r})"

    @property
    def dataset_runtime_patch(self) -> str:
        """Return the source patch for lazy RNA-FM preprocessing."""
        return f"""from pathlib import Path

path = Path({str(self.ensirna_dir / "data/dataset.py")!r})
text = path.read_text()
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
path.write_text(text)
"""

    @property
    def dataset_runtime_patch_runner(self) -> str:
        """Return a Python one-liner that applies the dataset patch."""
        return f"exec({self.dataset_runtime_patch!r})"

    @property
    def output_paths(self) -> tuple[str, ...]:
        """Return upstream-visible output paths relative to an output directory."""
        stem = self.input_stem
        return (
            f"{stem}.csv",
            f"{stem}.json",
            f"{stem}_pdb",
            f"{stem}_processed",
            f"{stem}_result.xlsx",
        )


@dataclass(frozen=True, slots=True)
class EnsirnaPdbChunkSpec:
    """One CPU Rosetta PDB preparation chunk."""

    chunk_name: str
    csv_path: str
    json_path: str
    pdb_dir: str


@dataclass(frozen=True, slots=True)
class EnsirnaPreparationPlan:
    """Volume-backed prepared-input contract for ENsiRNA inference."""

    cache_key: str
    prepared_dir: str
    json_path: str
    processed_dir: str
    result_xlsx: str
    candidate_count: int
    chunk_count: int
    chunks: list[EnsirnaPdbChunkSpec]
    cached: bool


def _cache_key_for_fasta(mrna_fasta_bytes: bytes) -> str:
    """Return a deterministic cache key for one mRNA FASTA payload."""
    return hash_string(
        "\n".join((APP_INFO.cache_namespace, mrna_fasta_bytes.decode("utf-8")))
    )


def _layout_for_cache_key(cache_key: str) -> AppRunLayout:
    """Return the shared volume layout for one prepared FASTA."""
    return AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / "cache" / cache_key[:2] / cache_key
    )


def _prepared_marker_path(layout: AppRunLayout) -> Path:
    """Return the prepared marker path for a cached ENsiRNA input."""
    return layout.markers_dir / APP_INFO.prepared_marker_name


def _required_prepared_paths(layout: AppRunLayout) -> tuple[Path, ...]:
    """Return paths required before GPU inference may start."""
    stem = APP_INFO.input_stem
    return (
        _prepared_marker_path(layout),
        layout.outputs_dir / f"{stem}.json",
        layout.outputs_dir / f"{stem}_processed" / "_metainfo",
    )


def _is_prepared(layout: AppRunLayout) -> bool:
    """Return whether the prepared-input cache is complete."""
    return all(path.exists() for path in _required_prepared_paths(layout))


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
        result_xlsx=str(layout.outputs_dir / f"{stem}_result.xlsx"),
        candidate_count=candidate_count,
        chunk_count=chunk_count,
        chunks=[],
        cached=True,
    )


def _cached_preparation_plan(
    *, cache_key: str, layout: AppRunLayout
) -> EnsirnaPreparationPlan | None:
    """Return a cached preparation plan when the prepared marker is valid."""
    if not _is_prepared(layout):
        return None
    import orjson

    marker = orjson.loads(_prepared_marker_path(layout).read_bytes())
    if marker.get("cache_key") != cache_key:
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

    layout.markers_dir.mkdir(parents=True, exist_ok=True)
    _prepared_marker_path(layout).write_bytes(
        orjson.dumps({
            "cache_key": plan.cache_key,
            "candidate_count": plan.candidate_count,
            "chunk_count": plan.chunk_count,
            "json_records": json_records,
            "json_path": plan.json_path,
            "processed_dir": plan.processed_dir,
        })
    )


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
        | {"MAMBA_ROOT_PREFIX": APP_INFO.mamba_root, "PATH": APP_INFO.mamba_bin_path}
    )
    .run_commands("curl -L micro.mamba.pm/install.sh | bash")
    .micromamba_install(
        *APP_INFO.conda_packages, channels=list(APP_INFO.conda_channels)
    )
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
    .uv_pip_install(*APP_INFO.pip_packages)
    .uv_pip_install(*APP_INFO.torch_packages, index_url=APP_INFO.torch_index_url)
    .uv_pip_install(*APP_INFO.extra_pip_packages)
    .pipe(patch_image_for_helper, ignore_dep_versions=True, skip_deps=["uniaf3"])
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


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
    run_name: str,
    max_prepare_jobs: int = 4,
    force: bool = False,
) -> EnsirnaPreparationPlan:
    """Create siRNA CSV and CPU PDB chunk work in the output volume."""
    import polars as pl

    del run_name
    cache_key = _cache_key_for_fasta(mrna_fasta_bytes)
    CONF.output_volume.reload()
    layout = _layout_for_cache_key(cache_key)
    if not force and (
        cached_plan := _cached_preparation_plan(cache_key=cache_key, layout=layout)
    ):
        return cached_plan

    if layout.run_root.exists():
        shutil.rmtree(layout.run_root)
    layout.inputs_dir.mkdir(parents=True)
    layout.outputs_dir.mkdir(parents=True)
    layout.prep_dir.mkdir(parents=True)

    mrna_fasta = layout.inputs_dir / APP_INFO.input_fasta_name
    mrna_fasta.write_bytes(mrna_fasta_bytes)
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
            str(layout.outputs_dir),
        ],
        cwd=APP_INFO.ensirna_dir,
        output_mode="capture",
    )

    stem = APP_INFO.input_stem
    csv_path = layout.outputs_dir / f"{stem}.csv"
    frame = pl.read_csv(csv_path)
    candidate_count = frame.height
    if candidate_count == 0:
        raise RuntimeError("ENsiRNA did not generate any siRNA candidates.")

    chunk_count = min(max(1, max_prepare_jobs), candidate_count)
    chunk_size = (candidate_count + chunk_count - 1) // chunk_count
    pdb_dir = layout.outputs_dir / f"{stem}_pdb"
    pdb_dir.mkdir()

    chunks = []
    for idx, offset in enumerate(range(0, candidate_count, chunk_size)):
        chunk_name = f"chunk_{idx:04d}"
        chunk_csv = layout.prep_dir / f"{chunk_name}.csv"
        frame.slice(offset, chunk_size).write_csv(chunk_csv)
        chunks.append(
            EnsirnaPdbChunkSpec(
                chunk_name=chunk_name,
                csv_path=str(chunk_csv),
                json_path=str(layout.prep_dir / f"{chunk_name}.json"),
                pdb_dir=str(pdb_dir),
            )
        )

    CONF.output_volume.commit()
    return EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=str(layout.run_root),
        json_path=str(layout.outputs_dir / f"{stem}.json"),
        processed_dir=str(layout.outputs_dir / f"{stem}_processed"),
        result_xlsx=str(layout.outputs_dir / f"{stem}_result.xlsx"),
        candidate_count=candidate_count,
        chunk_count=len(chunks),
        chunks=chunks,
        cached=False,
    )


@app.function(
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def ensirna_prepare_pdb_chunk(
    chunk: EnsirnaPdbChunkSpec, pdb_cores: int = 1
) -> dict[str, int | str]:
    """Run Rosetta PDB generation for one CSV chunk on CPU."""
    CONF.output_volume.reload()
    json_path = Path(chunk.json_path)
    if not json_path.exists():
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
                chunk.csv_path,
                "-p",
                chunk.pdb_dir,
            ],
            cwd=APP_INFO.ensirna_dir,
            env={APP_INFO.pdb_cores_env: str(max(1, pdb_cores))},
        )
        CONF.output_volume.commit()
    return {
        "chunk_name": chunk.chunk_name,
        "json_path": str(json_path),
        "cached": int(json_path.exists()),
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
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.prepared_dir)
    if cached_plan := _cached_preparation_plan(cache_key=plan.cache_key, layout=layout):
        return cached_plan

    json_path = Path(plan.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("wb") as out:
        for chunk in plan.chunks:
            chunk_json = Path(chunk.json_path)
            if not chunk_json.exists():
                raise FileNotFoundError(
                    f"ENsiRNA PDB chunk JSON not found: {chunk_json}"
                )
            data = chunk_json.read_bytes()
            if data:
                out.write(data)
                if not data.endswith(b"\n"):
                    out.write(b"\n")

    CONF.output_volume.commit()
    return EnsirnaPreparationPlan(
        cache_key=plan.cache_key,
        prepared_dir=plan.prepared_dir,
        json_path=plan.json_path,
        processed_dir=plan.processed_dir,
        result_xlsx=plan.result_xlsx,
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
) -> EnsirnaPreparationPlan:
    """Build the RNA-FM preprocessed dataset cache on GPU."""
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(plan.prepared_dir)
    if cached_plan := _cached_preparation_plan(cache_key=plan.cache_key, layout=layout):
        return cached_plan

    json_path = Path(plan.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"ENsiRNA merged JSON not found: {json_path}")

    processed_dir = Path(plan.processed_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    if not APP_INFO.rnafm_cache_path.exists():
        raise FileNotFoundError(
            f"RNA-FM checkpoint not found: {APP_INFO.rnafm_cache_path}"
        )
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
            str(json_path),
            "--save_dir",
            str(processed_dir),
        ],
        cwd=APP_INFO.ensirna_dir,
        env={APP_INFO.rnafm_device_env: "cuda"},
    )
    processed_marker = processed_dir / "_metainfo"
    if not processed_marker.exists():
        raise FileNotFoundError(
            f"ENsiRNA processed dataset marker not found: {processed_marker}"
        )

    json_data = json_path.read_bytes()
    json_records = (
        json_data.count(b"\n") + int(not json_data.endswith(b"\n")) if json_data else 0
    )
    prepared_plan = EnsirnaPreparationPlan(
        cache_key=plan.cache_key,
        prepared_dir=plan.prepared_dir,
        json_path=plan.json_path,
        processed_dir=plan.processed_dir,
        result_xlsx=plan.result_xlsx,
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
    CONF.output_volume.reload()
    layout = AppRunLayout.from_run_root(prepared_dir)
    if not _is_prepared(layout):
        missing = [
            str(path) for path in _required_prepared_paths(layout) if not path.exists()
        ]
        raise FileNotFoundError(f"ENsiRNA prepared inputs are incomplete: {missing}")

    result_xlsx = layout.outputs_dir / f"{APP_INFO.input_stem}_result.xlsx"
    if force or not result_xlsx.exists():
        _link_checkpoints()
        checkpoint_args = [
            str(APP_INFO.ensirna_dir / "pkl" / filename)
            for filename in APP_INFO.checkpoint_filenames
        ]
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
                str(layout.outputs_dir),
                "--gpu",
                "0",
                "--id",
                APP_INFO.input_stem,
            ],
            cwd=APP_INFO.ensirna_dir,
        )
        CONF.output_volume.commit()

    return package_outputs(
        layout.outputs_dir,
        paths_to_bundle=APP_INFO.output_paths,
    )


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_ensirna_task(
    mrna_fasta: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    prepare_workers: int = 4,
    pdb_cores: int = 1,
    force: bool = False,
) -> None:
    """Run ENsiRNA siRNA candidate design.

    Args:
        mrna_fasta: Local mRNA FASTA file to design siRNA candidates for.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to the mRNA FASTA
            filename stem.
        prepare_workers: Maximum number of CPU Modal workers for Rosetta PDB
            preparation chunks.
        pdb_cores: CPU cores used inside each PDB preparation worker.
        force: Rebuild prepared artifacts and rerun inference instead of using
            matching cached Modal volume outputs.
    """
    input_path = Path(mrna_fasta).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"mRNA FASTA not found: {input_path}")
    run_name = run_name or input_path.stem
    out_file = build_local_output_path(
        resolve_local_output_dir(out_dir),
        run_name=run_name,
        suffix="ensirna",
    )

    print(f"🧬 Submitting ENsiRNA run '{run_name}'")
    download_ensirna_models.remote(force=False)
    prepare_plan = ensirna_prepare_inputs.remote(
        mrna_fasta_bytes=input_path.read_bytes(),
        run_name=run_name,
        max_prepare_jobs=prepare_workers,
        force=force,
    )
    if prepare_plan.cached:
        print(f"🧬 Reusing prepared ENsiRNA inputs: {prepare_plan.prepared_dir}")
        prepared_plan = prepare_plan
    else:
        print(
            f"🧬 Preparing {prepare_plan.candidate_count} siRNAs across "
            f"{prepare_plan.chunk_count} CPU chunks"
        )

        def run_chunk(chunk: EnsirnaPdbChunkSpec) -> dict[str, int | str]:
            return ensirna_prepare_pdb_chunk.remote(chunk=chunk, pdb_cores=pdb_cores)

        bounded_map(
            prepare_plan.chunks,
            run_chunk,
            max_parallel=max(1, min(prepare_workers, len(prepare_plan.chunks))),
        )
        finalized_plan = ensirna_finalize_prepared_inputs.remote(prepare_plan)
        prepared_plan = ensirna_preprocess_dataset.remote(finalized_plan)

    tarball_bytes = run_ensirna_inference.remote(
        prepared_dir=prepared_plan.prepared_dir,
        force=force,
    )
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 ENsiRNA run complete! Results saved to {out_file}")
