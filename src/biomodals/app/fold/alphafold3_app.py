"""AlphaFold3 source repo: <https://github.com/google-deepmind/alphafold3>.

## Additional notes

This script only provides a runtime for AlphaFold3.
To acquire the model weights and MSA databases, please follow instructions at:

<https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>

Make sure the model checkpoint is available at `/AlphaFold3/af3.bin` in the `biomodals-store` volume,
and the MSA databases are available at the `AlphaFold3-msa-db` volume.

See <https://github.com/google-deepmind/alphafold3/tree/main/docs> for general docs.

## Outputs

See <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import modal
import orjson
from uniaf3.schema.alphafold3 import (
    AF3RNA,
    AF3Config,
    AF3Protein,
    AF3SequenceEntry,
    AF3Template,
)

from biomodals.app.config import AppConfig
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MODEL_VOLUME,
    MSA_CACHE_VOLUME,
)
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import (
    copy_files,
    package_outputs,
    run_command,
    run_command_with_log,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AlphaFold3",
    repo_url="https://github.com/y1zhou/alphafold3",
    repo_commit_hash="987ad1cb7d7028b6d35908cf63fe7d951d98d6b6",
    package_name="alphafold3",
    version="3.0.2",
    python_version="3.12",
    cuda_version="cu130",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "21600")),
)


@dataclass
class AppInfo:
    """Container for AlphaFold3-specific information and configurations."""

    # Volume mount path for genetic search databases
    msa_db_dir: str = f"/{CONF.name}-msa-db"
    # Volume mount path for MSA output cache
    msa_cache_dir: str = "/biomodals-msa-cache"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()

# Ref: https://github.com/google-deepmind/alphafold3/blob/main/docker/Dockerfile
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "zstd", "zlib1g-dev", "wget")
    .env(
        CONF.default_env
        | {
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .run_commands(
        " && ".join((
            # Clone AlphaFold3 repo
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            # Download, check hash, and extract HMMER
            "mkdir /hmmer_build",
            "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build",
            "cd /hmmer_build",
            "echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz' | sha256sum --check",
            "tar zxf hmmer-3.4.tar.gz",
            "rm hmmer-3.4.tar.gz",
            # Apply the --seq_limit patch to HMMER
            "cd /hmmer_build",
            f"patch -p0 < {CONF.git_clone_dir}/docker/jackhmmer_seq_limit.patch",
            # Build and install HMMER
            "cd /hmmer_build/hmmer-3.4",
            "./configure --prefix=/hmmer",
            "make -j",
            "make install",
            "cd /hmmer_build/hmmer-3.4/easel",
            "make install",
            "rm -rf /hmmer_build",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    # .uv_sync(frozen=True, extra_options="--no-editable")
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")  # installed in the previous step
    .env({"PATH": "/hmmer/bin:$PATH"})
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Helper functions
##########################################
def _load_conf_from_bytes(json_bytes: bytes) -> AF3Config:
    """Load AlphaFold3 config from JSON bytes."""
    with TemporaryDirectory() as temp_dir:
        f = Path(temp_dir) / "config.json"
        f.write_bytes(json_bytes)
        return AF3Config.from_file(f)


def _load_msa_cache(seq: AF3Protein | AF3RNA, msa_cache_dir: Path) -> Path:
    """Fetch MSA from cache if available."""
    seq_hash = hash_string(seq.sequence)
    seq_msa_cache_dir = msa_cache_dir / seq_hash[:2] / seq_hash
    unpaired_msa_path = seq_msa_cache_dir / "unpaired.a3m"
    if isinstance(seq, AF3Protein):
        # If either MSA path is not empty, skip cache lookup because the
        # data likely does not come from AF3 query data_pipeline
        if seq.pairedMsaPath is not None or seq.unpairedMsaPath is not None:
            return seq_msa_cache_dir

        # When the config does not contain MSA, and cache file exists,
        # add the MSA from cache
        if unpaired_msa_path.exists() and seq.unpairedMsa is None:
            seq.unpairedMsa = unpaired_msa_path.read_text()

        paired_msa_path = seq_msa_cache_dir / "paired.a3m"
        if paired_msa_path.exists() and seq.pairedMsa is None:
            seq.pairedMsa = paired_msa_path.read_text()

        template_json_path = seq_msa_cache_dir / "templates.json"
        if template_json_path.exists() and seq.templates is None:
            templates = orjson.loads(template_json_path.read_bytes())
            seq.templates = [AF3Template(**t) for t in templates]

        # https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#multiple-sequence-alignment
        # If only one of the MSA is set, remove them as a rerun is needed anyway
        # We only check for None cases because it's ok to have unpaired MSA filled,
        # and paired MSA being explicitly empty ("")
        if not (seq.unpairedMsa is not None and seq.pairedMsa is not None):
            seq.unpairedMsa = None
            seq.pairedMsa = None
    elif isinstance(seq, AF3RNA):
        # RNA does not have paired MSA, so checking unpaired MSA is sufficient
        if seq.unpairedMsaPath is not None:
            return seq_msa_cache_dir
        if unpaired_msa_path.exists() and seq.unpairedMsa is None:
            seq.unpairedMsa = unpaired_msa_path.read_text()
    else:
        raise TypeError(f"Expected AF3Protein or AF3RNA, got {type(seq)}")

    return seq_msa_cache_dir


def _save_msa_cache(seq: AF3Protein | AF3RNA, seq_msa_cache_dir: Path) -> None:
    """Save MSA results to cache files."""
    seq_msa_cache_dir.mkdir(parents=True, exist_ok=True)
    unpaired_msa_path = seq_msa_cache_dir / "unpaired.a3m"
    if not unpaired_msa_path.exists() and seq.unpairedMsa is not None:
        unpaired_msa_path.write_text(seq.unpairedMsa)
    if isinstance(seq, AF3Protein):
        if seq.pairedMsa is not None:
            paired_msa_path = seq_msa_cache_dir / "paired.a3m"
            paired_msa_path.write_text(seq.pairedMsa)
        if (tmpl := seq.templates) is not None and tmpl:
            import orjson

            templates_path = seq_msa_cache_dir / "templates.json"
            templates_path.write_bytes(orjson.dumps(tmpl))


def _cache_conf_unpaired_msa(conf: AF3Config, msa_cache_dir: Path) -> AF3Config:
    """Cache MSA results in separate files for future reuse.

    If cache files are found, read and add them to the config object.
    """
    for seq in conf.sequences:
        if (prot_chain := seq.protein) is not None:
            prot_msa_dir = _load_msa_cache(prot_chain, msa_cache_dir)
            # When the config is from the data pipeline, cache MSA results
            _save_msa_cache(prot_chain, prot_msa_dir)
        elif (rna_chain := seq.rna) is not None:
            rna_msa_dir = _load_msa_cache(rna_chain, msa_cache_dir)
            _save_msa_cache(rna_chain, rna_msa_dir)
    return conf


def _af3_sanitised_name(name: str) -> str:
    """Return sanitised version of the name that can be used as a filename."""
    import string

    spaceless_name = name.replace(" ", "_")
    allowed_chars = set(string.ascii_letters + string.digits + "_-.")
    return "".join(x for x in spaceless_name if x in allowed_chars)


##########################################
# Inference functions
##########################################
@app.function(
    cpu=(0.125, 32.125),  # 8c per database searched with HMMER
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    # Protein sequences: 304.8GiB
    # RNA sequences: 88.6GiB
    # mmCIF templates .tar.zst: 57.6GiB
    # ephemeral_disk=1024 * round(304.8 + 5),  # MiB, billed by memory at 20:1 ratio
    timeout=CONF.timeout,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME,
        APP_INFO.msa_db_dir: AF3_MSA_DB_VOLUME,
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME,
    },
)
def run_data_pipeline(json_bytes: bytes, copy_msa_to_ssd: bool = True) -> bytes:
    """Run AlphaFold3 data pipeline (CPU-only)."""
    import sys
    from tempfile import mkdtemp

    # Try to fill config with cached MSA
    msa_cache_dir = Path(APP_INFO.msa_cache_dir) / CONF.name
    conf = _load_conf_from_bytes(json_bytes)
    MSA_CACHE_VOLUME.reload()
    conf = _cache_conf_unpaired_msa(conf, msa_cache_dir)
    MSA_CACHE_VOLUME.commit()

    # Check if all protein/RNA sequences have MSA results
    temp_dir: Path = Path(mkdtemp(prefix="alphafold3_data_"))
    run_name = _af3_sanitised_name(conf.name)
    input_json_path = temp_dir / f"{run_name}.json"
    conf.to_files(temp_dir, run_name)
    all_protein_msa_filled = all(
        prot_seq.unpairedMsa is not None and prot_seq.pairedMsa is not None
        for seq in conf.sequences
        if (prot_seq := seq.protein) is not None
    )
    all_rna_msa_filled = all(
        rna_seq.unpairedMsa is not None
        for seq in conf.sequences
        if (rna_seq := seq.rna) is not None
    )
    if all_protein_msa_filled and all_rna_msa_filled:
        print("💊 MSA cache hit, returning results...")
        return input_json_path.read_bytes()

    # TODO: test sharded DB
    # https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md
    # Copy volume db files to /tmp for faster access (~651.7GiB)
    print("💊 MSA cache not hit, running data pipeline...")
    msa_db_dir: list[str] = []
    if copy_msa_to_ssd:
        db_dir = temp_dir / "db"
        db_dir.mkdir()
        msa_db_path = Path(APP_INFO.msa_db_dir)
        db_files = [
            # Protein sequence databases
            "bfd-first_non_consensus_sequences.fasta",
            "uniref90_2022_05.fa",
            "uniprot_all_2021_04.fa",
            "mgy_clusters_2022_05.fa",
            "pdb_seqres_2022_09_28.fasta",
            # RNA sequence databases
            # "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
            # "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
            # "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        ]

        print(f"💊 Copying database files to local SSD {db_dir}")
        # p = run_background_command(
        #     f"tar -I zstd -xf {msa_db_path / 'pdb_2022_09_28_mmcif_files.tar.zst'} -C {db_dir}"
        # )
        copy_files({msa_db_path / db_file: db_dir / db_file for db_file in db_files})
        # p.wait()
        # if p.returncode != 0:
        #     raise RuntimeError("Failed to extract mmCIF template files")
        msa_db_dir.append(str(db_dir))
    msa_db_dir.append(APP_INFO.msa_db_dir)  # fallback for mmCIF templates and RNA

    cmd = [
        sys.executable,
        str(CONF.git_clone_dir / "run_alphafold.py"),
        "--run_inference=false",
        f"--json_path={input_json_path}",
        f"--output_dir={temp_dir}",
        f"--model_dir={CONF.model_dir}",
        *(f"--db_dir={d}" for d in msa_db_dir),
        "--jackhmmer_n_cpu=8",
        "--nhmmer_n_cpu=8",
    ]
    run_command(cmd, verbose=True)

    # Cache unpaired MSA files in separate directories for future use
    msa_json_path = temp_dir / run_name / f"{run_name}_data.json"
    if not msa_json_path.exists():
        print([x.relative_to(temp_dir) for x in temp_dir.rglob("*")])
        raise FileNotFoundError(f"MSA JSON file not found: {msa_json_path}")
    _ = _cache_conf_unpaired_msa(AF3Config.from_file(msa_json_path), msa_cache_dir)
    MSA_CACHE_VOLUME.commit()
    return msa_json_path.read_bytes()


def search_msa_and_templates(
    config_path: str | Path, search_chains_in_parallel: bool
) -> bytes:
    """Manage AlphaFold3 data pipeline(s)."""
    conf = AF3Config.from_file(config_path)
    msa_chains: list[tuple[int, AF3SequenceEntry]] = [
        (i, chain)
        for i, chain in enumerate(conf.sequences)
        if chain.protein is not None or chain.rna is not None
    ]

    if not search_chains_in_parallel:
        msa_json_bytes = run_data_pipeline.remote(
            json_bytes=Path(config_path).read_bytes(),
            copy_msa_to_ssd=len(msa_chains) > 1,
        )
        return msa_json_bytes

    # Parallelize MSA search by chains
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_pipeline_futures = []
        for i, msa_chain in msa_chains:
            input_conf = conf.model_copy(update={"sequences": [msa_chain]})
            input_conf.to_files(tmp_path, str(i))
            data_pipeline_futures.append(
                run_data_pipeline.spawn(
                    json_bytes=(tmp_path / f"{i}.json").read_bytes(),
                    copy_msa_to_ssd=False,
                )
            )
    msa_bytes = modal.FunctionCall.gather(*data_pipeline_futures)

    # Merge into one AF3Config
    for i, _ in msa_chains:
        msa_conf = _load_conf_from_bytes(msa_bytes[i])
        conf.sequences[i] = msa_conf.sequences[0]

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        conf.to_files(tmp_path, conf.name)
        return (tmp_path / f"{conf.name}.json").read_bytes()


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),  # burst for tar compression
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME,  # JAX cache
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.read_only(),
    },
)
def run_inference_pipeline(
    json_bytes: bytes, recycle: int, sample: int, model_seeds: list[int]
) -> bytes:
    """Run AlphaFold3 structure prediction.

    Returns:
        Tarball bytes of inference outputs (CIF files + confidence JSONs).

    """
    import sys

    with TemporaryDirectory(prefix="alphafold3_inference_") as temp_dir:
        temp_path = Path(temp_dir)
        input_json_path = temp_path / "input.json"
        input_json_path.write_bytes(json_bytes)

        conf = AF3Config.from_file(input_json_path)
        run_name = conf.name
        conf.modelSeeds = model_seeds
        conf.to_files(temp_path, "input")
        print(f"💊 Running inference for {run_name} with seeds {model_seeds}")

        out_dir = temp_path / run_name
        cmd = [
            sys.executable,
            str(CONF.git_clone_dir / "run_alphafold.py"),
            "--run_inference=true",
            "--run_data_pipeline=false",
            f"--json_path={input_json_path}",
            f"--output_dir={out_dir}",
            f"--model_dir={CONF.model_dir}",
            f"--jax_compilation_cache_dir={CONF.model_dir / 'jax_cache'}",
            f"--num_recycles={recycle}",
            f"--num_diffusion_samples={sample}",
        ]
        run_command_with_log(
            cmd, log_file=out_dir / f"{run_name}_inference.log", verbose=True
        )
        return package_outputs(out_dir / run_name)


def predict_structures(
    conf: AF3Config,
    local_out_dir: Path,
    recycle: int,
    sample: int,
    num_containers: int,
    *,
    poll_timeout: int = 5,
) -> Path:
    """Run AF3 inference pipeline and save outputs to .tar.zst file."""
    run_name = conf.name
    out_file = build_local_output_path(local_out_dir, run_name=run_name)
    if out_file.exists():
        print(f"🧬 File already exists, skipping inference: {out_file}")
        return out_file

    # Directly run inference pipeline if only one container is specified
    json_bytes = conf.to_json().encode()
    model_seeds = conf.modelSeeds
    if num_containers == 1:
        tarball_content = run_inference_pipeline.remote(
            json_bytes, recycle=recycle, sample=sample, model_seeds=model_seeds
        )
        write_local_tarball(out_file, tarball_content)
        return out_file

    tar_binary = shutil.which("tar") or None
    if tar_binary is None:
        raise RuntimeError("🧬 tar command not found")
    tar_cmd = [tar_binary, "-I", "zstd"]

    def _part_file(i: int) -> Path:
        return local_out_dir / f"{run_name}_part{i}.tar.zst"

    def _is_good_tarball(tarball_file: Path) -> bool:
        """Return whether an existing tarball is good enough to skip."""
        if not tarball_file.exists() or tarball_file.stat().st_size == 0:
            return False
        try:
            run_command([*tar_cmd, "-tf", str(tarball_file)], verbose=False)
        except Exception as exc:
            print(
                f"🧬 Existing part tarball is not readable; rerunning {tarball_file}: {exc}"
            )
            return False
        return True

    # Run inference in parallel for parts that are missing
    inference_func_calls: dict[int, modal.FunctionCall] = {}
    good_part_indices: set[int] = set()
    for i in range(num_containers):
        tarball_file = _part_file(i)
        if _is_good_tarball(tarball_file):
            good_part_indices.add(i)
            continue
        fc = run_inference_pipeline.spawn(
            json_bytes,
            recycle=recycle,
            sample=sample,
            model_seeds=model_seeds[i::num_containers],
        )
        inference_func_calls[i] = fc

    # Collect results as they become available
    failures: list[tuple[int, Exception]] = []
    while inference_func_calls:
        for i, fc in inference_func_calls.copy().items():
            try:
                tarball_content = fc.get(timeout=poll_timeout)
            except TimeoutError:
                print(f"🧬 Task {i} still running...")
                continue
            except Exception as exc:
                failures.append((i, exc))
                del inference_func_calls[i]
                print(f"🧬 Task {i} failed: {exc}")
                continue

            tarball_file = _part_file(i)
            tmp_file = tarball_file.with_suffix(".tmp")
            write_local_tarball(tmp_file, tarball_content, overwrite=True)
            tmp_file.replace(tarball_file)
            del inference_func_calls[i]

    # Go through all expected tarball part files
    tarball_part_files = [_part_file(i) for i in range(num_containers)]
    for i, tarball_part_file in enumerate(tarball_part_files):
        if i not in good_part_indices and _is_good_tarball(tarball_part_file):
            good_part_indices.add(i)
    unusable_part_files = [
        p for i, p in enumerate(tarball_part_files) if i not in good_part_indices
    ]
    if unusable_part_files:
        saved = (
            ", ".join(str(tarball_part_files[i]) for i in sorted(good_part_indices))
            or "none"
        )
        failed = "; ".join(f"part {i}: {exc}" for i, exc in failures) or "unknown"
        raise RuntimeError(
            "Some AlphaFold3 inference parts failed or did not produce readable "
            "tarballs. "
            f"Saved part tarballs: {saved}. Failed parts: {failed}. "
            "Rerun the command to resume only missing parts."
        )

    # Run local extraction after everything is saved to avoid errors
    with TemporaryDirectory() as tmp_dir:
        for tar_filename in tarball_part_files:
            run_command(
                [*tar_cmd, "-xf", str(tar_filename)], verbose=False, cwd=tmp_dir
            )

        # Combine the parts into a single .tar.zst file
        tarball_content = package_outputs(Path(tmp_dir) / run_name)
        write_local_tarball(out_file, tarball_content)
    print(
        f"🧬 Note that top-level {run_name}_*.{{cif,json,csv}} may not be correct since they are from parallel workers"
    )
    return out_file


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_alphafold3_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    search_msa: bool = True,
    search_chains_in_parallel: bool = True,
    max_num_gpus: int = 1,
    recycle: int = 10,
    sample: int = 5,
) -> None:
    """Run AlphaFold3 on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file.
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to `name` in the AF3 JSON config)
        search_msa: Whether to run MSA and template search data pipeline.
        search_chains_in_parallel: Whether to spawn multiple `data_pipeline`
            jobs when there is more than one protein/RNA chain to query MSA.
            If True, a 32-core job will be spawned for *each* chain. If False,
            a single container will be used for all chains sequentially.
        max_num_gpus: Maximum number of GPUs to use during inference. If >1,
            multiple `model_inference` jobs will be spawned in parallel based
            on the number of model seeds in the JSON config.
        recycle: Number of Pairformer recycles to use during inference.
        sample: Number of diffusion samples to generate per seed.

    """
    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    conf = AF3Config.from_file(input_path)
    if run_name is None:
        run_name = conf.name
    conf.name = run_name

    # Run inference
    if search_msa:
        print(f"🧬 Running {CONF.name} data pipeline...")
        json_bytes = search_msa_and_templates(input_path, search_chains_in_parallel)
    else:
        json_bytes = input_path.read_bytes()

    local_out_dir = resolve_local_output_dir(out_dir)

    new_conf = _load_conf_from_bytes(json_bytes)
    new_conf.name = run_name
    new_conf.modelSeeds = conf.modelSeeds
    num_seeds = len(new_conf.modelSeeds)
    num_containers = max(1, min(max_num_gpus, num_seeds))
    print(f"🧬 Running {CONF.name} inference pipeline with {num_containers=}...")
    out_file = predict_structures(
        new_conf, local_out_dir, recycle, sample, num_containers
    )
    print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
