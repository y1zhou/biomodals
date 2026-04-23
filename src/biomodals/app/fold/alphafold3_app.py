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
from dataclasses import dataclass
from pathlib import Path

import modal
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.config import AppConfig
from biomodals.app.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MODEL_VOLUME,
    MSA_CACHE_VOLUME,
)
from biomodals.app.helper import hash_string, patch_image_for_helper
from biomodals.app.helper.shell import copy_files, package_outputs, run_command_with_log

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AlphaFold3",
    repo_url="https://github.com/google-deepmind/alphafold3",
    repo_commit_hash="87bd9e678d9acacc4aa9baa05e820f32b80e1b49",
    package_name="alphafold3",
    version="3.0.1",
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
runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
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
        " && ".join(
            (
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
            )
        )
    )
    .workdir(str(CONF.git_clone_dir))
    # .uv_sync(frozen=True, extra_options="--no-editable")
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")  # installed in the previous step
    .env({"PATH": "/hmmer/bin:$PATH"})
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Inference functions
##########################################
def _cache_conf_unpaired_msa(conf: AF3Config, msa_cache_dir: Path) -> AF3Config:
    """Cache unpaired MSA results in separate files for future reuse.

    If cache files are found, read and add them to the config object.
    """
    for seq in conf.sequences:
        if (prot_chain := seq.protein) is not None:
            # When the config does not contain MSA, and cache file exists,
            # add the MSA from cache
            seq_hash = hash_string(prot_chain.sequence)
            single_msa_path = msa_cache_dir / seq_hash[:2] / seq_hash / "single.a3m"
            if (
                single_msa_path.exists()
                and prot_chain.unpairedMsa is None
                and prot_chain.unpairedMsaPath is None
                # https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md#multiple-sequence-alignment
                and (prot_chain.pairedMsa is not None or prot_chain.pairedMsa == "")
            ):
                prot_chain.unpairedMsa = single_msa_path.read_text()
                continue

            # Skip caching if a MSA path is provided - it likely did not come
            # the AlphaFold3 data pipeline.
            if prot_chain.unpairedMsaPath is not None:
                continue

            # When the config is from the data pipeline, cache MSA results
            if not single_msa_path.exists() and prot_chain.unpairedMsa is not None:
                single_msa_path.parent.mkdir(parents=True, exist_ok=True)
                single_msa_path.write_text(prot_chain.unpairedMsa)
        elif (rna_chain := seq.rna) is not None:
            seq_hash = hash_string(rna_chain.sequence)
            single_msa_path = msa_cache_dir / seq_hash[:2] / seq_hash / "single.a3m"
            if (
                single_msa_path.exists()
                and rna_chain.unpairedMsa is None
                and rna_chain.unpairedMsaPath is None
            ):
                rna_chain.unpairedMsa = single_msa_path.read_text()
                continue
            if rna_chain.unpairedMsaPath is not None:
                continue
            if not single_msa_path.exists() and rna_chain.unpairedMsa is not None:
                single_msa_path.parent.mkdir(parents=True, exist_ok=True)
                single_msa_path.write_text(rna_chain.unpairedMsa)
    return conf


def _get_cache_key(conf: AF3Config, separator: str = ":") -> str:
    """Get cache key for given protein and RNA sequences."""
    protein_seqs: list[str] = []
    rna_seqs: list[str] = []
    for seq in conf.sequences:
        if (prot_chain := seq.protein) is not None:
            protein_seqs.append(prot_chain.sequence)
        elif (rna_chain := seq.rna) is not None:
            rna_seqs.append(rna_chain.sequence)
    return hash_string(separator.join(protein_seqs + rna_seqs))


@app.function(
    cpu=(8.125, 32.125),  # 8c per database searched with HMMER
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
def run_data_pipeline(json_bytes: bytes) -> Path:
    """Run AlphaFold3 data pipeline (CPU-only)."""
    import sys
    from tempfile import mkdtemp

    temp_dir: Path = Path(mkdtemp(prefix="alphafold3_data_"))
    json_path = temp_dir / "input.json"
    json_path.write_bytes(json_bytes)

    # Determine cache key for input
    conf = AF3Config.from_file(json_path)
    hash_key = _get_cache_key(conf)
    cache_base_dir = Path(APP_INFO.msa_cache_dir) / CONF.name
    cache_dir = cache_base_dir / hash_key[:2] / hash_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fill existing unpaired MSA with cached results
    conf = _cache_conf_unpaired_msa(conf, cache_base_dir)

    # Check cache_dir for existing results and return early if found
    run_name = conf.name
    cache_data_file = cache_dir / run_name / f"{run_name}_data.json"
    if cache_data_file.exists():
        msa_conf = AF3Config.from_file(cache_data_file)
        _ = _cache_conf_unpaired_msa(msa_conf, cache_base_dir)
        return cache_data_file

    # Copy volume db files to /tmp for faster access (~651.7GiB)
    # TODO: test sharded DB
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

    # TODO: more performant runs when multiple inputs share same chains
    # https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md
    input_json_path = cache_dir / f"{run_name}.json"
    conf.to_files(cache_dir, run_name)
    MSA_CACHE_VOLUME.commit()

    cmd = [
        sys.executable,
        str(CONF.git_clone_dir / "run_alphafold.py"),
        "--run_inference=false",
        f"--json_path={input_json_path}",
        f"--output_dir={cache_dir}",
        f"--model_dir={CONF.model_dir}",
        f"--db_dir={db_dir}",
        f"--db_dir={msa_db_path}",  # fallback for mmCIF templates and RNA
        "--jackhmmer_n_cpu=8",
        "--nhmmer_n_cpu=8",
    ]
    run_command_with_log(cmd, log_file=cache_dir / f"{conf.name}.log")

    # Cache unpaired MSA files in separate directories for future use
    msa_conf = _cache_conf_unpaired_msa(
        AF3Config.from_file(cache_data_file), cache_base_dir
    )
    return cache_data_file


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME,  # JAX cache
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.read_only(),
    },
)
def run_inference_pipeline(json_path: Path, recycle: int, sample: int) -> bytes:
    """Run AlphaFold3 structure prediction.

    Returns:
        Tarball bytes of inference outputs (CIF files + confidence JSONs).

    """
    import sys
    from tempfile import TemporaryDirectory

    from uniaf3.schema.alphafold3 import AF3Config

    run_name = AF3Config.from_file(json_path).name
    with TemporaryDirectory(prefix=f"alphafold3_inference_{run_name}_") as temp_dir:
        out_dir = Path(temp_dir) / run_name
        cmd = [
            sys.executable,
            str(CONF.git_clone_dir / "run_alphafold.py"),
            "--run_inference=true",
            "--run_data_pipeline=false",
            f"--json_path={json_path}",
            f"--output_dir={out_dir}",
            f"--model_dir={CONF.model_dir}",
            f"--jax_compilation_cache_dir={CONF.model_dir / 'jax_cache'}",
            f"--num_recycles={recycle}",
            f"--num_diffusion_samples={sample}",
        ]
        run_command_with_log(cmd, log_file=out_dir / f"{run_name}_inference.log")
        return package_outputs(out_dir / run_name)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_alphafold3_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    search_msa: bool = True,
    recycle: int = 10,
    sample: int = 5,
) -> None:
    """Run AlphaFold3 on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file.
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to input filename stem)
        search_msa: Whether to run MSA and template search data pipeline.
        recycle: Number of Pairformer recycles to use during inference.
        sample: Number of diffusion samples to generate per seed.
    """
    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if run_name is None:
        run_name = input_path.stem

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    # Run inference
    if search_msa:
        print(f"🧬 Running {CONF.name} data pipeline...")
        json_bytes = input_path.read_bytes()
        json_path = run_data_pipeline.remote(json_bytes)
    else:
        # Upload the input JSON as-is when skipping the data pipeline
        cache_key = _get_cache_key(AF3Config.from_file(input_path))
        with CONF.get_out_volume().batch_upload() as batch:
            batch.put_file(
                input_path,
                f"/{CONF.name}/{cache_key[:2]}/{cache_key}/{input_path.name}",
            )
        json_path = (
            Path(APP_INFO.msa_cache_dir)
            / CONF.name
            / cache_key[:2]
            / cache_key
            / input_path.name
        )

    print(f"🧬 Running {CONF.name} inference pipeline...")
    tarball_bytes = run_inference_pipeline.remote(
        json_path, recycle=recycle, sample=sample
    )

    # Save results locally
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
