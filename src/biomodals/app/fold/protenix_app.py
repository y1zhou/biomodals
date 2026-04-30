"""Protenix source repo: <https://github.com/y1zhou/Protenix>.

## Notes

* The default `--msa-server-mode protenix` uses the Protenix remote MSA server,
  so no local MSA databases are required. Switch to `colabfold` if you have a
  pre-populated database volume.
* MSA/template preprocessing is run in a CPU-only Modal function and cached in a
  persistent Modal volume before GPU inference.
* Templates are only used when `--use-template` is passed. Template support
  requires the v1.0.0 model checkpoints.
* RNA MSA is only supported by v1.0.0 model checkpoints.
* The `protenix_base_constraint_v0.5.0` model supports pocket, contact, and
  substructure constraints specified in the input JSON.
* For large structures (>2000 tokens), consider using an A100 (80GB) or H100
  GPU by setting the `GPU` environment variable.

## Outputs

* Results will be saved to the specified `--out-dir` as `<run-name>.tar.zst`.
* For prediction runs, the tarball contains predicted `.cif` structure files and
  `*_summary_confidence.json` files with pLDDT, pAE, and ranking scores.
* For `--score-only` runs, the tarball contains per-structure confidence JSON
  files produced by `protenixscore score`.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MAX_TIMEOUT, MODEL_VOLUME, MSA_CACHE_VOLUME
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.output import (
    build_local_output_path,
    ensure_output_file_available,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import (
    package_outputs,
    run_command,
    run_command_with_log,
)
from biomodals.helper.structure import struct2seq
from biomodals.helper.web import download_files

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Protenix",
    repo_url="https://github.com/y1zhou/Protenix",
    repo_commit_hash="7e1de70749910c401339dd49aa62735510c22959",
    package_name="protenix",
    version="2.0.0",
    python_version="3.11",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)


@dataclass
class AppInfo:
    """Container for app-specific configuration and constants."""

    # https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
    # CUDA version should be no greater than host CUDA version
    # "devel" image includes the full CUDA toolkit, which is required for
    # building custom LayerNorm kernels
    cuda_tag = f"{CONF.cuda_version_numeric}-devel-ubuntu24.04"

    # Volume for preprocessed MSA/template intermediates (MSA_CACHE_VOLUME)
    msa_cache_dir: str = "/protenix-msa"
    # Volume for prediction outputs (enables skip/resume across interrupted runs)
    model_dir: Path = CONF.model_dir

    # Base URL for downloading checkpoints and data caches
    # https://github.com/bytedance/Protenix/blob/main/protenix/web_service/dependency_url.py
    base_url: str = "https://protenix.tos-cn-beijing.volces.com"

    # Supported model checkpoints
    supported_models: Sequence[str] = (
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        # "protenix-v2.pt",  # TODO: keep an eye on protenix-v2
    )
    # CCD and other data caches required for inference
    data_cache: Sequence[str] = (
        "common/components.cif",
        "common/components.cif.rdkit_mol.pkl",
        "common/clusters-by-entity-40.txt",
        "common/obsolete_release_date.csv",
    )
    # Additional files needed when templates are enabled
    template_cache: Sequence[str] = (
        "common/obsolete_to_successor.json",
        "common/release_date_cache.json",
    )


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = patch_image_for_helper(
    modal.Image
    .from_registry(f"nvidia/cuda:{APP_INFO.cuda_tag}", add_python=CONF.python_version)
    .entrypoint([])  # remove verbose logging in the base image
    .apt_install("git", "build-essential", "zstd", "hmmer", "kalign", "wget")
    .env(
        CONF.default_env
        | {
            "PYTHONUNBUFFERED": "1",
            "PROTENIX_ROOT_DIR": str(CONF.model_dir),
            "PROTENIX_CHECKPOINT_DIR": str(CONF.model_dir / "checkpoint"),
        }
    )
    .uv_pip_install(
        f"{CONF.package_name}[{CONF.cuda_version}] @ "
        f"git+{CONF.repo_url}@{CONF.repo_commit_hash}"
    )
    # Trigger kernel compilation
    .run_commands(
        "python -m protenix.model.layer_norm.layer_norm",
        gpu=CONF.gpu,
        env={"LAYERNORM_TYPE": "fast_layernorm"},  # default, but just in case
    )
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights and data caches
##########################################
@app.function(
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME}, timeout=CONF.timeout
)
def download_protenix_data(
    model_name: str = "protenix_base_default_v1.0.0",
    force: bool = False,
    include_templates: bool = False,
) -> None:
    """Download Protenix model checkpoint and shared data caches.

    Args:
        model_name: Name of the model checkpoint to download.
        force: Force re-download even if files already exist.
        include_templates: Also download template-related data files.

    """
    data_root = CONF.model_dir
    files_to_download: dict[str, str | Path] = {}

    # Download common data caches
    data_caches = {
        f"{APP_INFO.base_url}/{rel_path}": data_root / rel_path
        for rel_path in APP_INFO.data_cache
    }
    files_to_download = files_to_download | data_caches

    # Download template data if requested
    if include_templates:
        template_caches = {
            f"{APP_INFO.base_url}/{rel_path}": data_root / rel_path
            for rel_path in APP_INFO.template_cache
        }
        files_to_download = files_to_download | template_caches

    # TODO: https://github.com/bytedance/Protenix/blob/main/scripts/database/download_protenix_data.sh

    # Download model checkpoint
    ckpt_url = f"{APP_INFO.base_url}/checkpoint/{model_name}.pt"
    files_to_download = files_to_download | {
        ckpt_url: data_root / "checkpoint" / f"{model_name}.pt"
    }
    download_files(
        files_to_download, force=force, progress_bar_desc="💊 Downloading Protenix data"
    )
    MODEL_VOLUME.commit()
    print("💊 Download complete")


##########################################
# Inference functions
##########################################
@app.function(timeout=CONF.timeout, volumes={APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME})
def query_protenix_msa_server(
    query_command: str, input_json_path: str, output_dir: str, msa_server_mode: str
) -> None:
    """Query the Protenix remote MSA server with the given command."""
    from uniaf3.schema import ProtenixConfig

    cmd = [
        "protenix",
        query_command,
        f"--input={input_json_path}",
        f"--out_dir={output_dir}",
        f"--msa_server_mode={msa_server_mode}",
    ]
    run_command(cmd)

    # Move the searched files out of the run_name subdir such that future
    # runs with different names could hit the same cache
    out_path = Path(output_dir)
    msa_out_dir = next(out_path.glob("*/msa"))
    run_name_dir = msa_out_dir.parent
    run_name = run_name_dir.name
    for subdir in run_name_dir.iterdir():
        subdir.rename(out_path / subdir.name)

    # Also need to update the file paths in the JSON to reflect new locations
    def _get_new_location(old_path: str | None) -> str | None:
        if old_path is None:
            return
        old_path_file = Path(old_path)
        if old_path_file.is_relative_to(run_name_dir):
            return str(out_path / old_path_file.relative_to(run_name_dir))
        return old_path

    for conf_json in out_path.glob(f"{run_name}-*.json"):
        conf = ProtenixConfig.from_file(conf_json)
        for task in conf.root:
            for seq in task.sequences:
                if seq.proteinChain is not None:
                    seq.proteinChain.unpairedMsaPath = _get_new_location(
                        seq.proteinChain.unpairedMsaPath
                    )
                    seq.proteinChain.pairedMsaPath = _get_new_location(
                        seq.proteinChain.pairedMsaPath
                    )
                if seq.rnaSequence is not None:
                    seq.rnaSequence.unpairedMsaPath = _get_new_location(
                        seq.rnaSequence.unpairedMsaPath
                    )
        conf.to_files(out_path, conf_json.stem)

    run_name_dir.rmdir()
    MSA_CACHE_VOLUME.commit()


@app.function(timeout=CONF.timeout, volumes={APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME})
def prepare_protenix_inputs(
    input_bytes: bytes,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
) -> bytes:
    """Run CPU preprocessing and cache prepared JSON + search outputs in a volume."""
    from tempfile import mkdtemp

    from uniaf3.schema import ProtenixConfig

    tmpdir = Path(mkdtemp(prefix="protenix_prep_"))  # cleaned on container exit
    tmp_json_path = tmpdir / "input.json"
    tmp_json_path.write_bytes(input_bytes)

    # `protenix prep` (inputprep) runs MSA + template + RNA MSA search.
    # It first produces `input-update-msa.json`, then (if template or RNA
    # MSA updates were actually made) renames it to `input-final-updated.json`.
    # `protenix mt` skips the RNA MSA search.
    # `protenix msa` only runs protein sequence MSA search.
    match (use_template, use_rna_msa):
        case (True, True):
            protenix_command = "prep"
            updated_suffix = "final-updated"
        case (True, False):
            protenix_command = "mt"
            updated_suffix = "final-updated"
        case (False, False):
            protenix_command = "msa"
            updated_suffix = "update-msa"
        case _:
            raise ValueError("RNA MSA without templates is not supported")

    # Load protein and RNA sequences from input JSON
    conf = ProtenixConfig.from_file(tmp_json_path)
    msa_tasks = []
    output_dirs: list[str] = []
    for task in conf.root:
        protein_seqs: list[str] = []
        rna_seqs: list[str] = []
        for seq in task.sequences:
            if (prot_chain := seq.proteinChain) is not None:
                protein_seqs.append(prot_chain.sequence)
            elif (rna_chain := seq.rnaSequence) is not None:
                rna_seqs.append(rna_chain.sequence)

        hash_key = (
            hash_string(":".join(protein_seqs + rna_seqs))
            if use_rna_msa
            else hash_string(":".join(protein_seqs))
        )
        cache_dir = (
            Path(APP_INFO.msa_cache_dir)
            / CONF.name
            / msa_server_mode
            / hash_key[:2]
            / hash_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(str(cache_dir))

        # Skip run if expected output is already cached for this input
        task_name = task.name
        if (cache_dir / f"{task_name}-{updated_suffix}.json").exists():
            continue

        ProtenixConfig([task]).to_files(cache_dir, task_name)
        MSA_CACHE_VOLUME.commit()
        msa_task = query_protenix_msa_server.spawn(
            protenix_command,
            f"{cache_dir}/{task_name}.json",
            str(cache_dir),
            msa_server_mode,
        )
        msa_tasks.append(msa_task)

    _ = modal.FunctionCall.gather(*msa_tasks)
    MSA_CACHE_VOLUME.reload()

    # Add the MSA paths back to the input JSON
    for task_idx, task in enumerate(conf.root):
        task_name = task.name
        cache_dir = Path(output_dirs[task_idx])
        updated_suffix = (
            "update-msa" if not (use_template or use_rna_msa) else "final-updated"
        )
        updated_json_path = cache_dir / f"{task_name}-{updated_suffix}.json"
        if not updated_json_path.exists():
            raise FileNotFoundError(
                f"Expected MSA output not found: {updated_json_path}"
            )
        updated_conf = ProtenixConfig.from_file(updated_json_path)
        conf.root[task_idx] = updated_conf.root[0]

    return conf.to_json().encode()


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME,
    },
)
def run_protenix(
    input_bytes: bytes,
    run_name: str,
    model_name: str = "protenix_base_default_v1.0.0",
    seeds: str = "101",
    cycle: int = 10,
    step: int = 200,
    sample: int = 5,
    dtype: str = "bf16",
    use_msa: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_tfg_guidance: bool = False,
    use_fast_layernorm: bool = True,
    extra_args: str | None = None,
    score_only: bool = False,
) -> bytes:
    """Run Protenix structure prediction or confidence scoring.

    Args:
        input_bytes: Input JSON for prediction, or PDB/CIF in `score_only` mode.
        run_name: Name for this run (used for output directory).
        prep_cache_key: Cache key from a prior prepare_protenix_inputs call.
        model_name: Model checkpoint name.
        seeds: Comma-separated random seeds.
        cycle: Pairformer cycle number.
        step: Number of diffusion steps.
        sample: Number of samples per seed.
        dtype: Inference dtype (bf16 or fp32).
        use_msa: Whether to use MSA features.
        msa_server_mode: MSA search mode (protenix or colabfold).
        use_template: Whether to use templates.
        use_rna_msa: Whether to use RNA MSA.
        use_tfg_guidance: Enable Training-Free Guidance (TFG) for refined sampling.
        use_fast_layernorm: Whether to enable the custom CUDA layernorm kernel.
        extra_args: Additional CLI arguments as a string.
        score_only: When True, score an existing PDB/CIF structure using
            ``protenixscore score`` instead of running diffusion prediction.

    Returns:
        Tarball bytes of inference outputs (CIF files + confidence JSONs)
        or scoring outputs when score_only is True.

    """
    from tempfile import mkdtemp

    run_env = os.environ.copy()
    if use_fast_layernorm:
        run_env["LAYERNORM_TYPE"] = "fast_layernorm"

    tmpdir_path = Path(mkdtemp(prefix="protenix_run_"))  # cleaned on container exit
    out_dir = tmpdir_path / run_name
    out_dir.mkdir()

    # Score an existing structure with the Protenix confidence head
    if score_only:
        # Detect CIF vs PDB format: CIF files start with 'data_' (after
        # any leading comment lines starting with '#'), PDB files with
        # record types like HEADER/ATOM/REMARK.
        input_ext = ".pdb"
        for _line in input_bytes.splitlines():
            stripped = _line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            if stripped.startswith(b"data_"):
                input_ext = ".cif"
            break
        input_file = tmpdir_path / f"{run_name}{input_ext}"
        input_file.write_bytes(input_bytes)

        # Map use_msa → --use_msas (both | false)
        # ProtenixScore's use_msas controls which chain roles receive MSAs.
        use_msas_val = "both" if use_msa else "false"

        # Map msa_server_mode → --msa_host_url.
        # The protenix remote server URL matches what `protenix msa` uses
        # (MMSEQS_SERVICE_HOST_URL in protenix/web_service/colab_request_parser.py).
        msa_host_url = (
            "https://protenix-server.com/api/msa"
            if msa_server_mode == "protenix"
            else "https://api.colabfold.com"
        )

        # Cache fetched MSAs so they can be reused across runs
        input_seqs = struct2seq(input_file)
        cache_key = hash_string(":".join(x[1] for x in input_seqs))
        score_msa_cache_dir = (
            Path(APP_INFO.msa_cache_dir)
            / f"{CONF.name}_score"
            / msa_server_mode
            / cache_key[:2]
            / cache_key
        )
        score_msa_cache_dir.mkdir(parents=True, exist_ok=True)
        # TODO: split MSA search and score steps
        cmd = [
            "protenixscore",
            "score",
            f"--input={input_file}",
            f"--output={tmpdir_path}",
            f"--model_name={model_name}",
            f"--dtype={dtype}",
            f"--use_msas={use_msas_val}",
            f"--msa_host_url={msa_host_url}",
            f"--msa_cache_dir={score_msa_cache_dir}",
            "--msa_cache_mode=readwrite",
        ]
        run_command_with_log(
            cmd,
            log_file=out_dir / f"{run_name}.log",
            verbose=True,
            env=run_env,
            cwd=tmpdir_path,
        )

        # Persist MSA cache back to the volume for reuse in future runs
        MSA_CACHE_VOLUME.commit()
        print("💊 Packaging ProtenixScore results...")
        tarball_bytes = package_outputs(out_dir)
        return tarball_bytes

    # --- Prediction mode ---
    input_json_path = tmpdir_path / f"{run_name}.json"
    input_json_path.write_bytes(input_bytes)
    cmd = [
        "protenix",
        "pred",
        f"--input={input_json_path}",
        f"--out_dir={tmpdir_path}",
        f"--seeds={seeds}",
        f"--cycle={cycle}",
        f"--step={step}",
        f"--sample={sample}",
        f"--dtype={dtype}",
        f"--model_name={model_name}",
        f"--use_msa={use_msa}",
        f"--msa_server_mode={msa_server_mode}",
        f"--use_template={use_template}",
        f"--use_rna_msa={use_rna_msa}",
        f"--use_tfg_guidance={use_tfg_guidance}",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    run_command_with_log(
        cmd,
        log_file=out_dir / f"{run_name}.log",
        verbose=True,
        env=run_env,
        cwd=tmpdir_path,
    )

    # Package outputs
    print("💊 Packaging Protenix results...")
    tarball_bytes = package_outputs(out_dir)
    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_protenix_task(
    input_file: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    model_name: str = "protenix_base_default_v1.0.0",
    seeds: str = "101",
    cycle: int = 10,
    step: int = 200,
    sample: int = 5,
    dtype: str = "bf16",
    use_msa: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
    use_tfg_guidance: bool = False,
    use_fast_layernorm: bool = True,
    force_redownload: bool = False,
    extra_args: str | None = None,
    score_only: bool = False,
) -> None:
    """Run Protenix structure prediction on Modal and fetch results to `out_dir`.

    Args:
        input_file: Path to input JSON file, or a PDB/CIF file in `score_only` mode.
            For a description of the JSON schema, see
            <https://github.com/y1zhou/Protenix/blob/main/docs/infer_json_format.md>.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to input filename stem.
        model_name: Model checkpoint name. See `APP_INFO.supported_models`
            for available models.
        seeds: Comma-separated random seeds for inference.
        cycle: Pairformer cycle number.
        step: Number of diffusion steps.
        sample: Number of samples per seed.
        dtype: Inference dtype (bf16 or fp32).
        use_msa: Whether to use MSA features. Pass `--no-use-msa` to disable.
        msa_server_mode: MSA search mode (`protenix` or `colabfold`).
        use_template: Whether to use templates. Requires Protenix data files.
        use_rna_msa: Whether to use RNA MSA features.
        use_tfg_guidance: Enable Training-Free Guidance (TFG) for refined sampling.
        use_fast_layernorm: Whether to enable the custom CUDA layernorm kernel.
        force_redownload: Whether to force re-download of model weights.
        extra_args: Additional CLI arguments passed to `protenix pred`.
        score_only: When True, score an existing PDB/CIF structure using
            ``protenixscore score`` instead of running prediction.
    """
    # Validate model name
    if model_name not in APP_INFO.supported_models:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {', '.join(APP_INFO.supported_models)}"
        )

    # Validate input and output paths
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if run_name is None:
        run_name = input_path.stem

    local_out_dir = resolve_local_output_dir(out_dir)
    out_file = build_local_output_path(
        local_out_dir,
        run_name=run_name,
        suffix=f"_{CONF.name}",
    )
    ensure_output_file_available(out_file)

    # Ensure models and data caches are available
    print(f"🧬 Checking Protenix model and data caches for {model_name}...")
    download_protenix_data.remote(
        model_name=model_name, force=force_redownload, include_templates=use_template
    )

    input_bytes = input_path.read_bytes()
    if score_only:
        # Score an existing structure; MSA fetching is handled inside run_protenix
        print(f"🧬 Scoring structure with {model_name}...")
        tarball_bytes = run_protenix.remote(
            input_bytes=input_bytes,
            run_name=run_name,
            model_name=model_name,
            dtype=dtype,
            use_msa=use_msa,
            msa_server_mode="colabfold",
            use_fast_layernorm=use_fast_layernorm,
            score_only=True,
        )
    else:
        # Preprocess (MSA/template) on CPU and cache in volume for reuse
        if use_msa or use_template or use_rna_msa:
            print("🧬 Running Protenix preprocessing and caching intermediates...")
            input_bytes = prepare_protenix_inputs.remote(
                input_bytes=input_bytes,
                msa_server_mode=msa_server_mode,
                use_template=use_template,
                use_rna_msa=use_rna_msa,
            )

        # Run inference
        print(f"🧬 Running inference with {model_name}...")
        tarball_bytes = run_protenix.remote(
            input_bytes=input_bytes,
            run_name=run_name,
            model_name=model_name,
            seeds=seeds,
            cycle=cycle,
            step=step,
            sample=sample,
            dtype=dtype,
            use_msa=use_msa,
            msa_server_mode=msa_server_mode,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
            use_tfg_guidance=use_tfg_guidance,
            use_fast_layernorm=use_fast_layernorm,
            extra_args=extra_args,
        )

    # Save results locally
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 Protenix run complete! Results saved to {out_file}")
