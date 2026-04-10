"""Protenix source repo: <https://github.com/y1zhou/Protenix>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-json` | **Required** | Path to input JSON file (or PDB/CIF when `--score-only`). For a description of the JSON schema, see https://github.com/y1zhou/Protenix/blob/main/docs/infer_json_format.md. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved in the current working directory. |
| `--run-name` | stem name of `--input-json` | Optional run name used to name output files. |
| `--model-name` | `protenix_base_default_v1.0.0` | Model checkpoint name. Supported models: `protenix_base_default_v1.0.0`, `protenix_base_20250630_v1.0.0`. |
| `--seeds` | `"101"` | Comma-separated random seeds for inference. |
| `--cycle` | `10` | Pairformer cycle number. |
| `--step` | `200` | Number of diffusion steps. |
| `--sample` | `5` | Number of samples per seed. |
| `--dtype` | `"bf16"` | Inference dtype (`bf16` or `fp32`). |
| `--use-msa`/`--no-use-msa` | `--use-msa` | Whether to use MSA features. |
| `--msa-server-mode` | `protenix` | MSA search mode: `protenix` (remote server, no local DB needed) or `colabfold`. |
| `--use-template`/`--no-use-template` | `--no-use-template` | Whether to search for and use templates. |
| `--use-rna-msa`/`--no-use-rna-msa` | `--no-use-rna-msa` | Whether to use RNA MSA features. |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download model weights and data caches, then exit without running inference. |
| `--force-redownload`/`--no-force-redownload` | `--no-force-redownload` | Whether to force re-download even if files already exist. |
| `--score-only` | `False` | If True, score an existing structure with the Protenix confidence head via `protenixscore score` instead of running prediction. The `--input-json` argument should be a PDB or CIF file path. |
| `--extra-args` | `None` | Additional CLI arguments passed directly to `protenix pred` as a single string. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `Protenix` | Name of the Modal app to use. |
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `3600` | Timeout for each Modal function in seconds. |

## Additional notes

* The default `--msa-server-mode protenix` uses the Protenix remote MSA server,
  so no local MSA databases are required. Switch to `colabfold` if you have a
  pre-populated database volume.
* MSA/template preprocessing is run in a CPU-only Modal function and cached in a
  persistent Modal volume (`protenix-msa`) before GPU inference.
* Templates are only used when `--use-template` is passed. Template support
  requires the v1.0.0 model checkpoints.
* RNA MSA is only supported by v1.0.0 model checkpoints.
* The `protenix_base_constraint_v0.5.0` model supports pocket, contact, and
  substructure constraints specified in the input JSON.
* For large structures (>2000 tokens), consider using an A100 (80GB) or H100
  GPU by setting the `GPU` environment variable.
* Prediction outputs are cached in a persistent Modal volume (`protenix-outputs`)
  keyed by run name and inference parameters. Interrupted runs resume from the
  last completed seed automatically.

## Outputs

* Results will be saved to the specified `--out-dir` as
  `<run-name>_protenix_outputs.tar.zst`.
* For prediction runs, the tarball contains predicted `.cif` structure files and
  `*_summary_confidence.json` files with pLDDT, pAE, and ranking scores.
* For `--score-only` runs, the tarball contains per-structure confidence JSON
  files produced by `protenixscore score`.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shlex
import shutil
from hashlib import sha256
from pathlib import Path

from modal import App, Image

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME, MSA_CACHE_VOLUME
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import package_outputs, run_command
from biomodals.app.helper.web import download_files

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    name="Protenix",
    repo_url="https://github.com/y1zhou/Protenix",
    repo_commit_hash="7e1de70749910c401339dd49aa62735510c22959",
    package_name="protenix",
    version="2.0.0",
    python_version="3.11",
    cuda_version="cu130",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "86400")),
)

# Volume for preprocessed MSA/template intermediates (MSA_CACHE_VOLUME)
MSA_CACHE_DIR = "/protenix-msa"

# Volume for prediction outputs (enables skip/resume across interrupted runs)
OUTPUTS_VOLUME_NAME, OUTPUTS_VOLUME = CONF.out_volume()
OUTPUTS_DIR = CONF.output_volume_mountpoint
MODEL_DIR = CONF.model_dir

# Base URL for downloading checkpoints and data caches
# https://github.com/bytedance/Protenix/blob/main/protenix/web_service/dependency_url.py
PROTENIX_DOWNLOAD_BASE = "https://protenix.tos-cn-beijing.volces.com"

# Supported model checkpoints

SUPPORTED_MODELS = (
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    # "protenix-v2.pt",  # TODO: keep an eye on protenix-v2
)

# CCD and other data caches required for inference
DATA_CACHE_FILES = (
    "common/components.cif",
    "common/components.cif.rdkit_mol.pkl",
    "common/clusters-by-entity-40.txt",
    "common/obsolete_release_date.csv",
)

# Additional files needed when templates are enabled
TEMPLATE_CACHE_FILES = (
    "common/obsolete_to_successor.json",
    "common/release_date_cache.json",
)

##########################################
# Image and app definitions
##########################################

# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = CONF.cuda_version_numeric  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

runtime_image = patch_image_for_helper(
    Image.from_registry(f"nvidia/cuda:{tag}", add_python=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "zstd",
        "hmmer",
        "kalign",
        "wget",
    )
    .env(
        CONF.default_env
        | {
            "PYTHONUNBUFFERED": "1",
            "PROTENIX_ROOT_DIR": str(CONF.model_dir),
            "PROTENIX_CHECKPOINT_DIR": str(CONF.model_dir / "checkpoint"),
        }
    )
    .uv_pip_install(f"git+{CONF.repo_url}@{CONF.repo_commit_hash}[{CONF.cuda_version}]")
    # Trigger kernel compilation
    .run_commands(
        "python -m protenix.model.layer_norm.layer_norm",
        gpu=CONF.gpu,
        env={"LAYERNORM_TYPE": "fast_layernorm"},  # default, but just in case
    )
)
app = App(CONF.name, image=runtime_image)


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
        f"{PROTENIX_DOWNLOAD_BASE}/{rel_path}": data_root / rel_path
        for rel_path in DATA_CACHE_FILES
    }
    files_to_download = files_to_download | data_caches

    # Download template data if requested
    if include_templates:
        template_caches = {
            f"{PROTENIX_DOWNLOAD_BASE}/{rel_path}": data_root / rel_path
            for rel_path in TEMPLATE_CACHE_FILES
        }
        files_to_download = files_to_download | template_caches

    # TODO: https://github.com/bytedance/Protenix/blob/main/scripts/database/download_protenix_data.sh

    # Download model checkpoint
    ckpt_url = f"{PROTENIX_DOWNLOAD_BASE}/checkpoint/{model_name}.pt"
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
# TODO: fix cache keys (preferrably re-use uniaf3 code)
@app.function(
    timeout=CONF.timeout,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME,
        MSA_CACHE_DIR: MSA_CACHE_VOLUME,
    },
)
def prepare_protenix_inputs(
    json_str: bytes,
    use_msa: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
) -> str:
    """Run CPU preprocessing and cache prepared JSON + search outputs in a volume.

    Command selection:
    - ``use_template=True``:  ``protenix prep`` (MSA + template + RNA MSA);
      output is ``input-final-updated.json``, falling back to
      ``input-update-msa.json`` if no template/RNA changes were applied.
    - ``use_template=False``: ``protenix msa`` (MSA search only);
      output is ``input-update-msa.json``.

    """
    h = sha256()
    h.update(json_str)
    h.update(b"\x00")
    h.update(f"{use_msa}|{msa_server_mode}|{use_template}|{use_rna_msa}".encode())
    cache_key = h.hexdigest()
    cache_dir = Path(MSA_CACHE_DIR) / CONF.name / cache_key[:2] / cache_key
    prepared_json_path = cache_dir / "prepared.json"
    if prepared_json_path.exists():
        print(f"💊 Reusing cached preprocessed inputs: {cache_key}")
        return cache_key

    cache_dir.mkdir(parents=True, exist_ok=True)
    input_json_path = cache_dir / "input.json"
    input_json_path.write_bytes(json_str)
    # The stem "input" is used by Protenix to derive output filenames such as
    # "input-update-msa.json" and "input-final-updated.json".
    input_stem = input_json_path.stem

    if use_msa or use_template or use_rna_msa:
        if use_template:
            # `protenix prep` (inputprep) runs MSA + template + RNA MSA search.
            # It first produces `input-update-msa.json`, then (if template or RNA
            # MSA updates were actually made) renames it to `input-final-updated.json`.
            cmd = [
                "protenix",
                "prep",
                "--input",
                str(input_json_path),
                "--out_dir",
                str(cache_dir),
                "--msa_server_mode",
                msa_server_mode,
            ]
            run_command(cmd, cwd=cache_dir)

            # Prefer the fully-updated file; fall back to MSA-only update.
            expected_files = [
                cache_dir / f"{input_stem}-final-updated.json",
                cache_dir / f"{input_stem}-update-msa.json",
            ]
        else:
            # `protenix msa` runs MSA search only and produces `input-update-msa.json`.
            cmd = [
                "protenix",
                "msa",
                "--input",
                str(input_json_path),
                "--out_dir",
                str(cache_dir),
                "--msa_server_mode",
                msa_server_mode,
            ]
            run_command(cmd, cwd=cache_dir)

            expected_files = [
                cache_dir / f"{input_stem}-update-msa.json",
            ]

        found_output_json: Path | None = next(
            (p for p in expected_files if p.exists()), None
        )
        if found_output_json is None:
            raise RuntimeError(
                f"Expected output JSON not found for cache key {cache_key}. "
                f"Looked for: {[p.name for p in expected_files]}"
            )
        shutil.copyfile(found_output_json, prepared_json_path)
    else:
        shutil.copyfile(input_json_path, prepared_json_path)

    MSA_CACHE_VOLUME.commit()
    print(f"💊 Cached preprocessed inputs: {cache_key}")
    return cache_key


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
        MSA_CACHE_DIR: MSA_CACHE_VOLUME,
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
)
def run_protenix(
    json_str: bytes | None,
    run_name: str,
    prep_cache_key: str | None = None,
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
    use_fast_layernorm: bool = False,
    extra_args: str | None = None,
    score_only: bool = False,
) -> bytes:
    """Run Protenix structure prediction or confidence scoring.

    Args:
        json_str: Input JSON bytes for prediction, or PDB/CIF bytes when
            score_only is True.
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
    import tempfile

    run_env = os.environ.copy()
    if use_fast_layernorm:
        run_env["LAYERNORM_TYPE"] = "fast_layernorm"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_dir = tmpdir_path / "output"
        out_dir.mkdir()

        if score_only:
            # Score an existing structure with the Protenix confidence head
            if json_str is None:
                raise ValueError("json_str must be provided when score_only=True")

            # Detect CIF vs PDB format: CIF files start with 'data_' (after
            # any leading comment lines starting with '#'), PDB files with
            # record types like HEADER/ATOM/REMARK.
            input_ext = ".pdb"
            for _line in json_str.splitlines():
                stripped = _line.strip()
                if not stripped or stripped.startswith(b"#"):
                    continue
                if stripped.startswith(b"data_"):
                    input_ext = ".cif"
                break
            input_file = tmpdir_path / f"{run_name}{input_ext}"
            input_file.write_bytes(json_str)

            # Map use_msa → --use_msas (both | false)
            # ProtenixScore's use_msas controls which chain roles receive MSAs.
            use_msas_val = "both" if use_msa else "false"

            # Map msa_server_mode → --msa_host_url.
            # The protenix remote server URL matches what `protenix msa` uses
            # (MMSEQS_SERVICE_HOST_URL in protenix/web_service/colab_request_parser.py).
            _PROTENIX_MSA_HOST = "https://protenix-server.com/api/msa"
            _COLABFOLD_MSA_HOST = "https://api.colabfold.com"
            msa_host_url = (
                _PROTENIX_MSA_HOST
                if msa_server_mode == "protenix"
                else _COLABFOLD_MSA_HOST
            )

            # Cache fetched MSAs in the PREP volume so they can be reused across
            # runs (separate sub-directory from the `protenix msa` cache).
            score_msa_cache_dir = Path(MSA_CACHE_DIR) / "score_msa_cache"
            score_msa_cache_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "protenixscore",
                "score",
                "--input",
                str(input_file),
                "--output",
                str(out_dir),
                "--model_name",
                model_name,
                "--dtype",
                dtype,
                "--use_msas",
                use_msas_val,
                "--msa_host_url",
                msa_host_url,
                "--msa_cache_dir",
                str(score_msa_cache_dir),
                "--msa_cache_mode",
                "readwrite",
            ]
            run_command(cmd, env=run_env, cwd=out_dir)

            # Persist MSA cache back to the volume for reuse in future runs
            MSA_CACHE_VOLUME.commit()
            print("💊 Packaging ProtenixScore results...")
            tarball_bytes = package_outputs(out_dir)
            return tarball_bytes

        # --- Prediction mode ---
        # Compute a cache key for this run (excludes seeds so we can track per-seed)
        h_out = sha256()
        if prep_cache_key is not None:
            h_out.update(prep_cache_key.encode("utf-8"))
        elif json_str is not None:
            h_out.update(json_str)
        h_out.update(
            f"{model_name}|{cycle}|{step}|{sample}|{dtype}"
            f"|{use_msa}|{msa_server_mode}|{use_template}|{use_rna_msa}".encode()
        )
        output_cache_key = h_out.hexdigest()
        vol_run_dir = Path(OUTPUTS_DIR) / run_name / output_cache_key

        # Sync the volume to see outputs from any previous (possibly interrupted) run
        OUTPUTS_VOLUME.reload()
        vol_run_dir.mkdir(parents=True, exist_ok=True)

        # Determine which seeds have already been saved to the volume.
        # Protenix outputs follow <vol_run_dir>/<sample_name>/seed_<N>/ so we
        # only need to look one directory level deep.
        seed_list = [int(s.strip()) for s in seeds.split(",")]
        done_seeds = [
            s
            for s in seed_list
            if any(d.is_dir() for d in vol_run_dir.glob(f"*/seed_{s}"))
        ]
        missing_seeds = [s for s in seed_list if s not in done_seeds]

        if not missing_seeds:
            print(
                f"💊 All {len(seed_list)} seed(s) found in output cache; "
                "skipping inference."
            )
            shutil.copytree(str(vol_run_dir), str(out_dir), dirs_exist_ok=True)
        else:
            if done_seeds:
                print(
                    f"💊 {len(done_seeds)} seed(s) cached; "
                    f"running {len(missing_seeds)} remaining seed(s)."
                )
                # Restore already-completed seeds so the final tarball is complete
                shutil.copytree(str(vol_run_dir), str(out_dir), dirs_exist_ok=True)

            # Resolve input JSON path
            if prep_cache_key is not None:
                input_json_path = (
                    Path(MSA_CACHE_DIR)
                    / prep_cache_key[:2]
                    / prep_cache_key
                    / "prepared.json"
                )
                if not input_json_path.exists():
                    raise FileNotFoundError(
                        f"Prepared input not found for cache key: {prep_cache_key}"
                    )
            else:
                if json_str is None:
                    raise ValueError(
                        "Either prep_cache_key or json_str must be provided"
                    )
                input_json_path = tmpdir_path / f"{run_name}.json"
                input_json_path.write_bytes(json_str)

            # Run protenix pred for missing seeds only
            missing_seeds_str = ",".join(str(s) for s in missing_seeds)
            cmd = [
                "protenix",
                "pred",
                "-i",
                str(input_json_path),
                "-o",
                str(out_dir),
                "-s",
                missing_seeds_str,
                "-c",
                str(cycle),
                "-p",
                str(step),
                "-e",
                str(sample),
                "-d",
                dtype,
                "-n",
                model_name,
                "--use_msa",
                str(use_msa),
                "--msa_server_mode",
                msa_server_mode,
                "--use_template",
                str(use_template),
                "--use_rna_msa",
                str(use_rna_msa),
                "--use_tfg_guidance",
                str(use_tfg_guidance),
            ]

            if extra_args:
                cmd.extend(shlex.split(extra_args))

            run_command(cmd, env=run_env, cwd=out_dir)

            # Persist new outputs to the volume so interrupted runs can resume
            shutil.copytree(str(out_dir), str(vol_run_dir), dirs_exist_ok=True)
            OUTPUTS_VOLUME.commit()

        # Package outputs
        print("💊 Packaging Protenix results...")
        tarball_bytes = package_outputs(out_dir)

    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_protenix_task(
    input_json: str,
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
    use_fast_layernorm: bool = False,
    download_models: bool = False,
    force_redownload: bool = False,
    extra_args: str | None = None,
    score_only: bool = False,
) -> None:
    """Run Protenix structure prediction on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file, or a PDB/CIF file when score_only
            is True.
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to input filename stem)
        model_name: Model checkpoint name
        seeds: Comma-separated random seeds for inference
        cycle: Pairformer cycle number
        step: Number of diffusion steps
        sample: Number of samples per seed
        dtype: Inference dtype (bf16 or fp32)
        use_msa: Whether to use MSA features
        msa_server_mode: MSA search mode (protenix or colabfold)
        use_template: Whether to use templates
        use_rna_msa: Whether to use RNA MSA features
        use_tfg_guidance: Enable Training-Free Guidance (TFG) for refined sampling
        use_fast_layernorm: Whether to enable the custom CUDA layernorm kernel
        download_models: Whether to download model weights and skip running
        force_redownload: Whether to force re-download of model weights
        extra_args: Additional CLI arguments passed to protenix pred
        score_only: When True, score an existing PDB/CIF structure using
            ``protenixscore score`` instead of running prediction.

    """
    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # Handle download-only mode
    if download_models:
        print(f"🧬 Downloading Protenix model: {model_name}")
        download_protenix_data.remote(
            model_name=model_name,
            force=force_redownload,
            include_templates=use_template,
        )
        print("🧬 Download complete!")
        return

    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if run_name is None:
        run_name = input_path.stem

    json_str = input_path.read_bytes()

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}_protenix_outputs.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    # Ensure models and data caches are available
    print(f"🧬 Checking Protenix model and data caches for {model_name}...")
    download_protenix_data.remote(
        model_name=model_name,
        force=False,
        include_templates=use_template,
    )

    if score_only:
        # Score an existing structure; MSA fetching is handled inside run_protenix
        print(f"🧬 Scoring structure with {model_name}...")
        tarball_bytes = run_protenix.remote(
            json_str=json_str,
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
        prep_cache_key: str | None = None
        if use_msa or use_template or use_rna_msa:
            print(
                "🧬 Running Protenix preprocessing (CPU-only) and caching intermediates..."
            )
            prep_cache_key = prepare_protenix_inputs.remote(
                json_str=json_str,
                use_msa=use_msa,
                msa_server_mode=msa_server_mode,
                use_template=use_template,
                use_rna_msa=use_rna_msa,
            )

        # Run inference
        print(f"🧬 Running Protenix inference with {model_name}...")
        tarball_bytes = run_protenix.remote(
            json_str=None if prep_cache_key is not None else json_str,
            run_name=run_name,
            prep_cache_key=prep_cache_key,
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
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 Protenix run complete! Results saved to {out_file}")
