"""AntiFold source repo: <https://github.com/oxpig/AntiFold>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--struct-file` | **Required** | Path to input PDB or mmCIF file containing the antibody structure. **The antibody chains in the file must be IMGT-numbered.** |
| `--run-name` | None | Prefix used to name the output directory and files. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved to the current working directory. |
| `--heavy-chain` | 1st chain in structure file | Chain ID of the heavy chain. |
| `--light-chain` | 2nd chain in structure file | Chain ID of the light chain. |
| `--antigen-chain` | None | Chain ID of the antigen, if present. |
| `--nanobody-chain` | None | Chain ID of the nanobody, if applicable. |
| `--regions` | `CDR1 CDR2 CDR3` | Space-separated string specifying the regions to design. See <https://github.com/oxpig/AntiFold/blob/789d46786624c01eb44f177ef4c0deeeb6e77469/antifold/antiscripts.py#L738> for options. |
| `--num-seq-per-target` | `0` | Number of sequences to generate. |
| `--sampling-temp` | `0.2` | Sampling temperature controls generated sequence diversity, by scaling the inverse folding probabilities before sampling. Temperature = 1 means no change, while temperature ~ 0 only samples the most likely amino-acid at each position (acts as argmax). |
| `--limit-variation` | `False` | If set, limits variation to as many mutations as expected from temperature sampling. |
| `--extract-embeddings` | `False` | If set, extracts and saves per-residue embeddings and return them in an NumPy array. |
| `--num-threads` | `0` | Number of CPU threads to use. Defaults to all available. |
| `--seed` | `42` | Random seed for reproducibility. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `GPU` | `A10G` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Notes

* By default there would be two files in the output archive file:
  * `log.txt`: Log file for the AntiFold run.
  * `<run_name>_<vh_chain><vl_chain>.csv`: table of the residue indices and scores.
* If `--extract-embeddings` is set, there would be an additional file:
  * `<run_name>_<vh_chain><vl_chain>_embeddings.npy`: NumPy array of per-residue embeddings.
* The model does not like large antigens in the input structure. In our benchmarks antigens don't seem to affect results much, so for performance you may want to remove the antigen chains from the input structure and only provide the antibody chain(s).
* Make sure *all* antibody chains are IMGT-numbered!
"""
# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import sys
from pathlib import Path

from modal import App, Image

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import package_outputs, run_command

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    name="AntiFold",
    repo_url="https://github.com/oxpig/AntiFold",
    repo_commit_hash="789d46786624c01eb44f177ef4c0deeeb6e77469",
    version="0.3.1",
    python_version="3.10",
    cuda_version="cu121",
    gpu=os.environ.get("GPU", "A10G"),
)
MODEL_DIR = CONF.model_dir

##########################################
# Image and app definitions
##########################################
runtime_image = patch_image_for_helper(
    Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "wget")
    .env(CONF.default_env)
    .uv_pip_install(f"git+{CONF.repo_url}@{CONF.repo_commit_hash}")
    .uv_pip_install(["torch==2.2.0", "torchvision"])
    .uv_pip_install(
        "torch-scatter",
        find_links=f"https://data.pyg.org/whl/torch-2.2.0+{CONF.cuda_version}.html",
        extra_options="--no-build-isolation",  # https://github.com/astral-sh/uv/issues/5040
    )
)

app = App(CONF.name, image=runtime_image)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME},
)
def antifold_inference(
    struct_bytes: bytes,
    struct_file_type: str,  # or "cif"
    output_id: str,
    heavy_chain: str | None = None,  # 1st chain if not specified
    light_chain: str | None = None,  # 2nd chain if not specified
    antigen_chain: str | None = None,
    nanobody_chain: str | None = None,
    regions: str = "CDR1 CDR2 CDR3",
    num_seq_per_target: int = 0,
    sampling_temp: float = 0.2,
    limit_variation: bool = False,
    extract_embeddings: bool = False,
    num_threads: int = 0,
    seed: int = 42,
) -> bytes:
    """Manage AntiFold runs and return all inference results."""
    from tempfile import TemporaryDirectory

    # AntiFold hard-coded the download logic to look for models in ./models/model.pt
    model_path = (
        Path(sys.exec_prefix)
        / "lib"
        / f"python{CONF.python_version}"
        / "site-packages"
        / "models"
        / "model.pt"
    )
    cache_model_path = MODEL_DIR / "model.pt"
    if cache_model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.symlink_to(cache_model_path)

    with TemporaryDirectory() as tmpdir:
        work_path = Path(tmpdir) / f"{output_id}_antifold"
        work_path.mkdir()
        input_struct = work_path.parent / f"{output_id}.{struct_file_type}"
        with open(input_struct, "wb") as f:
            f.write(struct_bytes)
        cmd = [
            sys.executable,
            "-m",
            "antifold.main",
            "--pdb_file",
            str(input_struct),
            "--out_dir",
            str(work_path),
            "--regions",
            regions,
            "--num_seq_per_target",
            str(num_seq_per_target),
            "--sampling_temp",
            str(sampling_temp),
            "--seed",
            str(seed),
            "--num_threads",
            str(num_threads),
        ]
        if heavy_chain is not None:
            cmd.extend(("--heavy_chain", heavy_chain))
        if light_chain is not None:
            if nanobody_chain is not None:
                raise ValueError("Cannot specify both light_chain and nanobody_chain.")
            cmd.extend(("--light_chain", light_chain))
        if antigen_chain is not None:
            cmd.extend(("--antigen_chain", antigen_chain))
        if limit_variation:
            cmd.append("--limit_variation")
        if extract_embeddings:
            cmd.append("--extract_embeddings")
        if nanobody_chain is not None:
            cmd.extend(("--nanobody_chain", nanobody_chain))

        run_command(cmd)

        print("💊 Packaging results...")
        tarball_bytes = package_outputs(work_path)

    if not cache_model_path.exists():
        # Cache the model for future runs
        import shutil

        shutil.copyfile(model_path, cache_model_path)
        MODEL_VOLUME.commit()

    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_antifold_task(
    # Input and output
    struct_file: str,
    run_name: str | None = None,
    out_dir: str | None = None,
    # AntiFold parameters
    heavy_chain: str | None = None,
    light_chain: str | None = None,
    antigen_chain: str | None = None,
    nanobody_chain: str | None = None,
    regions: str = "CDR1 CDR2 CDR3",
    num_seq_per_target: int = 0,
    sampling_temp: float = 0.2,
    limit_variation: bool = False,
    extract_embeddings: bool = False,
    num_threads: int = 0,
    seed: int = 42,
) -> None:
    """Run AntiFold inverse folding for a given antibody(-antigen) structure.

    Args:
        run_name: Prefix used to name the output directory and files.
        struct_file: Path to input PDB or mmCIF file containing the antibody structure.
            The antibody chains in the file should be IMGT-numbered.
        out_dir: Local directory where the results are persisted; defaults to the current working directory.
        heavy_chain: Chain ID of the heavy chain; defaults to the first chain in the structure file.
        light_chain: Chain ID of the light chain; defaults to the second chain in the structure file.
        antigen_chain: Chain ID of the antigen, if present.
        nanobody_chain: Chain ID of the nanobody, if applicable.
        regions: Space-separated string specifying the regions to design.
        num_seq_per_target: Number of sequences to generate.
        sampling_temp: Sampling temperature controls generated sequence diversity,
            by scaling the inverse folding probabilities before sampling.
            Temperature = 1 means no change, while temperature ~ 0 only samples the most
            likely amino-acid at each position (acts as argmax).
        limit_variation: If True, limits variation to as many mutations as expected
            from temperature sampling.
        extract_embeddings: If True, extracts and saves per-residue embeddings.
        num_threads: Number of CPU threads to use.
        seed: Random seed for reproducibility.
    """
    # Set up output paths
    print("🧬 Starting AntiFold run...")
    struct_file_path = Path(struct_file).expanduser().resolve()
    if not struct_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_file_path}")
    struct_file_type = struct_file_path.suffix.removeprefix(".").lower()
    if struct_file_type not in {"pdb", "cif"}:
        raise ValueError(
            f"Unsupported structure file type: {struct_file_type}. Must be 'pdb' or 'cif'."
        )

    if run_name is None:
        run_name = struct_file_path.stem

    local_out_dir = (
        (Path(out_dir) if out_dir is not None else Path.cwd()).expanduser().resolve()
    )
    out_zst_file = local_out_dir / f"{run_name}_antifold.tar.zst"
    if out_zst_file.exists():
        raise FileExistsError(f"Output file already exists: {out_zst_file}")

    # Submit scoring job based on model type
    print("🧬 Running AntiFold inverse folding...")
    with open(struct_file, "rb") as f:
        struct_bytes = f.read()
    antifold_outputs: bytes = antifold_inference.remote(  # pyrefly: ignore[bad-assignment,invalid-param-spec]
        struct_bytes,
        struct_file_type,
        run_name,
        heavy_chain,
        light_chain,
        antigen_chain,
        nanobody_chain,
        regions,
        num_seq_per_target,
        sampling_temp,
        limit_variation,
        extract_embeddings,
        num_threads,
        seed,
    )
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_zst_file.write_bytes(antifold_outputs)
    print(
        f"🧬 AntiFold run complete! Results saved to {local_out_dir} in {out_zst_file.name}"
    )
