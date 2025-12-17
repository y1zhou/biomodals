"""AntiFold source repo: <https://github.com/oxpig/AntiFold>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--run-name` | **Required** | Prefix used to name the output directory and files. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved to the current working directory. |
| `--struct-file` | **Required** | Path to input PDB or mmCIF file containing the antibody structure. **The antibody chains in the file must be IMGT-numbered.** |
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
| `MODAL_APP` | `AntiFold` | Name of the Modal app to use. |
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
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
import sys
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "AntiFold")

# Volume for model cache
ANTIFOLD_VOLUME = Volume.from_name("antifold-models", create_if_missing=True)
ANTIFOLD_MODEL_DIR = "/antifold-models"

# Repositories and commit hashes
ANTIFOLD_REPO = "https://github.com/oxpig/AntiFold"
ANTIFOLD_COMMIT = "789d46786624c01eb44f177ef4c0deeeb6e77469"
ANTIFOLD_REPO_DIR = "/opt/antifold"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential", "wget", "zstd")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu121",  # find best torch and CUDA versions
            "HF_HOME": str(ANTIFOLD_MODEL_DIR),  # store boltzgen model weights
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # speed up downloads
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {ANTIFOLD_REPO} {ANTIFOLD_REPO_DIR}",
                f"cd {ANTIFOLD_REPO_DIR}",
                f"git checkout {ANTIFOLD_COMMIT}",
                "uv venv --python 3.10",
                "uv pip install .",
            ),
        )
    )
    .env({"PATH": f"{ANTIFOLD_REPO_DIR}/.venv/bin:$PATH"})
    .uv_pip_install(["torch==2.2.0", "torchvision"])
    .uv_pip_install(
        "torch-scatter",
        find_links="https://data.pyg.org/whl/torch-2.2.0+cu121.html",
        extra_options="--no-build-isolation",  # https://github.com/astral-sh/uv/issues/5040
    )
    .workdir(ANTIFOLD_REPO_DIR)
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def package_outputs(
    dir: str, tar_args: list[str] | None = None, num_threads: int = 16
) -> bytes:
    """Package directory into a tar.zst archive and return as bytes."""
    import os
    import subprocess as sp

    dir_path = Path(dir)
    cmd = ["tar", "--zstd"]
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-cf", "-", dir_path.name])

    return sp.check_output(
        cmd, cwd=dir_path.parent, env=os.environ | {"ZSTD_NBTHREADS": str(num_threads)}
    )  # noqa: S603


def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={ANTIFOLD_MODEL_DIR: ANTIFOLD_VOLUME},
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
        Path(ANTIFOLD_REPO_DIR)
        / ".venv"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "models"
        / "model.pt"
    )
    cache_model_path = Path(ANTIFOLD_MODEL_DIR) / "model.pt"
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
            "antifold/main.py",
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

        print("Packaging results...")
        tarball_bytes = package_outputs(str(work_path))
        print("Packaging complete.")

    if not cache_model_path.exists():
        # Cache the model for future runs
        import shutil

        shutil.copyfile(model_path, cache_model_path)
        ANTIFOLD_VOLUME.commit()

    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_antifold_task(
    # Input and output
    run_name: str,
    struct_file: str,
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
    local_out_dir = (
        (Path(out_dir) if out_dir is not None else Path.cwd()).expanduser().resolve()
    )
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_zst_file = local_out_dir / f"{run_name}_antifold.tar.zst"
    if out_zst_file.exists():
        raise FileExistsError(f"Output file already exists: {out_zst_file}")

    # Submit scoring job based on model type
    print("🧬 Running AntiFold inverse folding...")
    struct_file_path = Path(struct_file).expanduser().resolve()
    if not struct_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_file_path}")
    struct_file_type = struct_file_path.suffix.removeprefix(".").lower()
    if struct_file_type not in {"pdb", "cif"}:
        raise ValueError(
            f"Unsupported structure file type: {struct_file_type}. Must be 'pdb' or 'cif'."
        )
    with open(struct_file, "rb") as f:
        struct_bytes = f.read()
    antifold_outputs: bytes = (
        antifold_inference.remote(  # pyrefly: ignore[bad-assignment,invalid-param-spec]
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
    )

    out_zst_file.write_bytes(antifold_outputs)
    print(
        f"🧬 AntiFold run complete! Results saved to {local_out_dir} in {out_zst_file.name}"
    )
