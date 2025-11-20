"""Run AbNatiV on Modal GPU instances.

https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ

"""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "AbNatiV")

# Volume for model cache
ABNATIV_VOLUME = Volume.from_name("abnativ-models", create_if_missing=True)
ABNATIV_MODEL_DIR = "/root/.abnativ/models/pretrained_models"

# Volume for outputs
OUTPUTS_VOLUME = Volume.from_name("abnativ-outputs", create_if_missing=True, version=2)
OUTPUTS_DIR = "/abnativ-outputs"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.micromamba(python_version="3.12")
    .apt_install("git", "build-essential")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu128",  # find best torch and CUDA versions
        }
    )
    .micromamba_install(["openmm", "pdbfixer", "biopython"], channels=["conda-forge"])
    .micromamba_install(["anarci"], channels=["bioconda"])
    .uv_pip_install("abnativ==2.0.3")
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def package_outputs(
    dir: str, tar_args: list[str] | None = None, num_threads: int = 16
) -> bytes:
    """Package directory into a tar.zst archive and return as bytes."""
    import subprocess as sp

    dir_path = Path(dir)
    cmd = ["tar", "--zstd"]
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-cf", "-", dir_path.name])

    return sp.check_output(
        cmd, cwd=dir_path.parent, env={"ZSTD_NBTHREADS": str(num_threads)}
    )  # noqa: S603


##########################################
# Fetch model weights
##########################################
@app.function(volumes={ABNATIV_MODEL_DIR: ABNATIV_VOLUME}, timeout=TIMEOUT)
def download_abnativ_models(force: bool = False) -> None:
    """Download AbNatiV models into the mounted volume."""
    import subprocess as sp

    # Download all artifacts
    print("Downloading AbNatiV models...")
    cmd = ["abnativ", "init"]
    if force:
        cmd.append("--force_update")
    sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf-8")
    print("Model download complete")

    ABNATIV_VOLUME.commit()


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=(0.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={ABNATIV_MODEL_DIR: ABNATIV_VOLUME, OUTPUTS_DIR: OUTPUTS_VOLUME},
)
def collect_abnativ_scores_single(
    fasta_bytes: bytes,
    output_id: str,
    nativeness_type: str,
    mean_score_only: bool,
    align_before_scoring: bool,
    ncpu: int,
    is_vhh: bool,
    plot_profiles: bool,
):
    """Manage AbNatiV runs and return all score results."""
    import subprocess as sp
    from hashlib import sha256

    input_hash = sha256(fasta_bytes).hexdigest()

    work_path = Path(OUTPUTS_DIR) / input_hash / output_id
    if work_path.exists():
        print(f"Output directory already exists, skipping run: {work_path}")
        print("Packaging AbNatiV results...")
        tarball_bytes = package_outputs(str(work_path))
        print("Packaging complete.")
        return tarball_bytes

    work_path.mkdir(parents=True, exist_ok=True)
    input_fasta = work_path / f"{output_id}_input.fasta"
    with open(input_fasta, "wb") as f:
        f.write(fasta_bytes)
    cmd = [
        "abnativ",
        "score",
        "-nat",
        nativeness_type,
        "-i",
        str(input_fasta),
        "-odir",
        str(work_path),
        "--output_id",
        output_id,
        "--ncpu",
        str(ncpu),
    ]
    if mean_score_only:
        cmd.append("--mean_score_only")
    if align_before_scoring:
        cmd.append("--do_align")
    if is_vhh:
        cmd.append("--is_vhh")
    if plot_profiles:
        cmd.append("--plot")

    OUTPUTS_VOLUME.reload()
    process = sp.run(cmd, capture_output=True, check=True)
    print(process.stdout.decode("utf-8"))
    print(process.stderr.decode("utf-8"))

    OUTPUTS_VOLUME.commit()
    print("Packaging results...")
    tarball_bytes = package_outputs(str(work_path))
    print("Packaging complete.")
    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_abnativ_task(
    input_fasta_or_seq: str,
    run_name: str,
    out_dir: str | None = None,
    download_models: bool = False,
    force_redownload: bool = False,
    nativeness_type: str = "VH",
    mean_score_only: bool = False,
    align_before_scoring: bool = False,
    num_workers: int = 1,
    is_vhh: bool = False,
    plot_profiles: bool = False,
) -> None:
    """Run AbNatiV scoring on modal and fetch results to `out_dir`.

    See `abnativ score -h` for details on input arguments.

    Args:
        input_fasta_or_seq: Path to a FASTA file or a single-sequence string.
        run_name: Prefix used to name the output directory and files.
        out_dir: Local directory where the results are persisted; defaults to the current working directory.
        download_models: If True, download the AbNatiV models before inference.
        force_redownload: Force re-download of the models even if they already exist.
        nativeness_type: Selects the AbNatiV trained model (VH, VKappa, VLambda, VHH).
        mean_score_only: When True, only export a per-sequence score file instead of both sequence and per-position nativeness profiles.
        align_before_scoring: Align and clean the sequences before scoring; can be slow for large sets.
        num_workers: Number of workers to parallelize the alignment process.
        is_vhh: Use the VHH alignment seed, which is better for nanobody sequences.
        plot_profiles: Generate and save per-sequence profile plots under `{output_directory}/{output_id}_profiles`.
    """
    # Load input and prepend ">A" if it is not a file path
    input_path = Path(input_fasta_or_seq)
    if input_path.exists():
        with open(input_path, "rb") as f:
            fasta_bytes = f.read()
    else:
        if "\n" in input_fasta_or_seq or " " in input_fasta_or_seq:
            raise ValueError(
                "Input sequence does not appear to be a valid file path. "
                "Please provide a valid FASTA file path or a single-line sequence."
            )
        fasta_str = f">single_seq\n{input_fasta_or_seq.strip()}\n"
        fasta_bytes = fasta_str.encode("utf-8")

    if out_dir is None:
        out_dir = Path.cwd()
    local_out_dir = Path(out_dir) / run_name
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_zst_file = local_out_dir / f"{run_name}_abnativ_scores.tar.zst"
    if out_zst_file.exists():
        raise FileExistsError(f"Output file already exists: {out_zst_file}")

    print("🧬 Starting AbNatiV run...")

    if download_models:
        print("🧬 Checking AbNatiV inference dependencies...")
        download_abnativ_models.remote(force=force_redownload)

    print(f"🧬 Running AbNatiV and collecting results to {out_zst_file}")
    abnativ_scores = collect_abnativ_scores_single.remote(
        fasta_bytes=fasta_bytes,
        output_id=run_name,
        nativeness_type=nativeness_type,
        mean_score_only=mean_score_only,
        align_before_scoring=align_before_scoring,
        ncpu=num_workers,
        is_vhh=is_vhh,
        plot_profiles=plot_profiles,
    )
    out_zst_file.write_bytes(abnativ_scores)

    print(
        f"🧬 AbNatiV run complete! Results saved to {local_out_dir} in {out_zst_file.name}"
    )
