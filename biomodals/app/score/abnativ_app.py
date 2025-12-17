r"""AbNatiV source repo: <https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ>.

## Configuration

**General flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-name` | **Required** | Prefix used to name the output directory and files. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved to the current working directory. |
| `--model-type` | `VH` | Selects the AbNatiV trained model (VH, VKappa, VLambda, VHH), or AbNatiV2 models (VH2, VL2, VHH2). If set to "paired", the paired V2 model will be used. |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download model weights and skip running. |
| `--force-redownload` | `--no-force-redownload` | Whether to force re-download of model weights even if they exist. |
| `--mean-score-only`/`--no-mean-score-only` | `--no-mean-score-only` | When True, only export a per-sequence score file instead of both sequence and per-position nativeness profiles. |
| `--align-before-scoring`/`--no-align-before-scoring` | `--align-before-scoring` | Align and clean the sequences before scoring. |
| `--num-workers` | `1` | Number of workers to parallelize the alignment process. |
| `--plot-profiles`/`--no-plot-profiles` | `--plot-profiles` | Generate and save per-sequence profile plots under `{output_directory}/{output_id}_profiles`. |

**Single-sequence model flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-fasta-or-seq` | `None` | Path to a FASTA file or a single-sequence string. Required for single-chain models. |
| `--is-vhh`/`--no-is-vhh` | `--no-is-vhh` | Use the VHH alignment seed, which is better for nanobody sequences. |

**Paired-sequence model flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-paired-csv` | `None` | Path to a CSV file containing paired VH and VL sequences. The CSV file should have columns "ID", "vh_seq", and "vl_seq". Required for paired model unless the following args are both provided. |
| `--input-vh-seq` | `None` | A single-sequence string for VH sequences. Used if `--input-paired-csv` is not provided. |
| `--input-vl-seq` | `None` | A single-sequence string for VL sequences. Used if `--input-paired-csv` is not provided. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `AbNatiV` | Name of the Modal app to use. |
| `GPU` | `A10G` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Notes

* Always check the `--model-type` argument to ensure you are using the correct model for your sequences.
* When `--model-type` is set to `paired`, the `--input-fasta-or-seq` argument is ignored, and sequences are read from either `--input-paired-csv` or the combination of `--input-vh-seq` and `--input-vl-seq`.
* In paired mode, `--input-paired-csv` takes precedence over `--input-vh-seq` and `--input-vl-seq` if both are provided.
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

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.micromamba(python_version="3.12")
    .apt_install("git", "build-essential", "wget", "zstd")
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
# Fetch model weights
##########################################
@app.function(
    cpu=(1.125, 16.125),
    volumes={ABNATIV_MODEL_DIR: ABNATIV_VOLUME},
    timeout=TIMEOUT * 10,
)
def download_abnativ_models(force: bool = False) -> None:
    """Download AbNatiV models into the mounted volume."""
    # Download all artifacts
    print("Downloading AbNatiV models...")
    cmd = ["abnativ", "init"]
    if force:
        cmd.append("--force_update")

    run_command(cmd, bufsize=8)
    ABNATIV_VOLUME.commit()
    print("Model download complete")


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={ABNATIV_MODEL_DIR: ABNATIV_VOLUME.read_only()},
)
def abnativ_score_unpaired(
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
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        work_path = Path(tmpdir) / f"{output_id}_abnativ_{nativeness_type}"
        work_path.mkdir()
        input_fasta = work_path.parent / f"{output_id}.fasta"
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
            # "--ncpu",
            # str(ncpu),  # bug in upstream code
        ]
        if mean_score_only:
            cmd.append("--mean_score_only")
        if align_before_scoring:
            cmd.append("--do_align")
        if is_vhh:
            cmd.append("--is_VHH")
        if plot_profiles:
            cmd.append("-plot")

        run_command(cmd)

        print("Packaging results...")
        tarball_bytes = package_outputs(str(work_path))
        print("Packaging complete.")

    return tarball_bytes


@app.function(
    gpu=GPU,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={ABNATIV_MODEL_DIR: ABNATIV_VOLUME.read_only()},
)
def abnativ_score_paired(
    csv_bytes: bytes,
    output_id: str,
    mean_score_only: bool,
    align_before_scoring: bool,
    ncpu: int,
    plot_profiles: bool,
):
    """Manage AbNatiV runs and return all score results."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        work_path = Path(tmpdir) / f"{output_id}_abnativ_paired"
        work_path.mkdir()
        input_csv = work_path.parent / f"{output_id}.csv"
        with open(input_csv, "wb") as f:
            f.write(csv_bytes)
        cmd = [
            "abnativ",
            "paired_score",
            "-i",
            str(input_csv),
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
        if plot_profiles:
            cmd.append("-plot")

        run_command(cmd)

        print("Packaging results...")
        tarball_bytes = package_outputs(str(work_path))
        print("Packaging complete.")

    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_abnativ_task(
    # Inputs and outputs
    run_name: str,
    out_dir: str | None = None,
    input_fasta_or_seq: str | None = None,
    input_paired_csv: str | None = None,
    input_vh_seq: str | None = None,
    input_vl_seq: str | None = None,
    # Model download options
    download_models: bool = False,
    force_redownload: bool = False,
    # AbNatiV configs
    model_type: str = "VH",
    mean_score_only: bool = False,
    align_before_scoring: bool = True,
    num_workers: int = 1,
    is_vhh: bool = False,
    plot_profiles: bool = True,
) -> None:
    """Run AbNatiV scoring on modal and fetch results to `out_dir`.

    See `abnativ score -h` for details on input arguments.

    Args:
        run_name: Prefix used to name the output directory and files.
        out_dir: Local directory where the results are persisted; defaults to the current working directory.
        input_fasta_or_seq: Path to a FASTA file or a single-sequence string.
        input_paired_csv: (For paired model only) Path to a CSV file containing paired VH and VL sequences.
            The CSV file should have columns "ID", "vh_seq", and "vl_seq".
        input_vh_seq: (For paired model only) A single-sequence string for VH sequences.
        input_vl_seq: (For paired model only) A single-sequence string for VL sequences.
        download_models: If True, download the AbNatiV models before inference.
        force_redownload: Force re-download of the models even if they already exist.
        model_type: Selects the AbNatiV trained model (VH, VKappa, VLambda, VHH), or AbNatiV2 models (VH2, VL2, VHH2).
            If set to "paired", the paired V2 model will be used. `input_fasta_or_seq` will be ignored, and sequences
            will be read from `input_{paired_csv, vh_seq, vl_seq}` instead.
        mean_score_only: When True, only export a per-sequence score file instead of both sequence and per-position nativeness profiles.
        align_before_scoring: Align and clean the sequences before scoring; can be slow for large sets.
            If not set, all input sequences need to be Aho-numbered.
        num_workers: Number of workers to parallelize the alignment process.
        is_vhh: Use the VHH alignment seed, which is better for nanobody sequences.
        plot_profiles: Generate and save per-sequence profile plots under `{output_directory}/{output_id}_profiles`.
    """
    # Ignore everything else if downloading models
    if download_models:
        print("🧬 Checking AbNatiV inference dependencies...")
        download_abnativ_models.remote(force=force_redownload)
        return

    # Set up output paths
    print("🧬 Starting AbNatiV run...")
    if out_dir is None:
        out_dir = Path.cwd()
    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_zst_file = local_out_dir / f"{run_name}_abnativ_{model_type}.tar.zst"
    if out_zst_file.exists():
        raise FileExistsError(f"Output file already exists: {out_zst_file}")

    # Submit scoring job based on model type
    match model_type:
        case "paired":
            if input_paired_csv is None and (
                input_vh_seq is None or input_vl_seq is None
            ):
                raise ValueError(
                    "For paired model_type, either input_paired_csv or both "
                    "input_vh_seq and input_vl_seq must be provided."
                )
            if input_paired_csv is not None:
                input_path = Path(input_paired_csv)
                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Input paired CSV file not found: {input_paired_csv}"
                    )
                with open(input_path, "rb") as f:
                    csv_bytes = f.read()
            elif input_vh_seq is not None and input_vl_seq is not None:
                # Build CSV from single-sequence inputs
                if "\n" in input_vh_seq or " " in input_vh_seq:
                    raise ValueError(
                        "Input VH sequence does not appear to be a valid single sequence."
                    )
                if "\n" in input_vl_seq or " " in input_vl_seq:
                    raise ValueError(
                        "Input VL sequence does not appear to be a valid single sequence."
                    )
                csv_str = f"ID,vh_seq,vl_seq\nsingle_seq,{input_vh_seq.strip()},{input_vl_seq.strip()}\n"
                csv_bytes = csv_str.encode("utf-8")

            print("🧬 Running AbNatiV paired mode...")
            abnativ_scores = abnativ_score_paired.remote(
                csv_bytes=csv_bytes,
                output_id=run_name,
                mean_score_only=mean_score_only,
                align_before_scoring=align_before_scoring,
                ncpu=num_workers,
                plot_profiles=plot_profiles,
            )
        case "VH" | "VKappa" | "VLambda" | "VHH" | "VH2" | "VL2" | "VHH2":
            # Load fasta, or build one if it is not a file path
            if input_fasta_or_seq is None:
                raise ValueError(
                    "input_fasta_or_seq must be provided for single-chain AbNatiV scoring."
                )
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

            print(f"🧬 Running AbNatiV unpaired {model_type} mode...")
            abnativ_scores = abnativ_score_unpaired.remote(
                fasta_bytes=fasta_bytes,
                output_id=run_name,
                nativeness_type=model_type,
                mean_score_only=mean_score_only,
                align_before_scoring=align_before_scoring,
                ncpu=num_workers,
                is_vhh=is_vhh,
                plot_profiles=plot_profiles,
            )
        case _:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be one of "
                "'VH', 'VKappa', 'VLambda', 'VHH', 'VH2', 'VL2', 'VHH2', or 'paired'."
            )

    out_zst_file.write_bytes(abnativ_scores)
    print(
        f"🧬 AbNatiV run complete! Results saved to {local_out_dir} in {out_zst_file.name}"
    )
