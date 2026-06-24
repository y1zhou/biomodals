r"""AbNatiV source repo: <https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ>.

## Notes

* Always check the `--model-type` argument to ensure you are using the correct model for your sequences.
* When `--model-type` is set to `paired`, the `--input-fasta-or-seq` argument is ignored, and sequences are read from either `--input-paired-csv` or the combination of `--input-vh-seq` and `--input-vl-seq`.
* In paired mode, `--input-paired-csv` takes precedence over `--input-vh-seq` and `--input-vl-seq` if both are provided.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415
import os
from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import MAX_TIMEOUT, MODEL_VOLUME
from biomodals.helper.shell import package_outputs, run_command

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AbNatiV",
    repo_url="https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ",
    package_name="abnativ",
    version="2.0.8",
    python_version="3.12",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "A10G"),
    model_volume_mountpoint="/root/.abnativ/models/pretrained_models",
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Metadata for model files and patch locations."""

    model_files = {
        "VH": "vh_model.ckpt",
        "VHH": "vhh_model.ckpt",
        "VKappa": "vkappa_model.ckpt",
        "VLambda": "vlambda_model.ckpt",
        "VH2": "vh2_model.ckpt",
        "VL2": "vl2_model.ckpt",
        "VHH2": "vhh2_model.ckpt",
        "paired": "vpaired2_model.ckpt",
    }
    pkg_path = f"/opt/conda/lib/python{CONF.python_version}/site-packages/abnativ"
    plotter_path = "model/alignment/plotter.py"
    scoring_functions_path = "model/scoring_functions.py"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "wget")
    .env(CONF.default_env | {"ABNATIV_MODELS_DIR": f"{CONF.model_volume_mountpoint}"})
    .micromamba_install(
        ["anarci", "hmmer", "pdbfixer"], channels=["bioconda", "conda-forge"]
    )
    .uv_pip_install("numba>=0.61.2", f"{CONF.package_name}=={CONF.version}")
    # AbNatiV 2.0.8 defines two adjacent strucorr colormap points at 0.25,
    # which strict Matplotlib builds reject while importing abnativ.
    .workdir(APP_INFO.pkg_path)
    .run_commands(
        " && ".join((
            f"grep -q '(0.25, \"DimGray\")' {APP_INFO.plotter_path}",
            f'sed -i \'s/(0\\.25, "DimGray")/(0.250001, "DimGray")/\' {APP_INFO.plotter_path}',
            f"grep -q '(0.250001, \"DimGray\")' {APP_INFO.plotter_path}",
        ))
    )
    # AbNatiV 2.0.8 saves one figure per sequence profile without closing it.
    .run_commands(
        " && ".join((
            "grep -q 'plt.savefig(fig_fp_save, dpi=200, "
            f'bbox_inches="tight")\' {APP_INFO.scoring_functions_path}',
            "sed -i '/plt.savefig(fig_fp_save, dpi=200, "
            f'bbox_inches="tight")/a\\    plt.close(fig)\' {APP_INFO.scoring_functions_path}',
            f"grep -q 'plt.close(fig)' {APP_INFO.scoring_functions_path}",
        ))
    )
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Model validation helpers
##########################################
def _checkpoint_error(checkpoint_path: Path) -> str | None:
    """Return a reason when a checkpoint is missing or unreadable."""
    import zipfile

    if not checkpoint_path.exists():
        return "missing"
    if not checkpoint_path.is_file():
        return "not a regular file"

    size = checkpoint_path.stat().st_size
    if size < 1024:
        return f"too small ({size} bytes)"

    with checkpoint_path.open("rb") as checkpoint_file:
        header = checkpoint_file.read(4)

    if header.startswith(b"PK"):
        try:
            with zipfile.ZipFile(checkpoint_path) as checkpoint_zip:
                corrupt_member = checkpoint_zip.testzip()
        except zipfile.BadZipFile as exc:
            return f"invalid PyTorch zip checkpoint: {exc}"
        if corrupt_member is not None:
            return f"corrupt zip member: {corrupt_member}"
        return None

    try:
        import torch  # type: ignore[ty:unresolved-import]

        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"cannot be loaded by torch: {exc}"
    return None


def _invalid_model_files(model_dir: Path, filenames: list[str]) -> dict[str, str]:
    invalid_files = {}
    for filename in filenames:
        if error := _checkpoint_error(model_dir / filename):
            invalid_files[filename] = error
    return invalid_files


def _redownload_abnativ_model_files(filenames: list[str]) -> None:
    from abnativ.init import ensure_zenodo_models  # type: ignore[ty:unresolved-import]

    ensure_zenodo_models(filenames, do_force_update=True)


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=MAX_TIMEOUT
)
def ensure_abnativ_model(model_type: str, force: bool = False) -> None:
    """Ensure the checkpoint needed for one AbNatiV scoring mode is readable."""
    filename = APP_INFO.model_files[model_type]
    model_dir = Path(CONF.model_volume_mountpoint)
    checkpoint_path = model_dir / filename

    if force:
        print(f"💊 Force redownloading AbNatiV checkpoint {filename}")
        _redownload_abnativ_model_files([filename])

    if error := _checkpoint_error(checkpoint_path):
        print(f"💊 AbNatiV checkpoint {filename} is unreadable: {error}")
        _redownload_abnativ_model_files([filename])

    if error := _checkpoint_error(checkpoint_path):
        raise RuntimeError(
            f"AbNatiV checkpoint {checkpoint_path} is still unreadable after "
            f"redownload: {error}"
        )

    MODEL_VOLUME.commit()
    print(f"💊 AbNatiV checkpoint {filename} is available")


##########################################
# Inference functions
##########################################
@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True),
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
        print("💊 Packaging results...")
        tarball_bytes = package_outputs(work_path)

    return tarball_bytes


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True),
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

        print("💊 Packaging results...")
        tarball_bytes = package_outputs(work_path)

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
        out_dir: Local directory where the results are persisted. If not specified,
            outputs will be saved to the current working directory.
        input_fasta_or_seq: Path to a FASTA file or a single-sequence string.
            Required for single-chain models.
        input_paired_csv: (For paired model only) Path to a CSV file containing
            paired VH and VL sequences. The CSV file should have columns "ID",
            "vh_seq", and "vl_seq". Required for paired model unless `input_vh_seq`
            and `input_vl_seq` are both provided.
        input_vh_seq: (For paired model only) A single-sequence string for VH sequences.
            Used if `input_paired_csv` is not provided.
        input_vl_seq: (For paired model only) A single-sequence string for VL sequences.
            Used if `input_paired_csv` is not provided.
        force_redownload: Force re-download of the models even if they already exist.
        model_type: Selects the AbNatiV trained model (VH, VKappa, VLambda, VHH),
            or AbNatiV2 models (VH2, VL2, VHH2). If set to "paired", the paired V2
            model will be used. `input_fasta_or_seq` will be ignored, and sequences
            will be read from `input_{paired_csv, vh_seq, vl_seq}` instead.
        mean_score_only: When True, only export a per-sequence score file instead of
            both sequence and per-position nativeness profiles.
        align_before_scoring: Align and clean the sequences before scoring.
            If not set, all input sequences need to be Aho-numbered.
            Note that this can be slow for large sets.
        num_workers: Number of workers to parallelize the alignment process.
        is_vhh: Use the VHH alignment seed, which is better for nanobody sequences.
            Only applicable for single-chain models.
        plot_profiles: Generate and save per-sequence profile plots under `{output_directory}/{output_id}_profiles`.
    """
    # Set up output paths
    print("🧬 Starting AbNatiV run...")
    local_out_dir = (
        (Path(out_dir) if out_dir is not None else Path.cwd()).expanduser().resolve()
    )
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
            ensure_abnativ_model.remote(model_type="paired", force=force_redownload)
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
            ensure_abnativ_model.remote(model_type=model_type, force=force_redownload)
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
