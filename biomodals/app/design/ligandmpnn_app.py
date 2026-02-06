"""LigandMPNN source repo: <https://github.com/dauparas/LigandMPNN>.

## Model checkpoints

See <https://github.com/dauparas/LigandMPNN#available-models> for details.

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `LigandMPNN` | Name of the Modal app to use. |
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
"""

# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # for inputs and startup in seconds
APP_NAME = os.environ.get("MODAL_APP", "LigandMPNN")

# Volume for model cache
LIGANDMPNN_VOLUME_NAME = "ligandmpnn-models"
LIGANDMPNN_VOLUME = Volume.from_name(LIGANDMPNN_VOLUME_NAME, create_if_missing=True)

# Repositories and commit hashes
REPO_URL = "https://github.com/dauparas/LigandMPNN"
REPO_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
REPO_DIR = "/opt/LigandMPNN"

##########################################
# Image and app definitions
##########################################
runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential", "zstd")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu121",  # find best torch and CUDA versions
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {REPO_URL} {REPO_DIR}",
                f"cd {REPO_DIR}",
                f"git checkout {REPO_COMMIT}",
                "uv venv --python 3.11",
                "uv pip install -r requirements.txt",
            ),
        )
    )
    .env({"PATH": f"{REPO_DIR}/.venv/bin:$PATH"})
    .run_commands("uv pip install polars[pandas,numpy,calamine,xlsxwriter] tqdm")
    .apt_install("wget", "fd-find")
    .workdir(REPO_DIR)
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"💊 Running command: {' '.join(cmd)}")
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


def package_outputs(
    root: str | Path,
    paths_to_bundle: Iterable[str | Path],
    tar_args: list[str] | None = None,
    num_threads: int = 16,
) -> bytes:
    """Package directories into a tar.zst archive and return as bytes.

    We make an assumption here that all paths to bundle are under the same root.
    This should be safe for `collect_boltzgen_data` usage.

    Args:
        root: Root directory in the archive. All paths will be relative to this.
        paths_to_bundle: Specific paths (relative to root) to include in the archive.
        tar_args: Additional arguments to pass to `tar`.
        num_threads: Number of threads to use for compression.

    """
    import subprocess as sp
    from pathlib import Path

    root_path = Path(root)  # don't resolve, as the mapped location could be a soft link
    cmd = ["tar", "-I", f"zstd -T{num_threads}"]  # ZSTD_NBTHREADS
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-c"])

    # Our volume file structure is: outputs/[run_id]/...
    # We want to preserve the relative paths
    for p in paths_to_bundle:
        out_path = root_path.joinpath(p)
        if out_path.exists():
            cmd.append(str(out_path.relative_to(root_path.parent)))
        else:
            print(f"💊 Warning: path {out_path} does not exist and will be skipped.")

    return sp.check_output(cmd, cwd=root_path.parent)


def torch_to_numpy(pt_file: str | Path) -> dict[str, Any]:
    """Convert a PyTorch .pt file to a dictionary of numpy arrays.

    Args:
        pt_file: Path to the .pt file.

    Returns:
        A dictionary where keys are tensor names and values are lists of floats.

    """
    import torch

    pt_path = Path(pt_file)
    if not pt_path.exists():
        raise FileNotFoundError(f".pt file not found: {pt_path}")

    tensor_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    np_dict = {
        key: v.cpu().numpy().flatten() if isinstance(v, torch.Tensor) else v
        for key, v in tensor_dict.items()
    }
    return np_dict


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes={f"{REPO_DIR}/model_params": LIGANDMPNN_VOLUME},
    timeout=TIMEOUT,
    image=runtime_image,
)
def download_weights() -> None:
    """Download ProteinMPNN models into the mounted volume."""
    print("💊 Downloading boltzgen models...")
    cmd = ["bash", f"{REPO_DIR}/get_model_params.sh", f"{REPO_DIR}/model_params"]

    run_command(cmd, cwd=REPO_DIR)
    LIGANDMPNN_VOLUME.commit()
    print("💊 Model download complete")


##########################################
# Inference functions
##########################################
def build_base_command(
    run_name: str,
    script_mode: str,
    struct_bytes: bytes,
    seeds: list[int],
    cli_args: dict[str, str | int | float | bool],
    bias_aa_per_residue_bytes: bytes | None = None,
    omit_aa_per_residue_bytes: bytes | None = None,
) -> tuple[list[str], Path]:
    """Build base command for LigandMPNN execution."""
    import tempfile
    from pathlib import Path

    workdir = Path(tempfile.gettempdir()) / f"{run_name}-{script_mode}"
    for d in ("inputs", "outputs"):
        (workdir / d).mkdir(parents=True, exist_ok=True)

    # Build command
    # cli_args["--out_folder"] = str(workdir / "outputs")
    input_pdb_file = workdir / "inputs" / f"{run_name}.pdb"
    with open(input_pdb_file, "wb") as f:
        f.write(struct_bytes)
        cli_args["--pdb_path"] = str(input_pdb_file)

    if bias_aa_per_residue_bytes is not None:
        bias_aa_per_res_file = workdir / "inputs" / "bias_AA_per_residue.json"
        with open(bias_aa_per_res_file, "wb") as f:
            f.write(bias_aa_per_residue_bytes)
            cli_args["--bias_AA_per_residue"] = str(bias_aa_per_res_file)
    if omit_aa_per_residue_bytes is not None:
        omit_aa_per_res_file = workdir / "inputs" / "omit_AA_per_residue.json"
        with open(omit_aa_per_res_file, "wb") as f:
            f.write(omit_aa_per_residue_bytes)
            cli_args["--omit_AA_per_residue"] = str(omit_aa_per_res_file)

    cmd = ["python", f"{REPO_DIR}/{script_mode}.py"]
    for arg, val in cli_args.items():
        if isinstance(val, bool):
            cmd.extend([str(arg), str(int(val))])
        else:
            cmd.extend([str(arg), str(val)])

    return cmd, workdir


@app.function(
    gpu=GPU,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={f"{REPO_DIR}/model_params": LIGANDMPNN_VOLUME.read_only()},
    image=runtime_image,
)
def ligandmpnn_run(
    run_name: str,
    script_mode: str,
    struct_bytes: bytes,
    seeds: list[int],
    cli_args: dict[str, str | int | float | bool],
    bias_aa_per_residue_bytes: bytes | None = None,
    omit_aa_per_residue_bytes: bytes | None = None,
) -> bytes:
    """Run LigandMPNN with the specifi ed CLI arguments.

    Returns:
        Outputs bundled into a `.tar.zst` file.

    """
    import subprocess as sp
    import time
    from datetime import UTC, datetime

    import numpy as np
    from tqdm import tqdm

    base_cmd, workdir = build_base_command(
        run_name,
        script_mode,
        struct_bytes,
        seeds,
        cli_args,
        bias_aa_per_residue_bytes,
        omit_aa_per_residue_bytes,
    )

    log_path = workdir / "ligandmpnn-run.log"
    print(f"💊 Running LigandMPNN, saving logs to {log_path}")
    for seed in tqdm(seeds, desc="Inference seeds"):
        cmd = base_cmd + [
            "--seed",
            str(seed),
            "--out_folder",
            str(workdir / "outputs" / f"seed-{seed}"),
        ]
        with (
            sp.Popen(
                cmd,
                bufsize=1,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                encoding="utf-8",
                cwd=REPO_DIR,
            ) as p,
            open(log_path, "a", buffering=1) as log_file,
        ):
            if p.stdout is None:
                raise RuntimeError("Failed to capture stdout from the command.")

            now = time.time()
            banner = "=" * 100
            log_file.write(f"\n{banner}\nTime: {str(datetime.now(UTC))}\n")
            log_file.write(f"Running command: {' '.join(cmd)}\n{banner}\n")

            while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
                log_file.write(buffered_output)  # not realtime without volume commit
                print(buffered_output, end="", flush=True)

            log_file.write(f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n")
            log_file.write(f"Elapsed time: {time.time() - now:.2f} seconds\n")

            if p.returncode != 0:
                print(f"💊 LigandMPNN run failed. Error log is in {log_path}")
                raise sp.CalledProcessError(p.returncode, cmd)

    # Convert .pt outputs to numpy
    print("💊 Converting .pt outputs to numpy...")
    for f in (workdir / "outputs").glob("**/*.pt"):
        np_dict = torch_to_numpy(f)
        npz_path = f.with_suffix(".npz")

        np.savez(npz_path, **np_dict)
        print(f"💊 Saved numpy output: {npz_path}")
        f.unlink()  # remove .pt file

    tar_bytes = package_outputs(workdir, ["outputs", log_path.name])
    return tar_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
# https://github.com/copilot/share/423a1120-4ba0-8023-9113-00096484408d
@app.local_entrypoint()
def submit_ligandmpnn_task(
    # Input and output
    input_pdb: str,
    script_mode: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    download_models: bool = False,
    # Model configuration
    model_type: str = "soluble_mpnn",
    checkpoint: str | None = None,
    seeds: str = "0",
    batch_size: int = 1,
    number_of_batches: int = 1,
    temperature: float = 0.1,
    ligand_mpnn_use_atom_context: bool = True,
    ligand_mpnn_cutoff_for_score: float = 8.0,
    ligand_mpnn_use_side_chain_context: bool = False,
    global_transmembrane_label: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
    pack_side_chains: bool = False,
    number_of_packs_per_design: int = 4,
    sc_num_denoising_steps: int = 3,
    sc_num_samples: int = 16,
    repack_everything: bool = False,
    pack_with_ligand_context: bool = True,
    # Input-specific options
    fixed_residues: str | None = None,
    redesigned_residues: str | None = None,
    bias_aa: str | None = None,
    bias_aa_per_residue: str | None = None,
    omit_aa: str | None = None,
    omit_aa_per_residue: str | None = None,
    symmetry_residues: str | None = None,
    is_homo_oligomer: bool = False,
    chains_to_design: str | None = None,
    parse_these_chains_only: str | None = None,
    transmembrane_buried: str | None = None,
    transmembrane_interface: str | None = None,
    # Score mode arguments
    use_sequence: bool = True,
    autoregressive_score: bool = False,
    single_aa_score: bool = True,
) -> None:
    """Run a variant of the ProteinMPNN models with results saved to `out_dir`.

    Args:
        input_pdb: Path to the input PDB structure file
        script_mode: One of `run` or `score`
        out_dir: Local output directory; defaults to $PWD
        run_name: Name for this run; defaults to input structure stem
        download_models: Whether to download model weights and skip running

        model_type: One of: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn,
            global_label_membrane_mpnn, soluble_mpnn
        checkpoint: Optional path to model weights. Note that the name should match
            the `model_type` specified.
        seeds: Comma-separated random seeds for design generation
        batch_size: Number of sequence to generate per one pass
        number_of_batches: Number of times to design sequence using a chosen batch size
        temperature: Sampling temperature for design generation
        ligand_mpnn_use_atom_context: Whether to use atom-level context in LigandMPNN
        ligand_mpnn_cutoff_for_score: Cutoff in angstroms between protein and context
            atoms to select residues for reporting score
        ligand_mpnn_use_side_chain_context: Whether to use side chain atoms as ligand
            context for the fixed residues
        global_transmembrane_label: Whether to provide global label for the
            `global_label_membrane_mpnn` model. 1 - transmembrane, 0 - soluble
        parse_atoms_with_zero_occupancy: Whether to parse atoms with 0 occupancy
        pack_side_chains: Whether to run side chain packer
        number_of_packs_per_design: Number of independent side chain packing samples to return per design
        sc_num_denoising_steps: Number of denoising/recycling steps to make for side chain packing
        sc_num_samples: Number of samples to draw from a mixture distribution
            and then take a sample with the highest likelihood
        repack_everything: 1 - repacks side chains of all residues including the fixed ones;
            0 - keeps the side chains fixed for fixed residues
        pack_with_ligand_context: 1 - pack side chains using ligand context
             0 - do not use it

        fixed_residues: Space-separated list of residue to keep fixed,
            e.g. "A12 A13 A14 B2 B25"
        redesigned_residues: Space-separated list of residues to redesign,
            e.g. "A15 A16 A17 B3 B4". Everything else will be fixed.
        bias_aa: Bias generation of amino acids, e.g. "A:-1.024,P:2.34,C:-12.34"
        bias_aa_per_residue: Path to json mapping of bias,
            e.g. {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}
        omit_aa: Exclude amino acids from generation, e.g. "ACG"
        omit_aa_per_residue: Path to json mapping of amino acids to exclude,
            e.g. {'A12': 'APQ', 'A13': 'QST'}
        symmetry_residues: Add list of lists for which residues need to be symmetric,
            e.g. "A12,A13,A14|C2,C3|A5,B6"
        is_homo_oligomer: This flag will automatically set `--symmetry_residues` and
            `--symmetry_weights` to do homooligomer design with equal weighting
        chains_to_design: Specify which chains to redesign and all others will be kept fixed.
            e.g. "A,B,C,F"
        parse_these_chains_only: Provide chains letters for parsing backbones,
            e.g. "A,B,C,F"
        transmembrane_buried: Provide buried residues when using the model
            `checkpoint_per_residue_label_membrane_mpnn`, e.g. "A12 A13 A14 B2 B25"
        transmembrane_interface: Provide interface residues when using the model
            `checkpoint_per_residue_label_membrane_mpnn`, e.g. "A12 A13 A14 B2 B25"

        use_sequence: This only applies when using `script_mode` "score"!
            1 - get scores using amino acid sequence info;
            0 - get scores using backbone info only
        autoregressive_score: This only applies when using `script_mode` "score"!
            Run autoregressive scoring function: p(AA_1|backbone); p(AA_2|backbone, AA_1) etc.
        single_aa_score: This only applies when using `script_mode` "score"!
            Run single amino acid scoring function: p(AA_i|backbone, AA_{all except ith one})

    """
    from pathlib import Path

    if download_models:
        download_weights.remote()
        return

    print("🧬 Checking input arguments...")
    input_path = Path(input_pdb).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input structure file not found: {input_path}")
    if script_mode not in {"run", "score"}:
        raise ValueError(
            f"Invalid script_mode: {script_mode}. Must be 'run' or 'score'."
        )
    if run_name is None:
        run_name = input_path.stem

    score_mode = script_mode == "score"
    seeds: list[int] = [
        int(s_num) for s in seeds.split(",") if (s_num := s.strip()).isdigit()
    ]
    cli_args = {
        "--model_type": model_type,
        "--batch_size": str(batch_size),
        "--number_of_batches": str(number_of_batches),
        # 0/1 flags
        "--ligand_mpnn_use_atom_context": ligand_mpnn_use_atom_context,
        "--ligand_mpnn_cutoff_for_score": str(ligand_mpnn_cutoff_for_score),
        "--ligand_mpnn_use_side_chain_context": ligand_mpnn_use_side_chain_context,
        "--global_transmembrane_label": global_transmembrane_label,
        "--parse_atoms_with_zero_occupancy": parse_atoms_with_zero_occupancy,
    }
    # Mode-specific args
    if score_mode:
        cli_args |= {
            "--use_sequence": use_sequence,
            "--autoregressive_score": autoregressive_score,
            "--single_aa_score": single_aa_score,
        }
    else:
        cli_args |= {
            "--temperature": str(temperature),
            "--save_stats": "1",
            "--pack_side_chains": pack_side_chains,
            "--number_of_packs_per_design": str(number_of_packs_per_design),
            "--sc_num_denoising_steps": str(sc_num_denoising_steps),
            "--sc_num_samples": str(sc_num_samples),
            "--repack_everything": repack_everything,
            "--pack_with_ligand_context": pack_with_ligand_context,
        }
    # Non-default args
    if checkpoint is not None:
        cli_args[f"--checkpoint_{model_type}"] = checkpoint
    if fixed_residues is not None:
        cli_args["--fixed_residues"] = fixed_residues
    if redesigned_residues is not None:
        cli_args["--redesigned_residues"] = redesigned_residues
    if symmetry_residues is not None:
        cli_args["--symmetry_residues"] = symmetry_residues
    if is_homo_oligomer:
        cli_args["--homo_oligomer"] = "1"
    if chains_to_design is not None:
        cli_args["--chains_to_design"] = chains_to_design
    if parse_these_chains_only is not None:
        cli_args["--parse_these_chains_only"] = (
            "".join(parse_these_chains_only.split(","))
            if score_mode
            else parse_these_chains_only
        )
    if transmembrane_buried is not None:
        if model_type != "per_residue_label_membrane_mpnn":
            print(
                "⚠ --transmembrane_buried only applies when model_type == 'per_residue_label_membrane_mpnn'"
            )
        else:
            cli_args["--transmembrane_buried"] = transmembrane_buried
    if transmembrane_interface is not None:
        if model_type != "per_residue_label_membrane_mpnn":
            print(
                "⚠ --transmembrane_interface only applies when model_type == 'per_residue_label_membrane_mpnn'"
            )
        else:
            cli_args["--transmembrane_interface"] = transmembrane_interface

    # Run-mode only args
    if bias_aa is not None and not score_mode:
        cli_args["--bias_AA"] = bias_aa
    if omit_aa is not None and not score_mode:
        cli_args["--omit_AA"] = omit_aa

    bias_AA_per_residue_bytes = None
    if bias_aa_per_residue is not None and not score_mode:
        bias_AA_per_res_path = Path(bias_aa_per_residue).expanduser()
        if not bias_AA_per_res_path.exists():
            raise FileNotFoundError(
                f"Bias AA per residue file not found: {bias_AA_per_res_path}"
            )
        bias_AA_per_residue_bytes = bias_AA_per_res_path.read_bytes()

    omit_AA_per_residue_bytes = None
    if omit_aa_per_residue is not None and not score_mode:
        omit_AA_per_res_path = Path(omit_aa_per_residue).expanduser()
        if not omit_AA_per_res_path.exists():
            raise FileNotFoundError(
                f"Omit AA per residue file not found: {omit_AA_per_res_path}"
            )
        omit_AA_per_residue_bytes = omit_AA_per_res_path.read_bytes()

    print("🧬 Running LigandMPNN...")
    struct_bytes = input_path.read_bytes()
    res_bytes = ligandmpnn_run.remote(
        run_name,
        script_mode,
        struct_bytes,
        seeds,
        cli_args,
        bias_AA_per_residue_bytes,
        omit_AA_per_residue_bytes,
    )
    local_out_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else Path.cwd() / f"{run_name}-{script_mode}"
    )
    local_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧬 Downloading results for {run_name}...")
    (local_out_dir / f"{run_name}-{script_mode}.tar.zst").write_bytes(res_bytes)
    # run_command(
    #     ["modal", "volume", "get", OUTPUTS_VOLUME_NAME, str(remote_results_dir)],
    #     cwd=local_out_dir,
    # )
    print(f"🧬 Results saved to: {local_out_dir.resolve()}")
