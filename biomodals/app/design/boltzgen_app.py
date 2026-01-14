"""BoltzGen source repo: <https://github.com/HannesStark/boltzgen>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-yaml` | **Required** | Path to YAML design specification file. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved in a Modal volume only. |
| `--run-name` | stem name of `--input-yaml` | Optional run name used to name output directory. |
| `--num-parallel-runs` | `1` | Number of parallel runs to submit. |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download model weights and skip running. |
| `--force-redownload` | `--no-force-redownload` | Whether to force re-download of model weights even if they exist. |
| `--protocol` | `nanobody-anything` | Design protocol, one of: protein-anything, peptide-anything, protein-small_molecule, `antibody-anything`, or nanobody-anything. |
| `--num-designs` | `10` | Number of designs to generate *per run*. |
| `--steps` | `None` | Specific pipeline steps to run (e.g. "design,inverse_folding"). |
| `--extra-args` | `None` | Additional CLI arguments as a single string. |
| `--salvage-mode`/`--no-salvage-mode` | `--no-salvage-mode` | Whether to try to finish incomplete runs. |

For a complete set of BoltzGen CLI options that can be passed via `--extra-args`, see <https://github.com/HannesStark/boltzgen#all-command-line-arguments>.

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `BoltzGen` | Name of the Modal app to use. |
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
* The `--run-name` and `--salvage-mode` flags can be used together to continue previous incomplete runs. When finished, all results under the same run name will be packaged and returned.
"""

# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # for inputs and startup in seconds
APP_NAME = os.environ.get("MODAL_APP", "BoltzGen")

# Volume for model cache
BOLTZGEN_VOLUME_NAME = "boltzgen-models"
BOLTZGEN_VOLUME = Volume.from_name(BOLTZGEN_VOLUME_NAME, create_if_missing=True)
BOLTZGEN_MODEL_DIR = "/boltzgen-models"

# Volume for outputs
OUTPUTS_VOLUME_NAME = "boltzgen-outputs"
OUTPUTS_VOLUME = Volume.from_name(
    OUTPUTS_VOLUME_NAME, create_if_missing=True, version=2
)
OUTPUTS_DIR = "/boltzgen-outputs"

# Repositories and commit hashes
BOLTZGEN_REPO = "https://github.com/HannesStark/boltzgen"
BOLTZGEN_COMMIT = "a941e4eb3d4457ac6a9b636d7bfdd024df8063cf"
BOLTZGEN_REPO_DIR = "/opt/boltzgen"

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
            "UV_TORCH_BACKEND": "cu128",  # find best torch and CUDA versions
            "HF_HOME": str(BOLTZGEN_MODEL_DIR),  # store boltzgen model weights
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # speed up downloads
        }
    )
    .run_commands(
        " && ".join(
            (
                f"git clone {BOLTZGEN_REPO} {BOLTZGEN_REPO_DIR}",
                f"cd {BOLTZGEN_REPO_DIR}",
                f"git checkout {BOLTZGEN_COMMIT}",
                "uv venv --python 3.12",
                "uv pip install .",
            ),
        )
    )
    .env({"PATH": f"{BOLTZGEN_REPO_DIR}/.venv/bin:$PATH"})
    .run_commands("uv pip install polars[pandas,numpy,calamine,xlsxwriter] tqdm")
    .apt_install("fd-find")
    .workdir(BOLTZGEN_REPO_DIR)
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
@app.function(
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
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


def warmup_directory(dir_path: str | Path, file_pattern: str = ".") -> None:
    """Warm up the disk cache for all files in a directory matching a pattern."""
    cmd = [
        "fdfind",
        "-tf",
        file_pattern,
        str(dir_path),
        "-j256",
        "-x",
        "dd",
        "if={}",
        "of=/dev/null",
        "bs=1M",
        "status=none",
    ]
    run_command(cmd)


class YAMLReferenceLoader:
    """Class to load referenced files from YAML files.

    BoltzGen configs might reference other cif or yaml files.
    We need to recursively parse all yaml files to find all used cif templates.

    The file paths need to be relative to the parent directory of the
    input yaml, because we need to recreate the file structure on the remote.
    """

    def __init__(self, input_yaml_file: str | Path) -> None:
        """Initialize the loader with the input YAML file path."""
        self.input_path = Path(input_yaml_file).expanduser().resolve()
        self.ref_dir = self.input_path.parent

        # key: relative path to self.ref_dir, value: file content bytes
        self.additional_files: dict[str, bytes] = {}

        # absolute paths for tracking and recursive parsing
        self.parsed_files: set[Path] = set()
        self.queue: set[Path] = set()
        self.queue.add(self.input_path)
        self.load()

    def load(self) -> None:
        """Load referenced files from a YAML."""
        while self.queue:
            file = self.queue.pop()
            if file in self.parsed_files:
                continue

            new_ref_files = self.find_paths_from_yaml(file)
            for ref_file in new_ref_files:
                ref_path = file.parent.joinpath(ref_file).resolve()
                if ref_path.exists():
                    rel_path = ref_path.relative_to(self.ref_dir, walk_up=True)
                    self.additional_files[str(rel_path)] = ref_path.read_bytes()
                if (
                    ref_path.suffix in {".yaml", ".yml"}
                    and ref_path not in self.parsed_files
                ):
                    self.queue.add(ref_path)

    def find_paths_from_yaml(self, yaml_file: Path) -> set[Path]:
        """Load referenced files from a YAML."""
        import yaml

        yaml_path = Path(yaml_file).expanduser().resolve()
        if yaml_path in self.parsed_files:
            return set()

        with yaml_path.open() as f:
            conf = yaml.safe_load(f)

        file_refs: set[Path] = set()
        self.find_paths_in_dict(conf, yaml_path.parent, file_refs)
        self.parsed_files.add(yaml_path)
        return file_refs

    def find_paths_in_dict(
        self, yaml_content: dict, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for v in yaml_content.values():
            if isinstance(v, str):
                if (p := (ref_dir / v)).exists():
                    file_refs.add(p)
            elif isinstance(v, list):
                self.find_paths_in_list(v, ref_dir, file_refs)
            elif isinstance(v, dict):
                self.find_paths_in_dict(v, ref_dir, file_refs)
            else:
                continue

    def find_paths_in_list(
        self, sublist: list, ref_dir: Path, file_refs: set[Path]
    ) -> None:
        """Recursively find all file references in the yaml content."""
        for item in sublist:
            if isinstance(item, str):
                if (p := (ref_dir / item)).exists():
                    file_refs.add(p)
            elif isinstance(item, dict):
                self.find_paths_in_dict(item, ref_dir, file_refs)
            elif isinstance(item, list):
                self.find_paths_in_list(item, ref_dir, file_refs)
            else:
                continue


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes={BOLTZGEN_MODEL_DIR: BOLTZGEN_VOLUME}, timeout=TIMEOUT, image=runtime_image
)
def boltzgen_download(force: bool = False) -> None:
    """Download BoltzGen models into the mounted volume."""
    # Download all artifacts (~/.cache overridden to volume mount)
    print("💊 Downloading boltzgen models...")
    cmd = ["boltzgen", "download", "all", "--cache", BOLTZGEN_MODEL_DIR]
    if force:
        cmd.append("--force_download")
    run_command(cmd, cwd=BOLTZGEN_REPO_DIR)

    BOLTZGEN_VOLUME.commit()
    print("💊 Model download complete")


##########################################
# Inference functions
##########################################
@app.function(
    timeout=TIMEOUT, volumes={OUTPUTS_DIR: OUTPUTS_VOLUME}, image=runtime_image
)
def prepare_boltzgen_run(
    yaml_content: bytes, run_name: str, additional_files: dict[str, bytes]
) -> None:
    """Prepare BoltzGen input and output directories."""
    workdir = Path(OUTPUTS_DIR) / run_name
    for d in ("inputs", "outputs"):
        (workdir / d).mkdir(parents=True, exist_ok=True)

    # Write yaml to file
    conf_path = workdir / "inputs" / "config"
    conf_path.mkdir(parents=True, exist_ok=True)
    (conf_path / f"{run_name}.yaml").write_bytes(yaml_content)

    # Write any additional files (e.g., .cif files referenced in yaml)
    for rel_path, content in additional_files.items():
        file_path = conf_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    OUTPUTS_VOLUME.commit()


@app.function(
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
def collect_boltzgen_data(
    run_name: str,
    num_parallel_runs: int,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    budget: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
    salvage_mode: bool = False,
    focus_run_ids: str | None = None,
    ignore_run_ids: str | None = None,
    filter_results: bool = True,
) -> bytes | list[str]:
    """Collect BoltzGen output data from multiple runs."""
    from datetime import UTC, datetime
    from uuid import uuid4

    outdir = Path(OUTPUTS_DIR) / run_name / "outputs"
    if salvage_mode:
        all_run_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        if not all_run_dirs:
            raise RuntimeError(
                f"💊 No existing run directories found for run name '{run_name}'."
            )
        run_dirs = [
            d
            for d in all_run_dirs
            if not (
                (d_final_dir := (d / "final_ranked_designs")).exists()
                and (d_final_dir / "results_overview.pdf").exists()
            )
        ]
        run_ids = [d.name for d in all_run_dirs]
        if focus_run_ids is not None:
            focus_set = set(focus_run_ids.split(","))
            run_dirs = [d for d in run_dirs if d.name in focus_set]
            run_ids = [d for d in run_ids if d in focus_set]
        if ignore_run_ids is not None:
            ignore_set = set(ignore_run_ids.split(","))
            run_dirs = [d for d in run_dirs if d.name not in ignore_set]
            run_ids = [d for d in run_ids if d not in ignore_set]

    else:
        today: str = datetime.now(UTC).strftime("%Y%m%d")
        run_dirs = [outdir / f"{today}-{uuid4().hex}" for _ in range(num_parallel_runs)]
        run_ids = [d.name for d in run_dirs]

    kwargs = {
        "input_yaml_path": str(
            outdir.parent / "inputs" / "config" / f"{run_name}.yaml"
        ),
        "protocol": protocol,
        "num_designs": num_designs,
        "steps": steps,
        "extra_args": extra_args,
    }
    cli_args_json_path = outdir.parent / "inputs" / "config" / "cli-args.json"
    if not cli_args_json_path.exists():
        import json

        # Save a copy of the CLI args for reference
        with cli_args_json_path.open("w") as f:
            json.dump(kwargs, f, indent=2)

    if run_dirs:
        for boltzgen_dir in boltzgen_run.map(run_dirs, kwargs=kwargs):
            print(f"💊 BoltzGen run completed: {boltzgen_dir}")

    OUTPUTS_VOLUME.reload()
    if filter_results:
        # Rerun BoltzGen filters on all run IDs, and only download the designs
        # that passed all filters (also limited by the `budget`)
        print("💊 Collecting BoltzGen outputs...")
        combine_multiple_runs.remote(run_name, run_ids)
        print("💊 Filtering combined BoltzGen designs...")
        refilter_designs.remote(run_name, budget)
        OUTPUTS_VOLUME.reload()

        print("💊 Packaging filtered BoltzGen outputs...")
        tarball_bytes = package_outputs.remote(
            outdir.parent / "pass-filter-designs",
            [
                "all-designs.parquet",
                "top-designs.parquet",
                "boltzgen-cif/",
                "refold-cif/",
            ],
        )
        print("💊 Packaging complete.")
        return tarball_bytes
    else:
        print("💊 Skipping refiltering of BoltzGen outputs.")
        print(
            f"💊 Results are available at: '{outdir.relative_to(OUTPUTS_DIR)}' in volume '{OUTPUTS_VOLUME_NAME}'."
        )
        return run_ids


@app.function(
    gpu=GPU,
    cpu=1.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        BOLTZGEN_MODEL_DIR: BOLTZGEN_VOLUME.read_only(),
    },
    image=runtime_image,
)
def boltzgen_run(
    out_dir: str,
    input_yaml_path: str,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    budget: int = 1,
    steps: str | None = None,
    extra_args: str | None = None,
) -> str:
    """Run BoltzGen on a yaml specification.

    Args:
        out_dir: Output directory path
        input_yaml_path: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        budget: Number of designs to keep after filtering. This is not very useful
            here because we are likely to run multiple parallel runs and combine later.
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string

    Returns:
    -------
        Path to output directory as string.
    """
    import subprocess as sp
    import time
    from datetime import UTC, datetime

    # Build command
    cmd = [
        "boltzgen",
        "run",
        str(input_yaml_path),
        "--protocol",
        protocol,
        "--output",
        str(out_dir),
        "--num_designs",
        str(num_designs),
        "--budget",
        str(budget),
        "--cache",
        str(BOLTZGEN_MODEL_DIR),
    ]

    if steps:
        cmd.extend(["--steps", *steps.split()])
    if extra_args:
        cmd.extend(extra_args.split())

    out_path = Path(out_dir)
    # Handle preempted runs by continuing from existing output
    if out_path.exists():
        cmd.append("--reuse")
        warmup_directory(out_path)

    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "boltzgen-run.log"
    print(f"💊 Running BoltzGen, saving logs to {log_path}")
    with (
        sp.Popen(
            cmd,
            bufsize=8,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            encoding="utf-8",
            cwd=BOLTZGEN_REPO_DIR,
        ) as p,
        open(log_path, "a", buffering=1) as log_file,
    ):
        now = time.time()
        banner = "=" * 100
        log_file.write(f"\n{banner}\nTime: {str(datetime.now(UTC))}\n")
        log_file.write(f"Running command: {' '.join(cmd)}\n{banner}\n")

        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            log_file.write(buffered_output)  # not realtime without volume commit
            print(buffered_output)

        log_file.write(f"\n{banner}\nFinished at: {str(datetime.now(UTC))}\n")
        log_file.write(f"Elapsed time: {time.time() - now:.2f} seconds\n")

        if p.returncode != 0:
            print(f"💊 BoltzGen run failed. Error log is in {log_path}")
            raise sp.CalledProcessError(p.returncode, cmd)

    OUTPUTS_VOLUME.commit()
    return str(out_dir)


@app.function(
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
def combine_multiple_runs(run_name: str, run_ids: list[str]):
    """Combine outputs from multiple BoltzGen runs into a single table."""
    import gzip
    import pickle

    import polars as pl
    from tqdm import tqdm

    workdir = Path(OUTPUTS_DIR) / run_name / "outputs"
    out_dir = Path(OUTPUTS_DIR) / run_name / "combined-outputs"
    (out_dir / "refold_cif").mkdir(parents=True, exist_ok=True)
    OUTPUTS_VOLUME.reload()

    metrics_dfs: list[pl.DataFrame] = []
    ca_coords_seqs_dfs: list[pl.DataFrame] = []
    print(f"💊 Combining outputs from runs: {run_ids}")
    for run_id in run_ids:
        run_design_dir = workdir / run_id / "intermediate_designs_inverse_folded"

        # Metrics table required for downstream filtering
        metrics_df = pl.read_csv(run_design_dir / "aggregate_metrics_analyze.csv")

        # ID, seqs, and coords required for diversity
        with gzip.open(run_design_dir / "ca_coords_sequences.pkl.gz", "rb") as f:
            ca_coords_seqs_df = pl.from_pandas(pickle.load(f))  # noqa: S301

        # Prepend run_id to `id` and `file_name` columns to ensure uniqueness
        metrics_df = metrics_df.with_columns(
            pl.concat_str(pl.lit(run_id), pl.col("id"), separator="_").alias("id"),
            pl.concat_str(pl.lit(run_id), pl.col("file_name"), separator="_").alias(
                "file_name"
            ),
        )
        ca_coords_seqs_df = ca_coords_seqs_df.with_columns(
            pl.concat_str(pl.lit(run_id), pl.col("id"), separator="_").alias("id")
        )
        metrics_dfs.append(metrics_df)
        ca_coords_seqs_dfs.append(ca_coords_seqs_df)

        # Copy files to out_dir for later use
        cif_files = list(run_design_dir.glob("*.cif"))
        refold_cif_files = list(run_design_dir.glob("refold_cif/*.cif"))

        for f in tqdm(cif_files, desc=f"Copying CIFs from {run_id}"):
            dest = out_dir / f"{run_id}_{f.name}"
            if not dest.exists():
                # Make soft link instead of copy to save space
                dest.symlink_to(f)
                # shutil.copyfile(f, dest)

        for f in tqdm(refold_cif_files, desc=f"Copying refolded CIFs from {run_id}"):
            dest = out_dir / "refold_cif" / f"{run_id}_{f.name}"
            if not dest.exists():
                dest.symlink_to(f)
                # shutil.copyfile(f, dest)

    metrics_df = pl.concat(metrics_dfs, how="diagonal")
    ca_coords_seqs_df = pl.concat(ca_coords_seqs_dfs, how="vertical")
    if (not (out_dir / "aggregate_metrics_analyze.csv").exists()) or (
        pl.scan_csv(out_dir / "aggregate_metrics_analyze.csv")
        .select(pl.len())
        .collect()
        .item()
        != metrics_df.height
    ):
        metrics_df.write_csv(out_dir / "aggregate_metrics_analyze.csv")
        with gzip.open(out_dir / "ca_coords_sequences.pkl.gz", "wb") as f:
            pickle.dump(ca_coords_seqs_df.to_pandas(), f)


@app.function(
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=86400,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME},
    image=runtime_image,
)
def refilter_designs(
    run_name: str,
    budget: int = 100,
    rmsd_threshold: float = 4.0,
    modality: str = "antibody",  # or "peptide"
):
    """Refilter combined BoltzGen designs using boltzgen.task.filter.Filter."""
    import polars as pl
    from boltzgen.task.filter.filter import Filter

    workdir = Path(OUTPUTS_DIR) / run_name
    warmup_directory(workdir / "combined-outputs")
    OUTPUTS_VOLUME.reload()

    filter_task = Filter(
        design_dir=workdir / "combined-outputs",
        outdir=workdir / "refiltered",
        budget=budget,  # How many designs to subselect from all designs
        filter_cysteine=True,  # remove designs with cysteines
        use_affinity=False,  # When designing binders to small molecules this should be true
        filter_bindingsite=True,  # This filters out everything that does not have a residue within 4A of a binding site residue
        filter_designfolding=False,  # Filter by the RMSD when refolding only the designed part (usually true for proteins and false for nanobodies or peptides)
        refolding_rmsd_threshold=rmsd_threshold,
        modality=modality,
        alpha=0.001,  # for diversity quality optimization: 0 = quality-only, 1 = diversity-only
        metrics_override={  # larger value down-weights the metric's rank
            "neg_min_design_to_target_pae": 1,
            "design_to_target_iptm": 1,
            "design_ptm": 2,
            "plip_hbonds_refolded": 4,
            "plip_saltbridge_refolded": 4,
            "delta_sasa_refolded": 4,
            "neg_design_hydrophobicity": 7,
        },
        # size_buckets=[
        #     {"num_designs": 10, "min": 50, "max": 100}, # maximum number of designs that are allowed in the final selected diverse set
        #     {"num_designs": 10, "min": 100, "max": 150},
        #     {"num_designs": 10, "min": 150, "max": 200},
        # ],
        # additional_filters=[
        #     {"feature": "design_ptm", "lower_is_better": False, "threshold": 0.7},
        #     {"feature": "sheet", "lower_is_better": True, "threshold": 0.8},
        # ],
    )
    filter_task.run(jupyter_nb=False)

    # All designs
    # filter_task.outdir
    refiltered_df = pl.read_csv(
        workdir / "refiltered" / "final_ranked_designs" / "all_designs_metrics.csv"
    )

    # Final designs
    final_df = pl.read_csv(
        workdir
        / "refiltered"
        / "final_ranked_designs"
        / f"final_designs_metrics_{filter_task.budget}.csv"
    )

    out_dir = workdir / "pass-filter-designs"
    for subdir in ("boltzgen-cif", "refold-cif"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    refiltered_df.write_parquet(out_dir / "all-designs.parquet")
    final_df.write_parquet(out_dir / "top-designs.parquet")
    for r in final_df.filter("pass_filters").iter_rows(named=True):
        r_id = r["id"]
        r_cif_path = workdir / "combined-outputs" / f"{r_id}.cif"
        refold_cif_path = workdir / "combined-outputs" / "refold_cif" / f"{r_id}.cif"

        r_save_cif_path = out_dir / "boltzgen-cif" / f"{r_id}.cif"
        r_save_refold_cif_path = out_dir / "refold-cif" / f"{r_id}.cif"
        if not r_save_cif_path.exists():
            shutil.copyfile(r_cif_path, r_save_cif_path, follow_symlinks=True)
        if not r_save_refold_cif_path.exists():
            shutil.copyfile(
                refold_cif_path, r_save_refold_cif_path, follow_symlinks=True
            )


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_boltzgen_task(
    input_yaml: str | None = None,
    out_dir: str | None = None,
    run_name: str | None = None,
    num_parallel_runs: int = 1,
    download_models: bool = False,
    force_redownload: bool = False,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    budget: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
    salvage_mode: bool = False,
    focus_run_ids: str | None = None,
    ignore_run_ids: str | None = None,
    filter_results: bool = False,
) -> None:
    """Run BoltzGen with results saved as a tarball to `out_dir`.

    Args:
        input_yaml: Path to YAML design specification file
        out_dir: Local output directory; defaults to $PWD
        run_name: Name for this BoltzGen run; defaults to yaml file stem. Can be used
            together with `salvage_mode` to continue previous runs.
        num_parallel_runs: Number of parallel runs to submit
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        protocol: Design protocol, one of: protein-anything, peptide-anything,
            protein-small_molecule, or nanobody-anything
        num_designs: Number of designs to generate
        budget: Number of designs to keep after filtering
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string
        salvage_mode: Whether to only try to finish incomplete runs
        focus_run_ids: Comma-separated run IDs to focus on (only used in salvage mode)
        ignore_run_ids: Comma-separated run IDs to ignore (only used in salvage mode).
            Note that `ignore_run_ids` takes precedence over `focus_run_ids`.
        filter_results: If true, bundle top `'budget` results into a tarball and download to `out_dir`.
            Otherwise, use subprocesses to call `modal volume get` for downloads.
            This flag is useless if `out_dir` is None.
    """
    from pathlib import Path

    if download_models:
        boltzgen_download.remote(force=force_redownload)
        return

    # NOTE: make sure names are unique for different inputs
    if run_name is None:
        if input_yaml is None:
            raise ValueError("input_yaml must be provided if run_name is not set.")
        run_name = Path(input_yaml).stem

    # Prepare BoltzGen run inputs if we're not re-running incomplete jobs
    if not salvage_mode:
        # Find any file references in the yaml (path: something.cif)
        # File paths in yaml are relative to the yaml file location
        print("🧬 Checking if input yaml references additional files...")
        if input_yaml is None:
            raise ValueError("input_yaml must be provided for new BoltzGen runs.")
        yaml_path = Path(input_yaml)
        yml_parser = YAMLReferenceLoader(yaml_path)
        if yml_parser.additional_files:
            print(
                f"🧬 Including additional referenced files: {list(yml_parser.additional_files.keys())}"
            )

        print(f"🧬 Submitting BoltzGen run for yaml: {input_yaml}")
        yaml_str = yaml_path.read_bytes()

        prepare_boltzgen_run.remote(
            yaml_content=yaml_str,
            run_name=run_name,
            additional_files=yml_parser.additional_files,
        )
    else:
        print(f"🧬 Salvage mode enabled; skipping input preparation for {run_name}.")

    print("🧬 Running BoltzGen...")
    budget = min(budget, num_designs)
    outputs = collect_boltzgen_data.remote(
        run_name=run_name,
        num_parallel_runs=num_parallel_runs,
        protocol=protocol,
        num_designs=num_designs,
        budget=budget,
        steps=steps,
        extra_args=extra_args,
        salvage_mode=salvage_mode,
        focus_run_ids=focus_run_ids,
        ignore_run_ids=ignore_run_ids,
        filter_results=filter_results and out_dir is not None,
    )
    if out_dir is None:
        return

    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    if filter_results:
        (local_out_dir / f"{run_name}.tar.zst").write_bytes(outputs)
    else:
        (local_out_dir / "outputs").mkdir(exist_ok=True)
        for run_id in outputs:
            run_out_dir: Path = local_out_dir / "outputs" / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            remote_root_dir = f"{run_name}/outputs/{run_id}"
            print(f"🧬 Downloading results for run ID {run_id}...")
            for subdir in (
                "boltzgen-run.log",
                f"{run_name}.cif",
                "final_ranked_designs",
                "intermediate_designs_inverse_folded",
            ):
                if (run_out_dir / subdir).exists():
                    continue

                run_command(
                    [
                        "modal",
                        "volume",
                        "get",
                        OUTPUTS_VOLUME_NAME,
                        f"{remote_root_dir}/{subdir}",
                    ],
                    cwd=run_out_dir,
                )

    print(f"🧬 Results saved to: {local_out_dir}")
