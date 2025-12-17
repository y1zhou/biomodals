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
    cmd = ["tar", "--zstd"]
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-cf", "-"])

    # Our volume file structure is: outputs/[run_id]/...
    # We want to preserve the relative paths
    for p in paths_to_bundle:
        out_path = root_path.joinpath(p)
        if out_path.exists():
            cmd.append(str(out_path.relative_to(root_path.parent)))
        else:
            print(f"Warning: path {out_path} does not exist and will be skipped.")

    return sp.check_output(
        cmd, cwd=root_path.parent, env=os.environ | {"ZSTD_NBTHREADS": str(num_threads)}
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
    print("Downloading boltzgen models...")
    cmd = ["boltzgen", "download", "all", "--cache", BOLTZGEN_MODEL_DIR]
    if force:
        cmd.append("--force_download")
    run_command(cmd, cwd=BOLTZGEN_REPO_DIR)

    BOLTZGEN_VOLUME.commit()
    print("Model download complete")


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
    steps: str | None = None,
    extra_args: str | None = None,
    salvage_mode: bool = False,
    package_results: bool = True,
) -> bytes:
    """Collect BoltzGen output data from multiple runs."""
    from datetime import UTC, datetime
    from uuid import uuid4

    outdir = Path(OUTPUTS_DIR) / run_name / "outputs"
    if salvage_mode:
        all_run_dirs = [d for d in outdir.iterdir() if d.is_dir()]
        run_dirs = [
            d
            for d in all_run_dirs
            if not (
                (d_final_dir := (d / "final_ranked_designs")).exists()
                and (d_final_dir / "results_overview.pdf").exists()
            )
        ]
        run_ids = [d.name for d in all_run_dirs]
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
            print(f"BoltzGen run completed: {boltzgen_dir}")

    OUTPUTS_VOLUME.reload()
    if package_results:
        print("Packaging BoltzGen outputs...")
        tarball_bytes = package_outputs.remote(
            outdir,
            run_ids,
            tar_args=[
                "--exclude",
                "intermediate_designs",  # intermediate_designs_inverse_folded is enough
                "--exclude",
                "lightning_logs",
                "--exclude",
                "metrics_tmp",  # design_seq, ca_coords, ca_coords_refolded
            ],
        )
        print("Packaging complete.")
        return tarball_bytes
    else:
        print("Skipping packaging of BoltzGen outputs.")
        print(
            f"Results are available at: '{outdir.relative_to(OUTPUTS_DIR)}' in volume '{OUTPUTS_VOLUME_NAME}'."
        )


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
    steps: str | None = None,
    extra_args: str | None = None,
) -> str:
    """Run BoltzGen on a yaml specification.

    Args:
        out_dir: Output directory path
        input_yaml_path: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
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

    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "boltzgen-run.log"
    print(f"Running BoltzGen, saving logs to {log_path}")
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
            print(f"BoltzGen run failed. Error log is in {log_path}")
            raise sp.CalledProcessError(p.returncode, cmd)

    OUTPUTS_VOLUME.commit()
    return str(out_dir)


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
    steps: str | None = None,
    extra_args: str | None = None,
    salvage_mode: bool = False,
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
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string
        salvage_mode: Whether to only try to finish incomplete runs
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
        print("Checking if input yaml references additional files...")
        if input_yaml is None:
            raise ValueError("input_yaml must be provided for new BoltzGen runs.")
        yaml_path = Path(input_yaml)
        yml_parser = YAMLReferenceLoader(yaml_path)
        if yml_parser.additional_files:
            print(
                f"Including additional referenced files: {list(yml_parser.additional_files.keys())}"
            )

        print(f"Submitting BoltzGen run for yaml: {input_yaml}")
        yaml_str = yaml_path.read_bytes()

        prepare_boltzgen_run.remote(
            yaml_content=yaml_str,
            run_name=run_name,
            additional_files=yml_parser.additional_files,
        )
    else:
        print(f"Salvage mode enabled; skipping input preparation for {run_name}.")

    print("Running BoltzGen...")
    outputs = collect_boltzgen_data.remote(
        run_name=run_name,
        num_parallel_runs=num_parallel_runs,
        protocol=protocol,
        num_designs=num_designs,
        steps=steps,
        extra_args=extra_args,
        salvage_mode=salvage_mode,
        package_results=out_dir is not None,
    )
    if out_dir is not None:
        local_out_dir = Path(out_dir).expanduser().resolve()
        local_out_dir.mkdir(parents=True, exist_ok=True)
        (local_out_dir / f"{run_name}.tar.zst").write_bytes(outputs)

        print(f"Results saved to: {local_out_dir}")
