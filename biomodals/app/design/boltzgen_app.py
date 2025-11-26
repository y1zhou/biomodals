"""BoltzGen for VHH generation."""
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
BOLTZGEN_COMMIT = "8deda18e6f29f1505e00a9cdc2e0847c2be52ffa"
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
    cpu=16.125,  # burst for tar compression
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
    import os
    import subprocess as sp
    from pathlib import Path

    root_path = Path(root)  # don't resolve, as the mapped location could be a soft link
    cmd = ["tar", "-I", f"'zstd -T{num_threads}'"]
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-cf", "-"])

    # Our volume file structure is: outputs/[run_id]/...
    # We want to preserve the relative paths
    for p in paths_to_bundle:
        out_path = root_path.joinpath(p)
        if out_path.exists():
            cmd.append(out_path.relative_to(root_path.parent))
        else:
            print(f"Warning: path {out_path} does not exist and will be skipped.")

    return sp.check_output(cmd, cwd=root_path.parent)  # noqa: S603


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
        while (buffered_output := p.stdout.readline()) != "" or (
            return_code := p.poll()
        ) is None:
            print(buffered_output, end="", flush=True)

        if return_code != 0:
            raise sp.CalledProcessError(return_code, cmd, buffered_output)


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
    (conf_path / "design-specs.yaml").write_bytes(yaml_content)

    # Write any additional files (e.g., .cif files referenced in yaml)
    for rel_path, content in additional_files.items():
        file_path = conf_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    OUTPUTS_VOLUME.commit()


@app.function(
    cpu=(0.125, 16.125),  # burst for tar compression
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
) -> bytes:
    """Collect BoltzGen output data from multiple runs."""
    from uuid import uuid4

    outdir = Path(OUTPUTS_DIR) / run_name / "outputs"
    run_ids = [uuid4().hex for _ in range(num_parallel_runs)]

    run_dirs = [outdir / run_id for run_id in run_ids]
    kwargs = {
        "input_yaml_path": outdir.parent / "inputs" / "config" / "design-specs.yaml",
        "protocol": protocol,
        "num_designs": num_designs,
        "steps": steps,
        "extra_args": extra_args,
    }
    for boltzgen_dir in boltzgen_run.map(run_dirs, kwargs=kwargs):
        print(f"BoltzGen run completed: {boltzgen_dir}")

    OUTPUTS_VOLUME.reload()
    print("Packaging BoltzGen outputs...")
    tarball_bytes = package_outputs.remote(outdir, run_ids)
    print("Packaging complete.")
    return tarball_bytes


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
    out_dir: str | Path,
    input_yaml_path: str | Path,
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
        open(log_path, "w", buffering=1) as log_file,
    ):
        now = time.time()
        log_file.write(f"Time: {str(datetime.now(UTC))}\n")
        log_file.write(f"Running command: {' '.join(cmd)}\n\n")

        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            log_file.write(buffered_output)  # not realtime without volume commit
            print(buffered_output)

        log_file.write(f"\nFinished at: {str(datetime.now(UTC))}\n")
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
    input_yaml: str,
    out_dir: str | None = None,
    num_parallel_runs: int = 1,
    download_models: bool = False,
    force_redownload: bool = False,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
) -> None:
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        out_dir: Local output directory; defaults to $PWD
        num_parallel_runs: Number of parallel runs to submit
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        protocol: Design protocol, one of: protein-anything, peptide-anything,
            protein-small_molecule, or nanobody-anything
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        extra_args: Additional CLI arguments as string
    """
    from datetime import UTC, datetime
    from pathlib import Path

    if download_models:
        boltzgen_download.remote(force=force_redownload)
        return

    # Find any file references in the yaml (path: something.cif)
    # File paths in yaml are relative to the yaml file location
    print("Checking if input yaml references additional files...")
    yaml_path = Path(input_yaml)
    yml_parser = YAMLReferenceLoader(yaml_path)
    if yml_parser.additional_files:
        print(
            f"Including additional referenced files: {list(yml_parser.additional_files.keys())}"
        )

    print(f"Submitting BoltzGen run for yaml: {input_yaml}")
    yaml_str = yaml_path.read_bytes()
    today: str = datetime.now(UTC).strftime("%Y%m%d")
    run_name = f"{today}-{yaml_path.stem}"

    prepare_boltzgen_run.remote(
        yaml_content=yaml_str,
        run_name=run_name,
        additional_files=yml_parser.additional_files,
    )

    print("Running BoltzGen...")
    outputs = collect_boltzgen_data.remote(
        run_name=run_name,
        num_parallel_runs=num_parallel_runs,
        protocol=protocol,
        num_designs=num_designs,
        steps=steps,
        extra_args=extra_args,
    )
    if out_dir is None:
        out_dir = Path.cwd()
    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    (local_out_dir / f"{run_name}.tar.zst").write_bytes(outputs)

    print(f"Results saved to: {local_out_dir}")
