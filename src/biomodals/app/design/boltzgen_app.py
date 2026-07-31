"""BoltzGen source repo: <https://github.com/HannesStark/boltzgen>.

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
* The `--run-name` and `--salvage-mode` flags can be used together to continue previous incomplete runs. When finished, all results under the same run name will be packaged and returned.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415
import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.app.design.boltzgen.execution_contracts import (
    boltzgen_output_claim_key,
    is_boltzgen_run_complete,
    write_boltzgen_task_publication,
    write_collection_publication,
)
from biomodals.app.design.boltzgen.execution_coordinator import (
    BoltzGenExecutionCoordinator,
)
from biomodals.app.design.boltzgen.execution_request import (
    load_execution_request_from_volume,
    prepare_execution_request,
    stage_execution_request,
)
from biomodals.execution import DeploymentIdentity, ExecutionSnapshot, RunStatus
from biomodals.execution.modal import (
    ModalCallDriver,
    deployed_execution_coordinator,
    development_modal_call_driver,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout, volume_path_from_mount_path
from biomodals.helper.constant import MAX_TIMEOUT, MODEL_VOLUME
from biomodals.helper.output_claim import acquire_output_claim
from biomodals.helper.shell import (
    copy_files,
    package_outputs,
    run_command,
    sanitize_filename,
    warmup_directory,
)

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="BoltzGen",
    repo_url="https://github.com/y1zhou/boltzgen",
    repo_commit_hash="7327d07d14a8a70ec20f967afbb6e3d842b9c11f",
    package_name="boltzgen",
    version="0.3.2",
    python_version="3.12",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
)

##########################################
# Image and app definitions
##########################################
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential")
    .env(CONF.default_env)
    .uv_pip_install("tqdm")
    .uv_pip_install(f"git+{CONF.repo_url}@{CONF.repo_commit_hash}")
    .workdir(str(CONF.git_clone_dir))
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.boltzgen")
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
BOLTZGEN_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_boltzgen_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8


##########################################
# Helper functions
##########################################
@app.function(
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
    image=runtime_image,
)
def package_outputs_helper(
    root: str | Path,
    paths_to_bundle: Iterable[str | Path],
    tar_args: list[str] | None = None,
    num_threads: int = 16,
) -> bytes:
    """Modal runner to package directories into a tar.zst archive and return as bytes."""
    warmup_directory(root)
    return package_outputs(
        root,
        paths_to_bundle=paths_to_bundle,
        tar_args=tar_args,
        num_threads=num_threads,
    )


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
    volumes=CONF.mounts(model_volume=True, model_ro=False, is_huggingface=True),
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MAX_TIMEOUT,
)
def boltzgen_download(force: bool = False) -> None:
    """Download BoltzGen models into the mounted volume."""
    # Download all artifacts to $HF_HOME
    print("💊 Downloading boltzgen models...")
    cmd = ["boltzgen", "download", "all"]
    if force:
        cmd.append("--force_download")
    run_command(cmd)

    MODEL_VOLUME.commit()
    print("💊 Model download complete")


##########################################
# Inference functions
##########################################
@app.function(timeout=CONF.timeout, volumes=CONF.mounts(output_volume=True))
def prepare_boltzgen_run(
    yaml_content: bytes, run_name: str, additional_files: dict[str, bytes]
) -> None:
    """Prepare BoltzGen input and output directories."""
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    for path in (layout.inputs_dir, layout.outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Write yaml to file
    conf_path = layout.inputs_dir / "config"
    conf_path.mkdir(parents=True, exist_ok=True)
    (conf_path / f"{run_name}.yaml").write_bytes(yaml_content)

    # Write any additional files (e.g., .cif files referenced in yaml)
    for rel_path, content in additional_files.items():
        file_path = conf_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    CONF.output_volume.commit()


@app.function(timeout=CONF.timeout, volumes=CONF.mounts(output_volume=True))
def get_run_ids(
    run_name: str,
    num_parallel_runs: int,
    salvage_mode: bool = False,
    focus_run_ids: str | None = None,
    ignore_run_ids: str | None = None,
    skip_finished: bool = False,
) -> list[str]:
    """Gather BoltzGen run IDs to collect data for."""
    from datetime import UTC, datetime
    from uuid import uuid4

    CONF.output_volume.reload()
    outdir = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_name
    ).outputs_dir

    if not salvage_mode:
        today: str = datetime.now(UTC).strftime("%Y%m%d")
        return [f"{today}-{uuid4().hex}" for _ in range(num_parallel_runs)]

    if not outdir.exists():
        raise RuntimeError(
            f"💊 No existing run directories found for run name '{run_name}'."
        )
    all_run_dirs = sorted(
        (d for d in outdir.iterdir() if d.is_dir()), key=lambda d: d.name
    )
    if not all_run_dirs:
        raise RuntimeError(
            f"💊 No existing run directories found for run name '{run_name}'."
        )
    if skip_finished:
        all_run_dirs = [d for d in all_run_dirs if not is_boltzgen_run_complete(d)]

    run_ids = [d.name for d in all_run_dirs]
    if focus_run_ids is not None:
        focus_set = set(focus_run_ids.split(","))
        run_ids = [d for d in run_ids if d in focus_set]
    if ignore_run_ids is not None:
        ignore_set = set(ignore_run_ids.split(","))
        run_ids = [d for d in run_ids if d not in ignore_set]

    return run_ids


@app.function(
    memory=(128, 65536),  # reserve 128MB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def collect_boltzgen_data(
    run_name: str,
    run_ids: list[str],
    task_fingerprints: dict[str, str],
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    budget: int = 10,
    steps: str | None = None,
    extra_args: str | None = None,
    filter_results: bool = True,
    filter_rmsd_threshold: float = 4.0,
    publication_path: str = "",
) -> dict[str, object]:
    """Validate completed Tasks and publish their deterministic collection."""
    if not publication_path:
        raise ValueError("publication_path cannot be empty")
    out_vol = CONF.output_volume
    out_vol.reload()
    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    outdir = layout.outputs_dir
    config_dir = layout.inputs_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    all_run_dirs = [outdir / x for x in run_ids]
    if set(task_fingerprints) != set(run_ids):
        raise ValueError("BoltzGen Task fingerprints do not match the requested runs")
    incomplete = [
        run_dir.name
        for run_dir in all_run_dirs
        if not is_boltzgen_run_complete(
            run_dir,
            task_fingerprint=task_fingerprints[run_dir.name],
        )
    ]
    if incomplete:
        raise RuntimeError(
            "Cannot collect incomplete BoltzGen Tasks: " + ", ".join(incomplete)
        )

    kwargs = {
        "input_yaml_path": str(config_dir / f"{run_name}.yaml"),
        "protocol": protocol,
        "num_designs": num_designs,
        "steps": steps,
        "extra_args": extra_args,
    }
    cli_args_json_path = config_dir / "cli-args.json"
    if not cli_args_json_path.exists():
        # Save a copy of the CLI args for reference
        with cli_args_json_path.open("wb") as f:
            f.write(orjson.dumps(kwargs, option=orjson.OPT_INDENT_2))

    vol_path = volume_path_from_mount_path(
        str(outdir), CONF.output_volume_mountpoint, CONF.output_volume_name
    )
    publication: dict[str, object] = {
        "run_name": run_name,
        "run_ids": run_ids,
        "filtered": filter_results,
    }
    if filter_results:
        print(f"💊 Collecting BoltzGen outputs in {vol_path}...")
        combine_multiple_runs.get_raw_f()(run_name, run_ids)
        print("💊 Filtering combined BoltzGen designs...")
        refilter_designs.get_raw_f()(run_name, budget, filter_rmsd_threshold)

        print("💊 Packaging filtered BoltzGen outputs...")
        tarball_bytes = package_outputs_helper.get_raw_f()(
            layout.run_root / "pass-filter-designs",
            [
                "all-designs.parquet",
                "top-designs.parquet",
                "boltzgen-cif/",
                "refold-cif/",
            ],
        )
        archive_path = Path(CONF.output_volume_mountpoint).joinpath(
            *Path(publication_path).with_suffix(".tar.zst").parts
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(tarball_bytes)
        publication.update({
            "archive_path": str(
                archive_path.relative_to(CONF.output_volume_mountpoint)
            ),
            "archive_size_bytes": len(tarball_bytes),
        })
    else:
        print(f"💊 Results are available at: {vol_path}.")
    out_vol.commit()
    record = write_collection_publication(
        CONF.output_volume_mountpoint,
        publication_path,
        publication,
    )
    out_vol.commit()
    return record


@app.function(
    gpu=CONF.gpu,
    cpu=1.125,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True, model_volume=True, is_huggingface=True),
    single_use_containers=True,
)
def run_boltzgen_task(
    out_dir: str,
    input_yaml_path: str,
    claim_owner: str,
    task_fingerprint: str,
    protocol: str = "nanobody-anything",
    num_designs: int = 10,
    budget: int = 1,
    steps: str | None = None,
    extra_args: str | None = None,
    replace_claim_owner: str | None = None,
) -> str:
    """Run one independently tracked BoltzGen Task."""
    output_root = Path(CONF.output_volume_mountpoint).resolve()
    out_path = Path(out_dir).resolve()
    input_path = Path(input_yaml_path).resolve()
    try:
        out_path.relative_to(output_root)
        input_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "BoltzGen worker paths must stay inside the output Volume"
        ) from error
    CONF.output_volume.reload()
    if is_boltzgen_run_complete(
        out_path,
        task_fingerprint=task_fingerprint,
    ):
        return str(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    acquire_output_claim(
        BOLTZGEN_OUTPUT_CLAIMS,
        claim_key=boltzgen_output_claim_key(
            out_path,
            output_root=CONF.output_volume_mountpoint,
        ),
        owner=claim_owner,
        replace_owner=replace_claim_owner,
    )

    cmd = [
        "boltzgen",
        "run",
        str(input_yaml_path),
        f"--protocol={protocol}",
        f"--output={out_dir}",
        f"--num_designs={num_designs}",
        f"--budget={budget}",
    ]
    if steps:
        cmd.extend(["--steps", *steps.split()])
    if extra_args:
        cmd.extend(extra_args.split())
    cmd.append("--reuse")
    warmup_directory(out_path)

    log_path = out_path / "boltzgen-run.log"
    log_vol_path = volume_path_from_mount_path(
        str(log_path), CONF.output_volume_mountpoint, CONF.output_volume_name
    )
    print(f"💊 Running BoltzGen, saving logs to {log_vol_path}")
    run_command(cmd, output_mode="log", log_file=log_path, cwd=out_path)
    if not is_boltzgen_run_complete(out_path):
        raise RuntimeError("BoltzGen returned without its final publication")
    CONF.output_volume.commit()
    write_boltzgen_task_publication(
        out_path,
        task_fingerprint=task_fingerprint,
    )
    CONF.output_volume.commit()
    return str(out_dir)


@app.function(
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(output_volume=True),
)
def combine_multiple_runs(run_name: str, run_ids: list[str]):
    """Combine outputs from multiple BoltzGen runs into a single table."""
    import gzip
    import pickle

    import polars as pl
    from tqdm import tqdm

    layout = AppRunLayout.from_run_root(Path(CONF.output_volume_mountpoint) / run_name)
    workdir = layout.outputs_dir
    out_dir = layout.run_root / "combined-outputs"
    (out_dir / "refold_cif").mkdir(parents=True, exist_ok=True)
    CONF.output_volume.reload()

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
            if not dest.exists(follow_symlinks=False):
                # Make soft link instead of copy to save space
                dest.symlink_to(f.relative_to(out_dir, walk_up=True))
                # shutil.copyfile(f, dest)

        refold_cif_out_dir = out_dir / "refold_cif"
        for f in tqdm(refold_cif_files, desc=f"Copying refolded CIFs from {run_id}"):
            dest = refold_cif_out_dir / f"{run_id}_{f.name}"
            if not dest.exists(follow_symlinks=False):
                dest.symlink_to(f.relative_to(refold_cif_out_dir, walk_up=True))
                # shutil.copyfile(f, dest)

    metrics_df = pl.concat(metrics_dfs, how="diagonal")
    ca_coords_seqs_df = pl.concat(ca_coords_seqs_dfs, how="vertical")
    if (not (out_dir / "aggregate_metrics_analyze.csv").exists()) or (
        pl
        .scan_csv(out_dir / "aggregate_metrics_analyze.csv")
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
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def refilter_designs(
    run_name: str,
    budget: int = 100,
    rmsd_threshold: float = 2.5,
    modality: str = "antibody",  # or "peptide"
):
    """Refilter combined BoltzGen designs using boltzgen.task.filter.Filter."""
    import polars as pl
    from boltzgen.task.filter.filter import Filter  # type: ignore[ty:unresolved-import]

    workdir = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_name
    ).run_root
    # warmup_directory(workdir / "outputs", file_pattern=r"\.cif$")

    filter_kwargs = dict(
        design_dir=workdir / "combined-outputs",
        outdir=workdir / "refiltered",
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
        # additional_filters=[
        #     {"feature": "design_ptm", "lower_is_better": False, "threshold": 0.7},
        #     {"feature": "sheet", "lower_is_better": True, "threshold": 0.8},
        # ],
    )

    # If budget <= 0, collect all designs that pass filters and skip sub-selection
    all_design_metrics_file = (
        workdir / "refiltered" / "final_ranked_designs" / "all_designs_metrics.csv"
    )
    if keep_all_pass_filter := (budget <= 0):
        print("💊 Budget<=0; will collect all designs that pass filters.")
        budget = 1

    filter_task = Filter(**filter_kwargs, budget=budget)
    filter_task.run(jupyter_nb=False)

    # All designs
    # filter_task.outdir
    refiltered_df = pl.read_csv(all_design_metrics_file)

    # Final designs
    final_df = pl.read_csv(
        all_design_metrics_file.parent
        / f"final_designs_metrics_{filter_task.budget}.csv"
    )

    out_dir = workdir / "pass-filter-designs"
    for subdir in ("boltzgen-cif", "refold-cif"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    refiltered_df.write_parquet(out_dir / "all-designs.parquet")
    final_df.write_parquet(out_dir / "top-designs.parquet")

    copy_cif_ids = (
        refiltered_df.filter("pass_filters").get_column("id")
        if keep_all_pass_filter
        else final_df.filter("pass_filters").get_column("id")
    )
    boltzgen_cif_dir = workdir / "combined-outputs"
    boltzgen_refold_cif_dir = boltzgen_cif_dir / "refold_cif"
    src_dest_mapping: dict[str | Path, str | Path] = {}
    for r_id in copy_cif_ids:
        r_cif_path = boltzgen_cif_dir / f"{r_id}.cif"
        refold_cif_path = boltzgen_refold_cif_dir / f"{r_id}.cif"

        r_save_cif_path = out_dir / "boltzgen-cif" / f"{r_id}.cif"
        r_save_refold_cif_path = out_dir / "refold-cif" / f"{r_id}.cif"
        src_dest_mapping[r_cif_path] = r_save_cif_path
        src_dest_mapping[refold_cif_path] = r_save_refold_cif_path

        # if not r_save_cif_path.exists():
        #     shutil.copyfile(r_cif_path, r_save_cif_path, follow_symlinks=True)
        # if not r_save_refold_cif_path.exists():
        #     shutil.copyfile(
        #         refold_cif_path, r_save_refold_cif_path, follow_symlinks=True
        #     )
    copy_files(src_dest_mapping, cp_args="-anL")


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with BoltzGen's worker functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh the output Volume before accepting lifecycle methods."""
        self._coordinator_adapter = None
        self._development = None
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionSnapshot:
        """Drive one staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionSnapshot:
        """Read this Run's durable kernel snapshot."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionSnapshot:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        return self._adapter().resume()

    @modal.method()
    def restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive one compatible Successor Run."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
        workload_plan_fingerprint: str,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
    ) -> ExecutionSnapshot:
        """Create a compatible Successor while inferring predecessor identity."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            expected_workload_plan_fingerprint=workload_plan_fingerprint,
        )

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached child calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return (
            UUID(self.execution_run_id),
            DeploymentIdentity(
                self.deployment_environment,
                self.deployment_name,
                self.deployment_version,
            ),
        )

    def _adapter(
        self,
        *,
        development: bool = False,
    ) -> BoltzGenExecutionCoordinator:
        adapter = getattr(self, "_coordinator_adapter", None)
        selected_mode = getattr(self, "_development", None)
        if adapter is not None:
            if selected_mode != development:
                raise ValueError("Coordinator execution mode cannot change in place")
            return adapter
        execution_run_id, deployment = self._identity()
        adapter = BoltzGenExecutionCoordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=Path(CONF.output_volume_mountpoint),
            output_volume=CONF.output_volume,
            modal_driver=_coordinator_modal_driver(development=development),
        )
        self._coordinator_adapter = adapter
        self._development = development
        return adapter


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source development handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "run_boltzgen_task": run_boltzgen_task,
            "collect_boltzgen_data": collect_boltzgen_data,
        },
        workload_name="BoltzGen",
    )


def _execution_coordinator_handle(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    use_deployed_coordinator: bool,
):
    """Resolve this run's exact deployed or current-source coordinator."""
    if use_deployed_coordinator:
        return deployed_execution_coordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
        )
    return ExecutionCoordinator(
        execution_run_id=str(execution_run_id),
        deployment_environment=deployment.environment,
        deployment_name=deployment.deployment_name,
        deployment_version=deployment.deployment_version,
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
    filter_rmsd_threshold: float = 2.5,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Run BoltzGen with results saved as a tarball to `out_dir`.

    Args:
        input_yaml: Path to YAML design specification file.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in a Modal volume only.
        run_name: Name for this BoltzGen run; defaults to yaml file stem. Can
            be used together with `salvage_mode` to continue previous runs.
        num_parallel_runs: Number of parallel runs to submit. Due to the stochastic
            nature of BoltzGen, running multiple parallel runs with the same
            YAML input would generate different results. Also caps concurrent
            BoltzGen child runs in salvage mode.
        download_models: Whether to download model weights and skip running.
        force_redownload: Whether to force re-download of model weights even if they exist.
        protocol: Design protocol, one of: protein-anything, peptide-anything,
            protein-small_molecule, antibody-anything, or nanobody-anything.
        num_designs: Number of designs to generate *per run*. Note that this
            is just the number of generated designs, and there is no guarantee
            that all designs will pass the filtering criteria.
        budget: Number of designs to keep after filtering. It is recommended
            to set this to a reasonably large number (e.g. 100) to get the best
            results, and do further filtering locally after combining multiple runs.
            If set to <=0, all designs that pass BoltzGen filters will be collected.
        steps: Specific pipeline steps to run (e.g. "design inverse_folding").
        extra_args: Additional CLI arguments as a string. See
            <https://github.com/HannesStark/boltzgen#all-command-line-arguments>.
        salvage_mode: Whether to only try to finish incomplete runs. In salvage mode,
            the app will look for existing run outputs under the same `run_name`
            and only run BoltzGen for runs that are not completed.
        focus_run_ids: Comma-separated run IDs to focus on
            (only used in `salvage_mode`).
        ignore_run_ids: Comma-separated run IDs to ignore
            (only used in `salvage_mode`).
            Note that `ignore_run_ids` takes precedence over `focus_run_ids`.
        filter_results: If true, bundle top `budget` results into a tarball and download to `out_dir`.
            Otherwise, use subprocesses to call `modal volume get` for downloads.
            This flag is useless if `out_dir` is not specified.
        filter_rmsd_threshold: RMSD threshold for refiltering designs. This is
            only used if `filter_results` is true. The RMSD calculation is
            between the designed structure and the refolded structure.
        use_deployed_coordinator: Target the exact deployed coordinator. The
            `biomodals app run` client supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric Modal deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    if download_models:
        boltzgen_download.remote(force=force_redownload)
        return

    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    predecessor_request = (
        None
        if predecessor_execution_run_id is None
        else load_execution_request_from_volume(
            CONF.output_volume,
            predecessor_execution_run_id,
        )
    )
    yaml_content: bytes | None = None
    additional_files: dict[str, bytes] = {}
    if input_yaml is not None:
        yaml_path = Path(input_yaml).expanduser().resolve()
        yaml_content = yaml_path.read_bytes()
        additional_files = YAMLReferenceLoader(yaml_path).additional_files
        if run_name is None:
            run_name = yaml_path.stem
    elif predecessor_request is not None:
        if run_name is None:
            run_name = predecessor_request.run_name
    elif not salvage_mode:
        raise ValueError("input_yaml must be provided for a new BoltzGen run.")

    if run_name is None:
        raise ValueError("run_name must be provided when input_yaml is omitted.")
    run_name = sanitize_filename(run_name)

    if predecessor_execution_run_id is None:
        if yaml_content is None:
            config_path = f"{run_name}/inputs/config/{run_name}.yaml"
            try:
                yaml_content = b"".join(CONF.output_volume.read_file(config_path))
            except FileNotFoundError as error:
                raise RuntimeError(
                    "Salvage mode could not load the staged BoltzGen YAML: "
                    f"{config_path}"
                ) from error
            print(f"🧬 Salvage mode enabled; reusing staged inputs for {run_name}.")
        else:
            print("🧬 Checking if input yaml references additional files...")
        if additional_files:
            print(f"🧬 Including additional referenced files: {list(additional_files)}")

        if not salvage_mode:
            print(f"🧬 Staging BoltzGen inputs for yaml: {input_yaml}")
            prepare_boltzgen_run.remote(
                yaml_content=yaml_content,
                run_name=run_name,
                additional_files=additional_files,
            )
        run_ids = tuple(
            get_run_ids.remote(
                run_name=run_name,
                num_parallel_runs=num_parallel_runs,
                salvage_mode=salvage_mode,
                focus_run_ids=focus_run_ids,
                ignore_run_ids=ignore_run_ids,
            )
        )
    else:
        if predecessor_request is None:
            raise RuntimeError("Predecessor request was not loaded")
        run_ids = predecessor_request.run_ids

    budget = min(budget, num_designs)
    if yaml_content is None:
        if predecessor_request is None:
            raise RuntimeError("BoltzGen request has no scientific input")
        request = replace(
            predecessor_request,
            max_active_provider_calls=num_parallel_runs,
            max_active_gpu_provider_calls=num_parallel_runs,
            replace_claim_owners=(),
        )
    else:
        app_version = CONF.version
        repo_commit_hash = CONF.repo_commit_hash
        if app_version is None or repo_commit_hash is None:
            raise RuntimeError("BoltzGen scientific version metadata is incomplete")
        request = prepare_execution_request(
            run_name=run_name,
            run_ids=run_ids,
            yaml_content=yaml_content,
            additional_files=additional_files,
            protocol=protocol,
            num_designs=num_designs,
            budget=budget,
            steps=steps,
            extra_args=extra_args,
            filter_results=filter_results and (out_dir is not None),
            filter_rmsd_threshold=filter_rmsd_threshold,
            app_version=app_version,
            repo_commit_hash=repo_commit_hash,
            max_active_provider_calls=num_parallel_runs,
            max_active_gpu_provider_calls=num_parallel_runs,
        )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    if predecessor_execution_run_id is None:
        stage_execution_request(CONF.output_volume, execution_run_id, request)
    coordinator = _execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(
            development=not use_deployed_coordinator,
        )
    else:
        call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
            workload_plan_fingerprint=(
                request.execution_plan.workload_plan_fingerprint
            ),
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=(request.max_active_gpu_provider_calls),
        )
    print(f"Execution Run ID: {execution_run_id}")
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    snapshot = call.get()
    if snapshot.run.status != RunStatus.SUCCEEDED:
        diagnostic = snapshot.run.status_message or (
            snapshot.run.status_reason.value
            if snapshot.run.status_reason is not None
            else snapshot.run.status.value
        )
        raise RuntimeError(
            f"{CONF.name} Execution Run ended as "
            f"{snapshot.run.status.value}: {diagnostic}"
        )
    completed_request = load_execution_request_from_volume(
        CONF.output_volume,
        execution_run_id,
    )
    if out_dir is None:
        return

    local_out_dir = Path(out_dir).expanduser().resolve()
    local_out_dir.mkdir(parents=True, exist_ok=True)
    if completed_request.filter_results:
        archive_path = completed_request.collection_publication_path.with_suffix(
            ".tar.zst"
        )
        archive = b"".join(CONF.output_volume.read_file(archive_path.as_posix()))
        (local_out_dir / f"{completed_request.run_name}.tar.zst").write_bytes(archive)
    else:
        (local_out_dir / "outputs").mkdir(exist_ok=True)
        for run_id in completed_request.run_ids:
            run_out_dir: Path = local_out_dir / "outputs" / run_id
            run_out_dir.mkdir(parents=True, exist_ok=True)
            remote_root_dir = f"{completed_request.run_name}/outputs/{run_id}"
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
                        CONF.output_volume_name,
                        f"{remote_root_dir}/{subdir}",
                    ],
                    cwd=run_out_dir,
                )

    print(f"🧬 Results saved to: {local_out_dir}")
