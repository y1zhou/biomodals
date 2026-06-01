"""ShortMD workflow for parallel short GROMACS production replicates.

This proof-of-concept accepts a directory of PDB files, prepares each structure
once with the GROMACS app, clones the prepared production inputs into replicate
run directories, and runs many short production trajectories in parallel through
the reusable Biomodals workflow runtime.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import modal

from biomodals.app.bioinfo import gromacs_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.helper.volume_run import volume_path_from_mount_path
from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    NodeExecutionPolicy,
    NodePlacement,
    VolumePath,
)
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
)

DEPENDENCY_APPS = ("gromacs",)
CONF = AppConfig(
    tags={"depends_on": ",".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="ShortMDWorkflow",
    package_name="biomodals-shortmd-workflow",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
GROMACS_OUTPUT_VOLUME = gromacs_app.CONF.output_volume
GROMACS_OUTPUT_VOLUME_NAME = gromacs_app.CONF.output_volume_name
GROMACS_OUTPUT_MOUNTPOINT = gromacs_app.CONF.output_volume_mountpoint


@dataclass(frozen=True)
class ShortMDGromacsSettings:
    """Shared GROMACS arguments for ShortMD prep and production nodes."""

    simulation_time_ns: int = 2
    run_pdbfixer: bool = False
    cpu_only: bool = False
    num_threads: int = 16
    use_openmp_threads: bool = False
    ld_seed: int = -1
    gen_seed: int = -1
    genion_seed: int = 0
    save_processed_traj: bool = True
    make_figures: bool = True


@dataclass(frozen=True)
class ShortMDModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    clear: modal.Function
    clone: modal.Function
    prepare_cpu: modal.Function
    prepare_gpu: modal.Function
    production_cpu: modal.Function
    production_gpu: modal.Function
    collect_stats: modal.Function


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes={GROMACS_OUTPUT_MOUNTPOINT: GROMACS_OUTPUT_VOLUME},
)
def clear_shortmd_gromacs_run(run_name: str) -> None:
    """Remove one ShortMD-managed GROMACS run directory from the app volume."""
    safe_run_name = sanitize_filename(run_name)
    GROMACS_OUTPUT_VOLUME.reload()
    run_dir = Path(GROMACS_OUTPUT_MOUNTPOINT) / safe_run_name
    if run_dir.is_dir():
        shutil.rmtree(run_dir)
    elif run_dir.exists():
        run_dir.unlink()
    GROMACS_OUTPUT_VOLUME.commit()


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes={GROMACS_OUTPUT_MOUNTPOINT: GROMACS_OUTPUT_VOLUME},
)
def clone_prepared_shortmd_run(
    source_storage_path: str,
    source_run_name: str,
    replicate_run_name: str,
    overwrite: bool = False,
) -> str:
    """Clone prepared GROMACS inputs into a ShortMD replicate directory."""
    source_storage_path = VolumePath(
        volume_name=GROMACS_OUTPUT_VOLUME_NAME,
        path=source_storage_path,
    ).path
    source_run_name = sanitize_filename(source_run_name)
    replicate_run_name = sanitize_filename(replicate_run_name)
    GROMACS_OUTPUT_VOLUME.reload()

    volume_root = Path(GROMACS_OUTPUT_MOUNTPOINT)
    source_dir = volume_root / source_storage_path
    replicate_dir = volume_root / replicate_run_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prepared GROMACS run not found: {source_dir}")

    if overwrite and replicate_dir.exists():
        shutil.rmtree(replicate_dir)

    created_clone = False
    if not replicate_dir.exists():
        shutil.copytree(source_dir, replicate_dir)
        created_clone = True
    else:
        replicate_dir.mkdir(parents=True, exist_ok=True)

    source_pdb = replicate_dir / f"{source_run_name}.pdb"
    if not source_pdb.exists():
        source_pdb = source_dir / f"{source_run_name}.pdb"
    if not source_pdb.exists():
        raise FileNotFoundError(f"Prepared PDB not found: {source_pdb}")
    shutil.copy2(source_pdb, replicate_dir / f"{replicate_run_name}.pdb")

    source_tpr = replicate_dir / f"production_{source_run_name}.tpr"
    if not source_tpr.exists():
        source_tpr = source_dir / f"production_{source_run_name}.tpr"
    if not source_tpr.exists():
        raise FileNotFoundError(f"Prepared production TPR not found: {source_tpr}")
    shutil.copy2(source_tpr, replicate_dir / f"production_{replicate_run_name}.tpr")

    if created_clone:
        keep_tpr = f"production_{replicate_run_name}.tpr"
        for path in replicate_dir.glob("production_*"):
            if path.name != keep_tpr and (path.is_file() or path.is_symlink()):
                path.unlink()
        for pattern in (
            "rmsd_production_*",
            "rg_production_*",
            "rmsf_production_*",
            "production_*_nopbc*",
            "production_*_last_frame.pdb",
        ):
            for path in replicate_dir.glob(pattern):
                if path.is_file() or path.is_symlink():
                    path.unlink()

    GROMACS_OUTPUT_VOLUME.commit()
    return str(replicate_dir)


@dataclass
class ShortMDPrepNode(AppBackedNode):
    """Workflow node that prepares one PDB for GROMACS production replicates."""

    pdb_content: bytes
    run_name: str
    modal_namespace: ShortMDModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    overwrite_existing: bool = False
    gromacs: ShortMDGromacsSettings = field(default_factory=ShortMDGromacsSettings)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RESUME
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run GROMACS prep and return a workflow artifact for the run directory."""
        safe_run_name = sanitize_filename(self.run_name)
        if self.overwrite_existing:
            self.modal_namespace.clear.remote(run_name=safe_run_name)
        app_function = (
            self.modal_namespace.prepare_cpu
            if self.gromacs.cpu_only
            else self.modal_namespace.prepare_gpu
        )
        remote_workdir = app_function.remote(
            pdb_content=self.pdb_content,
            run_name=safe_run_name,
            simulation_time_ns=self.gromacs.simulation_time_ns,
            run_pdbfixer=self.gromacs.run_pdbfixer,
            num_threads=self.gromacs.num_threads,
            use_openmp_threads=self.gromacs.use_openmp_threads,
            ld_seed=self.gromacs.ld_seed,
            gen_seed=self.gromacs.gen_seed,
            genion_seed=self.gromacs.genion_seed,
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="prepared_gromacs_run",
                    kind=ArtifactKind.DIRECTORY,
                    storage=volume_path_from_mount_path(
                        remote_path=str(remote_workdir),
                        mount_root=GROMACS_OUTPUT_MOUNTPOINT,
                        volume_name=GROMACS_OUTPUT_VOLUME_NAME,
                    ),
                    metadata={"stage": "prep", "run_name": safe_run_name},
                )
            ],
        )


@dataclass
class ShortMDCloneNode(WorkflowNativeNode):
    """Workflow-native adapter that clones prepared inputs for one replicate."""

    source_run_name: str
    replicate_run_name: str
    modal_namespace: ShortMDModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    overwrite_clone: bool = False
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RESUME
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Clone prepared inputs into a replicate run directory."""
        prepared_artifacts = context.inputs.get("prepared") or []
        if len(prepared_artifacts) != 1:
            raise ValueError(
                "ShortMD clone node requires exactly one prepared input artifact"
            )
        prepared_artifact = prepared_artifacts[0]
        if prepared_artifact.storage.volume_name != GROMACS_OUTPUT_VOLUME_NAME:
            raise ValueError(
                "ShortMD prepared artifact volume does not match the GROMACS "
                f"output volume: {prepared_artifact.storage.volume_name}"
            )
        safe_source_run_name = sanitize_filename(
            str(prepared_artifact.metadata.get("run_name") or self.source_run_name)
        )
        safe_replicate_run_name = sanitize_filename(self.replicate_run_name)
        remote_workdir = self.modal_namespace.clone.remote(
            source_storage_path=prepared_artifact.storage.path,
            source_run_name=safe_source_run_name,
            replicate_run_name=safe_replicate_run_name,
            overwrite=self.overwrite_clone,
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="cloned_gromacs_run",
                    kind=ArtifactKind.DIRECTORY,
                    storage=volume_path_from_mount_path(
                        remote_path=str(remote_workdir),
                        mount_root=GROMACS_OUTPUT_MOUNTPOINT,
                        volume_name=GROMACS_OUTPUT_VOLUME_NAME,
                    ),
                    metadata={
                        "stage": "clone",
                        "run_name": safe_replicate_run_name,
                        "source_run_name": safe_source_run_name,
                    },
                )
            ],
        )


@dataclass
class ShortMDReplicateNode(AppBackedNode):
    """Workflow node that runs one short production replicate through GROMACS."""

    source_run_name: str
    replicate_run_name: str
    modal_namespace: ShortMDModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    gromacs: ShortMDGromacsSettings = field(default_factory=ShortMDGromacsSettings)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RESUME
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Launch one GROMACS production run, then collect trajectory stats."""
        cloned_artifacts = context.inputs.get("cloned") or []
        if len(cloned_artifacts) != 1:
            raise ValueError(
                "ShortMD replicate node requires exactly one cloned input artifact"
            )
        cloned_artifact = cloned_artifacts[0]
        if cloned_artifact.storage.volume_name != GROMACS_OUTPUT_VOLUME_NAME:
            raise ValueError(
                "ShortMD cloned artifact volume does not match the GROMACS "
                f"output volume: {cloned_artifact.storage.volume_name}"
            )
        safe_source_run_name = sanitize_filename(
            str(cloned_artifact.metadata.get("source_run_name") or self.source_run_name)
        )
        safe_replicate_run_name = sanitize_filename(
            str(cloned_artifact.metadata.get("run_name") or self.replicate_run_name)
        )
        app_function = (
            self.modal_namespace.production_cpu
            if self.gromacs.cpu_only
            else self.modal_namespace.production_gpu
        )
        _ = app_function.remote(
            run_name=safe_replicate_run_name,
            simulation_time_ns=self.gromacs.simulation_time_ns,
            num_threads=self.gromacs.num_threads,
            use_openmp_threads=self.gromacs.use_openmp_threads,
        )
        remote_workdir = self.modal_namespace.collect_stats.remote(
            "production_",
            run_name=safe_replicate_run_name,
            save_processed_traj=self.gromacs.save_processed_traj,
            make_figures=self.gromacs.make_figures,
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="gromacs_production",
                    kind=ArtifactKind.DIRECTORY,
                    storage=volume_path_from_mount_path(
                        remote_path=str(remote_workdir),
                        mount_root=GROMACS_OUTPUT_MOUNTPOINT,
                        volume_name=GROMACS_OUTPUT_VOLUME_NAME,
                    ),
                    metadata={
                        "stage": "production",
                        "run_name": safe_replicate_run_name,
                        "source_run_name": safe_source_run_name,
                    },
                )
            ],
        )


@dataclass
class ShortMDSummaryNode(WorkflowNativeNode):
    """Workflow-native node that emits a manifest of production replicates."""

    replicates: int
    max_parallel: int

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Write a Markdown summary of all replicate output artifacts."""
        artifacts = [
            artifact
            for artifacts in context.inputs.values()
            for artifact in artifacts
            if artifact.kind == ArtifactKind.DIRECTORY
        ]
        artifacts.sort(key=lambda artifact: artifact.metadata.get("run_name", ""))
        lines = [
            "# ShortMD Workflow Summary",
            "",
            f"- Replicates per input: {self.replicates}",
            f"- Max parallel workflow nodes: {self.max_parallel}",
            "",
            "| Source run | Replicate run | Volume | Path |",
            "| --- | --- | --- | --- |",
        ]
        for artifact in artifacts:
            source_run_name = str(artifact.metadata.get("source_run_name") or "")
            run_name = str(artifact.metadata.get("run_name") or artifact.artifact_id)
            lines.append(
                "| "
                f"{source_run_name} | "
                f"{run_name} | "
                f"{artifact.storage.volume_name} | "
                f"{artifact.storage.path} |"
            )
        summary = "\n".join(lines) + "\n"
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="shortmd_summary",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=summary.encode("utf-8"),
                        filename="shortmd-summary.md",
                        media_type="text/markdown",
                    ),
                    metadata={
                        "replicates": str(self.replicates),
                        "max_parallel": str(self.max_parallel),
                    },
                )
            ],
        )


def discover_pdb_inputs(input_dir: str | Path) -> list[tuple[str, bytes]]:
    """Return ``(filename, bytes)`` pairs for PDB files in a directory."""
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(input_path)
    pdb_paths = list(input_path.glob("*.pdb"))
    if not pdb_paths:
        raise ValueError(f"No PDB files found in {input_path}")
    return [(path.name, path.read_bytes()) for path in pdb_paths]


def build_shortmd_workflow(
    *,
    input_pdbs: list[tuple[str, bytes]],
    run_namespace: str | None = None,
    replicates: int = 50,
    simulation_time_ns: int = 2,
    run_pdbfixer: bool = False,
    cpu_only: bool = False,
    num_threads: int = 16,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
    max_parallel: int = 16,
    overwrite_existing: bool = False,
) -> Workflow:
    """Build a ShortMD workflow DAG from local PDB payloads."""
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    workflow = Workflow("shortmd")
    safe_run_namespace = (
        sanitize_filename(run_namespace) if run_namespace is not None else None
    )
    gromacs = ShortMDGromacsSettings(
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
    )
    modal_namespace = ShortMDModalNamespace(
        clear=clear_shortmd_gromacs_run,
        clone=clone_prepared_shortmd_run,
        prepare_cpu=gromacs_app.prepare_tpr_cpu,
        prepare_gpu=gromacs_app.prepare_tpr_gpu,
        production_cpu=gromacs_app.production_run_cpu,
        production_gpu=gromacs_app.production_run_gpu,
        collect_stats=gromacs_app.collect_traj_stats,
    )
    used_run_names: set[str] = set()
    replicate_handles = {}

    for file_name, pdb_content in input_pdbs:
        pdb_run_name = sanitize_filename(Path(file_name).stem)
        run_name = (
            f"{safe_run_namespace}-{pdb_run_name}"
            if safe_run_namespace is not None
            else pdb_run_name
        )
        if run_name in used_run_names:
            raise ValueError(f"Duplicate sanitized PDB run name: {run_name}")
        used_run_names.add(run_name)
        prep = workflow.add_node(
            ShortMDPrepNode(
                pdb_content=pdb_content,
                run_name=run_name,
                modal_namespace=modal_namespace,
                overwrite_existing=overwrite_existing,
                gromacs=gromacs,
            ),
            id=f"prep-{run_name}",
        )
        for replicate_idx in range(1, replicates + 1):
            replicate_run_name = f"{run_name}-r{replicate_idx:03d}"
            clone = workflow.add_node(
                ShortMDCloneNode(
                    source_run_name=run_name,
                    replicate_run_name=replicate_run_name,
                    modal_namespace=modal_namespace,
                    overwrite_clone=overwrite_existing,
                ),
                id=f"clone-{replicate_run_name}",
                inputs={"prepared": prep.outputs(kind=ArtifactKind.DIRECTORY)},
            )
            replicate = workflow.add_node(
                ShortMDReplicateNode(
                    source_run_name=run_name,
                    replicate_run_name=replicate_run_name,
                    modal_namespace=modal_namespace,
                    gromacs=gromacs,
                ),
                id=f"replicate-{replicate_run_name}",
                inputs={"cloned": clone.outputs(kind=ArtifactKind.DIRECTORY)},
            )
            replicate_handles[replicate_run_name] = replicate

    workflow.add_node(
        ShortMDSummaryNode(replicates=replicates, max_parallel=max_parallel),
        id="summary",
        inputs={
            replicate_run_name: handle.outputs(kind=ArtifactKind.DIRECTORY)
            for replicate_run_name, handle in replicate_handles.items()
        },
    )
    return workflow


@app.local_entrypoint()
def submit_shortmd_workflow(
    input_dir: str,
    run_id: str | None = None,
    replicates: int = 50,
    simulation_time_ns: int = 2,
    run_pdbfixer: bool = False,
    cpu_only: bool = False,
    num_threads: int = 16,
    use_openmp_threads: bool = False,
    ld_seed: int = -1,
    gen_seed: int = -1,
    genion_seed: int = 0,
    force: bool = False,
    wait: bool = True,
    max_parallel: int = 16,
) -> None:
    """Run ShortMD production replicate workflow for a directory of PDB files.

    Args:
        input_dir: Directory containing `.pdb` files. Each filename stem becomes
            the prepared GROMACS run name.
        run_id: Stable workflow run id for durable ledger state. Defaults to
            the input directory name.
        replicates: Number of short production replicates per input PDB.
        simulation_time_ns: Production simulation length in nanoseconds for
            each replicate and the prepared production TPR.
        run_pdbfixer: Whether to run PDBFixer during preparation.
        cpu_only: Whether to run GROMACS preparation and production on CPU only.
        num_threads: Number of CPU threads to pass to GROMACS.
        use_openmp_threads: Whether to use OpenMP threading in GROMACS.
        ld_seed: Random seed for Langevin dynamics during preparation.
        gen_seed: Random seed for initial velocity generation during preparation.
        genion_seed: Random seed for ion placement during preparation.
        force: Replace an existing workflow run ledger before running.
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_parallel: Maximum number of ready workflow nodes to execute
            concurrently in one scheduler wave.
    """
    input_path = Path(input_dir).expanduser().resolve()
    input_pdbs = discover_pdb_inputs(input_path)
    resolved_run_id = sanitize_filename(run_id or input_path.name)
    workflow = build_shortmd_workflow(
        input_pdbs=input_pdbs,
        run_namespace=resolved_run_id,
        replicates=replicates,
        simulation_time_ns=simulation_time_ns,
        run_pdbfixer=run_pdbfixer,
        cpu_only=cpu_only,
        num_threads=num_threads,
        use_openmp_threads=use_openmp_threads,
        ld_seed=ld_seed,
        gen_seed=gen_seed,
        genion_seed=genion_seed,
        max_parallel=max_parallel,
        overwrite_existing=force,
    )

    orchestrator_handle = orchestrator.WorkflowOrchestrator()
    orchestrator_kwargs = {
        "workflow": workflow,
        "run_id": resolved_run_id,
        "force": force,
        "max_ready_workers": max_parallel,
    }
    print(
        f"Submitting ShortMD workflow '{resolved_run_id}' with "
        f"{len(input_pdbs)} input PDB(s), {replicates} replicate(s) each",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(
            orchestrator_handle.run.remote(**orchestrator_kwargs)
        )
    else:
        function_call = orchestrator_handle.run.spawn(**orchestrator_kwargs)
        result = str(getattr(function_call, "object_id", function_call))
    if isinstance(result, AppRunResult):
        print(f"ShortMD workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"ShortMD workflow run submitted. FunctionCall id: {result}", flush=True)
