"""RFdiffusion to LigandMPNN workflow.

This workflow fans out slow RFdiffusion trajectories for one input PDB, then
runs one LigandMPNN design node for each RFdiffusion output PDB.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import modal

from biomodals.app.design import ligandmpnn_app, rfdiffusion_app
from biomodals.execution import DeploymentIdentity
from biomodals.helper import patch_image_for_helper
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
    WorkflowArtifact,
)
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    RemoteNodeCall,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
    print_workflow_dag,
)
from biomodals.workflow.core.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)

DEPENDENCY_APPS = ("rfdiffusion", "ligandmpnn")
CONF = AppConfig(
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="RFDLigandMPNNWorkflow",
    package_name="biomodals-rfd-ligandmpnn-workflow",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .uv_pip_install("gemmi")
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
RFDIFFUSION_OUTPUT_VOLUME = rfdiffusion_app.CONF.output_volume
RFDIFFUSION_OUTPUT_VOLUME_NAME = rfdiffusion_app.CONF.output_volume_name
RFDIFFUSION_OUTPUT_MOUNTPOINT = rfdiffusion_app.CONF.output_volume_mountpoint


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes={RFDIFFUSION_OUTPUT_MOUNTPOINT: RFDIFFUSION_OUTPUT_VOLUME},
)
def check_rfd_ligandmpnn_external_artifact(
    artifact: WorkflowArtifact,
) -> ArtifactAvailability:
    """Validate RFdiffusion artifacts referenced by the workflow runtime."""
    RFDIFFUSION_OUTPUT_VOLUME.reload()
    return check_external_artifact_status(
        artifact,
        workflow_volume_name=orchestrator.OUT_VOLUME_NAME,
        volume_roots={RFDIFFUSION_OUTPUT_VOLUME_NAME: RFDIFFUSION_OUTPUT_MOUNTPOINT},
    )


@dataclass(frozen=True)
class LigandMPNNDesignSettings:
    """Shared LigandMPNN arguments for each RFdiffusion output structure."""

    model_type: str
    seeds: tuple[int, ...]
    batch_size: int
    number_of_batches: int
    sc_num_samples: int
    number_of_packs_per_design: int


@app.function(
    image=runtime_image,
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes={RFDIFFUSION_OUTPUT_MOUNTPOINT: RFDIFFUSION_OUTPUT_VOLUME},
)
def select_rfdiffusion_design(
    *, rfd_output_storage_path: str, rfd_run_name: str, design_index: int
) -> AppRunResult:
    """Read one RFdiffusion PDB/TRB pair and infer LigandMPNN redesign residues."""
    storage_path = VolumePath(
        volume_name=RFDIFFUSION_OUTPUT_VOLUME_NAME, path=rfd_output_storage_path
    ).path
    safe_run_name = sanitize_filename(rfd_run_name)
    RFDIFFUSION_OUTPUT_VOLUME.reload()
    scaffolds_dir = Path(RFDIFFUSION_OUTPUT_MOUNTPOINT) / storage_path

    def design_sort_key(path: Path) -> tuple[int, str]:
        prefix = f"{safe_run_name}_"
        suffix = path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem
        try:
            return int(suffix), path.name
        except ValueError:
            return 1_000_000, path.name

    # Select PDB and TRB based on stable index
    pdbs = sorted(scaffolds_dir.glob(f"{safe_run_name}_*.pdb"), key=design_sort_key)
    if not pdbs:
        pdbs = sorted(scaffolds_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No RFdiffusion PDB outputs found in {scaffolds_dir}")
    if design_index < 0 or design_index >= len(pdbs):
        raise IndexError(
            f"RFdiffusion design index {design_index} is out of range for "
            f"{len(pdbs)} PDB output(s) in {scaffolds_dir}"
        )
    pdb_path = pdbs[design_index]
    trb_path = pdb_path.with_suffix(".trb")
    if not trb_path.exists():
        raise FileNotFoundError(f"RFdiffusion TRB metadata not found: {trb_path}")
    pdb_bytes = pdb_path.read_bytes()

    # TRB files are RFdiffusion's own pickled run metadata from the app volume.
    # We parse the file to find the fixed residues and RFdiffusion-perturbed positions
    trb_metadata = pickle.loads(trb_path.read_bytes())  # noqa: S301
    if not isinstance(trb_metadata, dict):
        raise TypeError(f"RFdiffusion TRB metadata must be a dict: {trb_path}")
    try:
        is_fixed = list(trb_metadata["mask_1d"])
    except KeyError as exc:
        raise ValueError(
            f"RFdiffusion TRB metadata missing mask_1d: {trb_path}"
        ) from exc
    except TypeError as exc:
        raise TypeError(
            f"RFdiffusion TRB metadata mask_1d must be iterable: {trb_path}"
        ) from exc

    # We use gemmi to parse the PDB and extract residue labels for comparison
    # Also sanity check the B-factors of the perturbed positions
    import gemmi

    structure = gemmi.read_structure(str(pdb_path), format=gemmi.CoorFormat.Detect)
    structure.setup_entities()
    if len(structure) == 0:
        raise ValueError(f"No model found in RFdiffusion PDB output: {pdb_path}")

    residue_records: list[tuple[str, float]] = []
    for chain in structure[0]:
        chain_id = chain.name.strip()
        for residue in chain:
            res_idx = residue.seqid.num  # assume icode is empty
            if res_idx is None:
                raise ValueError(f"Invalid residue ID in chain {chain_id}: {residue}")
            label = f"{chain_id}{res_idx}"
            residue_b_factor = float(max((atom.b_iso for atom in residue), default=0.0))
            residue_records.append((label, residue_b_factor))
    if len(is_fixed) != len(residue_records):
        raise ValueError(
            f"RFdiffusion TRB mask_1d length {len(is_fixed)} does not match "
            f"{len(residue_records)} residue(s) parsed from {pdb_path}: {trb_path}"
        )

    fixed_labels: list[str] = []
    redesigned_labels: list[str] = []
    bfactor_mismatches: list[str] = []
    for idx, (label, residue_b_factor) in enumerate(residue_records):
        if is_fixed[idx]:
            fixed_labels.append(label)
            if residue_b_factor == 0.0:
                bfactor_mismatches.append(label)
            continue
        if residue_b_factor != 0.0:
            bfactor_mismatches.append(label)
        redesigned_labels.append(label)
    if bfactor_mismatches:
        print(
            "💊 RFdiffusion redesign-set B-factor sanity check mismatches: "
            f"{' '.join(bfactor_mismatches)}",
            flush=True,
        )
    if not redesigned_labels:
        raise ValueError(f"No redesigned residues inferred for {pdb_path}")
    redesigned_residues = " ".join(redesigned_labels)
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="selected_rfd_design",
                kind=ArtifactKind.STRUCTURES,
                storage=InlineBytes(
                    data=pdb_bytes,
                    filename=pdb_path.name,
                    media_type="chemical/x-pdb",
                ),
                metadata={
                    "rfd_run_name": safe_run_name,
                    "design_index": str(design_index),
                    "redesigned_residues": redesigned_residues,
                    "trb_name": trb_path.name,
                },
            )
        ],
    )


@dataclass
class RFdiffusionTrajectoryNode(AppBackedNode):
    """Workflow node that runs one RFdiffusion trajectory."""

    pdb_content: bytes
    input_pdb_name: str
    run_name: str
    contigs: str
    hotspot_res: str
    num_designs: int
    noise_scale_ca: float = 1.0
    noise_scale_frame: float = 1.0
    rfd_args: str = ""

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare the RFdiffusion call for kernel submission."""
        safe_run_name = sanitize_filename(self.run_name)
        return RemoteNodeCall(
            function_name="rfdiffusion_infer",
            uses_gpu=True,
            kwargs={
                "input_pdb_bytes": self.pdb_content,
                "input_pdb_name": self.input_pdb_name,
                "run_name": safe_run_name,
                "hydra_overrides": (
                    rfdiffusion_app.build_rfdiffusion_hydra_overrides(
                        contigs=self.contigs,
                        num_designs=self.num_designs,
                        hotspot_res=self.hotspot_res,
                        noise_scale_ca=self.noise_scale_ca,
                        noise_scale_frame=self.noise_scale_frame,
                        rfd_args=self.rfd_args,
                    )
                ),
            },
        )


@dataclass
class RFdiffusionSelectionNode(AppBackedNode):
    """Select one RFdiffusion design through a tracked provider call."""

    rfd_run_name: str
    design_index: int

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare one deterministic RFdiffusion output selection."""
        rfd_artifacts = context.inputs.get("rfd_output") or []
        if len(rfd_artifacts) != 1:
            raise ValueError(
                "RFdiffusion selection node requires exactly one RFdiffusion output"
            )
        artifact = rfd_artifacts[0]
        if artifact.storage.volume_name != RFDIFFUSION_OUTPUT_VOLUME_NAME:
            raise ValueError(
                "RFdiffusion artifact volume does not match the RFdiffusion "
                f"output volume: {artifact.storage.volume_name}"
            )
        run_name = sanitize_filename(
            str(artifact.metadata.get("run_name") or self.rfd_run_name)
        )
        return RemoteNodeCall(
            function_name="select_rfdiffusion_design",
            uses_gpu=False,
            kwargs={
                "rfd_output_storage_path": artifact.storage.path,
                "rfd_run_name": run_name,
                "design_index": self.design_index,
            },
        )


@dataclass
class LigandMPNNDesignNode(AppBackedNode):
    """Workflow node that designs sequences for one RFdiffusion output PDB."""

    rfd_run_name: str
    design_index: int
    run_name: str
    settings: LigandMPNNDesignSettings

    def _select_ligandmpnn_inputs(
        self, context: NodeRunContext
    ) -> tuple[bytes, str, dict[str, str]]:
        selected_artifacts = context.inputs.get("selected_design") or []
        if len(selected_artifacts) != 1:
            raise ValueError(
                "LigandMPNN design node requires exactly one selected RFdiffusion "
                "design"
            )
        selected = selected_artifacts[0]
        pdb_bytes = context.resolve_workflow_artifact(selected).read_bytes()
        safe_rfd_run_name = sanitize_filename(
            str(selected.metadata.get("rfd_run_name") or self.rfd_run_name)
        )
        redesigned_residues = str(selected.metadata["redesigned_residues"])
        return (
            pdb_bytes,
            redesigned_residues,
            {
                "rfd_run_name": safe_rfd_run_name,
                "design_index": str(self.design_index),
                "redesigned_residues": redesigned_residues,
            },
        )

    def prepare_remote(self, context: NodeRunContext) -> RemoteNodeCall:
        """Prepare the LigandMPNN call for kernel submission."""
        pdb_bytes, redesigned_residues, metadata = self._select_ligandmpnn_inputs(
            context
        )
        cli_args = ligandmpnn_app.build_ligandmpnn_cli_args(
            script_mode="run",
            model_type=self.settings.model_type,
            batch_size=self.settings.batch_size,
            number_of_batches=self.settings.number_of_batches,
            parse_atoms_with_zero_occupancy=True,
            pack_side_chains=True,
            number_of_packs_per_design=self.settings.number_of_packs_per_design,
            sc_num_samples=self.settings.sc_num_samples,
            repack_everything=True,
            redesigned_residues=redesigned_residues,
        )
        return RemoteNodeCall(
            function_name="ligandmpnn_run",
            uses_gpu=True,
            kwargs={
                "run_name": sanitize_filename(self.run_name),
                "script_mode": "run",
                "struct_bytes": pdb_bytes,
                "seeds": list(self.settings.seeds),
                "cli_args": cli_args,
            },
            metadata=metadata,
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Attach workflow selection metadata to LigandMPNN app outputs."""
        result = AppRunResult.model_validate(result)
        for output in result.outputs:
            output.metadata.setdefault("rfd_run_name", str(metadata["rfd_run_name"]))
            output.metadata.setdefault("design_index", str(metadata["design_index"]))
            output.metadata.setdefault(
                "redesigned_residues",
                str(metadata["redesigned_residues"]),
            )
        return result


@dataclass
class RFDLigandMPNNSummaryNode(WorkflowNativeNode):
    """Workflow-native node that emits a manifest of LigandMPNN design outputs."""

    num_rfdiffusion_trajectories: int
    num_rfdiffusion_designs: int
    max_parallel: int

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Write a Markdown summary of LigandMPNN output artifacts."""
        artifacts = [
            artifact
            for artifacts in context.inputs.values()
            for artifact in artifacts
            if artifact.kind == ArtifactKind.ARCHIVE
        ]
        artifacts.sort(
            key=lambda artifact: (
                str(artifact.metadata.get("rfd_run_name") or ""),
                str(artifact.metadata.get("design_index") or ""),
                str(artifact.metadata.get("run_name") or artifact.artifact_id),
            )
        )
        lines = [
            "# RFdiffusion + LigandMPNN Workflow Summary",
            "",
            f"- RFdiffusion trajectories: {self.num_rfdiffusion_trajectories}",
            f"- RFdiffusion designs per trajectory: {self.num_rfdiffusion_designs}",
            f"- Max parallel workflow nodes: {self.max_parallel}",
            "",
            "| RFdiffusion run | Design index | LigandMPNN run | Volume | Path |",
            "| --- | --- | --- | --- | --- |",
        ]
        for artifact in artifacts:
            rfd_run_name = str(artifact.metadata.get("rfd_run_name") or "")
            design_index = str(artifact.metadata.get("design_index") or "")
            run_name = str(artifact.metadata.get("run_name") or artifact.artifact_id)
            lines.append(
                "| "
                f"{rfd_run_name} | "
                f"{design_index} | "
                f"{run_name} | "
                f"{artifact.storage.volume_name} | "
                f"{artifact.storage.path} |"
            )
        summary = "\n".join(lines) + "\n"
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="rfd_ligandmpnn_summary",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=summary.encode("utf-8"),
                        filename="rfd-ligandmpnn-summary.md",
                        media_type="text/markdown",
                    ),
                    metadata={
                        "num_rfdiffusion_trajectories": str(
                            self.num_rfdiffusion_trajectories
                        ),
                        "num_rfdiffusion_designs": str(self.num_rfdiffusion_designs),
                        "max_parallel": str(self.max_parallel),
                    },
                )
            ],
        )


def _parse_seeds(seeds: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(seeds, str):
        parsed = tuple(int(seed) for part in seeds.split(",") if (seed := part.strip()))
    else:
        parsed = tuple(int(seed) for seed in seeds)
    if not parsed:
        raise ValueError("seeds must contain at least one integer")
    return parsed


def build_rfd_ligandmpnn_workflow(
    *,
    input_pdb: tuple[str, bytes],
    contigs: str,
    hotspot_res: str,
    run_namespace: str | None = None,
    num_rfdiffusion_trajectories: int = 1,
    num_rfdiffusion_designs: int = 1,
    model_type: str = "protein_mpnn",
    seeds: Sequence[int] = (0,),
    batch_size: int = 1,
    number_of_batches: int = 1,
    sc_num_samples: int = 16,
    number_of_packs_per_design: int = 4,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
    rfd_args: str = "",
    max_parallel: int = 16,
) -> Workflow:
    """Build an RFdiffusion to LigandMPNN workflow DAG from one PDB payload."""
    if num_rfdiffusion_trajectories < 1:
        raise ValueError("num_rfdiffusion_trajectories must be at least 1")
    if num_rfdiffusion_designs < 1:
        raise ValueError("num_rfdiffusion_designs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if number_of_batches < 1:
        raise ValueError("number_of_batches must be at least 1")
    settings = LigandMPNNDesignSettings(
        model_type=model_type,
        seeds=_parse_seeds(seeds),
        batch_size=batch_size,
        number_of_batches=number_of_batches,
        sc_num_samples=sc_num_samples,
        number_of_packs_per_design=number_of_packs_per_design,
    )
    input_pdb_name, pdb_content = input_pdb
    input_stem = sanitize_filename(Path(input_pdb_name).stem)
    safe_run_namespace = (
        sanitize_filename(run_namespace) if run_namespace is not None else input_stem
    )
    workflow = Workflow("rfd_ligandmpnn")
    mpnn_handles = {}

    for trajectory_idx in range(1, num_rfdiffusion_trajectories + 1):
        rfd_run_name = f"{safe_run_namespace}-rfd{trajectory_idx:03d}"
        rfd = workflow.add_node(
            RFdiffusionTrajectoryNode(
                pdb_content=pdb_content,
                input_pdb_name=input_pdb_name,
                run_name=rfd_run_name,
                contigs=contigs,
                hotspot_res=hotspot_res,
                num_designs=num_rfdiffusion_designs,
                noise_scale_ca=noise_scale_ca,
                noise_scale_frame=noise_scale_frame,
                rfd_args=rfd_args,
            ),
            id=f"rfd-{rfd_run_name}",
        )
        for design_index in range(num_rfdiffusion_designs):
            mpnn_run_name = f"{rfd_run_name}-d{design_index:03d}-mpnn"
            selection = workflow.add_node(
                RFdiffusionSelectionNode(
                    rfd_run_name=rfd_run_name,
                    design_index=design_index,
                ),
                id=f"select-{rfd_run_name}-d{design_index:03d}",
                inputs={"rfd_output": rfd.outputs(kind=ArtifactKind.DIRECTORY)},
            )
            mpnn = workflow.add_node(
                LigandMPNNDesignNode(
                    rfd_run_name=rfd_run_name,
                    design_index=design_index,
                    run_name=mpnn_run_name,
                    settings=settings,
                ),
                id=f"ligandmpnn-{rfd_run_name}-d{design_index:03d}",
                inputs={
                    "selected_design": selection.outputs(kind=ArtifactKind.STRUCTURES)
                },
            )
            mpnn_handles[f"{rfd_run_name}-d{design_index:03d}"] = mpnn

    workflow.add_node(
        RFDLigandMPNNSummaryNode(
            num_rfdiffusion_trajectories=num_rfdiffusion_trajectories,
            num_rfdiffusion_designs=num_rfdiffusion_designs,
            max_parallel=max_parallel,
        ),
        id="summary",
        inputs={
            design_id: handle.outputs(kind=ArtifactKind.ARCHIVE)
            for design_id, handle in mpnn_handles.items()
        },
    )
    return workflow


@app.local_entrypoint()
def submit_rfd_ligandmpnn_workflow(
    input_pdb: str,
    contigs: str,
    hotspot_res: str,
    run_id: str | None = None,
    num_rfdiffusion_trajectories: int = 1,
    num_rfdiffusion_designs: int = 1,
    model_type: str = "protein_mpnn",
    seeds: str = "0",
    batch_size: int = 1,
    number_of_batches: int = 1,
    sc_num_samples: int = 16,
    number_of_packs_per_design: int = 4,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
    rfd_args: str = "",
    wait: bool = True,
    max_parallel: int = 16,
    dry_run: bool = False,
    strict_artifact_checks: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "development",
    deployment_name: str | None = None,
    deployment_version: int = 1,
) -> None:
    """Run RFdiffusion trajectories followed by LigandMPNN sequence design.

    Args:
        input_pdb: Local input PDB path.
        contigs: RFdiffusion contig string, passed as `contigmap.contigs`.
        hotspot_res: RFdiffusion hotspot residues, comma- or space-separated.
        run_id: Stable workflow run id. Defaults to the input PDB stem.
        num_rfdiffusion_trajectories: Independent RFdiffusion nodes to fan out.
        num_rfdiffusion_designs: `inference.num_designs` per RFdiffusion node.
        model_type: LigandMPNN model type.
        seeds: Comma-separated LigandMPNN seeds.
        batch_size: LigandMPNN `--batch_size`.
        number_of_batches: LigandMPNN `--number_of_batches`.
        sc_num_samples: LigandMPNN side-chain packing samples.
        number_of_packs_per_design: LigandMPNN side-chain packs per design.
        noise_scale_ca: RFdiffusion denoiser CA noise scale.
        noise_scale_frame: RFdiffusion denoiser frame noise scale.
        rfd_args: Extra RFdiffusion Hydra overrides.
        wait: Wait locally for the remote workflow result.
        max_parallel: Maximum ready workflow nodes per scheduler wave.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
        strict_artifact_checks: Validate referenced RFdiffusion volume artifacts
            before reusing completed workflow nodes.
        use_deployed_coordinator: Submit through an exact named deployment.
        deployment_environment: Modal Environment containing the deployment.
        deployment_name: Modal app deployment name. Defaults to this workflow.
        deployment_version: Exact numeric Modal deployment version.
    """
    input_path = Path(input_pdb).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDB not found: {input_pdb}")
    resolved_run_id = sanitize_filename(run_id or input_path.stem)
    workflow = build_rfd_ligandmpnn_workflow(
        input_pdb=(input_path.name, input_path.read_bytes()),
        run_namespace=resolved_run_id,
        contigs=contigs,
        hotspot_res=hotspot_res,
        num_rfdiffusion_trajectories=num_rfdiffusion_trajectories,
        num_rfdiffusion_designs=num_rfdiffusion_designs,
        model_type=model_type,
        seeds=_parse_seeds(seeds),
        batch_size=batch_size,
        number_of_batches=number_of_batches,
        sc_num_samples=sc_num_samples,
        number_of_packs_per_design=number_of_packs_per_design,
        noise_scale_ca=noise_scale_ca,
        noise_scale_frame=noise_scale_frame,
        rfd_args=rfd_args,
        max_parallel=max_parallel,
    )
    if dry_run:
        print_workflow_dag(workflow.validate())
        return
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        environment=(
            deployment_environment if use_deployed_coordinator else "development"
        ),
        deployment_name=(
            (deployment_name or CONF.name) if use_deployed_coordinator else CONF.name
        ),
        deployment_version=deployment_version if use_deployed_coordinator else 1,
    )
    coordinator = orchestrator.execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    orchestrator_kwargs = {
        "workflow": workflow,
        "workload_run_key": resolved_run_id,
        "max_active_provider_calls": max_parallel,
        "max_active_gpu_provider_calls": max_parallel,
    }
    if not use_deployed_coordinator:
        orchestrator_kwargs["development_function_handles"] = {
            "rfdiffusion_infer": rfdiffusion_app.rfdiffusion_infer,
            "select_rfdiffusion_design": select_rfdiffusion_design,
            "ligandmpnn_run": ligandmpnn_app.ligandmpnn_run,
            "check_rfd_ligandmpnn_external_artifact": (
                check_rfd_ligandmpnn_external_artifact
            ),
        }
    if strict_artifact_checks:
        orchestrator_kwargs["strict_external_artifact_checks"] = True
        orchestrator_kwargs["external_artifact_checker_function_name"] = (
            "check_rfd_ligandmpnn_external_artifact"
        )
    total_structures = num_rfdiffusion_trajectories * num_rfdiffusion_designs
    print(
        f"Submitting {CONF.name} '{resolved_run_id}' with "
        f"{num_rfdiffusion_trajectories} RFdiffusion trajector"
        f"{'y' if num_rfdiffusion_trajectories == 1 else 'ies'}, "
        f"{num_rfdiffusion_designs} design(s) per trajectory, "
        f"{total_structures} LigandMPNN node(s)",
        flush=True,
    )
    fc = coordinator.run.spawn(**orchestrator_kwargs)
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}",
        flush=True,
    )
    print(f"Execution Run ID: {execution_run_id}", flush=True)
    print(
        f"Coordinator FunctionCall ID: {getattr(fc, 'object_id', fc)}",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(fc.get())
        print(f"{CONF.name} run finished with status: {result.status}", flush=True)
    else:
        result = str(getattr(fc, "object_id", fc))
        print(f"{CONF.name} run submitted. FunctionCall id: {result}", flush=True)
