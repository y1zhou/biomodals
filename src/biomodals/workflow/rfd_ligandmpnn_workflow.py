"""RFdiffusion to LigandMPNN workflow.

This workflow fans out slow RFdiffusion trajectories for one input PDB, then
runs one LigandMPNN design node for each RFdiffusion output PDB.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import pickle
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal

from biomodals.app.design import ligandmpnn_app, rfdiffusion_app
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

DEPENDENCY_APPS = ("rfdiffusion", "ligandmpnn")
CONF = AppConfig(
    tags={"depends_on": ",".join(DEPENDENCY_APPS)},
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
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
RFDIFFUSION_OUTPUT_VOLUME = rfdiffusion_app.CONF.output_volume
RFDIFFUSION_OUTPUT_VOLUME_NAME = rfdiffusion_app.CONF.output_volume_name
RFDIFFUSION_OUTPUT_MOUNTPOINT = rfdiffusion_app.CONF.output_volume_mountpoint


@dataclass(frozen=True)
class LigandMPNNDesignSettings:
    """Shared LigandMPNN arguments for each RFdiffusion output structure."""

    model_type: str
    seeds: tuple[int, ...]
    batch_size: int
    number_of_batches: int
    sc_num_samples: int
    number_of_packs_per_design: int


@dataclass(frozen=True)
class RFDLigandMPNNModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    rfdiffusion_infer: modal.Function
    ligandmpnn_run: modal.Function
    select_rfd_design: modal.Function


def _pdb_residue_labels(pdb_bytes: bytes) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for line in pdb_bytes.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21:22].strip()
        residue_number = line[22:26].strip()
        insertion_code = line[26:27].strip()
        if not residue_number:
            continue
        label = (
            f"{chain_id}{residue_number}{insertion_code}"
            if chain_id
            else f"{residue_number}{insertion_code}"
        )
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _as_python_value(value: object) -> object:
    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[no-any-return]
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[no-any-return]
    return value


def _residue_label(value: object) -> str | None:
    value = _as_python_value(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        label = value.strip()
        return label or None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        parts = [_as_python_value(part) for part in value]
        if len(parts) < 2 or tuple(parts[:2]) == ("_", "_"):
            return None
        chain_id = "" if parts[0] is None else str(parts[0]).strip()
        residue_number = "" if parts[1] is None else str(parts[1]).strip()
        if not residue_number:
            return None
        insertion_code = "" if len(parts) < 3 else str(parts[2]).strip()
        return (
            f"{chain_id}{residue_number}{insertion_code}"
            if chain_id
            else f"{residue_number}{insertion_code}"
        )
    return None


def _labels_from_mapping(values: object) -> set[str]:
    values = _as_python_value(values)
    if not isinstance(values, Iterable) or isinstance(values, str | bytes):
        return set()
    return {label for value in values if (label := _residue_label(value)) is not None}


def _fixed_output_labels(trb_metadata: dict[str, Any]) -> set[str]:
    if "complex_con_hal_pdb_idx" in trb_metadata:
        labels = _labels_from_mapping(trb_metadata["complex_con_hal_pdb_idx"])
        if labels:
            return labels
    labels: set[str] = set()
    for key in ("con_hal_pdb_idx", "receptor_con_hal_pdb_idx"):
        labels.update(_labels_from_mapping(trb_metadata.get(key, ())))
    return labels


def _design_pdbs(scaffolds_dir: Path, rfd_run_name: str) -> list[Path]:
    expected = scaffolds_dir / f"{rfd_run_name}_0.pdb"
    if expected.exists():
        return sorted(
            scaffolds_dir.glob(f"{rfd_run_name}_*.pdb"),
            key=lambda path: _design_index_from_name(path, rfd_run_name),
        )
    pdbs = sorted(scaffolds_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No RFdiffusion PDB outputs found in {scaffolds_dir}")
    return pdbs


def _design_index_from_name(path: Path, rfd_run_name: str) -> tuple[int, str]:
    prefix = f"{rfd_run_name}_"
    suffix = path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem
    try:
        return int(suffix), path.name
    except ValueError:
        return 1_000_000, path.name


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes={RFDIFFUSION_OUTPUT_MOUNTPOINT: RFDIFFUSION_OUTPUT_VOLUME},
)
def select_rfdiffusion_design(
    *,
    rfd_output_storage_path: str,
    rfd_run_name: str,
    design_index: int,
) -> dict[str, bytes | str]:
    """Read one RFdiffusion PDB/TRB pair and infer LigandMPNN redesign residues."""
    storage_path = VolumePath(
        volume_name=RFDIFFUSION_OUTPUT_VOLUME_NAME,
        path=rfd_output_storage_path,
    ).path
    safe_run_name = sanitize_filename(rfd_run_name)
    RFDIFFUSION_OUTPUT_VOLUME.reload()
    scaffolds_dir = Path(RFDIFFUSION_OUTPUT_MOUNTPOINT) / storage_path
    pdbs = _design_pdbs(scaffolds_dir, safe_run_name)
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
    trb_metadata = pickle.loads(trb_path.read_bytes())  # noqa: S301
    if not isinstance(trb_metadata, dict):
        raise TypeError(f"RFdiffusion TRB metadata must be a dict: {trb_path}")
    fixed_labels = _fixed_output_labels(trb_metadata)
    redesigned_labels = [
        label for label in _pdb_residue_labels(pdb_bytes) if label not in fixed_labels
    ]
    if not redesigned_labels:
        raise ValueError(f"No redesigned residues inferred for {pdb_path}")
    return {
        "pdb_name": pdb_path.name,
        "pdb_bytes": pdb_bytes,
        "trb_name": trb_path.name,
        "redesigned_residues": " ".join(redesigned_labels),
    }


@dataclass
class RFdiffusionTrajectoryNode(AppBackedNode):
    """Workflow node that runs one RFdiffusion trajectory."""

    pdb_content: bytes
    input_pdb_name: str
    run_name: str
    contigs: str
    hotspot_res: str
    num_designs: int
    modal_namespace: RFDLigandMPNNModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    noise_scale_ca: float = 1.0
    noise_scale_frame: float = 1.0
    rfd_args: str = ""
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RESUME
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run RFdiffusion and return the app-compatible output directory result."""
        safe_run_name = sanitize_filename(self.run_name)
        overrides = [
            f"inference.num_designs={int(self.num_designs)}",
            f"denoiser.noise_scale_ca={self.noise_scale_ca}",
            f"denoiser.noise_scale_frame={self.noise_scale_frame}",
            f"contigmap.contigs=[{self.contigs}]",
            f"ppi.hotspot_res=[{self.hotspot_res.replace(' ', ',')}]",
        ]
        if self.rfd_args.strip():
            overrides.extend(shlex.split(self.rfd_args))
        return AppRunResult.model_validate(
            self.modal_namespace.rfdiffusion_infer.remote(
                input_pdb_bytes=self.pdb_content,
                input_pdb_name=self.input_pdb_name,
                run_name=safe_run_name,
                hydra_overrides=shlex.join(overrides),
            )
        )


@dataclass
class LigandMPNNDesignNode(AppBackedNode):
    """Workflow node that designs sequences for one RFdiffusion output PDB."""

    rfd_run_name: str
    design_index: int
    run_name: str
    modal_namespace: RFDLigandMPNNModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    settings: LigandMPNNDesignSettings
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RESUME
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Select one RFdiffusion PDB and run LigandMPNN in run mode."""
        rfd_artifacts = context.inputs.get("rfd_output") or []
        if len(rfd_artifacts) != 1:
            raise ValueError(
                "LigandMPNN design node requires exactly one RFdiffusion output"
            )
        rfd_artifact = rfd_artifacts[0]
        if rfd_artifact.storage.volume_name != RFDIFFUSION_OUTPUT_VOLUME_NAME:
            raise ValueError(
                "RFdiffusion artifact volume does not match the RFdiffusion "
                f"output volume: {rfd_artifact.storage.volume_name}"
            )
        safe_rfd_run_name = sanitize_filename(
            str(rfd_artifact.metadata.get("run_name") or self.rfd_run_name)
        )
        selected = self.modal_namespace.select_rfd_design.remote(
            rfd_output_storage_path=rfd_artifact.storage.path,
            rfd_run_name=safe_rfd_run_name,
            design_index=self.design_index,
        )
        pdb_bytes = selected["pdb_bytes"]
        if not isinstance(pdb_bytes, bytes):
            raise TypeError("RFdiffusion selector must return PDB bytes")
        redesigned_residues = str(selected["redesigned_residues"])
        cli_args: dict[str, str | int | float | bool] = {
            "--model_type": self.settings.model_type,
            "--batch_size": str(self.settings.batch_size),
            "--number_of_batches": str(self.settings.number_of_batches),
            "--parse_atoms_with_zero_occupancy": True,
            "--pack_side_chains": True,
            "--number_of_packs_per_design": str(
                self.settings.number_of_packs_per_design
            ),
            "--repack_everything": True,
            "--sc_num_samples": str(self.settings.sc_num_samples),
            "--redesigned_residues": redesigned_residues,
        }
        result = AppRunResult.model_validate(
            self.modal_namespace.ligandmpnn_run.remote(
                run_name=sanitize_filename(self.run_name),
                script_mode="run",
                struct_bytes=pdb_bytes,
                seeds=list(self.settings.seeds),
                cli_args=cli_args,
            )
        )
        for output in result.outputs:
            output.metadata.setdefault("rfd_run_name", safe_rfd_run_name)
            output.metadata.setdefault("design_index", str(self.design_index))
            output.metadata.setdefault("redesigned_residues", redesigned_residues)
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
    modal_namespace = RFDLigandMPNNModalNamespace(
        rfdiffusion_infer=rfdiffusion_app.rfdiffusion_infer,
        ligandmpnn_run=ligandmpnn_app.ligandmpnn_run,
        select_rfd_design=select_rfdiffusion_design,
    )
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
                modal_namespace=modal_namespace,
                noise_scale_ca=noise_scale_ca,
                noise_scale_frame=noise_scale_frame,
                rfd_args=rfd_args,
            ),
            id=f"rfd-{rfd_run_name}",
        )
        for design_index in range(num_rfdiffusion_designs):
            mpnn_run_name = f"{rfd_run_name}-d{design_index:03d}-mpnn"
            mpnn = workflow.add_node(
                LigandMPNNDesignNode(
                    rfd_run_name=rfd_run_name,
                    design_index=design_index,
                    run_name=mpnn_run_name,
                    modal_namespace=modal_namespace,
                    settings=settings,
                ),
                id=f"ligandmpnn-{rfd_run_name}-d{design_index:03d}",
                inputs={"rfd_output": rfd.outputs(kind=ArtifactKind.DIRECTORY)},
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
    force: bool = False,
    wait: bool = True,
    max_parallel: int = 16,
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
        force: Replace an existing workflow run ledger before running.
        wait: Wait locally for the remote workflow result.
        max_parallel: Maximum ready workflow nodes per scheduler wave.
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
    orchestrator_handle = orchestrator.WorkflowOrchestrator()
    orchestrator_kwargs = {
        "workflow": workflow,
        "run_id": resolved_run_id,
        "force": force,
        "max_ready_workers": max_parallel,
    }
    total_structures = num_rfdiffusion_trajectories * num_rfdiffusion_designs
    print(
        f"Submitting RFdiffusion + LigandMPNN workflow '{resolved_run_id}' with "
        f"{num_rfdiffusion_trajectories} RFdiffusion trajector"
        f"{'y' if num_rfdiffusion_trajectories == 1 else 'ies'}, "
        f"{num_rfdiffusion_designs} design(s) per trajectory, "
        f"{total_structures} LigandMPNN node(s)",
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
        print(
            "RFdiffusion + LigandMPNN workflow run finished with status: "
            f"{result.status}",
            flush=True,
        )
    else:
        print(
            "RFdiffusion + LigandMPNN workflow run submitted. FunctionCall id: "
            f"{result}",
            flush=True,
        )
