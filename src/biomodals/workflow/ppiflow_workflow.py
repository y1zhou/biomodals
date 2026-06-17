"""PPIFlow workflow definition built on the reusable workflow runtime."""

from __future__ import annotations

import os
import shlex
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal
import yaml

from biomodals.app.bioinfo import rosetta_app
from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold import alphafold3_app, flowpacker_app
from biomodals.app.score import af3score_app, dockq_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import (
    AppRunLayout,
    volume_app_output,
    volume_path_from_mount_path,
)
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
    WorkflowArtifact,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    RemoteNodeSubmission,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
    print_workflow_dag,
)
from biomodals.workflow.core.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)

PPI_FLOW_OUTPUT_LAYOUT = (
    "stage1/",
    "stage2/",
    "design_output/",
    "design_output/ranked_designs.csv",
    "design_output/design_report.md",
)
PPI_FLOW_APP_STEPS = ("PPIFlowStep", "PartialStep")
PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS = ("outputs/**/*.pdb", "outputs/**/*.cif")
APP_RUN_OUTPUT_STRUCTURE_PATTERNS = ("outputs/**/*.pdb", "outputs/**/*.cif")
STRUCTURE_SUFFIXES = {".pdb", ".cif"}

DEPENDENCY_APPS = (
    "ppiflow",
    "rosetta",
    "flowpacker",
    "ligandmpnn",
    "dockq",
    "af3score",
    "alphafold3",
)
CONF = AppConfig(
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="PPIFlowWorkflow",
    package_name="biomodals-ppiflow-workflow",
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
PPI_FLOW_OUTPUT_VOLUME = ppiflow_app.CONF.output_volume
PPI_FLOW_OUTPUT_VOLUME_NAME = ppiflow_app.CONF.output_volume_name
PPI_FLOW_OUTPUT_MOUNTPOINT = ppiflow_app.CONF.output_volume_mountpoint
FLOWPACKER_OUTPUT_VOLUME = flowpacker_app.CONF.output_volume
FLOWPACKER_OUTPUT_VOLUME_NAME = flowpacker_app.CONF.output_volume_name
FLOWPACKER_OUTPUT_MOUNTPOINT = flowpacker_app.CONF.output_volume_mountpoint
AF3SCORE_OUTPUT_VOLUME = af3score_app.CONF.output_volume
AF3SCORE_OUTPUT_VOLUME_NAME = af3score_app.CONF.output_volume_name
AF3SCORE_OUTPUT_MOUNTPOINT = af3score_app.CONF.output_volume_mountpoint
ROSETTA_OUTPUT_VOLUME = rosetta_app.CONF.output_volume
ROSETTA_OUTPUT_VOLUME_NAME = rosetta_app.CONF.output_volume_name
ROSETTA_OUTPUT_MOUNTPOINT = rosetta_app.CONF.output_volume_mountpoint
WORKFLOW_OUTPUT_VOLUME = orchestrator.OUT_VOLUME
WORKFLOW_OUTPUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
WORKFLOW_OUTPUT_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
PPI_FLOW_SOURCE_VOLUME_ROOTS = {
    PPI_FLOW_OUTPUT_VOLUME_NAME: PPI_FLOW_OUTPUT_MOUNTPOINT,
    FLOWPACKER_OUTPUT_VOLUME_NAME: FLOWPACKER_OUTPUT_MOUNTPOINT,
    AF3SCORE_OUTPUT_VOLUME_NAME: AF3SCORE_OUTPUT_MOUNTPOINT,
    ROSETTA_OUTPUT_VOLUME_NAME: ROSETTA_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME_NAME: WORKFLOW_OUTPUT_MOUNTPOINT,
}
PPI_FLOW_SOURCE_VOLUME_MOUNTS = {
    PPI_FLOW_OUTPUT_MOUNTPOINT: PPI_FLOW_OUTPUT_VOLUME,
    FLOWPACKER_OUTPUT_MOUNTPOINT: FLOWPACKER_OUTPUT_VOLUME,
    AF3SCORE_OUTPUT_MOUNTPOINT: AF3SCORE_OUTPUT_VOLUME,
    ROSETTA_OUTPUT_MOUNTPOINT: ROSETTA_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME,
}


def _reload_ppiflow_source_volumes() -> None:
    PPI_FLOW_OUTPUT_VOLUME.reload()
    FLOWPACKER_OUTPUT_VOLUME.reload()
    AF3SCORE_OUTPUT_VOLUME.reload()
    ROSETTA_OUTPUT_VOLUME.reload()
    WORKFLOW_OUTPUT_VOLUME.reload()


def _artifact_mount_path(artifact: WorkflowArtifact) -> Path:
    mountpoint = PPI_FLOW_SOURCE_VOLUME_ROOTS.get(artifact.storage.volume_name)
    if mountpoint is None:
        raise ValueError(
            "PPIFlow workflow cannot read artifact volume "
            f"{artifact.storage.volume_name!r}"
        )
    return artifact.storage.at_mountpoint(mountpoint)


def _matches_structure_pattern(path: str, patterns: Sequence[str] | None) -> bool:
    import fnmatch

    suffix = Path(path).suffix.lower()
    if suffix not in STRUCTURE_SUFFIXES:
        return False
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _structure_patterns_from_metadata(
    artifact: WorkflowArtifact,
    patterns: Sequence[str] | None,
) -> Sequence[str] | None:
    if patterns is not None:
        return patterns
    metadata_patterns = artifact.metadata.get("structure_patterns")
    if isinstance(metadata_patterns, str):
        return tuple(
            pattern.strip()
            for pattern in metadata_patterns.split(",")
            if pattern.strip()
        )
    if isinstance(metadata_patterns, Sequence):
        return tuple(str(pattern) for pattern in metadata_patterns)
    return None


def _safe_selected_file_name(artifact_id: str, member_name: str) -> str:
    parts = [sanitize_filename(part) for part in Path(member_name).parts if part]
    return sanitize_filename("__".join([artifact_id, *parts]))


def _artifact_is_zstd_archive(artifact: WorkflowArtifact, path: Path) -> bool:
    return (
        artifact.kind == ArtifactKind.ARCHIVE
        or artifact.storage.media_type == ZSTD_MEDIA_TYPE
        or artifact.metadata.get("archive_format") == "tar.zst"
        or path.name.endswith(".tar.zst")
    )


def _structure_files_from_tar_zst(
    artifact: WorkflowArtifact,
    archive_path: Path,
    patterns: Sequence[str] | None,
) -> list[tuple[str, bytes]]:
    import zstandard as zstd

    selected: list[tuple[str, bytes]] = []
    with archive_path.open("rb") as compressed:
        reader = zstd.ZstdDecompressor().stream_reader(compressed)
        with reader, tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not _matches_structure_pattern(member.name, patterns):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                selected.append((
                    _safe_selected_file_name(artifact.artifact_id, member.name),
                    extracted.read(),
                ))
    return selected


def _structure_files_from_artifact(
    artifact: WorkflowArtifact,
    patterns: Sequence[str] | None,
) -> list[tuple[str, bytes]]:
    patterns = _structure_patterns_from_metadata(artifact, patterns)
    root = _artifact_mount_path(artifact)
    if not root.exists():
        raise FileNotFoundError(f"PPIFlow input artifact path not found: {root}")
    if root.is_file():
        if _artifact_is_zstd_archive(artifact, root):
            return _structure_files_from_tar_zst(artifact, root, patterns)
        if _matches_structure_pattern(root.name, patterns):
            return [
                (
                    _safe_selected_file_name(artifact.artifact_id, root.name),
                    root.read_bytes(),
                )
            ]
        return []

    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if _matches_structure_pattern(relative, patterns):
            files.append((
                _safe_selected_file_name(artifact.artifact_id, relative),
                path.read_bytes(),
            ))
    return files


def _select_structure_files_from_artifacts(
    artifacts: Sequence[WorkflowArtifact],
    *,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[tuple[str, bytes]]:
    selected = [
        structure_file
        for artifact in artifacts
        for structure_file in _structure_files_from_artifact(artifact, patterns)
    ]
    selected.sort(key=lambda item: item[0])
    if max_files is not None:
        selected = selected[:max_files]
    if not selected:
        raise FileNotFoundError("No PPIFlow structure files were found in inputs")
    return selected


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def select_ppiflow_structure_files(
    *,
    artifacts: list[WorkflowArtifact],
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[tuple[str, bytes]]:
    """Read structure files from mounted PPIFlow workflow artifacts."""
    _reload_ppiflow_source_volumes()
    return _select_structure_files_from_artifacts(
        artifacts,
        patterns=patterns,
        max_files=max_files,
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def check_ppiflow_external_artifact(
    artifact: WorkflowArtifact,
) -> ArtifactAvailability:
    """Validate app-owned artifacts referenced by the PPIFlow workflow."""
    _reload_ppiflow_source_volumes()
    return check_external_artifact_status(
        artifact,
        workflow_volume_name=orchestrator.OUT_VOLUME_NAME,
        volume_roots=PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def copy_ppiflow_structure_artifacts(
    *,
    artifacts: list[WorkflowArtifact],
    run_id: str,
    node_id: str,
    attempt_id: str,
    output_name: str,
    metadata: dict[str, object] | None = None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> AppRunResult:
    """Copy selected structure artifacts into the workflow output volume."""
    _reload_ppiflow_source_volumes()
    selected = _select_structure_files_from_artifacts(
        artifacts=artifacts,
        patterns=patterns,
        max_files=max_files,
    )
    output_dir = (
        Path(WORKFLOW_OUTPUT_MOUNTPOINT)
        / "ppiflow"
        / sanitize_filename(run_id)
        / sanitize_filename(node_id)
        / sanitize_filename(attempt_id)
        / sanitize_filename(output_name)
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, file_bytes in selected:
        (output_dir / sanitize_filename(file_name)).write_bytes(file_bytes)
    WORKFLOW_OUTPUT_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            volume_app_output(
                name=output_name,
                kind=ArtifactKind.STRUCTURES,
                remote_path=str(output_dir),
                mount_root=WORKFLOW_OUTPUT_MOUNTPOINT,
                volume_name=WORKFLOW_OUTPUT_VOLUME_NAME,
                metadata={
                    "structure_count": len(selected),
                    "files": [file_name for file_name, _ in selected],
                }
                | dict(metadata or {}),
            )
        ],
    )


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def stage_ppiflow_input_structure(
    *,
    artifacts: list[WorkflowArtifact],
    run_id: str,
    step_name: str,
    field_name: str,
    structure_index: int = 0,
    patterns: Sequence[str] | None = None,
) -> str:
    """Copy one upstream structure into the PPIFlow volume and return its path."""
    _reload_ppiflow_source_volumes()
    selected = _select_structure_files_from_artifacts(
        artifacts=artifacts,
        patterns=patterns,
        max_files=None,
    )
    if structure_index < 0 or structure_index >= len(selected):
        raise IndexError(
            f"PPIFlow input structure index {structure_index} is out of range for "
            f"{len(selected)} selected structure(s)"
        )
    file_name, file_bytes = selected[structure_index]
    remote_path = (
        Path(PPI_FLOW_OUTPUT_MOUNTPOINT)
        / sanitize_filename(run_id)
        / sanitize_filename(step_name)
        / sanitize_filename(field_name)
        / sanitize_filename(file_name)
    )
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path.write_bytes(file_bytes)
    PPI_FLOW_OUTPUT_VOLUME.commit()
    return str(remote_path)


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def stage_af3score_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    run_name: str,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Stage selected PDB files into AF3Score's input directory."""
    _reload_ppiflow_source_volumes()
    selected = _select_structure_files_from_artifacts(
        artifacts=artifacts,
        patterns=patterns,
        max_files=max_files,
    )
    layout = AppRunLayout.from_run_root(
        Path(AF3SCORE_OUTPUT_MOUNTPOINT) / sanitize_filename(run_name)
    )
    if layout.inputs_dir.exists():
        shutil.rmtree(layout.inputs_dir)
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    input_names = []
    for file_name, file_bytes in selected:
        pdb_name = sanitize_filename(Path(file_name).with_suffix(".pdb").name)
        (layout.inputs_dir / pdb_name).write_bytes(file_bytes)
        input_names.append(pdb_name)
    AF3SCORE_OUTPUT_VOLUME.commit()
    return input_names


@app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)
def stage_rosetta_inputs(
    *,
    artifacts: list[WorkflowArtifact],
    run_name: str,
    run_id: str,
    rosetta_binary: str,
    rosetta_script: str | None = None,
    flags_file: str | None = None,
    patterns: Sequence[str] | None = None,
    max_files: int | None = None,
) -> dict[str, object]:
    """Stage selected structures and enqueue Rosetta jobs."""
    _reload_ppiflow_source_volumes()
    selected = _select_structure_files_from_artifacts(
        artifacts=artifacts,
        patterns=patterns,
        max_files=max_files,
    )
    safe_run_name = sanitize_filename(run_name)
    safe_run_id = sanitize_filename(run_id)
    layout = AppRunLayout.from_run_root(
        Path(ROSETTA_OUTPUT_MOUNTPOINT) / f"{safe_run_name}-{safe_run_id}"
    )
    if layout.run_root.exists():
        shutil.rmtree(layout.run_root)
    layout.inputs_dir.mkdir(parents=True, exist_ok=True)
    remote_script = None
    if rosetta_script:
        remote_script = "inputs/_script/workflow.xml"
        script_path = layout.run_root / remote_script
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            _resolve_rosetta_config_text(rosetta_script, "rosetta_script"),
            encoding="utf-8",
        )
    remote_flags = None
    if flags_file:
        remote_flags = "inputs/_flags/workflow.flags"
        flags_path = layout.run_root / remote_flags
        flags_path.parent.mkdir(parents=True, exist_ok=True)
        flags_path.write_text(
            _resolve_rosetta_config_text(flags_file, "flags_file"),
            encoding="utf-8",
        )

    queue = modal.Queue.from_name(
        f"{rosetta_app.CONF.name}-queue-{safe_run_id}",
        create_if_missing=True,
    )
    for index, (file_name, file_bytes) in enumerate(selected, start=1):
        remote_pdb = f"inputs/{index}/{sanitize_filename(file_name)}"
        pdb_path = layout.run_root / remote_pdb
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(file_bytes)
        queue.put({
            "index": index,
            "binary": rosetta_binary,
            "pdb": remote_pdb,
            "rosetta_script": remote_script,
            "flags_file": remote_flags,
        })
    ROSETTA_OUTPUT_VOLUME.commit()
    return {
        "run_name": safe_run_name,
        "run_id": safe_run_id,
        "run_root": str(layout.run_root),
        "num_jobs": len(selected),
    }


@dataclass(frozen=True)
class PPIFlowModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    ppiflow_run: modal.Function
    ligandmpnn_run: modal.Function
    flowpacker_run: modal.Function
    af3score_manage_lock: modal.Function
    af3score_prepare: modal.Function
    af3score_run: modal.Function
    af3score_postprocess: modal.Function
    dockq_run: modal.Function
    rosetta_run: modal.Function
    alphafold3_search_msa: modal.Function
    alphafold3_predict_structures: modal.Function
    select_structures: modal.Function
    copy_structures: modal.Function
    stage_ppiflow_input: modal.Function
    stage_af3score_inputs: modal.Function
    stage_rosetta_inputs: modal.Function


@dataclass
class _ConfiguredAppStepNode(AppBackedNode):
    """Base class for configured PPIFlow app-backed workflow nodes."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def _run_name(self, context: NodeRunContext) -> str:
        run_name = sanitize_filename(
            str(self.config.get("run_name") or f"{context.run_id}-{self.step_name}")
        )
        return run_name

    def _select_structures(
        self,
        context: NodeRunContext,
        *,
        max_files: int | None = None,
        default_patterns: Sequence[str] | None = None,
    ) -> list[tuple[str, bytes]]:
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(
                f"PPIFlow workflow step {self.step_name!r} requires structure inputs"
            )
        patterns = _patterns_from_config(
            self.config,
            default=default_patterns,
        )
        return self.modal_namespace.select_structures.remote(
            artifacts=artifacts,
            patterns=patterns,
            max_files=max_files,
        )

    def _select_one_structure(
        self,
        context: NodeRunContext,
        *,
        default_patterns: Sequence[str] | None = None,
    ) -> tuple[str, bytes]:
        selected = self._select_structures(
            context,
            max_files=self.config.get("max_structures"),
            default_patterns=default_patterns,
        )
        explicit_index = "structure_index" in self.config
        if len(selected) > 1 and not explicit_index:
            raise ValueError(
                f"{self.step_name} selected {len(selected)} structures; set an "
                "explicit structure_index or structure_patterns/max_structures "
                "to avoid silently discarding candidates"
            )
        structure_index = int(self.config.get("structure_index", 0))
        if structure_index < 0 or structure_index >= len(selected):
            raise IndexError(
                f"{self.step_name} structure_index {structure_index} is out of "
                f"range for {len(selected)} selected structure(s)"
            )
        return selected[structure_index]


@dataclass
class _PPIFlowRunNode(_ConfiguredAppStepNode):
    """App-backed node implemented by the PPIFlow workflow app function."""

    def _app_kwargs(self, context: NodeRunContext) -> dict[str, object]:
        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        return {"args": app_args, "run_name": self._run_name(context)}

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the PPIFlow app function directly from the orchestrator."""
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ppiflow_run.spawn(
                **self._app_kwargs(context)
            ),
            function_name="ppiflow_run",
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose PPIFlow app output directories as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {
                "step_name": self.step_name,
                "structure_patterns": PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS,
            }
            | dict(metadata),
        )


@dataclass
class PPIFlowDesignNode(_PPIFlowRunNode):
    """Initial PPIFlow design step."""


@dataclass
class PPIFlowPartialNode(_PPIFlowRunNode):
    """PPIFlow partial-design step for stage 2."""

    def _app_kwargs(self, context: NodeRunContext) -> dict[str, object]:
        raw_args = deepcopy(self.config.get("args", self.config))
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")
        if context.inputs.get("structures"):
            field_name = "complex_pdb" if "complex_pdb" in raw_args else "input_pdb"
            raw_args[field_name] = self.modal_namespace.stage_ppiflow_input.remote(
                artifacts=context.inputs["structures"],
                run_id=self._run_name(context),
                step_name=self.step_name,
                field_name=field_name,
                structure_index=int(self.config.get("structure_index", 0)),
                patterns=None,
            )
        if "fixed_positions" not in raw_args:
            for artifact in context.inputs.get("structures", []):
                fixed_positions = artifact.metadata.get("fixed_positions")
                if fixed_positions:
                    raw_args["fixed_positions"] = str(fixed_positions)
                    break

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        return {"args": app_args, "run_name": self._run_name(context)}


@dataclass
class LigandMPNNNode(_ConfiguredAppStepNode):
    """LigandMPNN or AbMPNN design step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the LigandMPNN app function."""
        selected_name, selected_bytes = self._select_one_structure(context)
        script_mode = str(self.config.get("script_mode", "run"))
        model_type = str(
            self.config.get(
                "model_type",
                "abmpnn" if self.step_name.startswith("AbMPNN") else "protein_mpnn",
            )
        )
        cli_kwargs = _ligandmpnn_cli_kwargs(
            self.config,
            script_mode=script_mode,
            model_type=model_type,
        )
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.ligandmpnn_run.spawn(
                run_name=self._run_name(context),
                script_mode=script_mode,
                struct_bytes=selected_bytes,
                seeds=_parse_seed_values(self.config.get("seeds", [0])),
                cli_args=ligandmpnn_app.build_ligandmpnn_cli_args(**cli_kwargs),
                bias_aa_per_residue_bytes=self.config.get("bias_aa_per_residue_bytes"),
                omit_aa_per_residue_bytes=self.config.get("omit_aa_per_residue_bytes"),
            ),
            function_name="ligandmpnn_run",
            metadata={"selected_structure": selected_name},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose LigandMPNN archives as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class FlowPackerNode(_ConfiguredAppStepNode):
    """FlowPacker side-chain packing step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the FlowPacker app function."""
        selected_structures = self._select_structures(
            context,
            max_files=self.config.get("max_structures"),
        )
        kwargs = {
            key: self.config[key]
            for key in (
                "model_name",
                "use_confidence",
                "n_samples",
                "num_steps",
                "sample_coeff",
                "use_gt_masks",
                "inpaint",
                "save_traj",
                "seed",
            )
            if key in self.config
        }
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.flowpacker_run.spawn(
                input_files=selected_structures,
                run_name=self._run_name(context),
                **kwargs,
            ),
            function_name="run_flowpacker_workflow",
            metadata={"structure_count": len(selected_structures)},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose FlowPacker archives as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class AF3ScoreNode(_ConfiguredAppStepNode):
    """AF3Score structure scoring step."""

    placement: NodePlacement = NodePlacement.ORCHESTRATOR

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run the AF3Score app's prepare/run/postprocess sequence."""
        run_name = self._run_name(context)
        input_names = self.modal_namespace.stage_af3score_inputs.remote(
            artifacts=context.inputs.get("structures") or [],
            run_name=run_name,
            patterns=_patterns_from_config(self.config, default=("*.pdb",)),
            max_files=self.config.get("max_structures"),
        )
        if not input_names:
            raise ValueError(f"{self.step_name} requires at least one AF3Score input")

        self.modal_namespace.af3score_manage_lock.remote(
            run_name=run_name,
            acquire=True,
        )
        try:
            task_spec = self.modal_namespace.af3score_prepare.remote(
                run_name=run_name,
                input_files=input_names,
                num_jobs=int(
                    self.config.get("num_jobs", self.config.get("max_batches", 10))
                ),
                prepare_workers=int(self.config.get("prepare_workers", 8)),
            )
            calls = []
            chunk_specs = (
                task_spec.get("chunk_specs", [])
                if isinstance(task_spec, Mapping)
                else getattr(task_spec, "chunk_specs", [])
            )
            for chunk in chunk_specs:
                batch_name = (
                    chunk["batch_name"]
                    if isinstance(chunk, Mapping)
                    else chunk.batch_name
                )
                batch_json_dir = (
                    chunk["batch_json_dir"]
                    if isinstance(chunk, Mapping)
                    else chunk.batch_json_dir
                )
                batch_pdb_dir = (
                    chunk["batch_pdb_dir"]
                    if isinstance(chunk, Mapping)
                    else chunk.batch_pdb_dir
                )
                calls.append(
                    self.modal_namespace.af3score_run.spawn(
                        run_name=run_name,
                        batch_name=batch_name,
                        batch_json_dir=batch_json_dir,
                        batch_pdb_dir=batch_pdb_dir,
                    )
                )
            for call in calls:
                call.get()
            metrics = self.modal_namespace.af3score_postprocess.remote(
                run_name=run_name,
                input_files=input_names,
            )
        finally:
            self.modal_namespace.af3score_manage_lock.remote(
                run_name=run_name,
                acquire=False,
            )

        metrics_csv = str(metrics["metrics_csv"])
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="af3score_metrics",
                    kind=ArtifactKind.SCORES,
                    storage=volume_path_from_mount_path(
                        metrics_csv,
                        AF3SCORE_OUTPUT_MOUNTPOINT,
                        AF3SCORE_OUTPUT_VOLUME_NAME,
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "run_name": run_name,
                    }
                    | dict(metrics),
                )
            ],
        )


@dataclass
class _RosettaNode(_ConfiguredAppStepNode):
    """Base class for PPIFlow Rosetta steps."""

    placement: NodePlacement = NodePlacement.ORCHESTRATOR

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Stage inputs, run Rosetta workers, and return the output directory."""
        run_name = self._run_name(context)
        run_id = sanitize_filename(
            f"{context.run_id}-{context.node_id}-{context.attempt_id}"
        )
        staged = self.modal_namespace.stage_rosetta_inputs.remote(
            artifacts=context.inputs.get("structures") or [],
            run_name=run_name,
            run_id=run_id,
            rosetta_binary=str(self.config.get("rosetta_binary", "relax")),
            rosetta_script=self.config.get("rosetta_script"),
            flags_file=self.config.get("flags_file"),
            patterns=None,
            max_files=self.config.get("max_structures"),
        )
        num_jobs = int(staged["num_jobs"])
        if num_jobs < 1:
            raise ValueError(f"{self.step_name} requires at least one Rosetta input")
        num_cpu_per_pod = min(30, max(1, num_jobs))
        max_num_pods = max(1, int(self.config.get("max_num_pods", 1)))
        num_pods = min(
            max_num_pods, (num_jobs + num_cpu_per_pod - 1) // num_cpu_per_pod
        )
        calls = [
            self.modal_namespace.rosetta_run.spawn(
                str(staged["run_name"]),
                str(staged["run_id"]),
                num_cpu_per_pod,
            )
            for _ in range(num_pods)
        ]
        for call in calls:
            call.get()

        # TODO: add a small workflow helper for best-effort Rosetta queue cleanup
        # after workers finish. Doing it directly here makes unit tests hit the
        # Modal control plane instead of the hydrated namespace.

        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                volume_app_output(
                    name="rosetta_outputs",
                    kind=ArtifactKind.STRUCTURES,
                    remote_path=str(staged["run_root"]),
                    mount_root=ROSETTA_OUTPUT_MOUNTPOINT,
                    volume_name=ROSETTA_OUTPUT_VOLUME_NAME,
                    metadata={
                        "step_name": self.step_name,
                        "run_name": str(staged["run_name"]),
                        "run_id": str(staged["run_id"]),
                        "num_jobs": num_jobs,
                        "structure_patterns": APP_RUN_OUTPUT_STRUCTURE_PATTERNS,
                    },
                )
            ],
        )


@dataclass
class RosettaFixNode(_RosettaNode):
    """Rosetta fixed-position analysis step."""


@dataclass
class RosettaRelaxNode(_RosettaNode):
    """Rosetta relaxation step."""


@dataclass
class ReFoldNode(_ConfiguredAppStepNode):
    """AlphaFold3 refolding step."""

    placement: NodePlacement = NodePlacement.ORCHESTRATOR

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run AlphaFold3 refolding for one selected structure."""
        structure_name, structure_bytes = self._select_one_structure(
            context,
            default_patterns=("*.pdb",),
        )
        run_name = self._run_name(context)
        conf = _af3_config_for_refold(
            structure_name=structure_name,
            structure_bytes=structure_bytes,
            run_name=run_name,
            config=self.config,
        )
        json_bytes = conf.model_dump_json().encode("utf-8")
        if bool(self.config.get("search_msa", False)):
            json_bytes = self.modal_namespace.alphafold3_search_msa.remote(
                json_bytes=json_bytes,
                copy_msa_to_ssd=True,
            )
        function_call = self.modal_namespace.alphafold3_predict_structures.spawn(
            json_bytes=json_bytes,
            recycle=int(self.config.get("recycle", 10)),
            sample=int(self.config.get("sample", 5)),
            model_seeds=list(conf.modelSeeds),
        )
        tarball_bytes = function_call.get()
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="alphafold3_refolded_structures",
                    kind=ArtifactKind.STRUCTURES,
                    storage=InlineBytes(
                        data=tarball_bytes,
                        filename=f"{run_name}_alphafold3.tar.zst",
                        media_type=ZSTD_MEDIA_TYPE,
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "run_name": run_name,
                        "source_structure": structure_name,
                        "archive_format": "tar.zst",
                    },
                )
            ],
        )


@dataclass
class DockQNode(_ConfiguredAppStepNode):
    """DockQ model/reference scoring step."""

    def submit_remote(self, context: NodeRunContext) -> RemoteNodeSubmission:
        """Submit the DockQ app function."""
        references = self._select_structures(context)
        model_artifacts = context.inputs.get("models") or []
        if not model_artifacts:
            raise ValueError(f"{self.step_name} requires model structure inputs")
        models = self.modal_namespace.select_structures.remote(
            artifacts=model_artifacts,
            patterns=None,
            max_files=self.config.get("max_models"),
        )
        if len(references) != len(models):
            raise ValueError(
                f"{self.step_name} requires the same number of reference and model "
                f"structures; found {len(references)} reference(s) and "
                f"{len(models)} model(s)"
            )
        pairs = []
        for pair_idx, (
            (reference_name, reference_bytes),
            (model_name, model_bytes),
        ) in enumerate(
            zip(references, models, strict=True),
            start=1,
        ):
            pairs.append({
                "id": f"dockq_pair_{pair_idx}",
                "model_name": model_name,
                "model_bytes": model_bytes,
                "reference_name": reference_name,
                "reference_bytes": reference_bytes,
                "mapping": self.config.get("mapping"),
            })
        if not pairs:
            raise ValueError(f"{self.step_name} did not find any DockQ pairs")
        dockq_args = self.config.get("dockq_args", "--short")
        if isinstance(dockq_args, str):
            dockq_args = shlex.split(dockq_args)
        return RemoteNodeSubmission(
            function_call=self.modal_namespace.dockq_run.spawn(
                pairs=pairs,
                run_name=self._run_name(context),
                dockq_args=dockq_args,
            ),
            function_name="run_dockq_workflow",
            metadata={"pair_count": len(pairs)},
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Attach DockQ workflow metadata to score outputs."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.SCORES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class ExistingStructuresNode(WorkflowNativeNode):
    """Reference existing structures for stage-2-only PPIFlow runs."""

    step_name: str
    storage: VolumePath
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Return the configured existing structure location as an artifact."""
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="stage2_input_structures",
                    kind=ArtifactKind.STRUCTURES,
                    storage=self.storage,
                    metadata={
                        "step_name": self.step_name,
                        "run_name": self.config.get("run_name", context.run_id),
                    },
                )
            ],
        )


@dataclass
class FilterStructuresNode(WorkflowNativeNode):
    """Filter structures using score artifacts."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute filtering logic."""
        _ = context
        raise NotImplementedError(
            f"{self.step_name} score-based filtering is not implemented yet; "
            "disable this step or add a workflow-native score parser before "
            "using it in production"
        )


@dataclass
class FixedPositionsNode(WorkflowNativeNode):
    """Convert Rosetta residue energies into fixed-position constraints."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute fixed-position conversion logic."""
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(f"{self.step_name} requires structure inputs")
        fixed_positions = str(self.config.get("fixed_positions") or "")
        if not fixed_positions:
            for artifact in artifacts:
                value = artifact.metadata.get("fixed_positions")
                if value:
                    fixed_positions = str(value)
                    break
        if not fixed_positions:
            raise ValueError(
                f"{self.step_name} requires fixed_positions in step config or "
                "upstream artifact metadata; Rosetta residue-energy parsing is "
                "not implemented yet"
            )
        return AppRunResult.model_validate(
            self.modal_namespace.copy_structures.remote(
                artifacts=artifacts,
                run_id=context.run_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                output_name="fixed_position_structures",
                metadata={
                    "step_name": self.step_name,
                    "fixed_positions": fixed_positions,
                },
                patterns=None,
                max_files=self.config.get("max_structures"),
            )
        )


@dataclass
class RankNode(WorkflowNativeNode):
    """Rank final designs."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute ranking logic."""
        _ = context
        raise NotImplementedError(
            f"{self.step_name} score-aware ranking is not implemented yet; "
            "disable this step or add a ranking parser for score artifacts"
        )


@dataclass
class ReportNode(WorkflowNativeNode):
    """Write the final design report."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute report generation logic."""
        if "rank" not in context.inputs:
            raise NotImplementedError(
                f"{self.step_name} report generation requires ranked designs; "
                "score-aware ranking is not implemented yet"
            )
        artifacts = [
            artifact
            for artifact_list in context.inputs.values()
            for artifact in artifact_list
        ]
        lines = [
            "# PPIFlow Workflow Report",
            "",
            f"- Step: {self.step_name}",
            f"- Input artifacts: {len(artifacts)}",
            "",
            "| Artifact | Kind | Volume | Path |",
            "| --- | --- | --- | --- |",
        ]
        for artifact in sorted(artifacts, key=lambda item: item.artifact_id):
            lines.append(
                f"| {artifact.artifact_id} | {artifact.kind} | "
                f"{artifact.storage.volume_name} | {artifact.storage.path} |"
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="design_report",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=("\n".join(lines) + "\n").encode("utf-8"),
                        filename="design_report.md",
                        media_type="text/markdown",
                    ),
                    metadata={"step_name": self.step_name},
                )
            ],
        )


def _result_with_output_kind(
    result: AppRunResult,
    kind: ArtifactKind,
    metadata: Mapping[str, object],
) -> AppRunResult:
    outputs = [
        output.model_copy(
            update={
                "kind": kind,
                "metadata": dict(output.metadata) | dict(metadata),
            }
        )
        for output in result.outputs
    ]
    return result.model_copy(update={"outputs": outputs})


def _parse_seed_values(value: object) -> list[int]:
    if isinstance(value, str):
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        seeds = [value]
    elif isinstance(value, Sequence):
        seeds = [int(seed) for seed in value]
    else:
        raise TypeError("seeds must be an integer, comma-separated string, or sequence")
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    return seeds


def _patterns_from_config(
    config: Mapping[str, object],
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...] | None:
    raw_patterns = config.get("structure_patterns") or config.get("patterns")
    if raw_patterns is None:
        return tuple(default) if default is not None else None
    if isinstance(raw_patterns, str):
        return tuple(
            pattern.strip() for pattern in raw_patterns.split(",") if pattern.strip()
        )
    return tuple(str(pattern) for pattern in raw_patterns)


def _resolve_rosetta_config_text(value: str, field_name: str) -> str:
    config_path = Path(value).expanduser()
    has_newline = "\n" in value
    looks_like_path = (
        config_path.suffix in {".xml", ".flags"} or "/" in value or "\\" in value
    )
    if looks_like_path and not has_newline and config_path.exists():
        return config_path.read_text(encoding="utf-8")
    if looks_like_path and not has_newline:
        raise FileNotFoundError(
            f"Rosetta {field_name} path was not found locally or in the mounted "
            f"container filesystem: {value}"
        )
    return value


def _inline_rosetta_config_files(steps_doc: dict[str, Any]) -> dict[str, Any]:
    staged_steps = deepcopy(steps_doc)
    for step_name in ("RosettaFixStep", "RosettaRelaxStep"):
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        for field_name in ("rosetta_script", "flags_file"):
            value = cfg.get(field_name)
            if isinstance(value, str):
                cfg[field_name] = _resolve_rosetta_config_text(value, field_name)
    return staged_steps


def _ligandmpnn_cli_kwargs(
    config: Mapping[str, object],
    *,
    script_mode: str,
    model_type: str,
) -> dict[str, object]:
    excluded = {
        "run_name",
        "seeds",
        "script_mode",
        "structure_index",
        "max_structures",
        "structure_patterns",
        "patterns",
        "bias_aa_per_residue_bytes",
        "omit_aa_per_residue_bytes",
    }
    allowed = set(ligandmpnn_app.build_ligandmpnn_cli_args.__annotations__)
    allowed.discard("return")
    kwargs = {
        key: value
        for key, value in config.items()
        if key in allowed and key not in excluded
    }
    kwargs["script_mode"] = script_mode
    kwargs["model_type"] = model_type
    return kwargs


def _af3_config_for_refold(
    *,
    structure_name: str,
    structure_bytes: bytes,
    run_name: str,
    config: Mapping[str, object],
):
    if config.get("af3_config_json") is not None:
        conf = alphafold3_app.AF3Config.model_validate_json(
            str(config["af3_config_json"])
        )
        conf.name = run_name
        return conf

    residue_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for line in structure_bytes.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain_id = line[21].strip() or "A"
        residue_id = line[22:27].strip()
        residue_key = (chain_id, residue_id)
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue_name = line[17:20].strip().upper()
        chains.setdefault(chain_id, []).append(residue_map.get(residue_name, "X"))
    if not chains:
        raise ValueError(f"Could not derive AlphaFold3 sequence from {structure_name}")

    return alphafold3_app.AF3Config(
        name=run_name,
        modelSeeds=[
            int(seed) for seed in _parse_seed_values(config.get("model_seeds", [1]))
        ],
        sequences=[
            alphafold3_app.AF3SequenceEntry(
                protein=alphafold3_app.AF3Protein(
                    id=chain_id,
                    sequence="".join(sequence),
                )
            )
            for chain_id, sequence in sorted(chains.items())
        ],
    )


def build_ppiflow_workflow(
    *,
    task_yaml_bytes: bytes,
    steps_yaml_bytes: bytes,
    stage: int | None = None,
    modal_namespace: PPIFlowModalNamespace | None = None,
) -> Workflow:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    if modal_namespace is None:
        modal_namespace = PPIFlowModalNamespace(
            ppiflow_run=ppiflow_app.ppiflow_run_workflow,
            ligandmpnn_run=ligandmpnn_app.ligandmpnn_run,
            flowpacker_run=flowpacker_app.run_flowpacker_workflow,
            af3score_manage_lock=af3score_app.af3score_manage_lock,
            af3score_prepare=af3score_app.af3score_prepare,
            af3score_run=af3score_app.af3score_run,
            af3score_postprocess=af3score_app.af3score_postprocess,
            dockq_run=dockq_app.run_dockq_workflow,
            rosetta_run=rosetta_app.run_rosetta,
            alphafold3_search_msa=alphafold3_app.run_data_pipeline,
            alphafold3_predict_structures=alphafold3_app.run_inference_pipeline,
            select_structures=select_ppiflow_structure_files,
            copy_structures=copy_ppiflow_structure_artifacts,
            stage_ppiflow_input=stage_ppiflow_input_structure,
            stage_af3score_inputs=stage_af3score_inputs,
            stage_rosetta_inputs=stage_rosetta_inputs,
        )

    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    workflow = Workflow("ppiflow-v2")

    stage1_tail = None
    if stage in {None, 1}:
        stage1_tail = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            modal_namespace=modal_namespace,
        )

    if stage in {None, 2}:
        stage2_upstream = stage1_tail
        if stage == 2:
            stage2_upstream = workflow.add_node(
                _stage2_input_node(task, steps_doc),
                id="stage2-existing-input",
            )
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage2_upstream,
            modal_namespace=modal_namespace,
        )

    return workflow


def _add_stage1_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    modal_namespace: PPIFlowModalNamespace,
):
    tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        tail = workflow.add_node(
            PPIFlowDesignNode(
                "PPIFlowStep",
                modal_namespace,
                _step_cfg(steps, "PPIFlowStep"),
            ),
            id="stage1-ppiflow-design",
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage1"):
        mpnn_step = ("stage1-ligandmpnn", "MPNNStep_stage1")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage1"
    ):
        mpnn_step = ("stage1-abmpnn", "AbMPNNStep_stage1")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(step_name, modal_namespace, _step_cfg(steps, step_name)),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage1",
                modal_namespace,
                _step_cfg(steps, "FlowpackerStep_stage1"),
            ),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = workflow.add_node(
            AF3ScoreNode(
                "AF3scoreStep_stage1",
                modal_namespace,
                _step_cfg(steps, "AF3scoreStep_stage1"),
            ),
            id="stage1-af3score",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FilterStep_stage1"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        tail = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage1",
                modal_namespace,
                _step_cfg(steps, "FilterStep_stage1"),
            ),
            id="stage1-filter",
            inputs=inputs,
        )
    return tail


def _add_stage2_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    modal_namespace: PPIFlowModalNamespace,
) -> None:
    tail = upstream
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = workflow.add_node(
            RosettaFixNode(
                "RosettaFixStep",
                modal_namespace,
                _step_cfg(steps, "RosettaFixStep"),
            ),
            id="stage2-rosetta-fix",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "RosettaFixStep") and _step_enabled(
        enabled, "PartialStep"
    ):
        tail = workflow.add_node(
            FixedPositionsNode(
                "FixedPositions",
                modal_namespace,
                _step_cfg(steps, "FixedPositions"),
            ),
            id="stage2-fixed-positions",
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            PPIFlowPartialNode(
                "PartialStep",
                modal_namespace,
                _step_cfg(steps, "PartialStep"),
            ),
            id="stage2-partial-ppiflow",
            inputs=_structure_inputs(tail),
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage2"):
        mpnn_step = ("stage2-ligandmpnn", "MPNNStep_stage2")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage2"
    ):
        mpnn_step = ("stage2-abmpnn", "AbMPNNStep_stage2")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(step_name, modal_namespace, _step_cfg(steps, step_name)),
            id=node_id,
            inputs=_structure_inputs(tail),
        )

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage2",
                modal_namespace,
                _step_cfg(steps, "FlowpackerStep_stage2"),
            ),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
        )

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = workflow.add_node(
            AF3ScoreNode(
                "AF3scoreStep_stage2",
                modal_namespace,
                _step_cfg(steps, "AF3scoreStep_stage2"),
            ),
            id="stage2-af3score",
            inputs=_structure_inputs(tail),
        )

    filtered = tail
    if _step_enabled(enabled, "FilterStep_stage2"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        filtered = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage2",
                modal_namespace,
                _step_cfg(steps, "FilterStep_stage2"),
            ),
            id="stage2-filter",
            inputs=inputs,
        )

    refold = None
    if _step_enabled(enabled, "ReFoldStep"):
        refold = workflow.add_node(
            ReFoldNode(
                "ReFoldStep",
                modal_namespace,
                _step_cfg(steps, "ReFoldStep"),
            ),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
        )

    dockq = None
    if _step_enabled(enabled, "DockQStep"):
        inputs = _structure_inputs(filtered)
        if refold is not None:
            inputs["models"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        dockq = workflow.add_node(
            DockQNode(
                "DockQStep",
                modal_namespace,
                _step_cfg(steps, "DockQStep"),
            ),
            id="stage2-dockq",
            inputs=inputs,
        )

    relaxed = None
    if _step_enabled(enabled, "RosettaRelaxStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        relaxed = workflow.add_node(
            RosettaRelaxNode(
                "RosettaRelaxStep",
                modal_namespace,
                _step_cfg(steps, "RosettaRelaxStep"),
            ),
            id="stage2-rosetta-relax",
            inputs=inputs,
        )

    rank = None
    if _step_enabled(enabled, "RankStep"):
        inputs = _structure_inputs(relaxed or filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        rank = workflow.add_node(
            RankNode("RankStep", _step_cfg(steps, "RankStep")),
            id="stage2-rank",
            inputs=inputs,
        )

    if _step_enabled(enabled, "ReportStep"):
        inputs = (
            {"rank": rank.outputs(kind=ArtifactKind.TABLE)}
            if rank is not None
            else _structure_inputs(filtered)
        )
        workflow.add_node(
            ReportNode("ReportStep", _step_cfg(steps, "ReportStep")),
            id="stage2-report",
            inputs=inputs,
        )


def _structure_inputs(upstream) -> dict[str, Any]:
    if upstream is None:
        return {}
    return {"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)}


def _stage2_input_node(
    task: Mapping[str, Any],
    steps: Mapping[str, Any],
) -> ExistingStructuresNode:
    raw_cfg = steps.get("Stage2Input") or task.get("stage2_input")
    if not isinstance(raw_cfg, Mapping):
        raise ValueError(
            "stage=2 PPIFlow runs require a Stage2Input step config or "
            "task.stage2_input mapping with an existing structure path"
        )
    raw_path = raw_cfg.get("path")
    if raw_path is None:
        raise ValueError("Stage2Input requires a 'path' value")
    volume_name = str(raw_cfg.get("volume_name", PPI_FLOW_OUTPUT_VOLUME_NAME))
    path = str(raw_path)
    if path.startswith("/"):
        for known_volume, mountpoint in PPI_FLOW_SOURCE_VOLUME_ROOTS.items():
            try:
                storage = volume_path_from_mount_path(path, mountpoint, known_volume)
            except ValueError:
                continue
            return ExistingStructuresNode("Stage2Input", storage, dict(raw_cfg))
        raise ValueError(f"Stage2Input path is not under a known mountpoint: {path}")
    return ExistingStructuresNode(
        "Stage2Input",
        VolumePath(volume_name=volume_name, path=path),
        dict(raw_cfg),
    )


def _load_yaml_bytes(data: bytes) -> dict[str, Any]:
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return loaded


def _task_section(task_doc: dict[str, Any]) -> dict[str, Any]:
    section = task_doc.get("task", task_doc)
    if not isinstance(section, dict):
        raise ValueError("task.yaml must contain a mapping under 'task'")
    return section


def _enabled_section(task_doc: dict[str, Any]) -> dict[str, bool]:
    enabled = task_doc.get("steps", {})
    if not isinstance(enabled, dict):
        raise ValueError("task.yaml 'steps' section must be a mapping")
    return {str(key): bool(value) for key, value in enabled.items()}


def _step_enabled(enabled: dict[str, bool], step_name: str) -> bool:
    return bool(enabled.get(step_name, False))


def _step_cfg(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    cfg = steps.get(step_name, {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"steps.yaml entry {step_name!r} must be a mapping")
    return cfg


def _ppiflow_input_fields(args: object) -> tuple[str, ...]:
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyConfig):
        return ("antigen_pdb", "framework_pdb")
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyPartialConfig):
        return ("complex_pdb",)
    if isinstance(
        args,
        (ppiflow_app.SampleBinderConfig, ppiflow_app.SampleBinderPartialConfig),
    ):
        return ("input_pdb",)
    raise TypeError(f"Unsupported PPIFlow args type: {type(args).__name__}")


def _active_ppiflow_app_steps(
    task_doc: dict[str, Any], stage: int | None
) -> tuple[str, ...]:
    """Return PPIFlow app steps that should be staged for the selected run."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    enabled = _enabled_section(task_doc)
    active_steps: list[str] = []
    if stage in {None, 1} and _step_enabled(enabled, "PPIFlowStep"):
        active_steps.append("PPIFlowStep")
    return tuple(active_steps)


def _stage_ppiflow_app_inputs(
    *,
    steps_doc: dict[str, Any],
    run_id: str,
    app_steps: tuple[str, ...],
) -> dict[str, Any]:
    """Upload local PPIFlow app inputs and rewrite step args to mounted paths."""
    staged_steps = deepcopy(steps_doc)
    uploads: list[tuple[Path, str]] = []
    volume_root = Path(ppiflow_app.CONF.output_volume_mountpoint)

    for step_name in app_steps:
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        raw_args = cfg.get("args", cfg)
        if not isinstance(raw_args, dict):
            continue

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        for field_name in _ppiflow_input_fields(app_args.args):
            current_value = getattr(app_args.args, field_name)
            current_path = Path(current_value)
            if current_path.is_absolute() and current_path.is_relative_to(volume_root):
                continue

            local_path = current_path.expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PPIFlow {step_name} input {field_name!r} was not found "
                    f"locally or in the mounted output volume: {current_value}"
                )

            remote_rel = (
                Path(run_id)
                / sanitize_filename(step_name)
                / sanitize_filename(field_name)
                / sanitize_filename(local_path.name)
            )
            raw_args[field_name] = str(volume_root / remote_rel)
            uploads.append((local_path, remote_rel.as_posix()))

    if uploads:
        with ppiflow_app.CONF.output_volume.batch_upload() as batch:
            for local_path, remote_rel in uploads:
                remote_storage = volume_path_from_mount_path(
                    str(volume_root / remote_rel),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                print(
                    f"Uploading PPIFlow input '{local_path}' to {remote_storage}",
                    flush=True,
                )
                batch.put_file(local_path, f"/{remote_storage.path}")
    return staged_steps


@app.local_entrypoint()
def submit_ppiflow_workflow(
    task_yaml: str,
    steps_yaml: str,
    run_id: str | None = None,
    stage: int | None = None,
    force: bool = False,
    wait: bool = True,
    max_parallel: int = 16,
    dry_run: bool = False,
    strict_artifact_checks: bool = False,
) -> None:
    """Build and submit a PPIFlow workflow from task and step YAML files.

    Args:
        task_yaml: Path to the PPIFlow task YAML declaring enabled workflow
            steps and design mode.
        steps_yaml: Path to the YAML file containing per-step app arguments.
        run_id: Stable workflow run id for durable ledger state. Defaults to
            the task YAML filename stem.
        stage: Optional stage selector. Use 1 for stage 1 only, 2 for stage 2
            only, or omit to build both stages.
        force: Replace an existing workflow run ledger before running.
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_parallel: Maximum number of ready workflow nodes to execute
            concurrently in one scheduler wave.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
        strict_artifact_checks: Validate referenced app-owned volume artifacts
            before reusing completed workflow nodes.
    """
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    steps_yaml_bytes = steps_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    if dry_run:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=task_yaml_bytes,
            steps_yaml_bytes=steps_yaml_bytes,
            stage=stage,
        )
        print_workflow_dag(workflow.validate())
        return

    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_bytes),
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
    )
    steps_doc = _inline_rosetta_config_files(steps_doc)
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=task_yaml_bytes,
        steps_yaml_bytes=yaml.safe_dump(steps_doc).encode("utf-8"),
        stage=stage,
    )

    orchestrator_handle = orchestrator.WorkflowOrchestrator()
    orchestrator_kwargs = {
        "workflow": workflow,
        "run_id": resolved_run_id,
        "force": force,
        "max_ready_workers": max_parallel,
    }
    if strict_artifact_checks:
        orchestrator_kwargs["strict_external_artifact_checks"] = True
        orchestrator_kwargs["external_artifact_checker"] = (
            check_ppiflow_external_artifact.remote
        )
    print(
        f"Submitting PPIFlow workflow '{resolved_run_id}' with "
        f"{len(workflow.validate().nodes)} node(s)",
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
        print(f"PPIFlow workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"PPIFlow workflow run submitted. FunctionCall id: {result}", flush=True)
