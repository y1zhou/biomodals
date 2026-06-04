"""PPIFlow workflow definition built on the reusable workflow runtime."""

from __future__ import annotations

import ast
import base64
import html as html_lib
import os
import re
import shutil
import string
import subprocess as sp
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal
import orjson
import polars as pl
import yaml

from biomodals.app.bioinfo import rosetta_app
from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold import alphafold3_app, flowpacker_app
from biomodals.app.score import af3score_app, dockq_app
from biomodals.helper import patch_image_for_helper
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.helper.volume_run import (
    build_volume_run_paths,
    volume_path_from_mount_path,
)
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
from biomodals.workflow.core import (
    AppBackedNode,
    NodeRunContext,
    Workflow,
    WorkflowNativeNode,
    orchestrator,
)

PPI_FLOW_OUTPUT_LAYOUT = (
    "stage1/",
    "stage2/",
    "design_output/",
    "design_output/ranked_designs.csv",
    "design_output/design_report.html",
)
PPI_FLOW_APP_STEPS = ("PPIFlowStep", "PartialStep")
AF3_REFOLD_SEEDS = (1, 10, 3090, 999, 42, 101, 1024, 1012, 4090, 1020)
VALID_FILTER_OPERATORS = (">", ">=", "<", "<=", "==", "!=")
_PYROSETTA_INIT_DONE = False
PPI_FLOW_ROSETTA_NATIVE_XML = """<ROSETTASCRIPTS>
        <SCOREFXNS>
        </SCOREFXNS>
    <RESIDUE_SELECTORS>
        <Neighborhood name="nbrhood" resnums="29L-147L,18R-131R" distance="20.0"/>
        <Not name="others" selector="nbrhood" />
    </RESIDUE_SELECTORS>
    <TASKOPERATIONS>
        <OperateOnResidueSubset name="turn_off_others" selector="others">
            <PreventRepackingRLT/>
        </OperateOnResidueSubset>
        <RestrictToRepacking name="rtrp"/>
    </TASKOPERATIONS>
        <FILTERS>
        <ScoreCutoffFilter name="scofilter" report_residue_pair_energies="1" cutoff="5000000.0" />
        </FILTERS>
    <SIMPLE_METRICS>
    </SIMPLE_METRICS>
        <MOVERS>
        <FastRelax name="fast_relax" task_operations="rtrp,turn_off_others" relaxscript="rosettacon2018" repeats="5">
            <MoveMap name="backbone">
                    <Chain number="1" chi="1" bb="1"/>
                    <Chain number="2" chi="1" bb="0"/>
            </MoveMap>
        </FastRelax>
        </MOVERS>
        <APPLY_TO_POSE>
        </APPLY_TO_POSE>
        <PROTOCOLS>
        <Add mover_name="fast_relax"/>
        <Add filter="scofilter" />
        </PROTOCOLS>
</ROSETTASCRIPTS>
"""
PDB_RESIDUE_TO_ONE_LETTER = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

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
    tags={"depends_on": ".".join(DEPENDENCY_APPS)},
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


@app.function(
    image=runtime_image,
    cpu=(0.125, 4.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes={
        orchestrator.CONF.output_volume_mountpoint: orchestrator.OUT_VOLUME,
        ppiflow_app.CONF.output_volume_mountpoint: ppiflow_app.CONF.output_volume,
    },
)
def stage_ppiflow_app_outputs(
    *,
    run_id: str,
    node_id: str,
    outputs: list[dict[str, object]],
) -> list[dict[str, str | int]]:
    """Copy PPIFlow app-volume outputs into the workflow output volume."""
    orchestrator.OUT_VOLUME.reload()
    ppiflow_app.CONF.output_volume.reload()
    workflow_root = Path(orchestrator.CONF.output_volume_mountpoint)
    ppiflow_root = Path(ppiflow_app.CONF.output_volume_mountpoint)
    staged_outputs: list[dict[str, str | int]] = []
    for output in outputs:
        source = ppiflow_root / str(output["path"])
        if not source.exists():
            raise FileNotFoundError(f"PPIFlow app output not found: {source}")
        destination_rel = (
            Path("ppiflow_app_outputs")
            / sanitize_filename(run_id)
            / sanitize_filename(node_id)
            / sanitize_filename(str(output["name"]))
        )
        destination = workflow_root / destination_rel
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        else:
            shutil.copy2(source, destination)
        staged_outputs.append({
            "index": int(output["index"]),
            "volume_name": orchestrator.OUT_VOLUME_NAME,
            "path": destination_rel.as_posix(),
        })
    orchestrator.OUT_VOLUME.commit()
    return staged_outputs


@app.function(
    image=runtime_image,
    cpu=(0.125, 8.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes={
        orchestrator.CONF.output_volume_mountpoint: orchestrator.OUT_VOLUME,
        af3score_app.CONF.output_volume_mountpoint: af3score_app.CONF.output_volume,
    },
)
def stage_ppiflow_af3score_inputs(run_name: str, source_paths: list[str]) -> list[str]:
    """Copy workflow-volume PDBs into AF3Score's staged input directory."""
    safe_run_name = sanitize_filename(run_name)
    orchestrator.OUT_VOLUME.reload()
    af3score_app.CONF.output_volume.reload()
    workflow_root = Path(orchestrator.CONF.output_volume_mountpoint)
    run_paths = build_volume_run_paths(
        af3score_app.CONF.output_volume_mountpoint,
        safe_run_name,
        metrics_filename=af3score_app.APP_INFO.metrics_filename,
    )
    input_dir = run_paths["inputs_dir"]
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)

    input_names: list[str] = []
    seen_names: set[str] = set()
    for source_path in source_paths:
        source = workflow_root / source_path
        if not source.is_file():
            raise FileNotFoundError(f"AF3Score source PDB not found: {source}")
        input_name = _af3score_safe_input_name(source.name)
        if input_name in seen_names:
            raise ValueError(f"Duplicated AF3Score input name: {input_name}")
        seen_names.add(input_name)
        shutil.copy2(source, input_dir / input_name)
        input_names.append(input_name)

    af3score_app.CONF.output_volume.commit()
    return input_names


@app.function(
    image=runtime_image,
    cpu=(0.125, 4.125),
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes={
        orchestrator.CONF.output_volume_mountpoint: orchestrator.OUT_VOLUME,
        af3score_app.CONF.output_volume_mountpoint: af3score_app.CONF.output_volume,
    },
)
def stage_ppiflow_af3score_metrics(
    run_name: str,
    metrics_csv: str,
) -> dict[str, str]:
    """Copy AF3Score metrics from its app volume into the workflow volume."""
    safe_run_name = sanitize_filename(run_name)
    orchestrator.OUT_VOLUME.reload()
    af3score_app.CONF.output_volume.reload()

    source = Path(metrics_csv)
    af3score_root = Path(af3score_app.CONF.output_volume_mountpoint)
    if not source.is_absolute():
        source = af3score_root / metrics_csv
    if not source.is_relative_to(af3score_root):
        raise ValueError(f"AF3Score metrics path is outside its volume: {metrics_csv}")
    if not source.is_file():
        raise FileNotFoundError(f"AF3Score metrics CSV not found: {source}")

    workflow_root = Path(orchestrator.CONF.output_volume_mountpoint)
    destination_rel = Path("ppiflow_af3score") / safe_run_name / source.name
    destination = workflow_root / destination_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    orchestrator.OUT_VOLUME.commit()
    return {
        "volume_name": orchestrator.OUT_VOLUME_NAME,
        "path": destination_rel.as_posix(),
    }


@app.function(
    image=runtime_image,
    cpu=(0.125, 4.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes={
        orchestrator.CONF.output_volume_mountpoint: orchestrator.OUT_VOLUME,
        flowpacker_app.CONF.output_volume_mountpoint: flowpacker_app.CONF.output_volume,
    },
)
def stage_ppiflow_flowpacker_archive(source: dict[str, str]) -> bytes:
    """Read a FlowPacker workflow archive from its owning volume."""
    volume_name = source.get("volume_name", "")
    storage_path = source.get("path", "")
    if not volume_name or not storage_path:
        raise ValueError("FlowPacker archive source requires volume_name and path")

    if volume_name == flowpacker_app.CONF.output_volume_name:
        flowpacker_app.CONF.output_volume.reload()
        archive_path = Path(flowpacker_app.CONF.output_volume_mountpoint) / storage_path
    elif volume_name == orchestrator.OUT_VOLUME_NAME:
        orchestrator.OUT_VOLUME.reload()
        archive_path = Path(orchestrator.CONF.output_volume_mountpoint) / storage_path
    else:
        raise ValueError(f"Unsupported FlowPacker archive volume: {volume_name}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"FlowPacker archive not found: {archive_path}")
    return archive_path.read_bytes()


@app.function(
    image=rosetta_app.runtime_image,
    cpu=(1.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
)
def stage_ppiflow_rosetta_relax_scores(
    *,
    pdb_inputs: list[dict[str, bytes]],
    batch_idx: str,
    dump_pdb: bool = False,
) -> bytes:
    """Run upstream-style PyRosetta complex scoring for RosettaRelaxStep."""
    global _PYROSETTA_INIT_DONE

    import tempfile

    try:
        from pyrosetta import ScoreFunction, get_score_function, init, pose_from_pdb
        from pyrosetta.rosetta.core.scoring import (
            dslf_fa13,
            fa_atr,
            fa_dun,
            fa_elec,
            fa_intra_rep,
            fa_intra_sol,
            fa_rep,
            fa_sol,
            hbond_bb_sc,
            hbond_lr_bb,
            hbond_sc,
            hbond_sr_bb,
            lk_ball_wtd,
            omega,
            p_aa_pp,
            pro_close,
            rama_prepro,
            ref,
            rg,
            yhh_planarity,
        )
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.protocols.relax import FastRelax
    except ImportError as exc:  # pragma: no cover - depends on Rosetta image contents
        raise RuntimeError(
            "RosettaRelaxStep requires PyRosetta in the Rosetta app image to "
            "generate upstream rosetta_complex CSV output."
        ) from exc

    if not pdb_inputs:
        raise ValueError("RosettaRelaxStep received no PDB inputs")
    if not str(batch_idx).strip():
        raise ValueError("RosettaRelaxStep requires a batch_idx")

    if not _PYROSETTA_INIT_DONE:
        init(
            "-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res "
            "-ignore_zero_occupancy false -load_PDB_components false "
            "-relax:default_repeats 2 -no_fconfig"
        )
        _PYROSETTA_INIT_DONE = True
    score_terms = [
        ("fa_atr", fa_atr),
        ("fa_rep", fa_rep),
        ("fa_intra_rep", fa_intra_rep),
        ("fa_sol", fa_sol),
        ("lk_ball_wtd", lk_ball_wtd),
        ("fa_intra_sol", fa_intra_sol),
        ("fa_elec", fa_elec),
        ("hbond_lr_bb", hbond_lr_bb),
        ("hbond_sr_bb", hbond_sr_bb),
        ("hbond_bb_sc", hbond_bb_sc),
        ("hbond_sc", hbond_sc),
        ("dslf_fa13", dslf_fa13),
        ("rama_prepro", rama_prepro),
        ("p_aa_pp", p_aa_pp),
        ("fa_dun", fa_dun),
        ("omega", omega),
        ("pro_close", pro_close),
        ("yhh_planarity", yhh_planarity),
        ("ref", ref),
        ("rg", rg),
    ]
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for pdb_input in pdb_inputs:
            pdb_name = Path(str(pdb_input["name"])).name
            pdb_path = tmp_path / pdb_name
            pdb_path.write_bytes(pdb_input["contents"])
            pose = pose_from_pdb(str(pdb_path))
            original_pose = pose.clone()
            full_score = get_score_function()
            fast_relax = FastRelax()
            fast_relax.set_scorefxn(full_score)
            fast_relax.max_iter(170)
            if not os.getenv("DEBUG"):
                fast_relax.apply(pose)

            interface = InterfaceAnalyzerMover()
            interface.set_skip_reporting(True)
            interface.apply(pose)

            row: dict[str, object] = {
                "pdb_name": pdb_path.stem,
                "relaxed": full_score(pose),
                "interface_score": interface.get_interface_dG(),
                "original": full_score(original_pose),
                "delta": full_score(pose) - full_score(original_pose),
            }
            for column_name, score_term in score_terms:
                score_function = ScoreFunction()
                score_function.set_weight(score_term, 1.0)
                row[column_name] = score_function(pose)
            row["get_complexed_sasa"] = interface.get_complexed_sasa()
            row["get_interface_delta_sasa"] = interface.get_interface_delta_sasa()
            if dump_pdb:
                pose.dump_pdb(str(tmp_path / f"relax_{pdb_name}"))
            rows.append(row)

    columns = [
        "pdb_name",
        "relaxed",
        "interface_score",
        "original",
        "delta",
        *[column_name for column_name, _score_term in score_terms],
        "get_complexed_sasa",
        "get_interface_delta_sasa",
    ]
    return pl.DataFrame(rows, orient="row").select(columns).write_csv().encode("utf-8")


@app.function(
    image=runtime_image,
    cpu=(0.125, 8.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes={
        orchestrator.CONF.output_volume_mountpoint: orchestrator.OUT_VOLUME,
        rosetta_app.CONF.output_volume_mountpoint: rosetta_app.CONF.output_volume,
    },
)
def stage_ppiflow_rosetta_tasks(
    *,
    run_name: str,
    run_id: str,
    source_paths: list[str],
    step_name: str,
    config: dict[str, Any],
) -> dict[str, str | int]:
    """Copy workflow PDBs into Rosetta's volume and enqueue Rosetta jobs."""
    safe_run_name = sanitize_filename(run_name)
    safe_run_id = sanitize_filename(run_id)
    orchestrator.OUT_VOLUME.reload()
    rosetta_app.CONF.output_volume.reload()

    workflow_root = Path(orchestrator.CONF.output_volume_mountpoint)
    rosetta_mount = Path(rosetta_app.CONF.output_volume_mountpoint)
    run_root = rosetta_mount / f"{safe_run_name}-{safe_run_id}"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)

    binary = str(
        config.get("rosetta_bin")
        or config.get("rosetta_binary")
        or ("relax" if step_name == "RosettaRelaxStep" else "rosetta_scripts")
    )
    native_xml_text = (
        _rosetta_xml_template_text(config) if step_name == "RosettaFixStep" else None
    )
    remote_script = _stage_rosetta_support_file(
        run_root=run_root,
        run_root_name=run_root.name,
        config_value=(
            None
            if step_name == "RosettaFixStep"
            else (
                config.get("native_xml")
                or config.get("rosetta_script")
                or config.get("input_rosetta_script")
            )
        ),
        subdir="_script",
    )
    remote_flags = _stage_rosetta_flags_file(
        run_root=run_root,
        run_root_name=run_root.name,
        step_name=step_name,
        config=config,
    )
    queue = modal.Queue.from_name(
        f"{rosetta_app.CONF.name}-queue-{safe_run_id}",
        create_if_missing=True,
    )
    task_rows: list[dict[str, str | int | None]] = []
    for index, source_path in enumerate(source_paths, start=1):
        source = workflow_root / source_path
        if not source.is_file():
            raise FileNotFoundError(f"Rosetta source PDB not found: {source}")
        input_dir = run_root / str(index)
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = input_dir / source.name
        shutil.copy2(source, destination)
        job_script = remote_script
        if native_xml_text is not None:
            update_xml_path = input_dir / "update.xml"
            update_xml_path.write_text(
                _ppiflow_rosetta_fix_update_xml(
                    pdb_text=destination.read_text(encoding="utf-8", errors="replace"),
                    native_xml_text=native_xml_text,
                    gentype=str(config.get("gentype", "binder")),
                ),
                encoding="utf-8",
            )
            job_script = f"{run_root.name}/{index}/update.xml"
        queue.put({
            "index": index,
            "binary": binary,
            "pdb": f"{run_root.name}/{index}/{destination.name}",
            "rosetta_script": job_script,
            "flags_file": remote_flags,
        })
        task_rows.append({
            "index": index,
            "binary": binary,
            "pdb": str(source),
            "rosetta_script": job_script,
            "flags_file": remote_flags,
        })

    pl.DataFrame(
        task_rows,
        schema={
            "index": pl.Int64,
            "binary": pl.Utf8,
            "pdb": pl.Utf8,
            "rosetta_script": pl.Utf8,
            "flags_file": pl.Utf8,
        },
        orient="row",
    ).write_parquet(run_root / "tasks.parquet")

    rosetta_app.CONF.output_volume.commit()
    return {
        "run_name": safe_run_name,
        "run_id": safe_run_id,
        "num_cpu_per_pod": int(config.get("num_cpu_per_pod", 1)),
        "root": str(run_root),
    }


def _stage_rosetta_support_file(
    *,
    run_root: Path,
    run_root_name: str,
    config_value: object,
    subdir: str,
) -> str | None:
    if config_value in {None, ""}:
        return None
    local_path = Path(str(config_value)).expanduser()
    if not local_path.is_file():
        local_path = rosetta_app.ROSETTA_DIR / str(config_value)
    if not local_path.is_file():
        raise FileNotFoundError(f"Rosetta support file not found: {config_value}")
    destination = run_root / subdir / local_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, destination)
    return f"{run_root_name}/{subdir}/{destination.name}"


def _stage_rosetta_flags_file(
    *,
    run_root: Path,
    run_root_name: str,
    step_name: str,
    config: dict[str, Any],
) -> str | None:
    config_value = config.get("flags_file") or config.get("input_flags_file")
    if config_value not in {None, ""}:
        return _stage_rosetta_support_file(
            run_root=run_root,
            run_root_name=run_root_name,
            config_value=config_value,
            subdir="_flags",
        )
    if step_name != "RosettaFixStep":
        return None
    destination = run_root / "_flags" / "ppiflow-rosetta-fix.flags"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "-overwrite\n-ignore_zero_occupancy false\n",
        encoding="utf-8",
    )
    return f"{run_root_name}/_flags/{destination.name}"


def _rosetta_xml_template_text(config: dict[str, Any]) -> str:
    config_value = (
        config.get("native_xml")
        or config.get("rosetta_script")
        or config.get("input_rosetta_script")
    )
    if config_value not in {None, ""}:
        local_path = Path(str(config_value)).expanduser()
        if not local_path.is_file():
            local_path = rosetta_app.ROSETTA_DIR / str(config_value)
        if not local_path.is_file():
            raise FileNotFoundError(f"Rosetta XML template not found: {config_value}")
        return local_path.read_text(encoding="utf-8", errors="replace")
    return PPI_FLOW_ROSETTA_NATIVE_XML


def _ppiflow_rosetta_fix_update_xml(
    *,
    pdb_text: str,
    native_xml_text: str,
    gentype: str,
) -> str:
    """Generate the upstream RosettaFix ``update.xml`` for one PDB."""
    chains = _ppiflow_rosetta_pdb_chain_ranges(pdb_text)
    if not chains:
        raise ValueError("No valid ATOM records/chains parsed for RosettaFixStep")
    resnums = ",".join(
        f"{start}{chain_id}-{end}{chain_id}"
        for chain_id, (start, end) in chains.items()
    )
    xml_text = re.sub(r'resnums="[^"]+"', f'resnums="{resnums}"', native_xml_text)
    if gentype == "antibody":
        antibody_movemap = (
            '<MoveMap name="backbone">\n'
            '                <Chain number="1" chi="1" bb="1"/>\n'
            '                <Chain number="2" chi="1" bb="1"/>\n'
            '                <Chain number="3" chi="1" bb="0"/>\n'
            "            </MoveMap>"
        )
        updated = re.sub(
            r'<MoveMap name="backbone">.*?</MoveMap>',
            antibody_movemap,
            xml_text,
            flags=re.DOTALL,
        )
        if updated == xml_text:
            raise ValueError('Failed to locate <MoveMap name="backbone"> in XML')
        xml_text = updated
    return xml_text


def _ppiflow_rosetta_pdb_chain_ranges(
    pdb_text: str,
) -> dict[str, tuple[int, int]]:
    chains: dict[str, tuple[int, int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or len(line) < 26:
            continue
        chain_id = line[21].strip()
        if not chain_id:
            continue
        try:
            res_seq = int(line[22:26].strip())
        except ValueError:
            continue
        current = chains.get(chain_id)
        if current is None:
            chains[chain_id] = (res_seq, res_seq)
        else:
            chains[chain_id] = (min(current[0], res_seq), max(current[1], res_seq))
    return chains


@dataclass(frozen=True)
class PPIFlowModalNamespace:
    """Hydrated Modal objects carried across the orchestrator boundary."""

    ppiflow_run: modal.Function
    ppiflow_stage_outputs: modal.Function | None = None
    ligandmpnn_run: modal.Function | None = None
    flowpacker_run: modal.Function | None = None
    flowpacker_stage_archive: modal.Function | None = None
    af3score_stage_inputs: modal.Function | None = None
    af3score_stage_metrics: modal.Function | None = None
    af3score_manage_lock: modal.Function | None = None
    af3score_prepare: modal.Function | None = None
    af3score_run: modal.Function | None = None
    af3score_postprocess: modal.Function | None = None
    dockq_run: modal.Function | None = None
    rosetta_stage_tasks: modal.Function | None = None
    rosetta_run: modal.Function | None = None
    rosetta_package: modal.Function | None = None
    rosetta_relax_score: modal.Function | None = None
    alphafold3_data: modal.Function | None = None
    alphafold3_inference: modal.Function | None = None


@dataclass
class ExistingVolumePathNode(WorkflowNativeNode):
    """Expose an existing workflow-volume artifact to a stage-only DAG."""

    output_name: str
    kind: ArtifactKind
    storage: VolumePath

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Validate a configured existing artifact path and return it unchanged."""
        if self.storage.volume_name == orchestrator.OUT_VOLUME_NAME:
            path = _resolve_volume_storage_path(self.storage, context.cache_dir)
            if not path.exists():
                raise FileNotFoundError(
                    f"Configured stage input was not found: {self.storage}"
                )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=self.output_name,
                    kind=self.kind,
                    storage=self.storage,
                    metadata={"source": "existing_volume_path"},
                )
            ],
        )


@dataclass
class PPIFlowWorkflowNode(AppBackedNode):
    """Base class for PPIFlow v2 app-backed workflow nodes."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run a workflow-compatible PPIFlow app step."""
        if self.step_name not in PPI_FLOW_APP_STEPS:
            raise NotImplementedError(
                f"PPIFlow workflow step {self.step_name!r} does not yet have a "
                "workflow-compatible app adapter."
            )

        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")

        run_name = sanitize_filename(
            str(self.config.get("run_name") or f"{context.run_id}-{self.step_name}")
        )
        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        result = AppRunResult.model_validate(
            self.modal_namespace.ppiflow_run.remote(
                args=app_args,
                run_name=run_name,
            )
        )
        staged_result = _stage_ppiflow_result_outputs(
            result=result,
            modal_namespace=self.modal_namespace,
            context=context,
        )
        return _retag_result_outputs(staged_result, ArtifactKind.STRUCTURES)


@dataclass
class PPIFlowDesignNode(PPIFlowWorkflowNode):
    """Run upstream PPIFlowStep design generation."""


@dataclass
class PPIFlowPartialWorkflowNode(AppBackedNode):
    """Run upstream PartialStep through per-PDB PPIFlow app calls."""

    step_name: str
    gentype: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run one partial PPIFlow app call for each selected input PDB."""
        pdb_files = _input_pdb_files(context, input_name="structures")
        fixed_positions = _load_partial_fixed_positions(context)
        outputs: list[AppOutput] = []
        metrics: dict[str, str | int | float | bool] = {}
        for pdb_path in pdb_files:
            pdb_name = pdb_path.name
            if pdb_name not in fixed_positions:
                raise ValueError(f"No fixed_positions entry found for PDB: {pdb_name}")
            pdb_stem = pdb_path.stem
            run_name = sanitize_filename(f"{context.run_id}-partial-{pdb_stem}")
            app_args = ppiflow_app.PPIFlowArgs(
                args=_partial_app_config(
                    gentype=self.gentype,
                    pdb_path=pdb_path,
                    fixed_positions=fixed_positions[pdb_name],
                    config=self.config,
                )
            )
            result = AppRunResult.model_validate(
                self.modal_namespace.ppiflow_run.remote(
                    args=app_args,
                    run_name=run_name,
                )
            )
            staged_result = _stage_ppiflow_result_outputs(
                result=result,
                modal_namespace=self.modal_namespace,
                context=context,
            )
            for output in staged_result.outputs:
                outputs.append(
                    output.model_copy(
                        update={
                            "kind": ArtifactKind.STRUCTURES,
                            "metadata": output.metadata
                            | {
                                "step_name": self.step_name,
                                "run_name": run_name,
                                "input_pdb": pdb_name,
                            },
                        }
                    )
                )
            metrics[f"{pdb_stem}_status"] = result.status.value
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=outputs,
            metrics=metrics,
        )


class PPIFlowPartialNode(PPIFlowPartialWorkflowNode):
    """Run upstream PartialStep through per-PDB PPIFlow app calls."""


@dataclass
class LigandMPNNWorkflowNode(AppBackedNode):
    """Run upstream MPNNStep or AbMPNNStep through LigandMPNN app functions."""

    step_name: str
    gentype: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Call the hydrated LigandMPNN app handle for this sequence-design step."""
        if self.modal_namespace.ligandmpnn_run is None:
            return _marker_result(self.step_name, ArtifactKind.STRUCTURES)
        pdb_files = _input_pdb_files(context, input_name="structures")
        fixed_positions = _load_ligandmpnn_fixed_positions(
            context,
            use_abmpnn=self.step_name.startswith("AbMPNNStep"),
        )
        output_dir = context.cache_dir / "ligandmpnn_outputs"
        _reset_output_dir(output_dir)
        extracted_root = output_dir / "extracted"
        mpnn_pdbs_dir = output_dir / "mpnn_pdbs"
        extracted_root.mkdir(parents=True)
        mpnn_pdbs_dir.mkdir(parents=True)
        manifest: list[dict[str, str | int]] = []
        for pdb_path in pdb_files:
            cli_args = _ligandmpnn_cli_args(
                self.config,
                use_abmpnn=self.step_name.startswith("AbMPNNStep"),
                fixed_positions=_fixed_positions_for_mpnn_pdb(
                    pdb_path,
                    fixed_positions,
                ),
            )
            run_name = sanitize_filename(
                f"{context.run_id}-{self.step_name}-{pdb_path.stem}"
            )
            result = self.modal_namespace.ligandmpnn_run.remote(
                run_name=run_name,
                script_mode="run",
                struct_bytes=pdb_path.read_bytes(),
                seeds=_seed_list(self.config),
                cli_args=cli_args,
                bias_aa_per_residue_bytes=None,
                omit_aa_per_residue_bytes=None,
            )
            if isinstance(result, AppRunResult):
                archive_bytes = b""
            elif isinstance(result, bytes):
                archive_bytes = result
            else:
                raise TypeError("LigandMPNN app did not return bytes or AppRunResult")
            archive_path = output_dir / f"{pdb_path.stem}.tar.zst"
            archive_path.write_bytes(archive_bytes)
            extract_dir = extracted_root / pdb_path.stem
            extract_dir.mkdir(parents=True)
            _extract_tar_zst_archive(archive_path, extract_dir)
            manifest.append({
                "input_pdb": pdb_path.name,
                "run_name": run_name,
                "archive": archive_path.name,
                "size_bytes": len(archive_bytes),
            })
        link_prefixes = _ligandmpnn_link_prefixes(
            self.step_name,
            self.config,
            context,
            pdb_files,
        )
        _link_mpnn_input_pdbs(
            pdb_files,
            mpnn_pdbs_dir,
            prefixes_by_stem=link_prefixes,
        )
        _write_mpnn_sequences_csv(
            fasta_root=extracted_root,
            output_csv=mpnn_pdbs_dir / "mpnn_seqs.csv",
            prefixes_by_stem=link_prefixes,
        )
        output_dir.joinpath("ligandmpnn_manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ligandmpnn_outputs",
                    kind=ArtifactKind.STRUCTURES,
                    storage=_context_volume_path(
                        output_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "structures": str(len(pdb_files)),
                    },
                )
            ],
        )


class LigandMPNNNode(LigandMPNNWorkflowNode):
    """Run upstream MPNNStep or AbMPNNStep through LigandMPNN app functions."""


@dataclass
class FlowPackerWorkflowNode(AppBackedNode):
    """Run upstream FlowpackerStep through the FlowPacker app."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Call FlowPacker with LigandMPNN-staged PDB bytes and wrap its tarball."""
        if self.modal_namespace.flowpacker_run is None:
            return _marker_result(self.step_name, ArtifactKind.STRUCTURES)
        input_files, mpnn_seqs_csv = _flowpacker_input_files(context)
        run_name = sanitize_filename(f"{context.run_id}-{self.step_name}")
        flowpacker_result = self.modal_namespace.flowpacker_run.remote(
            input_files=input_files,
            run_name=run_name,
            model_name=str(self.config.get("model_name", "cluster")),
            use_confidence=bool(self.config.get("use_confidence", False)),
            n_samples=int(self.config.get("n_samples", 1)),
            num_steps=int(
                self.config.get(
                    "num_steps",
                    flowpacker_app.APP_INFO.default_num_steps,
                )
            ),
            sample_coeff=float(
                self.config.get(
                    "sample_coeff",
                    flowpacker_app.APP_INFO.default_sample_coeff,
                )
            ),
            use_gt_masks=bool(self.config.get("use_gt_masks", True)),
            inpaint=self.config.get("inpaint"),
            save_traj=bool(self.config.get("save_traj", False)),
            seed=int(self.config.get("seed", 42)),
        )
        archive_bytes = _archive_bytes_from_app_result(
            flowpacker_result,
            app_name="FlowPacker",
            stage_archive=self.modal_namespace.flowpacker_stage_archive,
        )

        output_dir = context.cache_dir / "flowpacker_outputs"
        _reset_output_dir(output_dir)
        archive_path = output_dir / f"{run_name}.tar.zst"
        archive_path.write_bytes(archive_bytes)
        extracted_dir = output_dir / "extracted"
        _extract_tar_zst_archive(archive_path, extracted_dir)
        _flatten_single_child_directory(extracted_dir)
        output_dir.joinpath("flowpacker_manifest.json").write_bytes(
            orjson.dumps(
                {
                    "run_name": run_name,
                    "archive": archive_path.name,
                    "input_files": [name for name, _content in input_files],
                    "mpnn_seqs_csv": mpnn_seqs_csv.name,
                    "size_bytes": len(archive_bytes),
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            )
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="flowpacker_outputs",
                    kind=ArtifactKind.STRUCTURES,
                    storage=_context_volume_path(
                        extracted_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "run_name": run_name,
                        "archive": archive_path.name,
                        "mpnn_seqs_csv": mpnn_seqs_csv.name,
                    },
                )
            ],
        )


class FlowPackerNode(FlowPackerWorkflowNode):
    """Run upstream FlowpackerStep through the FlowPacker app."""


@dataclass
class AF3ScoreWorkflowNode(AppBackedNode):
    """Run upstream AF3scoreStep through AF3Score app functions."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Run AF3Score's prepare/run/postprocess app functions as one Workflow Node."""
        if (
            self.modal_namespace.af3score_prepare is None
            or self.modal_namespace.af3score_postprocess is None
        ):
            return _marker_result(self.step_name, ArtifactKind.SCORES)
        run_name = sanitize_filename(f"{context.run_id}-{self.step_name}")
        input_files = _af3score_input_files(
            context,
            run_name=run_name,
            modal_namespace=self.modal_namespace,
        )
        task = self.modal_namespace.af3score_prepare.remote(
            run_name=run_name,
            input_files=input_files,
            num_jobs=int(
                self.config.get("num_jobs", self.config.get("num_workers", 1))
            ),
            prepare_workers=int(self.config.get("prepare_workers", 1)),
        )
        if self.modal_namespace.af3score_run is not None:
            for chunk in getattr(task, "chunk_specs", []):
                self.modal_namespace.af3score_run.remote(
                    run_name=run_name,
                    batch_name=chunk.batch_name,
                    batch_json_dir=chunk.batch_json_dir,
                    batch_pdb_dir=chunk.batch_pdb_dir,
                )
        postprocess = self.modal_namespace.af3score_postprocess.remote(
            run_name=run_name,
            input_files=getattr(task, "input_files", input_files),
        )
        metrics_csv = (
            str(postprocess.get("metrics_csv", ""))
            if isinstance(postprocess, dict)
            else ""
        )
        metrics_exists = (
            bool(postprocess.get("metrics_csv_exists", 0))
            if isinstance(postprocess, dict)
            else False
        )
        storage: InlineBytes | VolumePath
        if metrics_csv and metrics_exists:
            storage = _af3score_metrics_storage(
                run_name=run_name,
                metrics_csv=metrics_csv,
                modal_namespace=self.modal_namespace,
            )
        else:
            storage = InlineBytes(
                data=metrics_csv.encode("utf-8"),
                filename="af3score-metrics-path.txt",
                media_type="text/plain",
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="af3score_metrics",
                    kind=ArtifactKind.SCORES,
                    storage=storage,
                    metadata={"step_name": self.step_name, "run_name": run_name},
                )
            ],
            metrics=postprocess if isinstance(postprocess, dict) else {},
        )


class AF3ScoreNode(AF3ScoreWorkflowNode):
    """Run upstream AF3scoreStep through AF3Score app functions."""


@dataclass
class FilterStructuresNode(WorkflowNativeNode):
    """Filter structures using score artifacts."""

    step_name: str
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute filtering logic."""
        filters = _parse_filters_cfg(self.config.get("filters"))
        filename_col = str(self.config.get("filename_col", "description"))
        pdb_files = _input_pdb_files(context, input_name="structures")
        metrics_csv = _input_score_csv(context)

        filtered_rows = _filter_metrics_csv(
            metrics_csv=metrics_csv,
            filters=filters,
            filename_col=filename_col,
        )
        output_name = _filter_output_name(self.step_name)
        output_dir = context.cache_dir / output_name
        _reset_output_dir(output_dir)
        linked_count = _link_filtered_pdbs(
            filtered_rows=filtered_rows,
            pdb_files=pdb_files,
            output_dir=output_dir,
            filename_col=filename_col,
        )
        filtered_rows.write_csv(output_dir / f"{output_name}.csv")
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="filtered_structures",
                    kind=ArtifactKind.STRUCTURES,
                    storage=_context_volume_path(
                        output_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "metrics_csv": metrics_csv.name,
                        "rows": str(filtered_rows.height),
                        "linked_files": str(linked_count),
                    },
                )
            ],
        )


@dataclass
class RosettaWorkflowNode(AppBackedNode):
    """Run upstream RosettaFixStep or RosettaRelaxStep through Rosetta app handles."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    gentype: str = "binder"
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Stage Rosetta inputs, run the app, and extract packaged outputs."""
        if (
            self.modal_namespace.rosetta_stage_tasks is None
            or self.modal_namespace.rosetta_run is None
            or self.modal_namespace.rosetta_package is None
        ):
            return _marker_result(self.step_name, ArtifactKind.DIRECTORY)

        run_name = sanitize_filename(f"{context.run_id}-{self.step_name}")
        run_id = sanitize_filename(f"{context.node_id}-{context.attempt_id}")
        pdb_files = _input_pdb_files(context, input_name="structures")
        source_paths = [
            _context_volume_path(
                pdb_file,
                context=context,
                seed_artifacts=context.inputs.get("structures", []),
            ).path
            for pdb_file in pdb_files
        ]
        stage_config = dict(self.config)
        stage_config.setdefault("gentype", self.gentype)
        stage_info = self.modal_namespace.rosetta_stage_tasks.remote(
            run_name=run_name,
            run_id=run_id,
            source_paths=source_paths,
            step_name=self.step_name,
            config=stage_config,
        )
        if not isinstance(stage_info, dict):
            raise TypeError("Rosetta staging did not return run metadata")
        staged_run_name = str(stage_info.get("run_name") or run_name)
        staged_run_id = str(stage_info.get("run_id") or run_id)
        self.modal_namespace.rosetta_run.remote(
            run_name=staged_run_name,
            run_id=staged_run_id,
            num_cpu_per_pod=int(
                stage_info.get("num_cpu_per_pod", self.config.get("num_cpu_per_pod", 1))
            ),
        )
        archive_bytes = self.modal_namespace.rosetta_package.remote(
            root=str(
                stage_info.get("root")
                or (
                    Path(rosetta_app.CONF.output_volume_mountpoint)
                    / f"{staged_run_name}-{staged_run_id}"
                )
            )
        )
        if not isinstance(archive_bytes, bytes):
            raise TypeError("Rosetta package helper did not return tar.zst bytes")

        output_dir = context.cache_dir / _rosetta_output_name(self.step_name)
        _reset_output_dir(output_dir)
        archive_path = output_dir / f"{staged_run_name}-{staged_run_id}.tar.zst"
        archive_path.write_bytes(archive_bytes)
        extracted_dir = output_dir / "extracted"
        _extract_tar_zst_archive(archive_path, extracted_dir)
        _flatten_single_child_directory(extracted_dir)
        _promote_directory_contents(extracted_dir, output_dir)
        if self.step_name == "RosettaFixStep":
            _normalize_rosetta_fix_output(
                output_dir=output_dir,
                pdb_files=pdb_files,
                gentype=self.gentype,
                interface_dist=float(self.config.get("interface_dist", 12.0)),
            )
        elif self.step_name == "RosettaRelaxStep":
            _normalize_rosetta_relax_output(output_dir=output_dir)
            if not (output_dir / "rosetta_complex_0.csv").is_file():
                relax_score = self.modal_namespace.rosetta_relax_score
                if relax_score is None:
                    raise FileNotFoundError(
                        "RosettaRelaxStep did not produce rosetta_complex_0.csv and "
                        "no workflow relax-score wrapper is configured."
                    )
                batch_idx = str(self.config.get("batch_idx", "0"))
                score_csv_bytes = relax_score.remote(
                    pdb_inputs=[
                        {"name": pdb_file.name, "contents": pdb_file.read_bytes()}
                        for pdb_file in pdb_files
                    ],
                    batch_idx=batch_idx,
                    dump_pdb=bool(self.config.get("dump_pdb", False)),
                )
                if not isinstance(score_csv_bytes, bytes):
                    raise TypeError(
                        "RosettaRelaxStep score wrapper did not return CSV bytes"
                    )
                output_dir.joinpath(
                    f"rosetta_complex_{sanitize_filename(batch_idx)}.csv"
                ).write_bytes(score_csv_bytes)
                _normalize_rosetta_relax_output(output_dir=output_dir)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name=_rosetta_output_name(self.step_name),
                    kind=ArtifactKind.DIRECTORY,
                    storage=_context_volume_path(
                        output_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "run_name": staged_run_name,
                        "run_id": staged_run_id,
                    },
                )
            ],
        )


@dataclass
class RosettaFixNode(RosettaWorkflowNode):
    """Run upstream RosettaFixStep through Rosetta app handles."""


class RosettaRelaxNode(RosettaWorkflowNode):
    """Run upstream RosettaRelaxStep through Rosetta app handles."""


RosettaFixWorkflowNode = RosettaFixNode
RosettaRelaxWorkflowNode = RosettaRelaxNode


@dataclass
class FixedPositionsNode(WorkflowNativeNode):
    """Convert Rosetta residue energies into PartialStep fixed-position inputs."""

    step_name: str = "FixedPositionsStep"
    gentype: str = "binder"
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Write a fixed-position CSV from RosettaFix residue energies."""
        rosetta_dir = _input_directory(context, input_name="rosetta")
        output_csv = context.cache_dir / "fixed_positions.csv"
        fixed_positions = _build_fixed_positions_frame(
            rosetta_fix_output_dir=rosetta_dir,
            gentype=self.gentype,
            energy_threshold=float(self.config.get("energy_threshold", -5)),
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fixed_positions.write_csv(output_csv)
        outputs = [
            AppOutput(
                name="fixed_positions",
                kind=ArtifactKind.TABLE,
                storage=_context_volume_path(
                    output_csv,
                    context=context,
                    seed_artifacts=context.inputs.get("rosetta", []),
                ),
                metadata={
                    "step_name": self.step_name,
                    "gentype": self.gentype,
                    "rows": str(fixed_positions.height),
                },
            )
        ]
        if context.inputs.get("structures"):
            before_partial_dir = context.cache_dir / "before_partial_pdbs"
            _reset_output_dir(before_partial_dir)
            linked_count = _link_before_partial_pdbs(
                fixed_positions=fixed_positions,
                pdb_files=_input_pdb_files(context, input_name="structures"),
                output_dir=before_partial_dir,
            )
            outputs.append(
                AppOutput(
                    name="before_partial_pdbs",
                    kind=ArtifactKind.STRUCTURES,
                    storage=_context_volume_path(
                        before_partial_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "linked_files": str(linked_count),
                    },
                )
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=outputs,
        )


@dataclass
class ReFoldWorkflowNode(AppBackedNode):
    """Run upstream ReFoldStep using AlphaFold3 app functions."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Call AlphaFold3 data and inference handles for refolding."""
        if (
            self.modal_namespace.alphafold3_data is None
            or self.modal_namespace.alphafold3_inference is None
        ):
            return _marker_result(self.step_name, ArtifactKind.STRUCTURES)

        pdb_files = _input_pdb_files(context, input_name="structures")
        output_dir = context.cache_dir / "refold_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_msa = bool(self.config.get("run_MSA", True))
        run_inference = bool(self.config.get("run_AF3_inference", True))
        remove_template = bool(self.config.get("remove_template", False))
        seed_count = int(self.config.get("seed_num", 5))
        recycle = int(self.config.get("recycle", self.config.get("num_recycles", 10)))
        sample = int(
            self.config.get("sample", self.config.get("num_diffusion_samples", 5))
        )

        manifest: list[dict[str, str | int | bool]] = []
        metrics_rows: list[dict[str, object]] = []
        for pdb_path in pdb_files:
            run_name = sanitize_filename(pdb_path.stem)
            chain_ids = [
                chain_id
                for chain_id, _sequence in _pdb_sequences_by_chain(
                    pdb_path.read_text(encoding="utf-8", errors="replace")
                )
            ]
            json_bytes = _alphafold3_refold_json_bytes(
                name=run_name,
                pdb_bytes=pdb_path.read_bytes(),
                seed_count=seed_count,
            )
            input_json_path = output_dir / f"{run_name}.json"
            input_json_path.write_bytes(json_bytes)

            prepared_json_bytes = (
                self.modal_namespace.alphafold3_data.remote(json_bytes=json_bytes)
                if run_msa
                else json_bytes
            )
            if not isinstance(prepared_json_bytes, bytes):
                prepared_json_bytes = json_bytes
            if remove_template:
                prepared_json_bytes = _remove_alphafold3_templates_json(
                    prepared_json_bytes
                )
            prepared_json_path = output_dir / f"{run_name}_data.json"
            prepared_json_path.write_bytes(prepared_json_bytes)

            archive_path = output_dir / f"{run_name}.tar.zst"
            if run_inference:
                archive_bytes = self.modal_namespace.alphafold3_inference.remote(
                    json_bytes=prepared_json_bytes,
                    recycle=recycle,
                    sample=sample,
                    model_seeds=_alphafold3_model_seeds(seed_count),
                )
                if not isinstance(archive_bytes, bytes):
                    raise TypeError("AlphaFold3 inference did not return bytes")
                archive_path.write_bytes(archive_bytes)
                metrics_rows.extend(
                    _extract_refold_metrics(
                        archive_path=archive_path,
                        extract_dir=output_dir / "extracted" / run_name,
                        pdb_name=run_name,
                        chain_ids=chain_ids,
                    )
                )
            manifest.append({
                "run_name": run_name,
                "input_pdb": pdb_path.name,
                "input_json": input_json_path.name,
                "prepared_json": prepared_json_path.name,
                "archive": archive_path.name if run_inference else "",
                "run_MSA": run_msa,
                "run_AF3_inference": run_inference,
                "remove_template": remove_template,
                "seed_num": seed_count,
                "recycle": recycle,
                "sample": sample,
            })

        output_dir.joinpath("refold_manifest.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )
        _write_refold_metrics(output_dir, metrics_rows)
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="alphafold3_refold_outputs",
                    kind=ArtifactKind.STRUCTURES,
                    storage=_context_volume_path(
                        output_dir,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "structures": str(len(pdb_files)),
                    },
                )
            ],
        )


class ReFoldNode(ReFoldWorkflowNode):
    """Run upstream ReFoldStep using AlphaFold3 app functions."""


@dataclass
class DockQWorkflowNode(AppBackedNode):
    """Run upstream DockQStep through the DockQ app."""

    step_name: str
    modal_namespace: PPIFlowModalNamespace = field(
        repr=False,
        compare=False,
        metadata={"dag_hash": False},
    )
    config: dict[str, Any] = field(default_factory=dict)
    execution_policy: NodeExecutionPolicy = NodeExecutionPolicy.RERUN
    placement: NodePlacement = NodePlacement.REMOTE

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Call DockQ for prepared model/reference pairs."""
        if self.modal_namespace.dockq_run is None:
            return _marker_result(self.step_name, ArtifactKind.SCORES)
        pairs = _dockq_pairs(context)
        run_name = sanitize_filename(f"{context.run_id}-{self.step_name}")
        result = self.modal_namespace.dockq_run.remote(
            pairs=pairs,
            run_name=run_name,
            dockq_args=list(self.config.get("dockq_args", ["--short"])),
        )
        if not isinstance(result, bytes):
            raise TypeError("DockQ app did not return tar.zst bytes")

        output_dir = context.cache_dir / "dockq_outputs"
        _reset_output_dir(output_dir)
        archive_path = output_dir / f"{run_name}.tar.zst"
        archive_path.write_bytes(result)
        extracted_dir = output_dir / "extracted"
        _extract_tar_zst_archive(archive_path, extracted_dir)
        _flatten_single_child_directory(extracted_dir)
        _promote_directory_contents(extracted_dir, output_dir)
        dockq_csv = output_dir / "dockq_results.csv"
        if not dockq_csv.is_file():
            candidates = sorted(output_dir.rglob("dockq_results.csv"))
            if not candidates:
                raise FileNotFoundError(
                    "DockQ archive did not contain dockq_results.csv"
                )
            dockq_csv = candidates[0]
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="dockq_results",
                    kind=ArtifactKind.SCORES,
                    storage=_context_volume_path(
                        dockq_csv,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "run_name": run_name,
                        "pairs": str(len(pairs)),
                    },
                )
            ],
        )


class DockQNode(DockQWorkflowNode):
    """Run upstream DockQStep through the DockQ app."""


@dataclass
class RankAndReportNode(WorkflowNativeNode):
    """Rank final designs and write report artifacts."""

    step_name: str
    gentype: str = "binder"
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Generate the final PPIFlow design report HTML."""
        report_cfg = _report_config(self.config)
        ranked_csv = _input_csv_file(
            context,
            input_name="ranked",
            preferred_names=(
                str(report_cfg.get("output_csv_name", "ranked_designs.csv")),
            ),
        )
        output_dir = ranked_csv.parent
        report_path = output_dir / str(
            report_cfg.get("report_filename", "design_report.html")
        )
        report_path.write_text(
            _render_report_html(
                context=context,
                gentype=self.gentype,
                design_name=str(report_cfg.get("name") or context.run_id),
                ranked_csv=ranked_csv,
            ),
            encoding="utf-8",
        )
        storage = _context_volume_path(
            report_path,
            context=context,
            seed_artifacts=context.inputs.get("ranked", []),
        ).model_copy(update={"media_type": "text/html"})
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="design_report",
                    kind=ArtifactKind.REPORT,
                    storage=storage,
                    metadata={
                        "step_name": self.step_name,
                        "ranked_csv": ranked_csv.name,
                    },
                )
            ],
        )


class ReportNode(RankAndReportNode):
    """Write upstream ReportStep artifacts."""


@dataclass
class RankDesignsNode(WorkflowNativeNode):
    """Rank final designs using DockQ, Rosetta, AF3, and filtered structures."""

    step_name: str
    gentype: str = "binder"
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Merge DockQ, Rosetta, and ReFold metrics into ranked design outputs."""
        output_dir = context.cache_dir / "design_output"
        _reset_output_dir(output_dir)
        ranked_csv = output_dir / str(
            self.config.get("output_csv_name", "ranked_designs.csv")
        )
        rows = _rank_design_rows(
            context=context,
            gentype=self.gentype,
            dockq_threshold=float(self.config.get("dockq_threshold", 0.49)),
        )
        _write_ranked_designs_csv(ranked_csv, rows)
        _copy_ranked_design_pdbs(
            rows=rows,
            pdb_files=_input_pdb_files(context, input_name="structures"),
            output_dir=output_dir / "pdbs",
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ranked_designs",
                    kind=ArtifactKind.TABLE,
                    storage=_context_volume_path(
                        ranked_csv,
                        context=context,
                        seed_artifacts=context.inputs.get("structures", []),
                    ),
                    metadata={
                        "step_name": self.step_name,
                        "rows": str(len(rows)),
                    },
                )
            ],
        )


class RankNode(RankDesignsNode):
    """Rank final designs using DockQ, Rosetta, AF3, and filtered structures."""


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
            ppiflow_stage_outputs=stage_ppiflow_app_outputs,
            ligandmpnn_run=ligandmpnn_app.ligandmpnn_run,
            flowpacker_run=flowpacker_app.run_flowpacker_workflow,
            flowpacker_stage_archive=stage_ppiflow_flowpacker_archive,
            af3score_stage_inputs=stage_ppiflow_af3score_inputs,
            af3score_stage_metrics=stage_ppiflow_af3score_metrics,
            af3score_manage_lock=af3score_app.af3score_manage_lock,
            af3score_prepare=af3score_app.af3score_prepare,
            af3score_run=af3score_app.af3score_run,
            af3score_postprocess=af3score_app.af3score_postprocess,
            dockq_run=dockq_app.run_dockq_batch,
            rosetta_stage_tasks=stage_ppiflow_rosetta_tasks,
            rosetta_run=rosetta_app.run_rosetta,
            rosetta_package=rosetta_app.package_outputs_helper,
            rosetta_relax_score=stage_ppiflow_rosetta_relax_scores,
            alphafold3_data=alphafold3_app.run_data_pipeline,
            alphafold3_inference=alphafold3_app.run_inference_pipeline,
        )

    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    workflow = Workflow("ppiflow")

    stage1_tail = None
    stage1_score = None
    if stage in {None, 1}:
        stage1_tail, stage1_score = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            task=task,
            gentype=gentype,
            modal_namespace=modal_namespace,
        )
    elif stage == 2:
        stage1_tail, stage1_score = _add_stage2_input_nodes(
            workflow=workflow,
            steps=steps_doc,
        )

    if stage in {None, 2}:
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage1_tail,
            stage1_score=stage1_score,
            modal_namespace=modal_namespace,
        )

    return workflow


def _add_stage2_input_nodes(*, workflow: Workflow, steps: dict[str, Any]):
    stage2_inputs = _stage2_inputs_cfg(steps)
    filtered_config = stage2_inputs.get("filtered_structures")
    if filtered_config is None:
        raise ValueError(
            "stage=2 requires Stage2Inputs.filtered_structures to reference "
            "an existing stage 1 filtered structures artifact."
        )
    filtered = workflow.add_node(
        ExistingVolumePathNode(
            output_name="filtered_structures",
            kind=ArtifactKind.STRUCTURES,
            storage=_configured_volume_path(filtered_config),
        ),
        id="stage2-input-filtered",
    )

    stage1_score = None
    if stage1_af3score_config := stage2_inputs.get("stage1_af3score"):
        stage1_score = workflow.add_node(
            ExistingVolumePathNode(
                output_name="stage1_af3score",
                kind=ArtifactKind.SCORES,
                storage=_configured_volume_path(stage1_af3score_config),
            ),
            id="stage2-input-stage1-af3score",
        )
    return filtered, stage1_score


def _add_stage1_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    task: dict[str, Any],
    gentype: str,
    modal_namespace: PPIFlowModalNamespace,
):
    tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        tail = workflow.add_node(
            _app_step_node(steps, task, gentype, "PPIFlowStep", modal_namespace),
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
            LigandMPNNNode(
                step_name=step_name,
                gentype=gentype,
                modal_namespace=modal_namespace,
                config=_stage1_ligandmpnn_cfg(steps, step_name, task),
            ),
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
                _step_cfg(steps, "FilterStep_stage1"),
            ),
            id="stage1-filter",
            inputs=inputs,
        )
    return tail, score


def _add_stage2_nodes(
    *,
    workflow: Workflow,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    stage1_score,
    modal_namespace: PPIFlowModalNamespace,
) -> None:
    tail = upstream
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = workflow.add_node(
            RosettaFixNode(
                step_name="RosettaFixStep",
                modal_namespace=modal_namespace,
                gentype=gentype,
                config=_step_cfg(steps, "RosettaFixStep"),
            ),
            id="stage2-rosetta-fix",
            inputs=_structure_inputs(tail),
        )

    fixed_positions = None
    if _step_enabled(enabled, "PartialStep"):
        fixed_inputs = {}
        if tail is not None:
            fixed_inputs["rosetta"] = tail.outputs(kind=ArtifactKind.DIRECTORY)
        if upstream is not None:
            fixed_inputs["structures"] = upstream.outputs(kind=ArtifactKind.STRUCTURES)
        fixed_positions = workflow.add_node(
            FixedPositionsNode(
                gentype=gentype,
                config=_fixed_positions_cfg(steps),
            ),
            id="stage2-fixed-positions",
            inputs=fixed_inputs,
        )

    if _step_enabled(enabled, "PartialStep"):
        inputs = {}
        if fixed_positions is not None:
            inputs["fixed_positions"] = fixed_positions.outputs(kind=ArtifactKind.TABLE)
            inputs["structures"] = fixed_positions.outputs(kind=ArtifactKind.STRUCTURES)
        else:
            inputs = _structure_inputs(upstream)
        tail = workflow.add_node(
            PPIFlowPartialNode(
                "PartialStep",
                gentype,
                modal_namespace,
                _step_cfg(steps, "PartialStep"),
            ),
            id="stage2-partial-ppiflow",
            inputs=inputs,
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
        inputs = _structure_inputs(tail)
        if fixed_positions is not None:
            inputs["fixed_positions"] = fixed_positions.outputs(kind=ArtifactKind.TABLE)
        tail = workflow.add_node(
            LigandMPNNNode(
                step_name=step_name,
                gentype=gentype,
                modal_namespace=modal_namespace,
                config=_step_cfg(steps, step_name),
            ),
            id=node_id,
            inputs=inputs,
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
            id="stage2-refold",
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

    relax = None
    if _step_enabled(enabled, "RosettaRelaxStep"):
        relax = workflow.add_node(
            RosettaRelaxNode(
                step_name="RosettaRelaxStep",
                modal_namespace=modal_namespace,
                gentype=gentype,
                config=_step_cfg(steps, "RosettaRelaxStep"),
            ),
            id="stage2-rosetta-relax",
            inputs=_structure_inputs(filtered),
        )

    rank = None
    if _step_enabled(enabled, "RankStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        if relax is not None:
            inputs["rosetta"] = relax.outputs(kind=ArtifactKind.DIRECTORY)
        if refold is not None:
            inputs["refold"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        rank = workflow.add_node(
            RankNode(
                "RankStep",
                gentype=gentype,
                config=_step_cfg(steps, "RankStep"),
            ),
            id="stage2-rank",
            inputs=inputs,
        )

    if _step_enabled(enabled, "ReportStep"):
        inputs = {}
        if stage1_score is not None:
            inputs["stage1_af3score"] = stage1_score.outputs(kind=ArtifactKind.SCORES)
        if score is not None:
            inputs["stage2_af3score"] = score.outputs(kind=ArtifactKind.SCORES)
        if refold is not None:
            inputs["refold"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        if rank is not None:
            inputs["ranked"] = rank.outputs(kind=ArtifactKind.TABLE)
        workflow.add_node(
            ReportNode(
                "ReportStep",
                gentype=gentype,
                config=_rank_report_cfg(steps),
            ),
            id="stage2-report",
            inputs=inputs,
        )


def _structure_inputs(upstream) -> dict[str, Any]:
    if upstream is None:
        return {}
    return {"structures": upstream.outputs(kind=ArtifactKind.STRUCTURES)}


def _stage1_ligandmpnn_cfg(
    steps: dict[str, Any],
    step_name: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(_step_cfg(steps, step_name))
    task_name = task.get("name")
    if task_name is None:
        task_name = _ppiflow_step_arg_overrides(_step_cfg(steps, "PPIFlowStep")).get(
            "name"
        )
    if task_name is not None:
        cfg.setdefault("name", task_name)
    return cfg


def _app_step_node(
    steps: dict[str, Any],
    task: dict[str, Any],
    gentype: str,
    step_name: str,
    modal_namespace: PPIFlowModalNamespace,
) -> PPIFlowWorkflowNode:
    config = (
        _ppiflow_design_cfg(steps=steps, task=task, gentype=gentype)
        if step_name == "PPIFlowStep"
        else _step_cfg(steps, step_name)
    )
    node_class = (
        PPIFlowDesignNode if step_name == "PPIFlowStep" else PPIFlowWorkflowNode
    )
    return node_class(
        step_name=step_name,
        modal_namespace=modal_namespace,
        config=config,
    )


def _ppiflow_design_cfg(
    *,
    steps: dict[str, Any],
    task: dict[str, Any],
    gentype: str,
) -> dict[str, Any]:
    cfg = dict(_step_cfg(steps, "PPIFlowStep"))
    step_args = _ppiflow_step_arg_overrides(cfg)
    merged_args = dict(task)
    merged_args.update(step_args)
    if gentype == "binder":
        merged_args.setdefault("target_chain", "B")
        merged_args.setdefault("binder_chain", "A")
    elif gentype in {"antibody", "nanobody"}:
        merged_args.setdefault("heavy_chain", "A")
    if cfg.get("config") is not None and "config" not in merged_args:
        merged_args["config"] = cfg["config"]
    output_cfg = {
        key: value
        for key, value in cfg.items()
        if key not in {"args", "config", "model_weights", "output_dir", "python"}
    }
    output_cfg["args"] = merged_args
    return output_cfg


def _ppiflow_step_arg_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg.get("args"), dict):
        return dict(cfg["args"])
    return {
        key: value
        for key, value in cfg.items()
        if key not in {"run_name", "model_weights", "output_dir", "python"}
    }


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


def _rank_report_cfg(steps: dict[str, Any]) -> dict[str, Any]:
    return {
        "RankStep": _step_cfg(steps, "RankStep"),
        "ReportStep": _step_cfg(steps, "ReportStep"),
    }


def _stage2_inputs_cfg(steps: dict[str, Any]) -> dict[str, Any]:
    cfg = steps.get("Stage2Inputs", steps.get("stage2_inputs", {}))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Stage2Inputs must be a mapping")
    return cfg


def _configured_volume_path(value: object) -> VolumePath:
    if isinstance(value, str):
        return _volume_path_from_config(path=value, volume_name=None)
    if not isinstance(value, dict):
        raise ValueError("Configured volume path must be a string or mapping")
    raw_path = value.get("path")
    if raw_path in {None, ""}:
        raise ValueError("Configured volume path is missing required 'path'")
    volume_name = value.get("volume_name")
    return _volume_path_from_config(path=str(raw_path), volume_name=volume_name)


def _volume_path_from_config(path: str, volume_name: object | None) -> VolumePath:
    mountpoint = str(orchestrator.CONF.output_volume_mountpoint)
    if path.startswith(f"{mountpoint}/"):
        return volume_path_from_mount_path(
            path,
            mountpoint,
            str(volume_name or orchestrator.OUT_VOLUME_NAME),
        )
    return VolumePath(
        volume_name=str(volume_name or orchestrator.OUT_VOLUME_NAME),
        path=path.lstrip("/"),
    )


def _fixed_positions_cfg(steps: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(_step_cfg(steps, "PartialStep"))
    cfg.update(_step_cfg(steps, "RosettaFixStep"))
    return cfg


def _rosetta_output_name(step_name: str) -> str:
    if step_name == "RosettaFixStep":
        return "rosetta_fix_output"
    if step_name == "RosettaRelaxStep":
        return "rosetta_relax_output"
    return sanitize_filename(step_name)


def _flowpacker_input_files(
    context: NodeRunContext,
) -> tuple[list[tuple[str, bytes]], Path]:
    """Resolve upstream-style ``mpnn_pdbs`` inputs for FlowPacker."""
    structures_dir = _input_directory(context, input_name="structures")
    mpnn_pdbs_dir = (
        structures_dir / "mpnn_pdbs"
        if (structures_dir / "mpnn_pdbs").is_dir()
        else structures_dir
    )
    mpnn_seqs_csv = mpnn_pdbs_dir / "mpnn_seqs.csv"
    if not mpnn_seqs_csv.is_file():
        raise FileNotFoundError(
            f"Expected LigandMPNN sequence CSV for FlowPacker: {mpnn_seqs_csv}"
        )
    pdb_files = sorted(
        path for path in mpnn_pdbs_dir.iterdir() if path.suffix == ".pdb"
    )
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found for FlowPacker in {mpnn_pdbs_dir}")
    return [(path.name, path.read_bytes()) for path in pdb_files], mpnn_seqs_csv


def _af3score_input_files(
    context: NodeRunContext,
    *,
    run_name: str,
    modal_namespace: PPIFlowModalNamespace,
) -> list[str]:
    """Stage FlowPacker PDB artifacts into AF3Score's app volume."""
    if not context.inputs.get("structures"):
        return []
    pdb_files = _input_pdb_files(context, input_name="structures")
    seed_artifacts = context.inputs.get("structures", [])
    source_paths = [
        _context_volume_path(
            pdb_file,
            context=context,
            seed_artifacts=seed_artifacts,
        ).path
        for pdb_file in pdb_files
    ]
    if modal_namespace.af3score_stage_inputs is None:
        return [_af3score_safe_input_name(path.name) for path in pdb_files]
    staged_names = modal_namespace.af3score_stage_inputs.remote(
        run_name=run_name,
        source_paths=source_paths,
    )
    if not isinstance(staged_names, list) or not all(
        isinstance(name, str) for name in staged_names
    ):
        raise TypeError("AF3Score staging did not return input file names")
    return staged_names


def _af3score_metrics_storage(
    *,
    run_name: str,
    metrics_csv: str,
    modal_namespace: PPIFlowModalNamespace,
) -> VolumePath:
    if modal_namespace.af3score_stage_metrics is not None:
        staged = modal_namespace.af3score_stage_metrics.remote(
            run_name=run_name,
            metrics_csv=metrics_csv,
        )
        if not isinstance(staged, dict):
            raise TypeError("AF3Score metrics staging did not return a mapping")
        return VolumePath(
            volume_name=str(staged["volume_name"]),
            path=str(staged["path"]),
        )
    return volume_path_from_mount_path(
        metrics_csv,
        af3score_app.CONF.output_volume_mountpoint,
        af3score_app.CONF.output_volume_name,
    )


def _af3score_safe_input_name(filename: str) -> str:
    """Normalize a PDB filename the same way AF3Score local staging does."""
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    safe_name = "".join(
        char for char in filename.lower().replace(" ", "_") if char in allowed_chars
    )
    if not safe_name:
        raise ValueError(f"Input file name has no AF3Score-safe characters: {filename}")
    return safe_name


def _extract_tar_zst_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract one app tar.zst archive into a workflow-owned directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_binary = shutil.which("tar")
    if tar_binary is None:
        raise RuntimeError("tar is required to extract app archives")
    sp.run(  # noqa: S603
        [tar_binary, "-I", "zstd", "-xf", str(archive_path), "-C", str(output_dir)],
        check=True,
    )


def _archive_bytes_from_app_result(
    result: object,
    *,
    app_name: str,
    stage_archive: modal.Function | None = None,
) -> bytes:
    """Return archive bytes from raw bytes or a workflow-compatible app result."""
    if isinstance(result, bytes):
        return result
    if not isinstance(result, AppRunResult):
        raise TypeError(f"{app_name} app did not return bytes or AppRunResult")
    if result.status != AppRunStatus.SUCCEEDED:
        raise ValueError(f"{app_name} app result was not successful: {result.status}")
    archive_output = _archive_output(result)
    if isinstance(archive_output.storage, InlineBytes):
        return archive_output.storage.data
    if stage_archive is None:
        raise ValueError(f"{app_name} archive VolumePath requires a staging function")
    staged_bytes = stage_archive.remote(
        source={
            "volume_name": archive_output.storage.volume_name,
            "path": archive_output.storage.path,
        }
    )
    if not isinstance(staged_bytes, bytes):
        raise TypeError(f"{app_name} archive staging did not return bytes")
    return staged_bytes


def _archive_output(result: AppRunResult) -> AppOutput:
    for output in result.outputs:
        if output.kind == ArtifactKind.ARCHIVE:
            return output
    for output in result.outputs:
        if output.metadata.get("archive_format") == "tar.zst":
            return output
    raise ValueError("AppRunResult did not contain an archive output")


def _flatten_single_child_directory(directory: Path) -> None:
    """Move a single archived root directory's contents up one level."""
    children = list(directory.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return
    child = children[0]
    for nested_path in child.iterdir():
        destination = directory / nested_path.name
        if destination.exists() or os.path.lexists(destination):
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        nested_path.rename(destination)
    child.rmdir()


def _promote_directory_contents(source_dir: Path, destination_dir: Path) -> None:
    """Move extracted archive contents into their final output directory."""
    for nested_path in source_dir.iterdir():
        destination = destination_dir / nested_path.name
        if destination.exists() or os.path.lexists(destination):
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        nested_path.rename(destination)
    source_dir.rmdir()


def _link_mpnn_input_pdbs(
    pdb_files: list[Path],
    output_dir: Path,
    *,
    prefixes_by_stem: dict[str, str] | None = None,
) -> int:
    """Populate upstream-style ``mpnn_pdbs`` links from input PDB files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for pdb_path in pdb_files:
        prefix = (prefixes_by_stem or {}).get(pdb_path.stem)
        link_name = (
            f"{prefix}_{pdb_path.name}".lower() if prefix else pdb_path.name.lower()
        )
        link_path = output_dir / link_name
        if os.path.lexists(link_path):
            link_path.unlink()
        link_path.symlink_to(pdb_path.resolve())
        linked += 1
    return linked


def _write_mpnn_sequences_csv(
    *,
    fasta_root: Path,
    output_csv: Path,
    prefixes_by_stem: dict[str, str] | None = None,
) -> None:
    """Collect LigandMPNN FASTA files into FlowPacker's ``mpnn_seqs.csv``."""
    rows: list[dict[str, str | int]] = []
    for fasta_path in sorted([
        *fasta_root.rglob("*.fa"),
        *fasta_root.rglob("*.fasta"),
    ]):
        rows.extend(
            _mpnn_sequence_rows(
                fasta_path,
                fasta_root=fasta_root,
                prefixes_by_stem=prefixes_by_stem,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "link_name": pl.Utf8,
            "sequence_dict": pl.Utf8,
            "seq_idx": pl.Int64,
        },
        orient="row",
    ).write_csv(output_csv)


def _mpnn_sequence_rows(
    fasta_path: Path,
    *,
    fasta_root: Path,
    prefixes_by_stem: dict[str, str] | None = None,
) -> list[dict[str, str | int]]:
    """Parse one LigandMPNN FASTA file like upstream ``_process_one_fasta``."""
    fasta_name = fasta_path.name
    if fasta_name.endswith(".fa"):
        pdb_filename = f"{fasta_name[:-3]}.pdb"
    elif fasta_name.endswith(".fasta"):
        pdb_filename = f"{fasta_name[:-6]}.pdb"
    else:
        pdb_filename = f"{fasta_path.stem}.pdb"

    sequences = _read_fasta_sequences(fasta_path)
    rows: list[dict[str, str | int]] = []
    for seq_idx, sequence in enumerate(sequences[1:], start=1):
        parts = sequence.split("/")
        if len(parts) >= 2:
            sequence_dict = str({"A": parts[0], "B": parts[1]})
        else:
            sequence_dict = str({"A": parts[0]})
        prefix = _mpnn_fasta_link_prefix(
            fasta_path,
            fasta_root=fasta_root,
            prefixes_by_stem=prefixes_by_stem,
        )
        link_name = (
            f"{prefix}_{pdb_filename}".lower() if prefix else pdb_filename.lower()
        )
        rows.append({
            "link_name": link_name,
            "sequence_dict": sequence_dict,
            "seq_idx": seq_idx,
        })
    return rows


def _mpnn_fasta_link_prefix(
    fasta_path: Path,
    *,
    fasta_root: Path,
    prefixes_by_stem: dict[str, str] | None,
) -> str | None:
    """Resolve the input-PDB prefix for a LigandMPNN FASTA path."""
    if not prefixes_by_stem:
        return None
    try:
        relative_parts = fasta_path.relative_to(fasta_root).parts
    except ValueError:
        relative_parts = ()
    candidates = [relative_parts[0]] if relative_parts else []
    candidates.append(fasta_path.stem)
    for candidate in candidates:
        if prefix := prefixes_by_stem.get(candidate):
            return prefix
    return None


def _read_fasta_sequences(fasta_path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for raw_line in fasta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _load_partial_fixed_positions(context: NodeRunContext) -> dict[str, str]:
    """Load ``filename -> fixed_positions`` for PartialStep."""
    artifacts = context.inputs.get("fixed_positions") or []
    for artifact in artifacts:
        csv_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if csv_path.is_file():
            frame = pl.read_csv(csv_path)
            missing = {"filename", "fixed_positions"} - set(frame.columns)
            if missing:
                raise ValueError(
                    "fixed_positions CSV is missing column(s): "
                    + ", ".join(sorted(missing))
                )
            return {
                str(row["filename"]).strip(): str(row["fixed_positions"]).strip()
                for row in frame.iter_rows(named=True)
                if str(row["filename"]).strip()
            }
    raise FileNotFoundError("No fixed_positions CSV found in workflow inputs")


def _load_ligandmpnn_fixed_positions(
    context: NodeRunContext,
    *,
    use_abmpnn: bool,
) -> dict[str, str]:
    """Load optional stage-2 fixed positions for MPNN or AbMPNN app calls."""
    artifacts = context.inputs.get("fixed_positions") or []
    for artifact in artifacts:
        csv_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if not csv_path.is_file():
            continue
        frame = pl.read_csv(csv_path)
        missing = {"filename", "fixed_positions"} - set(frame.columns)
        if missing:
            raise ValueError(
                "fixed_positions CSV is missing column(s): "
                + ", ".join(sorted(missing))
            )
        fixed_by_name: dict[str, str] = {}
        for row in frame.iter_rows(named=True):
            filename = str(row["filename"]).strip()
            fixed_raw = str(row["fixed_positions"]).strip()
            if not filename:
                continue
            fixed_value = (
                _raw_fixed_positions_for_abmpnn(fixed_raw)
                if use_abmpnn
                else _normalize_fixed_positions_for_ligandmpnn(fixed_raw)
            )
            if fixed_value is None:
                continue
            fixed_by_name[filename.lower()] = fixed_value
            fixed_by_name[Path(filename).stem.lower()] = fixed_value
        return fixed_by_name
    return {}


def _fixed_positions_for_mpnn_pdb(
    pdb_path: Path,
    fixed_positions: dict[str, str],
) -> str | None:
    """Return the fixed-position string matching one MPNN input PDB."""
    candidates = [pdb_path.name.lower(), pdb_path.stem.lower()]
    candidates.extend(parent.name.lower() for parent in pdb_path.parents)
    for candidate in candidates:
        if candidate in fixed_positions:
            return fixed_positions[candidate]
    return None


def _normalize_fixed_positions_for_ligandmpnn(fixed_positions: str) -> str | None:
    """Convert upstream binder fixed positions into LigandMPNN residue numbers."""
    text = fixed_positions.strip()
    if not text or text.upper() == "NONE":
        return None
    positions: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"[A-Za-z]?(\d+)", token)
        if match is None:
            raise ValueError(f"Invalid fixed position token: {token!r}")
        positions.append(match.group(1))
    return " ".join(positions) if positions else None


def _raw_fixed_positions_for_abmpnn(fixed_positions: str) -> str | None:
    """Return upstream AbMPNN fixed positions without binder-only normalization."""
    text = fixed_positions.strip()
    if not text or text.upper() == "NONE":
        return None
    return text


def _ligandmpnn_link_prefixes(
    step_name: str,
    config: dict[str, Any],
    context: NodeRunContext,
    pdb_files: list[Path],
) -> dict[str, str]:
    """Return upstream ``mpnn_pdbs`` link prefixes keyed by input PDB stem."""
    if step_name.endswith("_stage1"):
        if not (name := config.get("name")):
            return {}
        prefix = sanitize_filename(str(name)).lower()
        return {pdb_path.stem: prefix for pdb_path in pdb_files}
    if not step_name.endswith("_stage2"):
        return {}

    artifact_roots = [
        _resolve_artifact_storage_path(artifact, context.cache_dir)
        for artifact in context.inputs.get("structures", [])
    ]
    prefixes: dict[str, str] = {}
    for pdb_path in pdb_files:
        for artifact_root in artifact_roots:
            if not artifact_root.is_dir():
                continue
            try:
                relative_path = pdb_path.relative_to(artifact_root)
            except ValueError:
                continue
            if len(relative_path.parts) > 1:
                prefixes[pdb_path.stem] = sanitize_filename(
                    relative_path.parts[0]
                ).lower()
            break
    return prefixes


def _partial_app_config(
    *,
    gentype: str,
    pdb_path: Path,
    fixed_positions: str,
    config: dict[str, Any],
) -> (
    ppiflow_app.SampleBinderPartialConfig
    | ppiflow_app.SampleAntibodyNanobodyPartialConfig
):
    """Build the PPIFlow app config for one upstream PartialStep input PDB."""
    if gentype == "binder":
        specified_hotspots = _pdb_bfactor_residues(
            pdb_path,
            bfactor=2.0,
            chain_id="B",
        )
        if not specified_hotspots:
            raise ValueError(
                "No binder specified_hotspots found from B-chain bfactor=2 "
                f"for PDB: {pdb_path}"
            )
        binder_args: dict[str, Any] = {
            "name": _partial_sample_name(pdb_path, config),
            "input_pdb": str(pdb_path),
            "target_chain": str(config.get("target_chain", "B")),
            "binder_chain": str(config.get("binder_chain", "A")),
            "fixed_positions": fixed_positions,
            "specified_hotspots": specified_hotspots,
            "samples_per_target": int(config.get("samples_per_target", 10)),
            "start_t": float(config.get("start_t", 0.7)),
            "sample_hotspot_rate_min": float(
                config.get("sample_hotspot_rate_min", 0.2)
            ),
            "sample_hotspot_rate_max": float(
                config.get("sample_hotspot_rate_max", 0.5)
            ),
        }
        if config.get("config") is not None:
            binder_args["config"] = config["config"]
        return ppiflow_app.SampleBinderPartialConfig(**binder_args)

    if gentype not in {"antibody", "nanobody"}:
        raise ValueError(f"Unsupported gentype: {gentype}")

    specified_hotspots = _pdb_bfactor_residues(pdb_path, bfactor=1.0, chain_id="C")
    cdr_position = _pdb_bfactor_residues(pdb_path, bfactor=2.0, chain_id=None)
    if not fixed_positions or fixed_positions.upper() == "NONE":
        raise ValueError(f"{gentype} PartialStep requires fixed_positions: {pdb_path}")
    if not specified_hotspots:
        raise ValueError(
            "No antigen specified_hotspots found from C-chain bfactor=1 "
            f"for PDB: {pdb_path}"
        )
    if not cdr_position:
        raise ValueError(f"No CDR positions found from bfactor=2 for PDB: {pdb_path}")
    antibody_args: dict[str, Any] = {
        "name": _partial_sample_name(pdb_path, config),
        "complex_pdb": str(pdb_path),
        "fixed_positions": fixed_positions,
        "cdr_position": cdr_position,
        "antigen_chain": str(config.get("antigen_chain", "C")),
        "heavy_chain": str(config.get("heavy_chain", "A")),
        "light_chain": config.get("light_chain") if gentype == "antibody" else None,
        "specified_hotspots": specified_hotspots,
        "samples_per_target": int(config.get("samples_per_target", 10)),
        "start_t": float(config.get("start_t", 0.8)),
        "retry_Limit": int(config.get("retry_Limit", 10)),
    }
    if config.get("config") is not None:
        antibody_args["config"] = config["config"]
    return ppiflow_app.SampleAntibodyNanobodyPartialConfig(**antibody_args)


def _partial_sample_name(pdb_path: Path, config: dict[str, Any]) -> str:
    pdb_stem = pdb_path.stem
    if name := config.get("name"):
        return f"{name}_{pdb_stem}"
    return pdb_stem


def _pdb_bfactor_residues(
    pdb_path: Path,
    *,
    bfactor: float,
    chain_id: str | None,
) -> str:
    """Return comma-separated residue IDs with any atom at a target B-factor."""
    residues: dict[tuple[str, str], str] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) < 66 or not line.startswith("ATOM"):
            continue
        if line[17:20].strip() not in PDB_RESIDUE_TO_ONE_LETTER:
            continue
        current_chain = line[21].strip()
        if chain_id is not None and current_chain != chain_id:
            continue
        try:
            atom_bfactor = float(line[60:66])
        except ValueError:
            continue
        if abs(atom_bfactor - bfactor) >= 1e-6:
            continue
        residue_id = f"{line[22:26].strip()}{line[26:27].strip()}"
        key = (current_chain, residue_id)
        residues.setdefault(key, f"{current_chain}{residue_id}")
    return ",".join(residues.values())


def _input_directory(context: NodeRunContext, *, input_name: str) -> Path:
    """Resolve the first directory artifact for a workflow input."""
    artifacts = context.inputs.get(input_name) or []
    for artifact in artifacts:
        artifact_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if artifact_path.is_dir():
            return artifact_path
    raise FileNotFoundError(
        f"No directory artifact found in workflow input {input_name!r}"
    )


def _normalize_rosetta_fix_output(
    *,
    output_dir: Path,
    pdb_files: list[Path],
    gentype: str,
    interface_dist: float,
) -> None:
    """Create upstream RosettaFix residue-energy CSVs from Rosetta app logs."""
    for pair_dir_name, binder_chain, target_chain in _rosetta_fix_chain_pairs(gentype):
        pair_output_dir = output_dir / pair_dir_name
        residue_energy_csv = pair_output_dir / "residue_energy.csv"
        if residue_energy_csv.is_file():
            continue
        rows: list[dict[str, object]] = []
        for index, pdb_file in enumerate(pdb_files, start=1):
            rosetta_log = _rosetta_fix_log_path(
                output_dir=output_dir,
                index=index,
                pdb_file=pdb_file,
            )
            if rosetta_log is None:
                continue
            interface_pairs = _pdb_interface_residue_pairs(
                pdb_file=pdb_file,
                binder_chain=binder_chain,
                target_chain=target_chain,
                distance_threshold=interface_dist,
            )
            binder_energy = _rosetta_fix_binder_energy(
                rosetta_log=rosetta_log,
                binder_chain=binder_chain,
                target_chain=target_chain,
                interface_pairs=interface_pairs,
            )
            rows.append({
                "pdbpath": str(pdb_file),
                "pdbname": pdb_file.stem,
                "rosetta_path": str(rosetta_log),
                "target_id": target_chain,
                "binder_id": binder_chain,
                "binder_energy": repr(binder_energy),
            })
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            rows,
            schema={
                "pdbpath": pl.Utf8,
                "pdbname": pl.Utf8,
                "rosetta_path": pl.Utf8,
                "target_id": pl.Utf8,
                "binder_id": pl.Utf8,
                "binder_energy": pl.Utf8,
            },
            orient="row",
        ).write_csv(residue_energy_csv)


def _rosetta_fix_chain_pairs(gentype: str) -> list[tuple[str, str, str]]:
    if gentype == "binder":
        return [("interface_energy_A_B", "A", "B")]
    if gentype == "nanobody":
        return [("interface_energy_A_C", "A", "C")]
    if gentype == "antibody":
        return [
            ("interface_energy_A_C", "A", "C"),
            ("interface_energy_B_C", "B", "C"),
        ]
    raise ValueError(f"Unsupported gentype: {gentype}")


def _rosetta_fix_log_path(
    *,
    output_dir: Path,
    index: int,
    pdb_file: Path,
) -> Path | None:
    candidates = [
        output_dir / str(index) / "rosetta.log",
        output_dir / str(index) / f"{pdb_file.stem}.out",
        output_dir / pdb_file.stem / "out" / f"{pdb_file.stem}.out",
        output_dir / pdb_file.stem / "rosetta.log",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _pdb_interface_residue_pairs(
    *,
    pdb_file: Path,
    binder_chain: str,
    target_chain: str,
    distance_threshold: float,
) -> set[tuple[int, int]]:
    residues = _pdb_cb_or_ca_coordinates(
        pdb_file.read_text(encoding="utf-8", errors="replace")
    )
    binder_residues = residues.get(binder_chain, {})
    target_residues = residues.get(target_chain, {})
    threshold_sq = distance_threshold * distance_threshold
    interface_pairs: set[tuple[int, int]] = set()
    for binder_resseq, binder_coord in binder_residues.items():
        for target_resseq, target_coord in target_residues.items():
            if _distance_sq(binder_coord, target_coord) <= threshold_sq:
                interface_pairs.add((binder_resseq, target_resseq))
    return interface_pairs


def _pdb_cb_or_ca_coordinates(
    pdb_text: str,
) -> dict[str, dict[int, tuple[float, float, float]]]:
    residues: dict[str, dict[int, tuple[str, tuple[float, float, float]]]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name not in {"CB", "CA"}:
            continue
        chain_id = line[21].strip()
        if not chain_id:
            continue
        try:
            res_seq = int(line[22:26].strip())
            coord = (
                float(line[30:38].strip()),
                float(line[38:46].strip()),
                float(line[46:54].strip()),
            )
        except ValueError:
            continue
        chain = residues.setdefault(chain_id, {})
        current = chain.get(res_seq)
        if current is None or (current[0] == "CA" and atom_name == "CB"):
            chain[res_seq] = (atom_name, coord)
    return {
        chain_id: {
            resseq: coord for resseq, (_atom_name, coord) in chain_residues.items()
        }
        for chain_id, chain_residues in residues.items()
    }


def _distance_sq(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return (
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    )


def _rosetta_fix_binder_energy(
    *,
    rosetta_log: Path,
    binder_chain: str,
    target_chain: str,
    interface_pairs: set[tuple[int, int]],
) -> dict[int, float]:
    binder_energy: dict[int, float] = {}
    for row in _parse_rosetta_resrese_rows(rosetta_log):
        left = _parse_rosetta_residue_token(str(row.get("Res1", "")))
        right = _parse_rosetta_residue_token(str(row.get("Res2", "")))
        total = _safe_float(row.get("total"))
        if left is None or right is None or total is None or total >= 0:
            continue
        left_chain, left_resseq = left
        right_chain, right_resseq = right
        binder_resseq: int | None = None
        if (
            left_chain == binder_chain
            and right_chain == target_chain
            and (left_resseq, right_resseq) in interface_pairs
        ):
            binder_resseq = left_resseq
        elif (
            left_chain == target_chain
            and right_chain == binder_chain
            and (right_resseq, left_resseq) in interface_pairs
        ):
            binder_resseq = right_resseq
        if binder_resseq is None:
            continue
        binder_energy[binder_resseq] = binder_energy.get(binder_resseq, 0.0) + total
    return dict(sorted(binder_energy.items()))


def _parse_rosetta_resrese_rows(rosetta_log: Path) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in rosetta_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ResResE"):
            continue
        parts = line.split()
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        if _safe_float(row.get("total")) is None:
            continue
        rows.append(row)
    return rows


def _parse_rosetta_residue_token(token: str) -> tuple[str, int] | None:
    residue = token.split("_")[-1]
    if match := re.fullmatch(r"(\d{2})(-?\d+)", residue):
        chain_ord = int(match.group(1))
        if 65 <= chain_ord <= 90:
            return chr(chain_ord), int(match.group(2))
    if match := re.fullmatch(r"([A-Za-z])(-?\d+)", residue):
        return match.group(1), int(match.group(2))
    if match := re.fullmatch(r"(-?\d+)([A-Za-z])", residue):
        return match.group(2), int(match.group(1))
    return None


def _normalize_rosetta_relax_output(*, output_dir: Path) -> None:
    """Create upstream RosettaRelax ``rosetta_complex_0.csv`` when possible."""
    output_csv = output_dir / "rosetta_complex_0.csv"
    if output_csv.is_file():
        return
    existing_csv = _existing_rosetta_complex_csv(output_dir)
    if existing_csv is not None:
        shutil.copy2(existing_csv, output_csv)
        return
    rows = _rosetta_scorefile_rows(output_dir)
    if rows:
        pl.DataFrame(rows, orient="row").write_csv(output_csv)


def _existing_rosetta_complex_csv(output_dir: Path) -> Path | None:
    for csv_path in sorted(output_dir.rglob("*.csv")):
        if csv_path.name == "rosetta_complex_0.csv":
            continue
        columns = _csv_columns(csv_path)
        if {"pdb_name", "interface_score"}.issubset(columns):
            return csv_path
    return None


def _csv_columns(csv_path: Path) -> set[str]:
    try:
        return set(pl.read_csv(csv_path, n_rows=0).columns)
    except (OSError, pl.exceptions.PolarsError):
        return set()


def _rosetta_scorefile_rows(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scorefile in sorted(output_dir.rglob("*.sc")):
        rows.extend(_parse_rosetta_scorefile(scorefile))
    return rows


def _parse_rosetta_scorefile(scorefile: Path) -> list[dict[str, object]]:
    header: list[str] | None = None
    rows: list[dict[str, object]] = []
    for line in scorefile.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SCORE:"):
            continue
        parts = line.split()[1:]
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            continue
        raw_row = dict(zip(header, parts, strict=True))
        interface_score = raw_row.get("interface_score") or raw_row.get(
            "interface_delta"
        )
        if interface_score is None:
            continue
        row: dict[str, object] = {
            "pdb_name": Path(str(raw_row.get("description", ""))).stem
        }
        for key in header:
            if key == "description":
                continue
            value: object = raw_row[key]
            if (number := _safe_float(value)) is not None:
                value = number
            row[key] = value
        row["interface_score"] = float(interface_score)
        rows.append(row)
    return rows


def _build_fixed_positions_frame(
    *,
    rosetta_fix_output_dir: Path,
    gentype: str,
    energy_threshold: float,
) -> pl.DataFrame:
    """Convert RosettaFix residue-energy CSVs into PartialStep fixed positions."""
    energy_specs = _fixed_position_energy_specs(gentype)
    merged_energy: dict[str, dict[str, dict[int, float]]] = {}

    for subdir_name, binder_chain in energy_specs:
        energy_csv = rosetta_fix_output_dir / subdir_name / "residue_energy.csv"
        if not energy_csv.is_file():
            raise FileNotFoundError(
                f"Expected residue_energy.csv not found: {energy_csv}"
            )

        for row in pl.read_csv(energy_csv).iter_rows(named=True):
            pdbname = str(row.get("pdbname") or "")
            if not pdbname:
                pdbname = Path(str(row.get("pdbpath") or "")).stem
            if not pdbname:
                continue

            parts = pdbname.split("_")
            structure_name = "_".join(parts[:-1]) if len(parts) > 2 else parts[0]
            binder_energy = _literal_energy_dict(row.get("binder_energy"))
            chain_energy = merged_energy.setdefault(structure_name, {}).setdefault(
                binder_chain, {}
            )
            for resseq, energy in binder_energy.items():
                try:
                    resseq_int = int(resseq)
                    energy_value = float(energy)
                except (TypeError, ValueError):
                    continue
                existing = chain_energy.get(resseq_int)
                chain_energy[resseq_int] = (
                    energy_value if existing is None else min(existing, energy_value)
                )

    output_rows: list[dict[str, str]] = []
    for structure_name in sorted(merged_energy):
        positions: list[str] = []
        for binder_chain in sorted(merged_energy[structure_name]):
            chain_energy = merged_energy[structure_name][binder_chain]
            positions.extend(
                f"{binder_chain}{resseq}"
                for resseq, energy in sorted(chain_energy.items())
                if energy < energy_threshold
            )
        output_rows.append({
            "filename": f"{structure_name}.pdb",
            "fixed_positions": ",".join(positions) if positions else "NONE",
        })

    return pl.DataFrame(
        output_rows,
        schema={"filename": pl.Utf8, "fixed_positions": pl.Utf8},
        orient="row",
    )


def _link_before_partial_pdbs(
    *,
    fixed_positions: pl.DataFrame,
    pdb_files: list[Path],
    output_dir: Path,
) -> int:
    """Link PDBs named in ``fixed_positions.csv`` for PartialStep inputs."""
    pdb_by_name = {path.name: path for path in pdb_files}
    linked_count = 0
    for row in fixed_positions.select("filename").iter_rows(named=True):
        filename = str(row["filename"]).strip()
        if not filename:
            continue
        source = pdb_by_name.get(Path(filename).name)
        if source is None or not os.path.lexists(source):
            continue
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            destination.unlink()
        destination.symlink_to(source.resolve())
        linked_count += 1
    return linked_count


def _fixed_position_energy_specs(gentype: str) -> list[tuple[str, str]]:
    if gentype == "binder":
        return [("interface_energy_A_B", "A")]
    if gentype == "nanobody":
        return [("interface_energy_A_C", "A")]
    if gentype == "antibody":
        return [
            ("interface_energy_A_C", "A"),
            ("interface_energy_B_C", "B"),
        ]
    raise ValueError(f"Unsupported gentype: {gentype}")


def _literal_energy_dict(value: object) -> dict[object, object]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_filter_string(expr: str) -> dict[str, float]:
    """Parse upstream filter clauses such as ``'> 0.7, <= 1.0'``."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"Invalid filter expression: {expr!r}")

    parsed: dict[str, float] = {}
    pattern = re.compile(r"^(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$")
    for part in (part.strip() for part in expr.split(",") if part.strip()):
        match = pattern.match(part)
        if match is None:
            raise ValueError(
                f"Invalid filter clause: {part!r}. "
                "Expected format like '> 0.7' or '<= 10'"
            )
        op, value = match.groups()
        parsed[op] = float(value)
    if not parsed:
        raise ValueError(f"Invalid filter expression: {expr!r}")
    return parsed


def _parse_filters_cfg(filters_cfg: object) -> dict[str, float | dict[str, float]]:
    """Convert upstream YAML filter config into normalized filter conditions."""
    if not isinstance(filters_cfg, dict) or not filters_cfg:
        raise ValueError("'filters' config must be a non-empty dict.")

    parsed: dict[str, float | dict[str, float]] = {}
    for metric, condition in filters_cfg.items():
        metric_name = str(metric)
        if isinstance(condition, str):
            parsed[metric_name] = _parse_filter_string(condition)
        elif isinstance(condition, int | float):
            parsed[metric_name] = float(condition)
        elif isinstance(condition, dict):
            parsed_ops: dict[str, float] = {}
            for op, value in condition.items():
                if op not in VALID_FILTER_OPERATORS:
                    raise ValueError(f"Invalid operator: {op}")
                if not isinstance(value, int | float):
                    raise ValueError("Threshold must be numeric.")
                parsed_ops[str(op)] = float(value)
            parsed[metric_name] = parsed_ops
        else:
            raise ValueError(
                f"Unsupported filter config for metric {metric_name!r}: {condition!r}"
            )
    return parsed


def _filter_metrics_csv(
    *,
    metrics_csv: Path,
    filters: dict[str, float | dict[str, float]],
    filename_col: str,
) -> pl.DataFrame:
    """Read AF3Score metrics and return rows passing all configured filters."""
    frame = pl.read_csv(metrics_csv)
    missing_columns = {filename_col, *filters.keys()} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Metrics CSV is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    predicate = pl.lit(True)
    for metric, condition in filters.items():
        metric_col = pl.col(metric).cast(pl.Float64)
        if isinstance(condition, dict):
            for op, threshold in condition.items():
                predicate = predicate & _filter_expr(metric_col, op, threshold)
        else:
            predicate = predicate & (metric_col >= condition)
    return frame.filter(predicate)


def _filter_expr(column: pl.Expr, op: str, threshold: float) -> pl.Expr:
    if op == ">":
        return column > threshold
    if op == ">=":
        return column >= threshold
    if op == "<":
        return column < threshold
    if op == "<=":
        return column <= threshold
    if op == "==":
        return column == threshold
    if op == "!=":
        return column != threshold
    raise ValueError(f"Invalid operator: {op}")


def _input_score_csv(context: NodeRunContext) -> Path:
    """Resolve the AF3Score metrics CSV from workflow score artifacts."""
    artifacts = context.inputs.get("scores") or []
    for artifact in artifacts:
        score_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if score_path.is_file() and score_path.suffix == ".csv":
            return score_path
        if score_path.is_dir():
            metrics_path = score_path / af3score_app.APP_INFO.metrics_filename
            if metrics_path.is_file():
                return metrics_path
            csv_files = sorted(score_path.rglob("*.csv"))
            if csv_files:
                return csv_files[0]
    raise FileNotFoundError("No AF3Score metrics CSV found in workflow score inputs")


def _filter_output_name(step_name: str) -> str:
    if step_name == "FilterStep_stage1":
        return "filtered_iptm07"
    if step_name == "FilterStep_stage2":
        return "filtered_iptm08"
    return sanitize_filename(step_name)


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _link_filtered_pdbs(
    *,
    filtered_rows: pl.DataFrame,
    pdb_files: list[Path],
    output_dir: Path,
    filename_col: str,
) -> int:
    pdb_by_name = {path.name: path for path in pdb_files}
    linked_count = 0
    for row in filtered_rows.select(filename_col).iter_rows(named=True):
        raw_filename = row.get(filename_col)
        if raw_filename is None:
            continue
        filename = str(raw_filename)
        source = pdb_by_name.get(Path(filename).name)
        if source is None or not source.is_file():
            continue
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(destination):
            destination.unlink()
        destination.symlink_to(source.resolve())
        linked_count += 1
    return linked_count


def _input_pdb_files(context: NodeRunContext, *, input_name: str) -> list[Path]:
    """Resolve PDB files from a workflow input artifact collection."""
    return _input_structure_files(context, input_name=input_name, suffixes={".pdb"})


def _input_structure_files(
    context: NodeRunContext,
    *,
    input_name: str,
    suffixes: set[str],
) -> list[Path]:
    """Resolve structure files from a workflow input artifact collection."""
    artifacts = context.inputs.get(input_name) or []
    structure_files: list[Path] = []
    for artifact in artifacts:
        artifact_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if artifact_path.is_file() and artifact_path.suffix.lower() in suffixes:
            structure_files.append(artifact_path)
            continue
        if artifact_path.is_dir():
            structure_files.extend(
                sorted(
                    path
                    for path in artifact_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in suffixes
                )
            )
    if not structure_files:
        suffix_label = ", ".join(sorted(suffixes))
        raise FileNotFoundError(
            f"No structure files ({suffix_label}) found in workflow input {input_name!r}"
        )
    return sorted(structure_files)


def _dockq_pairs(context: NodeRunContext) -> list[dict[str, object]]:
    """Pair ReFold model structures with filtered reference complexes."""
    reference_files = _input_pdb_files(context, input_name="structures")
    model_files = _input_structure_files(
        context,
        input_name="models",
        suffixes={".pdb", ".cif"},
    )
    references_by_stem = {path.stem: path for path in reference_files}
    pairs: list[dict[str, object]] = []
    for model_path in model_files:
        reference_path = _matching_dockq_reference(
            model_path,
            references_by_stem=references_by_stem,
        )
        if reference_path is None:
            continue
        pairs.append({
            "id": reference_path.stem,
            "model_name": model_path.name,
            "model_bytes": model_path.read_bytes(),
            "reference_name": reference_path.name,
            "reference_bytes": reference_path.read_bytes(),
        })
    if not pairs:
        raise FileNotFoundError("No DockQ model/reference pairs could be assembled")
    return pairs


def _matching_dockq_reference(
    model_path: Path,
    *,
    references_by_stem: dict[str, Path],
) -> Path | None:
    """Find the filtered reference corresponding to one ReFold model."""
    candidates = [model_path.stem, model_path.parent.name]
    candidates.extend(
        model_path.stem.removeprefix(prefix)
        for prefix in ("fold_", "model_")
        if model_path.stem.startswith(prefix)
    )
    candidates.extend(
        model_path.stem.removesuffix(suffix)
        for suffix in ("_model", "_rank_001", "_seed_1")
        if model_path.stem.endswith(suffix)
    )
    for candidate in candidates:
        reference = references_by_stem.get(candidate)
        if reference is not None:
            return reference
    return None


def _rank_design_rows(
    *,
    context: NodeRunContext,
    gentype: str,
    dockq_threshold: float,
) -> list[dict[str, object]]:
    """Build ranked design rows from DockQ, Rosetta, ReFold, and filtered PDBs."""
    dockq_rows = _best_dockq_rows(
        _input_csv_file(
            context, input_name="dockq", preferred_names=("dockq_results.csv",)
        ),
        dockq_threshold=dockq_threshold,
    )
    if not dockq_rows:
        return []
    rosetta_rows = _csv_rows(
        _input_csv_file(
            context,
            input_name="rosetta",
            preferred_names=("rosetta_complex_0.csv",),
        )
    )
    refold_rows = _csv_rows(
        _input_csv_file(
            context,
            input_name="refold",
            preferred_names=("af3_iptm@1.csv",),
        )
    )
    if not rosetta_rows or not refold_rows:
        raise ValueError("Rosetta or ReFold ranking CSV is empty.")
    rank_metric_col = _rank_metric_col(gentype, refold_rows)
    rosetta_index = _index_rank_rows(rosetta_rows)
    refold_index = _index_rank_rows(refold_rows)
    pdb_by_stem = {
        path.stem: path for path in _input_pdb_files(context, input_name="structures")
    }

    ranked_rows: list[dict[str, object]] = []
    for dockq_row in dockq_rows:
        target_name = str(dockq_row["target_name"])
        rosetta_row = rosetta_index.get(target_name)
        refold_row = refold_index.get(target_name)
        if rosetta_row is None or refold_row is None:
            continue
        interface_score = _safe_float(rosetta_row.get("interface_score"))
        rank_metric = _safe_float(refold_row.get(rank_metric_col))
        if interface_score is None or rank_metric is None:
            continue
        merged: dict[str, object] = dict(refold_row)
        merged.update({
            "target_name": target_name,
            "reference_pdb": dockq_row["reference_pdb"],
            "model_pdb": dockq_row["model_pdb"],
            "dockq_mean": dockq_row["dockq_mean"],
            "pdb_name": str(rosetta_row.get("pdb_name") or target_name),
            "interface_score": interface_score,
            "rank_score": interface_score + 100.0 * rank_metric,
        })
        if pdb_path := pdb_by_stem.get(target_name):
            sequences = dict(
                _pdb_sequences_by_chain(
                    pdb_path.read_text(encoding="utf-8", errors="replace")
                )
            )
            if gentype in {"binder", "nanobody"}:
                merged["binder_seq"] = sequences.get("A", "")
            elif gentype == "antibody":
                merged["heavy_seq"] = sequences.get("A", "")
                merged["light_seq"] = sequences.get("B", "")
        ranked_rows.append(merged)
    ranked_rows.sort(key=lambda row: float(row["rank_score"]), reverse=True)
    return ranked_rows


def _best_dockq_rows(
    dockq_csv: Path,
    *,
    dockq_threshold: float,
) -> list[dict[str, object]]:
    """Return the best DockQ row per target above the configured threshold."""
    best_rows: dict[str, dict[str, object]] = {}
    for row in _csv_rows(dockq_csv):
        target_name = _dockq_target_name(row)
        if not target_name:
            continue
        dockq_mean = _dockq_mean(row)
        if dockq_mean is None or dockq_mean <= dockq_threshold:
            continue
        candidate = {
            "target_name": target_name,
            "reference_pdb": _first_nonempty(row, "reference_pdb", "reference"),
            "model_pdb": _first_nonempty(row, "model_pdb", "model"),
            "dockq_mean": dockq_mean,
        }
        existing = best_rows.get(target_name)
        if existing is None or dockq_mean > float(existing["dockq_mean"]):
            best_rows[target_name] = candidate
    return list(best_rows.values())


def _dockq_target_name(row: dict[str, object]) -> str:
    raw_name = _first_nonempty(row, "target_name", "id", "reference", "reference_pdb")
    if not raw_name:
        return ""
    return Path(raw_name).stem


def _dockq_mean(row: dict[str, object]) -> float | None:
    if dockq := _safe_float(row.get("dockq")):
        return dockq
    numeric_values = [
        value
        for key, raw_value in row.items()
        if key
        not in {
            "id",
            "target_name",
            "model",
            "model_pdb",
            "reference",
            "reference_pdb",
            "mapping",
            "returncode",
            "error",
            "log",
        }
        if (value := _safe_float(raw_value)) is not None
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _rank_metric_col(gentype: str, rows: list[dict[str, object]]) -> str:
    columns = set(rows[0]) if rows else set()
    if gentype == "binder":
        if "iptm" not in columns:
            raise ValueError("ReFold CSV must contain column 'iptm' for binder mode.")
        return "iptm"
    if gentype in {"antibody", "nanobody"}:
        for candidate in ("iptm_A_C", "chain_A_iptm"):
            if candidate in columns:
                return candidate
        raise ValueError(
            "ReFold CSV must contain 'iptm_A_C' or 'chain_A_iptm' for antibody/nanobody mode."
        )
    raise ValueError(f"Unsupported gentype: {gentype}")


def _index_rank_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for row in rows:
        for key in ("pdb_name", "target_name", "description", "id"):
            raw_value = row.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if not value:
                continue
            index.setdefault(value, row)
            index.setdefault(Path(value).stem, row)
    return index


def _write_ranked_designs_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    pl.DataFrame(rows, orient="row").write_csv(csv_path)


def _copy_ranked_design_pdbs(
    *,
    rows: list[dict[str, object]],
    pdb_files: list[Path],
    output_dir: Path,
) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_by_stem = {path.stem: path for path in pdb_files}
    for row in rows:
        target_name = str(row.get("target_name") or row.get("pdb_name") or "").strip()
        source = pdb_by_stem.get(target_name)
        if source is None:
            continue
        shutil.copy2(source, output_dir / f"{target_name}.pdb")


def _input_csv_file(
    context: NodeRunContext,
    *,
    input_name: str,
    preferred_names: tuple[str, ...],
) -> Path:
    """Resolve a CSV file from one named workflow input."""
    artifacts = context.inputs.get(input_name) or []
    for artifact in artifacts:
        artifact_path = _resolve_artifact_storage_path(artifact, context.cache_dir)
        if artifact_path.is_file() and artifact_path.suffix.lower() == ".csv":
            return artifact_path
        if artifact_path.is_dir():
            for preferred_name in preferred_names:
                preferred = artifact_path / preferred_name
                if preferred.is_file():
                    return preferred
            csv_files = sorted(artifact_path.rglob("*.csv"))
            if csv_files:
                return csv_files[0]
    raise FileNotFoundError(f"No CSV file found in workflow input {input_name!r}")


def _csv_rows(csv_path: Path) -> list[dict[str, object]]:
    """Read CSV rows with polars and normalize null cells."""
    return [
        {key: "" if value is None else value for key, value in row.items()}
        for row in pl.read_csv(csv_path, infer_schema_length=0).iter_rows(named=True)
    ]


def _first_nonempty(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_config(config: dict[str, Any]) -> dict[str, Any]:
    report_cfg = config.get("ReportStep", config)
    return report_cfg if isinstance(report_cfg, dict) else {}


def _render_report_html(
    *,
    context: NodeRunContext,
    gentype: str,
    design_name: str,
    ranked_csv: Path,
) -> str:
    """Render a compact final PPIFlow design report."""
    ranked_rows = _csv_rows(ranked_csv)
    stage1_rows = _optional_csv_rows(
        context,
        input_name="stage1_af3score",
        preferred_names=("af3score_metrics.csv",),
    )
    stage2_rows = _optional_csv_rows(
        context,
        input_name="stage2_af3score",
        preferred_names=("af3score_metrics.csv",),
    )
    refold_rows = _optional_csv_rows(
        context,
        input_name="refold",
        preferred_names=("af3_iptm@1.csv",),
    )
    top_row = max(
        ranked_rows,
        key=lambda row: _safe_float(row.get("rank_score")) or float("-inf"),
        default=None,
    )
    top_pdb_path = _top_ranked_pdb_path(ranked_csv.parent, top_row)
    top_pdb_b64 = (
        base64.b64encode(top_pdb_path.read_bytes()).decode("ascii")
        if top_pdb_path is not None
        else ""
    )
    top_pdb_name = top_pdb_path.name if top_pdb_path is not None else "N/A"
    table_columns = _report_table_columns(ranked_rows, refold_rows, gentype)
    table_rows = _report_table_rows(ranked_rows, refold_rows, table_columns)
    return "\n".join([
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>PPIFlow Design Report - {html_lib.escape(design_name)}</title>",
        "</head>",
        "<body>",
        "  <h1>PPIFlow Design Report</h1>",
        f"  <p>Design: {html_lib.escape(design_name)} | Type: {html_lib.escape(gentype)}</p>",
        "  <h2>Pipeline Funnel - Design Counts</h2>",
        "  <table>",
        "    <tr><th>Step</th><th>Designs</th></tr>",
        f"    <tr><td>AF3scoreStep stage 1</td><td>{len(stage1_rows)}</td></tr>",
        f"    <tr><td>AF3scoreStep stage 2</td><td>{len(stage2_rows)}</td></tr>",
        f"    <tr><td>ReFoldStep</td><td>{len(refold_rows)}</td></tr>",
        f"    <tr><td>RankStep</td><td>{len(ranked_rows)}</td></tr>",
        "  </table>",
        "  <h2>Top-Ranked Design Structure</h2>",
        f"  <p>{html_lib.escape(top_pdb_name)}</p>",
        f'  <script type="application/pdb" id="top-pdb">{top_pdb_b64}</script>',
        "  <h2>All Ranked Designs</h2>",
        _html_table(table_columns, table_rows),
        "</body>",
        "</html>",
    ])


def _optional_csv_rows(
    context: NodeRunContext,
    *,
    input_name: str,
    preferred_names: tuple[str, ...],
) -> list[dict[str, object]]:
    try:
        return _csv_rows(
            _input_csv_file(
                context,
                input_name=input_name,
                preferred_names=preferred_names,
            )
        )
    except FileNotFoundError:
        return []


def _top_ranked_pdb_path(
    output_dir: Path,
    top_row: dict[str, object] | None,
) -> Path | None:
    if top_row is None:
        return None
    pdb_name = _first_nonempty(top_row, "pdb_name", "target_name")
    if not pdb_name:
        return None
    pdb_path = output_dir / "pdbs" / f"{Path(pdb_name).stem}.pdb"
    return pdb_path if pdb_path.is_file() else None


def _report_table_columns(
    ranked_rows: list[dict[str, object]],
    refold_rows: list[dict[str, object]],
    gentype: str,
) -> list[str]:
    if not ranked_rows:
        return []
    refold_columns = set(refold_rows[0]) if refold_rows else set()
    metric_candidates = (
        ("chain_A_ptm", "iptm")
        if gentype in {"binder", "nanobody"}
        else ("chain_A_ptm", "chain_B_ptm", "iptm_A_C", "iptm_B_C")
    )
    sequence_columns = [
        column
        for column in ("binder_seq", "heavy_seq", "light_seq")
        if column in ranked_rows[0]
    ]
    return [
        "pdb_name",
        "rank_score",
        "interface_score",
        "dockq_mean",
        *sequence_columns,
        *(column for column in metric_candidates if column in refold_columns),
    ]


def _report_table_rows(
    ranked_rows: list[dict[str, object]],
    refold_rows: list[dict[str, object]],
    columns: list[str],
) -> list[dict[str, object]]:
    refold_index = _index_rank_rows(refold_rows)
    table_rows: list[dict[str, object]] = []
    for ranked_row in ranked_rows:
        pdb_name = _first_nonempty(ranked_row, "pdb_name", "target_name")
        refold_row = refold_index.get(pdb_name, {})
        table_rows.append({
            column: ranked_row.get(column, refold_row.get(column, ""))
            for column in columns
        })
    return table_rows


def _html_table(columns: list[str], rows: list[dict[str, object]]) -> str:
    if not columns:
        return "  <p>No ranked designs.</p>"
    header = "".join(f"<th>{html_lib.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html_lib.escape(str(row.get(column, '')))}</td>"
            for column in columns
        )
        body_rows.append(f"    <tr>{cells}</tr>")
    return "\n".join([
        "  <table>",
        f"    <tr>{header}</tr>",
        *body_rows,
        "  </table>",
    ])


def _resolve_artifact_storage_path(artifact: WorkflowArtifact, cache_dir: Path) -> Path:
    """Find an input artifact path mounted under the current workflow volume."""
    return _resolve_volume_storage_path(artifact.storage, cache_dir)


def _resolve_volume_storage_path(storage: VolumePath, cache_dir: Path) -> Path:
    """Find a volume path mounted under the current workflow volume."""
    relative_path = storage.path
    for root in (cache_dir, *cache_dir.parents):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Workflow artifact path not found: {storage}")


def _context_volume_path(
    path: Path,
    *,
    context: NodeRunContext,
    seed_artifacts: list[WorkflowArtifact],
) -> VolumePath:
    """Build a workflow VolumePath for a file written under ``context.cache_dir``."""
    resolved_path = path.resolve()
    orchestrator_mount = Path(orchestrator.CONF.output_volume_mountpoint).resolve()
    try:
        return VolumePath(
            volume_name=orchestrator.OUT_VOLUME_NAME,
            path=resolved_path.relative_to(orchestrator_mount).as_posix(),
        )
    except ValueError:
        pass

    for artifact in seed_artifacts:
        for root in (context.cache_dir, *context.cache_dir.parents):
            if not root.joinpath(artifact.storage.path).exists():
                continue
            try:
                return VolumePath(
                    volume_name=artifact.storage.volume_name,
                    path=resolved_path.relative_to(root.resolve()).as_posix(),
                )
            except ValueError:
                continue

    return VolumePath(
        volume_name=orchestrator.OUT_VOLUME_NAME,
        path=resolved_path.relative_to(context.cache_dir.resolve()).as_posix(),
    )


def _extract_refold_metrics(
    *,
    archive_path: Path,
    extract_dir: Path,
    pdb_name: str,
    chain_ids: list[str],
) -> list[dict[str, object]]:
    """Extract AlphaFold3 summary confidences from one ReFold archive."""
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    try:
        _extract_tar_zst_archive(archive_path, extract_dir)
    except (RuntimeError, sp.CalledProcessError):
        return []
    _flatten_single_child_directory(extract_dir)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(extract_dir.rglob("summary_confidences.json")):
        summary = orjson.loads(summary_path.read_bytes())
        row: dict[str, object] = {
            "pdb_name": pdb_name,
            "seed": summary_path.parent.name,
            "ptm": summary.get("ptm", 0.0),
            "iptm": summary.get("iptm", 0.0),
        }
        chain_ptm = summary.get("chain_ptm", [])
        chain_iptm = summary.get("chain_iptm", [])
        for index, chain_id in enumerate(chain_ids):
            row[f"chain_{chain_id}_ptm"] = _list_value(chain_ptm, index)
            row[f"chain_{chain_id}_iptm"] = _list_value(chain_iptm, index)
        chain_pair_iptm = summary.get("chain_pair_iptm", [])
        for left_idx, left_chain in enumerate(chain_ids):
            for right_idx, right_chain in enumerate(
                chain_ids[left_idx + 1 :], left_idx + 1
            ):
                row[f"iptm_{left_chain}_{right_chain}"] = _matrix_value(
                    chain_pair_iptm,
                    left_idx,
                    right_idx,
                )
        rows.append(row)
    return rows


def _write_refold_metrics(output_dir: Path, rows: list[dict[str, object]]) -> None:
    """Write all ReFold metrics plus the best-iptm row per PDB."""
    if not rows:
        return
    pl.DataFrame(rows, orient="row").write_csv(output_dir / "af3_metrics.csv")
    best_by_pdb: dict[str, dict[str, object]] = {}
    for row in rows:
        pdb_name = str(row.get("pdb_name") or "")
        iptm = _safe_float(row.get("iptm")) or float("-inf")
        existing = best_by_pdb.get(pdb_name)
        existing_iptm = (
            _safe_float(existing.get("iptm")) if existing is not None else None
        )
        if existing is None or iptm > (existing_iptm or float("-inf")):
            best_by_pdb[pdb_name] = row
    pl.DataFrame(list(best_by_pdb.values()), orient="row").write_csv(
        output_dir / "af3_iptm@1.csv"
    )


def _list_value(values: object, index: int) -> object:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def _matrix_value(values: object, row_index: int, col_index: int) -> object:
    if (
        isinstance(values, list)
        and row_index < len(values)
        and isinstance(values[row_index], list)
        and col_index < len(values[row_index])
    ):
        return values[row_index][col_index]
    return None


def _alphafold3_refold_json_bytes(
    *,
    name: str,
    pdb_bytes: bytes,
    seed_count: int,
) -> bytes:
    """Convert one PDB complex into an AlphaFold3 input JSON payload."""
    chains = _pdb_sequences_by_chain(pdb_bytes.decode("utf-8", errors="replace"))
    if not chains:
        raise ValueError(f"No protein CA residues found in PDB for {name!r}")

    sequences = [
        {
            "protein": {
                "id": chain_id,
                "sequence": sequence,
                "unpairedMsa": None,
                "pairedMsa": None,
                "templates": None,
            }
        }
        for chain_id, sequence in chains
    ]
    return orjson.dumps(
        {
            "name": sanitize_filename(name),
            "sequences": sequences,
            "modelSeeds": _alphafold3_model_seeds(seed_count),
            "dialect": "alphafold3",
            "version": 1,
        },
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )


def _pdb_sequences_by_chain(pdb_text: str) -> list[tuple[str, str]]:
    """Extract chain sequences from PDB CA records in first-seen chain order."""
    residues: dict[str, list[tuple[str, str]]] = {}
    for line in pdb_text.splitlines():
        if len(line) < 26 or not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        residue = PDB_RESIDUE_TO_ONE_LETTER.get(line[17:20].strip())
        if residue is None:
            continue
        chain_id = line[21].strip() or "A"
        residue_id = f"{line[22:26].strip()}{line[26:27].strip()}"
        chain_residues = residues.setdefault(chain_id, [])
        if chain_residues and chain_residues[-1][0] == residue_id:
            continue
        chain_residues.append((residue_id, residue))
    return [
        (chain_id, "".join(residue for _, residue in chain_residues))
        for chain_id, chain_residues in residues.items()
    ]


def _alphafold3_model_seeds(seed_count: int) -> list[int]:
    """Return upstream ReFold/gen_json-compatible AlphaFold3 model seeds."""
    if seed_count < 1:
        raise ValueError("seed_num must be at least 1")
    fixed_seeds = list(AF3_REFOLD_SEEDS)
    extended = fixed_seeds + [seed for seed in range(1, 201) if seed not in fixed_seeds]
    return extended[: min(seed_count, len(extended))]


def _remove_alphafold3_templates_json(json_bytes: bytes) -> bytes:
    """Replace every ``templates`` value in an AlphaFold3 JSON document with ``[]``."""
    data = orjson.loads(json_bytes)
    _set_templates_empty(data)
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def _set_templates_empty(node: Any) -> None:
    if isinstance(node, dict):
        if "templates" in node:
            node["templates"] = []
        for value in node.values():
            _set_templates_empty(value)
    elif isinstance(node, list):
        for item in node:
            _set_templates_empty(item)


def _marker_result(step_name: str, kind: ArtifactKind) -> AppRunResult:
    """Return a small text artifact for workflow-native adapter placeholders."""
    safe_step_name = sanitize_filename(step_name)
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name=f"{safe_step_name}_manifest",
                kind=kind,
                storage=InlineBytes(
                    data=f"{step_name}\n".encode(),
                    filename=f"{safe_step_name}.txt",
                    media_type="text/plain",
                ),
                metadata={"step_name": step_name},
            )
        ],
    )


def _retag_result_outputs(result: AppRunResult, kind: ArtifactKind) -> AppRunResult:
    """Return a copy of an app result with every output retagged for selectors."""
    return result.model_copy(
        update={
            "outputs": [
                output.model_copy(update={"kind": kind}) for output in result.outputs
            ]
        }
    )


def _stage_ppiflow_result_outputs(
    *,
    result: AppRunResult,
    modal_namespace: PPIFlowModalNamespace,
    context: NodeRunContext,
) -> AppRunResult:
    """Copy PPIFlow app-volume outputs into the workflow volume when needed."""
    if modal_namespace.ppiflow_stage_outputs is None:
        return result
    outputs_to_stage: list[dict[str, object]] = []
    for index, output in enumerate(result.outputs):
        if (
            isinstance(output.storage, VolumePath)
            and output.storage.volume_name == ppiflow_app.CONF.output_volume_name
        ):
            outputs_to_stage.append({
                "index": index,
                "name": output.name,
                "volume_name": output.storage.volume_name,
                "path": output.storage.path,
            })
    if not outputs_to_stage:
        return result
    staged_outputs = modal_namespace.ppiflow_stage_outputs.remote(
        run_id=context.run_id,
        node_id=context.node_id,
        outputs=outputs_to_stage,
    )
    if not isinstance(staged_outputs, list):
        raise TypeError("PPIFlow output staging did not return output metadata")
    replacements: dict[int, VolumePath] = {}
    for staged_output in staged_outputs:
        if not isinstance(staged_output, dict):
            raise TypeError("PPIFlow staged output metadata must be a mapping")
        replacements[int(staged_output["index"])] = VolumePath(
            volume_name=str(staged_output["volume_name"]),
            path=str(staged_output["path"]),
        )
    return result.model_copy(
        update={
            "outputs": [
                output.model_copy(update={"storage": replacements[index]})
                if index in replacements
                else output
                for index, output in enumerate(result.outputs)
            ]
        }
    )


def _seed_list(config: dict[str, Any]) -> list[int]:
    """Parse a LigandMPNN seed config into a list of integers."""
    raw_seeds = config.get("seeds", config.get("seed", "0"))
    if isinstance(raw_seeds, int):
        return [raw_seeds]
    if isinstance(raw_seeds, list):
        return [int(seed) for seed in raw_seeds]
    return [int(seed.strip()) for seed in str(raw_seeds).split(",") if seed.strip()]


def _ligandmpnn_cli_args(
    config: dict[str, Any],
    *,
    use_abmpnn: bool,
    fixed_positions: str | None = None,
) -> dict[str, str | int | float | bool]:
    """Translate upstream MPNNStep config into LigandMPNN CLI arguments."""
    use_soluble_model = bool(config.get("use_soluble_model", False))
    model_type = (
        "abmpnn"
        if use_abmpnn
        else ("soluble_mpnn" if use_soluble_model else "protein_mpnn")
    )
    cli_args: dict[str, str | int | float | bool] = {
        "--model_type": "protein_mpnn" if model_type == "abmpnn" else model_type,
        "--batch_size": str(config.get("batch_size", 1)),
        "--number_of_batches": str(config.get("num_seq_per_target", 1)),
        "--temperature": str(config.get("sampling_temp", 0.1)),
        "--save_stats": "1",
        "--pack_side_chains": bool(config.get("pack_side_chains", False)),
        "--number_of_packs_per_design": str(
            config.get("number_of_packs_per_design", 4)
        ),
        "--sc_num_denoising_steps": str(config.get("sc_num_denoising_steps", 3)),
        "--sc_num_samples": str(config.get("sc_num_samples", 16)),
        "--repack_everything": bool(config.get("repack_everything", False)),
        "--pack_with_ligand_context": bool(
            config.get("pack_with_ligand_context", True)
        ),
        "--ligand_mpnn_use_atom_context": bool(
            config.get("ligand_mpnn_use_atom_context", True)
        ),
        "--ligand_mpnn_cutoff_for_score": str(
            config.get("ligand_mpnn_cutoff_for_score", 8.0)
        ),
        "--ligand_mpnn_use_side_chain_context": bool(
            config.get("ligand_mpnn_use_side_chain_context", False)
        ),
        "--global_transmembrane_label": bool(
            config.get("global_transmembrane_label", False)
        ),
        "--parse_atoms_with_zero_occupancy": bool(
            config.get("parse_atoms_with_zero_occupancy", False)
        ),
    }
    if use_abmpnn:
        cli_args["--checkpoint_protein_mpnn"] = str(
            Path(ligandmpnn_app.CONF.model_volume_mountpoint)
            / "model_params"
            / "abmpnn.pt"
        )
    if chains_to_design := (config.get("chains_to_design") or config.get("chain_list")):
        cli_args["--chains_to_design"] = str(chains_to_design)
    if omit_aas := config.get("omit_AAs"):
        cli_args["--omit_AA"] = str(omit_aas)
    if fixed_positions is not None:
        cli_args["--fixed_residues"] = fixed_positions
    elif config_fixed_positions := config.get("fixed_positions"):
        cli_args["--fixed_residues"] = str(config_fixed_positions)
    if bias_aa := config.get("bias_AA_list"):
        cli_args["--bias_AA"] = str(bias_aa)
    return cli_args


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
    task_doc: dict[str, Any] | None = None,
    run_id: str,
    app_steps: tuple[str, ...],
) -> dict[str, Any]:
    """Upload local PPIFlow app inputs and rewrite step args to mounted paths."""
    staged_steps = deepcopy(steps_doc)
    uploads: list[tuple[Path, str]] = []
    volume_root = Path(ppiflow_app.CONF.output_volume_mountpoint)
    task = _task_section(task_doc) if task_doc is not None else {}
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")

    for step_name in app_steps:
        if step_name not in staged_steps:
            continue
        if step_name == "PPIFlowStep" and task_doc is not None:
            cfg = _ppiflow_design_cfg(
                steps=staged_steps,
                task=task,
                gentype=gentype,
            )
            staged_steps[step_name] = cfg
        else:
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
    """
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_path.read_bytes()),
        task_doc=task_doc,
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
    )
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
