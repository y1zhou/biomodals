"""CPU-only import and endpoint validation before production continuation."""

from __future__ import annotations

import math
import re
import shutil
from hashlib import file_digest, md5, sha256
from pathlib import Path

import orjson

from biomodals.app.bioinfo.gromacs.execution import PREPARE_CONTINUATION, PREPARE_RESULT
from biomodals.app.bioinfo.gromacs.execution_runtime import (
    GromacsExecutionRequest,
    gromacs_node_paths,
    gromacs_publication_path,
    load_execution_request,
    parse_gromacs_publication,
)
from biomodals.execution import ContentBoundFileSet
from biomodals.helper.artifacts import replace_bytes_atomic
from biomodals.helper.io import require_safe_filename_component
from biomodals.helper.shell import run_command
from biomodals.schema import ArtifactFile


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Continuation evidence must be regular files")
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def native_endpoint(
    gmx: str, tpr: Path, checkpoint: Path, scratch: Path
) -> tuple[int, float, tuple[str, ...]]:
    """Verify checkpoint completion against its actual TPR, not XTC timestamps."""
    mdp = scratch / "source-parameters.mdp"
    run_command(
        [gmx, "dump", "-s", str(tpr), "-om", str(mdp)],
        output_mode="discard",
        show_command=False,
    )
    parameters = {}
    for line in mdp.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            parameters[key.strip()] = value.partition(";")[0].strip()
    start = int(parameters.get("init-step", "0"))
    steps = int(parameters["nsteps"])
    dt = float(parameters["dt"])
    end_step = start + steps
    end_time = float(parameters.get("tinit", "0")) + end_step * dt
    dump = scratch / "checkpoint.txt"
    # run_command appends logs; each inspection must parse only this checkpoint.
    dump.unlink(missing_ok=True)
    run_command(
        [gmx, "dump", "-cp", str(checkpoint)],
        output_mode="log",
        log_file=dump,
        show_command=False,
    )
    checkpoint_values = {}
    outputs: list[dict[str, str]] = []
    with dump.open() as handle:
        for line in handle:
            if "Checkpoint file is corrupted or truncated" in line:
                raise ValueError("Source checkpoint is corrupted or truncated")
            match = re.fullmatch(
                r"\s*(step|t|number of output files)\s*=\s*(\S+)\s*", line
            )
            if match:
                checkpoint_values[match[1]] = match[2]
            key, sep, value = line.strip().partition(" = ")
            if sep and key == "output filename":
                outputs.append({key: value})
            elif sep and key.startswith("file_") and outputs:
                outputs[-1][key] = value
    if (
        steps <= 0
        or dt <= 0
        or not math.isfinite(end_time)
        or int(checkpoint_values["step"]) != end_step
        or not math.isclose(
            float(checkpoint_values["t"]), end_time, rel_tol=0, abs_tol=1e-5
        )
    ):
        raise ValueError("Source checkpoint has not reached the completed TPR endpoint")
    if not outputs or len(outputs) != int(checkpoint_values["number of output files"]):
        raise ValueError("Checkpoint append inventory is incomplete")
    for output in outputs:
        name = output["output filename"]
        require_safe_filename_component(name, field_name="checkpoint append output")
        path = checkpoint.parent / name
        offset = (int(output["file_offset_high"]) << 32) | (
            int(output["file_offset_low"]) & 0xFFFFFFFF
        )
        size = int(output["file_checksum_size"])
        if (
            path.is_symlink()
            or not path.is_file()
            or offset < 0
            or path.stat().st_size < offset
            or size != min(offset, 1024 * 1024)
        ):
            raise ValueError(f"Checkpoint append output is unavailable: {name}")
        # Pinned GROMACS validates this final 1 MiB window, not the entire file.
        with path.open("rb") as handle:
            handle.seek(offset - size)
            digest = md5(handle.read(size), usedforsecurity=False).hexdigest()
        if digest != output["file_checksum"]:
            raise ValueError(f"Checkpoint append checksum differs: {name}")
    return end_step, end_time, tuple(output["output filename"] for output in outputs)


def prepare_continuation_files(
    request: GromacsExecutionRequest, volume_root: Path
) -> Path:
    """Copy into an isolated child; never replace a progressing child checkpoint."""
    source = request.continuation
    if source is None:
        raise ValueError("Continuation source is required")
    source_root = volume_root / source.run_name
    root = request.run_root(volume_root)
    if source_root.is_symlink() or root.is_symlink():
        raise ValueError("Continuation directories cannot be symlinks")
    prepared = ContentBoundFileSet(
        root=root,
        marker_path=volume_root
        / gromacs_publication_path(request, PREPARE_CONTINUATION),
        expected_paths=gromacs_node_paths(request, PREPARE_CONTINUATION),
        identity={
            "node_key": PREPARE_CONTINUATION,
            "workload_plan_fingerprint": request.execution_plan.workload_plan_fingerprint,
        },
    )
    if prepared.load() is not None:
        return root
    parent = load_execution_request(volume_root, source.execution_run_id)
    if sha256(parent.to_bytes()).hexdigest() != source.request_sha256:
        raise ValueError("Continuation source request changed")
    marker = (
        volume_root / gromacs_publication_path(parent, PREPARE_RESULT)
    ).read_bytes()
    if sha256(marker).hexdigest() != source.publication_sha256:
        raise ValueError("Continuation source publication changed")
    published = parse_gromacs_publication(parent, PREPARE_RESULT, marker)
    if published is None:
        raise ValueError("Continuation source publication is invalid")
    prefix = f"production_{source.file_stem}"
    checkpoint = source_root / f"{prefix}.cpt"
    if _digest(checkpoint) != source.checkpoint_sha256:
        raise ValueError("Continuation source checkpoint changed")
    ready = root / "continuation.json"
    child_checkpoint = root / checkpoint.name
    if (
        child_checkpoint.exists()
        and _digest(child_checkpoint) != source.checkpoint_sha256
    ):
        raise ValueError(
            "Cannot replace child progress after continuation preparation was lost"
        )

    gmx = shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError("GROMACS binary not found")
    scratch = root / ".biomodals" / "continuation"
    scratch.mkdir(parents=True, exist_ok=True)
    source_tpr = source_root / f"{prefix}.tpr"
    # Validate only files we reuse. Production plots/processed XTC are regenerated.
    required = {
        file.path: file
        for file in published
        if file.path.startswith((
            "rmsd_nvt_",
            "rg_nvt_",
            "rmsf_nvt_",
            "rmsd_npt_",
            "rg_npt_",
            "rmsf_npt_",
        ))
        or file.path in {"production.mdp", f"{prefix}.tpr", f"{prefix}.edr"}
    }
    production = "production_run_cpu" if parent.cpu_only else "production_run_gpu"
    production_marker = (
        volume_root / gromacs_publication_path(parent, production)
    ).read_bytes()
    native_files = parse_gromacs_publication(parent, production, production_marker)
    if native_files is None:
        raise ValueError("Source native production publication is invalid")
    required.update({file.path: file for file in native_files})
    for name, file in required.items():
        path = source_root / name
        if (
            path.stat().st_size != file.size_bytes
            or _digest(path) != file.content_sha256
        ):
            raise ValueError(f"Source file no longer matches its publication: {name}")

    input_pdb = source_root / f"{source.file_stem}.pdb"
    if _digest(input_pdb) != sha256(parent.pdb_content).hexdigest():
        raise ValueError("Source input structure changed")
    step, time_ps, append_files = native_endpoint(gmx, source_tpr, checkpoint, scratch)
    if not math.isclose(
        time_ps, source.simulation_time_ns * 1000, rel_tol=0, abs_tol=1e-5
    ):
        raise ValueError("Source native endpoint differs from its saved request")
    names = set(required) | {input_pdb.name, checkpoint.name, *append_files}
    for name in sorted(names):
        path = source_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe native append file: {name}")
        if name != source_tpr.name:
            shutil.copyfile(path, root / name)
    shutil.copyfile(source_tpr, root / "source.tpr")
    additional_ns = request.simulation_time_ns - source.simulation_time_ns
    # Always extend the immutable source TPR, never an earlier child TPR.
    (root / source_tpr.name).unlink(missing_ok=True)
    run_command(
        [
            gmx,
            "convert-tpr",
            "-s",
            str(root / "source.tpr"),
            "-extend",
            str(additional_ns * 1000),
            "-o",
            str(root / f"{prefix}.tpr"),
        ],
        output_mode="capture",
    )
    replace_bytes_atomic(
        ready,
        orjson.dumps(
            {
                "workload_plan_fingerprint": request.execution_plan.workload_plan_fingerprint,
                "source": source.model_dump(mode="json"),
                "source_checkpoint_step": step,
                "source_checkpoint_time_ps": time_ps,
                "additional_time_ns": additional_ns,
                "target_time_ns": request.simulation_time_ns,
                "trajectory_scope": "cumulative",
                "equilibration_analysis": "inherited",
                "production_mdp": "original input; extended TPR is authoritative",
            },
            option=orjson.OPT_SORT_KEYS,
        ),
    )
    prepared.write(
        tuple(
            ArtifactFile(
                path=name,
                size_bytes=(root / name).stat().st_size,
                content_sha256=_digest(root / name),
            )
            for name in prepared.expected_paths
        )
    )
    return root
