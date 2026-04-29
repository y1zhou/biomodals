"""Task-queue orchestrator for PPIFlow-style multi-stage design workflows.

This workflow coordinates deployed Modal functions from:

- ``biomodals.app.design.ppiflow_app``
- ``biomodals.app.fol d.flowpacker_app``
- ``biomodals.app.design.ligandmpnn_app``
- ``biomodals.app.bioinfo.rosetta_app``
- ``biomodals.app.score.af3score_app``

The orchestrator builds a dependency-aware queue and runs independent steps in
parallel worker threads from the local entrypoint.
"""

from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal

DEFAULT_APP_NAMES = {
    "ppiflow": "PPIFlow",
    "flowpacker": "FlowPacker",
    "ligandmpnn": "LigandMPNN",
    "rosetta": "Rosetta",
    "af3score": "AF3Score",
}


@dataclass
class WorkflowTask:
    """One queued workflow step."""

    task_id: str
    app_key: str
    fn_name: str
    kwargs: dict[str, Any]
    needs: set[str] = field(default_factory=set)


def _lookup_function(task: WorkflowTask):
    app_name = DEFAULT_APP_NAMES[task.app_key]
    return modal.Function.lookup(app_name, task.fn_name)


def _run_task(task: WorkflowTask) -> tuple[str, Any]:
    fn = _lookup_function(task)
    print(f"🧬 starting task '{task.task_id}' via {DEFAULT_APP_NAMES[task.app_key]}.{task.fn_name}")
    result = fn.remote(**task.kwargs)
    print(f"🧬 completed task '{task.task_id}'")
    return task.task_id, result


def _schedule(tasks: list[WorkflowTask], max_workers: int = 4) -> dict[str, Any]:
    by_id = {task.task_id: task for task in tasks}
    remaining = {task.task_id: set(task.needs) for task in tasks}
    ready = deque([task_id for task_id, deps in remaining.items() if not deps])

    outputs: dict[str, Any] = {}
    in_flight: dict[Any, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while ready or in_flight:
            while ready and len(in_flight) < max_workers:
                task_id = ready.popleft()
                fut = pool.submit(_run_task, by_id[task_id])
                in_flight[fut] = task_id

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                task_id = in_flight.pop(fut)
                finished_task_id, result = fut.result()
                outputs[finished_task_id] = result

                for waiting_task_id, deps in remaining.items():
                    if task_id in deps:
                        deps.remove(task_id)
                        if not deps and waiting_task_id not in outputs and waiting_task_id not in ready and waiting_task_id not in in_flight.values():
                            ready.append(waiting_task_id)

    unresolved = {k: v for k, v in remaining.items() if v and k not in outputs}
    if unresolved:
        raise RuntimeError(f"Unresolved workflow dependencies: {unresolved}")

    return outputs


app = modal.App("PPIFlowWorkflow")


@app.local_entrypoint()
def submit_ppiflow_workflow(
    run_name: str,
    ppiflow_args_json: str,
    ligandmpnn_input_pdb: str,
    flowpacker_input_pdb: str,
    rosetta_input_pdb: str,
    rosetta_script: str,
    af3_paths_json: str,
    max_workers: int = 4,
) -> None:
    """Execute a queued PPIFlow workflow.

    Args:
        run_name: Stable workflow run identifier.
        ppiflow_args_json: JSON payload compatible with ``PPIFlowArgs`` model.
        ligandmpnn_input_pdb: Path to input pdb for LigandMPNN.
        flowpacker_input_pdb: Path to input pdb for FlowPacker.
        rosetta_input_pdb: Path to input pdb for Rosetta.
        rosetta_script: Path to RosettaScript XML file.
        af3_paths_json: JSON dict with ``run_root/inputs_dir/output_dir/failed_dir/metrics_csv``.
        max_workers: Number of parallel local orchestration workers.
    """
    run_stamp = int(time.time())
    ligand_bytes = Path(ligandmpnn_input_pdb).expanduser().resolve().read_bytes()
    flowpacker_bytes = Path(flowpacker_input_pdb).expanduser().resolve().read_bytes()
    ppiflow_args = json.loads(ppiflow_args_json)

    rosetta_run_id = f"{run_name}-{run_stamp}"

    tasks = [
        WorkflowTask(
            task_id="ppiflow_design",
            app_key="ppiflow",
            fn_name="ppiflow_run",
            kwargs={"args": ppiflow_args, "run_name": f"{run_name}-ppiflow"},
        ),
        WorkflowTask(
            task_id="ligandmpnn_sequences",
            app_key="ligandmpnn",
            fn_name="ligandmpnn_run",
            kwargs={
                "run_name": f"{run_name}-ligandmpnn",
                "script_mode": "run",
                "struct_bytes": ligand_bytes,
                "seeds": [0, 1, 2],
                "cli_args": {"--verbose": True},
            },
            needs={"ppiflow_design"},
        ),
        WorkflowTask(
            task_id="flowpacker_pack",
            app_key="flowpacker",
            fn_name="run_flowpacker",
            kwargs={
                "input_files": [(Path(flowpacker_input_pdb).name, flowpacker_bytes)],
                "run_name": f"{run_name}-flowpacker",
                "n_samples": 1,
            },
            needs={"ppiflow_design"},
        ),
        WorkflowTask(
            task_id="rosetta_refine",
            app_key="rosetta",
            fn_name="run_rosetta",
            kwargs={
                "run_name": run_name,
                "run_id": rosetta_run_id,
                "num_cpu_per_pod": 1,
            },
            needs={"ppiflow_design"},
        ),
        WorkflowTask(
            task_id="af3_score",
            app_key="af3score",
            fn_name="af3score_postprocess",
            kwargs={
                "input_files": [Path(flowpacker_input_pdb).name],
                "paths": json.loads(af3_paths_json),
            },
            needs={"ligandmpnn_sequences", "flowpacker_pack", "rosetta_refine"},
        ),
    ]

    results = _schedule(tasks=tasks, max_workers=max_workers)
    print("🧬 Workflow completed. Task outputs:")
    for task_id, result in results.items():
        print(f"  - {task_id}: {type(result).__name__}")
