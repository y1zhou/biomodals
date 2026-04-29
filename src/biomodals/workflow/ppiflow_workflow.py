"""PPIFlow-style Modal workflow orchestrator.

Follows the staged workflow in Mingchenchen/PPIFlow while dispatching to already
-deployed Biomodals app functions.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import modal

APP_NAMES = {
    "ppiflow": "PPIFlow",
    "flowpacker": "FlowPacker",
    "ligandmpnn": "LigandMPNN",
    "rosetta": "Rosetta",
    "af3score": "AF3Score",
}


@dataclass
class TaskSpec:
    """One DAG node in the orchestrator task queue."""

    task_id: str
    app_key: str
    function_name: str
    kwargs: dict[str, Any]
    deps: set[str] = field(default_factory=set)


def _lookup_modal_function(app_key: str, function_name: str):
    return modal.Function.lookup(APP_NAMES[app_key], function_name)


def _run_remote_task(task: TaskSpec) -> tuple[str, Any]:
    fn = _lookup_modal_function(task.app_key, task.function_name)
    print(f"🧬 [{task.task_id}] start -> {APP_NAMES[task.app_key]}.{task.function_name}")
    result = fn.remote(**task.kwargs)
    print(f"🧬 [{task.task_id}] done")
    return task.task_id, result


def _run_task_queue(tasks: list[TaskSpec], max_workers: int) -> dict[str, Any]:
    task_map = {task.task_id: task for task in tasks}
    unmet_deps = {task.task_id: set(task.deps) for task in tasks}
    ready = deque([task_id for task_id, deps in unmet_deps.items() if not deps])

    outputs: dict[str, Any] = {}
    running: dict[Future, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while ready or running:
            while ready and len(running) < max_workers:
                task_id = ready.popleft()
                fut = pool.submit(_run_remote_task, task_map[task_id])
                running[fut] = task_id

            completed_futures, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for fut in completed_futures:
                done_task_id = running.pop(fut)
                _, result = fut.result()
                outputs[done_task_id] = result

                for task_id, deps in unmet_deps.items():
                    if done_task_id in deps:
                        deps.remove(done_task_id)
                        if not deps and task_id not in outputs and task_id not in ready:
                            if task_id not in running.values():
                                ready.append(task_id)

    unresolved = {task_id: deps for task_id, deps in unmet_deps.items() if deps}
    if unresolved:
        raise RuntimeError(f"DAG has unresolved dependencies: {unresolved}")
    return outputs


def _read_optional_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    return Path(path).expanduser().resolve().read_bytes()


app = modal.App("PPIFlowWorkflow")


@app.local_entrypoint()
def submit_ppiflow_workflow(
    ppiflow_run_name: str,
    ppiflow_args_json: str,
    run_flowpacker: bool = True,
    run_ligandmpnn: bool = True,
    run_rosetta: bool = False,
    run_af3score: bool = True,
    flowpacker_input_pdb: str | None = None,
    ligandmpnn_input_pdb: str | None = None,
    rosetta_run_id: str | None = None,
    rosetta_num_cpu_per_pod: int = 1,
    af3_input_files_json: str = "[]",
    af3_paths_json: str = "{}",
    max_workers: int = 4,
) -> None:
    """Run a PPIFlow-compatible staged workflow using deployed Modal functions.

    Args:
        ppiflow_run_name: Name used for the PPIFlow backbone generation step.
        ppiflow_args_json: JSON that can instantiate ``PPIFlowArgs`` for `ppiflow_run`.
        run_flowpacker: Whether to run the FlowPacker packing stage.
        run_ligandmpnn: Whether to run the LigandMPNN design stage.
        run_rosetta: Whether to run Rosetta refinement stage.
        run_af3score: Whether to run AF3Score postprocess stage.
        flowpacker_input_pdb: Input PDB for FlowPacker.
        ligandmpnn_input_pdb: Input PDB for LigandMPNN.
        rosetta_run_id: Optional explicit Rosetta run id.
        rosetta_num_cpu_per_pod: CPU parallelism passed to Rosetta run function.
        af3_input_files_json: JSON list of AF3 input file names.
        af3_paths_json: JSON dict of AF3 path bundle.
        max_workers: Max local orchestration workers for independent stages.
    """
    ppiflow_args: dict[str, Any] = json.loads(ppiflow_args_json)
    af3_input_files = json.loads(af3_input_files_json)
    af3_paths = json.loads(af3_paths_json)

    tasks: list[TaskSpec] = [
        TaskSpec(
            task_id="ppiflow_design",
            app_key="ppiflow",
            function_name="ppiflow_run",
            kwargs={"args": ppiflow_args, "run_name": ppiflow_run_name},
        )
    ]

    if run_ligandmpnn:
        if ligandmpnn_input_pdb is None:
            raise ValueError("`ligandmpnn_input_pdb` is required when run_ligandmpnn=True")
        ligand_bytes = Path(ligandmpnn_input_pdb).expanduser().resolve().read_bytes()
        tasks.append(
            TaskSpec(
                task_id="ligandmpnn_stage",
                app_key="ligandmpnn",
                function_name="ligandmpnn_run",
                kwargs={
                    "run_name": f"{ppiflow_run_name}-ligandmpnn",
                    "script_mode": "run",
                    "struct_bytes": ligand_bytes,
                    "seeds": [0, 1, 2],
                    "cli_args": {},
                    "bias_aa_per_residue_bytes": None,
                    "omit_aa_per_residue_bytes": None,
                },
                deps={"ppiflow_design"},
            )
        )

    if run_flowpacker:
        if flowpacker_input_pdb is None:
            raise ValueError("`flowpacker_input_pdb` is required when run_flowpacker=True")
        flowpacker_bytes = Path(flowpacker_input_pdb).expanduser().resolve().read_bytes()
        tasks.append(
            TaskSpec(
                task_id="flowpacker_stage",
                app_key="flowpacker",
                function_name="run_flowpacker",
                kwargs={
                    "input_files": [(Path(flowpacker_input_pdb).name, flowpacker_bytes)],
                    "run_name": f"{ppiflow_run_name}-flowpacker",
                    "model_name": "cluster",
                    "use_confidence": False,
                    "n_samples": 1,
                },
                deps={"ppiflow_design"},
            )
        )

    if run_rosetta:
        tasks.append(
            TaskSpec(
                task_id="rosetta_stage",
                app_key="rosetta",
                function_name="run_rosetta",
                kwargs={
                    "run_name": ppiflow_run_name,
                    "run_id": rosetta_run_id or f"{ppiflow_run_name}-rosetta",
                    "num_cpu_per_pod": rosetta_num_cpu_per_pod,
                },
                deps={"ppiflow_design"},
            )
        )

    if run_af3score:
        deps = {"ppiflow_design"}
        if run_ligandmpnn:
            deps.add("ligandmpnn_stage")
        if run_flowpacker:
            deps.add("flowpacker_stage")
        if run_rosetta:
            deps.add("rosetta_stage")

        tasks.append(
            TaskSpec(
                task_id="af3score_stage",
                app_key="af3score",
                function_name="af3score_postprocess",
                kwargs={"input_files": af3_input_files, "paths": af3_paths},
                deps=deps,
            )
        )

    results = _run_task_queue(tasks=tasks, max_workers=max_workers)
    print("🧬 workflow finished")
    for task_id, result in results.items():
        print(f"🧬 {task_id}: {type(result).__name__}")
