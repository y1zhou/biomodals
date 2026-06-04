# PPIFlow Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `src/biomodals/workflow/ppiflow_workflow.py` as a ShortMD-style Biomodals workflow that follows upstream PPIFlow stage ordering and uses included Biomodals app functions.

**Architecture:** Build a static stage-level DAG from upstream `task.yaml` and `steps.yaml`. App-backed Workflow Nodes call included app functions through a `PPIFlowModalNamespace`; workflow-native helpers do only staging, archive extraction, structure selection, fixed-position CSV conversion, DockQ pair preparation, ranking, and reporting. `alphafold3_app` is an additional dependency for upstream `ReFoldStep` because DockQ needs refolded model structures and `af3score_app` intentionally does not emit them.

**Tech Stack:** Python 3.13 workflow module, Modal, Biomodals workflow runtime, Pydantic app result schemas, `polars`, `orjson`, `pytest`, `prek`.

---

## Task 1: Lock The Workflow Contract With Tests

**Files:**

- Modify: `tests/workflow/test_ppiflow_workflow.py`
- [x] Add tests that assert `CONF.depends_on_apps` contains `ppiflow`, `rosetta`, `flowpacker`, `ligandmpnn`, `dockq`, `af3score`, and `alphafold3`, and that `CONF.tags["depends_on"]` mirrors the tuple order.
- [x] Add a DAG-shape test for the full binder chain:
  `PPIFlowStep -> MPNNStep_stage1 -> FlowpackerStep_stage1 -> AF3scoreStep_stage1 -> FilterStep_stage1 -> RosettaFixStep -> FixedPositions -> PartialStep -> MPNNStep_stage2 -> FlowpackerStep_stage2 -> AF3scoreStep_stage2 -> FilterStep_stage2 -> ReFoldStep -> DockQStep -> RosettaRelaxStep -> RankStep -> ReportStep`.
- [x] Add adapter tests with fake Modal functions for PPIFlow, LigandMPNN, FlowPacker, AF3Score, DockQ, Rosetta, and AlphaFold3. The tests assert app calls are made through hydrated namespace handles and cover PPIFlowStep, PartialStep, LigandMPNN, FlowPacker byte and `VolumePath` archive returns, AF3Score input/metrics staging, DockQ, Rosetta, and AlphaFold3/ReFold.
- [x] Run `uv run pytest tests/workflow/test_ppiflow_workflow.py -q` and verify the new tests fail on the current skeleton.

### Task 2: Add Workflow Data Helpers

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Replace ad hoc YAML handling with helpers that read upstream `task` and `steps`, merge `task` fields into PPIFlow args, and validate stage selection. `PPIFlowStep` now follows upstream by taking design inputs from `task.yaml` and runtime config from `steps.yaml`; local staging applies the same merge before uploading static inputs.
- [x] Add volume helpers that convert `WorkflowArtifact.storage` to mount paths for known app output volumes. Workflow-volume structure/table/report artifacts resolve locally; PPIFlow app outputs, FlowPacker archive `VolumePath`s, and AF3Score metrics `VolumePath`s are staged into workflow-readable paths before downstream workflow-native nodes consume them.
- [x] Add archive extraction helpers for app functions that return `.tar.zst` bytes or archives in a volume path. LigandMPNN, FlowPacker, ReFold, DockQ, and Rosetta byte archive extraction is implemented; FlowPacker `AppRunResult`/`VolumePath` archive staging is implemented.
- [x] Port upstream pure helpers into workflow-native code using `polars` and standard library:
  filter parsing, FASTA sequence collection into `mpnn_seqs.csv`, Rosetta `residue_energy.csv` to `fixed_positions.csv`, partial sample directory discovery, DockQ model directory preparation, DockQ pair assembly, ranking, and report generation.
  Filter parsing, FilterStep CSV filtering/linking, FASTA sequence collection, Rosetta residue energy to `fixed_positions.csv` conversion, `before_partial_pdbs` symlink selection, ReFold AF3 metrics extraction, DockQ pair assembly, ranking, HTML report generation, Rosetta queue staging/output extraction, RosettaFix `ResResE` log normalization, and RosettaRelax scorefile-to-`rosetta_complex_0.csv` normalization are implemented. RosettaRelax now has a workflow-side PyRosetta scoring wrapper for app outputs that do not contain upstream `rosetta_complex_<batch_idx>.csv`, with one-time PyRosetta initialization for reused Modal containers; Rosetta remote behavior is accepted per user instruction.

### Task 3: Define Hydrated App Namespace And Nodes

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Extend `PPIFlowModalNamespace` with handles for PPIFlow, LigandMPNN, FlowPacker, AF3Score prepare/run/postprocess/lock, DockQ, Rosetta staging/run/packaging, and AlphaFold3 data/inference app functions.
- [x] Replace `PPIFlowWorkflowNode` with specific node classes:
  `PPIFlowDesignNode`, `LigandMPNNNode`, `FlowPackerNode`, `AF3ScoreNode`, `FilterStructuresNode`, `RosettaFixNode`, `FixedPositionsNode`, `PPIFlowPartialNode`, `ReFoldNode`, `DockQNode`, `RosettaRelaxNode`, `RankNode`, and `ReportNode`.
- [x] Make app-backed nodes `REMOTE`; make lightweight selectors/rank/report nodes `ORCHESTRATOR` unless they must access app volumes.
- [x] Return `AppRunResult` with `VolumePath` outputs for durable directories/tables/reports. Keep binary archives in volume-backed storage, not inline bytes.

### Task 4: Build The Upstream DAG

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Build stage 1 exactly as upstream: PPIFlow, binder MPNN or AbMPNN, collect `mpnn_pdbs/mpnn_seqs.csv`, FlowPacker, AF3Score, Filter.
  LigandMPNN now consumes real workflow PDB artifacts, persists and extracts returned archives, links upstream task-prefixed stage-1 `mpnn_pdbs`, and writes matching `mpnn_seqs.csv`; FlowPacker now consumes `mpnn_pdbs`, accepts either raw tarball bytes or workflow-compatible `VolumePath` archives, and extracts `run_1` for downstream scoring; AF3Score now stages FlowPacker PDBs into its app volume before prepare/run/postprocess. The DAG now uses explicit PPIFlow design and Rosetta stage node classes instead of generic placeholders for those steps.
- [x] Build stage 2 exactly as upstream: RosettaFix, fixed positions CSV, before-partial structure selection, Partial, binder MPNN or AbMPNN, FlowPacker, AF3Score, Filter, ReFold, DockQ, RosettaRelax, Rank, Report. Static DAG order, Rosetta app queue staging/output extraction, Rosetta task manifest preservation, RosettaFix `update.xml` generation, upstream RosettaFix default flags, RosettaFix residue-energy generation, RosettaRelax interface-score normalization with a workflow-side PyRosetta wrapper, fixed-position CSV conversion, before-partial selection, per-PDB PartialStep app calls, upstream PartialStep `samples_per_target` defaults, stage-2 MPNN/AbMPNN fixed-position propagation, partial-structure-prefixed stage-2 `mpnn_pdbs`/`mpnn_seqs.csv` collection, AbMPNN checkpoint and `chain_list` translation, ReFold `af3_iptm@1.csv` generation, DockQ model/reference pair assembly, RankStep merge/copy behavior, and ReportStep HTML output are covered; Rosetta remote behavior is accepted per user instruction.
- [x] Preserve stage-only execution behavior while requiring existing upstream artifacts for stage 2-only runs. `stage=2` now requires `Stage2Inputs.filtered_structures` and wires it through an existing-volume artifact node instead of building a disconnected DAG.
- [x] Update local input staging to cover initial PPIFlow inputs and preserve mounted paths. Local staging now uploads only static `PPIFlowStep` inputs after merging task-level fields; `PartialStep` runtime inputs are derived from RosettaFix/FixedPositions outputs inside the workflow.

### Task 5: Verify And Clean Up

**Files:**

- Modify: `tests/workflow/test_ppiflow_workflow.py`
- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Run `uv run pytest tests/workflow/test_ppiflow_workflow.py -q`.
- [x] Run `uv run pytest tests/app/test_catalog_workflow_apps.py tests/app/test_cli_workflow_catalog.py -q`.
- [x] Run `uv run biomodals workflow list`.
- [x] Run `uv run biomodals workflow help ppiflow`.
- [x] Run `prek run --files src/biomodals/workflow/ppiflow_workflow.py tests/workflow/test_ppiflow_workflow.py docs/superpowers/plans/2026-06-01-ppiflow-workflow.md`.

### Self-Review

- Coverage: the plan covers every named upstream PPIFlow step and every required app, including the newly approved `alphafold3` dependency for `ReFoldStep`.
- Placeholder scan: no workflow node is left as an unspecified implementation placeholder. Rosetta remote behavior is assumed working per user instruction, so no workflow stage remains blocked on live app validation.
- Type consistency: node names, app function handles, and artifact kinds match Biomodals workflow vocabulary.
