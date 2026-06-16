# PPIFlow Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `src/biomodals/workflow/ppiflow_workflow.py` as a ShortMD-style Biomodals workflow that follows upstream PPIFlow stage ordering and uses included Biomodals app functions.

**Architecture:** Build a static stage-level DAG from upstream `task.yaml` and `steps.yaml`. App-backed Workflow Nodes call included app functions through a `PPIFlowModalNamespace`; workflow-native helpers do only staging, archive extraction, structure selection, fixed-position CSV conversion, DockQ pair preparation, ranking, and reporting. `alphafold3_app` is an additional dependency for upstream `ReFoldStep` because DockQ needs refolded model structures and `af3score_app` intentionally does not emit them.

**Tech Stack:** Python 3.13 workflow module, Modal, Biomodals workflow runtime, Pydantic app result schemas, `orjson`, `pytest`, `prek`.

---

## Task 1: Lock The Workflow Contract With Tests

**Files:**

- Modify: `tests/workflow/test_ppiflow_workflow.py`
- [x] Add tests that assert `CONF.depends_on_apps` contains `ppiflow`, `rosetta`, `flowpacker`, `ligandmpnn`, `dockq`, `af3score`, and `alphafold3`, and that `CONF.tags["depends_on"]` mirrors the tuple order.
- [x] Add a DAG-shape test for the full binder chain:
  `PPIFlowStep -> MPNNStep_stage1 -> FlowpackerStep_stage1 -> AF3scoreStep_stage1 -> FilterStep_stage1 -> RosettaFixStep -> FixedPositions -> PartialStep -> MPNNStep_stage2 -> FlowpackerStep_stage2 -> AF3scoreStep_stage2 -> FilterStep_stage2 -> ReFoldStep -> DockQStep -> RosettaRelaxStep -> RankStep -> ReportStep`.
  Current coverage also asserts that those DAG vertices use specific node
  classes instead of the old generic `PPIFlowWorkflowNode`.
- [x] Add adapter tests with fake Modal functions for PPIFlow, LigandMPNN, FlowPacker, AF3Score, DockQ, Rosetta, and AlphaFold3. The tests assert that app calls are made through the hydrated namespace, not deployed lookup strings, and cover PartialStep staging through the PPIFlow volume.
- [x] Run `uv run pytest tests/workflow/test_ppiflow_workflow.py -q` and verify the new tests fail on the current skeleton.

### Task 2: Add Workflow Data Helpers

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Replace ad hoc YAML handling with helpers that read upstream `task` and `steps`, parse enabled steps, validate stage selection, and require explicit `Stage2Input` configuration for stage-2-only runs.
- [x] Add volume helpers that convert `WorkflowArtifact.storage` to mount paths for known app output volumes.
- [x] Add archive extraction helpers for app functions that return `.tar.zst` bytes or archives in a volume path.
- [x] Port upstream pure helpers into workflow-native code using standard library:
  filter parsing, FASTA sequence collection into `mpnn_seqs.csv`, Rosetta `residue_energy.csv` to `fixed_positions.csv`, partial sample directory discovery, DockQ model directory preparation, DockQ pair assembly, ranking, and report generation.
  Current implementation provides workflow-owned structure selection/copying, partial input staging, DockQ pair assembly, ranking, and report generation through standard `AppRunResult` artifacts.

### Task 3: Define Hydrated App Namespace And Nodes

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Extend `PPIFlowModalNamespace` with handles for PPIFlow, LigandMPNN, FlowPacker, AF3Score prepare/run/postprocess/lock, DockQ, Rosetta, Rosetta packaging, and AlphaFold3 data/inference app functions.
- [x] Replace `PPIFlowWorkflowNode` with specific node classes:
  `PPIFlowDesignNode`, `LigandMPNNNode`, `FlowPackerNode`, `AF3ScoreNode`, `FilterStructuresNode`, `RosettaFixNode`, `FixedPositionsNode`, `PPIFlowPartialNode`, `ReFoldNode`, `DockQNode`, `RosettaRelaxNode`, `RankNode`, and `ReportNode`.
- [x] Make app-backed nodes `REMOTE`; make lightweight selectors/rank/report nodes `ORCHESTRATOR` unless they must access app volumes.
- [x] Return `AppRunResult` with durable `VolumePath` outputs for app-owned directories and metrics tables, and materializable inline outputs for small workflow-native tables/reports and app archive bytes.

### Task 4: Build The Upstream DAG

**Files:**

- Modify: `src/biomodals/workflow/ppiflow_workflow.py`
- [x] Build stage 1 exactly as upstream: PPIFlow, binder MPNN or AbMPNN, collect `mpnn_pdbs/mpnn_seqs.csv`, FlowPacker, AF3Score, Filter.
- [x] Build stage 2 exactly as upstream: RosettaFix, fixed positions CSV, before-partial structure selection, Partial, binder MPNN or AbMPNN, FlowPacker, AF3Score, Filter, ReFold, DockQ, RosettaRelax, Rank, Report.
- [x] Preserve stage-only execution behavior while requiring existing upstream artifacts for stage 2-only runs.
- [x] Update local input staging to cover initial PPIFlow inputs and preserve mounted paths.

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
- Placeholder scan: no task is left as an unspecified implementation placeholder; the only remote biological execution details are delegated to the named app functions.
- Type consistency: node names, app function handles, and artifact kinds match Biomodals workflow vocabulary.
